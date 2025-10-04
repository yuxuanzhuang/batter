import os
import numpy as np
import MDAnalysis as mda
from rdkit import Chem
from openff.toolkit import Molecule
from abc import ABC, abstractmethod
from loguru import logger
import random
import string
import tempfile
import shutil
from openfe import SmallMoleculeComponent
from typing import List, Union

from batter.utils import (
    run_with_log,
    save_state,
    safe_directory,
    antechamber,
    tleap,
    cpptraj,
    parmchk2,
    charmmlipid2amber,
    obabel,
    vmd)

import hashlib
import re
import json

FORBIDDEN_MOL_NAMES = {
    'add', 'all', 'and', 'any'
}

def _base26_triplet(n: int) -> str:
    """Map nonnegative int -> 3 lowercase letters (base-26, a..z)."""
    n = n % (26**3)
    a = n // (26**2)
    b = (n // 26) % 26
    c = n % 26
    return chr(a + 97) + chr(b + 97) + chr(c + 97)

def _stable_hash_int(s: str) -> int:
    """Deterministic int from string (md5, not Python's salted hash)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16)

def _convert_mol_name_to_unique(mol_name: str, ind: int, smiles: str, exist_mol_names: set[str]) -> str:
    """
    Convert a molecule name to a unique identifier of exactly 3 characters
    (letters or digits), deterministic across runs, and never all digits.
    """
    # 1) normalize to letters and digits only (lowercase), take at most 3
    base = re.sub(r'[^a-zA-Z0-9]', '', mol_name or '').lower()[:3]

    # 2) pad with letters from index if shorter
    if len(base) < 3:
        pad = _base26_triplet(ind)
        base = (base + pad)[:3]

    # If still somehow empty, synthesize from index (letters only)
    if not base:
        base = _base26_triplet(ind)

    # 2.5) enforce: cannot be all digits
    if len(base) == 3 and base.isdigit():
        base = 'l' + base[:2]

    # 3) ensure uniqueness
    if base not in exist_mol_names and base not in FORBIDDEN_MOL_NAMES:
        return base

    # Collision: derive from SMILES hash; if still collides, perturb deterministically
    seed = _stable_hash_int(smiles) if smiles else _stable_hash_int(base)
    attempt = 0
    while attempt < 200:
        candidate = _base26_triplet(seed + attempt)  # letters only, so not all digits
        if candidate not in exist_mol_names and candidate not in FORBIDDEN_MOL_NAMES:
            return candidate
        attempt += 1

    # Last resort: index-perturbed triplet (letters only)
    attempt = 0
    while attempt < 200:
        triplet = _base26_triplet(ind + attempt)
        if triplet not in exist_mol_names and triplet not in FORBIDDEN_MOL_NAMES:
            return triplet
        attempt += 1
    raise ValueError(f"Could not derive unique 3-char name for molecule {mol_name} (smiles: {smiles})")
    
class LigandProcessing(ABC):
    """
    Base class for ligand processing.
    It will read the ligand file, calculate the
    partial charges, and generate the ligand
    topology in AMBER format.

    Properties
    ----------
    ligand_file : str
        The ligand file path.
    openff_molecule : openff.toolkit.Molecule
        The openff molecule object.
    ligand_sdf_path : str
        The ligand sdf file path.
    ligand_charge : float
        The ligand charge.
    """
    def __init__(self,
                ligand_file,
                index,
                output_dir,
                ligand_name=None,
                charge='am1bcc',
                retain_lig_prot=True,
                ligand_ff='gaff2',
                unique_mol_names=[],
                ):
        self.ligand_file = ligand_file
        self.index = index
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = os.path.abspath(output_dir)
        self._name = ligand_name.lower() if ligand_name is not None else None
        self.charge = charge
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ff = ligand_ff

        from openff.toolkit.typing.engines.smirnoff.forcefield import get_available_force_fields
        available_amber_ff = ['gaff', 'gaff2']
        available_openff_ff = [ff.removesuffix(".offxml") for ff in get_available_force_fields() if 'openff' in ff]
        if ligand_ff in available_amber_ff:
            self.force_field = 'amber'
        elif ligand_ff in available_openff_ff:
            self.force_field = 'openff'
        else:
            raise ValueError(f"Unsupported force field: {ligand_ff}. "
                             f"Supported force fields are: {available_amber_ff + available_openff_ff}")
        logger.debug(f'Using force field: {self.force_field} for ligand {ligand_file}')

        self.unique_mol_names = unique_mol_names
        ligand_rdkit = self._load_ligand()
        self._cano_smiles = Chem.MolToSmiles(ligand_rdkit, canonical=True)
        
        if ligand_name is None:
            ligand = SmallMoleculeComponent(ligand_rdkit)
        else:
            ligand = SmallMoleculeComponent(ligand_rdkit, name=ligand_name)

        self.ligand_object = ligand
        self.openff_molecule = ligand.to_openff()
        self._generate_unique_name()
 
    def _generate_unique_name(self):
        self._get_mol_name()
        self._name = _convert_mol_name_to_unique(
            self._name,
            self.index,
            self.smiles,
            self.unique_mol_names
        )
        logger.debug(f'Ligand {self.index}: {self.name}')
        self.openff_molecule.name = self.name.lower()

        # needed for construction of residues
        self.openff_molecule.add_default_hierarchy_schemes()
        self.openff_molecule.residues[0].residue_name = self.name.lower()

        self.openff_molecule.to_file(self.ligand_sdf_path, file_format='sdf')
    
    def to_dict(self):
        return {
            'ligand_file': self.ligand_file,
            'charge_type': self.charge,
            'smiles': self.smiles,
            'ligand_sdf_path': self.ligand_sdf_path,
            'ligand_charge': getattr(self, 'ligand_charge', None),
            'ligand_ff': self.ligand_ff,
            'ligand_name': self.name,
            'retain_lig_prot': self.retain_lig_prot,
        }

    @abstractmethod
    def _load_ligand(self):
        raise NotImplementedError("Subclasses must implement _load_ligand method")
    
    def _get_mol_name(self):
        if self._name is not None:
            return
        if self.ligand_object.name is not None:
            mol_name = self.ligand_object.name
        else:
            mol_name = os.path.basename(self.ligand_file).split('.')[0].lower()
        self._name = mol_name

    @property
    def name(self):
        return self._name

    @property
    def smiles(self):
        return self._cano_smiles

    @property
    def ligand_sdf_path(self):
        return os.path.join(self.output_dir, f'{self.name}.sdf')

    def _calculate_partial_charge(self):
        """
        This function calculates the partial charges of the ligand
        using the openff toolkit with gasteiger method.
        It is only for fast estimation of the ligand charge.
        and antechamber will use bcc method to calculate the partial charges.
        """
        molecule = self.openff_molecule
        charge_method = self.charge if self.force_field == 'openff' else 'gasteiger'
        molecule.assign_partial_charges(
            partial_charge_method=charge_method
        )
        ligand_charge = np.round(np.sum([charge._magnitude
            for charge in molecule.partial_charges]))
        self.ligand_charge = ligand_charge
        logger.info(f'The net charge of the ligand {self.name} in {self.ligand_file} is {ligand_charge}.')

    def prepare_ligand_parameters(self):
        if self.force_field == 'openff':
            self.prepare_ligand_parameters_openff()
        elif self.force_field == 'amber':
            self.prepare_ligand_parameters_amberff()
        else:
            raise ValueError(f"Unsupported force field: {self.force_field}. "
                             f"Supported force fields are: amber, openff")
        # save a json file with ligand info
        with open(f"{self.output_dir}/{self.name}.json", 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def prepare_ligand_parameters_openff(self):
        """
        Prepare ligand parameters using OpenFF toolkit.
        It will generate the ligand topology with openff toolkit.
        We will use the prmtop file in the later steps.
        """
        # First prepare the ligand parameters with AMBER force field
        # this is for building the system with tleap
        # as openff doesn't generate frcmod file for the ligand
        # we are using a fast charge method 'gas' to calculate the partial charges
        
        ligand_ff_openff = self.ligand_ff
        self.ligand_ff = 'gaff2'
        self.prepare_ligand_parameters_amberff(charge_method='gas')
        self.ligand_ff = ligand_ff_openff

        from openff.toolkit import ForceField, Molecule, Topology
        from openfe import SmallMoleculeComponent

        mol = self.name
        logger.debug(f'Preparing ligand {mol} parameters with OpenFF force field {self.ligand_ff}.')

        openff_ff = ForceField(f"{self.ligand_ff}.offxml")
        topology = Topology()
        topology.add_molecule(self.openff_molecule)

        interchange = openff_ff.create_interchange(topology,
                            charge_from_molecules=[self.openff_molecule])

        # somehow topology doesn't capture the residue info of openff_molecule
        for residue in interchange.topology.hierarchy_iterator("residues"):
            residue.residue_name = mol.lower()
        
        # add atom name from mol2 file
        for atom, atn in zip(interchange.topology.atoms, self.atomnames):
            atom.name = atn
        interchange.to_prmtop(f"{self.output_dir}/{mol}.prmtop")
        logger.info(f'Ligand {mol} OpenFF parameters prepared: {self.output_dir}/{mol}.prmtop')
        
    def prepare_ligand_parameters_amberff(self, charge_method='bcc'):
        """
        Prepare ligand parameters using AMBER force field (GAFF or GAFF2).
        It will generate the ligand topology with tleap and antechamber.
        We will use the prmtop file in the later steps.
        """
        mol = self.name
        logger.debug(f'Preparing ligand {mol} parameters with AMBER force field {self.ligand_ff}.')
        self._calculate_partial_charge()
        abspath_sdf = os.path.abspath(self.ligand_sdf_path)

        antechamber_command = f'{antechamber} -i {abspath_sdf} -fi sdf -o {self.output_dir}/{mol}_ante.mol2 -fo mol2 -c {charge_method} -s 2 -at {self.ligand_ff} -nc {self.ligand_charge} -rn {mol} -dr no'

        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(antechamber_command, working_dir=tmpdir)
        shutil.copy(f"{self.output_dir}/{mol}_ante.mol2", f"{self.output_dir}/{mol}.mol2")
        self._ligand_mol2_path = f"{self.output_dir}/{mol}.mol2"

        if self.ligand_ff == 'gaff':
            run_with_log(f'{parmchk2} -i {self.output_dir}/{mol}_ante.mol2 -f mol2 -o {self.output_dir}/{mol}.frcmod -s 1')
        elif self.ligand_ff == 'gaff2':
            run_with_log(f'{parmchk2} -i {self.output_dir}/{mol}_ante.mol2 -f mol2 -o {self.output_dir}/{mol}.frcmod -s 2')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(
                f'{antechamber} -i {self.ligand_sdf_path} -fi sdf -o {self.output_dir}/{mol}_ante.pdb -fo pdb -rn {mol} -dr no', working_dir=tmpdir)

        # copy _ante.pdb to .pdb
        shutil.copy(f"{self.output_dir}/{mol}_ante.pdb", f"{self.output_dir}/{mol}.pdb")

        # get lib file
        tleap_script = f"""
        source leaprc.protein.ff14SB
        source leaprc.{self.ligand_ff}
        lig = loadmol2 {self.output_dir}/{mol}.mol2
        loadamberparams {self.output_dir}/{mol}.frcmod
        saveoff lig {self.output_dir}/{mol}.lib
        saveamberparm lig {self.output_dir}/{mol}.prmtop {self.output_dir}/{mol}.inpcrd

        quit
        """
        with open(f"{self.output_dir}/tleap.in", 'w') as f:
            f.write(tleap_script)
        run_with_log(f"{tleap} -f tleap.in",
                        working_dir=self.output_dir)
        
        # save ligand atomnames from pdb file for future reference 
        lig_u = mda.Universe(f"{self.output_dir}/{mol}.pdb")
        self.atomnames = lig_u.atoms.names

        logger.info(f'Ligand {mol} AMBER parameters prepared: {self.output_dir}/{mol}.lib')

    def fetch_from_existing_db(self, database):
        """
        Search for the ligand in the database.

        The database is a directory containing ligand files in SDF or MOL2 format.
        """
        # first try to find name match
        db_files = [f for f in os.listdir(database) if f.endswith(('.frcmod'))]
        db_names = [os.path.splitext(f)[0] for f in db_files]
        if self.name in db_names:
            ligand_info = json.load(open(f"{database}/{self.name}.json"))
            # check if these matches
            # 1) charge type
            # 2) ligand ff
            if ligand_info['charge_type'] != self.charge:
                logger.warning(f"Ligand {self.name} found in database {database}, but charge type mismatch: {ligand_info['charge_type']} vs {self.charge}. Will re-parameterize.")
                return False
            if ligand_info['ligand_ff'] != self.ligand_ff:
                logger.warning(f"Ligand {self.name} found in database {database}, but ligand ff mismatch: {ligand_info['ligand_ff']} vs {self.ligand_ff}. Will re-parameterize.")
                return False
            # check if smiles
            if ligand_info['smiles'] != self.smiles:
                logger.warning(f"Ligand {self.name} found in database {database}, but smiles mismatch: {ligand_info['smiles']} vs {self.smiles}. Will re-parameterize.")
                return False
            if not self.retain_lig_prot and ligand_info['retain_lig_prot']:
                logger.warning(f"Ligand {self.name} found in database {database}, but retain_lig_prot mismatch: {ligand_info['retain_lig_prot']} vs {self.retain_lig_prot}. Will re-parameterize.")
                return False
            # check if SDF atom order matches
            # TODO
            suffixes = ['frcmod', 'lib', 'prmtop', 'inpcrd', 'mol2', 'pdb', 'json', 'sdf']
            for suffix in suffixes:
                if not os.path.exists(f"{database}/{self.name}.{suffix}"):
                    logger.warning(f"Ligand {self.name} found in database {database}, but {self.name}.{suffix} not found. Will re-parameterize.")
                    return False
            for suffix in suffixes:
                shutil.copy(f"{database}/{self.name}.{suffix}", f"{self.output_dir}/{self.name}.{suffix}")
            logger.info(f"Ligand {self.name} found in database {database}. Files copied to {self.output_dir}.")
            return True
        return False
            


    

class PDB_LigandProcessing(LigandProcessing):
    def _load_ligand(self):
        ligand = mda.Universe(self.ligand_file)
        self._ligand_u = ligand
        ligand_rdkit = ligand.atoms.convert_to("RDKIT")

        if ligand_rdkit is None:
            raise ValueError(f"Failed to load ligand from {self.ligand_file}"
                                " with RDKit. Check if the ligand is correct")
        if not self.retain_lig_prot:
            # remove
            ligand_rdkit = Chem.RemoveHs(ligand_rdkit)
            ligand_rdkit = Chem.AddHs(ligand_rdkit, addCoords=True)
        return ligand_rdkit

    def _get_mol_name(self):
        if self._name is not None:
            return
        self._name = self._ligand_u.atoms.resnames[0]


class SDF_LigandProcessing(LigandProcessing):
    def _load_ligand(self):
        ligand_rdkit = Chem.SDMolSupplier(
            self.ligand_file,
            removeHs=not self.retain_lig_prot)[0]
        if not self.retain_lig_prot:
            ligand_rdkit = Chem.RemoveHs(ligand_rdkit)
            ligand_rdkit = Chem.AddHs(ligand_rdkit, addCoords=True)
        else:
            if ligand_rdkit.GetNumAtoms() == ligand_rdkit.GetNumHeavyAtoms():
                logger.warning(f"Probabaly no explicit hydrogens in {self.ligand_file}."
                                f"smiles: {Chem.MolToSmiles(ligand_rdkit)}."
                                " But `retain_lig_prot` is set to True. "
                                "This may cause issues in the ligand parameterization.")
        if ligand_rdkit is None:
            raise ValueError(f"Failed to load ligand from {self.ligand_file}"
                                " with RDKit. Check if the ligand is correct")
        return ligand_rdkit

class MOL2_LigandProcessing(LigandProcessing):
    def _load_ligand(self):
        ligand_rdkit = Chem.MolFromMol2File(
            self.ligand_file,
            removeHs=not self.retain_lig_prot)
        if not self.retain_lig_prot:
            ligand_rdkit = Chem.RemoveHs(ligand_rdkit)
            ligand_rdkit = Chem.AddHs(ligand_rdkit)
        if ligand_rdkit is None:
            raise ValueError(f"Failed to load ligand from {self.ligand_file}"
                                " with RDKit. Check if the ligand is correct")
        return ligand_rdkit


class LigandFactory:
    def create_ligand(self,
                      ligand_file,
                      index,
                      output_dir,
                      ligand_name=None,
                      charge='am1bcc',
                      retain_lig_prot=True,
                      ligand_ff='gaff2',
                      unique_mol_names=[],
                    ):
        """
        Create a ligand object based on the ligand file format.
        
        Parameters
        ----------
        ligand_file : str
            The ligand file path.
        index : int
            The ligand index.
        output_dir : str
            The output directory.
        ligand_name : str, optional
            The ligand name.
            Default is None, which will be processed from the ligand file.
        charge : str, optional
            The ligand charge method. Default is ``"am1bcc"``.
            Supported charge methods (by openfe) are:
                - ``"am1bcc"``
                - ``"am1bccelf10"`` (requires OpenEye Toolkits)
                - ``"am1-mulliken"``
                - ``"mmff94"``
                - ``"gasteiger"``
        retain_lig_prot : bool, optional
            Whether to retain the ligand hydrogen atoms.
            Default is True.
        ligand_ff : str, optional
            The ligand force field. Default is ``"gaff2"``.
        unique_mol_names : list, optional
            The list of unique molecule names to avoid name conflicts.
            Default is an empty list.
        """
        if ligand_file.lower().endswith('.pdb'):
            raise ValueError("PDB file format is not supported. Please use SDF or MOL2 format.")
            return PDB_LigandProcessing(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
                unique_mol_names=unique_mol_names,
            )
        elif ligand_file.lower().endswith('.sdf'):
            return SDF_LigandProcessing(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
                unique_mol_names=unique_mol_names,
            )
        elif ligand_file.lower().endswith('.mol2'):
            return MOL2_LigandProcessing(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
                unique_mol_names=unique_mol_names,
            )
        else:
            raise ValueError(f"Unsupported ligand file format: {ligand_file}")


def batch_ligand_process(
    ligand_paths: Union[List[str], dict[str, str]],
    output_path: str,
    retain_lig_prot: bool = True,
    ligand_ph: float = 7.0,
    ligand_ff: str = 'gaff2',
    overwrite: bool = False,
    run_with_slurm: bool = False,
    max_slurm_jobs: int = 100,
    run_with_slurm_kwargs: dict = None,
    job_extra_directives: list = None,
):
    """
    Batch process ligands from a list of ligand files.
    It takes either a list of ligand files or a directory containing ligand files.
    
    Parameters
    ----------
    ligand_paths : list or dict
        A list of ligand file paths or a dictionary mapping ligand names to file paths.
    output_path : str
        The output directory to store the processed ligand files.
    retain_lig_prot : bool, optional
        Whether to retain the ligand hydrogen atoms.
        Default is True.
    ligand_ph : float, optional
        The pH value to protonate the ligand.
        Default is 7.0.
    ligand_ff : str, optional
        The ligand force field. Default is ``"gaff2"``.
    overwrite : bool, optional
        Whether to overwrite existing ligand files.
        Default is False.
    run_with_slurm : bool, optional
        Whether to run the ligand processing with SLURM.
        Default is False.
    max_slurm_jobs : int, optional
        The maximum number of SLURM jobs to submit.
        Default is 100.
    run_with_slurm_kwargs : dict, optional
        The keyword arguments for the SLURM job submission.
        Default is None.
    job_extra_directives : list, optional
        Extra directives to add to the SLURM job script.
        Default is None.
    """
     # always store a unique identifier for the ligand
    if isinstance(ligand_paths, list):
        ligand_list = {
            f'lig{i}': path
            for i, path in enumerate(ligand_paths)
        }
    else:
        #ligand_list = {ligand_name: os.path.relpath(ligand_path, output_path) for ligand_name, ligand_path in ligand_paths.items()}
        ligand_list = ligand_paths
        
    for ligand_path in ligand_list.values():
        if not os.path.exists(ligand_path):
            raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
    
    ligand_names = list(ligand_list.keys())
    logger.info(f"# {len(ligand_paths)} ligands.")

    os.makedirs(output_path, exist_ok=True)
    ligandff_folder=output_path

    from openff.toolkit.typing.engines.smirnoff.forcefield import get_available_force_fields
    available_amber_ff = ['gaff', 'gaff2']
    available_openff_ff = [ff.removesuffix(".offxml") for ff in get_available_force_fields() if 'openff' in ff]
    if ligand_ff not in available_amber_ff + available_openff_ff:
        raise ValueError(f"Unsupported force field: {ligand_ff}. "
                            f"Supported force fields are: {available_amber_ff + available_openff_ff}")

    unique_ligand_paths = {}
    for ligand_path, ligand_name in zip(ligand_paths.values(), ligand_names):
        if ligand_path not in unique_ligand_paths:
            unique_ligand_paths[ligand_path] = []
        unique_ligand_paths[ligand_path].append(ligand_name)
    logger.debug(f' Unique ligand paths: {unique_ligand_paths}')
    
    unique_mol_names = []

    mols = []
    ligands_to_be_prepared = []
    # only process the unique ligand paths
    # for ABFESystem, it will be a single ligand
    # for MBABFE and RBFE, it will be multiple ligands
    for ind, (ligand_path, ligand_names) in enumerate(unique_ligand_paths.items(), start=1):
        logger.debug(f'Processing ligand {ind}: {ligand_path} for {ligand_names}')
        # first if self.mols is not empty, then use it as the ligand name
        try:
            ligand_name = mols[ind-1]
        except:
            ligand_name = ligand_names[0]

        ligand_factory = LigandFactory()
        ligand = ligand_factory.create_ligand(
                ligand_file=ligand_path,
                index=ind,
                output_dir=ligandff_folder,
                ligand_name=ligand_name,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
                unique_mol_names=unique_mol_names
        )

        mols.append(ligand.name)
        unique_mol_names.append(ligand.name)
        if overwrite or not os.path.exists(f"{ligandff_folder}/{ligand.name}.frcmod"):
            ligands_to_be_prepared.append(ligand)
        else:
            with open(f"{ligand.output_dir}/{ligand.name}.json", 'w') as f:
                json.dump(ligand.to_dict(), f, indent=4)

    
    if len(ligands_to_be_prepared) == 0:
        logger.info("All ligands have been processed. Skip ligand parameterization.")
        return mols
        
    if run_with_slurm and len(ligands_to_be_prepared) > 0:
        logger.info('Running ligand preparation with SLURM Cluster')
        from dask_jobqueue import SLURMCluster
        from dask.distributed import Client
        from distributed.utils import TimeoutError

        log_dir = os.path.expanduser('~/.batter_jobs')
        os.makedirs(log_dir, exist_ok=True)
        n_workers = min(len(ligands_to_be_prepared), max_slurm_jobs)
        slurm_kwargs = {
            # https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
            'n_workers': n_workers,
            'queue': 'owners',
            'cores': 1,
            'memory': '5GB',
            'walltime': '00:30:00',
            'processes': 1,
            'nanny': False,
            'job_extra_directives': [
                '--job-name=batter-lig-prep',
                f'--output={log_dir}/dask-%j.out',
                f'--error={log_dir}/dask-%j.err',
            ],
            'worker_extra_args': [
                "--no-dashboard",
                "--resources prepare=1"
            ],
            # 'account': 'your_slurm_account',
        }
        if run_with_slurm_kwargs is not None:
            slurm_kwargs.update(run_with_slurm_kwargs)
        if job_extra_directives is not None:
            slurm_kwargs['job_extra_directives'].extend(job_extra_directives)

        cluster = SLURMCluster(
            **slurm_kwargs,
        )
        logger.info(f'SLURM Cluster created.')

        client = Client(cluster)
        logger.info(f'Dask Dashboard Link: {client.dashboard_link}')
        # Wait for all expected workers
        try:
            logger.info(f'Waiting for {slurm_kwargs['n_workers']} workers to start...')
            client.wait_for_workers(n_workers=n_workers, timeout=200)
        except TimeoutError:
            logger.warning(f"Timeout: Only {len(client.scheduler_info()['workers'])} workers started.")
            # scale down the cluster to the number of available workers
            if len(client.scheduler_info()['workers']) == 0:
                client.close()
                cluster.close()
                
                raise TimeoutError("No workers started in 200 sec. Check SLURM job status or run without SLURM.")
            cluster.scale(jobs=len(client.scheduler_info()['workers']))

        futures = []
        for ligand in ligands_to_be_prepared:
            logger.debug(f'Submitting lig preparation for {ligand.name}')
            fut = client.submit(
                ligand.prepare_ligand_parameters,
                pure=True,
                resources={'prepare': 1},
                key=f'prepare_{ligand.name}',
                retries=3,
            )
            futures.append(fut)
        
        logger.info(f'{len(futures)} parametrization jobs submitted to SLURM Cluster')
        logger.info('Waiting for parametrization jobs to complete...')
        _ = client.gather(futures, errors='skip')
        
        logger.info('Ligand parametrization with SLURM Cluster completed')
        client.close()
        cluster.close()
    else:
        for ligand in ligands_to_be_prepared:
            logger.debug(f'Preparing ligand {ligand.name} parameters locally.')
            ligand.prepare_ligand_parameters()

    logger.info(f"Preparing parameters for {len(ligands_to_be_prepared)} ligands with {ligand_ff} force field.")