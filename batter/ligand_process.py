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
    Convert a molecule name to a unique identifier of exactly 3 lowercase letters,
    as required. Deterministic across runs.

    Strategy:
      1) Normalize name -> keep letters only, lowercase, take up to 3.
      2) If <3 letters, pad with base-26 letters from `ind`.
      3) If collision, fall back to deterministic hash of SMILES, then perturb.
    """
    # 1) normalize to letters only (lowercase), take at most 3
    base = re.sub(r'[^a-zA-Z]', '', mol_name or '').lower()[:3]

    # 2) pad with letters from index if shorter
    if len(base) < 3:
        # use base-26 encoding of ind to fill remaining chars
        pad = _base26_triplet(ind)
        base = (base + pad)[:3]

    # If still somehow empty (e.g., mol_name had no letters), synthesize from index
    if not base:
        base = _base26_triplet(ind)

    # 3) ensure uniqueness
    if base not in exist_mol_names:
        return base

    # Collision: derive from SMILES hash; if still collides, perturb deterministically
    seed = _stable_hash_int(smiles) if smiles else _stable_hash_int(base)
    attempt = 0
    while attempt < 200:
        candidate = _base26_triplet(seed + attempt)
        if candidate not in exist_mol_names:
            return candidate
        attempt += 1

    # Extremely unlikely; last resort: use index-perturbed triplet
    return _base26_triplet(ind + attempt)

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
                database=None,
                ):

        if database is not None:
            self.database = database
            self.search_for_ligand()

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

    def search_for_ligand(self):
        """
        Search for the ligand in the database.
        """
        database = self.database
        raise NotImplementedError("Not implemented yet")

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
                      database=None,
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
        database : str, optional
            The ligand database. Default is None.
            If provided, the ligand will be searched in the database.
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
                database=database,
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
                database=database,
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
                database=database,
            )
        else:
            raise ValueError(f"Unsupported ligand file format: {ligand_file}")