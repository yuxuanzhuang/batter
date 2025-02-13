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

def random_three_letter_name():
    return ''.join(random.choices(string.ascii_lowercase, k=3))

def _convert_mol_name_to_unique(mol_name, ind, mol_names):
    """
    Convert the molecule name to a unique name that is 
    at most 3 lowercase characters, as required by AMBER.
    """
    # Ensure lowercase and truncate to 3 characters
    mol_name = mol_name.lower()
    
    # Handle short names
    if len(mol_name) == 1:
        mol_name = f"{mol_name}{ind:02d}"[:3]  # Ensure at most 3 chars
    elif len(mol_name) == 2:
        mol_name = f"{mol_name}{ind}"[:3]  # Ensure at most 3 chars
    else:
        mol_name = mol_name[:3]  # Truncate longer names

    # Ensure uniqueness
    if mol_name in mol_names:
        for _ in range(100):  # Try up to 100 times
            new_name = random_three_letter_name()
            if new_name not in mol_names:
                mol_name = new_name
                break
        else:
            raise ValueError("Failed to generate a unique 3-letter name after 100 tries.")

    return mol_name

class Ligand(ABC):
    """
    Base class for ligand processing.
    It will read the ligand file, calculate the
    partial charges, and generate the ligand
    topology in AMBER format.
    """
    def __init__(self,
                ligand_file,
                index,
                output_dir,
                ligand_name=None,
                charge='am1bcc',
                retain_lig_prot=True,
                ligand_ff='gaff2',
                ):
        self.ligand_file = ligand_file
        self.index = index
        self.output_dir = os.path.abspath(output_dir)
        self._name = ligand_name.lower() if ligand_name is not None else None
        self.charge = charge
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ff = ligand_ff
        self._load_ligand()

    def generate_unique_name(self, exist_mol_names=[]):
        self._get_mol_name()
        self._name = _convert_mol_name_to_unique(
            self._name,
            self.index,
            exist_mol_names)
        logger.info(f'Ligand {self.index}: {self.name}')
    
    @abstractmethod
    def _load_ligand(self):
        pass

    def _get_mol_name(self):
        if self._name is not None:
            return
        if self._ligand_object.HasProp('_Name'):
            mol_name = self._ligand_object.GetProp('_Name')
        else:
            mol_name = os.path.basename(self.ligand_file).split('.')[0].lower()
        self._name = mol_name

    @property
    def name(self):
        return self._name

    def _write_sdf(self):
        # Assign stereochemistry explicitly
        #Chem.AssignAtomChiralTagsFromStructure(self._ligand_object)
        #Chem.AssignStereochemistry(self._ligand_object, cleanIt=True, force=True)
        for i, a in enumerate(self._ligand_object.GetAtoms()):
            a.SetAtomMapNum(i)
        sdf_file = os.path.join(self.output_dir, f'{self.name}.sdf')
        w = Chem.SDWriter(sdf_file)
        w.write(self._ligand_object)
        w.close()
        self._ligand_sdf_path = sdf_file

    def _calculate_partial_charge(self):
        self._write_sdf()
        molecule = Molecule(self._ligand_sdf_path)
        molecule.assign_partial_charges(
            partial_charge_method=self.charge,
        )
        ligand_charge = np.round(np.sum([charge._magnitude
            for charge in molecule.partial_charges]))
        self.ligand_charge = ligand_charge
        logger.info(f'The net charge of the ligand {self.name} in {self.ligand_file} is {ligand_charge}')
        molecule.to_file(self._ligand_sdf_path, file_format='sdf')

    def prepare_ligand_parameters_sdf(self):
        self._calculate_partial_charge()
        mol = self.name
        logger.debug(f'Preparing ligand {mol} parameters')
        abspath_sdf = os.path.abspath(self._ligand_sdf_path)
        antechamber_command = f'{antechamber} -i {abspath_sdf} -fi sdf -o {self.output_dir}/{mol}_ante.mol2 -fo mol2 -c bcc -s 2 -at {self.ligand_ff} -nc {self.ligand_charge} -rn {mol}'

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
                f'{antechamber} -i {self._ligand_sdf_path} -fi sdf -o {self.output_dir}/{mol}_ante.pdb -fo pdb -rn {mol}', working_dir=tmpdir)

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


class PDB_Ligand(Ligand):
    def _load_ligand(self):
        ligand = mda.Universe(self.ligand_file)
        self._ligand_u = ligand
        self._ligand_object = ligand.atoms.convert_to("RDKIT")

    def _get_mol_name(self):
        self._name = self._ligand_u.atoms.resnames[0]


class SDF_Ligand(Ligand):
    def _load_ligand(self):
        ligand = Chem.MolFromMolFile(
                self.ligand_file,
                removeHs=not self.retain_lig_prot)
        if not self.retain_lig_prot:
            ligand = Chem.AddHs(ligand)
        self._ligand_object = ligand


class MOL2_Ligand(Ligand):
    def _load_ligand(self):
        ligand = Chem.MolFromMol2File(
            self.ligand_file,
            removeHs=not self.retain_lig_prot)
        if not self.retain_lig_prot:
            ligand = Chem.AddHs(ligand)
        self._ligand_object = ligand


class LigandFactory:
    def create_ligand(self,
                      ligand_file,
                      index,
                      output_dir,
                      ligand_name=None,
                      charge='am1bcc',
                      retain_lig_prot=True,
                      ligand_ff='gaff2',
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
        """
        if ligand_file.lower().endswith('.pdb'):
            return PDB_Ligand(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
            )
        elif ligand_file.lower().endswith('.sdf'):
            return SDF_Ligand(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
            )
        elif ligand_file.lower().endswith('.mol2'):
            return MOL2_Ligand(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
            )
        else:
            raise ValueError(f"Unsupported ligand file format: {ligand_file}")