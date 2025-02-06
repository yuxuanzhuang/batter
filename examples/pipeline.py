"""
This script is used to run the pipeline.

The pipeline is used to calculate the absolute binding free energy of ligand to a protein
in a lipid bilayer.

Input:

protein_file: str
    Path to the protein file in PDB format.
    It should be exported from Maestro,
    which means the protonation states of the protein are assigned.
    Water and ligand can be present in the file,
    but they will be removed during preparation.

system_file: str
    PDB file of a prepared simulation system with `dabble`.
    The ligand does not need to be present.
    It will be used to generate the topology of the system.

ligand_files: list of str
    List of ligand files. It can be either PDB or mol2 format.

system_coordinate: str
    The coordinate file for the system.
    The coordiantes and box dimensions will be used for the system.
    It can be an INPCRD file prepared from `dabble` or
    it can be a snapshot of the equilibrated system.
    If it is not provided, the coordinates from the system_topology
    will be used if available.

output_folder: str
    Path to the output folder.
    The folder will be created if it does not exist.

input_file: str
    Path to the input file for the ABFE calculation
    that stores the simulation configuration.

overwrite: bool
    If True, the output folder will be overwritten if it exists.
"""

from batter import MABFESystem, RBFESystem
import os
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run MABFE System with specified parameters.")
parser.add_argument('--state', required=True, help='State of the system')
parser.add_argument('--rep', required=True, help='Replica identifier')
parser.add_argument('--bfe', required=True, help='Type of binding free energy calculation')
parser.add_argument('--overwrite', default='False', help='Overwrite existing data (True/False)')
parser.add_argument('--only_equil', default='False', help='Only run the equilibration step (True/False)')
args = parser.parse_args()

# --- Input ---
state = args.state
rep = args.rep
bfe_type = args.bfe
overwrite = args.overwrite == 'True'
only_equil = args.only_equil == 'True'
print(f'Running {bfe_type} for {state} replica {rep}'
      f' with overwrite={overwrite} and only_equil={only_equil}')

protein_file = f'inputs/{state}/protein_input.pdb'
system_file = f'inputs/{state}/{state}.pdb'

ligand_files = [f'inputs/{state}/ligand_mp.pdb',
                f'inputs/{state}/ligand_fen.pdb',
                f'inputs/{state}/ligand_fna.pdb',
                f'inputs/{state}/ligand_lof.pdb',
                f'inputs/{state}/ligand_buf.pdb',
]

#system_inpcrd = 'data/7T2G_mp/system_input.inpcrd'
equilibrated_rst = f'inputs/{state}/{state}_eq.rst'

if bfe_type == 'ABFE':
    input_file = f'inputs/{state}/abfe.in'
elif bfe_type == 'RBFE':
    input_file = f'inputs/{state}/rbfe.in'

avg_struc = f'inputs/unbiased/{state}_avg.pdb'
rmsf_file = f'inputs/unbiased/{state}_rmsf.txt'

# make sure everyfile exists
for file in [protein_file, system_file, equilibrated_rst, input_file, avg_struc, rmsf_file] + ligand_files:
    assert os.path.exists(file), f'{file} does not exist'
    
if bfe_type == 'ABFE':
    output_folder = f'ABFE/{state}_rep{rep}'
    system = MABFESystem(folder=output_folder)
elif bfe_type == 'RBFE':
    output_folder = f'RBFE/{state}_rep{rep}'
    system = RBFESystem(folder=output_folder)
else:
    raise ValueError(f'Unknown BFE type: {bfe_type}; available options are ABFE and RBFE')

system.create_system(
            system_name='MOR',
            protein_input=protein_file,
            system_topology=system_file,
            system_coordinate=equilibrated_rst,
            ligand_paths=ligand_files,
            overwrite=overwrite,
            lipid_mol=['POPC'])

system.run_pipeline(
    input_file=input_file,
    avg_struc=avg_struc,
    rmsf_file=rmsf_file,
    only_equil=only_equil,
)