"""
This script is used to run the ABFE pipeline.

The ABFE pipeline is used to calculate the absolute binding free energy of ligand to a protein
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

from batter import MABFESystem

# --- Input ---
protein_file = 'data/7T2G_mp/protein_input.pdb'
system_file = 'data/7T2G_mp/system_input.pdb'

ligand_files = ['data/7T2G_mp/ligand_MP.pdb',
                'data/7T2G_mp/ligand_MP2.pdb',
                'data/7T2G_mp/ligand_DAMGO.pdb']

#system_coordinate = 'data/7T2G_mp/system_input.inpcrd'
system_coordinate = 'data/7T2G_mp/equilibrated.rst'
output_folder = 'test/abfe'
input_file = 'data/input_files/abfe.in'
overwrite = False

# --- Run the pipeline ---
system = MABFESystem(folder=output_folder)

system.create_system(
            system_name='7T2G',
            protein_input=protein_file,
            system_topology=system_file,
            system_coordinate=system_coordinate,
            ligand_paths=ligand_files,
            overwrite=overwrite,
            lipid_mol=['POPC'])

system.run_pipeline(
    input_file=input_file,
#    avg_struc=f'test/active_avg.pdb',
#    rmsf_file=f'test/active_rmsf.txt'
)