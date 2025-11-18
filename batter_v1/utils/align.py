"""A script to process the final structure output from batter"""

import click
import numpy as np
import MDAnalysis as mda
import tempfile
from batter_v1.utils import run_with_log
from loguru import logger
import shutil
from pathlib import Path
import sys
import os

_repo_root = Path(__file__).resolve().parents[2]
usalign = _repo_root / "batter" / "utils" / "USalign"

def align_with_usalign(input_file, reference_file, remove_residues=[], output_file=None):
    """Align the structures in the input files to the reference file"""
    u = mda.Universe(input_file)
    try:
        u.atoms.chainID = 'X'
    except AttributeError:
        pass
    temp_file = tempfile.mktemp(suffix='.pdb')
    if remove_residues:
        ag = u.select_atoms(f'not resname {" ".join(remove_residues)}')
    else:
        ag = u.atoms
    ag.write(temp_file)

    # replace basename from input_file
    if not output_file:
        output_file_align = input_file.replace('.pdb', f'_aligned')
    else:
        output_file_align = output_file.replace('.pdb', f'_aligned')
    # usalign
    reference_file =os.path.abspath(reference_file)
    run_with_log(f'{usalign} {temp_file} {reference_file} -mm 0 -ter 2 -o {temp_file.replace(".pdb", "")}',
                    working_dir=Path(temp_file).parent,
                    error_match='Cannot parse file')
    # create the folder
    Path(f'{output_file_align}.pdb').parent.mkdir(parents=True, exist_ok=True)
    shutil.move(temp_file, f'{output_file_align}.pdb')
    logger.info(f'Saved as {output_file_align}.pdb')

@click.command()
@click.option('--input_file', '-i', type=click.Path(exists=True), required=True)
@click.option('--reference_file', '-r', type=click.Path(exists=True), required=True)
@click.option('--remove_residues', '-no', multiple=True, type=str)
@click.option('--output_file', '-o', type=click.Path())
@click.option('--verbose', '-v', is_flag=True)
def aligning(input_file,
             reference_file,
             remove_residues=[],
             output_file=None,
             verbose=False):
    """Align the structures in the input files to the reference file"""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level='DEBUG')
    align_with_usalign(input_file, reference_file, remove_residues, output_file)


if __name__ == '__main__':
    aligning()
