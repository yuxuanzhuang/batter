import MDAnalysis as mda
import numpy as np
import rocklinc
import shutil
import subprocess
import os
import sys
from contextlib import contextmanager
import tempfile

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def run_rocklin_correction(universe, mol_name, box, lig_netq, other_netq, temp, water_model):
    """
    Run Rocklin correction for a given system.

    Parameters
    ----------
    universe : MDAnalysis Universe
        The MDAnalysis universe containing the system.
    mol_name : str
        The residue name of the ligand.
    box : array-like
        The average simulation box dimensions.
    lig_netq : float
        The net charge of the ligand.
    other_netq : float
        The net charge of the rest of the system.
    temp : float
        The temperature in Kelvin.
    water_model : str
        The water model used ('TIP3P' or 'TIP4P').

    Returns
    -------
    float
        The Rocklin correction in kcal/mol.
    """
    apbs_exe = shutil.which("apbs")
    if apbs_exe is None:
        raise RuntimeError("APBS executable not found in PATH"
                            "Please install APBS and ensure it is in your PATH.")

    if water_model == 'TIP3P':
        water = rocklinc.waters.TIP3P
    elif water_model == 'TIP4P':
        water = rocklinc.waters.TIP4P
    else:
        raise ValueError("Unsupported water model. Use 'TIP3P' or 'TIP4P'.")

    rc = rocklinc.RocklinCorrection(box, lig_netq, other_netq, temp,
                                water)
    with tempfile.TemporaryDirectory() as tmpdir, suppress_output():
        os.chdir(tmpdir)
        rc.make_APBS_input(universe, f'resname {mol_name}')
        rc.run_APBS(
            apbs_exe=apbs_exe,
            #apbs_exe='/scratch/users/yuzhuang/miniforge3_0808/envs/batter_dev/bin/apbs'
        )
        rc.read_APBS()
        result = rc.compute()
        # return kcal/mol
    return result.magnitude / 1000