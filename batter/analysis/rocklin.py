import MDAnalysis as mda
import numpy as np
import rocklinc
import shutil
import subprocess
import os
from contextlib import contextmanager
import tempfile
from pathlib import Path

@contextmanager
def suppress_output_fds(stderr=False):
    """
    Silence OS-level stdout (and optionally stderr) so child processes are quiet.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2) if stderr else None
    try:
        os.dup2(devnull_fd, 1)     # stdout -> /dev/null
        if stderr:
            os.dup2(devnull_fd, 2) # stderr -> /dev/null
        yield
    finally:
        os.dup2(saved_out, 1)
        os.close(saved_out)
        if stderr:
            os.dup2(saved_err, 2)
            os.close(saved_err)
        os.close(devnull_fd)

def run_rocklin_correction(universe, mol_name, box, lig_netq, other_netq, temp, water_model):
    """
    Compute Rocklin finite-size correction for solvation FE (kcal/mol).
    """
    apbs_exe = shutil.which("apbs")
    if apbs_exe is None:
        raise RuntimeError(
            "APBS executable not found in PATH. "
            "Install it (e.g., `conda install -c conda-forge apbs`) or set $APBS."
        )

    water_map = {
        "TIP3P": rocklinc.waters.TIP3P,
        "TIP4P": rocklinc.waters.TIP4P,
    }
    try:
        water = water_map[water_model.upper()]
    except KeyError:
        raise ValueError("Unsupported water model. Use 'TIP3P' or 'TIP4P'.")

    rc = rocklinc.RocklinCorrection(box, lig_netq, other_netq, temp, water)

    old_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # Build APBS input for the ligand selection
            rc.make_APBS_input(universe, f"resname {mol_name}")

            # Run APBS quietly (stdout+stderr). Use stderr=False if you want to see errors.
            with suppress_output_fds(stderr=True):
                rc.run_APBS(apbs_exe=apbs_exe)

            # Parse and compute correction
            rc.read_APBS()
            q = rc.compute()  # typically a pint Quantity in J/mol
    finally:
        os.chdir(old_cwd)

    # Convert cal/mol -> kcal/mol
    j_per_kcal = 1000.0
    return float(q.magnitude) / j_per_kcal