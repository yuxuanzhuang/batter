"""
Utilities for locating bundled test fixtures.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DATA_DIR",
    "data_path",
    "TWO_CANDIDATE_SDF",
    "THREE_CANDIDATE_SDF",
    # directories
    "CONFORMATIONAL_RESTRAINTS_DIR",
    "EXTRA_RESTRAINTS_DIR",
    "LIGAND_PARAMS_DIR",
    "LIGANDS_DIR",
    "REFERENCE_DIR",
    "EQUIL_FINISHED_DIR",
    "FE_FINISHED_EXECUTION_DIR",
    # YAML fixtures
    "MABFE_YAML",
    "MABFE_LIGAND_YAML",
    "MABFE_NONMEMBRANE_YAML",
    "MABFE_END2END_YAML",
    "MABFE_GAFF2_END2END_YAML",
    "MASFE_YAML",
    "MD_YAML",
]

DATA_DIR = Path(__file__).resolve().parent


def data_path(*parts: str) -> Path:
    """
    Return an absolute path inside the ``tests/data`` directory.

    Parameters
    ----------
    *parts : str
        Relative path components appended to the data root.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested fixture.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    candidate = DATA_DIR.joinpath(*parts)
    if not candidate.exists():
        raise FileNotFoundError(f"No such test data: {candidate}")
    return candidate


TWO_CANDIDATE_SDF = data_path("ligands", "2_candidates.sdf")
THREE_CANDIDATE_SDF = data_path("ligands", "3_candidates.sdf")

CONFORMATIONAL_RESTRAINTS_DIR = data_path("conformational_restraints")
EXTRA_RESTRAINTS_DIR = data_path("extra_restraints")
LIGAND_PARAMS_DIR = data_path("ligand_params")
LIGANDS_DIR = data_path("ligands")
REFERENCE_DIR = data_path("reference")
EQUIL_FINISHED_DIR = data_path("equil_finished")
FE_FINISHED_EXECUTION_DIR = data_path("fe_finished", "executions", "rep1")

MABFE_YAML = data_path("mabfe.yaml")
MABFE_LIGAND_YAML = data_path("mabfe_ligand.yaml")
MABFE_NONMEMBRANE_YAML = data_path("mabfe_nonmembrane.yaml")
MABFE_END2END_YAML = data_path("mabfe_end2end.yaml")
MABFE_GAFF2_END2END_YAML = data_path("mabfe_gaff2_end2end.yaml")
MASFE_YAML = data_path("masfe_nofe_type.yaml")
MD_YAML = data_path("md.yaml")
