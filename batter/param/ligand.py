"""Ligand parameterisation helpers for GAFF/GAFF2 and OpenFF workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
from gufe import SmallMoleculeComponent
from loguru import logger
from MDAnalysis import Universe
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff.forcefield import get_available_force_fields
from rdkit import Chem

warnings.filterwarnings("ignore", category=UserWarning, module="gufe")

from batter.utils import (
    antechamber,
    parmchk2,
    run_with_log,
    tleap,
)

__all__ = [
    "LigandProcessing",
    "PDB_LigandProcessing",
    "SDF_LigandProcessing",
    "MOL2_LigandProcessing",
    "LigandFactory",
    "batch_ligand_process",
]

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

# forbidden molecule names in MDA selection language and AMBER residue names
FORBIDDEN_MOL_NAMES = {"add", "all", "and", "any", "not"}


def _base26_triplet(n: int) -> str:
    """Map nonnegative int → 3 lowercase letters (base-26, a..z)."""
    n = n % (26**3)
    a = n // (26**2)
    b = (n // 26) % 26
    c = n % 26
    return chr(a + 97) + chr(b + 97) + chr(c + 97)


def _stable_hash_int(s: str) -> int:
    """Deterministic int from string (md5, not Python's salted hash)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16)


def _convert_mol_name_to_unique(
    mol_name: str,
    ind: int,
    smiles: str,
    exist_mol_names: set[str],
) -> str:
    """
    Convert a molecule name to a unique 3-char identifier (letters/digits),
    deterministic across runs, and never all digits.

    Parameters
    ----------
    mol_name
        Suggested name.
    ind
        Index used as a deterministic tiebreaker.
    smiles
        Canonical SMILES used to derive a stable hash on collision.
    exist_mol_names
        Set of names already taken.

    Returns
    -------
    str
        Unique 3-character residue name.
    """
    base = re.sub(r"[^a-zA-Z0-9]", "", (mol_name or "")).lower()[:3]
    if len(base) < 3:
        base = (base + _base26_triplet(ind))[:3]
    if not base:
        base = _base26_triplet(ind)
    if len(base) == 3 and base.isdigit():
        base = "l" + base[:2]

    if base not in exist_mol_names and base not in FORBIDDEN_MOL_NAMES:
        return base

    seed = _stable_hash_int(smiles or base)
    for attempt in range(200):
        candidate = _base26_triplet(seed + attempt)
        if candidate not in exist_mol_names and candidate not in FORBIDDEN_MOL_NAMES:
            return candidate

    for attempt in range(200):
        candidate = _base26_triplet(ind + attempt)
        if candidate not in exist_mol_names and candidate not in FORBIDDEN_MOL_NAMES:
            return candidate

    raise ValueError(
        f"Could not derive unique 3-char name for molecule {mol_name!r} (smiles: {smiles})"
    )


def _ensure_sdf_internal_name(sdf_path: Union[str, Path], name: str) -> None:
    """Normalize the ``_Name`` property of an SDF file.

    Parameters
    ----------
    sdf_path
        Input SDF file.
    name
        Value written to the ``_Name`` header field.
    """
    p = str(sdf_path)
    suppl = Chem.SDMolSupplier(p, removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        logger.warning(f"[ligand] Could not read SDF to set name: {p}")
        return
    mol = mols[0]
    try:
        mol.SetProp("_Name", name)
    except Exception as e:
        logger.warning(f"[ligand] Could not set _Name for {p}: {e}")
        return
    w = Chem.SDWriter(p)
    w.write(mol)
    w.close()
    logger.debug(f"[ligand] Set SDF _Name to '{name}' in {Path(p).name}")


def _rdkit_load(path: Union[str, Path], retain_h: bool) -> Chem.Mol:
    """
    Load a ligand file with RDKit (SDF/MOL2). Raise on failure.

    Parameters
    ----------
    path
        Input file path.
    retain_h
        Whether to preserve explicit hydrogens from the file.

    Returns
    -------
    Chem.Mol
        RDKit molecule.
    """
    p = str(path)
    mol: Optional[Chem.Mol] = None
    if p.lower().endswith(".sdf"):
        suppl = Chem.SDMolSupplier(p, removeHs=not retain_h)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
    elif p.lower().endswith(".mol2"):
        mol = Chem.MolFromMol2File(p, removeHs=not retain_h)
    else:
        raise ValueError(
            f"Unsupported ligand format for hashing: {p} (use .sdf or .mol2)"
        )

    if mol is None:
        raise ValueError(f"RDKit failed to load {p}")
    # Normalize: ensure a consistent H state for hashing
    if not retain_h:
        mol = Chem.RemoveHs(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def _canonical_payload(mol: Chem.Mol) -> str:
    """
    Canonical text used for hashing: canonical SMILES with explicit Hs.

    Parameters
    ----------
    mol
        RDKit molecule (with Hs added).

    Returns
    -------
    str
        Canonical SMILES (isomeric).
    """
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _hash_id(payload: str, ligand_ff: str, retain_h: bool) -> str:
    """
    Build a short, stable content hash for (mol, ff, retain flag).

    The final id is 12 hex chars of SHA256.

    """
    h = hashlib.sha256()
    h.update(payload.encode("utf-8"))
    h.update(f"|ff={ligand_ff}|retain={int(retain_h)}".encode("utf-8"))
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# core                                                                         #
# --------------------------------------------------------------------------- #


class LigandProcessing(ABC):
    """
    Base class for ligand processing and parameterization.

    It loads a ligand, determines a unique residue/name, estimates the charge,
    and generates AMBER/OpenFF parameters.

    Parameters
    ----------
    ligand_file
        Input ligand path (SDF/MOL2/PDB depending on subclass).
    index
        1-based index used for stable name generation.
    output_dir
        Output folder for generated files.
    ligand_name
        Optional preferred name; will be uniquified to 3 chars.
    charge
        Charge method for OpenFF pre-charge or quick estimate (e.g., ``"am1bcc"``).
    retain_lig_prot
        If ``True``, keep hydrogen atoms from input.
    ligand_ff
        One of ``"gaff"`` or ``"gaff2"`` or an OpenFF release like ``"openff-2.2.0"``.
    unique_mol_names
        Existing names to avoid collisions.

    Attributes
    ----------
    ligand_object : SmallMoleculeComponent
    openff_molecule : Molecule
    ligand_charge : float
        Estimated total charge (integer).
    atomnames : list[str]
        Atom names extracted from generated PDB (AMBER path).
    """

    def __init__(
        self,
        ligand_file: Union[str, Path],
        index: int,
        output_dir: Union[str, Path],
        ligand_name: Optional[str] = None,
        charge: str = "am1bcc",
        retain_lig_prot: bool = True,
        ligand_ff: str = "gaff2",
        unique_mol_names: Optional[List[str]] = None,
    ) -> None:
        self.ligand_file = str(ligand_file)
        self.index = index
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = str(Path(output_dir).absolute())
        self._name = ligand_name.lower() if ligand_name is not None else None
        self.charge = charge
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ff = ligand_ff

        available_amber_ff = ["gaff", "gaff2"]
        available_openff_ff = [
            ff.removesuffix(".offxml")
            for ff in get_available_force_fields()
            if "openff" in ff
        ]
        if ligand_ff in available_amber_ff:
            self.force_field = "amber"
        elif ligand_ff in available_openff_ff:
            self.force_field = "openff"
        else:
            raise ValueError(
                f"Unsupported force field: {ligand_ff}. "
                f"Supported: {available_amber_ff + available_openff_ff}"
            )
        logger.debug(
            "Using force field {} for ligand {}", self.force_field, ligand_file
        )

        self.unique_mol_names = set(unique_mol_names or [])
        ligand_rdkit = self._load_ligand()
        self._cano_smiles = Chem.MolToSmiles(ligand_rdkit, canonical=True)

        if ligand_name is None:
            ligand = SmallMoleculeComponent(ligand_rdkit)
        else:
            # skip warning
            ligand = SmallMoleculeComponent(ligand_rdkit, name=ligand_name)

        self.ligand_object = ligand
        self.openff_molecule = ligand.to_openff()
        self._generate_unique_name()

    # -------------------- name / io helpers --------------------

    def _generate_unique_name(self) -> None:
        """Derive a unique residue name for the ligand."""
        self._get_mol_name()
        self._name = _convert_mol_name_to_unique(
            self._name, self.index, self.smiles, self.unique_mol_names
        )
        logger.debug("Ligand {}: {}", self.index, self.name)
        self.openff_molecule.name = self.name.lower()

        # needed for residue construction
        self.openff_molecule.add_default_hierarchy_schemes()
        self.openff_molecule.residues[0].residue_name = self.name.lower()

        self.openff_molecule.to_file(self.ligand_sdf_path, file_format="sdf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_file": self.ligand_file,
            "charge_type": self.charge,
            "smiles": self.smiles,
            "ligand_sdf_path": self.ligand_sdf_path,
            "ligand_charge": getattr(self, "ligand_charge", None),
            "ligand_ff": self.ligand_ff,
            "ligand_name": self.name,
            "retain_lig_prot": self.retain_lig_prot,
        }

    @abstractmethod
    def _load_ligand(self) -> Chem.Mol:
        """Subclasses must load and return an RDKit molecule."""
        raise NotImplementedError

    def _get_mol_name(self) -> None:
        """Populate ``self._name`` from user input or the source file."""
        if self._name is not None:
            return
        if self.ligand_object.name is not None:
            mol_name = self.ligand_object.name
        else:
            mol_name = Path(self.ligand_file).stem.lower()
        self._name = mol_name

    @property
    def name(self) -> str:
        """str: Three-character residue name used for generated artifacts."""
        return str(self._name)

    @property
    def smiles(self) -> str:
        """str: Canonical SMILES with explicit hydrogens."""
        return self._cano_smiles

    @property
    def ligand_sdf_path(self) -> str:
        """str: Path to the canonicalised SDF stored on disk."""
        return str(Path(self.output_dir) / f"{self.name}.sdf")

    # -------------------- parameterization --------------------

    def _calculate_partial_charge(self) -> None:
        """Estimate the net charge using the configured partial-charge method."""
        molecule = self.openff_molecule
        charge_method = self.charge if self.force_field == "openff" else "gasteiger"
        molecule.assign_partial_charges(partial_charge_method=charge_method)
        ligand_charge = np.round(
            np.sum([charge._magnitude for charge in molecule.partial_charges])
        )
        self.ligand_charge = float(ligand_charge)
        logger.debug(
            "Net charge of ligand {} in {} is {}.",
            self.name,
            self.ligand_file,
            self.ligand_charge,
        )

    def prepare_ligand_parameters(self) -> None:
        """
        Generate parameters using either AMBER (GAFF/GAFF2) or OpenFF path.

        Notes
        -----
        - OpenFF path first creates AMBER artifacts for tleap-based system build.
        - Writes a ``<name>.json`` metadata file to the output folder.
        """
        if self.force_field == "openff":
            self.prepare_ligand_parameters_openff()
        elif self.force_field == "amber":
            self.prepare_ligand_parameters_amberff()
        else:
            raise ValueError("Unsupported force field; expected 'amber' or 'openff'.")

        metadata = self.to_dict()
        charge = metadata.get("ligand_charge")
        json_path = Path(self.output_dir) / f"{self.name}.json"
        with json_path.open("w") as f:
            json.dump(metadata, f, indent=4)

    def prepare_ligand_parameters_openff(self) -> None:
        """
        Prepare ligand parameters using OpenFF toolkit (and AMBER bootstrap).

        Behavior
        --------
        - Runs a **fast** AMBER bootstrap (GAFF2 + gas charges) so tleap artifacts exist.
        - Generates an OpenFF `prmtop` for downstream if you prefer OpenMM/OpenFF.
        """
        # Bootstrap via AMBER with fast charges
        ligand_ff_openff = self.ligand_ff
        self.ligand_ff = "gaff2"
        self.prepare_ligand_parameters_amberff(charge_method="gas")
        self.ligand_ff = ligand_ff_openff

        from openff.toolkit import ForceField, Topology

        mol = self.name
        logger.debug(
            "Preparing ligand {} parameters with OpenFF force field {}.",
            mol,
            self.ligand_ff,
        )

        openff_ff = ForceField(f"{self.ligand_ff}.offxml")
        topology = Topology()
        topology.add_molecule(self.openff_molecule)
        interchange = openff_ff.create_interchange(
            topology, charge_from_molecules=[self.openff_molecule]
        )

        # ensure residue + atom names are present
        for residue in interchange.topology.hierarchy_iterator("residues"):
            residue.residue_name = mol.lower()

        for atom, atn in zip(interchange.topology.atoms, self.atomnames):
            atom.name = atn

        interchange.to_prmtop(f"{self.output_dir}/{mol}.prmtop")
        logger.debug(
            f"Ligand {mol} OpenFF parameters prepared: {self.output_dir}/{mol}.prmtop",
        )

    def prepare_ligand_parameters_amberff(self, charge_method: str = "bcc") -> None:
        """
        Prepare ligand parameters using AMBER (GAFF/GAFF2): mol2/frcmod/lib/prmtop.

        Parameters
        ----------
        charge_method
            Antechamber charge method (e.g., ``"bcc"`` or ``"gas"``).
        """
        mol = self.name
        logger.debug(
            "Preparing ligand {} parameters with AMBER force field {}.",
            mol,
            self.ligand_ff,
        )
        self._calculate_partial_charge()
        abspath_sdf = str(Path(self.ligand_sdf_path).absolute())

        ante_cmd = (
            f"{antechamber} -i {abspath_sdf} -fi sdf -o {self.output_dir}/{mol}_ante.mol2 "
            f"-fo mol2 -c {charge_method} -s 2 -at {self.ligand_ff} -nc {int(self.ligand_charge)} "
            f"-rn {mol} -dr no"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(ante_cmd, working_dir=tmpdir)

        shutil.copy(
            f"{self.output_dir}/{mol}_ante.mol2", f"{self.output_dir}/{mol}.mol2"
        )
        self._ligand_mol2_path = f"{self.output_dir}/{mol}.mol2"

        if self.ligand_ff == "gaff":
            run_with_log(
                f"{parmchk2} -i {self.output_dir}/{mol}_ante.mol2 -f mol2 -o {self.output_dir}/{mol}.frcmod -s 1"
            )
        elif self.ligand_ff == "gaff2":
            run_with_log(
                f"{parmchk2} -i {self.output_dir}/{mol}_ante.mol2 -f mol2 -o {self.output_dir}/{mol}.frcmod -s 2"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_with_log(
                f"{antechamber} -i {self.ligand_sdf_path} -fi sdf -o {self.output_dir}/{mol}_ante.pdb -fo pdb -rn {mol} -dr no",
                working_dir=tmpdir,
            )
        shutil.copy(f"{self.output_dir}/{mol}_ante.pdb", f"{self.output_dir}/{mol}.pdb")

        tleap_script = f"""
source leaprc.protein.ff14SB
source leaprc.{self.ligand_ff}
lig = loadmol2 {self.output_dir}/{mol}.mol2
loadamberparams {self.output_dir}/{mol}.frcmod
saveoff lig {self.output_dir}/{mol}.lib
saveamberparm lig {self.output_dir}/{mol}.prmtop {self.output_dir}/{mol}.inpcrd
quit
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "tleap.in"
            p.write_text(tleap_script)
            run_with_log(f"{tleap} -f {p.name}", working_dir=tmpdir)

        # atom names from generated pdb
        lig_u = Universe(f"{self.output_dir}/{mol}.pdb")
        self.atomnames = list(lig_u.atoms.names)

        logger.debug(
            "Ligand {} AMBER parameters prepared: {}/{}.lib", mol, self.output_dir, mol
        )

    # -------------------- DB lookup --------------------

    def fetch_from_existing_db(self, database: Union[str, Path]) -> bool:
        """
        Search and copy ligand artifacts from a local database.

        Parameters
        ----------
        database
            Directory containing ``<name>.(frcmod|lib|prmtop|inpcrd|mol2|pdb|json|sdf)``.

        Returns
        -------
        bool
            ``True`` if a full, matching entry was found and copied.
        """
        db = Path(database)
        if not db.exists():
            return False

        db_files = [f for f in db.iterdir() if f.suffix == ".frcmod"]
        db_names = [f.stem for f in db_files]

        if self.name not in db_names:
            return False

        ligand_info = json.loads((db / f"{self.name}.json").read_text())
        if ligand_info.get("charge_type") != self.charge:
            logger.warning(
                "Ligand {} found in DB {}, but charge mismatch: {} vs {}. Will re-parameterize.",
                self.name,
                db,
                ligand_info.get("charge_type"),
                self.charge,
            )
            return False
        if ligand_info.get("ligand_ff") != self.ligand_ff:
            logger.warning(
                "Ligand {} found in DB {}, but FF mismatch: {} vs {}. Will re-parameterize.",
                self.name,
                db,
                ligand_info.get("ligand_ff"),
                self.ligand_ff,
            )
            return False
        if ligand_info.get("smiles") != self.smiles:
            logger.warning(
                "Ligand {} found in DB {}, but SMILES mismatch: {} vs {}. Will re-parameterize.",
                self.name,
                db,
                ligand_info.get("smiles"),
                self.smiles,
            )
            return False
        if not self.retain_lig_prot and ligand_info.get("retain_lig_prot"):
            logger.warning(
                "Ligand {} found in DB {}, but retain_lig_prot mismatch: {} vs {}. Will re-parameterize.",
                self.name,
                db,
                ligand_info.get("retain_lig_prot"),
                self.retain_lig_prot,
            )
            return False

        suffixes = ["frcmod", "lib", "prmtop", "inpcrd", "mol2", "pdb", "json", "sdf"]
        for suffix in suffixes:
            if not (db / f"{self.name}.{suffix}").exists():
                logger.warning(
                    "Ligand {} found in DB {}, but missing {}. Will re-parameterize.",
                    self.name,
                    db,
                    f"{self.name}.{suffix}",
                )
                return False

        for suffix in suffixes:
            shutil.copy(
                db / f"{self.name}.{suffix}",
                Path(self.output_dir) / f"{self.name}.{suffix}",
            )

        logger.debug(
            "Ligand {} found in database {}. Files copied to {}.",
            self.name,
            db,
            self.output_dir,
        )
        return True


# --------------------------------------------------------------------------- #
# concrete loaders                                                             #
# --------------------------------------------------------------------------- #


class PDB_LigandProcessing(LigandProcessing):
    def _load_ligand(self) -> Chem.Mol:
        lig_u = Universe(self.ligand_file)
        self._ligand_u = lig_u
        ligand_rdkit = lig_u.atoms.convert_to("RDKIT")
        if ligand_rdkit is None:
            raise ValueError(
                f"Failed to load ligand from {self.ligand_file} with RDKit. Check the input."
            )
        if not self.retain_lig_prot:
            ligand_rdkit = Chem.RemoveHs(ligand_rdkit)
            ligand_rdkit = Chem.AddHs(ligand_rdkit, addCoords=True)
        return ligand_rdkit

    def _get_mol_name(self) -> None:
        if self._name is not None:
            return
        self._name = self._ligand_u.atoms.resnames[0]  # type: ignore[attr-defined]


class SDF_LigandProcessing(LigandProcessing):
    def _load_ligand(self) -> Chem.Mol:
        supplier = Chem.SDMolSupplier(
            self.ligand_file, removeHs=not self.retain_lig_prot
        )
        ligand_rdkit = supplier[0] if supplier and len(supplier) > 0 else None
        if not self.retain_lig_prot and ligand_rdkit is not None:
            ligand_rdkit = Chem.RemoveHs(ligand_rdkit)
            ligand_rdkit = Chem.AddHs(ligand_rdkit, addCoords=True)
        else:
            if (
                ligand_rdkit is not None
                and ligand_rdkit.GetNumAtoms() == ligand_rdkit.GetNumHeavyAtoms()
            ):
                logger.warning(
                    "Probably no explicit H in {} (SMILES: {}). retain_lig_prot=True may cause issues.",
                    self.ligand_file,
                    Chem.MolToSmiles(ligand_rdkit) if ligand_rdkit else "<none>",
                )
        if ligand_rdkit is None:
            raise ValueError(
                f"Failed to load ligand from {self.ligand_file} with RDKit. Check the input."
            )
        return ligand_rdkit


class MOL2_LigandProcessing(LigandProcessing):
    def _load_ligand(self) -> Chem.Mol:
        ligand_rdkit = Chem.MolFromMol2File(
            self.ligand_file, removeHs=not self.retain_lig_prot
        )
        if not self.retain_lig_prot and ligand_rdkit is not None:
            ligand_rdkit = Chem.RemoveHs(ligand_rdkit)
            ligand_rdkit = Chem.AddHs(ligand_rdkit)
        if ligand_rdkit is None:
            raise ValueError(
                f"Failed to load ligand from {self.ligand_file} with RDKit. Check the input."
            )
        return ligand_rdkit


# --------------------------------------------------------------------------- #
# factory + batch                                                              #
# --------------------------------------------------------------------------- #


class LigandFactory:
    """
    Factory that chooses the appropriate loader/processor by file extension.
    """

    def create_ligand(
        self,
        ligand_file: Union[str, Path],
        index: int,
        output_dir: Union[str, Path],
        ligand_name: Optional[str] = None,
        charge: str = "am1bcc",
        retain_lig_prot: bool = True,
        ligand_ff: str = "gaff2",
        unique_mol_names: Optional[List[str]] = None,
    ) -> LigandProcessing:
        """Instantiate a concrete :class:`LigandProcessing` subclass.

        Parameters
        ----------
        ligand_file, index, output_dir, ligand_name, charge, retain_lig_prot,
        ligand_ff, unique_mol_names
            Forwarded to the underlying processor.

        Returns
        -------
        LigandProcessing
            Processor configured for the detected file type.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        """
        path = str(ligand_file).lower()
        if path.endswith(".pdb"):
            raise ValueError("PDB ligand input is not supported. Use SDF or MOL2.")
        if path.endswith(".sdf"):
            return SDF_LigandProcessing(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
                unique_mol_names=unique_mol_names or [],
            )
        if path.endswith(".mol2"):
            return MOL2_LigandProcessing(
                ligand_file=ligand_file,
                index=index,
                output_dir=output_dir,
                ligand_name=ligand_name,
                charge=charge,
                retain_lig_prot=retain_lig_prot,
                ligand_ff=ligand_ff,
                unique_mol_names=unique_mol_names or [],
            )
        raise ValueError(f"Unsupported ligand file format: {ligand_file!r}")


def batch_ligand_process(
    ligand_paths: Union[List[str], Dict[str, str]],
    output_path: Union[str, Path],
    retain_lig_prot: bool = True,
    ligand_ph: float = 7.0,
    ligand_ff: str = "gaff2",
    charge_method: str = "am1bcc",
    overwrite: bool = False,
    run_with_slurm: bool = False,
    max_slurm_jobs: int = 50,
    run_with_slurm_kwargs: Optional[Dict[str, Any]] = None,
    job_extra_directives: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
    """Parameterise ligands into a content-addressed store.

    Artifacts for each ligand are written under::

        <output_path>/<hash_id>/*

    where ``hash_id = sha256(canonical_smiles + ligand_ff + retain).hexdigest()[:12]``.

    Parameters
    ----------
    ligand_paths
        List of file paths or mapping {alias: path}. Only the file path affects hashing.
    output_path
        Output directory for the content-addressed store.
    retain_lig_prot
        Whether to retain hydrogens from inputs.
    ligand_ph
        Target protonation pH (reserved for future use).
    ligand_ff
        Force field ('gaff'/'gaff2' or a valid OpenFF release name).
    charge_method
        Charge method for ligand.
    overwrite
        If True, re-parameterize even if <hash_id> already exists.
    run_with_slurm
        If True, distribute parametrization with Dask+SLURM (same behavior as before).
    max_slurm_jobs, run_with_slurm_kwargs, job_extra_directives
        SLURM/Dask configuration.

    Returns
    -------
    list of str
        Hash identifiers in processing order (duplicates preserved).
    dict
        Mapping from the provided input path to ``(hash_id, canonical_smiles)``.
    """
    # --- normalize inputs ---
    if isinstance(ligand_paths, list):
        lig_map: Dict[str, str] = {f"lig{i}": p for i, p in enumerate(ligand_paths)}
    else:
        lig_map = dict(ligand_paths)

    for lp in lig_map.values():
        if not Path(lp).exists():
            raise FileNotFoundError(f"Ligand file not found: {lp}")

    out_root = Path(output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    available_amber_ff = ["gaff", "gaff2"]
    available_openff_ff = [
        ff.removesuffix(".offxml")
        for ff in get_available_force_fields()
        if "openff" in ff
    ]
    if ligand_ff not in (available_amber_ff + available_openff_ff):
        raise ValueError(
            f"Unsupported force field: {ligand_ff}. "
            f"Supported: {available_amber_ff + available_openff_ff}"
        )

    # --- compute content hashes for unique physical inputs ---
    # key: path (string) → (hash_id, canonical_smiles)
    unique: Dict[str, Tuple[str, str]] = {}
    for alias, path in lig_map.items():
        mol = _rdkit_load(path, retain_h=retain_lig_prot)
        smi = _canonical_payload(mol)
        hid = _hash_id(smi, ligand_ff=ligand_ff, retain_h=retain_lig_prot)
        unique[path] = (hid, smi)

    # order by first appearance of path in input list (stable)
    ordered_paths = []
    seen = set()
    ligand_names = []
    hash_order = []
    for name, p in lig_map.items():
        if p not in seen:
            ordered_paths.append(p)
            ligand_names.append(name)
            seen.add(p)
        hash_order.append(unique[p][0])

    # --- build LigandProcessing objects for each unique hash ---
    to_prepare: List[Tuple[str, "LigandProcessing"]] = []

    factory = LigandFactory()

    for idx, p in enumerate(ordered_paths, start=1):
        lig_name = ligand_names[idx - 1]
        hid, smi = unique[p]
        target_dir = out_root / hid
        target_dir.mkdir(parents=True, exist_ok=True)

        lig = factory.create_ligand(
            ligand_file=p,
            index=idx,
            output_dir=target_dir.as_posix(),
            # set to a generic name
            ligand_name="lig",
            retain_lig_prot=retain_lig_prot,
            charge=charge_method,
            ligand_ff=ligand_ff,
            unique_mol_names=[],
        )
        # Dump metadata for traceability
        meta = {
            "hash_id": hid,
            "input_path": str(Path(p).resolve()),
            "aliases": [name for name, path in lig_map.items() if path == p],
            "canonical_smiles": smi,
            "retain_lig_prot": bool(retain_lig_prot),
            "ligand_ff": ligand_ff,
            "prepared_base": lig_name,
        }
        (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Skip if artifacts already present and not overwriting
        marker_any = any(
            (target_dir / f"{lig.name}.{ext}").exists()
            for ext in ("frcmod", "lib", "prmtop", "xml")
        )
        if not overwrite and marker_any:
            logger.info("Reusing cached ligand @ {} ({})", hid, meta["prepared_base"])
        else:
            to_prepare.append((lig_name, hid, lig))

    # --- perform parametrization (local or SLURM) ---
    if to_prepare:
        if run_with_slurm:
            raise NotImplementedError("Not implemented yet.")
        else:
            for lig_name, hid, lig in to_prepare:
                logger.info(f"Preparing {lig_name} in {lig.output_dir}")
                lig.prepare_ligand_parameters()

    logger.success(f"Prepared all ligands into {out_root}")
    return hash_order, unique
