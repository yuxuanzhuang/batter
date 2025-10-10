from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from .core import SimSystem, SystemBuilder, CreateSystemLike


__all__ = [
    "MABFEBuilder",
    "make_ligand_subsystem",
    "prepare_subsystems_for_ligands",
]


class MABFEBuilder(SystemBuilder):
    """
    Builder for membrane/absolute free-energy (MABFE) systems.

    This builder prepares a *shared* working directory under ``system.root`` and,
    optionally, stages *all ligands at once* into per-ligand subfolders.

    Directory layout (relative to ``system.root``)
    ----------------------------------------------
    inputs/           # canonical copies of user-provided inputs
    artifacts/        # files produced by builders (e.g., PRMTOP, RST7)
    ligands/
      <LIG1>/inputs/ligand.sdf
              artifacts/
      <LIG2>/inputs/ligand.sdf
              artifacts/
      ...

    Idempotency
    -----------
    - If ``overwrite=False`` and the shared artifacts marker exists, existing
      files are kept.
    - If ``overwrite=True``, the *shared* ``artifacts/`` folder is wiped and
      re-created. Ligand subfolders are **not** touched (see ``build_all_ligands``).

    Notes
    -----
    This class does not perform equilibration or run FE windows. It only ensures
    that required on-disk artifacts and per-ligand folders exist for pipelines
    and backends to consume.
    """

    # -------------------- public API --------------------

    def build(self, system: SimSystem, args: CreateSystemLike) -> SimSystem:
        """
        Prepare the shared system area (stage protein/topology/coordinates/inputs).

        Parameters
        ----------
        system : SimSystem
            Descriptor for the shared system.
        args : CreateSystemLike
            Creation arguments (protein_input, ligand_paths, etc.).

        Returns
        -------
        SimSystem
            Updated descriptor with staged artifacts.
        """
        self._assert_names_match(system, args)

        root = system.root
        inputs_dir = root / "inputs"
        artifacts_dir = root / "artifacts"

        root.mkdir(parents=True, exist_ok=True)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        marker = artifacts_dir / ".prepared"
        if marker.exists() and not args.overwrite:
            logger.info("System already prepared at {} (overwrite=False) — keeping existing artifacts.", root)
            return self._assemble_system(system, inputs_dir, artifacts_dir, args)

        if args.overwrite:
            logger.warning("overwrite=True — wiping and re-preparing artifacts under {}", artifacts_dir)
            self._clean_dir(artifacts_dir)

        # Stage shared inputs
        staged_protein = self._stage_optional(inputs_dir, args.protein_input, "protein.pdb")
        staged_top = self._stage_optional(inputs_dir, args.system_input, "input.prmtop")
        staged_coord = self._stage_optional(inputs_dir, args.system_coordinate, "input.rst7")
        staged_ligs = self._stage_many(inputs_dir, args.ligand_paths.values(), "ligand_{i}.sdf")

        # Produce canonical shared artifacts (if provided)
        final_top = self._copy_optional(artifacts_dir, staged_top, "system.prmtop")
        final_coord = self._copy_optional(artifacts_dir, staged_coord, "system.rst7")

        marker.touch()

        updated = system.with_artifacts(
            protein=staged_protein,
            topology=final_top or staged_top,
            coordinates=final_coord or staged_coord,
            ligands=tuple(staged_ligs),
            lipid_mol=tuple(args.lipid_mol),
            anchors=tuple(args.anchor_atoms),
            meta={"ligand_ff": getattr(args, "ligand_ff", "gaff2")},
        )
        logger.info(
            "Prepared MABFE system '{}' at {} (ligands: {}, lipid_mol: {}, anchors: {})",
            updated.name, updated.root, len(updated.ligands), updated.lipid_mol, updated.anchors,
        )
        return updated

    def build_all_ligands(
        self,
        parent: SimSystem,
        lig_paths: Sequence[Path],
        overwrite: bool = False,
    ) -> Dict[str, SimSystem]:
        """
        Stage **all ligands at once** under ``parent.root/ligands/<NAME>/...``.

        Parameters
        ----------
        parent : SimSystem
            Shared system previously prepared by :meth:`build`.
        lig_paths : Sequence[pathlib.Path]
            List of ligand files to stage.
        overwrite : bool, default=False
            If ``True``, each per-ligand ``artifacts/`` directory is wiped
            before (re)creation. Inputs are always copied.

        Returns
        -------
        dict[str, SimSystem]
            Mapping ligand name (uppercase stem) → child :class:`SimSystem`.

        Notes
        -----
        Child systems reuse parent protein/topology/coordinates by reference to
        keep the store portable and avoid duplication.
        """
        lig_dir = parent.root / "ligands"
        lig_dir.mkdir(parents=True, exist_ok=True)

        children: Dict[str, SimSystem] = {}
        for src in lig_paths:
            p = Path(src)
            name = p.stem.upper()
            sub_root = lig_dir / name

            # ensure layout
            (sub_root / "inputs").mkdir(parents=True, exist_ok=True)
            art_dir = sub_root / "artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)

            if overwrite:
                logger.warning("overwrite=True — wiping ligand artifacts under {}", art_dir)
                self._clean_dir(art_dir)

            # stage ligand into its own inputs/
            dst = sub_root / "inputs" / "ligand.sdf"
            shutil.copy2(p, dst)

            child = SimSystem(
                name=f"{parent.name}:{name}",
                root=sub_root,
                protein=parent.protein,
                topology=parent.topology,
                coordinates=parent.coordinates,
                ligands=(dst,),
                lipid_mol=parent.lipid_mol,
                anchors=parent.anchors,
                meta={**(parent.meta or {}), "ligand": name},
            )
            children[name] = child

        logger.info("Staged {} ligand subsystems under {}", len(children), lig_dir)
        return children

    # ------------------ convenience helpers ------------------

    @staticmethod
    def make_child_for_ligand(parent: SimSystem, lig_name: str, lig_src: Path) -> SimSystem:
        """
        Create a single per-ligand child system under ``ligands/<NAME>/``.

        Parameters
        ----------
        parent : SimSystem
            Shared system.
        lig_name : str
            Ligand identifier (used as folder name; e.g., ``"LIG1"``).
        lig_src : path-like
            Path to the ligand file to stage.

        Returns
        -------
        SimSystem
            Child system descriptor.
        """
        lig_dir = parent.root / "ligands" / lig_name
        (lig_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (lig_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        dst = lig_dir / "inputs" / "ligand.sdf"
        shutil.copy2(lig_src, dst)

        return SimSystem(
            name=f"{parent.name}:{lig_name}",
            root=lig_dir,
            protein=parent.protein,
            topology=parent.topology,
            coordinates=parent.coordinates,
            ligands=(dst,),
            lipid_mol=parent.lipid_mol,
            anchors=parent.anchors,
            meta={**(parent.meta or {}), "ligand": lig_name},
        )

    # ------------------ internal utilities ------------------

    @staticmethod
    def _assert_names_match(system: SimSystem, args: CreateSystemLike) -> None:
        if system.name != args.system_name:
            raise ValueError(
                f"System name mismatch: SimSystem.name={system.name!r} vs args.system_name={args.system_name!r}"
            )

    @staticmethod
    def _clean_dir(path: Path) -> None:
        for p in path.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)

    @staticmethod
    def _stage_optional(dst_dir: Path, maybe_src: Optional[Path], filename: str) -> Optional[Path]:
        if maybe_src is None:
            return None
        dst = dst_dir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(maybe_src, dst)
        return dst

    @staticmethod
    def _stage_many(dst_dir: Path, sources: Sequence[Path], pattern: str) -> List[Path]:
        staged: List[Path] = []
        for i, src in enumerate(sources):
            name = pattern.format(i=i)
            dst = dst_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            staged.append(dst)
        return staged

    @staticmethod
    def _copy_optional(dst_dir: Path, maybe_src: Optional[Path], filename: str) -> Optional[Path]:
        if maybe_src is None:
            return None
        dst = dst_dir / filename
        shutil.copy2(maybe_src, dst)
        return dst

    @staticmethod
    def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
        for p in paths:
            if p and p.exists():
                return p
        return None

    def _assemble_system(
        self,
        system: SimSystem,
        inputs_dir: Path,
        artifacts_dir: Path,
        args: CreateSystemLike,
    ) -> SimSystem:
        protein = self._first_existing([artifacts_dir / "system.pdb", inputs_dir / "protein.pdb"])
        topology = self._first_existing([artifacts_dir / "system.prmtop", inputs_dir / "input.prmtop"])
        coordinates = self._first_existing([artifacts_dir / "system.rst7", inputs_dir / "input.rst7"])
        ligands = sorted(inputs_dir.glob("ligand_*.sdf"))
        return system.with_artifacts(
            protein=protein,
            topology=topology,
            coordinates=coordinates,
            ligands=tuple(ligands),
            lipid_mol=tuple(args.lipid_mol),
            anchors=tuple(args.anchor_atoms),
            meta={"ligand_ff": getattr(args, "ligand_ff", "gaff2")},
        )


# --------------------------------------------------------------------------
# Free helpers (functional style) if you prefer not to call methods directly
# --------------------------------------------------------------------------

def make_ligand_subsystem(parent: SimSystem, lig_name: str, lig_src: Path) -> SimSystem:
    """
    Functional helper that mirrors :meth:`MABFEBuilder.make_child_for_ligand`.
    """
    builder = MABFEBuilder()
    return builder.make_child_for_ligand(parent, lig_name, Path(lig_src))


def prepare_subsystems_for_ligands(parent: SimSystem, lig_paths: Iterable[Path]) -> Dict[str, SimSystem]:
    """
    Functional helper that mirrors :meth:`MABFEBuilder.build_all_ligands`
    with ``overwrite=False``.
    """
    builder = MABFEBuilder()
    return builder.build_all_ligands(parent, list(lig_paths), overwrite=False)