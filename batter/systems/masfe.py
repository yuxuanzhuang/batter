from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from loguru import logger

from .core import SimSystem, SystemBuilder, CreateSystemLike

__all__ = [
    "MASFEBuilder",
    "make_ligand_subsystem_masfe",
    "prepare_subsystems_for_ligands_masfe",
]


class MASFEBuilder(SystemBuilder):
    """
    Builder for membrane-free (solvation) absolute free-energy (MASFE) systems.

    This builder prepares a *shared* working directory under ``system.root`` and,
    optionally, stages *all ligands at once* into per-ligand subfolders.

    Differences vs MABFE:
      - No protein/topology/coordinates are required or staged.
      - Resulting SimSystem has protein/topology/coordinates = None.

    Directory layout (relative to ``system.root``)
    ----------------------------------------------
    inputs/           # canonical copies of user-provided ligand inputs
    artifacts/        # files produced by builders
    simulations/
      <LIG1>/inputs/ligand.<ext>
              artifacts/
      <LIG2>/inputs/ligand.<ext>
              artifacts/
      ...
    """

    # -------------------- public API --------------------

    def build(self, system: SimSystem, args: CreateSystemLike) -> SimSystem:
        """
        Prepare the shared system area (stage ligand inputs).

        Uses the **actual suffixes** from user inputs (no hard-coded extensions).
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
            logger.info("MASFE system found at {} (overwrite=False) — keeping existing artifacts.", root)
            return self._assemble_system(system, inputs_dir, artifacts_dir, args)

        if args.overwrite:
            logger.warning("overwrite=True — wiping and re-preparing artifacts under {}", artifacts_dir)
            self._clean_dir(artifacts_dir)

        # Stage ligands with <NAME>.<ext>
        staged_ligs = self._stage_ligands_named(inputs_dir, args.ligand_paths)

        marker.touch()

        updated = system.with_artifacts(
            protein=None,                 # MASFE: no receptor
            topology=None,
            coordinates=None,
            ligands=tuple(staged_ligs),
            lipid_mol=tuple(),            # not used for solvation FE
            anchors=tuple(),              # not used for solvation FE
            meta={"ligand_ff": getattr(args, "ligand_ff", "gaff2"), "mode": "MASFE"},
        )
        logger.info("Prepared MASFE system '{}' at {} (ligands: {})",
                    updated.name, updated.root, len(updated.ligands))
        logger.info("  Ligands: {}", ", ".join(l.stem for l in updated.ligands))
        return updated

    def build_all_ligands(
        self,
        parent: SimSystem,
        lig_paths: Sequence[Path],
        overwrite: bool = False,
    ) -> Dict[str, SimSystem]:
        """
        Stage **all ligands at once** under ``parent.root/simulations/<NAME>/...``.

        Ligands are copied as ``inputs/ligand.<ext>`` using each source's suffix.
        """
        lig_dir = parent.root / "simulations"
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

            # stage ligand into its own inputs/ as ligand.<ext>
            lig_dst = sub_root / "inputs" / f"ligand{p.suffix}"
            shutil.copy2(p, lig_dst)

            child = SimSystem(
                name=f"{parent.name}:{name}",
                root=sub_root,
                protein=None,
                topology=None,
                coordinates=None,
                ligands=(lig_dst,),
                lipid_mol=tuple(),
                anchors=tuple(),
                meta={**(parent.meta or {}), "ligand": name, "mode": "MASFE"},
            )
            children[name] = child

        logger.debug("Staged {} MASFE ligand subsystems under {}", len(children), lig_dir)
        return children

    # ------------------ convenience helpers ------------------

    @staticmethod
    def make_child_for_ligand(parent: SimSystem, lig_name: str, lig_src: Path) -> SimSystem:
        """
        Create a single per-ligand child system under ``simulations/<NAME>/`` with ligand.<ext>.
        """
        lig_dir = parent.root / "simulations" / lig_name
        (lig_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (lig_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        p = Path(lig_src)
        dst = lig_dir / "inputs" / f"ligand{p.suffix}"
        shutil.copy2(p, dst)

        return SimSystem(
            name=f"{parent.name}:{lig_name}",
            root=lig_dir,
            protein=None,
            topology=None,
            coordinates=None,
            ligands=(dst,),
            lipid_mol=tuple(),
            anchors=tuple(),
            meta={**(parent.meta or {}), "ligand": lig_name, "mode": "MASFE"},
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
    def _stage_ligands_named(dst_dir: Path, lig_map: Mapping[str, Path]) -> List[Path]:
        """
        Copy ligand files into dst_dir as <LIG_NAME>.<ext> using the keys of lig_map.
        """
        staged: List[Path] = []
        for name, src in lig_map.items():
            src = Path(src)
            dst = dst_dir / f"{name}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            staged.append(dst)
        return staged

    @staticmethod
    def _first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
        for p in paths:
            if p and Path(p).exists():
                return Path(p)
        return None

    def _assemble_system(
        self,
        system: SimSystem,
        inputs_dir: Path,
        artifacts_dir: Path,
        args: CreateSystemLike,
    ) -> SimSystem:
        """
        Re-assemble a MASFE SimSystem referencing already-staged ligands.
        (No protein/topology/coordinates are restored.)
        """
        ligands = (
            sorted(inputs_dir.glob("*.sdf"))
            + sorted(inputs_dir.glob("*.mol2"))
            + sorted(inputs_dir.glob("*.pdb"))
        )
        return system.with_artifacts(
            protein=None,
            topology=None,
            coordinates=None,
            ligands=tuple(ligands),
            lipid_mol=tuple(),
            anchors=tuple(),
            meta={"ligand_ff": getattr(args, "ligand_ff", "gaff2"), "mode": "MASFE"},
        )


# --------------------------------------------------------------------------
# Free helpers (functional style)
# --------------------------------------------------------------------------

def make_ligand_subsystem_masfe(parent: SimSystem, lig_name: str, lig_src: Path) -> SimSystem:
    builder = MASFEBuilder()
    return builder.make_child_for_ligand(parent, lig_name, Path(lig_src))


def prepare_subsystems_for_ligands_masfe(parent: SimSystem, lig_paths: Iterable[Path]) -> Dict[str, SimSystem]:
    builder = MASFEBuilder()
    return builder.build_all_ligands(parent, [Path(p) for p in lig_paths], overwrite=False)