from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

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
        Prepare the shared system area (stage protein/topology/coordinates/inputs).

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
            logger.info("System found at {} (overwrite=False) — keeping existing artifacts.", root)
            return self._assemble_system(system, inputs_dir, artifacts_dir, args)

        if args.overwrite:
            logger.warning("overwrite=True — wiping and re-preparing artifacts under {}", artifacts_dir)
            self._clean_dir(artifacts_dir)

        # Stage shared inputs with their actual suffixes
        staged_protein = self._stage_optional(
            inputs_dir,
            args.protein_input,
            f"protein{args.protein_input.suffix}" if args.protein_input else None,
        )
        staged_top = self._stage_optional(
            inputs_dir,
            args.system_input,
            f"system{args.system_input.suffix}" if args.system_input else None,
        )
        staged_coord = self._stage_optional(
            inputs_dir,
            args.system_coordinate,
            f"system{args.system_coordinate.suffix}" if args.system_coordinate else None,
        )

        # Stage ligands with <NAME>.<ext>
        staged_ligs = self._stage_ligands_named(inputs_dir, args.ligand_paths)

        marker.touch()

        updated = system.with_artifacts(
            protein=staged_protein,
            topology=staged_top,
            coordinates=staged_coord,
            ligands=tuple(staged_ligs),
            lipid_mol=tuple(args.lipid_mol),
            anchors=tuple(args.anchor_atoms),
            meta=system.meta.merge(ligand_ff=getattr(args, "ligand_ff", "gaff2")),
        )
        logger.info(
            f"Prepared MABFE system '{updated.name}' at {updated.root} (ligands: {len(updated.ligands)})")
        logger.info("  Protein:    {}", updated.protein)
        logger.info("  System Topology:   {}", updated.topology)
        logger.info("  System Coord:      {}", updated.coordinates)
        logger.info("  Ligands:    {}", ", ".join(l.stem for l in updated.ligands))
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
                protein=parent.protein,
                topology=parent.topology,
                coordinates=parent.coordinates,
                ligands=(lig_dst,),
                lipid_mol=parent.lipid_mol,
                anchors=parent.anchors,
                meta=parent.meta.merge(ligand=name),
            )
            children[name] = child

        logger.debug("Staged {} ligand subsystems under {}", len(children), lig_dir)
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
            protein=parent.protein,
            topology=parent.topology,
            coordinates=parent.coordinates,
            ligands=(dst,),
            lipid_mol=parent.lipid_mol,
            anchors=parent.anchors,
            meta=parent.meta.merge(ligand=lig_name),
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
    def _stage_optional(dst_dir: Path, maybe_src: Optional[Path], filename: Optional[str]) -> Optional[Path]:
        if maybe_src is None or filename is None:
            return None
        dst = dst_dir / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(maybe_src, dst)
        return dst

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
        Re-assemble a SimSystem referencing already-staged files.
        Prefer artifacts/, fallback to inputs/, using **suffixes from args** when available.
        """
        # Determine expected filenames based on original input suffixes (if provided)
        prot_candidates: List[Optional[Path]] = []
        if args.protein_input:
            prot_candidates += [
                artifacts_dir / f"protein{args.protein_input.suffix}",
                inputs_dir / f"protein{args.protein_input.suffix}",
            ]
        # Also fallback to a couple of common names if inputs absent
        prot_candidates += [artifacts_dir / "protein.pdb", inputs_dir / "protein.pdb"]

        top_candidates: List[Optional[Path]] = []
        if args.system_input:
            top_candidates += [
                artifacts_dir / f"system{args.system_input.suffix}",
                inputs_dir / f"system{args.system_input.suffix}",
            ]
        top_candidates += [inputs_dir / "input.prmtop", artifacts_dir / "system.prmtop"]

        coord_candidates: List[Optional[Path]] = []
        if args.system_coordinate:
            coord_candidates += [
                artifacts_dir / f"system{args.system_coordinate.suffix}",
                inputs_dir / f"system{args.system_coordinate.suffix}",
            ]
        coord_candidates += [inputs_dir / "input.rst7", artifacts_dir / "system.rst7"]

        protein = self._first_existing(prot_candidates)
        topology = self._first_existing(top_candidates)
        coordinates = self._first_existing(coord_candidates)
        ligands = sorted(inputs_dir.glob("*.sdf")) + sorted(inputs_dir.glob("*.mol2")) + sorted(inputs_dir.glob("*.pdb"))

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
# Free helpers (functional style)
# --------------------------------------------------------------------------

def make_ligand_subsystem(parent: SimSystem, lig_name: str, lig_src: Path) -> SimSystem:
    builder = MABFEBuilder()
    return builder.make_child_for_ligand(parent, lig_name, Path(lig_src))


def prepare_subsystems_for_ligands(parent: SimSystem, lig_paths: Iterable[Path]) -> Dict[str, SimSystem]:
    builder = MABFEBuilder()
    return builder.build_all_ligands(parent, [Path(p) for p in lig_paths], overwrite=False)
