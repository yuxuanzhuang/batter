from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import MDAnalysis as mda

from batter.analysis.sim_validation import check_universe_ring_penetration


PRMTOP = "full.hmr.prmtop"
MARKER = Path("RING_PENETRATION")
REPAIRED_MARKER = Path("RING_PENETRATION_REPAIRED")
REPAIR_FAILED_MARKER = Path("RING_PENETRATION_REPAIR_FAILED")
LIGAND_RESNAME = __BATTER_LIGAND_RESNAME__
LIGAND_LABEL = __BATTER_LIGAND_LABEL__
FIX_MODE = __BATTER_RING_FIX_MODE__


def _check_restart(path: str) -> bool:
    universe = mda.Universe(PRMTOP, path, format="RESTRT")
    return bool(check_universe_ring_penetration(universe))


def _set_penetration_marker(path: str) -> None:
    MARKER.write_text(f"Ring penetration detected in {path}.\n")


def _clear_markers() -> None:
    for marker in (MARKER, REPAIR_FAILED_MARKER):
        marker.unlink(missing_ok=True)


def _repair_restart(path: str) -> bool:
    from batter._internal.ops.ring_repair import repair_ring_penetrations
    from batter._internal.parmed_compat import import_parmed

    pmd = import_parmed()
    rst_path = Path(path)
    backup_path = rst_path.with_name(f"{rst_path.name}.pre_ring_repair")
    shutil.copy2(rst_path, backup_path)

    structure = pmd.load_file(PRMTOP, str(rst_path))
    result = repair_ring_penetrations(
        structure,
        fix_mode=FIX_MODE,
        ligand_resname=LIGAND_RESNAME,
        ligand_label=LIGAND_LABEL,
    )

    metadata = result.to_dict()
    metadata["coordinate_file"] = str(rst_path)
    metadata["pre_repair_coordinate_file"] = str(backup_path)
    Path(f"{rst_path.name}.ring_penetration_repair.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    if not result.repaired:
        REPAIR_FAILED_MARKER.write_text(
            "Ring penetration repair did not remove all detected penetrations.\n"
            f"initial_penetrations={result.initial_penetrations}\n"
            f"final_penetrations={result.final_penetrations}\n"
        )
        return False

    structure.save(str(rst_path), overwrite=True)
    if _check_restart(str(rst_path)):
        REPAIR_FAILED_MARKER.write_text(
            "Ring penetration repair wrote coordinates, but validation still detects penetration.\n"
        )
        return False

    REPAIRED_MARKER.write_text(
        f"Ring penetration repaired in {path}; original restart saved to {backup_path}.\n"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("restart")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Attempt BATTER local ring-penetration repair when penetration is detected.",
    )
    args = parser.parse_args()

    if not _check_restart(args.restart):
        _clear_markers()
        return 0

    _set_penetration_marker(args.restart)
    if not args.repair:
        return 0

    print(
        f"[INFO] Attempting BATTER ring-penetration repair for {args.restart} "
        f"(mode={FIX_MODE}, ligand_resname={LIGAND_RESNAME})."
    )
    if _repair_restart(args.restart):
        _clear_markers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
