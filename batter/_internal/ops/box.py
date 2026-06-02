from __future__ import annotations

import os
import glob
import json
import shutil
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from loguru import logger
import parmed as pmd

from batter.utils import run_with_log, tleap
from batter.utils.builder_utils import get_buffer_z
from batter._internal.builders.interfaces import BuildContext
from batter._internal.builders.fe_registry import register_create_box
from batter._internal.ops.helpers import (
    Anchors,
    PROTEIN_COM_ATOM_SELECTION,
    load_anchors,
    run_parmed_hmr_if_enabled,
    merge_first_n_molecules_in_prmtop,
    save_anchors,
)




_HY36_DIGITS_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HY36_DIGITS_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


def _decode_pure_base36(value: str, digits: str) -> int:
    decoded = 0
    for char in value:
        decoded *= len(digits)
        decoded += digits.index(char)
    return decoded


def _hy36decode(width: int, value: str) -> int:
    """Decode a hybrid-36 PDB number.

    AmberTools can use hybrid-36 residue IDs once a PDB exceeds the decimal
    residue field. MDAnalysis decodes hybrid-36 atom serials, but not residue
    IDs, so BATTER normalizes those fields before handing the PDB to MDAnalysis.
    """
    if len(value) != width:
        raise ValueError(f"invalid hybrid-36 field width: {value!r}")

    first = value[0]
    if first in {"-", " "} or first.isdigit():
        return int(value)
    if first in _HY36_DIGITS_UPPER:
        return (
            _decode_pure_base36(value, _HY36_DIGITS_UPPER)
            - 10 * 36 ** (width - 1)
            + 10**width
        )
    if first in _HY36_DIGITS_LOWER:
        return (
            _decode_pure_base36(value, _HY36_DIGITS_LOWER)
            + 16 * 36 ** (width - 1)
            + 10**width
        )
    raise ValueError(f"invalid hybrid-36 field: {value!r}")


def _pdb_coordinate_fields_are_parseable(line: str) -> bool:
    if len(line) < 54:
        return False
    try:
        float(line[30:38])
        float(line[38:46])
        float(line[46:54])
    except ValueError:
        return False
    return True


def _normalize_decimal_resid_overflow_line(line: str) -> str | None:
    line_body = line.rstrip("\n")
    line_ending = line[len(line_body) :]
    match = re.match(
        r"(?P<resid>-?\d{5,})\s+"
        r"(?P<x>[-+]?\d+\.\d+)\s+"
        r"(?P<y>[-+]?\d+\.\d+)\s*"
        r"(?P<z>[-+]?\d+\.\d+)"
        r"(?P<rest>\s+.*)$",
        line_body[22:],
    )
    if match is None:
        return None

    try:
        resid = int(match.group("resid"))
        x = float(match.group("x"))
        y = float(match.group("y"))
        z = float(match.group("z"))
    except ValueError:
        return None

    normalized = (
        f"{line_body[:22]}{resid % 10000:04d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{match.group('rest')}"
        f"{line_ending}"
    )
    return normalized if _pdb_coordinate_fields_are_parseable(normalized) else None


def _normalize_hybrid36_resids_for_mdanalysis(pdb_path: Path) -> Path | None:
    """Return a temp PDB with hybrid-36 residue IDs converted for MDAnalysis.

    MDAnalysis' PDB parser treats non-decimal residue fields such as ``A6VB`` as
    missing and assigns resid 1, which merges consecutive waters into one
    residue. For MDAnalysis parsing only, convert such fields to their decimal
    residue number modulo the 4-column PDB field. MDAnalysis' existing wraparound
    logic then restores monotonically increasing residue IDs for normal Amber
    output order. Older Amber five-digit residue output is left unchanged when
    coordinate columns remain parseable; six-digit decimal residue overflow is
    folded back into the 4-column residue field so coordinates realign.
    """
    normalized_lines: list[str] = []
    changed = False

    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
                resid_field = line[22:26]
                try:
                    int(resid_field)
                except ValueError:
                    try:
                        resid = _hy36decode(4, resid_field)
                    except ValueError:
                        normalized_lines.append(line)
                    else:
                        changed = True
                        normalized_lines.append(
                            f"{line[:22]}{resid % 10000:04d}{line[26:]}"
                        )
                else:
                    if _pdb_coordinate_fields_are_parseable(line):
                        normalized_lines.append(line)
                        continue
                    normalized = _normalize_decimal_resid_overflow_line(line)
                    if normalized is None:
                        normalized_lines.append(line)
                        continue
                    changed = True
                    normalized_lines.append(normalized)
            else:
                normalized_lines.append(line)

    if not changed:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w")
    try:
        tmp.writelines(normalized_lines)
    finally:
        tmp.close()
    return Path(tmp.name)


@contextmanager
def _mdanalysis_pdb_path(pdb_path: Path) -> Iterator[Path]:
    normalized = _normalize_hybrid36_resids_for_mdanalysis(pdb_path)
    if normalized is None:
        yield pdb_path
        return

    try:
        yield normalized
    finally:
        normalized.unlink(missing_ok=True)


def _cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_res_blocks(selection, out_pdb: Path) -> None:
    lines = []
    if len(selection.residues) != 0:
        prev = selection.residues.resids[0]
        for res in selection.residues:
            if res.resid != prev:
                lines.append("TER\n")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            res.atoms.write(tmp.name)
            tmp.close()
            with open(tmp.name) as f:
                lines += [ln for ln in f if ln.startswith("ATOM")]
            prev = res.resid
    out_pdb.write_text("".join(lines))


_TERMINAL_AMIDE_CAP_ATOMS = {"N1": "N", "H1": "HN1", "H2": "HN2"}
_TERMINAL_METHYLAMIDE_RESNAMES = {"NMA", "NME"}
_N_TERMINAL_CAP_RESNAMES = {"ACE"}
_C_TERMINAL_CAP_RESNAMES = {"NMA", "NME", "NHE"}
_PROTEIN_TERMINAL_CAP_RESNAME_SET = (
    _N_TERMINAL_CAP_RESNAMES | _C_TERMINAL_CAP_RESNAMES
)
_PROTEIN_TERMINAL_CAP_RESNAMES = "ACE NMA NME NHE"
_PROTEIN_WITH_TERMINAL_CAPS = f"(protein or resname {_PROTEIN_TERMINAL_CAP_RESNAMES})"
_EMBEDDED_METHYLAMIDE_CARBON_ALIASES = ("CH3", "C1", "CM", "CR")
_SEPARATE_METHYLAMIDE_ATOM_ALIASES = {
    "N": "N",
    "H": "H",
    "HNT": "H",
    "HN": "H",
    "HN1": "H",
    "CH3": "C",
    "C": "C",
    "CA": "C",
    "CAT": "C",
    "C1": "C",
    "CM": "C",
    "CR": "C",
    "HH31": "H1",
    "HH32": "H2",
    "HH33": "H3",
    "H31": "H1",
    "H32": "H2",
    "H33": "H3",
    "H31H": "H1",
    "H32H": "H2",
    "H33H": "H3",
    "H1": "H1",
    "H2": "H2",
    "H3": "H3",
    "1HA": "H1",
    "2HA": "H2",
    "3HA": "H3",
    "HT1": "H1",
    "HT2": "H2",
    "HT3": "H3",
    "HR1": "H1",
    "HR2": "H2",
    "HR3": "H3",
    "HM1": "H1",
    "HM2": "H2",
    "HM3": "H3",
}


def _pdb_atom_name(line: str) -> str:
    return line[12:16].strip()


def _pdb_residue_key(line: str) -> tuple[str, int, str] | None:
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
        return None
    try:
        resid = int(line[22:26])
    except ValueError:
        return None
    return (line[21].strip(), resid, line[17:20].strip())


def _replace_pdb_atom_name(line: str, atom_name: str) -> str:
    line_body = line.rstrip("\n")
    line_ending = line[len(line_body) :]
    if len(line_body) < 16:
        line_body = line_body.ljust(16)
    atom_field = atom_name[:4] if len(atom_name) >= 4 else f" {atom_name:<3}"
    return f"{line_body[:12]}{atom_field}{line_body[16:]}{line_ending}"


def _replace_pdb_residue(line: str, *, resname: str, resid: int) -> str:
    line_body = line.rstrip("\n")
    line_ending = line[len(line_body) :]
    if len(line_body) < 26:
        line_body = line_body.ljust(26)
    return f"{line_body[:17]}{resname:>3}{line_body[20:22]}{resid:4d}{line_body[26:]}{line_ending}"


def _residue_keys_in_order(block: list[str]) -> list[tuple[str, int, str]]:
    keys: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in block:
        key = _pdb_residue_key(line)
        if key is not None and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _atom_names_for_residue(
    block: list[str], residue_key: tuple[str, int, str]
) -> set[str]:
    return {
        _pdb_atom_name(line)
        for line in block
        if _pdb_residue_key(line) == residue_key
    }


def _embedded_methylamide_cap_atoms(atom_names: set[str]) -> dict[str, str] | None:
    methyl_carbons = [
        name for name in _EMBEDDED_METHYLAMIDE_CARBON_ALIASES if name in atom_names
    ]
    if not methyl_carbons:
        return None

    aliases = {"N1": "N", methyl_carbons[0]: "C"}
    if "HN" in atom_names:
        aliases["HN"] = "H"
        methyl_hydrogens = ("H1", "H2", "H3")
    elif "HN1" in atom_names:
        aliases["HN1"] = "H"
        methyl_hydrogens = ("H1", "H2", "H3")
    elif "H" in atom_names:
        aliases["H"] = "H"
        methyl_hydrogens = ("H1", "H2", "H3")
    else:
        aliases["H1"] = "H"
        methyl_hydrogens = ("H2", "H3", "H4")

    for source, target in zip(methyl_hydrogens, ("H1", "H2", "H3")):
        if source in atom_names:
            aliases[source] = target
    for source, target in {
        "HH31": "H1",
        "HH32": "H2",
        "HH33": "H3",
        "H31": "H1",
        "H32": "H2",
        "H33": "H3",
        "H31H": "H1",
        "H32H": "H2",
        "H33H": "H3",
        "HR1": "H1",
        "HR2": "H2",
        "HR3": "H3",
        "HM1": "H1",
        "HM2": "H2",
        "HM3": "H3",
    }.items():
        if source in atom_names:
            aliases[source] = target

    return aliases


def _rewrite_separate_terminal_methylamide_cap(
    block: list[str], residue_keys: list[tuple[str, int, str]]
) -> tuple[list[str], bool] | None:
    if len(residue_keys) < 2:
        return None

    terminal_key = residue_keys[-1]
    if terminal_key[2] not in _TERMINAL_METHYLAMIDE_RESNAMES:
        return None

    atom_names = _atom_names_for_residue(block, terminal_key)
    if (
        "N" not in atom_names
        or "O" in atom_names
        or not any(
            name in atom_names
            for name in ("CH3", "C", "CA", "CAT", "C1", "CM", "CR")
        )
        or not atom_names.issubset(_SEPARATE_METHYLAMIDE_ATOM_ALIASES)
    ):
        return None

    previous_key = residue_keys[-2]
    rewritten: list[str] = []
    emitted_cap_atoms: set[str] = set()
    changed = terminal_key[2] != "NME"
    for line in block:
        key = _pdb_residue_key(line)
        atom_name = _pdb_atom_name(line)
        if key == previous_key and atom_name == "OXT":
            changed = True
            continue
        if key == terminal_key:
            cap_atom = _SEPARATE_METHYLAMIDE_ATOM_ALIASES[atom_name]
            if cap_atom in emitted_cap_atoms:
                changed = True
                continue
            emitted_cap_atoms.add(cap_atom)
            changed = changed or cap_atom != atom_name
            cap_line = _replace_pdb_atom_name(line, cap_atom)
            cap_line = _replace_pdb_residue(
                cap_line,
                resname="NME",
                resid=terminal_key[1],
            )
            rewritten.append(cap_line)
            continue
        rewritten.append(line)

    return rewritten, changed


def _rewrite_terminal_amide_caps_for_leap(pdb_path: Path) -> int:
    """
    Rewrite terminal amide caps into Amber residue/atom names.

    Peptide inputs can encode a C-terminal amide on the final amino-acid residue
    itself. LEaP then treats cap atoms like ``N1`` as unknown atoms on ``CXXX``.
    Moving those atoms into following ``NHE`` or ``NME`` residues lets the
    standard aminoct library type the cap and bond it to the preceding residue.
    """
    lines = pdb_path.read_text().splitlines(True)
    rewritten: list[str] = []
    block: list[str] = []
    cap_count = 0
    used_resids = {
        (key[0], key[1])
        for key in (_pdb_residue_key(line) for line in lines)
        if key is not None
    }
    all_resids = {resid for _chain, resid in used_resids}
    next_cap_resid = max(all_resids, default=0) + 1

    def take_cap_resid(chain_id: str, after_resid: int) -> int:
        nonlocal next_cap_resid
        while (chain_id, next_cap_resid) in used_resids and next_cap_resid <= 9999:
            next_cap_resid += 1
        if next_cap_resid > 9999:
            candidate = min(max(int(after_resid) + 1, 1), 9999)
            for offset in range(9999):
                resid = ((candidate - 1 + offset) % 9999) + 1
                if (chain_id, resid) not in used_resids:
                    used_resids.add((chain_id, resid))
                    return resid
            raise ValueError(
                f"Unable to assign a unique PDB residue ID for terminal amide cap in {pdb_path}"
            )
        resid = next_cap_resid
        used_resids.add((chain_id, resid))
        next_cap_resid += 1
        return resid

    def flush_block() -> None:
        nonlocal cap_count
        if not block:
            return

        residue_keys = _residue_keys_in_order(block)
        if not residue_keys:
            rewritten.extend(block)
            block.clear()
            return

        separate_methylamide = _rewrite_separate_terminal_methylamide_cap(
            block, residue_keys
        )
        if separate_methylamide is not None:
            methylamide_lines, changed = separate_methylamide
            rewritten.extend(methylamide_lines)
            if changed:
                cap_count += 1
            block.clear()
            return

        terminal_key = residue_keys[-1]
        terminal_atom_names = _atom_names_for_residue(block, terminal_key)
        if "N1" not in terminal_atom_names:
            rewritten.extend(block)
            block.clear()
            return

        methylamide_atoms = _embedded_methylamide_cap_atoms(terminal_atom_names)
        if methylamide_atoms is not None:
            cap_atom_map = methylamide_atoms
            cap_resname = "NME"
        else:
            cap_atom_map = _TERMINAL_AMIDE_CAP_ATOMS
            cap_resname = "NHE"

        cap_lines: list[str] = []
        body_lines: list[str] = []
        has_amide_n = False
        cap_resid = take_cap_resid(terminal_key[0], terminal_key[1])

        for line in block:
            key = _pdb_residue_key(line)
            atom_name = _pdb_atom_name(line)
            if key == terminal_key and atom_name in cap_atom_map:
                has_amide_n = has_amide_n or atom_name == "N1"
                cap_atom = cap_atom_map[atom_name]
                cap_line = _replace_pdb_atom_name(line, cap_atom)
                cap_line = _replace_pdb_residue(
                    cap_line,
                    resname=cap_resname,
                    resid=cap_resid,
                )
                cap_lines.append(cap_line)
                continue
            if key == terminal_key and atom_name == "OXT":
                continue
            body_lines.append(line)

        if has_amide_n:
            rewritten.extend(body_lines)
            rewritten.extend(cap_lines)
            cap_count += 1
        else:
            rewritten.extend(block)
        block.clear()

    for line in lines:
        if line.startswith("TER"):
            flush_block()
            rewritten.append(line)
        else:
            block.append(line)
    flush_block()

    if cap_count:
        pdb_path.write_text("".join(rewritten))
    return cap_count


def _chain_id_from_renum(
    renum_df: pd.DataFrame, *, resid: int, resname: str
) -> str:
    """Return the original chain ID for a residue in an Amber-renumbered PDB."""
    candidates = renum_df.query(
        "new_resid == @resid and new_resname == @resname"
    )
    if candidates.empty:
        candidates = renum_df.query(
            "old_resid == @resid and old_resname == @resname"
        )
    if candidates.empty:
        raise ValueError(
            f"Unable to map Amber residue {resname} {resid} back to an input chain"
        )
    return candidates.old_chain.values[0]


def _renum_resname(row: pd.Series) -> str:
    new_resname = str(row.get("new_resname", "")).strip()
    return new_resname or str(row.get("old_resname", "")).strip()


def _resnames_match_for_renum(residue_resname: str, row: pd.Series) -> bool:
    residue_resname = residue_resname.strip()
    row_resnames = {
        str(row.get("old_resname", "")).strip(),
        str(row.get("new_resname", "")).strip(),
    }
    if residue_resname in row_resnames:
        return True
    return (
        residue_resname in _C_TERMINAL_CAP_RESNAMES
        and any(name in _C_TERMINAL_CAP_RESNAMES for name in row_resnames)
    )


def _collapse_terminal_cap_resid_values(
    renum_df: pd.DataFrame, resids: list[int] | np.ndarray
) -> list[int]:
    collapsed = [int(resid) for resid in resids]
    rows = renum_df.reset_index(drop=True)
    if len(rows) != len(collapsed):
        return collapsed

    for pos, row in rows.iterrows():
        resname = _renum_resname(row)
        if resname in _N_TERMINAL_CAP_RESNAMES:
            search_range = range(pos + 1, len(rows))
        elif resname in _C_TERMINAL_CAP_RESNAMES:
            search_range = range(pos - 1, -1, -1)
        else:
            continue

        chain = str(row["old_chain"]).strip()
        for neighbor_pos in search_range:
            neighbor = rows.iloc[neighbor_pos]
            if str(neighbor["old_chain"]).strip() != chain:
                continue
            if _renum_resname(neighbor) in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
                continue
            collapsed[pos] = collapsed[neighbor_pos]
            break
    return collapsed


def _residue_chain_id(residue) -> str:
    try:
        chain_ids = residue.atoms.chainIDs
    except Exception:
        chain_ids = []
    if len(chain_ids):
        return str(chain_ids[0]).strip()
    return str(getattr(residue, "segid", "")).strip()


def _collapse_terminal_cap_resids_in_place(residues) -> None:
    if len(residues) == 0:
        return

    resids = np.array(residues.resids, dtype=int)
    chain_ids = [_residue_chain_id(residue) for residue in residues]
    resnames = [str(residue.resname).strip() for residue in residues]

    for pos, resname in enumerate(resnames):
        if resname in _N_TERMINAL_CAP_RESNAMES:
            search_range = range(pos + 1, len(residues))
        elif resname in _C_TERMINAL_CAP_RESNAMES:
            search_range = range(pos - 1, -1, -1)
        else:
            continue

        for neighbor_pos in search_range:
            if chain_ids[neighbor_pos] != chain_ids[pos]:
                continue
            if resnames[neighbor_pos] in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
                continue
            resids[pos] = resids[neighbor_pos]
            break

    residues.resids = resids


def _renum_old_resids_for_residues(residues, renum_df: pd.DataFrame) -> list[int]:
    rows = renum_df.reset_index(drop=True)
    row_pos = 0
    old_resids: list[int] = []

    for residue in residues:
        resname = str(residue.resname).strip()
        if resname in ["HIS", "HIE", "HIP", "HID"]:
            resname = "HIS"

        if row_pos < len(rows) and _resnames_match_for_renum(
            resname, rows.iloc[row_pos]
        ):
            old_resids.append(int(rows.iloc[row_pos]["old_resid"]))
            row_pos += 1
            continue

        if resname in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
            old_resids.append(int(residue.resid))
            continue

        while row_pos < len(rows) and _renum_resname(
            rows.iloc[row_pos]
        ) in _PROTEIN_TERMINAL_CAP_RESNAME_SET:
            row_pos += 1
        if row_pos < len(rows):
            old_resids.append(int(rows.iloc[row_pos]["old_resid"]))
            row_pos += 1
        else:
            old_resids.append(int(residue.resid))

    return old_resids


def _restore_protein_resids_from_renum(atom_group, renum_df: pd.DataFrame) -> None:
    residues = atom_group.select_atoms(_PROTEIN_WITH_TERMINAL_CAPS).residues
    if len(residues) == 0:
        return
    residues.resids = _renum_old_resids_for_residues(residues, renum_df)
    _collapse_terminal_cap_resids_in_place(residues)


def _ligand_charge_from_metadata(meta_path: Path) -> int | None:
    """Return the integer ligand charge recorded during parametrization."""
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        charge_val = data.get("ligand_charge")
        if charge_val is None:
            return None
        return int(round(float(charge_val)))
    except Exception as exc:
        logger.debug(f"Failed to read ligand charge from {meta_path}: {exc}")
        return None


def _read_disulfide_pairs(sslink_path: Path) -> list[tuple[int, int]]:
    """Read pdb4amber's 1-based residue-index disulfide pairs."""
    if not sslink_path.exists():
        return []

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for line_no, line in enumerate(sslink_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 2:
            logger.warning(
                f"Skipping malformed disulfide record {sslink_path}:{line_no}: {line!r}"
            )
            continue
        try:
            first, second = int(fields[0]), int(fields[1])
        except ValueError:
            logger.warning(
                f"Skipping malformed disulfide record {sslink_path}:{line_no}: {line!r}"
            )
            continue
        if first <= 0 or second <= 0 or first == second:
            logger.warning(
                f"Skipping invalid disulfide record {sslink_path}:{line_no}: {line!r}"
            )
            continue

        pair = tuple(sorted((first, second)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _map_disulfide_pairs_to_resids(
    pairs: list[tuple[int, int]], revised_resids: list[int] | np.ndarray
) -> list[tuple[int, int]]:
    """Map pdb4amber residue indices to the residue IDs written to LEaP PDBs."""
    revised = [int(resid) for resid in revised_resids]
    mapped: list[tuple[int, int]] = []
    for first, second in pairs:
        if first > len(revised) or second > len(revised):
            logger.warning(
                f"Skipping disulfide pair {first} {second}: only {len(revised)} residues were mapped"
            )
            continue
        mapped.append((revised[first - 1], revised[second - 1]))
    return mapped


def _merge_disulfide_pairs(
    pairs: list[tuple[int, int]], extra_pairs: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for first, second in [*pairs, *extra_pairs]:
        pair = tuple(sorted((int(first), int(second))))
        if pair in seen:
            continue
        seen.add(pair)
        merged.append(pair)
    return merged


def _infer_cyx_disulfide_pairs_from_atoms(
    atoms: mda.AtomGroup, *, max_sg_distance: float = 2.8
) -> list[tuple[int, int]]:
    """Infer close CYX SG-SG pairs that pdb4amber may omit from sslink."""
    records: list[tuple[int, np.ndarray]] = []
    for residue in atoms.select_atoms("protein and resname CYX").residues:
        sg_atoms = residue.atoms.select_atoms("name SG")
        if sg_atoms.n_atoms != 1:
            continue
        records.append(
            (int(residue.resid), np.asarray(sg_atoms[0].position, dtype=float))
        )

    candidates: list[tuple[float, tuple[int, int]]] = []
    for idx, (first_resid, first_pos) in enumerate(records):
        for second_resid, second_pos in records[idx + 1 :]:
            distance = float(np.linalg.norm(first_pos - second_pos))
            if distance <= float(max_sg_distance):
                candidates.append((distance, tuple(sorted((first_resid, second_resid)))))

    inferred: list[tuple[int, int]] = []
    used_resids: set[int] = set()
    for _distance, pair in sorted(candidates, key=lambda item: item[0]):
        first, second = pair
        if first in used_resids or second in used_resids:
            continue
        used_resids.update(pair)
        inferred.append(pair)
    return inferred


def _mark_disulfide_residue_names(residues, disulfide_resids: set[int]) -> None:
    """Ensure disulfide cysteines are written as CYX before LEaP loads them."""
    if not disulfide_resids:
        return

    for residue in residues:
        if (
            int(residue.resid) in disulfide_resids
            and residue.resname in {"CYS", "CYX"}
        ):
            residue.resname = "CYX"


def _is_disulfide_thiol_hydrogen_line(line: str, disulfide_resids: set[int]) -> bool:
    """Return True for cysteine SG hydrogen records that should not survive as CYX."""
    if not disulfide_resids or not line.startswith(("ATOM  ", "HETATM")):
        return False
    atom_name = line[12:16].strip()
    if atom_name not in {"HG", "HG1"}:
        return False
    resname = line[17:20].strip()
    if resname != "CYX":
        return False
    try:
        resid = int(line[22:26])
    except ValueError:
        return False
    return resid in disulfide_resids


def _write_leap_disulfide_bonds(
    handle, unit_name: str, disulfide_pairs: list[tuple[int, int]]
) -> None:
    """Write explicit LEaP SG-SG bonds for pdb4amber-detected disulfides."""
    if not disulfide_pairs:
        return

    for first, second in disulfide_pairs:
        handle.write(f"bond {unit_name}.{first}.SG {unit_name}.{second}.SG\n")
    handle.write("\n")


def _map_disulfide_pairs_to_leap_indices(
    disulfide_pairs: list[tuple[int, int]], pdb_path: Path
) -> list[tuple[int, int]]:
    """Map PDB residue IDs to the contiguous residue indices used by LEaP."""
    if not disulfide_pairs:
        return []

    residue_order: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in pdb_path.read_text().splitlines():
        key = _pdb_residue_key(line)
        if key is None or key in seen:
            continue
        seen.add(key)
        residue_order.append(key)

    if not residue_order:
        return disulfide_pairs

    leap_index = residue_order[0][1]
    resid_to_leap_index: dict[int, int] = {}
    ambiguous_resids: set[int] = set()
    for _chain, resid, _resname in residue_order:
        if resid in resid_to_leap_index:
            ambiguous_resids.add(resid)
        else:
            resid_to_leap_index[resid] = leap_index
        leap_index += 1

    mapped: list[tuple[int, int]] = []
    for first, second in disulfide_pairs:
        if first in ambiguous_resids or second in ambiguous_resids:
            logger.warning(
                f"Skipping disulfide pair {first} {second}: duplicate residue IDs in {pdb_path}"
            )
            continue
        try:
            mapped.append((resid_to_leap_index[first], resid_to_leap_index[second]))
        except KeyError:
            logger.warning(
                f"Skipping disulfide pair {first} {second}: residue ID not present in {pdb_path}"
            )
    return mapped


def _replace_anchor_mask_resid(mask: str | None, resid: int) -> str | None:
    if not mask:
        return mask
    return re.sub(r":-?\d+(?=@)", f":{resid}", mask, count=1)


def _find_ligand_resid_in_pdb(pdb_path: Path, ligand_resname: str) -> int | None:
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip() != ligand_resname:
            continue
        key = _pdb_residue_key(line)
        if key is not None:
            return key[1]
    return None


def _sync_ligand_anchor_residue_with_pdb(
    working_dir: Path, pdb_path: Path, ligand_resname: str
) -> None:
    anchors_path = working_dir / "anchors.json"
    if not anchors_path.exists():
        return

    actual_lig_res = _find_ligand_resid_in_pdb(pdb_path, ligand_resname)
    if actual_lig_res is None:
        logger.warning(
            f"Could not find ligand residue {ligand_resname!r} in {pdb_path}; leaving anchors unchanged"
        )
        return

    anchors = load_anchors(working_dir)
    if str(actual_lig_res) == str(anchors.lig_res):
        return

    save_anchors(
        working_dir,
        Anchors(
            P1=anchors.P1,
            P2=anchors.P2,
            P3=anchors.P3,
            L1=_replace_anchor_mask_resid(anchors.L1, actual_lig_res),
            L2=_replace_anchor_mask_resid(anchors.L2, actual_lig_res),
            L3=_replace_anchor_mask_resid(anchors.L3, actual_lig_res),
            lig_res=str(actual_lig_res),
        ),
    )
    logger.info(
        "Updated ligand anchor residue from {} to {} after LEaP residue numbering.",
        anchors.lig_res,
        actual_lig_res,
    )


@register_create_box("z")
def create_box(ctx: BuildContext) -> None:
    """
    Create the solvated box for the given component and window.
    """
    work = ctx.working_dir
    comp = ctx.comp
    param_dir = work.parent.parent / "params" if comp != "q" else work.parent / "params"
    sim = ctx.sim
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    amber_dir = ctx.amber_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    membrane_builder = sim.membrane_simulation
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol

    ligand = ctx.ligand
    mol = ctx.residue_name

    for attr in ("buffer_x", "buffer_y", "buffer_z"):
        if not hasattr(sim, attr):
            raise AttributeError(
                f"SimulationConfig missing '{attr}'. Please specify this buffer in the YAML."
            )
    buffer_x = float(sim.buffer_x)
    buffer_y = float(sim.buffer_y)
    buffer_z = float(sim.buffer_z)
    if (not membrane_builder) and (buffer_x < 5 or buffer_y < 5 or buffer_z < 5):
        raise ValueError("For water systems, buffer_x/y/z must be ≥ 5 Å.")

    if membrane_builder:
        buffer_x = 0.0
        buffer_y = 0.0
    else:
        # for non-equilibration non-membrane systems,
        # reduce the buffer by existing solvation shell
        if comp != 'q':
            solv_shell = sim.solv_shell
            buffer_x = max(0.0, buffer_x - solv_shell)
            buffer_y = max(0.0, buffer_y - solv_shell)
            buffer_z = max(0.0, buffer_z - solv_shell)


    if comp != "q":
        sdr_dist, abs_z, buffer_z_left = map(float, open(window_dir / "sdr_info.txt").read().split())
    else:
        buffer_z_left = buffer_z

    if not hasattr(sim, "water_model"):
        raise AttributeError("SimulationConfig missing 'water_model'.")
    water_model = str(sim.water_model).upper()

    if not hasattr(sim, "ion_def"):
        raise AttributeError("SimulationConfig missing 'ion_def'.")
    ion_def = sim.ion_def

    if not hasattr(sim, "neut"):
        raise AttributeError("SimulationConfig missing 'neut'.")
    neut = str(sim.neut)

    if not hasattr(sim, "dec_method"):
        raise AttributeError("SimulationConfig missing 'dec_method'.")
    dec_method = str(sim.dec_method)

    # ---- copy FF artifacts (resolve ff/ relative to window_dir: ../../param) ----
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "json"):
        src = param_dir / f"{ctx.residue_name}.{ext}"
        shutil.copy2(src, window_dir / src.name)

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{ctx.residue_name}.{ext}"
        shutil.copy2(src, window_dir / f"vac_ligand.{ext}")

    shutil.copy2(build_dir / f"{ligand}.pdb", window_dir / f"{ligand}.pdb")

    # other_mol
    if other_mol:
        raise NotImplementedError("Other molecules not supported now.")

    # tleap template
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    _cp(src_tleap, window_dir / "tleap.in")

    # water box keyword
    if water_model == "TIP3PF":
        # still uses leaprc.water.fb3
        water_box = "FB3BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    else:
        water_box = f"{water_model}BOX"

    build_cap_count = _rewrite_terminal_amide_caps_for_leap(window_dir / "build.pdb")
    if build_cap_count:
        logger.info(
            "Rewrote {} terminal protein amide cap(s) as Amber NHE/NME residues before pre-solvation LEaP.",
            build_cap_count,
        )

    # --- tleap solvate pre ---
    tleap_solv_pre = window_dir / "tleap_solvate_pre.in"
    _cp(window_dir / "tleap.in", tleap_solv_pre)
    with tleap_solv_pre.open("a") as f:
        f.write("# Load the necessary parameters\n")
        for om in other_mol:
            f.write(f"loadamberparams {om.lower()}.frcmod\n")
            f.write(f"{om} = loadmol2 {om.lower()}.mol2\n")
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        if water_model != "TIP3PF":
            f.write(f"source leaprc.water.{water_model.lower()}\n\n")
        else:
            f.write("source leaprc.water.fb3\n\n")
        f.write("model = loadpdb build.pdb\n\n")
        f.write(
            f"solvatebox model {water_box} {{ {buffer_x} {buffer_y} {buffer_z_left} }} 1\n\n"
        )
        f.write("desc model\n")
        f.write("savepdb model full_pre.pdb\n")
        f.write("quit\n")
    run_with_log(
        f"{tleap} -s -f {tleap_solv_pre.name} > tleap_solvate_pre.log",
        working_dir=window_dir,
    )

    # Count waters in build.pdb
    num_waters = sum(
        1 for ln in (window_dir / "build.pdb").read_text().splitlines() if "WAT" in ln
    )

    # pdb4amber
    run_with_log("pdb4amber -i build.pdb -o build_amber.pdb -y", working_dir=window_dir)
    renum_df = pd.read_csv(
        window_dir / "build_amber_renum.txt",
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df["old_resname"] = renum_df["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    renum_df["new_resname"] = renum_df["new_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    revised_resids = []
    resid_counter = 1
    prev_resid = 0
    for _, row in renum_df.iterrows():
        if row["old_resid"] != prev_resid or row["old_resname"] not in lipid_mol:
            revised_resids.append(resid_counter)
            resid_counter += 1
        else:
            revised_resids.append(resid_counter - 1)
        prev_resid = row["old_resid"]
    disulfide_pairs = _map_disulfide_pairs_to_resids(
        _read_disulfide_pairs(window_dir / "build_amber_sslink"), revised_resids
    )
    disulfide_resids = {resid for pair in disulfide_pairs for resid in pair}

    # MDAnalysis universes
    with _mdanalysis_pdb_path(window_dir / "full_pre.pdb") as full_pre_pdb:
        u = mda.Universe(str(full_pre_pdb))
        final_system = u.atoms
        system_dimensions = u.dimensions[:3]

        if membrane_builder:
            u_ref = mda.Universe(str(window_dir / "equil-reference.pdb"))
            u.dimensions[0] = u_ref.dimensions[0]
            u.dimensions[1] = u_ref.dimensions[1]
            u.dimensions[2] = u.dimensions[2] - 3
            u.atoms.positions[:, 2] -= 3

            membrane_region = u.select_atoms(f'resname {" ".join(lipid_mol)}')
            memb_z_max = membrane_region.select_atoms("type P").positions[:, 2].max() - 10
            memb_z_min = membrane_region.select_atoms("type P").positions[:, 2].min() + 10
            water_in_mem = u.select_atoms(
                f"byres (resname WAT and prop z > {memb_z_min} and prop z < {memb_z_max})"
            )
            final_system = final_system - water_in_mem

        box_xy = [u.dimensions[0], u.dimensions[1]]
        water_around_prot = u.select_atoms("resname WAT").residues[:num_waters].atoms
        final_system = final_system | water_around_prot

        if membrane_builder:
            outside_wat = final_system.select_atoms(
                "byres (resname WAT and "
                f"((prop x > {box_xy[0]/2}) or (prop x < -{box_xy[0]/2}) or "
                f"(prop y > {box_xy[1]/2}) or (prop y < -{box_xy[1]/2})))"
            )
            final_system = final_system - outside_wat

        if comp in ["e", "v", "o", "z"]:
            min_pos = final_system.positions[:, 2].min()
            system_dimensions[2] = abs_z

            outside_wat_z = final_system.select_atoms(
                "byres (resname WAT and "
                f"(prop z > {abs_z + min_pos}))"
            )
            final_system = final_system - outside_wat_z

        # renumber residues
        revised_resids = np.array(revised_resids)
        total_residues = final_system.residues.n_residues
        final_resids = np.zeros(total_residues, dtype=int)
        final_resids[: len(revised_resids)] = revised_resids
        next_resnum = revised_resids[-1] + 1
        final_resids[len(revised_resids) :] = np.arange(
            next_resnum, total_residues - len(revised_resids) + next_resnum
        )
        final_system.residues.resids = final_resids
        if bool(getattr(sim, "infer_disulfide_bonds", True)):
            inferred_disulfide_pairs = _infer_cyx_disulfide_pairs_from_atoms(final_system)
            existing_disulfide_pairs = {tuple(sorted(pair)) for pair in disulfide_pairs}
            new_disulfide_pairs = [
                pair
                for pair in inferred_disulfide_pairs
                if tuple(sorted(pair)) not in existing_disulfide_pairs
            ]
            if new_disulfide_pairs:
                logger.info(
                    "Inferred additional CYX disulfide pair(s) from SG distances: {}. "
                    "Set create.infer_disulfide_bonds: false to disable this inference.",
                    ", ".join(
                        f"{first}-{second}" for first, second in new_disulfide_pairs
                    ),
                )
                disulfide_pairs = _merge_disulfide_pairs(
                    disulfide_pairs, new_disulfide_pairs
                )
                disulfide_resids = {resid for pair in disulfide_pairs for resid in pair}
        _mark_disulfide_residue_names(final_system.residues, disulfide_resids)

        # partitions
        final_system_dum = final_system.select_atoms("resname DUM")
        final_system_dum[0].position = final_system.select_atoms(PROTEIN_COM_ATOM_SELECTION).center_of_mass()
        if comp == 'z':
            final_system_dum[1].position = final_system.select_atoms(f"resname {mol}").residues[1].atoms.center_of_mass()
        final_system_prot = final_system.select_atoms(_PROTEIN_WITH_TERMINAL_CAPS)
        final_system_others = final_system - final_system_prot - final_system_dum
        final_system_ligs = final_system.select_atoms(f"resname {mol}")
        final_system_other_mol = (
            final_system_others.select_atoms("not resname WAT") - final_system_ligs
        )
        final_system_water = final_system_others.select_atoms("resname WAT")
        final_system_water_notaround = final_system.select_atoms(
            f"byres (resname WAT and not (around 6 {_PROTEIN_WITH_TERMINAL_CAPS}))"
        )
        final_system_water_around = final_system_water - final_system_water_notaround

        # write parts
        _write_res_blocks(final_system_dum, window_dir / "solvate_pre_dum.pdb")

        # set chainIDs using renum_df and write protein by chains
        for residue in final_system_prot.residues:
            resid_resname = (
                "HIS"
                if residue.resname in ["HIS", "HIE", "HIP", "HID"]
                else residue.resname
            )
            residue.atoms.chainIDs = _chain_id_from_renum(
                renum_df, resid=residue.resid, resname=resid_resname
            )
        _collapse_terminal_cap_resids_in_place(final_system_prot.residues)
        prot_lines = []
        for chain_name in np.unique(final_system_prot.atoms.chainIDs):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            final_system.select_atoms(f"chainID {chain_name}").write(tmp.name)
            tmp.close()
            with open(tmp.name) as f:
                prot_lines += [
                    ln
                    for ln in f
                    if ln.startswith("ATOM")
                    and not _is_disulfide_thiol_hydrogen_line(ln, disulfide_resids)
                ]
            prot_lines.append("TER\n")
        solvate_pre_prot = window_dir / "solvate_pre_prot.pdb"
        solvate_pre_prot.write_text("".join(prot_lines))
        cap_count = _rewrite_terminal_amide_caps_for_leap(solvate_pre_prot)
        if cap_count:
            logger.info(
                "Rewrote {} terminal protein amide cap(s) as Amber NHE/NME residues before LEaP.",
                cap_count,
            )
        leap_disulfide_pairs = _map_disulfide_pairs_to_leap_indices(
            disulfide_pairs, solvate_pre_prot
        )

        _write_res_blocks(final_system_ligs, window_dir / "solvate_pre_ligands.pdb")

        other_lines_exist = len(final_system_other_mol.residues) != 0
        if other_lines_exist:
            _write_res_blocks(final_system_other_mol, window_dir / "solvate_pre_others.pdb")

        outside_wat_exist = len(final_system_water_notaround.residues) != 0
        if outside_wat_exist:
            _write_res_blocks(
                final_system_water_notaround, window_dir / "solvate_pre_outside_wat.pdb"
            )

        around_wat_exist = len(final_system_water_around.residues) != 0
        if around_wat_exist:
            _write_res_blocks(
                final_system_water_around, window_dir / "solvate_pre_around_water.pdb"
            )

    # --- tleap parts (all with working_dir=window_dir) ---

    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_dum.in")
    with (window_dir / "tleap_solvate_dum.in").open("a") as f:
        f.write("dum = loadpdb solvate_pre_dum.pdb\n\n")
        f.write(
            f"set dum box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb dum solvate_dum.pdb\n")
        f.write("saveamberparm dum solvate_dum.prmtop solvate_dum.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_dum.in > tleap_dum.log", working_dir=window_dir
    )

    # prot
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_prot.in")
    with (window_dir / "tleap_solvate_prot.in").open("a") as f:
        f.write("prot = loadpdb solvate_pre_prot.pdb\n\n")
        _write_leap_disulfide_bonds(f, "prot", leap_disulfide_pairs)
        f.write(
            f"set prot box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb prot solvate_prot.pdb\n")
        f.write("saveamberparm prot solvate_prot.prmtop solvate_prot.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_prot.in > tleap_prot.log", working_dir=window_dir
    )

    # ligands
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_ligands.in")
    with (window_dir / "tleap_solvate_ligands.in").open("a") as f:
        f.write("# Load the necessary parameters\n")
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        if comp == "x":
            f.write(f"loadamberparams {mol}.frcmod\n")
            f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write("ligands = loadpdb solvate_pre_ligands.pdb\n\n")
        f.write(
            f"set ligands box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb ligands solvate_ligands.pdb\n")
        f.write(
            "saveamberparm ligands solvate_ligands.prmtop solvate_ligands.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_ligands.in > tleap_ligands.log",
        working_dir=window_dir,
    )

    # others
    if other_lines_exist:
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_others.in")
        with (window_dir / "tleap_solvate_others.in").open("a") as f:
            for om in other_mol:
                f.write(f"loadamberparams {om.lower()}.frcmod\n")
                f.write(f"{om} = loadmol2 {om.lower()}.mol2\n")
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("others = loadpdb solvate_pre_others.pdb\n\n")
            f.write(
                f"set others box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb others solvate_others.pdb\n")
            f.write(
                "saveamberparm others solvate_others.prmtop solvate_others.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_others.in > tleap_others.log",
            working_dir=window_dir,
        )

    # charge accounting
    def _sum_unit_charge_from_log(logfile: Path) -> Tuple[int, int]:
        neu_cat = neu_ani = 0
        if not logfile.exists():
            return 0, 0
        for line in logfile.read_text().splitlines():
            if "The unperturbed charge of the unit" in line:
                q = float(line.split()[6].strip("'\",.:;#()]["))
                if q < 0:
                    neu_cat += round(float(re.sub(r"[+-]", "", str(q))))
                elif q > 0:
                    neu_ani += round(float(re.sub(r"[+-]", "", str(q))))
        return neu_cat, neu_ani

    neu_cat, neu_ani = _sum_unit_charge_from_log(window_dir / "tleap_prot.log")
    if (window_dir / "tleap_others.log").exists():
        nc2, na2 = _sum_unit_charge_from_log(window_dir / "tleap_others.log")
        neu_cat += nc2
        neu_ani += na2
    lig_charge = _ligand_charge_from_metadata(param_dir / f"{ctx.residue_name}.json")
    lig_cat = max(0, -lig_charge)
    lig_ani = max(0, lig_charge)

    charge_neut = neu_cat - neu_ani + lig_cat - lig_ani
    neu_cat = max(0, charge_neut)
    neu_ani = max(0, -charge_neut)

    box_volume = system_dimensions[0] * system_dimensions[1] * system_dimensions[2]
    num_ions = round(ion_def[2] * 6.02e23 * box_volume * 1e-27)
    # put a minimum of 5 ions
    num_ions = max(5, num_ions)
    if membrane_builder:
        num_ions //= 2
    num_cat = num_ions
    num_ani = num_ions - neu_cat + neu_ani
    if num_ani < 0:
        num_cat = neu_cat
        num_ions = neu_cat
        num_ani = 0

    # outside water — ionization
    if (window_dir / "solvate_pre_outside_wat.pdb").exists():
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_outside_wat.in")
        with (window_dir / "tleap_solvate_outside_wat.in").open("a") as f:
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("outside_wat = loadpdb solvate_pre_outside_wat.pdb\n\n")
            if neut == "no":
                f.write(f"addionsrand outside_wat {ion_def[0]} {num_cat}\n")
                f.write(f"addionsrand outside_wat {ion_def[1]} {num_ani}\n")
            elif neut == "yes":
                if neu_cat:
                    f.write(f"addionsrand outside_wat {ion_def[0]} {neu_cat}\n")
                if neu_ani:
                    f.write(f"addionsrand outside_wat {ion_def[1]} {neu_ani}\n")
            f.write(
                f"set outside_wat box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb outside_wat solvate_outside_wat.pdb\n")
            f.write(
                "saveamberparm outside_wat solvate_outside_wat.prmtop solvate_outside_wat.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_outside_wat.in > tleap_outside_wat.log",
            working_dir=window_dir,
        )

    # around water
    if (window_dir / "solvate_pre_around_water.pdb").exists():
        _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_around_wat.in")
        with (window_dir / "tleap_solvate_around_wat.in").open("a") as f:
            if water_model != "TIP3PF":
                f.write(f"source leaprc.water.{water_model.lower()}\n\n")
            else:
                f.write("source leaprc.water.fb3\n\n")
            f.write("around_wat = loadpdb solvate_pre_around_water.pdb\n\n")
            f.write(
                f"set around_wat box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
            )
            f.write("savepdb around_wat solvate_around_wat.pdb\n")
            f.write(
                "saveamberparm around_wat solvate_around_wat.prmtop solvate_around_wat.inpcrd\nquit\n"
            )
        run_with_log(
            f"{tleap} -s -f tleap_solvate_around_wat.in > tleap_around_wat.log",
            working_dir=window_dir,
        )

    # combine with ParmEd
    dum_p = pmd.load_file(
        str(window_dir / "solvate_dum.prmtop"), str(window_dir / "solvate_dum.inpcrd")
    )
    prot_p = pmd.load_file(
        str(window_dir / "solvate_prot.prmtop"), str(window_dir / "solvate_prot.inpcrd")
    )
    ligand_p_1 = pmd.load_file(str(window_dir / f"{mol}.prmtop"))
    ligand_p_1.residues[0].name = mol
    ligand_p_1.save(str(window_dir / f"{mol}.prmtop"), overwrite=True)
    ligand_p_1 = pmd.load_file(str(window_dir / f"{mol}.prmtop"))

    lig_inp = pmd.load_file(str(window_dir / "solvate_ligands.inpcrd")).coordinates
    if dec_method == "dd" or comp == "q":
        ligands_p = ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp in ["z", "o", "s", "v"] and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    elif comp in ["e"] and dec_method == "sdr":
        ligands_p = ligand_p_1 + ligand_p_1 + ligand_p_1 + ligand_p_1
        ligands_p.coordinates = lig_inp
    else:
        raise ValueError(
            f"Unsupported comp={comp} with dec={dec_method} for custom ligand params."
        )

    combined = dum_p + prot_p + ligands_p
    vac = dum_p + prot_p + ligands_p
    other_parts = []

    if (window_dir / "solvate_others.prmtop").exists():
        others_p = pmd.load_file(
            str(window_dir / "solvate_others.prmtop"),
            str(window_dir / "solvate_others.inpcrd"),
        )
        combined += others_p
        other_parts.append(others_p)
    if (window_dir / "solvate_outside_wat.prmtop").exists():
        out_wat_pmd =  pmd.load_file(
            str(window_dir / "solvate_outside_wat.prmtop"),
            str(window_dir / "solvate_outside_wat.inpcrd"),
        )
        combined += out_wat_pmd
        other_parts.append(out_wat_pmd)
    if (window_dir / "solvate_around_wat.prmtop").exists():
        around_wat_pmd = pmd.load_file(
            str(window_dir / "solvate_around_wat.prmtop"),
            str(window_dir / "solvate_around_wat.inpcrd"),
        )
        combined += around_wat_pmd
        other_parts.append(around_wat_pmd)

    if len(other_parts) == 1:
        other_parts_pmd = other_parts[0]
    elif len(other_parts) == 2:
        other_parts_pmd = other_parts[0] + other_parts[1]
    elif len(other_parts) == 3:
        other_parts_pmd = other_parts[0] + other_parts[1] + other_parts[2]
    else:
        raise ValueError(f"Unsupported number of other_parts: {len(other_parts)}")

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)
    _sync_ligand_anchor_residue_with_pdb(work, window_dir / "vac.pdb", mol)

    other_parts_pmd.save(str(window_dir / "other_parts.prmtop"), overwrite=True)
    other_parts_pmd.save(str(window_dir / "other_parts.inpcrd"), overwrite=True)
    other_parts_pmd.save(str(window_dir / "other_parts.pdb"), overwrite=True)

    u_full = mda.Universe(str(window_dir / "full.pdb"))
    u_vac = mda.Universe(str(window_dir / "vac.pdb"))

    # renumber protein residues back to original ids
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    renum_df2 = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df2["old_resname"] = renum_df2["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    renum_df2["new_resname"] = renum_df2["new_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    _restore_protein_resids_from_renum(u_full, renum_df2)
    _restore_protein_resids_from_renum(u_vac, renum_df2)

    # rebuild segments by chain
    seg_txt = window_dir / "build_amber_renum.txt"
    seg_df = pd.read_csv(
        seg_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    chain_list = renum_df2.old_chain.values
    chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
    for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
        res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))
    u_vac.atoms.write(str(window_dir / "vac_orig.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    # merge DUM + DUM + PROT + LIG1 + LIG2 
    merge_first_n_molecules_in_prmtop(full_prmtop, 5, str(window_dir / "full_merged.prmtop"))
    return


@register_create_box("x")
def create_box_x(ctx: BuildContext) -> None:
    """
    Create the box for RBFE (x-component) ligand-pair systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir

    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    extra = ctx.extra or {}
    lig_ref = extra.get("ligand_ref")
    lig_alt = extra.get("ligand_alt")
    res_ref = extra.get("residue_ref") or ctx.residue_name
    res_alt = extra.get("residue_alt")

    if not res_alt:
        raise ValueError(
            "RBFE component 'x' requires residue_alt in BuildContext.extra."
        )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        param_dir = work.parent.parent / "params"
        src = param_dir / f"{res_ref}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_x] Optional/absent: {src}")
        param_dir = work.parent.parent.parent / lig_alt / "params"
        src = param_dir / f"{res_alt}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_x] Optional/absent: {src}")

    membrane_builder = sim.membrane_simulation
    lipid_mol = sim.lipid_mol
    other_mol = sim.other_mol
    
    # tleap template
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    _cp(src_tleap, window_dir / "tleap.in")

    # water box keyword
    water_model = str(sim.water_model).upper()

    if water_model == "TIP3PF":
        # still uses leaprc.water.fb3
        water_box = "FB3BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    else:
        water_box = f"{water_model}BOX"

    if water_model != "TIP3PF":
        water_line = f"source leaprc.water.{water_model.lower()}\n\n"
    else:
        water_line = "source leaprc.water.fb3\n\n"


    # combine with ParmEd
    vac_p = pmd.load_file(
        str(window_dir / "ref_vac.prmtop"), str(window_dir / "ref_vac.pdb")
    )
    other_part_p = pmd.load_file(
        str(window_dir / "other_parts.prmtop"),
        str(window_dir / "other_parts.pdb"),
    )
    ligand_alt = pmd.load_file(str(window_dir / f"{res_alt}.prmtop"))
    ligand_alt.residues[0].name = res_alt
    ligand_alt.save(str(window_dir / f"{res_alt}.prmtop"), overwrite=True)
    alter_ligands_p_site = pmd.load_file(
        str(window_dir / f"{res_alt}.prmtop"),
        str(window_dir / "alter_ligand_aligned_site.pdb"),
    )
    alter_ligands_p_solvent = pmd.load_file(
        str(window_dir / f"{res_alt}.prmtop"),
        str(window_dir / "alter_ligand_aligned_solvent.pdb"),
    )
    combined = vac_p + alter_ligands_p_site + alter_ligands_p_solvent + other_part_p

    # build the ion prmtop if exists
    if os.path.exists(window_dir / "ions.pdb"):
        tleap_ion_txt = (window_dir / "tleap.in").read_text().splitlines()
        tleap_ion_txt += [
            "# ion topology",
            water_line,
            f"ions = loadpdb ions.pdb",
            "saveamberparm ions ions.prmtop ions.inpcrd",
            "quit",
        ]
        _write(window_dir / "tleap_ions.in", "\n".join(tleap_ion_txt) + "\n")
        run_with_log(
            f"{tleap} -s -f tleap_ions.in > tleap_ions.log", working_dir=window_dir
        )
        ion_p = pmd.load_file(
        str(window_dir / "ions.prmtop"),
        str(window_dir / "ions.inpcrd"),
        )
        combined += ion_p

    vac = vac_p + alter_ligands_p_site + alter_ligands_p_solvent

    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    u_full = mda.Universe(str(window_dir / "full.pdb"))

    # renumber protein residues back to original ids
    renum_txt = build_dir / "protein_renum.txt"
    if not renum_txt.exists():
        renum_txt = build_dir.parent / build_dir.name / "protein_renum.txt"
    renum_df2 = pd.read_csv(
        renum_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    renum_df2["old_resname"] = renum_df2["old_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    renum_df2["new_resname"] = renum_df2["new_resname"].replace(
        ["HIS", "HIE", "HIP", "HID"], "HIS"
    )
    _restore_protein_resids_from_renum(u_full, renum_df2)

    # rebuild segments by chain
    seg_txt = window_dir / "build_amber_renum.txt"
    seg_df = pd.read_csv(
        seg_txt,
        sep=r"\s+",
        header=None,
        names=["old_resname", "old_chain", "old_resid", "new_resname", "new_resid"],
    )
    chain_list = renum_df2.old_chain.values
    chain_segments = {ch: u_full.add_Segment(segid=ch) for ch in chain_list}
    for res, ch in zip(u_full.residues[: len(chain_list)], chain_list):
        res.segment = chain_segments[ch]

    u_full.atoms.write(str(window_dir / "full.pdb"))

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    merge_first_n_molecules_in_prmtop(full_prmtop, 5, str(window_dir / "full_merged.prmtop"))

    # get mapping file

    mapping = json.load(open(window_dir / "mapping.json"))
    ref_site = u_full.select_atoms(f"resname {res_ref}").residues[0]
    ref_solvent = u_full.select_atoms(f"resname {res_ref}").residues[1]
    alt_site = u_full.select_atoms(f"resname {res_alt}").residues[0]
    alt_solvent = u_full.select_atoms(f"resname {res_alt}").residues[1]

    # select cc parts
    alt_index_list = [int(i) for i in mapping.keys()]
    ref_index_list = [int(i) for i in mapping.values()]
    cc_indices_site_t0 = ref_site.atoms[ref_index_list].indices + 1
    cc_indices_solvent_t0 = alt_solvent.atoms[alt_index_list].indices + 1
    cc_indices_solvent_t1 = ref_solvent.atoms[ref_index_list].indices + 1
    cc_indices_site_t1 = alt_site.atoms[alt_index_list].indices + 1
    all_indices_t0 = (
        np.concatenate((ref_site.atoms.indices, alt_solvent.atoms.indices)) + 1
    )
    all_indices_t1 = (
        np.concatenate((ref_solvent.atoms.indices, alt_site.atoms.indices)) + 1
    )

    dict_sc_mask = {
        "scmk1_all_indices": all_indices_t0.astype(int).tolist(),
        "scmk1_cc_site_indices": cc_indices_site_t0.astype(int).tolist(),
        "scmk1_cc_solvent_indices": cc_indices_solvent_t0.astype(int).tolist(),
        "scmk2_all_indices": all_indices_t1.astype(int).tolist(),
        "scmk2_cc_site_indices": cc_indices_site_t1.astype(int).tolist(),
        "scmk2_cc_solvent_indices": cc_indices_solvent_t1.astype(int).tolist(),
    }

    with open(window_dir / "scmask.json", "w") as f:
        json.dump(dict_sc_mask, f)

    return


@register_create_box("y")
def create_box_y(ctx: BuildContext) -> None:
    """
    Create the box for ligand-only (solvation FE) systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir
    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    mol = ctx.residue_name
    buffer_x = float(sim.buffer_x)
    buffer_y = float(sim.buffer_y)
    buffer_z = float(sim.buffer_z)
    if buffer_x < 10 or buffer_y < 10 or buffer_z < 10:
        raise ValueError(
            f"For water systems, buffer_x/y/z must be ≥ 10 Å; got {buffer_x}/{buffer_y}/{buffer_z}."
        )
    if not hasattr(sim, "water_model"):
        raise AttributeError("SimulationConfig missing 'water_model'.")
    water_model = str(sim.water_model).upper()

    if not hasattr(sim, "ion_def"):
        raise AttributeError("SimulationConfig missing 'ion_def'.")
    ion_def = sim.ion_def
    if len(ion_def) < 3:
        raise ValueError("`ion_def` must contain [cation, anion, concentration].")

    if not hasattr(sim, "neut"):
        raise AttributeError("SimulationConfig missing 'neut'.")
    neut = str(sim.neut).lower()

    comp = ctx.comp
    param_dir = (
        (work.parent.parent / "params") if comp != "q" else (work.parent / "params")
    )

    build_pdb = window_dir / "build.pdb"
    if not build_pdb.exists():
        fallback = build_dir / "build.pdb"
        if fallback.exists():
            _cp(fallback, build_pdb)
        else:
            raise FileNotFoundError(
                f"[create_box_y] build.pdb missing in {window_dir} (fallback: {fallback})."
            )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_y] Optional/absent: {src}")

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / f"vac_ligand.{ext}")

    # --- copy a base tleap template into window_dir ---
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    if not src_tleap.exists():
        raise FileNotFoundError(
            "No tleap template found (tleap.in[.amber16]) in amber_dir."
        )
    _cp(src_tleap, window_dir / "tleap.in")

    # --- build the vacuum unit from ligand PDB (vac.*) ---
    tleap_lig_txt = (window_dir / "tleap.in").read_text().splitlines()
    tleap_lig_txt += [
        "# ligand-only vacuum topology",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f'set {{{mol}.1}} name "{mol}"\n',
        f"lig = loadpdb {mol}.pdb",
        "desc lig",
        "savepdb lig vac.pdb",
        "saveamberparm lig vac.prmtop vac.inpcrd",
        "quit",
    ]
    _write(window_dir / "tleap_ligands.in", "\n".join(tleap_lig_txt) + "\n")
    run_with_log(
        f"{tleap} -s -f tleap_ligands.in > tleap_ligands.log", working_dir=window_dir
    )

    # --- determine water box keyword ---
    if water_model == "TIP3PF":
        water_box = "FB3BOX"  # leaprc.water.fb3
        water_leaprc = "leaprc.water.fb3"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
        water_leaprc = "leaprc.water.spce"
    else:
        water_box = f"{water_model}BOX"
        water_leaprc = f"leaprc.water.{water_model.lower()}"

    # --- read ligand net charge from tleap log (unperturbed unit charge line) ---
    def _unit_charge_from_log(logfile: Path) -> int:
        if not logfile.exists():
            return 0
        q = 0.0
        for ln in logfile.read_text().splitlines():
            if "The unperturbed charge of the unit" in ln:
                try:
                    q = float(ln.split()[6].strip("'\",.:;#()[]"))
                except Exception:
                    pass
        return int(round(q))

    lig_charge = _ligand_charge_from_metadata(param_dir / f"{ctx.residue_name}.json")
    # put a minimum of 5 ions
    box_volume_A3 = 2 * buffer_x * 2 * buffer_y * 2 * buffer_z
    num_ions = max(
        5,
        round(ion_def[2] * 6.02e23 * box_volume_A3 * 1e-27),
    )

    add_neu_cat = max(0, -lig_charge)
    add_neu_ani = max(0, lig_charge)

    tleap_solv_lines = (window_dir / "tleap.in").read_text().splitlines()
    tleap_solv_lines += [
        "# ligand-only solvation",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f"source {water_leaprc}",
        f'set {{{mol}.1}} name "{mol}"',
        f"model = loadpdb {build_pdb.name}",
        "",
        f"solvatebox model {water_box} {{ {buffer_x:.3f} {buffer_y:.3f} {buffer_z:.3f} }} 1",
        "",
        "# ions",
    ]
    if neut == "no":
        if num_ions > 0 or add_neu_cat > 0 or add_neu_ani > 0:
            tleap_solv_lines += [
                f"addionsrand model {ion_def[0]} {num_ions + add_neu_cat}",
                f"addionsrand model {ion_def[1]} {num_ions + add_neu_ani}",
            ]
    else:
        if add_neu_cat:
            tleap_solv_lines.append(f"addionsrand model {ion_def[0]} {add_neu_cat}")
        if add_neu_ani:
            tleap_solv_lines.append(f"addionsrand model {ion_def[1]} {add_neu_ani}")

    tleap_solv_lines += [
        "desc model",
        "savepdb model full_pre.pdb",
        "quit",
        "",
    ]
    _write(window_dir / "tleap_solvate.in", "\n".join(tleap_solv_lines))
    run_with_log(
        f"{tleap} -s -f tleap_solvate.in > tleap_solvate.log", working_dir=window_dir
    )

    # --- process full_pre.pdb into final full.{prmtop,inpcrd,pdb} ---
    #
    u = mda.Universe(str(window_dir / "full_pre.pdb"))
    final_system = u.atoms
    system_dimensions = u.dimensions[:3]
    final_system_dum = final_system.select_atoms("resname DUM")
    final_system_lig = final_system.select_atoms(f"resname {mol}")
    final_system_others = final_system - final_system_dum - final_system_lig

    _write_res_blocks(final_system_dum, window_dir / "solvate_pre_dum.pdb")
    _write_res_blocks(final_system_lig, window_dir / "solvate_pre_lig.pdb")
    _write_res_blocks(final_system_others, window_dir / "solvate_pre_others.pdb")

    # tleap parts
    # dum
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_dum.in")
    with (window_dir / "tleap_solvate_dum.in").open("a") as f:
        f.write("dum = loadpdb solvate_pre_dum.pdb\n\n")
        f.write(
            f"set dum box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb dum solvate_dum.pdb\n")
        f.write("saveamberparm dum solvate_dum.prmtop solvate_dum.inpcrd\nquit\n")
    run_with_log(
        f"{tleap} -s -f tleap_solvate_dum.in > tleap_dum.log", working_dir=window_dir
    )

    # ligand
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_lig.in")
    with (window_dir / "tleap_solvate_lig.in").open("a") as f:
        f.write(f"loadamberparams {mol}.frcmod\n")
        f.write(f"{mol} = loadmol2 {mol}.mol2\n\n")
        f.write(f'set {{{mol}.1}} name "{mol}"\n')
        f.write("lig = loadpdb solvate_pre_lig.pdb\n\n")
        f.write(
            f"set lig box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb lig solvate_ligands.pdb\n")
        f.write(
            "saveamberparm lig solvate_ligands.prmtop solvate_ligands.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_lig.in > tleap_lig.log", working_dir=window_dir
    )

    # others
    _cp(window_dir / "tleap.in", window_dir / "tleap_solvate_others.in")
    with (window_dir / "tleap_solvate_others.in").open("a") as f:
        if water_model != "TIP3PF":
            f.write(f"source leaprc.water.{water_model.lower()}\n\n")
        else:
            f.write("source leaprc.water.fb3\n\n")
        f.write("others = loadpdb solvate_pre_others.pdb\n\n")
        f.write(
            f"set others box {{{system_dimensions[0]:.6f} {system_dimensions[1]:.6f} {system_dimensions[2]:.6f}}}\n"
        )
        f.write("savepdb others solvate_others.pdb\n")
        f.write(
            "saveamberparm others solvate_others.prmtop solvate_others.inpcrd\nquit\n"
        )
    run_with_log(
        f"{tleap} -s -f tleap_solvate_others.in > tleap_others.log",
        working_dir=window_dir,
    )

    # combine with ParmEd
    dum_p = pmd.load_file(
        str(window_dir / "solvate_dum.prmtop"), str(window_dir / "solvate_dum.inpcrd")
    )
    ligand_p = pmd.load_file(str(window_dir / f"{mol}.prmtop"))
    ligand_p.residues[0].name = mol
    lig_inp = pmd.load_file(str(window_dir / "solvate_ligands.inpcrd")).coordinates
    ligand_p.coordinates = lig_inp
    ligand_p.save(str(window_dir / f"{mol}.prmtop"), overwrite=True)

    others = pmd.load_file(
        str(window_dir / "solvate_others.prmtop"),
        str(window_dir / "solvate_others.inpcrd"),
    )
    combined = dum_p + ligand_p + others
    combined.save(str(window_dir / "full.prmtop"), overwrite=True)
    combined.save(str(window_dir / "full.inpcrd"), overwrite=True)
    combined.save(str(window_dir / "full.pdb"), overwrite=True)

    vac = dum_p + ligand_p
    vac.save(str(window_dir / "vac.prmtop"), overwrite=True)
    vac.save(str(window_dir / "vac.inpcrd"), overwrite=True)
    vac.save(str(window_dir / "vac.pdb"), overwrite=True)

    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    return


@register_create_box("m")
def create_box_m(ctx: BuildContext) -> None:
    """
    Create the box for ligand-only (vacuum) systems.
    Produces vac.{prmtop,inpcrd,pdb} and full.{prmtop,inpcrd,pdb}.
    """
    work = ctx.working_dir
    sim = ctx.sim
    amber_dir = ctx.amber_dir
    build_dir = ctx.build_dir
    window_dir = ctx.window_dir
    window_dir.mkdir(parents=True, exist_ok=True)

    mol = ctx.residue_name
    
    comp = ctx.comp
    param_dir = (
        (work.parent.parent / "params") if comp != "q" else (work.parent / "params")
    )

    # --- stage required ligand artifacts into window_dir ---
    for ext in ("frcmod", "lib", "prmtop", "inpcrd", "mol2", "sdf", "pdb", "json"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / src.name)
        else:
            logger.debug(f"[create_box_m] Optional/absent: {src}")

    for ext in ("prmtop", "mol2", "sdf", "inpcrd"):
        src = param_dir / f"{mol}.{ext}"
        if src.exists():
            _cp(src, window_dir / f"vac_ligand.{ext}")

    # --- copy a base tleap template into window_dir ---
    src_tleap = amber_dir / "tleap.in.amber16"
    if not src_tleap.exists():
        src_tleap = amber_dir / "tleap.in"
    if not src_tleap.exists():
        raise FileNotFoundError(
            "No tleap template found (tleap.in[.amber16]) in amber_dir."
        )
    _cp(src_tleap, window_dir / "tleap.in")

    # --- build the vacuum unit from ligand PDB (vac.*) ---
    tleap_lig_txt = (window_dir / "tleap.in").read_text().splitlines()
    tleap_lig_txt += [
        "# ligand-only vacuum topology",
        f"loadamberparams {mol}.frcmod",
        f"{mol} = loadmol2 {mol}.mol2",
        f'set {{{mol}.1}} name "{mol}"\n',
        f"lig = loadpdb {mol}.pdb",
        # set box to 40
        "set lig box {40.000000 40.000000 40.000000}",
        "desc lig",
        "savepdb lig vac.pdb",
        "saveamberparm lig vac.prmtop vac.inpcrd",
        "quit",
    ]
    _write(window_dir / "tleap_ligands.in", "\n".join(tleap_lig_txt) + "\n")
    run_with_log(
        f"{tleap} -s -f tleap_ligands.in > tleap_ligands.log", working_dir=window_dir
    )

    # copy ligand_p to vac.prmtop
    ligand_p_file = window_dir / f"{mol}.prmtop"
    _cp(ligand_p_file, window_dir / "vac.prmtop")

    # copy vac to full
    _cp(window_dir / "vac.pdb", window_dir / "full.pdb")
    _cp(window_dir / "vac.prmtop", window_dir / "full.prmtop")
    _cp(window_dir / "vac.inpcrd", window_dir / "full.inpcrd")
    
    run_parmed_hmr_if_enabled(sim.hmr, amber_dir, window_dir)
    full_prmtop = str(window_dir / "full.prmtop") if not sim.hmr else str(window_dir / "full.hmr.prmtop")
    return
