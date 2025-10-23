# batter/_internal/ops/amber_templates.py
from __future__ import annotations

from pathlib import Path
from loguru import logger
import shutil
import os

from batter.config.simulation import SimulationConfig

from batter._internal.templates import AMBER_FILES_DIR as amber_files_orig  # type: ignore


def _resolve_ligand_ff(sim: SimulationConfig) -> str:
    """
    If user picked an OpenFF for the ligand, the legacy AMBER decks expect GAFF2 for build.
    """
    lig_ff = getattr(sim, "ligand_ff", "gaff2")
    if lig_ff and "openff" in str(lig_ff).lower():
        return "gaff2"
    return str(lig_ff or "gaff2")


def write_amber_templates(
    out_dir: Path,
    sim: SimulationConfig,
    *,
    membrane: bool,
    production: bool,
) -> None:
    """
    Render AMBER template files into `out_dir`.

    Notes
    -----
    This function ports your original `_create_amber_files` logic:
    - copy the template tree
    - perform string substitutions (dt, ntpr, ntwr, ntwe, ntwx, cut, gamma_ln, barostat, ff, p_coupling, c_surften, etc.)
    - handle `hmr` and membrane branches

    Placeholders supported (as in your legacy decks):
      _step_, _ntpr_, _ntwr_, _ntwe_, _ntwx_, _cutoff_, _gamma_ln_, _barostat_,
      _receptor_ff_, _ligand_ff_, _lipid_ff_, _p_coupling_, _c_surften_
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(amber_files_orig)
    
    # 1) Copy template tree
    shutil.copytree(src, out_dir, dirs_exist_ok=True)
    logger.debug(f"Copied AMBER templates {src} → {out_dir}")

    # 2) Compute substitutions
    dt = getattr(sim, "dt", 0.002)
    ntpr = getattr(sim, "ntpr", 500)
    ntwr = getattr(sim, "ntwr", 5000)
    ntwe = getattr(sim, "ntwe", 5000)
    ntwx = getattr(sim, "ntwx", 5000)
    cut = getattr(sim, "cut", 9.0)
    gamma_ln = getattr(sim, "gamma_ln", 2.0)
    barostat = getattr(sim, "barostat", "1")

    receptor_ff = getattr(sim, "receptor_ff", "protein.ff14SB")
    # only used for building the system
    ligand_ff = _resolve_ligand_ff(sim)
    lipid_ff = getattr(sim, "lipid_ff", "lipid21")

    # Membrane-specific knobs (legacy names preserved)
    if membrane:
        p_coupling = getattr(sim, "p_coupling", "3")   # semiisotropic
        c_surften = getattr(sim, "c_surften", "3")
    else:
        p_coupling = "1"  # isotropic/anisotropic off for water only
        c_surften = "0"

    if production:
        p_coupling = "0"

    replacements = {
        "_step_": str(dt),
        "_ntpr_": str(ntpr),
        "_ntwr_": str(ntwr),
        "_ntwe_": str(ntwe),
        "_ntwx_": str(ntwx),
        "_cutoff_": str(cut),
        "_gamma_ln_": str(gamma_ln),
        "_barostat_": str(barostat),
        "_receptor_ff_": str(receptor_ff),
        "_ligand_ff_": str(ligand_ff),
        "_lipid_ff_": str(lipid_ff),
        "_p_coupling_": str(p_coupling),
        "_c_surften_": str(c_surften),
    }

    # 3) Apply substitutions to all text files under out_dir
    #    (safe heuristic: try to decode as text; if fails, skip)
    changed = 0
    for root, _, files in os.walk(out_dir):
        for fname in files:
            fpath = Path(root) / fname
            try:
                txt = fpath.read_text()
            except Exception:
                continue  # likely binary
            orig = txt
            for k, v in replacements.items():
                txt = txt.replace(k, v)
            if txt != orig:
                fpath.write_text(txt)
                changed += 1

    logger.debug(
        f"Rendering AMBER templates to {out_dir} (production={production}, membrane={membrane}); "
        f"applied substitutions in {changed} file(s)."
    )