from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


def parse_results_dat(path: Path) -> Tuple[float | None, float | None, Dict[str, float]]:
    """
    Parse fe/Results/Results.dat and return totals plus per-component values.
    """
    per_comp: Dict[str, float] = {}
    total_dG, total_se = None, None
    for raw in path.read_text().splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [t for t in ln.replace("\t", " ").split() if t]
        if len(parts) < 3:
            continue
        label, fe_str, se_str = parts[0], parts[1], parts[2]
        try:
            fe = float(fe_str)
            se = float(se_str)
        except ValueError:
            continue
        if label.lower() == "total":
            total_dG, total_se = fe, se
        else:
            per_comp[f"{label}_fe"] = fe
            per_comp[f"{label}_se"] = se
    return total_dG, total_se, per_comp


def fallback_totals_from_json(results_dir: Path) -> Tuple[float | None, float | None]:
    """
    Look for totals in JSON outputs produced by analysis.
    """
    zjson = results_dir / "z_results.json"
    if zjson.exists():
        try:
            data = json.loads(zjson.read_text())
            fe = data.get("fe")
            se = data.get("fe_error")
            if fe is not None and se is not None:
                return float(fe), float(se)
        except Exception:
            pass

    for js in sorted(results_dir.glob("*_results.json")):
        try:
            data = json.loads(js.read_text())
            fe = data.get("fe")
            se = data.get("fe_error")
            if fe is not None and se is not None:
                return float(fe), float(se)
        except Exception:
            continue
    return None, None
