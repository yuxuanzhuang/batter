"""Network analysis and visualization helpers."""

from __future__ import annotations

from pathlib import Path


def plot_rbfe_network(
    *,
    network,
    out_dir: Path,
) -> None:
    """Best-effort network visualization using konnektor."""
    try:
        from konnektor.visualization import draw_ligand_network
    except Exception:
        return

    if network is None:
        return

    try:
        fig = draw_ligand_network(network=network, title=getattr(network, "name", None))
        out_path = out_dir / "rbfe_network.png"
        fig.savefig(out_path, dpi=200)
    except Exception:
        return
