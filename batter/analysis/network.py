"""Network analysis and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def plot_rbfe_network(
    *,
    available: List[str],
    lig_map: Dict[str, str],
    pairs: List[tuple[str, str]] | List[list[str]],
    mapping_name: str,
    mapping_file: str | Path | None,
    layout: str | None,
    out_dir: Path,
) -> None:
    """Best-effort network visualization using konnektor."""
    try:
        from konnektor.visualization import draw_ligand_network
        from konnektor.network_planners import ExplicitNetworkGenerator
        from gufe import SmallMoleculeComponent
        from kartograf.atom_mapper import KartografAtomMapper
    except Exception:
        return

    try:
        from batter.rbfe import _load_rdkit_mol, _resolve_konnektor_generator
    except Exception:
        return

    components: List[SmallMoleculeComponent] = []
    comp_by_name: Dict[str, SmallMoleculeComponent] = {}
    for lig in available:
        path = Path(lig_map[lig])
        mol = _load_rdkit_mol(path)
        comp = SmallMoleculeComponent(mol, name=lig)
        components.append(comp)
        comp_by_name[lig] = comp

    mapper = KartografAtomMapper()

    def _null_scorer(_mapping):
        return 0.0

    network = None
    try:
        if mapping_file:
            edges = [
                (comp_by_name[str(a)], comp_by_name[str(b)])
                for a, b in pairs
                if str(a) in comp_by_name and str(b) in comp_by_name
            ]
            generator = ExplicitNetworkGenerator(mappers=mapper, scorer=_null_scorer)
            network = generator.generate_ligand_network(edges=edges, nodes=components)
        elif mapping_name == "konnektor":
            generator_cls = _resolve_konnektor_generator(layout)
            try:
                generator = generator_cls(mappers=mapper, scorer=_null_scorer)
            except TypeError:
                generator = generator_cls(mappers=mapper)
            if hasattr(generator, "generate_ligand_network"):
                network = generator.generate_ligand_network(components)
            elif hasattr(generator, "generate_network"):
                network = generator.generate_network(components)
            elif callable(generator):
                network = generator(components)
        elif mapping_name in {"default", "star", "first"}:
            generator_cls = _resolve_konnektor_generator("star")
            try:
                generator = generator_cls(mappers=mapper, scorer=_null_scorer)
            except TypeError:
                generator = generator_cls(mappers=mapper)
            if hasattr(generator, "generate_ligand_network"):
                network = generator.generate_ligand_network(components)
            elif hasattr(generator, "generate_network"):
                network = generator.generate_network(components)
            elif callable(generator):
                network = generator(components)
    except Exception:
        network = None

    if network is None:
        return

    try:
        fig = draw_ligand_network(network=network, title=getattr(network, "name", None))
        out_path = out_dir / "rbfe_network.png"
        fig.savefig(out_path, dpi=200)
    except Exception:
        return
