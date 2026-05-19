"""Network analysis and visualization helpers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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


def _as_pair_list(pairs: Sequence[Sequence[str] | tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"RBFE planned-network pairs must be 2-tuples; got {pair!r}.")
        out.append((str(pair[0]), str(pair[1])))
    return out


def _planned_graph_with_layout(
    ligands: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> tuple[Any, dict[str, np.ndarray]]:
    from batter.analysis.cinnabar import (
        _ensure_node_spacing,
        _import_networkx,
        _initial_component_layout,
        _layout_node_radii,
        _pack_component_layouts,
    )

    nx = _import_networkx()
    graph = nx.DiGraph()
    graph.add_nodes_from(str(ligand) for ligand in ligands)
    for index, (ref, alt) in enumerate(pairs, start=1):
        graph.add_edge(str(ref), str(alt), pair_index=index)

    if graph.number_of_nodes() == 0:
        return graph, {}

    component_layouts: list[tuple[dict[str, np.ndarray], dict[str, float]]] = []
    for component_nodes in nx.connected_components(graph.to_undirected()):
        subgraph = graph.subgraph(component_nodes).copy()
        radii = _layout_node_radii(subgraph)
        positions = _initial_component_layout(subgraph)
        positions = _ensure_node_spacing(
            positions,
            radii,
            padding=26.0,
            iterations=260 if subgraph.number_of_nodes() > 12 else 200,
        )
        component_layouts.append((positions, radii))
    return graph, _pack_component_layouts(component_layouts)


def _planned_ligand_assets(
    ligands: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    ligand_files: Mapping[str, str | Path] | None,
) -> dict[str, dict[str, str]]:
    if ligand_files is None:
        ligand_files = {}

    assets: dict[str, dict[str, str]] = {}
    try:
        import pandas as pd

        from batter.analysis.cinnabar import _build_ligand_assets

        records = []
        for ref, alt in pairs:
            ref_path = str(Path(ligand_files.get(ref, "")).expanduser()) if ref in ligand_files else ""
            alt_path = str(Path(ligand_files.get(alt, "")).expanduser()) if alt in ligand_files else ""
            records.append(
                {
                    "edge_label": f"{ref}~{alt}",
                    "original_path": json.dumps([ref_path, alt_path]),
                    "canonical_smiles": json.dumps(["", ""]),
                }
            )
        if records:
            assets = _build_ligand_assets(pd.DataFrame(records), edge_separator="~")
    except Exception:
        assets = {}

    for ligand in ligands:
        label = str(ligand)
        if label in assets:
            continue
        input_path = str(Path(ligand_files.get(label, "")).expanduser()) if label in ligand_files else ""
        svg = ""
        if input_path:
            try:
                from batter.analysis.cinnabar import _mol_from_any_path, _mol_to_svg_text

                mol = _mol_from_any_path(input_path)
                if mol is not None:
                    svg = _mol_to_svg_text(mol)
            except Exception:
                svg = ""
        assets[label] = {
            "label": label,
            "smiles": "",
            "input_path": input_path,
            "svg": svg,
        }
    return assets


def _planned_metadata_items(metadata: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    if not metadata:
        return []
    ordered_keys = (
        "mapping",
        "mapping_file",
        "konnektor_layout",
        "atom_mapper",
        "both_directions",
    )
    items: list[tuple[str, str]] = []
    for key in ordered_keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if value in (None, "", False):
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        items.append((key.replace("_", " "), rendered))
    return items


def _node_label_svg(label: str, radius: float) -> str:
    escaped = html.escape(label)
    font_size = 18 if len(label) <= 8 else 15 if len(label) <= 14 else 12
    max_width = radius * 1.62
    estimate = len(label) * font_size * 0.58
    fit_attrs = ""
    if estimate > max_width:
        fit_attrs = f' textLength="{max_width:.2f}" lengthAdjust="spacingAndGlyphs"'
    return (
        f"<text text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"{font_size}\" "
        "font-weight=\"700\" fill=\"#102a43\" paint-order=\"stroke\" stroke=\"white\" "
        f"stroke-width=\"5\" stroke-linejoin=\"round\"{fit_attrs}>{escaped}</text>"
    )


def write_planned_rbfe_network_html(
    *,
    ligands: Sequence[str],
    pairs: Sequence[Sequence[str] | tuple[str, str]],
    out_path: Path,
    ligand_files: Mapping[str, str | Path] | None = None,
    title: str = "BATTER planned RBFE network",
    metadata: Mapping[str, Any] | None = None,
    edge_assets: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Write an interactive HTML visualization for a planned RBFE network."""
    pair_list = _as_pair_list(pairs)
    ligand_list = [str(ligand) for ligand in ligands]
    if not ligand_list or not pair_list:
        return False

    try:
        from batter.analysis.cinnabar import (
            _node_color_mapping,
            _normalize_vec,
            _quadratic_bezier_tangent,
            _resolve_label_positions,
            _rgba_to_hex,
        )
    except Exception:
        return False

    graph, pos = _planned_graph_with_layout(ligand_list, pair_list)
    if not pos:
        return False

    color_meta = _node_color_mapping(graph, None)
    node_values = color_meta["values"]
    norm = color_meta["norm"]
    cmap = color_meta["cmap"]
    ligand_assets = _planned_ligand_assets(ligand_list, pair_list, ligand_files)

    xs = [float(coord[0]) for coord in pos.values()]
    ys = [float(coord[1]) for coord in pos.values()]
    layout_min_x, layout_max_x = min(xs), max(xs)
    layout_min_y, layout_max_y = min(ys), max(ys)
    layout_span_x = max(layout_max_x - layout_min_x, 1.0)
    layout_span_y = max(layout_max_y - layout_min_y, 1.0)

    pad_x = max(110, int(0.10 * layout_span_x))
    pad_y = max(90, int(0.11 * layout_span_y))
    canvas_w = int(max(1100, layout_span_x + 2.0 * pad_x))
    canvas_h = int(max(640, layout_span_y + 2.0 * pad_y))

    def _to_xy(point: np.ndarray) -> tuple[float, float]:
        x = pad_x + (float(point[0]) - layout_min_x)
        y = pad_y + (layout_max_y - float(point[1]))
        return x, y

    def _edge_curvature(node_a: str, node_b: str) -> float:
        if graph.has_edge(node_b, node_a) and node_a != node_b:
            return 0.24
        return 0.0

    edge_color = "#7c3aed"
    node_degree = dict(graph.degree())
    node_radius = {node: 26.0 + 2.0 * node_degree[node] for node in graph.nodes}
    node_fill: dict[str, str] = {}
    for node, value in zip(graph.nodes, node_values):
        if norm is not None and cmap is not None and np.isfinite(value):
            node_fill[node] = _rgba_to_hex(cmap(norm(float(value))))
        else:
            node_fill[node] = "#88c0d0"

    planned_edges: dict[str, dict[str, Any]] = {}
    edge_svg: list[str] = []
    label_specs: list[dict[str, np.ndarray]] = []
    label_payloads: list[tuple[str, str]] = []
    for node_a, node_b, data in graph.edges(data=True):
        node_a = str(node_a)
        node_b = str(node_b)
        edge_key = f"{node_a}~{node_b}"
        pair_index = int(data.get("pair_index", len(planned_edges) + 1))
        planned_edges[edge_key] = {
            "edge_key": edge_key,
            "display_title": f"{node_a} -> {node_b}",
            "pair_index": pair_index,
            "ref": node_a,
            "alt": node_b,
        }
        if edge_assets and edge_key in edge_assets:
            planned_edges[edge_key].update(dict(edge_assets[edge_key]))

        curvature = _edge_curvature(node_a, node_b)
        start = np.asarray(_to_xy(pos[node_a]), dtype=float)
        end = np.asarray(_to_xy(pos[node_b]), dtype=float)
        direction = end - start
        unit_dir = _normalize_vec(direction, fallback=np.array([1.0, 0.0]))
        perp = np.array([-unit_dir[1], unit_dir[0]])
        stroke_width = 4.0
        head_length = 11.0 + 1.6 * stroke_width
        head_half_width = 4.5 + 0.85 * stroke_width
        start2 = start + unit_dir * (node_radius[node_a] + 5.0)
        tip = end - unit_dir * (node_radius[node_b] + 8.0)
        span = np.linalg.norm(tip - start2)
        control = 0.5 * (start2 + tip) + perp * curvature * span * 0.75
        tip_tangent = _normalize_vec(
            _quadratic_bezier_tangent(start2, control, tip, 1.0),
            fallback=unit_dir,
        )
        tip_normal = np.array([-tip_tangent[1], tip_tangent[0]])
        shaft_end = tip - tip_tangent * head_length
        arrow_left = shaft_end + tip_normal * head_half_width
        arrow_right = shaft_end - tip_normal * head_half_width
        path_d = (
            f"M {start2[0]:.2f} {start2[1]:.2f} "
            f"Q {control[0]:.2f} {control[1]:.2f} {shaft_end[0]:.2f} {shaft_end[1]:.2f}"
        )
        edge_svg.append(
            f"<g class=\"edge-path\" data-edge=\"{html.escape(edge_key)}\">"
            f"<path d=\"{path_d}\" fill=\"none\" stroke=\"transparent\" stroke-width=\"16\" "
            "stroke-linecap=\"round\" pointer-events=\"stroke\" />"
            f"<path d=\"{path_d}\" fill=\"none\" stroke=\"{edge_color}\" "
            f"stroke-width=\"{stroke_width:.2f}\" stroke-linecap=\"round\" stroke-opacity=\"0.94\" />"
            f"<polygon points=\"{tip[0]:.2f},{tip[1]:.2f} {arrow_left[0]:.2f},{arrow_left[1]:.2f} "
            f"{arrow_right[0]:.2f},{arrow_right[1]:.2f}\" fill=\"{edge_color}\" stroke=\"{edge_color}\" "
            "stroke-linejoin=\"round\" stroke-linecap=\"round\" />"
            "</g>"
        )

        text_pos = 0.25 * start2 + 0.5 * control + 0.25 * tip
        text_pos = text_pos + perp * curvature * span * 0.18
        label_specs.append({"base": text_pos, "tangent": unit_dir, "normal": perp})
        label_payloads.append((edge_key, f"T{pair_index}"))

    label_svg: list[str] = []
    resolved_label_positions = _resolve_label_positions(label_specs, box_size=(44.0, 26.0))
    for resolved_pos, (edge_key, label) in zip(resolved_label_positions, label_payloads):
        label_svg.append(
            f"<g class=\"edge-label\" data-edge=\"{html.escape(edge_key)}\">"
            f"<rect x=\"{resolved_pos[0] - 22:.2f}\" y=\"{resolved_pos[1] - 13:.2f}\" width=\"44\" height=\"26\" "
            "rx=\"6\" ry=\"6\" fill=\"white\" fill-opacity=\"0.94\" stroke=\"#cbd2d9\" stroke-width=\"1\" />"
            f"<text x=\"{resolved_pos[0]:.2f}\" y=\"{resolved_pos[1] + 4:.2f}\" text-anchor=\"middle\" "
            f"font-size=\"12\" font-weight=\"700\" fill=\"#243b53\">{html.escape(label)}</text>"
            "</g>"
        )

    node_svg: list[str] = []
    for node in graph.nodes:
        node = str(node)
        x, y = _to_xy(pos[node])
        escaped = html.escape(node)
        node_svg.append(
            "<g class=\"node\" "
            f"data-ligand=\"{escaped}\" transform=\"translate({x:.2f},{y:.2f})\">"
            f"<circle r=\"{node_radius[node]:.2f}\" fill=\"{node_fill[node]}\" "
            "stroke=\"#243b53\" stroke-width=\"3\" />"
            f"{_node_label_svg(node, node_radius[node])}"
            "</g>"
        )

    summary_items = [
        ("ligands", str(len(ligand_list))),
        ("planned transformations", str(len(pair_list))),
        *_planned_metadata_items(metadata),
    ]
    summary_html = "".join(
        f"<span class=\"summary-chip\"><strong>{html.escape(key)}</strong>{html.escape(value)}</span>"
        for key, value in summary_items
    )
    notes = [
        "Planned transformations are labeled T1, T2, ... in the order stored in rbfe_network.json.",
        "Node colors reflect graph degree at planning time.",
    ]

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7fb; color: #102a43; }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 18px 18px 28px; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; text-align: center; }}
    .summarybar {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 0 0 14px; }}
    .summary-chip {{ display: inline-flex; align-items: center; gap: 7px; border: 1px solid #cbd2d9; background: white; color: #334e68; border-radius: 8px; padding: 7px 11px; font-size: 13px; }}
    .summary-chip strong {{ color: #102a43; font-weight: 700; }}
    .panel {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); overflow: hidden; }}
    .network-toolbar {{ display: flex; justify-content: flex-end; gap: 8px; padding: 12px 14px 0; }}
    .zoom-btn {{ border: 1px solid #cbd2d9; background: white; color: #334e68; border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer; }}
    .zoom-btn:hover {{ border-color: #9fb3c8; background: #f8fafc; }}
    svg {{ width: 100%; height: auto; display: block; background: #f6f7fb; touch-action: none; user-select: none; }}
    .network-pan-surface {{ cursor: grab; }}
    .network-pan-surface.dragging {{ cursor: grabbing; }}
    .node, .edge-path, .edge-label {{ cursor: pointer; }}
    .notes {{ margin: 12px 14px 14px; padding: 10px 12px; border: 1px solid #cbd2d9; border-radius: 8px; background: rgba(255,255,255,0.96); color: #486581; white-space: pre-line; font-size: 13px; }}
    #stickies {{ position: fixed; inset: 0; pointer-events: none; z-index: 1000; }}
    .sticky-note {{ position: fixed; width: 280px; min-height: 130px; background: #fff9c4; border: 1px solid #e0c56e; border-radius: 8px; box-shadow: 0 16px 38px rgba(15, 23, 42, 0.18); padding: 12px 12px 10px; pointer-events: auto; }}
    .sticky-note.edge-note {{ width: 360px; background: #eef2ff; border-color: #c7d2fe; }}
    .sticky-header {{ display: flex; align-items: center; justify-content: space-between; font-weight: 700; margin-bottom: 8px; color: #6b4f00; cursor: move; }}
    .sticky-note.edge-note .sticky-header {{ color: #3730a3; }}
    .sticky-close {{ border: 0; background: transparent; color: inherit; font-size: 18px; line-height: 1; cursor: pointer; }}
    .sticky-body .smiles {{ margin-top: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #52606d; word-break: break-all; }}
    .sticky-body .empty {{ font-size: 12px; color: #7b8794; }}
    .sticky-body .mapping-image {{ display: block; width: 100%; max-height: 220px; object-fit: contain; border: 1px solid #cbd2d9; border-radius: 6px; background: white; margin: 6px 0 8px; }}
    .sticky-meta {{ margin-top: 8px; font-size: 12px; color: #52606d; line-height: 1.45; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <div class="summarybar">{summary_html}</div>
    <div class="panel">
      <div class="network-toolbar">
        <button class="zoom-btn" id="network-zoom-in" type="button">+</button>
        <button class="zoom-btn" id="network-zoom-out" type="button">-</button>
        <button class="zoom-btn" id="network-fit" type="button">Fit</button>
        <button class="zoom-btn" id="network-reset" type="button">Reset</button>
      </div>
      <svg id="network-svg" viewBox="0 0 {canvas_w} {canvas_h}" role="img" aria-label="{html.escape(title)}">
        <rect id="network-pan-surface" class="network-pan-surface" x="0" y="0" width="{canvas_w}" height="{canvas_h}" fill="#f6f7fb" />
        <g id="network-viewport">
          {''.join(edge_svg)}
          {''.join(label_svg)}
          {''.join(node_svg)}
        </g>
      </svg>
      <div class="notes">{html.escape(chr(10).join(notes))}</div>
    </div>
  </div>
  <div id="stickies"></div>
  <script>
    const ligandAssets = {json.dumps(ligand_assets)};
    const plannedEdges = {json.dumps(planned_edges)};
    const stickyRoot = document.getElementById('stickies');
    const networkSvg = document.getElementById('network-svg');
    const networkViewport = document.getElementById('network-viewport');
    const networkPanSurface = document.getElementById('network-pan-surface');
    let networkScale = 1.0;
    let networkPanX = 0.0;
    let networkPanY = 0.0;
    let networkDragging = false;
    let dragStartX = 0.0;
    let dragStartY = 0.0;
    let dragPanX = 0.0;
    let dragPanY = 0.0;
    let zCounter = 1000;

    function updateNetworkTransform() {{
      networkViewport.setAttribute(
        'transform',
        `translate(${{networkPanX.toFixed(2)}} ${{networkPanY.toFixed(2)}}) scale(${{networkScale.toFixed(5)}})`
      );
    }}

    function fitNetworkViewport(extraScale = 1.0) {{
      const bbox = networkViewport.getBBox();
      const viewBox = networkSvg.viewBox.baseVal;
      if (!bbox || bbox.width <= 0 || bbox.height <= 0) return;
      const pad = 32.0;
      const scaleX = (viewBox.width - 2.0 * pad) / bbox.width;
      const scaleY = (viewBox.height - 2.0 * pad) / bbox.height;
      networkScale = Math.min(scaleX, scaleY) * extraScale;
      networkPanX = viewBox.x + (viewBox.width - bbox.width * networkScale) * 0.5 - bbox.x * networkScale;
      networkPanY = viewBox.y + (viewBox.height - bbox.height * networkScale) * 0.5 - bbox.y * networkScale;
      updateNetworkTransform();
    }}

    function zoomNetwork(factor, clientX = null, clientY = null) {{
      const viewBox = networkSvg.viewBox.baseVal;
      const rect = networkSvg.getBoundingClientRect();
      const anchorX = clientX === null ? rect.left + rect.width * 0.5 : clientX;
      const anchorY = clientY === null ? rect.top + rect.height * 0.5 : clientY;
      const svgX = viewBox.x + ((anchorX - rect.left) / rect.width) * viewBox.width;
      const svgY = viewBox.y + ((anchorY - rect.top) / rect.height) * viewBox.height;
      const nextScale = Math.min(8.0, Math.max(0.25, networkScale * factor));
      const localX = (svgX - networkPanX) / networkScale;
      const localY = (svgY - networkPanY) / networkScale;
      networkScale = nextScale;
      networkPanX = svgX - localX * networkScale;
      networkPanY = svgY - localY * networkScale;
      updateNetworkTransform();
    }}

    function bringToFront(note) {{
      zCounter += 1;
      note.style.zIndex = String(zCounter);
    }}

    function makeDraggable(note) {{
      const header = note.querySelector('.sticky-header');
      let startX = 0, startY = 0, startLeft = 0, startTop = 0, dragging = false;
      header.addEventListener('pointerdown', (event) => {{
        if (event.target && event.target.closest('.sticky-close')) return;
        dragging = true;
        bringToFront(note);
        startX = event.clientX;
        startY = event.clientY;
        startLeft = parseFloat(note.style.left || '0');
        startTop = parseFloat(note.style.top || '0');
        header.setPointerCapture(event.pointerId);
      }});
      header.addEventListener('pointermove', (event) => {{
        if (!dragging) return;
        note.style.left = `${{startLeft + event.clientX - startX}}px`;
        note.style.top = `${{startTop + event.clientY - startY}}px`;
      }});
      function endDrag(event) {{
        dragging = false;
        try {{ header.releasePointerCapture(event.pointerId); }} catch (_e) {{}}
      }}
      header.addEventListener('pointerup', endDrag);
      header.addEventListener('pointercancel', endDrag);
    }}

    function stickyBodyHtml(label) {{
      const asset = ligandAssets[label] || {{}};
      const svg = asset.svg || '<div class="empty">No 2D structure available</div>';
      const smiles = asset.smiles ? `<div class="smiles">${{asset.smiles}}</div>` : '';
      return `<div class="sticky-body">${{svg}}${{smiles}}</div>`;
    }}

    function edgeBodyHtml(edgeKey) {{
      const edge = plannedEdges[edgeKey] || {{}};
      const ref = edge.ref || '';
      const alt = edge.alt || '';
      const index = edge.pair_index ? `T${{edge.pair_index}}` : edgeKey;
      const imgSrc = edge.image_data_uri || edge.image_src || '';
      const image = imgSrc
        ? `<img class="mapping-image" src="${{imgSrc}}" alt="${{edge.image_alt || 'Atom mapping'}}" />`
        : '';
      const nMapped = edge.n_mapped ? `<br />mapped atoms: ${{edge.n_mapped}}` : '';
      const mapper = edge.mapper ? `<br />mapper: ${{edge.mapper}}` : '';
      return `<div class="sticky-body">${{image}}<div class="sticky-meta">transformation: ${{index}}<br />reference: ${{ref}}<br />target: ${{alt}}${{mapper}}${{nMapped}}</div></div>`;
    }}

    function openSticky(kind, key, event) {{
      const selector = kind === 'edge'
        ? `.sticky-note[data-edge="${{CSS.escape(key)}}"]`
        : `.sticky-note[data-ligand="${{CSS.escape(key)}}"]`;
      const existing = document.querySelector(selector);
      if (existing) {{
        bringToFront(existing);
        return;
      }}
      const note = document.createElement('div');
      note.className = kind === 'edge' ? 'sticky-note edge-note' : 'sticky-note';
      if (kind === 'edge') note.dataset.edge = key;
      else note.dataset.ligand = key;
      const width = kind === 'edge' ? 360 : 320;
      note.style.left = `${{Math.min(window.innerWidth - width, Math.max(16, event.clientX + 12))}}px`;
      note.style.top = `${{Math.min(window.innerHeight - 280, Math.max(16, event.clientY + 12))}}px`;
      const edge = plannedEdges[key] || {{}};
      const title = kind === 'edge' ? (edge.display_title || key.replace('~', ' -> ')) : key;
      const body = kind === 'edge' ? edgeBodyHtml(key) : stickyBodyHtml(key);
      note.innerHTML = `
        <div class="sticky-header">
          <span>${{title}}</span>
          <button class="sticky-close" type="button" aria-label="Close">&times;</button>
        </div>
        ${{body}}
      `;
      stickyRoot.appendChild(note);
      bringToFront(note);
      makeDraggable(note);
      note.addEventListener('pointerdown', () => bringToFront(note));
      const closeButton = note.querySelector('.sticky-close');
      closeButton.addEventListener('pointerdown', (closeEvent) => closeEvent.stopPropagation());
      closeButton.addEventListener('click', (closeEvent) => {{
        closeEvent.preventDefault();
        closeEvent.stopPropagation();
        note.remove();
      }});
    }}

    document.querySelectorAll('.node').forEach((element) => {{
      element.addEventListener('click', (event) => {{
        const label = element.getAttribute('data-ligand') || '';
        if (label) openSticky('ligand', label, event);
      }});
    }});

    document.querySelectorAll('.edge-path, .edge-label').forEach((element) => {{
      element.addEventListener('click', (event) => {{
        const edgeKey = element.getAttribute('data-edge') || '';
        if (edgeKey) openSticky('edge', edgeKey, event);
      }});
    }});

    document.getElementById('network-zoom-in').addEventListener('click', () => zoomNetwork(1.18));
    document.getElementById('network-zoom-out').addEventListener('click', () => zoomNetwork(1.0 / 1.18));
    document.getElementById('network-fit').addEventListener('click', () => fitNetworkViewport(1.0));
    document.getElementById('network-reset').addEventListener('click', () => fitNetworkViewport(0.96));

    networkSvg.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoomNetwork(event.deltaY < 0 ? 1.12 : (1.0 / 1.12), event.clientX, event.clientY);
    }}, {{ passive: false }});

    networkSvg.addEventListener('pointerdown', (event) => {{
      if (event.target && event.target.closest('.node, .edge-path, .edge-label')) return;
      networkDragging = true;
      dragStartX = event.clientX;
      dragStartY = event.clientY;
      dragPanX = networkPanX;
      dragPanY = networkPanY;
      networkPanSurface.classList.add('dragging');
      networkSvg.setPointerCapture(event.pointerId);
    }});

    networkSvg.addEventListener('pointermove', (event) => {{
      if (!networkDragging) return;
      networkPanX = dragPanX + (event.clientX - dragStartX);
      networkPanY = dragPanY + (event.clientY - dragStartY);
      updateNetworkTransform();
    }});

    function endNetworkDrag(event) {{
      if (!networkDragging) return;
      networkDragging = false;
      networkPanSurface.classList.remove('dragging');
      try {{ networkSvg.releasePointerCapture(event.pointerId); }} catch (_e) {{}}
    }}
    networkSvg.addEventListener('pointerup', endNetworkDrag);
    networkSvg.addEventListener('pointercancel', endNetworkDrag);
    networkSvg.addEventListener('pointerleave', endNetworkDrag);
    fitNetworkViewport(0.96);
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path.exists()
