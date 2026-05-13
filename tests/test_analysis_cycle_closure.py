from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from batter.analysis.cycle_closure import (
    calculate_cycle_closure,
    cycle_closure_from_dataframe,
    cycle_closure_from_file,
    read_cycle_closure_file,
)


def test_sfc_corrects_inconsistent_triangle_without_cycle_enumeration() -> None:
    df = pd.DataFrame(
        [
            {"labelA": "A", "labelB": "B", "calc_DDG": 1.0, "calc_dDDG": 0.2},
            {"labelA": "B", "labelB": "C", "calc_DDG": 1.0, "calc_dDDG": 0.2},
            {"labelA": "A", "labelB": "C", "calc_DDG": 3.0, "calc_dDDG": 0.2},
        ]
    )

    result = cycle_closure_from_dataframe(df, reference="A", reference_free_energy=-5.0)

    assert result.method == "sfc"
    assert result.cycles == ()
    assert result.schemes == ("sfc", "wsfc1")

    nodes = result.node_results.set_index("label")
    assert nodes.loc["A", "dG_sfc"] == pytest.approx(-5.0)
    assert nodes.loc["B", "dG_sfc"] == pytest.approx(-5.0 + 4.0 / 3.0)
    assert nodes.loc["C", "dG_sfc"] == pytest.approx(-5.0 + 8.0 / 3.0)
    assert nodes.loc["B", "dG_wsfc1"] == pytest.approx(nodes.loc["B", "dG_sfc"])
    assert nodes.loc["C", "path_dependent_error"] == pytest.approx(1.0 / 3.0)

    edges = result.edge_results.set_index(["labelA", "labelB"])
    assert edges.loc[("A", "B"), "ddG_sfc"] == pytest.approx(4.0 / 3.0)
    assert edges.loc[("B", "C"), "ddG_sfc"] == pytest.approx(4.0 / 3.0)
    assert edges.loc[("A", "C"), "ddG_sfc"] == pytest.approx(8.0 / 3.0)
    assert edges.loc[("A", "B"), "pair_error"] == pytest.approx(1.0 / 3.0)


def test_wsfc_uses_uncertainty_weights() -> None:
    df = pd.DataFrame(
        [
            {"labelA": "A", "labelB": "B", "calc_DDG": 1.0, "calc_dDDG": 0.1},
            {"labelA": "B", "labelB": "C", "calc_DDG": 1.0, "calc_dDDG": 0.1},
            {"labelA": "A", "labelB": "C", "calc_DDG": 3.0, "calc_dDDG": 10.0},
        ]
    )

    result = cycle_closure_from_dataframe(df, reference="A", reference_free_energy=0.0)
    edges = result.edge_results.set_index(["labelA", "labelB"])

    assert abs(edges.loc[("A", "B"), "ddG_wsfc1"] - 1.0) < abs(
        edges.loc[("A", "B"), "ddG_sfc"] - 1.0
    )
    assert abs(edges.loc[("B", "C"), "ddG_wsfc1"] - 1.0) < abs(
        edges.loc[("B", "C"), "ddG_sfc"] - 1.0
    )
    assert edges.loc[("A", "C"), "pair_error_wsfc1"] > edges.loc[
        ("A", "B"), "pair_error_wsfc1"
    ]


def test_wsfc_accepts_zero_uncertainty_as_exact_weight() -> None:
    df = pd.DataFrame(
        [
            {"labelA": "A", "labelB": "B", "calc_DDG": 1.0, "calc_dDDG": 0.0},
            {"labelA": "B", "labelB": "C", "calc_DDG": 1.0, "calc_dDDG": 0.2},
            {"labelA": "A", "labelB": "C", "calc_DDG": 3.0, "calc_dDDG": 0.2},
        ]
    )

    result = cycle_closure_from_dataframe(df, reference="A", reference_free_energy=0.0)

    assert result.schemes == ("sfc", "wsfc1")
    edges = result.edge_results.set_index(["labelA", "labelB"])
    assert edges.loc[("A", "B"), "ddG_wsfc1"] == pytest.approx(1.0)


def test_sfc_reads_whitespace_input_files(tmp_path: Path) -> None:
    input_file = tmp_path / "sfc_input.dat"
    input_file.write_text("A B 1.0 0.2\nB C 1.0 0.2\nA C 3.0 0.2\n")

    parsed = read_cycle_closure_file(input_file)
    assert list(parsed.columns) == ["labelA", "labelB", "ddG", "std1"]

    result = cycle_closure_from_file(
        input_file,
        reference="A",
        reference_free_energy=-5.0,
    )

    nodes = result.node_results.set_index("label")
    assert nodes.loc["C", "dG_sfc"] == pytest.approx(-5.0 + 8.0 / 3.0)
    assert "dG_wsfc1" in nodes.columns


def test_sfc_does_not_require_a_cycle() -> None:
    edges = [("A", "B", 1.0), ("B", "C", 2.0)]

    result = calculate_cycle_closure(edges, reference="A", require_cycles=True)

    assert result.cycles == ()
    assert result.node_results.set_index("label").loc["C", "dG_sfc"] == pytest.approx(
        3.0
    )
