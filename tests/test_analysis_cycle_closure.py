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


WEIGHTED_CC_EXAMPLE = [
    ("3K", "4M", 0.0909541527899095, 0.022054792, 0.1363923531744726),
    ("3N", "3A", -1.7328677789257083, 0.018099453, 0.2927123086170466),
    ("3M", "3A", 0.7434501318561217, 0.017608079, 0.2505733279184034),
    ("3K", "3N", 2.4118388418509227, 0.012039911, 0.299493329915705),
    ("4P", "3K", -0.4075985196531118, 0.00881092, 0.0722943747848214),
    ("4M", "4P", -0.6467934031158151, 0.004644101, 0.0306028362704773),
    ("3K", "4L", 1.1256236732255829, 0.013223868, 0.1586001706093542),
    ("4M", "3M", -0.1800389837976137, 0.012025464, 0.1562222472274439),
    ("4N", "3K", -1.3824461331326638, 0.011242459, 0.0878525636390685),
    ("4I", "3N", 0.2267753903421312, 0.022879456, 0.3121259623630548),
    ("4I", "3M", -1.3273930485308971, 0.021831511, 0.2034708438753563),
    ("4M", "4L", 1.322931956624692, 0.012492437, 0.1963650175161637),
    ("4M", "4N", 0.5638892044637558, 0.007872439, 0.076181595662469),
]


def _example_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        WEIGHTED_CC_EXAMPLE,
        columns=["labelA", "labelB", "ddG", "bar_std", "slide_std"],
    )


def test_weighted_cycle_closure_matches_upstream_example() -> None:
    result = cycle_closure_from_dataframe(
        _example_dataframe(),
        ddg_col="ddG",
        uncertainty_cols=["bar_std", "slide_std"],
        reference="3A",
        reference_free_energy=-8.83,
    )

    assert len(result.cycles) == 15
    assert result.converged == (True, True, True)

    nodes = result.node_results.set_index("label")
    assert nodes.loc["3K", "dG_cc"] == pytest.approx(-9.6490, abs=5e-5)
    assert nodes.loc["3K", "dG_wcc1"] == pytest.approx(-9.8057, abs=5e-5)
    assert nodes.loc["3K", "dG_wcc2"] == pytest.approx(-9.9030, abs=5e-5)
    assert nodes.loc["3A", "dG_cc"] == pytest.approx(-8.83)
    assert nodes.loc["4P", "path_dependent_error"] == pytest.approx(0.9265, abs=5e-5)
    assert nodes.loc["4N", "path_independent_error"] == pytest.approx(0.5079, abs=5e-5)

    edges = result.edge_results.set_index(["labelA", "labelB"])
    assert edges.loc[("3K", "4M"), "ddG_cc"] == pytest.approx(0.3956, abs=5e-5)
    assert edges.loc[("3K", "4M"), "ddG_wcc1"] == pytest.approx(0.6956, abs=5e-5)
    assert edges.loc[("3K", "4M"), "ddG_wcc2"] == pytest.approx(0.7607, abs=5e-5)
    assert edges.loc[("4M", "4N"), "pair_error"] == pytest.approx(0.5079, abs=5e-5)


def test_cycle_closure_reads_weighted_cc_style_files(tmp_path: Path) -> None:
    input_file = tmp_path / "wcc_input.dat"
    input_file.write_text(
        "\n".join(
            f"{label_a} {label_b} {ddg} {std1} {std2}"
            for label_a, label_b, ddg, std1, std2 in WEIGHTED_CC_EXAMPLE
        )
    )

    parsed = read_cycle_closure_file(input_file)
    assert list(parsed.columns) == ["labelA", "labelB", "ddG", "std1", "std2"]

    result = cycle_closure_from_file(
        input_file,
        reference="3A",
        reference_free_energy=-8.83,
    )

    nodes = result.node_results.set_index("label")
    assert nodes.loc["4L", "dG_cc"] == pytest.approx(-8.2269, abs=5e-5)
    assert nodes.loc["4L", "dG_wcc2"] == pytest.approx(-8.3992, abs=5e-5)


def test_cycle_closure_from_cinnabar_edge_summary_defaults() -> None:
    df = pd.DataFrame(
        [
            {"labelA": "A", "labelB": "B", "calc_DDG": 1.0, "calc_dDDG": 0.2},
            {"labelA": "B", "labelB": "C", "calc_DDG": 1.0, "calc_dDDG": 0.3},
            {"labelA": "C", "labelB": "A", "calc_DDG": -3.0, "calc_dDDG": 0.4},
        ]
    )

    result = cycle_closure_from_dataframe(df, reference="A", reference_free_energy=-5.0)

    assert {"dG_cc", "dG_wcc1"}.issubset(result.node_results.columns)
    assert {"ddG_cc", "ddG_wcc1", "pair_error"}.issubset(result.edge_results.columns)
    assert result.node_results.set_index("label").loc["A", "dG_cc"] == pytest.approx(-5.0)


def test_cycle_closure_requires_a_cycle_by_default() -> None:
    edges = [("A", "B", 1.0), ("B", "C", 2.0)]

    with pytest.raises(ValueError, match="at least one graph cycle"):
        calculate_cycle_closure(edges, reference="A")

    result = calculate_cycle_closure(edges, reference="A", require_cycles=False)
    assert result.cycles == ()
    assert result.node_results.set_index("label").loc["C", "dG_cc"] == pytest.approx(3.0)
