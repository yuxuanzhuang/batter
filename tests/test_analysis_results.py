from batter.analysis.results import FEResult


def test_feresult_parses_component_lines(tmp_path):
    contents = "\n".join(
        [
            "m attach -1.00, 0.10",
            "e elec -2.50, 0.25",
            "Boresch restraint 0.50, 0.05",
            "Total free_energy -3.75, 0.30",
        ]
    )
    result_file = tmp_path / "results.dat"
    result_file.write_text(contents)

    fe = FEResult(result_file)

    assert fe.attach == -1.0
    assert fe.attach_std == 0.10
    assert fe.elec == -2.5
    assert fe.boresch == 0.5
    assert fe.fe == -3.75
    summary = fe.to_dict()
    assert summary["fe"]["std"] == 0.30


def test_feresult_marks_unbound(tmp_path):
    result_file = tmp_path / "results.dat"
    result_file.write_text("UNBOUND\n")

    fe = FEResult(result_file)

    assert fe.is_unbound
    assert fe.to_dict()["fe"]["value"] == "unbound"
