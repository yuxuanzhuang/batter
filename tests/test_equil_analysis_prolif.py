from __future__ import annotations

from pathlib import Path

import pandas as pd

from batter.exec.handlers.equil_analysis import (
    PROLIF_ARTIFACT_FILENAMES,
    _persistent_prolif_residue_ids,
    _records_from_prolif_dataframe,
    _run_prolif_fingerprint,
    _write_prolif_lignetwork_html,
    _write_prolif_artifacts,
)


def test_prolif_dataframe_records_persistent_protein_residues() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("LIG300.A", "ASP42.A", "HBAcceptor"),
            ("LIG300.A", "GLY77.A", "Hydrophobic"),
            ("LIG300.A", "LEU88.A", "VdWContact"),
        ]
    )
    df = pd.DataFrame(
        [
            [True, False, True],
            [True, True, True],
            [False, False, True],
        ],
        columns=columns,
    )

    interactions, persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.5,
    )

    assert any(item["interaction"] == "Hydrophobic" for item in interactions)
    assert any(item["interaction"] == "VdWContact" for item in interactions)
    hbond_record = next(
        item for item in interactions if item["interaction"] == "HBAcceptor"
    )
    assert hbond_record["protein"]["resid"] == 42
    assert hbond_record["occupancy"] == 2 / 3
    assert persistent == [
        {
            "resid": 42,
            "resname": "ASP",
            "chainID": "A",
            "max_occupancy": 2 / 3,
            "interactions": [
                {
                    "interaction": "HBAcceptor",
                    "occupancy": 2 / 3,
                    "active_frames": 2,
                    "ligand": {
                        "label": "LIG300.A",
                        "resname": "LIG",
                        "resid": 300,
                        "chainID": "A",
                    },
                }
            ],
        }
    ]
    assert _persistent_prolif_residue_ids(
        {"usable": True, "persistent_protein_residues": persistent}
    ) == [42]


def test_prolif_artifact_writer_saves_timeseries_and_pngs(tmp_path: Path) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("LIG300.A", "ASP42.A", "HBAcceptor"),
            ("LIG300.A", "GLY77.A", "Hydrophobic"),
        ]
    )
    df = pd.DataFrame(
        [
            [True, False],
            [False, True],
            [True, True],
        ],
        index=[10, 11, 12],
        columns=columns,
    )
    interactions, _persistent = _records_from_prolif_dataframe(
        df,
        occupancy_threshold=0.5,
    )

    artifacts, errors = _write_prolif_artifacts(
        prolif_path=tmp_path / "prolif_interactions.json",
        df=df,
        interactions=interactions,
        ligand_label="LIG",
    )

    assert errors == {}
    assert set(artifacts) == set(PROLIF_ARTIFACT_FILENAMES)
    timeseries = pd.read_csv(tmp_path / artifacts["timeseries_csv_gz"])
    assert timeseries["frame"].tolist() == [10, 11, 12]
    assert "LIG300.A|ASP42.A|HBAcceptor" in timeseries.columns
    lignetwork = tmp_path / artifacts["lignetwork_html"]
    assert lignetwork.exists()
    assert "LigNetwork unavailable" in lignetwork.read_text()
    for key in ("barcode_png", "occupancy_png", "interaction_diagram_png"):
        path = tmp_path / artifacts[key]
        assert path.exists()
        assert path.stat().st_size > 0


def test_run_prolif_fingerprint_disables_progress_when_supported() -> None:
    class FakeFingerprint:
        def __init__(self):
            self.progress = None

        def run(self, trajectory, ligand, protein, *, progress=True):
            self.progress = progress

    fp = FakeFingerprint()

    _run_prolif_fingerprint(fp, object(), object(), object())

    assert fp.progress is False


def test_write_prolif_lignetwork_html_uses_prolif_plot_lignetwork(
    tmp_path: Path,
) -> None:
    class FakeMolecule:
        @staticmethod
        def from_mda(selection):
            return {"selection": selection}

    class FakeProlif:
        Molecule = FakeMolecule

    class FakeView:
        def save(self, path):
            path.write_text("<html>native LigNetwork</html>\n")

    class FakeFingerprint:
        def __init__(self):
            self.calls = []

        def plot_lignetwork(self, ligand_mol, **kwargs):
            self.calls.append((ligand_mol, kwargs))
            return FakeView()

    fp = FakeFingerprint()
    path = tmp_path / "network.html"

    _write_prolif_lignetwork_html(
        fingerprint=fp,
        ligand_selection="ligand-selection",
        prolif_module=FakeProlif,
        path=path,
        threshold=0.3,
    )

    assert path.read_text() == "<html>native LigNetwork</html>\n"
    assert fp.calls == [
        (
            {"selection": "ligand-selection"},
            {
                "kind": "aggregate",
                "threshold": 0.3,
                "height": "650px",
                "show_interaction_data": True,
            },
        )
    ]
