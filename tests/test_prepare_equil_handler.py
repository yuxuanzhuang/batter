from __future__ import annotations

from types import SimpleNamespace

import pytest

from batter.exec.handlers import prepare_equil as prepare_equil_mod
from batter.pipeline.step import Step
from batter.systems.core import SimSystem


def test_prepare_equil_exception_names_offending_ligand(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    ligand = "OUTSIDE_POSE"
    system = SimSystem(
        name=f"sys:{ligand}:run1",
        root=tmp_path / "simulations" / ligand,
        meta={
            "ligand": ligand,
            "residue_name": "LIG",
            "param_dir_dict": {"LIG": str(tmp_path / "params")},
        },
    )

    class DummyPayload:
        sim = SimpleNamespace(infe=False)
        sys_params = None

        @staticmethod
        def get(_name, default=None):
            return default

    class FailingBuilder:
        def __init__(self, **_kwargs) -> None:
            pass

        def build(self):
            raise ValueError("ligand pose is outside the binding site")

    monkeypatch.setattr(
        prepare_equil_mod.StepPayload,
        "model_validate",
        staticmethod(lambda _params: DummyPayload()),
    )
    monkeypatch.setattr(prepare_equil_mod, "PrepareEquilBuilder", FailingBuilder)

    with pytest.raises(
        RuntimeError,
        match=r"failed for ligand 'OUTSIDE_POSE'.*outside the binding site",
    ):
        prepare_equil_mod.prepare_equil_handler(
            Step("prepare_equil"),
            system,
            {},
        )

    assert (system.root / "equil" / "prepare_equil.failed").exists()
