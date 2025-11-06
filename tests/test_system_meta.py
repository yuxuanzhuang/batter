from __future__ import annotations

from pathlib import Path

from batter.systems.core import SimSystem, SystemMeta


def test_system_meta_from_mapping_and_merge() -> None:
    data = {
        "ligand": "LIG1",
        "residue_name": "LIG",
        "mode": "MABFE",
        "param_dir_dict": {"LIG": "/store/LIG"},
        "foo": 1,
    }
    meta = SystemMeta.from_mapping(data)

    assert meta.ligand == "LIG1"
    assert meta.residue_name == "LIG"
    assert meta.param_dir_dict == {"LIG": "/store/LIG"}
    assert meta.get("foo") == 1

    merged = meta.merge(bar=2)
    assert merged is not meta
    assert merged.get("bar") == 2
    assert meta.get("bar") is None


def test_sim_system_converts_meta_dict(tmp_path: Path) -> None:
    sys = SimSystem(
        name="SYS",
        root=tmp_path,
        meta={"ligand": "L1", "residue_name": "LIG", "foo": 10},
    )

    assert isinstance(sys.meta, SystemMeta)
    assert sys.meta.ligand == "L1"
    assert sys.meta.get("foo") == 10

    updated = sys.with_artifacts(meta={"ligand": "L2"})
    assert isinstance(updated.meta, SystemMeta)
    assert updated.meta.ligand == "L2"
