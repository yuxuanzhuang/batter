from pathlib import Path

from batter.systems.core import SimSystem, SystemMeta


def test_sim_system_path_helper(tmp_path):
    system = SimSystem(name="SYS", root=tmp_path)
    sub_path = system.path("executions", "run1")
    assert sub_path == tmp_path / "executions" / "run1"


def test_with_meta_merges_updates():
    system = SimSystem(name="SYS", root=Path("."), meta=SystemMeta(ligand="L1"))
    updated = system.with_meta(residue_name="ABC")

    assert updated.meta.ligand == "L1"
    assert updated.meta.residue_name == "ABC"
    assert system.meta.residue_name is None
