from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from batter._internal.ops import remd


@pytest.mark.parametrize(
    ("remd_nstlim", "total_steps", "expected_interval"),
    [(1000, 10000, 1000), (100, 1000, 100)],
)
def test_patch_component_inputs_rewrites_existing_remd_template(
    tmp_path: Path,
    remd_nstlim: int,
    total_steps: int,
    expected_interval: int,
) -> None:
    comp_dir = tmp_path / "z"
    win_dir = comp_dir / "z00"
    win_dir.mkdir(parents=True)
    (win_dir / "mdin-template").write_text(
        "&cntrl\n"
        "  nstlim = 1000000,\n"
        "  numexchg = 1,\n"
        "  bar_intervall = 6250,\n"
        "  DISANG = disang.rest,\n"
        "/\n"
        " &wt type='DUMPFREQ', istep1=25000, /\n"
    )
    (win_dir / "mdin-remd-template").write_text("stale\n")

    sim = SimpleNamespace(
        remd_nstlim=remd_nstlim, dic_n_steps={"z": total_steps}
    )

    patched = remd.patch_component_inputs(
        comp_dir,
        "z",
        sim,
        add_numexchg=True,
    )

    assert patched == [win_dir / "mdin-remd-template"]
    text = (win_dir / "mdin-remd-template").read_text()
    assert text.startswith(f"! total_steps={total_steps}\n")
    assert "stale" not in text
    assert f"nstlim = {remd_nstlim}," in text
    assert "numexchg = 10," in text
    assert f"bar_intervall = {expected_interval}," in text
    assert "DISANG = z00/disang.rest," in text
    assert f"type='DUMPFREQ', istep1={expected_interval}" in text


def test_remd_groupfiles_always_use_merged_prmtop(tmp_path: Path) -> None:
    comp_dir = tmp_path / "z"
    sim = SimpleNamespace(hmr="yes")

    paths = remd.write_remd_groupfiles(comp_dir, "z", sim, n_windows=1)

    assert paths == [comp_dir / "remd" / "mini.in.remd.groupfile"]
    text = paths[0].read_text()
    assert "-p z-1/full_merged.prmtop" in text
    assert "full.hmr.prmtop" not in text
    assert "full.prmtop" not in text
