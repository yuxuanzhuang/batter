from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from batter._internal.ops import remd


def test_patch_component_inputs_rewrites_existing_remd_template(
    tmp_path: Path,
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

    sim = SimpleNamespace(remd_nstlim=100, dic_n_steps={"z": 1000})

    patched = remd.patch_component_inputs(
        comp_dir,
        "z",
        sim,
        add_numexchg=True,
    )

    assert patched == [win_dir / "mdin-remd-template"]
    text = (win_dir / "mdin-remd-template").read_text()
    assert text.startswith("! total_steps=1000\n")
    assert "stale" not in text
    assert "nstlim = 100," in text
    assert "numexchg = 10," in text
    assert "bar_intervall = 100," in text
    assert "DISANG = z00/disang.rest," in text
    assert "type='DUMPFREQ', istep1=100" in text


def test_remd_groupfiles_always_use_merged_prmtop(tmp_path: Path) -> None:
    comp_dir = tmp_path / "z"
    sim = SimpleNamespace(hmr="yes")

    paths = remd.write_remd_groupfiles(comp_dir, "z", sim, n_windows=1)

    assert paths == [comp_dir / "remd" / "mini.in.remd.groupfile"]
    text = paths[0].read_text()
    assert "-p z-1/full_merged.prmtop" in text
    assert "full.hmr.prmtop" not in text
    assert "full.prmtop" not in text
