from types import SimpleNamespace

from batter._internal.ops import simprep


def test_copy_simulation_dir_copies_disang(tmp_path):
    src = tmp_path / "src"
    dest = tmp_path / "dest" / "win"
    src.mkdir(parents=True)
    (src / "disang.rest").write_text("restraints")
    (src / "full.prmtop").write_text("prmtop")
    (src / "cv.in").write_text("cv")

    sim = SimpleNamespace(hmr="no")

    simprep.copy_simulation_dir(src, dest, sim)

    disang = dest / "disang.rest"
    assert disang.exists()
    assert not disang.is_symlink()
    assert disang.read_text() == "restraints"

    prmtop = dest / "full.prmtop"
    assert prmtop.exists()
    # other files may still be symlinked
    if prmtop.is_symlink():
        assert prmtop.resolve() == (src / "full.prmtop").resolve()
    else:
        assert prmtop.read_text() == "prmtop"

    cv = dest / "cv.in"
    assert cv.exists()
