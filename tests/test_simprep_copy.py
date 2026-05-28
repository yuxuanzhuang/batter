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


def test_read_ligand_anchor_names_allows_single_apo_anchor(tmp_path):
    anchors = tmp_path / "anchors-APO.txt"
    anchors.write_text("DU1\n")

    assert simprep._read_ligand_anchor_names(anchors) == ("DU1", None, None)


def test_write_build_from_aligned_uses_first_atom_in_dum_pdb(tmp_path):
    window_dir = tmp_path / "window"
    build_dir = tmp_path / "q_build_files"
    window_dir.mkdir()
    build_dir.mkdir()
    (build_dir / "dum1.pdb").write_text(
        "ATOM      1  Pb  DUM D   1       1.000   2.000   3.000  0.00  0.00\n"
        "END\n"
    )
    aligned_pdb = tmp_path / "aligned.pdb"
    aligned_pdb.write_text(
        "ATOM      1  CA  ALA A   1       4.000   5.000   6.000  0.00  0.00\n"
        "END\n"
    )

    simprep.write_build_from_aligned(
        lig="LIG",
        window_dir=window_dir,
        build_dir=build_dir,
        aligned_pdb=aligned_pdb,
        other_mol=[],
        lipid_mol=[],
        ion_mol=[],
    )

    first_atom = next(
        line for line in (window_dir / "build.pdb").read_text().splitlines()
        if simprep._is_atom_line(line)
    )
    assert simprep._field(first_atom, 12, 16) == "Pb"
    assert simprep._field(first_atom, 17, 20) == "DUM"
