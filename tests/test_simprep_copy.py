import math
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


def test_write_build_from_aligned_ter_marker_applies_once_per_residue(tmp_path):
    window_dir = tmp_path / "window"
    build_dir = tmp_path / "q_build_files"
    window_dir.mkdir()
    build_dir.mkdir()
    (build_dir / "dum1.pdb").write_text(
        "ATOM      1  Pb  DUM D   1       0.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    aligned_pdb = tmp_path / "aligned.pdb"
    aligned_pdb.write_text(
        "ATOM      1  N   ALA A   1       1.000   0.000   0.000  0.00  0.00\n"
        "ATOM      2  CA  ALA A   1       2.000   0.000   0.000  0.00  0.00\n"
        "ATOM      3  N   GLU A 214       3.000   0.000   0.000  0.00  0.00\n"
        "ATOM      4  CA  GLU A 214       4.000   0.000   0.000  0.00  0.00\n"
        "ATOM      5  C   GLU A 214       5.000   0.000   0.000  0.00  0.00\n"
        "ATOM      6  O   GLU A 214       6.000   0.000   0.000  0.00  0.00\n"
        "ATOM      7  N   SER A 215       7.000   0.000   0.000  0.00  0.00\n"
        "ATOM      8  CA  SER A 215       8.000   0.000   0.000  0.00  0.00\n"
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
        use_ter_markers=True,
        ter_residues={("A", 214)},
    )

    lines = (window_dir / "build.pdb").read_text().splitlines()
    glu_indices = [
        idx
        for idx, line in enumerate(lines)
        if simprep._is_atom_line(line)
        and simprep._field(line, 17, 20) == "GLU"
        and int(simprep._field(line, 22, 26)) == 215
    ]
    assert len(glu_indices) == 4
    assert all(line != "TER" for line in lines[glu_indices[0] : glu_indices[-1]])
    assert lines[glu_indices[-1] + 1] == "TER"


def test_write_build_from_aligned_can_use_template_for_shifted_ligand(tmp_path):
    window_dir = tmp_path / "window"
    build_dir = tmp_path / "q_build_files"
    window_dir.mkdir()
    build_dir.mkdir()
    (build_dir / "dum1.pdb").write_text(
        "ATOM      1  Pb  DUM D   1       0.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    (build_dir / "dum2.pdb").write_text(
        "ATOM      1  Pb  DUM D   2       1.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    aligned_pdb = tmp_path / "aligned.pdb"
    aligned_pdb.write_text(
        "ATOM      1  CA  ALA A   1       4.000   5.000   0.000  0.00  0.00\n"
        "ATOM      2  C1  LIG X   2       0.000   0.000   0.000  0.00  0.00\n"
        "ATOM      3  C2  LIG X   2       2.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    template_pdb = tmp_path / "initial_ligand.pdb"
    template_pdb.write_text(
        "ATOM      1  C1  lig L   1      10.000  10.000   0.000  0.00  0.00\n"
        "ATOM      2  C2  lig L   1      10.000  13.000   0.000  0.00  0.00\n"
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
        extra_ligand_shift=[True],
        sdr_dist=10.0,
        extra_ligand_source_pdb=template_pdb,
    )

    ligand_coords_by_resid = {}
    for line in (window_dir / "build.pdb").read_text().splitlines():
        if not simprep._is_atom_line(line) or simprep._field(line, 17, 20) != "LIG":
            continue
        resid = int(simprep._field(line, 22, 26))
        ligand_coords_by_resid.setdefault(resid, []).append(
            (
                float(simprep._field(line, 30, 38)),
                float(simprep._field(line, 38, 46)),
                float(simprep._field(line, 46, 54)),
            )
        )

    assert len(ligand_coords_by_resid) == 2
    groups = list(ligand_coords_by_resid.values())
    groups_by_distance = {
        round(math.dist(coords[0], coords[1]), 6): coords for coords in groups
    }
    site_coords = groups_by_distance[2.0]
    shifted_coords = groups_by_distance[3.0]

    shifted_distance = math.dist(shifted_coords[0], shifted_coords[1])
    assert shifted_distance == 3.0

    shifted_center = tuple(
        sum(coord[axis] for coord in shifted_coords) / len(shifted_coords)
        for axis in range(3)
    )
    site_center = tuple(
        sum(coord[axis] for coord in site_coords) / len(site_coords)
        for axis in range(3)
    )
    assert shifted_center == (4.0, 5.0, site_center[2] + 10.0)


def test_write_build_from_aligned_can_duplicate_two_template_solvent_copies(tmp_path):
    window_dir = tmp_path / "window"
    build_dir = tmp_path / "q_build_files"
    window_dir.mkdir()
    build_dir.mkdir()
    (build_dir / "dum1.pdb").write_text(
        "ATOM      1  Pb  DUM D   1       0.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    aligned_pdb = tmp_path / "aligned.pdb"
    aligned_pdb.write_text(
        "ATOM      1  CA  ALA A   1       4.000   5.000   0.000  0.00  0.00\n"
        "ATOM      2  C1  LIG X   2       0.000   0.000   0.000  0.00  0.00\n"
        "ATOM      3  C2  LIG X   2       2.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    template_pdb = tmp_path / "initial_ligand.pdb"
    template_pdb.write_text(
        "ATOM      1  C1  LIG L   1      10.000  10.000   0.000  0.00  0.00\n"
        "ATOM      2  C2  LIG L   1      10.000  13.000   0.000  0.00  0.00\n"
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
        extra_ligand_shift=[True, True],
        sdr_dist=10.0,
        extra_ligand_source_pdbs=[template_pdb, template_pdb],
        extra_ligand_duplicate_coordinates=True,
    )

    ligand_groups = []
    seen_resids = []
    for line in (window_dir / "build.pdb").read_text().splitlines():
        if not simprep._is_atom_line(line) or simprep._field(line, 17, 20) != "LIG":
            continue
        resid = int(simprep._field(line, 22, 26))
        if not seen_resids or resid != seen_resids[-1]:
            seen_resids.append(resid)
            ligand_groups.append([])
        ligand_groups[-1].append(
            (
                float(simprep._field(line, 30, 38)),
                float(simprep._field(line, 38, 46)),
                float(simprep._field(line, 46, 54)),
            )
        )

    assert len(ligand_groups) == 3
    bound, solvent_charge, solvent_neutral = ligand_groups
    assert math.dist(bound[0], bound[1]) == 2.0
    assert math.dist(solvent_charge[0], solvent_charge[1]) == 3.0
    assert solvent_charge == solvent_neutral


def test_write_build_from_aligned_can_copy_existing_solvent_ligand(tmp_path):
    window_dir = tmp_path / "window"
    build_dir = tmp_path / "q_build_files"
    window_dir.mkdir()
    build_dir.mkdir()
    (build_dir / "dum1.pdb").write_text(
        "ATOM      1  Pb  DUM D   1       0.000   0.000   0.000  0.00  0.00\n"
        "END\n"
    )
    aligned_pdb = tmp_path / "aligned.pdb"
    aligned_pdb.write_text(
        "ATOM      1  CA  ALA A   1       4.000   5.000   0.000  0.00  0.00\n"
        "ATOM      2  C1  LIG X   2       0.000   0.000   0.000  0.00  0.00\n"
        "ATOM      3  C2  LIG X   2       2.000   0.000   0.000  0.00  0.00\n"
        "ATOM      4  C1  LIG X   3      20.000  20.000   5.000  0.00  0.00\n"
        "ATOM      5  C2  LIG X   3      24.000  20.000   5.000  0.00  0.00\n"
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
        extra_ligand_shift=[False],
        extra_ligand_offsets=[(0.0, 0.0, 0.0)],
        extra_ligand_target_indices=[1],
    )

    ligand_groups = []
    seen_resids = []
    for line in (window_dir / "build.pdb").read_text().splitlines():
        if not simprep._is_atom_line(line) or simprep._field(line, 17, 20) != "LIG":
            continue
        resid = int(simprep._field(line, 22, 26))
        if not seen_resids or resid != seen_resids[-1]:
            seen_resids.append(resid)
            ligand_groups.append([])
        ligand_groups[-1].append(
            (
                float(simprep._field(line, 30, 38)),
                float(simprep._field(line, 38, 46)),
                float(simprep._field(line, 46, 54)),
            )
        )

    assert len(ligand_groups) == 3
    bound, solvent_charge, solvent_neutral = ligand_groups
    assert math.dist(bound[0], bound[1]) == 2.0
    assert math.dist(solvent_charge[0], solvent_charge[1]) == 4.0
    assert solvent_charge == solvent_neutral
