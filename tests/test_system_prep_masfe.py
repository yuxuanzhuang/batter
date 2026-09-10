from __future__ import annotations

from pathlib import Path

from batter.exec.handlers import system_prep_masfe as masfe_mod
from batter.exec.handlers.system_prep_masfe import _MASFESystemPrepRunner
from batter.systemprep.artifacts import ligand_pdb_path
from batter.systems.core import SimSystem


def test_masfe_conversion_files_cannot_overwrite_staged_ligands(
    monkeypatch, tmp_path: Path
) -> None:
    source_foo = tmp_path / "FOO.sdf"
    source_bar = tmp_path / "BAR.sdf"
    source_foo.write_text("foo molecule\n")
    source_bar.write_text("bar molecule\n")

    def _fake_ensure_pdb(source: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        converted = output_dir / f"{source.stem}.pdb"
        converted.write_text(source.read_text())
        return converted

    monkeypatch.setattr(masfe_mod, "_ensure_pdb", _fake_ensure_pdb)
    system = SimSystem(name="SOLV", root=tmp_path / "run")
    runner = _MASFESystemPrepRunner(system)

    manifest = runner.run(
        system_name="SOLV",
        ligand_paths={"BAR": str(source_foo), "FOO": str(source_bar)},
    )

    staged_bar = ligand_pdb_path(system.root, "BAR")
    staged_foo = ligand_pdb_path(system.root, "FOO")
    assert staged_bar.read_text() == "foo molecule\n"
    assert staged_foo.read_text() == "bar molecule\n"
    assert manifest["ligands"] == {
        "BAR": "ligands/BAR.pdb",
        "FOO": "ligands/FOO.pdb",
    }
