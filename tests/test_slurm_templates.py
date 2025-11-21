from __future__ import annotations

from pathlib import Path

from batter.utils.slurm_templates import render_slurm_with_header_body


def test_slurm_template_seeds_user_header(monkeypatch, tmp_path):
    """
    When no ~/.batter/<name> exists, render_slurm_with_header_body should
    create it from the packaged header and concatenate with the body.
    """
    # fake home for this test
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    name = "SLURMM-Am.header"
    default_header = tmp_path / "default.header"
    default_body = tmp_path / "default.body"

    default_header.write_text("#!/bin/bash\n#SBATCH --test\nTOKEN\n")
    default_body.write_text("echo BODY TOKEN\n")

    rendered = render_slurm_with_header_body(
        name, default_header, default_body, {"TOKEN": "VALUE"}
    )

    user_header = tmp_path / ".batter" / name
    assert user_header.exists(), "default header should be copied to ~/.batter/"
    assert user_header.read_text() == default_header.read_text()
    assert "#SBATCH --test" in rendered
    assert "echo BODY VALUE" in rendered


def test_slurm_template_uses_user_header(monkeypatch, tmp_path):
    """
    If a user header exists, it should be used verbatim instead of the packaged one.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    name = "custom.header"
    user_header = tmp_path / ".batter" / name
    user_header.parent.mkdir(parents=True, exist_ok=True)
    user_header.write_text("#!/bin/bash\n#SBATCH --user\nUSER_TOKEN\n")

    default_header = tmp_path / "default.header"
    default_header.write_text("#!/bin/bash\n#SBATCH --default\n")
    default_body = tmp_path / "default.body"
    default_body.write_text("BODY USER_TOKEN\n")

    rendered = render_slurm_with_header_body(
        name, default_header, default_body, {"USER_TOKEN": "X"}
    )

    assert "#SBATCH --user" in rendered
    assert "#SBATCH --default" not in rendered
    assert "BODY X" in rendered


def test_slurm_template_honors_header_root(tmp_path):
    """
    header_root should override the default ~/.batter lookup/seed location.
    """
    hdr_root = tmp_path / "custom_headers"
    name = "hdr.header"
    header = tmp_path / "pkg.header"
    body = tmp_path / "pkg.body"
    header.write_text("#!/bin/bash\n#PKG\nTOKEN\n")
    body.write_text("BODY TOKEN\n")

    rendered = render_slurm_with_header_body(
        name,
        header,
        body,
        {"TOKEN": "Z"},
        header_root=hdr_root,
    )

    assert (hdr_root / name).exists(), "header should be seeded under header_root"
    assert "#PKG" in rendered
    assert "BODY Z" in rendered


def test_slurm_template_reads_existing_header_root(tmp_path):
    """
    If a header exists under header_root, it should be used and not overwritten.
    """
    hdr_root = tmp_path / "hdrs"
    name = "hdr.header"
    user_hdr = hdr_root / name
    user_hdr.parent.mkdir(parents=True, exist_ok=True)
    user_hdr.write_text("#!/bin/bash\n#USER\n")

    pkg_header = tmp_path / "pkg.header"
    pkg_body = tmp_path / "pkg.body"
    pkg_header.write_text("#!/bin/bash\n#PKG\n")
    pkg_body.write_text("BODY\n")

    rendered = render_slurm_with_header_body(
        name, pkg_header, pkg_body, {}, header_root=hdr_root
    )

    assert "#USER" in rendered
    assert "#PKG" not in rendered
