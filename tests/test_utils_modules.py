import pytest

from batter.utils import (
    FEP_COMPONENTS,
    components_under,
    natural_keys,
    run_with_log,
)


def test_natural_keys_sorts_human_readable():
    items = ["win2", "win10", "win1", "win12", "win3"]
    assert sorted(items, key=natural_keys) == ["win1", "win2", "win3", "win10", "win12"]


def test_components_under_filters_known_components(tmp_path):
    root = tmp_path / "system"
    fe_root = root / "fe"
    fe_root.mkdir(parents=True)

    valid = FEP_COMPONENTS[:3]
    for comp in valid:
        (fe_root / comp).mkdir()
    # include directories that should be ignored
    (fe_root / "not_a_comp").mkdir()
    (fe_root / "xextra").mkdir()

    result = components_under(root)
    assert result == sorted(valid)


def test_run_with_log_success(tmp_path):
    script = "import pathlib; pathlib.Path('flag.txt').write_text('ok')"
    run_with_log(
        ["python3", "-c", script],
        shell=False,
        working_dir=str(tmp_path),
        level="info",
    )
    assert (tmp_path / "flag.txt").read_text() == "ok"


def test_run_with_log_failure_raises(tmp_path):
    with pytest.raises(RuntimeError):
        run_with_log(
            ["python3", "-c", "import sys; sys.exit(2)"],
            shell=False,
            working_dir=str(tmp_path),
        )


def test_process_exec_env_overrides(monkeypatch):
    import importlib
    from batter.utils import process

    overrides = {
        "BATTER_TLEAP": "/opt/tleap",
        "BATTER_CPPTRAJ": "/opt/cpptraj",
        "BATTER_USALIGN": "/custom/USalign",
    }
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)

    reloaded = importlib.reload(process)
    assert reloaded.tleap == "/opt/tleap"
    assert reloaded.cpptraj == "/opt/cpptraj"
    assert reloaded.usalign == "/custom/USalign"

    # cleanup: remove env and restore defaults
    for k in overrides:
        monkeypatch.delenv(k, raising=False)
    restored = importlib.reload(process)
    assert restored.tleap != "/opt/tleap"
