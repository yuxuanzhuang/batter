from batter.cli.run import _parse_jobname


def test_parse_jobname_remd():
    jobname = "fep_/tmp/foo/executions/run1/simulations/LIG_z_remd"
    meta = _parse_jobname(jobname)
    assert meta is not None
    assert meta["stage"] == "remd"
    assert meta["ligand"] == "LIG"
    assert meta["comp"] == "z"
    assert meta["win"] is None
