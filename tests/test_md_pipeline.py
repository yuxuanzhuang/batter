from batter.config import load_run_config
from batter.orchestrate.pipeline_utils import select_pipeline
from batter.pipeline.factory import make_md_pipeline
from tests.data import MABFE_YAML


def _sim_cfg():
    cfg = load_run_config(MABFE_YAML)
    return cfg.resolved_sim_config()


def test_md_pipeline_stops_after_equil_analysis():
    sim_cfg = _sim_cfg()
    pipeline = make_md_pipeline(sim_cfg, sys_params={})
    names = [step.name for step in pipeline.ordered_steps()]
    assert names == [
        "system_prep",
        "param_ligands",
        "prepare_equil",
        "equil",
        "equil_analysis",
    ]


def test_select_pipeline_md_branch():
    sim_cfg = _sim_cfg()
    pipeline = select_pipeline("md", sim_cfg, only_fe_prep=False, sys_params={})
    names = [step.name for step in pipeline.ordered_steps()]
    assert names[-1] == "equil_analysis"
    assert "prepare_fe" not in names
