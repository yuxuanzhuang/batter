from pathlib import Path

from batter.config.run import RunConfig
from batter import utils as utils_pkg
from batter.utils.process import apply_executable_overrides


def test_executable_overrides_applied(monkeypatch):
    yaml_text = """
run:
  output_folder: work
  executables:
    antechamber: ACUSTOM
    tleap: TLEAP_CUSTOM
    cpptraj: CPTRAJ_CUSTOM
    parmchk2: PARMCHK_CUSTOM
    charmmlipid2amber: CLA_CUSTOM
    usalign: USALIGN_CUSTOM
    obabel: OBABEL_CUSTOM
    vmd: VMD_CUSTOM
create:
  system_name: sys
  protein_input: prot.pdb
  ligand_paths: {LIG: lig.sdf}
fe_sim: {}
"""
    rc = RunConfig.model_validate_yaml(yaml_text)
    apply_executable_overrides(rc.run.executables)

    mod = utils_pkg.process
    assert mod.antechamber == "ACUSTOM"
    assert mod.tleap == "TLEAP_CUSTOM"
    assert mod.cpptraj == "CPTRAJ_CUSTOM"
    assert mod.parmchk2 == "PARMCHK_CUSTOM"
    assert mod.charmmlipid2amber == "CLA_CUSTOM"
    assert mod.usalign == "USALIGN_CUSTOM"
    assert mod.obabel == "OBABEL_CUSTOM"
    assert mod.vmd == "VMD_CUSTOM"
