from .data import (DD_OPENMM, EXC_CRYSTL,
                  EXC_OPENMM_DOCK, EXC_OPENMM_RANK,
                  MEXC_OPENMM_RANK, SDR_AMBER,
                  SDR_AMBER_MBAR, SDR_CRYSTAL,
                  SDR_OPENMM_RANK, SDR_OPENMM_MBAR,
                  TEX_AMBER_DOCK, SDR_AMBER_MBAR_LIPID)

from batter.input_process import get_configure_from_file, parse_input_file
import pytest
from loguru import logger
import json

@pytest.mark.parametrize('input_file', [DD_OPENMM, EXC_CRYSTL,
                                        EXC_OPENMM_DOCK, EXC_OPENMM_RANK,
                                        MEXC_OPENMM_RANK, SDR_AMBER,
                                        SDR_AMBER_MBAR, SDR_CRYSTAL,
                                        SDR_OPENMM_RANK, SDR_OPENMM_MBAR,
                                        TEX_AMBER_DOCK, SDR_AMBER_MBAR_LIPID])
def test_read_input(snapshot, input_file):
    logger.debug(f'input_file: {input_file}')
    sim_config = get_configure_from_file(input_file)

    # Convert sim_config to a JSON-compatible format for snapshot comparison
    sim_config_json = json.dumps(sim_config, indent=4, default=str)

    # Use snapshot.assert_match to compare the result with the stored snapshot
    snapshot.assert_match(sim_config_json, f"snapshots/{input_file.stem}.json")