from importlib import resources

_base_ref = resources.files('batter.tests.data')

# Input files
example_input_files = (_base_ref / 'example_input')

DD_OPENMM = (example_input_files / 'input-dd-openmm.in')
EXC_CRYSTL = (example_input_files / 'input-exc-crystal.in')
EXC_OPENMM_DOCK = (example_input_files / 'input-exc-openmm-dock.in')
EXC_OPENMM_RANK = (example_input_files / 'input-exc-openmm-rank.in')
MEXC_OPENMM_RANK = (example_input_files / 'input-mexc-openmm-rank.in')
SDR_AMBER = (example_input_files / 'input-sdr-amber.in')
SDR_AMBER_MBAR = (example_input_files / 'input-sdr-am-mbar.in')
SDR_CRYSTAL = (example_input_files / 'input-sdr-crystal.in')
SDR_OPENMM_RANK = (example_input_files / 'input-sdr-openmm-rank.in')
SDR_OPENMM_MBAR = (example_input_files / 'input-sdr-op-mbar.in')
TEX_AMBER_DOCK = (example_input_files / 'input-tex-amber-dock.in')

# lipid
SDR_AMBER_MBAR_LIPID = (example_input_files / 'input-sdr-am-mbar-lipid.in')

# System files
MOR_MP_INPUT = (_base_ref / '7T2G_mp')