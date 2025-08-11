from importlib import resources

_base_ref = resources.files('batter.data')

build_files = (_base_ref / 'build_files').as_posix()
amber_files = (_base_ref / 'amber_files').as_posix()
run_files = (_base_ref / 'run_files').as_posix()
openmm_files = (_base_ref / 'openmm_files').as_posix()
batch_files = (_base_ref / 'batch_files').as_posix()

charmmlipid2amber = (_base_ref / 'charmmlipid2amber.csv').as_posix()