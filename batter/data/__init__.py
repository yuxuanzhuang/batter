from importlib import resources

_base_ref = resources.files('batter.data')

charmmlipid2amber = (_base_ref / 'charmmlipid2amber.csv').as_posix()
# Use the header as the anchor path for manager templates; body is resolved alongside it.
job_manager = (_base_ref / 'job_manager.header').as_posix()
