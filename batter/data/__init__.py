from importlib import resources

_base_ref = resources.files('batter.data')

charmmlipid2amber = (_base_ref / 'charmmlipid2amber.csv').as_posix()