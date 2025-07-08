import MDAnalysis as mda
from batter.analysis.sim_validation import check_universe_ring_penetration

if __name__ == '__main__':
    u = mda.Universe('full.prmtop', 'eqnvt.rst7', format='RESTRT')

    if check_universe_ring_penetration(u):
        raise RuntimeError('Error: Ring penetration detected in the system.')