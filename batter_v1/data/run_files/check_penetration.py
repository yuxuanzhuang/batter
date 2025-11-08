import MDAnalysis as mda
from batter_v1.analysis.sim_validation import check_universe_ring_penetration
import sys
import os

if __name__ == '__main__':
    args = sys.argv[1:]
    u = mda.Universe('full.hmr.prmtop', args[0], format='RESTRT')

    if check_universe_ring_penetration(u):
        # write ring penetration detected
        with open('RING_PENETRATION', 'w') as f:
            f.write('Ring penetration detected in the system.\n')
    else:
        # remove file if it exists
        if os.path.exists('RING_PENETRATION'):
            os.remove('RING_PENETRATION')