"""
This scripts is used to run a simple simulation validation on
- the simulation box
- the rmsd of the protein
- the rmsd of the ligand
- the rmsf of the protein

Maybe in the future: the membrane properties
"""
import numpy as np
from pathlib import Path

import scipy.stats
import MDAnalysis as mda
from loguru import logger
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.core.groups import AtomGroup
import os

import networkx as nx
import re
import numpy as np
import itertools

#import lipyphilic as lpp

from MDAnalysis.analysis.results import Results

class SimValidator:
    """
    A class to validate a simulation

    Attributes
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis universe object
    ligand : str
        The resname of the ligand
    results : MDAnalysis.analysis.results.Results
        The results of the validation
    
    Methods
    -------
    plot_box()
        Plot the box size
    plot_rmsd()
        Plot the RMSD of the protein and the ligand
    plot_rmsf()
        Plot the RMSF of the protein
    """
    def __init__(self, universe, ligand=None, directory: str | Path = "."):
        """
        Parameters
        ----------
        universe : MDAnalysis.Universe
            The MDAnalysis universe object
        ligand : str, optional
            The resname of the ligand.
            If not provided, it will be guessed.
        """
        self.universe = universe
        self.workdir = Path(directory).resolve()
        if ligand is not None:
            self.ligand = ligand
        else:
            self._guess_ligand()
        self.results = Results()
        self._validate()

    def _guess_ligand(self):
        ligand_ag = self.universe.select_atoms('not protein and not resname HOH TIP3 WAT DUM OL PA PC')
        possible_resnames = np.unique(ligand_ag.resnames)
        # ligand resname should be lower case
        possible_resnames = [resname for resname in possible_resnames if resname.islower()]
        if len(possible_resnames) == 1:
            self.ligand = possible_resnames[0]
            logger.debug(f'Guessed ligand resname: {self.ligand}')
        elif len(possible_resnames) == 0:
            self.ligand = 'XXX'
            logger.warning('No ligand is found. If you know the ligand resname, '
                           'set it by `ligand` argument')
        else:
            raise ValueError('Could not guess ligand resname. It may be '
                             f'one of {possible_resnames}, set it by `ligand` argument')
    
    def _validate(self):
        self._box()
        self._rmsd()
        # self._rmsf()
        # self._membrane()
        self._ligand_bs()
        
    
    def _box(self):
        logger.debug('Calculating box size')
        results = []
        for ts in self.universe.trajectory:
            box = ts.dimensions[:3]
            results.append(box.copy())
        self.results['box'] = results

    def _rmsd(self):
        logger.debug('Calculating RMSD')
        from MDAnalysis.analysis.rms import RMSD
        rms = RMSD(self.universe,
                   self.universe,
                   groupselections=[f'resname {self.ligand}'],
                   select='name CA').run()
        self.results['protein_rmsd'] = rms.results.rmsd.T[2]
        self.results['ligand_rmsd'] = rms.results.rmsd.T[3]
    
    def _rmsf(self):
        logger.debug('Calculating RMSF')
        from MDAnalysis.analysis import rms, align
        u = self.universe
        average = align.AverageStructure(
                u,
                u,
                select='protein and name CA',
                ref_frame=0).run()

        ref = average.results.universe

        aligner = align.AlignTraj(u, ref,
                          select='protein and name CA',
                          in_memory=True).run()

        c_alphas = u.select_atoms('protein and name CA')
        R = rms.RMSF(c_alphas).run()

        self.results['ligand_rmsf'] = R.results.rmsf

    def _ligand_bs(self):
        logger.debug('Calculating ligand binding site')
        # Get the ligand atom group
        ligand_ag = self.universe.select_atoms(f'resname {self.ligand}')
        # Get the protein atom group
        bs_ag = self.universe.select_atoms(f"protein and byres around 5 resname {self.ligand}")
        
        # Calculate the distance between the ligand and the protein
        distances = []
        for ts in self.universe.trajectory:
            dist = distance_array(
                ligand_ag.center_of_mass(),
                bs_ag.center_of_mass(),
                box=self.universe.dimensions)[0]
            distances.append(dist)
        
        distances = np.array(distances)
        self.results['ligand_bs'] = distances
    
    def _ligand_dihedral(self):
        logger.debug('Calculating ligand dihedral')
        dihed_ligands_file = self.workdir / 'assign.in'
        if not os.path.exists(dihed_ligands_file):
            raise FileNotFoundError(f'{dihed_ligands_file} not found')
        
        
        with open(dihed_ligands_file, 'r') as f:
            lines = f.readlines()
            dihed_lines = [lines[i] for i in range(len(lines)) if lines[i].startswith('dihedral')]

        # The first few are for protein dihedrals
        dihed_lines = dihed_lines[3:]
        def selection_string(amber_sel):
            resid = amber_sel.split('@')[0].split(':')[1]
            resname = amber_sel.split('@')[1]
            return f'resid {resid} and name {resname}'

        ag_lists = []
        for line in dihed_lines:
            try:
                atoms_str = line.split()[2:6]
                atoms_str = [selection_string(a) for a in atoms_str]
                ag_group = AtomGroup([
                    self.universe.select_atoms(a).atoms[0] for a in atoms_str
                ])
                ag_lists.append(ag_group)
            except Exception as e:
                # an issue with Cl and CL naming
                pass
        
        diheds = []
        for ts in self.universe.trajectory:
            dihed = []
            for ag in ag_lists:
                dihed.append(ag.dihedral.value())
            diheds.append(dihed)
        diheds = np.array(diheds)

        self.results['ligand_dihedrals'] = diheds

    def _membrane(self):
        raise NotImplementedError('Membrane properties are not implemented yet')
        logger.debug('Calculating membrane properties')
        # Find which leaflet each lipid is in at each frame
        leaflets = lpp.AssignLeaflets(
            universe=self.universe,
            lipid_sel="resname OL PA PC"
            )

        leaflets.run()
        
        logger.debug('Calculating leaflet areas')
        areas = lpp.analysis.AreaPerLipid(
            universe=self.universe,
            lipid_sel="resname OL PA PC",
            leaflets=leaflets.leaflets
            )

        areas.run()
        self.results['leaflet_areas'] = areas.areas

    def plot_analysis(self, savefig=True):
        # plot ligand_bs, rmsd, dihedral in three rows
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        # Plot ligand binding site distance
        axes[0].plot(self.results['ligand_bs'], label='Ligand to binding site distance')
        axes[0].set_ylabel('Distance (Å)')
        axes[0].legend()
        # Plot RMSD
        axes[1].plot(self.results['protein_rmsd'], label='Protein RMSD')
        axes[1].plot(self.results['ligand_rmsd'], label='Ligand RMSD')
        axes[1].set_ylabel('RMSD (Å)')
        axes[1].legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(self.workdir / 'simulation_analysis.png')
        else:
            plt.show()
        plt.close(fig)

    def plot_box(self, savefig=True):
        logger.debug('Plotting box size')
        box_results = np.array(self.results['box'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(box_results[:, 0], label='x')
        ax.plot(box_results[:, 1], label='y')
        ax.plot(box_results[:, 2], label='z')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Box size (Å)')
        ax.legend()
        if savefig:
            plt.savefig(self.workdir / 'box_size.png')
        else:
            plt.show()
        plt.close(fig)
    
    def plot_ligand_bs(self, savefig=True):
        logger.debug('Plotting RMSD')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['ligand_bs'], label='Ligand to binding site distance')
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSD (Å)')
        ax.legend()
        if savefig:
            plt.savefig(self.workdir / 'ligand_bs.png')
        else:
            plt.show()
        plt.close(fig)

    def plot_rmsd(self, savefig=True):
        logger.debug('Plotting RMSD')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['protein_rmsd'], label='Protein')
        ax.plot(self.results['ligand_rmsd'], label='Ligand')
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSD (Å)')
        ax.legend()
        if savefig:
            plt.savefig(self.workdir / 'rmsd.png')
        else:
            plt.show()
        plt.close(fig)

    def plot_rmsf(self, savefig=True):
        logger.debug('Plotting RMSF')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['ligand_rmsf'], label='Ligand')
        ax.set_xlabel('Residue')
        ax.set_ylabel('RMSF (Å)')
        if savefig:
            plt.savefig(self.workdir / 'rmsf.png')
        else:
            plt.show()
        plt.close(fig)
    
    def plot_leaflet_areas(self):
        raise NotImplementedError('Membrane properties are not implemented yet')
        logger.debug('Plotting leaflet areas')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['leaflet_areas'])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Area per lipid (nm^2)')
        plt.show()
        plt.close(fig)
    
    # get the mode value of the dihedral
    def find_representative_snapshot(self, savefig=True):
        """
        Find the representative snapshot based on the mode dihedral values.
        """
        # convert to sin and cos values
        self._ligand_dihedral()
        dihed = self.results['ligand_dihedrals']
        dihed_rad = np.deg2rad(dihed)
        sin_dihed = np.sin(dihed_rad)
        cos_dihed = np.cos(dihed_rad)

        n_dihed = dihed.shape[1]

        # Calculate the mode dihedral values
        feat_dihed = np.concatenate([sin_dihed, cos_dihed], axis=1)

        mode_dihed = scipy.stats.mode(feat_dihed, axis=0, keepdims=True).mode
        
        # Calculate the absolute difference between each snapshot's dihedral and the mode
        abs_diff = np.abs(feat_dihed - mode_dihed)
        
        # Find the index of the snapshot with the smallest absolute difference
        representative_index = np.argmin(np.sum(abs_diff, axis=1))
        
        # plot 
        fig, ax = plt.subplots(1, n_dihed, figsize=(20, 5), sharex=True, sharey=True,
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        for i in range(n_dihed):
            ax[i].hist(dihed[:, i], bins=100, density=True, alpha=0.5, range=(-180, 180))
            ax[i].set_title(f"{i}")
            ax[i].vlines(dihed[representative_index, i], ymin=0, ymax=0.05,
                        color='r', linestyle='--', label='Representative')
        plt.tight_layout()
        if savefig:
            plt.savefig(self.workdir / 'dihed_hist.png')
        else:
            plt.show()
        plt.close(fig)

        return representative_index


class MultiligandSimValidator:
    """
    A class to validate a simulation

    Attributes
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis universe object
    ligand : str
        The resname(s) of the ligand
    results : MDAnalysis.analysis.results.Results
        The results of the validation
    
    Methods
    -------
    plot_box()
        Plot the box size
    plot_rmsd()
        Plot the RMSD of the protein and the ligand
    plot_rmsf()
        Plot the RMSF of the protein
    """
    def __init__(self, universe, ligand=None):
        """
        Parameters
        ----------
        universe : MDAnalysis.Universe
            The MDAnalysis universe object
        ligand : str, optional
            The resname of the ligand.
            If not provided, it will be guessed.
        """
        self.universe = universe
        if ligand is not None:
            ligand = ligand.split()
            ligand_ag = self.universe.select_atoms(f'resname {" ".join(ligand)}')
            if len(ligand_ag) == 0:
                logger.warning(f'No atoms are found with the provided ligand resname {ligand}')
                logger.warning('Guessing ligand resname')
                self._guess_ligand()
        else:
            ligand_ag = self._guess_ligand()

        self.ligands = [ag for ag in ligand_ag.residues]
        logger.debug(f'Found {len(self.ligands)} ligands')
        logger.debug(f'self.ligands: {self.ligands}')
        self.results = Results()
        self._validate()

    def _guess_ligand(self):
        ligand_ag = self.universe.select_atoms('not protein and not resname HOH TIP3 WAT DUM OL PA PC')
        possible_resnames = np.unique(ligand_ag.resnames)
        # ligand resname should be lower case
        possible_resnames = [resname for resname in possible_resnames if resname.islower()]
        if len(possible_resnames) == 1:
            ligand_name = possible_resnames[0]
            logger.debug(f'Guessed ligand resname: {ligand_name}')
            ligand_ag = [ag for ag in ligand_ag.residues]
        elif len(possible_resnames) == 0:
            ligand_name = 'XXX'
            ligand_ag = mda.AtomGroup()
            logger.warning('No ligand is found. If you know the ligand resname, '
                           'set it by `ligand` argument')
        else:
            raise ValueError('Could not guess ligand resname. It may be '
                             f'one of {possible_resnames}, set it by `ligand` argument')
        return ligand_ag
                    
    
    def _validate(self):
        self._box()
        self._rmsd()
        self._rmsf()
        # self._membrane()
    
    def _box(self):
        logger.debug('Calculating box size')
        results = []
        for ts in self.universe.trajectory:
            box = ts.dimensions[:3]
            results.append(box.copy())
        self.results['box'] = results

    def _rmsd(self):
        logger.debug('Calculating RMSD')
        from MDAnalysis.analysis.rms import RMSD
        ligand_indices = [ag.indices for ag in self.ligands]
        rms = RMSD(self.universe,
                   self.universe,
                   groupselections=[f'index {" ".join(map(str, indices))}' for indices in ligand_indices],
                   select='name CA').run()
        self.results['protein_rmsd'] = rms.results.rmsd.T[2]
        for i, ligand in enumerate(self.ligands):
            self.results[f'ligand_{i}_rmsd'] = rms.results.rmsd.T[3 + i]
    
    def _rmsf(self):
        logger.debug('Calculating RMSF')
        from MDAnalysis.analysis import rms, align
        u = self.universe
        average = align.AverageStructure(
                u,
                u,
                select='protein and name CA',
                ref_frame=0).run()

        ref = average.results.universe

        aligner = align.AlignTraj(u, ref,
                          select='protein and name CA',
                          in_memory=True).run()

        c_alphas = u.select_atoms('protein and name CA')
        R = rms.RMSF(c_alphas).run()

        self.results['ligand_rmsf'] = R.results.rmsf

    def _membrane(self):
        raise NotImplementedError('Membrane properties are not implemented yet')
        logger.debug('Calculating membrane properties')
        # Find which leaflet each lipid is in at each frame
        leaflets = lpp.AssignLeaflets(
            universe=self.universe,
            lipid_sel="resname OL PA PC"
            )

        leaflets.run()
        
        logger.debug('Calculating leaflet areas')
        areas = lpp.analysis.AreaPerLipid(
            universe=self.universe,
            lipid_sel="resname OL PA PC",
            leaflets=leaflets.leaflets
            )

        areas.run()
        self.results['leaflet_areas'] = areas.areas

    def plot_box(self):
        logger.debug('Plotting box size')
        box_results = np.array(self.results['box'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(box_results[:, 0], label='x')
        ax.plot(box_results[:, 1], label='y')
        ax.plot(box_results[:, 2], label='z')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Box size (Å)')
        ax.legend()
        plt.show()
    
    def plot_rmsd(self):
        logger.debug('Plotting RMSD')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['protein_rmsd'], label='Protein')
        for i, ligand in enumerate(self.ligands):
            ax.plot(self.results[f'ligand_{i}_rmsd'], label=f'Ligand {i}')
        ax.plot(self.results['ligand_rmsd'], label='Ligand')
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSD (Å)')
        ax.legend()
        plt.show()

    def plot_rmsf(self):
        logger.debug('Plotting RMSF')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['ligand_rmsf'], label='Ligand')
        ax.set_xlabel('Residue')
        ax.set_ylabel('RMSF (Å)')
        plt.show()
    
    def plot_leaflet_areas(self):
        raise NotImplementedError('Membrane properties are not implemented yet')
        logger.debug('Plotting leaflet areas')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['leaflet_areas'])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Area per lipid (nm^2)')
        plt.show()




# RING PENETRATION
# Modified from penetest.py file in CHARMM-GUI

def lsqp(atoms):
    com = atoms.mean(axis=0)
    #u, d, v = np.linalg.svd(atoms-com)

    axes = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        p1 = atoms[i]
        if i == len(atoms)-1:
            p2 = atoms[0]
        else:
            p2 = atoms[i+1]
        a = np.cross(p1, p2)
        axes += a
    u, d, v = np.linalg.svd(axes)
    i = 0
    d = -np.dot(v[i], com)
    n = -np.array((v[i,0], v[i,1], d))/v[i,2]
    return v[i], com, n


def intriangle(triangle, axis, u, p):
    # http://www.softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm
    p1, p2, p3 = triangle
    w0 = p - p1
    a = -np.dot(axis, w0)
    b = np.dot(axis, u)
    if (abs(b) < 0.01): return False

    r = a / b
    if r < 0.0: return False
    if r > 1.0: return False

    I = p + u * r

    u = p2 - p1
    v = p3 - p1
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    w = I - p1
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    D = uv * uv - uu * vv

    s = (uv * wv - vv * wu)/D
    if (s < 0 or s > 1): return False
    t = (uv * wu - uu * wv)/D
    if (t < 0 or (s+t) > 1): return False
    return True


def build_topology(universe, selection):
    g = nx.Graph()

    #  Atoms
    natom = universe.atoms.n_atoms
    for atom in universe.select_atoms(selection).atoms:  #  might be buggy
        g.add_node(atom.index + 1, **{'segid': atom.segid,
                                      'resname': atom.resname,
                                      'name': atom.name,
                                      'resid': atom.resid})
    #  Bonds
    for bond in universe.select_atoms(selection).bonds:
        num1, num2 = bond.atoms.indices + 1
        if g.has_node(num1) and g.has_node(num2):
            g.add_edge(num1, num2)
    return g


def check_ring_penetration(top, coord, pbc=[], xtl='rect', verbose=0):
    # ring penetration test
    # 1. find rings
    # 2. build least square plane
    # 3. project atoms ring constituent atoms onto the plane and build convex
    # 4. find two bonded atoms that are at the opposite side of the plane
    # 5. determine the point of intersection is enclosed in the ring
    #
    from networkx.algorithms.components import connected_components
    molecules =  (top.subgraph(c) for c in connected_components(top))

    allatoms = np.array([coord[num] for num in top.nodes()])
    atoms_map = np.array([num for num in top.nodes()])
    natoms = len(allatoms)
    if pbc:
        atoms_map_reverse = {}
        for i,num in enumerate(top.nodes()):
            atoms_map_reverse[num] = i

        a = float(pbc[0])
        b = float(pbc[1])
        n = len(allatoms)
        if xtl == 'rect':
            allatoms = np.tile(allatoms, (9,1))
            op = ((a,0),(a,b),(0,b),(-a,b),(-a,0),(-a,-b),(0,-b),(a,-b))
            for i in range(8):
                x,y = op[i]
                allatoms[n*(i+1):n*(i+2),0] += x
                allatoms[n*(i+1):n*(i+2),1] += y
            atoms_map = np.tile(atoms_map, 9)
        if xtl =='hexa':
            allatoms = np.tile(allatoms, (7,1))
            rot = lambda theta: np.matrix(((np.cos(np.radians(theta)), -np.sin(np.radians(theta))),
                                           (np.sin(np.radians(theta)),  np.cos(np.radians(theta)))))
            op = (rot(15), rot(75), rot(135), rot(195), rot(255), rot(315))
            d = np.array((a, 0))
            for i in range(6):
                xy = np.dot(d, op[i])
                allatoms[n*(i+1):n*(i+2),:2] = allatoms[n*(i+1):n*(i+2),:2] + xy
            atoms_map = np.tile(atoms_map, 7)

    pen_pairs = []
    pen_cycles = []

    for m in molecules:
        cycles = nx.cycle_basis(m)
        if not cycles: continue
        for cycle in cycles:
            flag = False
            atoms = np.array([coord[num] for num in cycle])
            if len(set([top.nodes[num]['resid'] for num in cycle])) > 1: continue
            if verbose:
                num = cycle[0]
                logger.info('found ring:', top.nodes[num]['segid'], top.nodes[num]['resid'], top.nodes[num]['resname'])

            # build least square fit plane
            axis, com, n = lsqp(atoms)

            # project atoms to the least square fit plane
            for i,atom in enumerate(atoms):
                w = np.dot(axis, atom-com)*axis + com
                atoms[i] = com + (atom - w)

            maxd = np.max(np.sqrt(np.sum(np.square(atoms - com), axis=1)))

            d = np.sqrt(np.sum(np.square(allatoms-com), axis=1))
            nums = np.squeeze(np.argwhere(d < 3))

            # find two bonded atoms that are at the opposite side of the plane
            for num in nums:
                num1 = atoms_map[num]

                for num2 in top[num1]:
                    if num1 in cycle or num2 in cycle: continue
                    if num > natoms:
                        # image atoms
                        offset = int(num / natoms)
                        coord1 = allatoms[num]
                        coord2 = allatoms[atoms_map_reverse[num2] + offset * natoms]
                    else:
                        coord1 = coord[num1]
                        coord2 = coord[num2]

                    v1 = np.dot(coord1 - com, axis)
                    v2 = np.dot(coord2 - com, axis)
                    if v1 * v2 > 0: continue

                    # point of intersection of the least square fit plane
                    s = -np.dot(axis, coord1-com)/np.dot(axis, coord2-coord1)
                    p = coord1 + s*(coord2-coord1)

                    d = np.sqrt(np.sum(np.square(p-com)))
                    if d > maxd: continue
                    if verbose:
                        logger.info('found potentially pentrarting bond:',
                              top.nodes[num1]['segid'],
                              top.nodes[num1]['resid'],
                              top.nodes[num1]['resname'],
                              top.nodes[num1]['name'],
                              top.nodes[num2]['name'])

                    d = 0
                    for i in range(0, len(atoms)):
                        p1 = atoms[i] - p
                        try: p2 = atoms[i+1] - p
                        except: p2 = atoms[0] - p
                        d += np.arccos(np.dot(p1, p2)/np.linalg.norm(p1)/np.linalg.norm(p2))

                    wn = d/2/np.pi
                    if wn > 0.9 and wn < 1.1:
                        # we have a case
                        pen_pairs.append((num1, num2))
                        pen_cycles.append(cycle)
                        flag = True
                        break

                if flag: break

    return pen_pairs, pen_cycles


def check_universe_ring_penetration(universe, verbose=0):
    """
    Check if there is any ring penetration in the universe
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        The MDAnalysis universe object; it shoud contain bond information
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    bool
        True if there is a ring penetration, False otherwise

    """
    selection = 'not resname TIP3 WAT and not (name H*)'
    faulty = False
    top = build_topology(universe, selection)
    ag = universe.select_atoms(selection)
    for frame, ts in enumerate(universe.trajectory):
        coord = dict(zip(ag.indices + 1, ag.positions))
        if len(top.nodes()) != len(coord):
            raise AtomMismatch('Number of atoms does not match')        
        #  only rect pbc have been tested
        pairs, rings = check_ring_penetration(top, coord, verbose=verbose)
        if pairs:
            logger.warning(f'In frame {frame} found a ring penetration:')
            for i, cycle in enumerate(rings):
                logger.warning(
                    f"- {top.nodes[pairs[i][0]]['segid']} {top.nodes[pairs[i][0]]['resid']} "
                    f"{top.nodes[pairs[i][0]]['resname']} {' '.join([top.nodes[num]['name'] for num in pairs[i]])} | "
                    f"{top.nodes[cycle[0]]['segid']} {top.nodes[cycle[0]]['resid']} "
                    f"{top.nodes[cycle[0]]['resname']} {' '.join([top.nodes[num]['name'] for num in cycle])}"
                )
            faulty = True
        else:
            logger.debug(f'In frame {frame} no ring penetration found')
    return faulty
