"""
This scripts is used to run a simple simulation validation on
- the simulation box
- the rmsd of the protein
- the rmsd of the ligand
- the rmsf of the protein

Maybe in the future: the membrane properties
"""
import numpy as np
import MDAnalysis as mda
from loguru import logger
import matplotlib.pyplot as plt
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.core.groups import AtomGroup
import os

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
            self.ligandz = ligand
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
        self._rmsf()
        # self._membrane()
        self._ligand_bs()
        self._ligand_dihedral()
    
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
        dihed_ligands_file = 'assign.in'
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
            atoms_str = line.split()[2:6]
            atoms_str = [selection_string(a) for a in atoms_str]
            ag_group = AtomGroup([
                self.universe.select_atoms(a).atoms[0] for a in atoms_str
            ])
            ag_lists.append(ag_group)
        
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
            plt.savefig('box_size.png')
    
    def plot_ligand_bs(self, savefig=True):
        logger.debug('Plotting RMSD')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['ligand_bs'], label='Ligand to binding site distance')
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSD (Å)')
        ax.legend()
        if savefig:
            plt.savefig('ligand_bs.png')
        
    def plot_rmsd(self, savefig=True):
        logger.debug('Plotting RMSD')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['protein_rmsd'], label='Protein')
        ax.plot(self.results['ligand_rmsd'], label='Ligand')
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSD (Å)')
        ax.legend()
        if savefig:
            plt.savefig('rmsd.png')

    def plot_rmsf(self, savefig=True):
        logger.debug('Plotting RMSF')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['ligand_rmsf'], label='Ligand')
        ax.set_xlabel('Residue')
        ax.set_ylabel('RMSF (Å)')
        if savefig:
            plt.savefig('rmsf.png')
    
    def plot_leaflet_areas(self):
        raise NotImplementedError('Membrane properties are not implemented yet')
        logger.debug('Plotting leaflet areas')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['leaflet_areas'])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Area per lipid (nm^2)')
        plt.show()
    
    # get the median value of the dihedral
    def find_representative_snapshot(self):
        """
        Find the representative snapshot based on the median dihedral values.
        """
        # convert to sin and cos values
        dihed = self.results['ligand_dihedrals']
        dihed_rad = np.deg2rad(dihed)
        sin_dihed = np.sin(dihed_rad)
        cos_dihed = np.cos(dihed_rad)

        n_dihed = dihed.shape[1]

        # Calculate the median dihedral values
        feat_dihed = np.concatenate([sin_dihed, cos_dihed], axis=1)

        median_dihed = np.median(feat_dihed, axis=0)
        
        # Calculate the absolute difference between each snapshot's dihedral and the median
        abs_diff = np.abs(feat_dihed - median_dihed)
        
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
        plt.savefig('dihed.png')
        plt.show()
        plt.close()

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
    