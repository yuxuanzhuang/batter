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
            logger.info(f'Guessed ligand resname: {self.ligand}')
        if len(possible_resnames) == 0:
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
    
    def _box(self):
        logger.info('Calculating box size')
        results = []
        for ts in self.universe.trajectory:
            box = ts.dimensions[:3]
            results.append(box.copy())
        self.results['box'] = results

    def _rmsd(self):
        logger.info('Calculating RMSD')
        from MDAnalysis.analysis.rms import RMSD
        rms = RMSD(self.universe,
                   self.universe,
                   groupselections=[f'resname {self.ligand}'],
                   select='name CA').run()
        self.results['protein_rmsd'] = rms.results.rmsd.T[2]
        self.results['ligand_rmsd'] = rms.results.rmsd.T[3]
    
    def _rmsf(self):
        logger.info('Calculating RMSF')
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
        logger.info('Calculating membrane properties')
        # Find which leaflet each lipid is in at each frame
        leaflets = lpp.AssignLeaflets(
            universe=self.universe,
            lipid_sel="resname OL PA PC"
            )

        leaflets.run()
        
        logger.info('Calculating leaflet areas')
        areas = lpp.analysis.AreaPerLipid(
            universe=self.universe,
            lipid_sel="resname OL PA PC",
            leaflets=leaflets.leaflets
            )

        areas.run()
        self.results['leaflet_areas'] = areas.areas

    def plot_box(self):
        logger.info('Plotting box size')
        box_results = np.array(self.results['box'])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(box_results[:, 0], label='x')
        ax.plot(box_results[:, 1], label='y')
        ax.plot(box_results[:, 2], label='z')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Box size (nm)')
        ax.legend()
        plt.show()
    
    def plot_rmsd(self):
        logger.info('Plotting RMSD')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['protein_rmsd'], label='Protein')
        ax.plot(self.results['ligand_rmsd'], label='Ligand')
        ax.set_xlabel('Frame')
        ax.set_ylabel('RMSD (nm)')
        ax.legend()
        plt.show()

    def plot_rmsf(self):
        logger.info('Plotting RMSF')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['ligand_rmsf'], label='Ligand')
        ax.set_xlabel('Residue')
        ax.set_ylabel('RMSF (nm)')
        plt.show()
    
    def plot_leaflet_areas(self):
        raise NotImplementedError('Membrane properties are not implemented yet')
        logger.info('Plotting leaflet areas')
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.results['leaflet_areas'])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Area per lipid (nm^2)')
        plt.show()
    