import pandas as pd
from joblib import Parallel, delayed
from pymbar.timeseries import detect_equilibration
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.convergence import forward_backward_convergence, block_average
from alchemlyb.visualisation import (
                plot_convergence,
                plot_mbar_overlap_matrix,
                plot_block_average
            )
import matplotlib.pyplot as plt
import numpy as np
import pickle

from abc import ABC, abstractmethod
from loguru import logger
import sys
import glob
import os

class FEAnalysisBase(ABC):
    """
    Abstract base class for analysis of each component.
    """
    def __init__(self):
        self.results = {
            'fe': None,
            'fe_error': None,
            'convergence': {}
        }

    @abstractmethod
    def run_analysis(self):
        """
        Run the analysis for the component.
        """
        pass

    @abstractmethod
    def plot_convergence(self, ax=None, **kwargs):
        """
        Plot the convergence of the component.
        """
        pass

    @property
    def fe(self):
        """
        Get the free energy results.
        """
        return self.results['fe']
    
    @property
    def fe_error(self):
        """
        Get the free energy error from bootstrapping.
        """
        return self.results['fe_error']

class MBARAnalysis(FEAnalysisBase):
    def __init__(self,
                comp_folder,
                component,
                windows,
                temperature,
                energy_unit='kcal/mol',
                log_level='INFO',
                sim_range=None,
                n_bootstraps=100,
                n_jobs=12,
                load=False,
                ):
        """
        Initialize the MBAR analysis with the component folder and parameters.
        
        Parameters
        ----------
        comp_folder : str
            The path to the component folder containing the simulation data.
        component : str
            The name of the component to analyze (e.g., 'v', 'e', 'o')
        windows : list of int
            The list of windows to analyze. Each window corresponds to a different lambda value.
        temperature : float
            The temperature in Kelvin for the analysis.
        energy_unit : str, optional
            The energy unit for the results. Can be 'kcal/mol', 'kJ/mol', or 'kT'. Default is 'kcal/mol'.
        log_level : str, optional
            The log level for logging the analysis. Default is 'INFO'.
        sim_range : tuple of int, optional
            The range of simulations to include in the analysis. If None, all simulations are included.
        n_bootstraps : int, optional
            The number of bootstraps to perform for error estimation. Default is 100.
        n_jobs : int, optional
            The number of parallel jobs to run for data extraction. Default is 12.
        load : bool, optional
            If True, load the data from a previously saved file instead of extracting it again. 
            """
        super().__init__()
        if not os.path.exists(comp_folder):
            raise ValueError(f"Component folder {comp_folder} does not exist.")
        self.comp_folder = comp_folder
        self.component = component

        self.windows = windows
        self.temperature = temperature
        if energy_unit not in ['kcal/mol', 'kJ/mol', 'kT']:
            raise ValueError(f"Energy unit {energy_unit} not recognized. Use 'kcal/mol', 'kJ/mol' or 'kT'.")
        self.energy_unit = energy_unit
        self.kT = 0.0019872041 * self.temperature
        self.sim_range = sim_range
        if sim_range is not None and not isinstance(sim_range, (list, tuple)):
            raise ValueError("sim_range must be a list or tuple of integers.")
        if isinstance(sim_range, (list, tuple)) and len(sim_range) == 2:
            self.sim_range = range(sim_range[0], sim_range[1])
        elif isinstance(sim_range, (list, tuple)) and len(sim_range) > 2:
            raise ValueError("sim_range must be a list or tuple of two integers.")

        self.n_bootstraps = n_bootstraps
        self.n_jobs = n_jobs
        self.load = load
        self._data_initialized = False

        #logger.remove()
        #logger.add(sys.stderr, level=log_level)

    def get_mbar_data(self):
        """
        Get the data for the component.
        """
        if os.path.exists(f'{self.comp_folder}/{self.component}_df_list.pickle') and self.load:
            with open(f'{self.comp_folder}/{self.component}_df_list.pickle', 'rb') as f:
                df_list = pickle.load(f)
        else:
            df_list = self._get_data_list()

        self._data_list = df_list
        self._u_df = pd.concat(df_list)
        self._data_initialized = True
    
    def run_analysis(self):
        if not self._data_initialized:
            self.get_mbar_data()

        mbar = MBAR(n_bootstraps=self.n_bootstraps)
        mbar.fit(self.u_df)
        self._mbar = mbar

        error = np.sqrt(
            sum([mbar.d_delta_f_.iloc[i, i+1]**2 for i in range(len(mbar.d_delta_f_)-1)])
        )
        # fe
        if self.energy_unit == 'kcal/mol':
            self.results['fe'] = mbar.delta_f_.iloc[0, -1] * self.kT
            self.results['fe_error'] = error * self.kT
        elif self.energy_unit == 'kJ/mol':
            self.results['fe'] = mbar.delta_f_.iloc[0, -1] * self.kT * 4.184
            self.results['fe_error'] = error * self.kT * 4.184
        elif self.energy_unit == 'kT':
            self.results['fe'] = mbar.delta_f_.iloc[0, -1]
            self.results['fe_error'] = error
        logger.info(f"Free energy results for {self.component}: {self.results['fe']:.2f} +- {self.results['fe_error']:.2f} {self.energy_unit}")

        # convergence
        logger.debug(f"Calculating convergence for {self.component}...")
        with SuppressLoguru():
            self.results['convergence']['time_convergence'] = forward_backward_convergence(self.data_list, 'MBAR')
            self.results['convergence']['block_convergence'] = block_average(self.data_list, 'MBAR')
            self.results['convergence']['overlap_matrix'] = mbar.overlap_matrix
            self.results['convergence']['mbar'] = mbar
        
        # Save the results
        with open(f'{self.comp_folder}/{self.component}_results.pickle', 'wb') as f:
            pickle.dump(self.results, f)

    @staticmethod
    def _extract_all_for_window(win_i, comp_folder, component, temperature,
                               sim_range):
        """
        Return a dataframe with the energy in kT units for the given window.
        """
        if sim_range is None:
            n_sims = len(glob.glob(f'{comp_folder}/{component}{win_i:02d}/mdin-*.out'))
            sim_range = range(n_sims)
        logger.debug(f"Extracting data for {component}{win_i:02d} with {len(sim_range)} simulations...")
        with SuppressLoguru():
            df = pd.concat([
                extract_u_nk(f'{comp_folder}/{component}{win_i:02d}/mdin-{i:02d}.out',
                            T=temperature, reduced=False)
                for i in sim_range
            ])
            t0, g, Neff_max = detect_equulibration(df.iloc[:, win_i], nskip=10)
        df = df[df.index.get_level_values(0) > t0]
        return df

    def _get_data_list(self):
        logger.debug(f"Extracting data for {self.component}...")

        df_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_all_for_window)(win_i=win_i,
                                                 comp_folder=self.comp_folder,
                                                 component=self.component,
                                                 temperature=self.temperature,
                                                 sim_range=self.sim_range)           
                for win_i in range(len(self.windows)))
        with open(f'{self.comp_folder}/{self.component}_df_list.pickle', 'wb') as f:
            pickle.dump(df_list, f)

        for df in df_list:
            df.attrs['temperature'] = self.temperature
            df.attrs['energy_unit'] = 'kT'
        
        return df_list

    @property
    def mbar(self):
        """
        The MBAR object containing the results of the analysis.
        """
        if not hasattr(self, '_mbar'):
            mbar = MBAR()
            mbar.fit(self.u_df)
            self._mbar = mbar
        return self._mbar

    @property
    def u_df(self):
        """
        The dataframe containing the free energy profiles.
        See https://alchemlyb.readthedocs.io/en/latest/parsing.html for more details.
        """
        return self._u_df

    @property
    def data_list(self):
        """
        The list of dataframes containing the free energy profiles for each lambda.
        See https://alchemlyb.readthedocs.io/en/latest/parsing.html for more details.
        """
        return self._data_list

    def plot_time_convergence(self, ax=None, **kwargs):
        df = self.results['convergence']['time_convergence']
        _ = plot_convergence(df, ax=ax, **kwargs)
    
    def plot_overlap_matrix(self, ax=None, **kwargs):
        _ = plot_mbar_overlap_matrix(self.results['convergence']['overlap_matrix'],
                                     ax=ax, **kwargs)

    def plot_block_convergence(self, ax=None, **kwargs):
        df = self.results['convergence']['block_convergence']
        _ = plot_block_average(df, ax=ax, **kwargs)

    def plot_convergence(self, save_path=None, title=None):
        logger.debug(f"Plotting convergence for {self.component}...")
        fig, axes = plt.subplot_mosaic(
            [["A", "A", "B"], ["C", "C", "C"]], figsize=(15, 10)
        )
        self.plot_time_convergence(ax=axes['A'],
                                units=self.energy_unit,
                                final_error=0.6)
        self.plot_overlap_matrix(ax=axes['B'])
        self.plot_block_convergence(ax=axes['C'],
                                    units=self.energy_unit,
                                    final_error=0.6)
        axes['A'].set_title('Time Convergence', fontsize=10)
        axes['B'].set_title('Overlap Matrix', fontsize=10)
        axes['C'].set_title('Block Convergence', fontsize=10)

        plt.tight_layout()
        if title:
            plt.suptitle(title, y=1.05)
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


class BoreschAnalysis(FEAnalysisBase):
    def __init__(self, disangfile, k_r, k_a, temperature):
        """
        Initialize the Boresch analysis with the disang file and parameters.

        Parameters
        ----------
        disangfile : str
            The path to the disang file containing the anchor atoms.
        k_r : float
            The force constant for the translation restraint.
        k_a : float
            The force constant for the angle and dihedral restraints.
            They are the same (they don't have to be).
        temperature : float
            The temperature in Kelvin for the analysis.
        """
        super().__init__()
        self.disangfile = disangfile
        self.k_r = k_r
        self.k_a = k_a
        self.temperature = temperature

    def run_analysis(self):
        """
        Run the analytical analysis for Boresch restraint.
        """
        logger.debug("Running analytical analysis for Boresch restraint")
        
        # Read disang file to get anchor atoms
        with open(self.disangfile, 'r') as f_in:        
            lines = [line.rstrip() for line in f_in]

            lines = list(line for line in lines if '#Lig_TR' in line)
            splitdata = lines[0].split()
            r0 = float(splitdata[6].strip(','))
            splitdata = lines[1].split()
            a1_0 = float(splitdata[6].strip(','))
            splitdata = lines[2].split()
            t1_0 = float(splitdata[6].strip(','))
            splitdata = lines[3].split()
            a2_0 = float(splitdata[6].strip(','))
            splitdata = lines[4].split()
            t2_0 = float(splitdata[6].strip(','))
            splitdata = lines[5].split()
            t3_0 = float(splitdata[6].strip(','))
            fe_bd = self.fe_int(r0, a1_0, t1_0, a2_0, t2_0, t3_0,
                                self.k_r, self.k_a, self.temperature)
            self.results['fe'] = fe_bd
            self.results['fe_error'] = 0.0
            logger.debug(f'Analytical release ligand TR: {fe_bd:.2f} kcal/mol')


    def _run_analysis(self,
                     protein_path,
                     protein_anchor_atmoms,
                     ligand_anchor_atmoms,
                     k_r,
                     k_a,
                     temperature):
        """
        Get analytical results of Bosrech restraint.

        P3-P2-P1-L1-L2-L3
        r0 = d(P1, L1)
        a1_0 = angle(P2, P1, L1)
        t1_0 = dihedral(P3, P2, P1, L1)
        a2_0 = angle(P1, L1, L2)
        t2_0 = dihedral(P2, P1, L1, L2)
        t3_0 = dihedral(P1, L1, L2, L3)

        Parameters
        ----------
        protein_path : str
            The path to the protein structure file.
        protein_anchor_atmoms : list of str
            The selection string for the protein anchor atoms in MDAnalysis.
        ligand_anchor_atmoms : list of str
            The selection string for the ligand anchor atoms in MDAnalysis.
        k_r : float
            The force constant for the translation restraint.
        ka : float
            The force constant for the angle and dihedral restraints.
            They are the same (they don't have to be).
        temperature : float
            The temperature in Kelvin for the analysis.
        """
        raise NotImplementedError("This method is not implemented yet.")
        logger.debug("Getting analytical results for Boresch restraint")

        u = mda.Universe(protein_path)
        l1 = u.select_atoms(protein_anchor_atmoms[0])
        l2 = u.select_atoms(protein_anchor_atmoms[1])
        l3 = u.select_atoms(protein_anchor_atmoms[2])        

        p1 = u.select_atoms(ligand_anchor_atmoms[0])
        p2 = u.select_atoms(ligand_anchor_atmoms[1])
        p3 = u.select_atoms(ligand_anchor_atmoms[2])

        # make sure they all have one atom
        if len(l1) != 1 or len(l2) != 1 or len(l3) != 1 or \
              len(p1) != 1 or len(p2) != 1 or len(p3) != 1:
            raise ValueError("Anchor atoms must be single atoms.")
        
        r0 = calc_bonds(l1.positions[0],
                          p1.position[0],
                          box=u.dimensions)
        a1_0 = calc_angles(p2.positions[0],
                          p1.positions[0], l1.positions[0],
                          box=u.dimensions)
        t1_0 = calc_dihedrals(p3.positions[0],
                              p2.positions[0], p1.positions[0], l1.positions[0],
                              box=u.dimensions)
        a2_0 = calc_angles(p1.positions[0],
                            l1.positions[0], l2.positions[0],
                            box=u.dimensions)
        t2_0 = calc_dihedrals(p2.positions[0],
                                p1.positions[0], l1.positions[0], l2.positions[0],
                                box=u.dimensions)
        t3_0 = calc_dihedrals(p1.positions[0],
                                l1.positions[0], l2.positions[0], l3.positions[0],
                                box=u.dimensions)
        rest = [r0, a1_0, t1_0, a2_0, t2_0, t3_0, k_r, k_a]
        logger.debug(f'Boresch restraint: {rest}')
        fe_bd = self.fe_int(r0, a1_0, t1_0, a2_0, t2_0, t3_0, k_r, k_a, temperature)
        logger.debug(f'Analytical release ligand TR: {fe_bd:.2f} kcal/mol')
        self.results['fe'] = fe_bd
        self.results['fe_error'] = 0.0

    def plot_convergence(self, ax=None, **kwargs):
        """
        no convergence for analytical results
        """
        pass

    @staticmethod
    def fe_int(r1_0, a1_0, t1_0, a2_0, t2_0, t3_0, k_r, k_a, temperature):
        """
        Calculate the analytical free energy of boresch restraint.
        from BAT.py
        """
        R = 1.987204118e-3  # kcal/mol-K, a.k.a. boltzman constant
        beta = 1/(temperature*R)
        r1lb, r1ub, r1st = [0.0, 100.0, 0.0001]
        a1lb, a1ub, a1st = [0.0, np.pi, 0.00005]
        t1lb, t1ub, t1st = [-np.pi, np.pi, 0.00005]
        a2lb, a2ub, a2st = [0.0, np.pi, 0.00005]
        t2lb, t2ub, t2st = [-np.pi, np.pi, 0.00005]
        t3lb, t3ub, t3st = [-np.pi, np.pi, 0.00005]

        def dih_per(lb, ub, st, t_0):
            drange = np.arange(lb, ub, st)
            delta = (drange-np.radians(t_0))
            for i in range(0, len(delta)):
                if delta[i] >= np.pi:
                    delta[i] = delta[i]-(2*np.pi)
                if delta[i] <= -np.pi:
                    delta[i] = delta[i]+(2*np.pi)
            return delta

        def f_r1(val):
            return (val**2)*np.exp(-beta*k_r*(val-r1_0)**2)

        def f_a1(val):
            return np.sin(val)*np.exp(-beta*k_a*(val-np.radians(a1_0))**2)

        def f_a2(val):
            return np.sin(val)*np.exp(-beta*k_a*(val-np.radians(a2_0))**2)

        def f_t1(delta):
            return np.exp(-beta*k_a*(delta)**2)

        def f_t2(delta):
            return np.exp(-beta*k_a*(delta)**2)

        def f_t3(delta):
            return np.exp(-beta*k_a*(delta)**2)

        # Integrate translation and rotation
        r1_int, a1_int, t1_int, a2_int, t2_int, t3_int = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        intrange = np.arange(r1lb, r1ub, r1st)
        r1_int = np.trapz(f_r1(intrange), intrange)
        intrange = np.arange(a1lb, a1ub, a1st)
        a1_int = np.trapz(f_a1(intrange), intrange)
        intrange = dih_per(t1lb, t1ub, t1st, t1_0)
        t1_int = np.trapz(f_t1(intrange), intrange)
        intrange = np.arange(a2lb, a2ub, a2st)
        a2_int = np.trapz(f_a2(intrange), intrange)
        intrange = dih_per(t2lb, t2ub, t2st, t2_0)
        t2_int = np.trapz(f_t2(intrange), intrange)
        intrange = dih_per(t3lb, t3ub, t3st, t3_0)
        t3_int = np.trapz(f_t3(intrange), intrange)
        return R*temperature*np.log((1/(8.0*np.pi*np.pi))*(1.0/1660.0)*r1_int*a1_int*t1_int*a2_int*t2_int*t3_int)





class SuppressLoguru:
    def __enter__(self):
        self.handler_ids = list(logger._core.handlers.keys())
        for handler_id in self.handler_ids:
            logger.remove(handler_id)
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger_format = ('{level} | <level>{message}</level> ')
        # format time to be human readable
        logger.add(sys.stderr, format=logger_format, level="INFO")
