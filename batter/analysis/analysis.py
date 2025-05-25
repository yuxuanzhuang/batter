import pandas as pd
from joblib import Parallel, delayed
from pymbar.timeseries import detect_equilibration
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
    def __init__(self, comp_folder):
        if not os.path.exists(comp_folder):
            raise ValueError(f"Component folder {comp_folder} does not exist.")
        self.comp_folder = comp_folder
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
                windows,
                temperature,
                energy_unit='kcal/mol',
                log_level='INFO',
                sim_range=None,
                n_bootstraps=100,
                n_jobs=12,
                load=False,
                ):
        super().__init__(comp_folder)
        self.windows = windows
        self.temperature = temperature
        if energy_unit not in ['kcal/mol', 'kJ/mol', 'kT']:
            raise ValueError(f"Energy unit {energy_unit} not recognized. Use 'kcal/mol', 'kJ/mol' or 'kT'.")
        self.energy_unit = energy_unit
        self.kT = 0.0019872041 * self.temperature
        self.sim_range = sim_range
        self.n_bootstraps = n_bootstraps
        self.n_jobs = n_jobs
        self.load = load
        self._data_initialized = False

        logger.remove()
        logger.add(sys.stderr, level=log_level)

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
        logger.info(f"Calculating convergence for {self.component}...")
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
            t0, g, Neff_max = detect_equilibration(df.iloc[:, win_i], nskip=10)
        df = df[df.index.get_level_values(0) > t0]
        return df

    def _get_data_list(self):
        logger.info(f"Extracting data for {self.component}...")

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
        logger.info(f"Plotting convergence for {self.component}...")
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


class E_MBARAnalysis(MBARAnalysis):
    """
    Analysis class for E_MBAR.
    """
    component = 'e'


class V_MBARAnalysis(MBARAnalysis):
    """
    Analysis class for V_MBAR.
    """
    component = 'v'


class O_MBARAnalysis(MBARAnalysis):
    """
    Analysis class for O_MBAR.
    """
    component = 'o'

class SuppressLoguru:
    def __enter__(self):
        self.handler_ids = list(logger._core.handlers.keys())
        for handler_id in self.handler_ids:
            logger.remove(handler_id)
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Add a default stderr logger back
        logger.add(sys.stderr)
