import numpy as np
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config
config.update("jax_enable_x64", True)
import pandas as pd

from loguru import logger


from alchemlyb.estimators import MBAR
from alchemlyb.convergence import forward_backward_convergence, block_average
from alchemlyb.visualisation import (
                plot_convergence,
                plot_mbar_overlap_matrix,
                plot_block_average
            )

import matplotlib.pyplot as plt

class MBARValidator:
    """
    Use raw input of df_list to validate the convergence of the free energy calculations using `alchemlyb`
    """
    def __init__(self,
                df_list: list,
                temperature: float = 310.0,
                energy_unit: str = 'kcal/mol',
                log_level: str = 'WARNING'
                ):
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        self._data_list = df_list
        self.temperature = temperature
        self.energy_unit = energy_unit

        #u_df_list = []
        for df in df_list:
            df.attrs['temperature'] = self.temperature
            df.attrs['energy_unit'] = self.energy_unit
            #df.set_index(['time', 'lambdas'], inplace=True)
           # u_df_list.append(df)
        
        self._data_list = df_list
        self._u_df = pd.concat(df_list)


    def analysis(self):
        """
        Perform the analysis of the free energy calculations.
        """
        # Perform the analysis
        mbar = MBAR()
        mbar.fit(self.u_df)
        self._mbar = mbar
    
    @property
    def mbar(self):
        """
        The MBAR object containing the results of the analysis.
        """
        if not hasattr(self, '_mbar'):
            raise ValueError("The analysis method must be called before accessing the mbar property.")
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
        df = forward_backward_convergence(self.data_list, 'MBAR')
        _ = plot_convergence(df, ax=ax, **kwargs)
    
    
    def plot_overlap_matrix(self, ax=None, **kwargs):
        _ = plot_mbar_overlap_matrix(self.mbar.overlap_matrix, ax=ax, **kwargs)


    def plot_block_convergence(self, ax=None, **kwargs):
        df = block_average(self.data_list, 'MBAR')
        _ = plot_block_average(df, ax=ax, **kwargs)

    def plot_convergence(self, save_path=None, title=None):
        fig, axes = plt.subplot_mosaic(
            [["A", "A", "B"], ["C", "C", "C"]], figsize=(15, 10)
        )
        self.plot_time_convergence(ax=axes['A'],
                                   units='kcal/mol',
                                   final_error=0.6)
        self.plot_overlap_matrix(ax=axes['B'])
        self.plot_block_convergence(ax=axes['C'],
                                    units='kcal/mol',
                                    final_error=0.6)
        axes['A'].set_title('Time Convergence', fontsize=10)
        axes['B'].set_title('Overlap Matrix', fontsize=10)
        axes['C'].set_title('Block Convergence', fontsize=10)

        if title:
            plt.suptitle(title, y=1.05)
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def write_results(self, save_path=None):
        """
        Write the results to a file.
        """
        if save_path:
            with open(save_path, 'w') as f:
                f.write(f"Temperature: {self.temperature} K\n")
                f.write(f"Energy unit: {self.energy_unit}\n")
                f.write(f"Number of dataframes: {len(self.data_list)}\n")
                for i, df in enumerate(self.data_list):
                    f.write(f"\nDataframe {i}:\n")
                    f.write(df.to_string())
        else:
            print(f"Temperature: {self.temperature} K")
            print(f"Energy unit: {self.energy_unit}")
            print(f"Number of dataframes: {len(self.data_list)}")
            for i, df in enumerate(self.data_list):
                print(f"\nDataframe {i}:\n")
                print(df)


class ConvergenceValidator(MBARValidator):
    """
    Class to validate the convergence of the free energy calculations using `alchemlyb` including
    - Time Convergence (forward and backward convergence)
    - Overlap Matrix
    """
    def __init__(self,
                Upot: np.ndarray,
                lambdas: list,
                temperature: float = 310.0,
                energy_unit: str = 'kcal/mol',
                log_level: str = 'WARNING'
                ):
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        self.Upot = Upot
        self.lambdas = lambdas
        self.temperature = temperature

        if self.Upot.ndim != 3:
            raise ValueError("Upot must be a 3D array, with shape (n_lambdas, n_lambdas, n_frames)")

        if self.Upot.shape[0] != len(lambdas) or self.Upot.shape[1] != len(lambdas):
            raise ValueError("The first two dimensions of Upot must match the length of lambdas")
        
        self._generate_df()
    
    def _generate_df(self):
        lambdas = self.lambdas
        u_df = pd.DataFrame(columns=['time', 'fep-lambda'] + list(lambdas))
        u_df.attrs['temperature'] = self.temperature
        u_df.attrs['energy_unit'] = 'kT'

        data_frames = []

        for i, u_pot_lambda in enumerate(self.Upot):
            fep = u_pot_lambda.T  # Transpose if needed
            time = np.arange(fep.shape[0])  # Create the time array based on the number of rows in `fep`
            fep_lambda = np.ones(fep.shape[0]) * lambdas[i]  # Create the fep-lambda array based on the current lambda
            # Stack the arrays horizontally
            result = np.hstack([time[:, None], fep_lambda[:, None], fep])  
            
            df = pd.DataFrame(result,
                            columns=['time', 'fep-lambda'] + list(lambdas))
            data_frames.append(df)

        # Combine all DataFrames into a single DataFrame
        u_df = pd.concat(data_frames, ignore_index=True)
        # set index to time and fep-lambda
        u_df.set_index(['time', 'fep-lambda'], inplace=True)
        self._u_df = u_df
        data_list = []
        # split u_def based on fep-lambda
        for fep, group in u_df.groupby('fep-lambda'):
            group.attrs['temperature'] = self.temperature
            group.attrs['energy_unit'] = 'kT'
            data_list.append(group)
        self._data_list = data_list
