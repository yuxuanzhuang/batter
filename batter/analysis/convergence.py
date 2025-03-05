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
from alchemlyb.visualisation import plot_convergence
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.visualisation import plot_mbar_overlap_matrix
import matplotlib.pyplot as plt


class ConvergenceValidator:
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


    def plot_convergence(self, save_path=None, title=None):
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        self.plot_time_convergence(ax=axes[0], units='kcal/mol')
        self.plot_overlap_matrix(ax=axes[1])
        axes[0].set_title('Time Convergence', fontsize=10)
        axes[1].set_title('Overlap Matrix', fontsize=10)
        if title:
            plt.suptitle(title, y=1.05)
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


    def _generate_df(self):
        lambdas = self.lambdas
        u_df = pd.DataFrame(columns=['time', 'fep-lambda'] + list(lambdas))
        u_df.attrs['temperature'] = self.temperature
        u_df.attrs['energy_unit'] = 'kcal/mol'

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
            group.attrs['energy_unit'] = 'kcal/mol'
            data_list.append(group)
        self._data_list = data_list

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
        df = forward_backward_convergence(self.data_list, 'mbar')
        _ = plot_convergence(df, ax=ax, **kwargs)
    
    
    def plot_overlap_matrix(self, ax=None, **kwargs):
        mbar = MBAR()
        mbar.fit(self.u_df)
        _ = plot_mbar_overlap_matrix(mbar.overlap_matrix, ax=ax, **kwargs)
