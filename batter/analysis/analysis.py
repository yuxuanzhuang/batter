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

from batter.utils import run_with_log, cpptraj


import matplotlib.pyplot as plt
import numpy as np
import pickle

from abc import ABC, abstractmethod
from loguru import logger
import sys
import glob
import os
from batter.utils import (
    COMPONENTS_FOLDER_DICT,
)

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
                pose_folder,
                component,
                windows,
                temperature,
                energy_unit='kcal/mol',
                sim_range=None,
                detect_equil=True,
                n_bootstraps=25,
                n_jobs=6,
                load=False,
                ):
        """
        Initialize the MBAR analysis with the component folder and parameters.
        
        Parameters
        ----------
        pose_folder : str
            The path to the pose folder containing the simulation data.
        component : str
            The name of the component to analyze (e.g., 'v', 'e', 'o')
        windows : list of int
            The list of windows to analyze. Each window corresponds to a different lambda value.
        temperature : float
            The temperature in Kelvin for the analysis.
        energy_unit : str, optional
            The energy unit for the results. Can be 'kcal/mol', 'kJ/mol', or 'kT'. Default is 'kcal/mol'.
        sim_range : tuple of int, optional
            The range of simulations to include in the analysis. If None, all simulations are included.
        detect_equil : bool, optional
            If True, detect equilibration time and truncate the data accordingly. Default is False.
        n_bootstraps : int, optional
            The number of bootstraps to perform for error estimation. Default is 0.
        n_jobs : int, optional
            The number of parallel jobs to run for data extraction. Default is 12.
        load : bool, optional
            If True, load the data from a previously saved file instead of extracting it again. 
            """
        super().__init__()
        self.pose_folder = pose_folder
        self.result_folder = f'{self.pose_folder}/Results'
        os.makedirs(f'{self.result_folder}', exist_ok=True)
        comp_folder = f'{self.pose_folder}/{COMPONENTS_FOLDER_DICT[component]}'
        if not os.path.exists(comp_folder):
            raise ValueError(f"Component folder {comp_folder} does not exist.")
        self.comp_folder = comp_folder
        self.component = component
        logger.debug(f"Initializing MBAR analysis for component {self.component} in folder {self.comp_folder}")

        self.windows = windows
        self.temperature = temperature
        if energy_unit not in ['kcal/mol', 'kJ/mol', 'kT']:
            raise ValueError(f"Energy unit {energy_unit} not recognized. Use 'kcal/mol', 'kJ/mol' or 'kT'.")
        self.energy_unit = energy_unit
        self.kT = 0.0019872041 * self.temperature
        if sim_range is not None:
            if not isinstance(sim_range, (list, tuple)):
                raise ValueError("sim_range must be a list or tuple of integers.")
            if len(sim_range) != 2:
                raise ValueError("sim_range must be a list or tuple of exactly two integers.")
            self.sim_range = range(sim_range[0], sim_range[1])
        else:
            self.sim_range = sim_range
        self.detect_equil = detect_equil
        self.n_bootstraps = n_bootstraps
        self.n_jobs = n_jobs
        self.load = load
        self._data_initialized = False

    def get_mbar_data(self):
        """
        Get the data for the component.
        """
        if os.path.exists(f'{self.result_folder}/{self.component}_df_list.pickle') and self.load:
            with open(f'{self.result_folder}/{self.component}_df_list.pickle', 'rb') as f:
                df_list = pickle.load(f)
        else:
            df_list = self._get_data_list()

        self._data_list = df_list
        self._u_df = pd.concat(df_list)
        self.timeseries = [df.index.get_level_values('time').values for df in df_list]
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
        logger.debug(f"Free energy results for {self.component}: {self.results['fe']:.2f} +- {self.results['fe_error']:.2f} {self.energy_unit}")

        # convergence

        logger.debug(f"Calculating convergence for {self.component}...")
        with SuppressLoguru():
            logger.debug("Calculating forward-backward convergence...")
            self.results['convergence']['time_convergence'] = forward_backward_convergence(self.data_list, 'MBAR')
            forward_end_time = [
                [series[int(len(series) * fraction)-1]
                for series in self.timeseries]
                for fraction in self.results['convergence']['time_convergence']['data_fraction']
            ]
            backward_start_time = [
                [series[int(len(series) * (1 - fraction))-1]
                for series in self.timeseries]
                for fraction in self.results['convergence']['time_convergence']['data_fraction']
            ]
            forward_FE = self.results['convergence']['time_convergence'].Forward.values
            forward_FE_err = self.results['convergence']['time_convergence'].Forward_Error.values
            backward_FE = self.results['convergence']['time_convergence'].Backward.values
            backward_FE_err = self.results['convergence']['time_convergence'].Backward_Error.values

            forward_end_time_tuples = [tuple(times) for times in forward_end_time]
            backward_start_time_tuples = [tuple(times) for times in backward_start_time]

            self.results['convergence']['forward_timeseries'] = pd.DataFrame(
                np.column_stack([forward_FE, forward_FE_err]),
                index=pd.MultiIndex.from_tuples(forward_end_time_tuples,
                                                names=[f"time_{i}" for i in range(len(forward_end_time_tuples[0]))]),
                columns=['FE', 'FE_Error']
            )
            self.results['convergence']['backward_timeseries'] = pd.DataFrame(
                np.column_stack([backward_FE, backward_FE_err]),
                index=pd.MultiIndex.from_tuples(backward_start_time_tuples,
                                                names=[f"time_{i}" for i in range(len(backward_start_time_tuples[0]))]),
                columns=['FE', 'FE_Error']
            )
            
            logger.debug("Calculating block average convergence...")
            num_blocks = 10
            self.results['convergence']['block_convergence'] = block_average(self.data_list,
                                                                             estimator='MBAR',
                                                                             num=num_blocks)

            block_FE = self.results['convergence']['block_convergence'].FE.values
            block_FE_err = self.results['convergence']['block_convergence'].FE_Error.values
            fractions = np.linspace(0, 1, num_blocks)
            block_start_time = [
                [series[int(len(series) * fraction)]
                for series in self.timeseries]
                for fraction in fractions[:-1]
            ]
            block_end_time = [
                [series[int(len(series) * fraction)-1]
                for series in self.timeseries]
                for fraction in fractions[1:]
            ]

            # Store both start and end times in a MultiIndex
            block_times = [
                (tuple(start), tuple(end))
                for start, end in zip(block_start_time, block_end_time)
            ]

            self.results['convergence']['block_timeseries'] = pd.DataFrame(
                np.column_stack([block_FE, block_FE_err]),
                index=pd.MultiIndex.from_tuples(block_times, names=["start_time", "end_time"]),
                columns=['FE', 'FE_Error']
            )

            logger.debug("Calculating overlap matrix...")
            self.results['convergence']['overlap_matrix'] = mbar.overlap_matrix
            self.results['convergence']['mbar'] = mbar
        
        # Save the results
        with open(f'{self.result_folder}/{self.component}_results.pickle', 'wb') as f:
            pickle.dump(self.results, f)

    @staticmethod
    def _extract_all_for_window(win_i, comp_folder, component, temperature,
                               sim_range, truncate):
        """
        Return a dataframe with the energy in kT units for the given window.
        """
        if sim_range is None:
            n_sims = len(glob.glob(f'{comp_folder}/{component}{win_i:02d}/mdin-*.out'))
            sim_range = range(n_sims)
        logger.debug(f"Extracting data for {component}{win_i:02d} with {len(sim_range)} simulations...")


        md_sim_files = []
        for i in sim_range:
            if os.path.exists(f'{comp_folder}/{component}{win_i:02d}/mdin-{i:02d}.out'):
                md_sim_files.append(f'{comp_folder}/{component}{win_i:02d}/mdin-{i:02d}.out')
            else:
                raise FileNotFoundError(f"Simulation file {comp_folder}/{component}{win_i:02d}/mdin-{i:02d}.out not found.")

        if len(md_sim_files) == 0:
            # the simulation is done probably in older versions
            md_sim_files = [
                #f'{comp_folder}/{component}{win_i:02d}/md-01.out',
                #f'{comp_folder}/{component}{win_i:02d}/md-02.out',
                f'{comp_folder}/{component}{win_i:02d}/md-03.out',
                f'{comp_folder}/{component}{win_i:02d}/md-04.out'
            ]
            if all(not os.path.exists(md_file) for md_file in md_sim_files):
                raise FileNotFoundError(f"No simulation files found for {component}{win_i:02d}")

        with SuppressLoguru():
            dfs = []
            for md_sim_file in md_sim_files:
                try:
                    df_part = extract_u_nk(md_sim_file, T=temperature, reduced=False, raise_error=False)
                    dfs.append(df_part)
                except Exception as e:
                    raise RuntimeError(f"Error processing {md_sim_file}: {e}")

            df = pd.concat(dfs)
            if truncate:
                t0, g, Neff_max = detect_equilibration(df.iloc[:, win_i], nskip=10)
                df = df[df.index.get_level_values(0) > t0]
        return df

    def _get_data_list(self):
        logger.debug(f"Extracting data for {self.component}...")

        logger.debug(f"windows: {self.windows}")
        logger.debug(f"sim_range: {self.sim_range}")
        logger.debug(f"temperature: {self.temperature}")
        logger.debug(f"component: {self.component}")
        df_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_all_for_window)(win_i=win_i,
                                                 comp_folder=self.comp_folder,
                                                 component=self.component,
                                                 temperature=self.temperature,
                                                 sim_range=self.sim_range,
                                                 truncate=self.detect_equil) 
                for win_i in range(len(self.windows)))
        for df in df_list:
            df.attrs['temperature'] = self.temperature
            df.attrs['energy_unit'] = 'kT'
        
        with open(f'{self.result_folder}/{self.component}_df_list.pickle', 'wb') as f:
            pickle.dump(df_list, f)

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
            [["A", "A", "B"], ["C", "C", "C"]], figsize=(25, 15)
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

class RESTMBARAnalysis(MBARAnalysis):
    """
    Analysis class for components that calculate applied restraints and use MBAR
    to estimate free energy of applying the restraint.
    """
    def _extract_restraints_from_windows(self):
        num_win = len(self.windows)
        component = self.component
        # Read disang file to get restraints type
        disang_file = f'{self.comp_folder}/{component}00/disang.rest'
        with open(disang_file, 'r') as f:
            disang =  f.readlines()

        num_rest = 0
        if (component == 't'):
            for line in disang:
                cols = line.split()
                if len(cols) != 0 and (cols[-1] == "#Lig_TR"):
                    num_rest += 1
        elif (component == 'l' or component == 'c'):
            for line in disang:
                cols = line.split()
                if len(cols) != 0 and (cols[-1] == "#Lig_C" or cols[-1] == "#Lig_D"):
                    num_rest += 1
        elif (component == 'a' or component == 'r'):
            for line in disang:
                cols = line.split()
                if len(cols) != 0 and (cols[-1] == "#Rec_C" or cols[-1] == "#Rec_D"):
                    num_rest += 1
        elif (component == 'm' or component == 'n'):
            for line in disang:
                cols = line.split()
                if len(cols) != 0 and (cols[-1] == "#Rec_C" or cols[-1] == "#Rec_D" or cols[-1] == "#Lig_TR" or cols[-1] == "#Lig_C" or cols[-1] == "#Lig_D"):
                    num_rest += 1
        
        rty = ['d'] * num_rest
        rfc = np.zeros([num_win, num_rest], dtype=float)
        req = np.zeros([num_win, num_rest], dtype=float)
                       
        for win in range(num_win):
            disang_file = f'{self.comp_folder}/{component}{win:02d}/disang.rest'
            with open(disang_file, 'r') as disang:
                disang = disang.readlines()

            # Read restraints from disang file
            r = 0
            for line in disang:
                cols = line.split()
                if (component == 't'):
                    if len(cols) != 0 and (cols[-1] == "#Lig_TR"):
                        natms = len(cols[2].split(','))-1
                        req[win, r] = float(cols[6].replace(",", ""))
                        if natms == 2:
                            rty[r] = 'd'
                            rfc[win, r] = float(cols[12].replace(",", ""))
                        elif natms == 3:
                            rty[r] = 'a'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        elif natms == 4:
                            rty[r] = 't'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        else:
                            sys.exit("not sure about restraint type!")
                        r += 1
                elif (component == 'l' or component == 'c'):
                    if len(cols) != 0 and (cols[-1] == "#Lig_C" or cols[-1] == "#Lig_D"):
                        natms = len(cols[2].split(','))-1
                        req[win, r] = float(cols[6].replace(",", ""))
                        if natms == 2:
                            rty[r] = 'd'
                            rfc[win, r] = float(cols[12].replace(",", ""))
                        elif natms == 3:
                            rty[r] = 'a'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        elif natms == 4:
                            rty[r] = 't'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        else:
                            sys.exit("not sure about restraint type!")
                        r += 1
                elif (component == 'a' or component == 'r'):
                    if len(cols) != 0 and (cols[-1] == "#Rec_C" or cols[-1] == "#Rec_D"):
                        natms = len(cols[2].split(','))-1
                        req[win, r] = float(cols[6].replace(",", ""))
                        if natms == 2:
                            rty[r] = 'd'
                            rfc[win, r] = float(cols[12].replace(",", ""))
                        elif natms == 3:
                            rty[r] = 'a'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        elif natms == 4:
                            rty[r] = 't'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        else:
                            sys.exit("not sure about restraint type!")
                        r += 1
                elif (component == 'm' or component == 'n'):
                    if len(cols) != 0 and (cols[-1] == "#Rec_C" or cols[-1] == "#Rec_D" or cols[-1] == "#Lig_TR" or cols[-1] == "#Lig_C" or cols[-1] == "#Lig_D"
                                        ):
                        natms = len(cols[2].split(','))-1
                        req[win, r] = float(cols[6].replace(",", ""))
                        if natms == 2:
                            rty[r] = 'd'
                            rfc[win, r] = float(cols[12].replace(",", ""))
                        elif natms == 3:
                            rty[r] = 'a'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        elif natms == 4:
                            rty[r] = 't'
                            rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)*(np.pi/180.0)  # Convert to degrees
                        else:
                            sys.exit("not sure about restraint type!")
                        r += 1

        return rfc, req, rty, num_rest

    def _get_data_list(self):
        logger.debug(f"Extracting data for {self.component}...")
        # add additional code to extract restraints from all windows ahead
        
        rfc, req, rty, num_rest = self._extract_restraints_from_windows()
        logger.debug(f"Extracted restraints for {self.component}: {num_rest} restraints found.")

        df_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_all_for_window)(
                                                 win_i=win_i,
                                                 comp_folder=self.comp_folder,
                                                 component=self.component,
                                                 temperature=self.temperature,
                                                 sim_range=self.sim_range,
                                                 truncate=self.detect_equil,
                                                 rfc=rfc,
                                                 req=req,
                                                 rty=rty,
                                                 num_rest=num_rest,
                                                 num_win=len(self.windows)
            )          
                for win_i in range(len(self.windows)))
        with open(f'{self.result_folder}/{self.component}_df_list.pickle', 'wb') as f:
            pickle.dump(df_list, f)

        for df in df_list:
            df.attrs['temperature'] = self.temperature
            df.attrs['energy_unit'] = 'kT'
        
        return df_list
    
    @staticmethod
    def _extract_all_for_window(win_i, comp_folder, component, temperature,
                               sim_range, rfc, req, rty, num_rest, num_win, truncate):
        """
        Return a dataframe with the energy in kT units for the given window.
        As no direct report of the MBAR energy is present in the output,
        we need to extract the data from the simulations.
        """
        kT = 0.0019872041 * temperature
        os.chdir(f'{comp_folder}/{component}{win_i:02d}')

        if sim_range is None:
            n_sims = len(glob.glob(f'mdin-*.nc'))
            sim_range = range(n_sims)
        logger.debug(f"Extracting data for {component}{win_i:02d} with {len(sim_range)} simulations...")

        md_sim_files = []
        for i in sim_range:
            if os.path.exists(f'mdin-{i:02d}.nc'):
                md_sim_files.append(f'mdin-{i:02d}.nc')
            else:
                raise FileNotFoundError(f"Simulation file mdin-{i:02d}.nc not found.")
        
        top = 'full'
        # make sure all nc files exist
        if len(md_sim_files) == 0:
            # the simulation is done probably in order versions
            md_sim_files = ['md01.nc', 'md02.nc', 'md03.nc', 'md04.nc']
            if all(not os.path.exists(md_file) for md_file in md_sim_files):
                raise FileNotFoundError(f"No simulation files found for {component}{win_i:02d}")
        try:
            generate_results_rest(md_sim_files, component, blocks=5, top=top)
        except:
            # try use vac
            top = 'vac'
            generate_results_rest(md_sim_files, component, blocks=5, top=top)

        logger.debug(f"Reading data for {component}{win_i:02d}...")

        # read simulation data

        # Read disang file to get restraints
        restraint_file = 'restraints.dat'
        with open(restraint_file, 'r') as f:
            restdat = f.readlines()
            
        val = np.zeros([len(restdat)-1, num_rest], dtype=float)

        n = 0
        for line in restdat:
            if line[0] != '#' and line[0] != '@':
                cols = line.split()
                for r in range(num_rest):
                    if rty[r] == 't':  # Do phase corrections
                        tmp = float(cols[r+1])
                        if tmp < req[win_i, r]-180.0:
                            val[n, r] = tmp + 360
                        elif tmp > req[win_i, r]+180.0:
                            val[n, r] = tmp - 360
                        else:
                            val[n, r] = tmp
                    else:
                        val[n, r] = float(cols[r+1])
                n += 1

        # get reduced potential
        if component != 'u':  # Attach/Release Restraints
            if rfc[win_i, 0] == 0:
                tmp = np.ones([num_rest], np.float64) * 0.001  # CHECK THIS!! might interfere on protein attach
                u = np.sum(tmp*((val-req[win_i])**2) / kT, axis=1)
            else:
                u = np.sum(rfc[win_i]*((val-req[win_i])**2) / kT, axis=1)
        else:  # Umbrella/Translation
            u = (rfc[win_i, 0]*((val[:, 0]-req[win_i, 0])**2) / kT)
        
        if truncate:
        # get equilibration time from the reduced potential
            with SuppressLoguru():
                t0, g, Neff_max = detect_equilibration(u, nskip=10)
                u = u[t0:]

        else:
            t0 = 0
        Upot = np.zeros([num_win, len(u)], np.float64)

        for win in range(num_win):
            if component != 'u':  # Attach Restraints
                Upot[win] = np.sum(rfc[win]*((val[t0:]-req[win])**2) / kT, axis=1)
            else:  # Umbrella/Translation
                Upot[win] = (rfc[win, 0]*((val[t0:, 0]-req[win, 0])**2) / kT)
            
        win_i_list = np.arange(num_win, dtype=np.float64)
        mbar_time = np.arange(len(u), dtype=np.float64)
        clambda = win_i
        mbar_df = pd.DataFrame(
            Upot,
            index=np.array(win_i_list, dtype=np.float64),
            columns=pd.MultiIndex.from_arrays(
                [mbar_time, np.repeat(clambda, len(mbar_time))],
                names=["time", "lambdas"],
            ),
        ).T
        return mbar_df


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


def generate_results_rest(md_sim_files,
                          comp,
                          blocks=5,
                          top='full'):
    data = []
    
    # Read the 'restraints.in' file
    with open('restraints.in', 'r') as f:
        lines = f.readlines()
    # remove lines contains 'trajin'
    lines = [line for line in lines if 'trajin' not in line]
    # get the line index of parm
    line_index = lines.index([line for line in lines if 'parm' in line][0])
    # replace 'vac.prmtop' with '{top}.prmtop'
    lines[line_index] = lines[line_index].replace('vac.prmtop',
                                                f'../{comp}-1/{top}.prmtop')
    with open('restraints_curr.in', 'w') as f:
        # Write lines up to and including the target line
        f.writelines(lines[:line_index + 1])
        # Append the sorted mdin files
        for mdin_file in md_sim_files:
            f.write(f'trajin {mdin_file}\n')
        # Write the remaining lines
        f.writelines(lines[line_index + 1:])
    # Run cpptraj with logging
    logger.debug('Running cpptraj')
    run_with_log(f"{cpptraj} -i restraints_curr.in > restraints.log 2>&1")
    logger.debug('cpptraj finished')

    with open("restraints.dat", "r") as fin:
        for line in fin:
            if not '#' in line:
                data.append(line)