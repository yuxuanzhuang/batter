from __future__ import annotations

import os
import re
import sys
import glob
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import logging
from loguru import logger
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
    plot_block_average,
)

import seaborn as sns
import matplotlib.pyplot as plt

from batter.utils import run_with_log, cpptraj
from batter.analysis.utils import exclude_outliers


COMPONENTS_DICT = {
    "rest": ["a", "l", "t", "c", "r", "m", "n"],
    "dd":   ["e", "v", "f", "w", "x", "o", "s", "z", "y"],
}

# sign that determines direction of contribution to total FE
COMPONENT_DIRECTION_DICT = {
    "m": -1,
    "n": +1,
    "e": -1,
    "v": -1,
    "o": -1,
    "z": -1,
    "y": +1,
    "Boresch": -1,
}

class SilenceAlchemlybOnly:
    def __enter__(self):
        logger.disable("alchemlyb")
        logger.disable("alchemlyb.parsing")
        logger.disable("alchemlyb.parsing.amber")
    def __exit__(self, *args):
        logger.enable("alchemlyb")
        logger.enable("alchemlyb.parsing")
        logger.enable("alchemlyb.parsing.amber")


class FEAnalysisBase(ABC):
    """
    Abstract base class for analysis of each component.
    """
    def __init__(self):
        self.results = {
            "fe": None,               # scalar in 'energy_unit'
            "fe_error": None,         # scalar (same unit)
            "convergence": {},        # dict of dataframes/arrays
            "fe_timeseries": None,    # Nx2 array: [FE, FE_err] across progress fractions
        }

    @abstractmethod
    def run_analysis(self): ...
    @abstractmethod
    def plot_convergence(self, ax=None, **kwargs): ...

    @property
    def fe(self): return self.results["fe"]
    @property
    def fe_error(self): return self.results["fe_error"]
    @property
    def convergence(self): return self.results["convergence"]
    @property
    def fe_timeseries(self): return self.results["fe_timeseries"]

    def dump(self, filename="results.json"):
        """Store results to JSON (omit heavy convergence tables)."""
        fe = float(self.fe) if self.fe is not None else None
        fe_err = float(self.fe_error) if self.fe_error is not None else None
        fets = self.fe_timeseries
        fets_list = fets.tolist() if isinstance(fets, np.ndarray) else fets
        with open(filename, "w") as f:
            json.dump(
                {"fe": fe, "fe_error": fe_err, "fe_timeseries": fets_list},
                f, indent=2
            )


class MBARAnalysis(FEAnalysisBase):
    def __init__(
        self,
        lig_folder: str,
        component: str,
        windows: List[int],
        temperature: float,
        energy_unit: str = "kcal/mol",
        sim_range: Optional[Tuple[int, int]] = None,
        detect_equil: bool = True,
        n_bootstraps: int = 0,
        n_jobs: int = 6,
        load: bool = False,
    ):
        super().__init__()
        self.lig_folder = lig_folder
        self.result_folder = f"{self.lig_folder}/Results"
        os.makedirs(self.result_folder, exist_ok=True)

        comp_folder = f"{self.lig_folder}/{component}"
        if not os.path.isdir(comp_folder):
            raise ValueError(f"Component folder not found: {comp_folder}")
        self.comp_folder = comp_folder
        self.component = component

        self.windows = windows
        self.temperature = float(temperature)
        if energy_unit not in ("kcal/mol", "kJ/mol", "kT"):
            raise ValueError("energy_unit must be 'kcal/mol', 'kJ/mol', or 'kT'")
        self.energy_unit = energy_unit
        self.kT = 0.0019872041 * self.temperature

        if sim_range is not None:
            if not (isinstance(sim_range, (list, tuple)) and len(sim_range) == 2):
                raise ValueError("sim_range must be a 2-tuple (start, end)")
        self.sim_range = sim_range

        self.detect_equil = bool(detect_equil)
        self.n_bootstraps = int(n_bootstraps)
        self.n_jobs = int(n_jobs)
        self.load = bool(load)
        self._data_initialized = False

    # public props used after get_mbar_data()
    @property
    def u_df(self) -> pd.DataFrame: return self._u_df
    @property
    def data_list(self) -> List[pd.DataFrame]: return self._data_list

    def get_mbar_data(self) -> None:
        pkl = f"{self.result_folder}/{self.component}_df_list.pickle"
        if self.load and os.path.exists(pkl):
            with open(pkl, "rb") as f:
                df_list = pickle.load(f)
        else:
            df_list = self._get_data_list()

        self._data_list = df_list
        self._u_df = pd.concat(df_list)
        self.timeseries = [df.index.get_level_values("time").values for df in df_list]
        self._data_initialized = True

    def run_analysis(self) -> None:
        if not self._data_initialized:
            self.get_mbar_data()

        mbar = MBAR(n_bootstraps=self.n_bootstraps)
        mbar.fit(self.u_df)
        self._mbar = mbar

        # accumulate error in kT space then convert
        err_kT = np.sqrt(sum(mbar.d_delta_f_.iloc[i, i + 1] ** 2
                             for i in range(len(mbar.d_delta_f_) - 1)))
        delta_kT = mbar.delta_f_.iloc[0, -1]

        if self.energy_unit == "kcal/mol":
            self.results["fe"] = float(delta_kT * self.kT)
            self.results["fe_error"] = float(err_kT * self.kT)
        elif self.energy_unit == "kJ/mol":
            self.results["fe"] = float(delta_kT * self.kT * 4.184)
            self.results["fe_error"] = float(err_kT * self.kT * 4.184)
        else:  # kT
            self.results["fe"] = float(delta_kT)
            self.results["fe_error"] = float(err_kT)

        # Convergence summaries
        with SilenceAlchemlybOnly():
            tc = forward_backward_convergence(self.data_list, "MBAR", error_tol=100, method="default")
            self.results["convergence"]["time_convergence"] = tc

            # forward/backward times (MultiIndex) + FE arrays (in kcal/mol)
            forward_FE = tc.Forward.values * self.kT
            forward_FE_err = tc.Forward_Error.values * self.kT
            backward_FE = tc.Backward.values * self.kT
            backward_FE_err = tc.Backward_Error.values * self.kT

            # fe_timeseries: N x 2 array (value, stderr)
            self.results["fe_timeseries"] = np.column_stack([forward_FE, forward_FE_err])

            # block average (10 blocks)
            ba = block_average(self.data_list, estimator="MBAR", num=10, method="default")
            self.results["convergence"]["block_convergence"] = ba

            block_FE = ba.FE.values * self.kT
            block_FE_err = ba.FE_Error.values * self.kT
            # pack in a simple dataframe with sequential fraction labels
            self.results["convergence"]["block_timeseries"] = pd.DataFrame(
                {"FE": block_FE, "FE_Error": block_FE_err},
                index=np.linspace(0.1, 1.0, len(block_FE))
            )

            self.results["convergence"]["overlap_matrix"] = mbar.overlap_matrix
            self.results["convergence"]["mbar"] = mbar

        # persist
        with open(f"{self.result_folder}/{self.component}_results.pickle", "wb") as f:
            pickle.dump(self.results, f)
        self.dump(f"{self.result_folder}/{self.component}_results.json")

    @staticmethod
    def _extract_all_for_window(
        win_i: int,
        comp_folder: str,
        component: str,
        temperature: float,
        sim_range: Optional[Tuple[int, int]],
        truncate: bool,
    ) -> pd.DataFrame:
        """
        Extract u_nk for one window using alchemlyb's Amber parser.
        Returns a kT-referenced dataframe (reduced potentials).
        """
        logger.remove()
        win_dir = f"{comp_folder}/{component}{win_i:02d}"
        n_sims = len(glob.glob(f"{win_dir}/mdin-*.out"))
        all_range = range(n_sims)

        # Collect mdin-XX.out, fallback to md-03/04.out if older layout
        mdouts: List[str] = []
        s0 = sim_range[0] if sim_range is not None else 0
        e0 = sim_range[1] if sim_range is not None else n_sims
        if s0 > n_sims:
            raise ValueError(f"sim_range start {s0} > available sims {n_sims}")
        if e0 > n_sims:
            logger.warning(f"sim_range end {e0} > available sims {n_sims}")

        for i in all_range[s0:e0]:
            path = f"{win_dir}/mdin-{i:02d}.out"
            if os.path.exists(path):
                mdouts.append(path)

        if not mdouts:
            candidates = [f"{win_dir}/md-03.out", f"{win_dir}/md-04.out"]
            mdouts = [c for c in candidates if os.path.exists(c)]
            if not mdouts:
                raise FileNotFoundError(f"No Amber out files in {win_dir}")

        dfs = []
        with SilenceAlchemlybOnly():
            for fn in mdouts:
                df_part = extract_u_nk(fn, T=temperature, reduced=False, raise_error=False)
                dfs.append(df_part)

        df = pd.concat(dfs)

        # detect_equilibration on the reference column of this window
        if truncate and df.shape[1] > win_i:
            with SilenceAlchemlybOnly():
                t0, _, _ = detect_equilibration(df.iloc[:, win_i], nskip=10)
            # time is level 0 of the MultiIndex
            df = df[df.index.get_level_values(0) > t0]

        # subtract reference (this window) to yield reduced potentials
        ref = df.iloc[:, win_i]
        df = df.subtract(ref, axis=0)

        # Mixed precision spikes guard
        df = exclude_outliers(df, iclam=win_i)
        return df

    def _get_data_list(self) -> List[pd.DataFrame]:
        df_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_all_for_window)(
                win_i=win_i,
                comp_folder=self.comp_folder,
                component=self.component,
                temperature=self.temperature,
                sim_range=self.sim_range,
                truncate=self.detect_equil,
            )
            for win_i in range(len(self.windows))
        )
        for df in df_list:
            df.attrs["temperature"] = self.temperature
            df.attrs["energy_unit"] = "kT"

        with open(f"{self.result_folder}/{self.component}_df_list.pickle", "wb") as f:
            pickle.dump(df_list, f)
        return df_list

    def plot_time_convergence(self, ax=None, **kwargs):
        df = self.results["convergence"]["time_convergence"]
        return plot_convergence(df, ax=ax, **kwargs)

    def plot_overlap_matrix(self, ax=None, **kwargs):
        return plot_mbar_overlap_matrix(self.results["convergence"]["overlap_matrix"], ax=ax, **kwargs)

    def plot_block_convergence(self, ax=None, **kwargs):
        df = self.results["convergence"]["block_convergence"]
        return plot_block_average(df, ax=ax, **kwargs)

    def plot_convergence(self, save_path: Optional[str] = None, title: Optional[str] = None):
        fig, axes = plt.subplot_mosaic([["A", "A", "B"], ["C", "C", "C"]], figsize=(24, 14))
        self.plot_time_convergence(ax=axes["A"], units=self.energy_unit, final_error=0.6)
        self.plot_overlap_matrix(ax=axes["B"])
        self.plot_block_convergence(ax=axes["C"], units=self.energy_unit, final_error=0.6)
        axes["A"].set_title("Time Convergence", fontsize=10)
        axes["B"].set_title("Overlap Matrix", fontsize=10)
        axes["C"].set_title("Block Convergence", fontsize=10)
        plt.tight_layout()
        if title:
            plt.suptitle(title, y=1.02)
        if save_path:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()


class RESTMBARAnalysis(MBARAnalysis):
    def _extract_restraints_from_windows(self):
        num_win = len(self.windows)
        component = self.component
        disang_file = f"{self.comp_folder}/{component}00/disang.rest"
        with open(disang_file, "r") as f:
            disang = f.readlines()

        num_rest = 0
        for line in disang:
            cols = line.split()
            if not cols:
                continue
            tag = cols[-1]
            if component == "t" and tag == "#Lig_TR":
                num_rest += 1
            elif component in ("l", "c") and tag in ("#Lig_C", "#Lig_D"):
                num_rest += 1
            elif component in ("a", "r") and tag in ("#Rec_C", "#Rec_D"):
                num_rest += 1
            elif component in ("m", "n") and tag in ("#Rec_C", "#Rec_D", "#Lig_TR", "#Lig_C", "#Lig_D"):
                num_rest += 1

        rty = ["d"] * num_rest
        rfc = np.zeros([num_win, num_rest], dtype=float)
        req = np.zeros([num_win, num_rest], dtype=float)

        for win in range(num_win):
            dpath = f"{self.comp_folder}/{component}{win:02d}/disang.rest"
            with open(dpath, "r") as fh:
                lines = fh.readlines()
            r = 0
            for line in lines:
                cols = line.split()
                if not cols:
                    continue
                tag = cols[-1]
                def _natms() -> int: return len(cols[2].split(",")) - 1
                if component == "t" and tag == "#Lig_TR":
                    req[win, r] = float(cols[6].replace(",", ""))
                    nat = _natms()
                    if nat == 2:
                        rty[r] = "d"; rfc[win, r] = float(cols[12].replace(",", ""))
                    elif nat == 3:
                        rty[r] = "a"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    elif nat == 4:
                        rty[r] = "t"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    else:
                        raise ValueError("Unknown restraint natoms")
                    r += 1
                elif component in ("l", "c") and tag in ("#Lig_C", "#Lig_D"):
                    req[win, r] = float(cols[6].replace(",", ""))
                    nat = _natms()
                    if nat == 2:
                        rty[r] = "d"; rfc[win, r] = float(cols[12].replace(",", ""))
                    elif nat == 3:
                        rty[r] = "a"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    elif nat == 4:
                        rty[r] = "t"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    else:
                        raise ValueError("Unknown restraint natoms")
                    r += 1
                elif component in ("a", "r") and tag in ("#Rec_C", "#Rec_D"):
                    req[win, r] = float(cols[6].replace(",", ""))
                    nat = _natms()
                    if nat == 2:
                        rty[r] = "d"; rfc[win, r] = float(cols[12].replace(",", ""))
                    elif nat == 3:
                        rty[r] = "a"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    elif nat == 4:
                        rty[r] = "t"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    else:
                        raise ValueError("Unknown restraint natoms")
                    r += 1
                elif component in ("m", "n") and tag in ("#Rec_C", "#Rec_D", "#Lig_TR", "#Lig_C", "#Lig_D"):
                    req[win, r] = float(cols[6].replace(",", ""))
                    nat = _natms()
                    if nat == 2:
                        rty[r] = "d"; rfc[win, r] = float(cols[12].replace(",", ""))
                    elif nat == 3:
                        rty[r] = "a"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    elif nat == 4:
                        rty[r] = "t"; rfc[win, r] = float(cols[12].replace(",", ""))*(np.pi/180.0)**2
                    else:
                        raise ValueError("Unknown restraint natoms")
                    r += 1

        return rfc, req, rty, num_rest

    def _get_data_list(self) -> List[pd.DataFrame]:
        rfc, req, rty, num_rest = self._extract_restraints_from_windows()

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
                num_win=len(self.windows),
            )
            for win_i in range(len(self.windows))
        )

        with open(f"{self.result_folder}/{self.component}_df_list.pickle", "wb") as f:
            pickle.dump(df_list, f)

        for df in df_list:
            df.attrs["temperature"] = self.temperature
            df.attrs["energy_unit"] = "kT"
        return df_list

    @staticmethod
    def _extract_all_for_window(
        win_i: int,
        comp_folder: str,
        component: str,
        temperature: float,
        sim_range: Optional[Tuple[int, int]],
        rfc: np.ndarray,
        req: np.ndarray,
        rty: List[str],
        num_rest: int,
        num_win: int,
        truncate: bool,
    ) -> pd.DataFrame:
        """Compute reduced potentials for REST components from restraint traces."""
        logger.remove()
        kT = 0.0019872041 * temperature
        win_dir = Path(f"{comp_folder}/{component}{win_i:02d}")
        cwd0 = Path.cwd()
        try:
            os.chdir(win_dir)

            # enumerate mdin-XX.nc (or fallback md01.nc..)
            nc_list: List[str] = []
            nsims = len(glob.glob("mdin-*.nc"))
            s0 = sim_range[0] if sim_range is not None else 0
            e0 = sim_range[1] if sim_range is not None else nsims
            if s0 > nsims:
                raise ValueError(f"sim_range start {s0} > n_sims {nsims}")
            if e0 > nsims:
                logger.warning(f"sim_range end {e0} > n_sims {nsims}")

            for i in range(s0, e0):
                fn = f"mdin-{i:02d}.nc"
                if os.path.exists(fn):
                    nc_list.append(fn)

            if not nc_list:
                fallback = ["md01.nc", "md02.nc", "md03.nc", "md04.nc"]
                nc_list = [f for f in fallback if os.path.exists(f)]
                if not nc_list:
                    raise FileNotFoundError("No NetCDF trajs for REST window")

            # generate restraint traces via cpptraj using current topology choice
            def _gen(top_choice: str):
                generate_results_rest(nc_list, component, blocks=5, top=top_choice)

            try:
                _gen("full")
            except Exception:
                _gen("vac")

            with open("restraints.dat", "r") as fin:
                lines = [ln for ln in fin if (ln and ln[0] not in "#@")]
            val = np.zeros((len(lines), num_rest), dtype=float)
            for n, line in enumerate(lines):
                cols = line.split()
                for r in range(num_rest):
                    if rty[r] == "t":
                        tmp = float(cols[r + 1])
                        if tmp < req[win_i, r] - 180.0:  tmp += 360.0
                        elif tmp > req[win_i, r] + 180.0: tmp -= 360.0
                        val[n, r] = tmp
                    else:
                        val[n, r] = float(cols[r + 1])

            # reduced potential at this window
            if component != "u":
                if rfc[win_i, 0] == 0:  # guard tiny zeros
                    tmp = np.ones((num_rest,), np.float64) * 1e-3
                    u = np.sum(tmp * (val - req[win_i]) ** 2 / kT, axis=1)
                else:
                    u = np.sum(rfc[win_i] * (val - req[win_i]) ** 2 / kT, axis=1)
            else:
                u = (rfc[win_i, 0] * (val[:, 0] - req[win_i, 0]) ** 2) / kT

            t0 = 0
            if truncate:
                with SilenceAlchemlybOnly():
                    t0, _, _ = detect_equilibration(u, nskip=10)
                u = u[t0:]
                val = val[t0:]

            Upot = np.zeros((num_win, len(u)), np.float64)
            for w in range(num_win):
                if component != "u":
                    Upot[w] = np.sum(rfc[w] * (val - req[w]) ** 2 / kT, axis=1)
                else:
                    Upot[w] = (rfc[w, 0] * (val[:, 0] - req[w, 0]) ** 2) / kT

            # Pack like alchemlyb (time,lambdas) MultiIndex
            win_i_list = np.arange(num_win, dtype=np.float64)
            mbar_time = np.arange(len(u), dtype=np.float64)
            clambda = float(win_i)

            mbar_df = pd.DataFrame(
                Upot,
                index=np.array(win_i_list, dtype=np.float64),
                columns=pd.MultiIndex.from_arrays(
                    [mbar_time, np.repeat(clambda, len(mbar_time))],
                    names=["time", "lambdas"],
                ),
            ).T
            return mbar_df
        finally:
            os.chdir(cwd0)


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
        def _extract_r2_val(line: str) -> float:
            m = re.search(r"\br2=\s*([+-]?\d+(?:\.\d+)?)", line)
            if not m:
                raise ValueError(f"Couldn't find r2= value in line: {line}")
            return float(m.group(1))

        # Read disang file to get anchor atoms
        with open(self.disangfile, 'r') as f_in:        
            lines = [line.rstrip() for line in f_in]

            tr_lines = list(line for line in lines if '#Lig_TR' in line)
            r0    = _extract_r2_val(tr_lines[0])  # P1–L1 distance (target at r2)
            a1_0  = _extract_r2_val(tr_lines[1])  # P2–P1–L1 angle
            t1_0  = _extract_r2_val(tr_lines[2])  # P3–P2–P1–L1 dihedral
            a2_0  = _extract_r2_val(tr_lines[3])  # P1–L1–L2 angle
            t2_0  = _extract_r2_val(tr_lines[4])  # P2–P1–L1–L2 dihedral
            t3_0  = _extract_r2_val(tr_lines[5])  # P1–L1–L2–L3 dihedral
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


def generate_results_rest(md_sim_files: List[str], comp: str, blocks: int = 5, top: str = "full") -> None:
    """
    Build a cpptraj input on the fly using 'restraints.in' template in cwd,
    swapping the topology to ../{comp}-1/{top}.prmtop and appending trajins.
    """
    with open("restraints.in", "r") as f:
        lines = f.readlines()

    # drop any existing trajin lines
    lines = [ln for ln in lines if "trajin" not in ln]

    # replace the parm line
    parm_idx = None
    for i, ln in enumerate(lines):
        if "parm " in ln:
            parm_idx = i
            break
    if parm_idx is None:
        raise ValueError("restraints.in missing a 'parm' line.")

    lines[parm_idx] = re.sub(r"parm\s+(\S+)",
                             f"parm ../{comp}-1/{top}.prmtop",
                             lines[parm_idx])

    with open("restraints_curr.in", "w") as f:
        f.writelines(lines[: parm_idx + 1])
        for mdin in md_sim_files:
            f.write(f"trajin {mdin}\n")
        f.writelines(lines[parm_idx + 1:])

    rc = run_with_log(f"{cpptraj} -i restraints_curr.in > restraints.log 2>&1")
    if rc != 0:
        raise RuntimeError("cpptraj failed; see restraints.log")

# ---- lig wrapper ------------------------------------------------------------

def analyze_lig_task(
    fe_folder: str,
    lig: str,
    components: List[str],
    rest: Tuple[float, float, float, float, float],
    temperature: float,
    water_model: str,
    component_windows_dict: Dict[str, List[int]],
    rocklin_correction: bool = False,
    sim_range: Optional[Tuple[int, int]] = None,
    raise_on_error: bool = True,
    mol: str = "LIG",
    n_workers: int = 4,
):
    """
    Analyze one lig under fe_folder/lig for the requested components.
    """
    lig_path = f"{fe_folder}/{lig}"
    os.makedirs(f"{lig_path}/Results", exist_ok=True)

    results_entries: List[str] = []
    LEN_FE_TIMESERIES = 10

    try:
        fe_values: List[float] = []
        fe_stds: List[float] = []
        fe_timeseries: Dict[str, np.ndarray] = {}

        # Analytical Boresch (if present)
        if "v" in components:
            boresch_file = f"{lig_path}/v/v-1/disang.rest"
        elif "o" in components:
            boresch_file = f"{lig_path}/o/o-1/disang.rest"
        elif "z" in components:
            boresch_file = f"{lig_path}/z/z-1/disang.rest"
        else:
            boresch_file = None

        if boresch_file:
            k_r, k_a = rest[2], rest[3]
            bor = BoreschAnalysis(disangfile=boresch_file, k_r=k_r, k_a=k_a, temperature=temperature)
            bor.run_analysis()
            fe_values.append(COMPONENT_DIRECTION_DICT["Boresch"] * bor.results["fe"])
            fe_stds.append(bor.results["fe_error"])
            fe_timeseries["Boresch"] = np.asarray([bor.results["fe"], 0.0])
            results_entries.append(f"Boresch\t{COMPONENT_DIRECTION_DICT['Boresch'] * bor.results['fe']:.2f}\t{bor.results['fe_error']:.2f}")

        for comp in components:
            comp_path = f"{lig_path}/{comp}"
            windows = component_windows_dict[comp]

            # skip 'n' if no conformational restraints are applied
            if comp == "n" and rest[1] == 0 and rest[4] == 0:
                logger.debug("Skipping 'n' (no conformational restraints).")
                continue

            if comp in COMPONENTS_DICT["dd"]:
                ana = MBARAnalysis(
                    lig_folder=lig_path,
                    component=comp,
                    windows=windows,
                    temperature=temperature,
                    sim_range=sim_range,
                    load=False,
                    n_jobs=n_workers,
                )
                ana.run_analysis()
                ana.plot_convergence(
                    save_path=f"{lig_path}/Results/{comp}_convergence.png",
                    title=f"Convergence for {comp} {mol}",
                )
                fe_values.append(COMPONENT_DIRECTION_DICT[comp] * ana.results["fe"])
                fe_stds.append(ana.results["fe_error"])
                fe_timeseries[comp] = ana.results["fe_timeseries"]
                results_entries.append(f"{comp}\t{COMPONENT_DIRECTION_DICT[comp]*ana.results['fe']:.2f}\t{ana.results['fe_error']:.2f}")

            elif comp in COMPONENTS_DICT["rest"]:
                ana = RESTMBARAnalysis(
                    lig_folder=lig_path,
                    component=comp,
                    windows=windows,
                    temperature=temperature,
                    sim_range=sim_range,
                    load=False,
                    n_jobs=n_workers,
                )
                ana.run_analysis()
                ana.plot_convergence(
                    save_path=f"{lig_path}/Results/{comp}_convergence.png",
                    title=f"Convergence for {comp} {mol}",
                )
                fe_values.append(COMPONENT_DIRECTION_DICT[comp] * ana.results["fe"])
                fe_stds.append(ana.results["fe_error"])
                fe_timeseries[comp] = ana.results["fe_timeseries"]
                results_entries.append(f"{comp}\t{COMPONENT_DIRECTION_DICT[comp]*ana.results['fe']:.2f}\t{ana.results['fe_error']:.2f}")

        # total FE and timeseries (sum in quadrature for std)
        fe_value = float(np.sum(fe_values)) if fe_values else float("nan")
        fe_std = float(np.sqrt(np.sum(np.array(fe_stds) ** 2))) if fe_stds else float("nan")

        fe_ts_val = np.zeros(LEN_FE_TIMESERIES, dtype=float)
        fe_ts_err2 = np.zeros(LEN_FE_TIMESERIES, dtype=float)
        for comp, ts in fe_timeseries.items():
            direction = COMPONENT_DIRECTION_DICT.get(comp, +1)
            if ts.ndim == 1:
                fe_ts_val += float(ts[0]) * direction
            else:
                # assume Nx2 (value, stderr)
                n = min(LEN_FE_TIMESERIES, ts.shape[0])
                fe_ts_val[:n] += ts[:n, 0] * direction
                fe_ts_err2[:n] += ts[:n, 1] ** 2
        fe_ts_err = np.sqrt(fe_ts_err2)

    except Exception as e:
        logger.error(f"Error during FE analysis for {lig}: {e}")
        if raise_on_error:
            raise
        fe_value = float("nan")
        fe_std = float("nan")
        fe_ts_val = np.zeros(LEN_FE_TIMESERIES) * np.nan
        fe_ts_err = np.zeros(LEN_FE_TIMESERIES) * np.nan

    # Optional Rocklin correction (component 'y')
    if rocklin_correction == "yes":
        from .rocklin import run_rocklin_correction  # local import
        if "y" not in components:
            raise ValueError("Rocklin correction requires component 'y'.")
        universe = mda.Universe(
            f"{lig_path}/y/y-1/full.prmtop",
            f"{lig_path}/y/y-1/eq_output.pdb",
        )
        box = universe.dimensions[:3]
        lig_ag = universe.select_atoms(f"resname {mol}")
        if len(lig_ag) == 0:
            raise ValueError(f"No ligand atoms found for Rocklin correction with resname {mol}")
        lig_netq = int(round(lig_ag.total_charge()))
        other_ag = universe.atoms - lig_ag
        other_netq = int(round(other_ag.total_charge()))
        if lig_netq != 0:
            corr = run_rocklin_correction(
                universe=universe,
                mol_name=mol,
                box=box,
                lig_netq=lig_netq,
                other_netq=other_netq,
                temp=temperature,
                water_model=water_model,
            )
            fe_value += corr
            results_entries.append(f"Rocklin\t{corr:.2f}\t0.00")
            fe_ts_val += corr

    results_entries.append(f"Total\t{fe_value:.2f}\t{fe_std:.2f}")
    with open(f"{lig_path}/Results/Results.dat", "w") as f:
        f.write("\n".join(results_entries))

    with open(f"{lig_path}/Results/fe_timeseries.json", "w") as f:
        json.dump(
            {"fe_value": fe_ts_val.tolist(), "fe_std": fe_ts_err.tolist()},
            f, indent=2
        )

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, LEN_FE_TIMESERIES + 1) / LEN_FE_TIMESERIES * 100.0
    ax.errorbar(x, fe_ts_val, yerr=fe_ts_err, fmt="-o", capsize=4)
    ax.axhline(fe_value, linestyle="--", label="FE value (±1 kcal/mol)")
    ax.fill_between(x, fe_value - 1.0, fe_value + 1.0, alpha=0.2)
    ax.set_xlabel("Simulation Progress (%)")
    ax.set_ylabel("Free Energy (kcal/mol)")
    ax.set_title(f"Free Energy Timeseries for {mol}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f"{lig_path}/Results/fe_timeseries.png", dpi=200)
    plt.close(fig)
    return