"""Utilities for inspecting replica-exchange simulations."""

from __future__ import annotations

# Copy from Amber FETools (refactored into a single class)
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

__all__ = ["RemdLog", "plot_trajectory"]

class RemdLog:
    r"""
    Read and analyse AMBER ``remlog`` files.

    The parser reconstructs the replica $\leftrightarrow$ state mapping at each
    exchange step and reports high-level metrics such as average single-pass
    duration and the number of round trips.

    Parameters
    ----------
    inputfile : str
        Path to the ``remlog`` text file produced by AMBER.
    """

    def __init__(self, inputfile: str):
        if not os.path.isfile(inputfile):
            raise FileNotFoundError(f"Input file '{inputfile}' does not exist.")
        self.inputfile: str = inputfile

        # Parsed data
        self.replica_trajectory: Optional[np.ndarray] = None      # shape: (n_replica, n_step+1)
        self.replica_state_count: Optional[np.ndarray] = None     # shape: (n_replica, n_state)
        self.replica_ex_count: Optional[np.ndarray] = None        # shape: (n_replica, n_state-1)
        self.replica_ex_succ: Optional[np.ndarray] = None         # shape: (n_replica, n_state-1)
        self.ARs: Optional[List[float]] = None                    # neighbor acceptance ratios

        # Meta
        self.n_replica: Optional[int] = None
        self.n_step: Optional[int] = None

        self._read_log()

    def _read_log(self) -> None:
        """Parse ``self.inputfile`` and populate cached arrays."""
        (
            self.replica_trajectory,
            self.replica_state_count,
            self.replica_ex_count,
            self.replica_ex_succ,
            self.ARs,
            self.n_replica,
            self.n_step,
        ) = self._read_rem_log()

    def analyze(self) -> Dict[str, float | List[float]]:
        """
        Summarise the replica trajectory.

        Returns
        -------
        dict
            Dictionary with the same keys as :meth:`get_remd_info`.
        """
        return self._remd_analysis(self.replica_trajectory, self.ARs)

    @classmethod
    def get_remd_info(cls, inputfile: str) -> Dict[str, float | List[float]]:
        """
        Convenience helper that parses and analyses a ``remlog`` file.

        Parameters
        ----------
        inputfile : str
            Path to the ``remlog`` text file.

        Returns
        -------
        dict
            Same structure as :meth:`analyze`.
        """
        rl = cls(inputfile)
        rl._read_log()
        return rl.analyze()

    # ---------- Internals ----------

    def _read_rem_log(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], int, int]:
        """
        Parse the on-disk remlog file.

        Returns
        -------
        tuple
            ``(replica_trajectory, replica_state_count, replica_ex_count,
            replica_ex_succ, neighbor_acceptance_ratios, n_replica, n_step)``.
        """
        logger.info("Analyzing remlog file: {}", self.inputfile)

        np.set_printoptions(precision=2, linewidth=150, formatter={"int": "{:2d}".format})

        rep: List[int] = []
        neigh: List[int] = []
        succ: List[str] = []

        try:
            with open(self.inputfile, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.inputfile}' not found.")

        count = 0
        n_replica = 0
        for line in lines:
            count += 1
            if not line or line[0] == "#":
                continue

            # Defensive slicing against short lines
            rep.append(int(line[0:6].strip()))
            neigh.append(int(line[6:12].strip()))

            # Older/newer formats differ where the success char sits
            # Prefer column 66 if it's T/F, else fallback to 91
            ch66 = line[66:67] if len(line) > 66 else ""
            ch91 = line[91:92] if len(line) > 91 else ""
            if ch66 in ("T", "F"):
                succ.append(ch66)
            else:
                succ.append(ch91)

            # Heuristic from original code to estimate number of replicas early
            if count > 200:
                n_replica = max(rep[0:200])

        logger.info("Done reading the remlog.")

        # Final replica/step counts
        n_replica = max(rep[0:200]) if n_replica == 0 else n_replica
        if n_replica <= 0:
            raise ValueError("Failed to infer number of replicas from remlog.")

        total_records = len(rep)
        if total_records % n_replica != 0:
            logger.warning(
                "Total records ({}) not divisible by n_replica ({}). "
                "Rounding down to an integer number of steps.",
                total_records,
                n_replica,
            )
        n_step = total_records // n_replica
        n_state = n_replica

        logger.info("# of Replicas: {}  # of Steps: {}", n_replica, n_step)

        # Parse neighbor acceptance ratios from the tail: last n_replica-1 lines
        # Original code used [-n_replica : -1]
        tail = lines[-n_replica:-1] if len(lines) >= n_replica else []
        ARs = []
        for t in tail:
            bits = t.strip().split()
            if bits:
                try:
                    ARs.append(float(bits[-1]))
                except ValueError:
                    pass  # ignore lines that don't parse cleanly
        # Fallback to empty or keep partial list
        # (Keep behavior close to original—no hard failure)

        # Allocate arrays
        replica_trajectory = np.zeros((n_replica, n_step + 1), np.int64)
        replica_state_count = np.zeros((n_replica, n_state), np.int64)
        replica_ex_count = np.zeros((n_replica, n_state - 1), np.int64)
        replica_ex_succ = np.zeros((n_replica, n_state - 1), np.int64)

        # Initialize each replica to its own state at step 0
        for i in range(n_replica):
            replica_trajectory[i, 0] = i + 1
            replica_state_count[i, i] = 1

        # Build trajectory over steps based on pairwise exchanges
        for m in range(n_step):
            # Carry forward last assignments
            replica_trajectory[:, m + 1] = replica_trajectory[:, m]

            # Odd-even neighbor swaps
            for i in range((m + 1) % 2, n_replica - 1, 2):
                k = m * n_replica + i

                # Find which rows currently hold state i+1 and i+2
                x = np.where(replica_trajectory[:, m + 1] == i + 1)[0]
                y = np.where(replica_trajectory[:, m + 1] == i + 2)[0]

                if x.size > 0:
                    replica_ex_count[x, i] += 1
                    if k < len(succ) and succ[k] == "T":
                        replica_ex_succ[x, i] += 1
                        if y.size > 0:
                            replica_trajectory[y, m + 1] = i + 1
                        replica_trajectory[x, m + 1] = i + 2

            # Update counts of time spent in each state
            idx = replica_trajectory[:, m + 1] - 1  # 0-based states
            for j in range(n_replica):
                replica_state_count[j, idx[j]] += 1

        return (
            replica_trajectory,
            replica_state_count,
            replica_ex_count,
            replica_ex_succ,
            ARs,
            n_replica,
            n_step,
        )

    @staticmethod
    def _remd_analysis(
        replica_trajectory: np.ndarray, ARs: List[float]
    ) -> Dict[str, float | List[float]]:
        """
        Compute REMD round-trip statistics from a replica/state table.

        Parameters
        ----------
        replica_trajectory : numpy.ndarray
            Array of shape ``(n_replica, n_step + 1)`` describing which thermodynamic
            state each replica occupied at every step.
        ARs : list[float]
            Neighbor acceptance ratios parsed from the tail of the remlog.

        Returns
        -------
        dict
            Summary containing the average single-pass length, round trips per
            replica, total round trips, and the provided acceptance ratios.
        """
        n_replica = int(np.size(replica_trajectory, 0))
        n_step = int(np.size(replica_trajectory, 1))

        logger.info("Analyzing trajectory: n_replica={}, n_step={}", n_replica, n_step)

        # Times to go from end-to-end (1 -> N and N -> 1), plus dwell times
        h1n: List[int] = []
        hn1: List[int] = []
        k1n: List[int] = []
        kn1: List[int] = []
        trip_count_1n = [0] * n_replica
        trip_count_n1 = [0] * n_replica

        for i in range(n_replica):
            first_step_at_1 = -1
            first_step_at_n = -1
            last_step_at_1 = -1
            last_step_at_n = -1
            at_1 = 0
            at_n = 0

            for j in range(n_step):
                state = replica_trajectory[i, j]
                if state == 1:
                    last_step_at_1 = j
                    if at_1 == 0:
                        at_1 = 1
                        at_n = 0
                        first_step_at_1 = j
                    if first_step_at_n >= 0:
                        hn1.append(j - first_step_at_n)
                        first_step_at_n = -1
                        trip_count_n1[i] += 1
                    if last_step_at_n >= 0:
                        kn1.append(j - last_step_at_n)
                        last_step_at_n = -1

                if state == n_replica:
                    last_step_at_n = j
                    if at_n == 0:
                        at_n = 1
                        at_1 = 0
                        first_step_at_n = j
                        if first_step_at_1 >= 0:
                            h1n.append(j - first_step_at_1)
                            first_step_at_1 = -1
                            trip_count_1n[i] += 1
                        if last_step_at_1 >= 0:
                            k1n.append(j - last_step_at_1)
                            last_step_at_1 = -1

        output_data: Dict[str, float | List[float]] = {}

        if len(h1n) == 0 or len(hn1) == 0:
            logger.warning("No single pass found (no 1↔N transitions detected).")
            output_data["Average single pass steps:"] = 1.0e8
            output_data["Round trips per replica:"] = 0.0
            output_data["Total round trips:"] = 0.0
            output_data["neighbor_acceptance_ratio"] = ARs
            return output_data

        hh = h1n + hn1
        mean_value = float(np.mean(hh))
        output_data["Average single pass steps:"] = mean_value
        output_data["Round trips per replica:"] = float(len(hh) / 2 / n_replica)
        output_data["Total round trips:"] = float(len(hh) / 2)
        output_data["neighbor_acceptance_ratio"] = ARs

        return output_data


def plot_trajectory(
    replica_trajectory,
    figsize=(10, 6),
    alpha=0.8,
    linewidth=1.5,
    subplot=False,
    ncols=4,
):
    """
    Visualise the replica walk through thermodynamic states.

    Parameters
    ----------
    replica_trajectory : numpy.ndarray
        Array of shape ``(n_replica, n_step + 1)`` containing state indices.
    figsize : tuple, optional
        Base figure size. When ``subplot=True`` the width/height apply to each
        panel instead of the aggregate.
    alpha : float, optional
        Line transparency used for individual replica traces.
    linewidth : float, optional
        Width of trajectory lines.
    subplot : bool, optional
        When ``True``, render one subplot per replica; otherwise plot all
        replicas on a shared axis.
    ncols : int, optional
        Number of subplot columns when ``subplot=True``.
    """
    import matplotlib.pyplot as plt  # deferred import to avoid heavy backends

    n_replica, n_step_plus1 = replica_trajectory.shape
    steps = np.arange(n_step_plus1)

    cmap = plt.cm.rainbow
    colors = [cmap(i / n_replica) for i in range(n_replica)]

    if not subplot:
        # --- Single axis plot ---
        plt.figure(figsize=figsize)
        for i in range(n_replica):
            plt.plot(
                steps,
                replica_trajectory[i],
                color=colors[i],
                alpha=alpha,
                linewidth=linewidth,
                label=f"Replica {i+1}" if n_replica <= 15 else None,
            )
        plt.xlabel("Step")
        plt.ylabel("State index")
        plt.title("Replica Trajectories")
        if n_replica <= 15:
            plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()

    else:
        # --- Subplot mode ---
        nrows = int(np.ceil(n_replica / ncols))
        fig_width = figsize[0] * ncols
        fig_height = figsize[1] * nrows
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
        axes = axes.flatten()

        for i in range(n_replica):
            ax = axes[i]
            ax.plot(
                steps,
                replica_trajectory[i],
                color=colors[i],
                alpha=alpha,
                linewidth=linewidth,
            )
            ax.set_title(f"Replica {i+1}", fontsize=9)
            ax.tick_params(labelsize=8)

        # Hide unused subplots if n_replica not multiple of ncols
        for j in range(n_replica, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Replica Trajectories", fontsize=12)
        fig.text(0.5, 0.04, "Step", ha="center")
        fig.text(0.04, 0.5, "State index", va="center", rotation="vertical")
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.show()
