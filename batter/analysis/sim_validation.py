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
from typing import Any, Sequence

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

STABLE_BORESCH_DISTANCE_SCHEMA_VERSION = 2
_VMD_FIRST_ANCHOR_ANGLE_TARGET = 90.0
_VMD_FIRST_ANCHOR_ANGLE_TOLERANCE = 70.0


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
    def __init__(
        self,
        universe,
        ligand=None,
        directory: str | Path = ".",
        protein_anchor_masks: list[str] | tuple[str, str, str] | None = None,
    ):
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
        self.protein_anchor_masks = (
            [m.strip() for m in protein_anchor_masks if isinstance(m, str) and m.strip()]
            if protein_anchor_masks is not None
            else []
        )
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
        if ligand_ag.n_atoms == 0:
            raise ValueError(f'No ligand atoms found for resname {self.ligand!r}.')

        ligand_heavy = self._heavy_atoms_or_all(ligand_ag)
        binding_site_ag = self._get_binding_site_atoms(ligand_heavy)
        if binding_site_ag is None or binding_site_ag.n_atoms == 0:
            logger.warning(
                'Could not resolve initial binding-site atoms; falling back to protein anchors.'
            )
            binding_site_ag = self._get_protein_anchor_atoms()
        if binding_site_ag is None or binding_site_ag.n_atoms == 0:
            raise ValueError(
                f'Could not resolve binding-site atoms for ligand_bs in {self.workdir}.'
            )

        # Distance metric: minimum distance from any ligand heavy atom to the
        # protein atoms that formed the initial binding-site pocket.
        distances = []
        for _ in self.universe.trajectory:
            dist_mat = distance_array(
                ligand_heavy.positions,
                binding_site_ag.positions,
                box=self.universe.dimensions,
            )
            distances.append(float(np.min(dist_mat)))

        self.results['ligand_bs'] = np.asarray(distances)

    @staticmethod
    def _heavy_atoms_or_all(atom_group: AtomGroup) -> AtomGroup:
        heavy = atom_group.select_atoms('not name H*')
        return heavy if heavy.n_atoms else atom_group

    def _get_binding_site_atoms(self, ligand_ag: AtomGroup, cutoff: float = 6.0):
        protein_ag = self.universe.select_atoms('protein')
        if protein_ag.n_atoms == 0:
            return None
        protein_ag = self._heavy_atoms_or_all(protein_ag)

        first_frame = self.universe.trajectory.frame
        try:
            self.universe.trajectory[0]
            dist_mat = distance_array(
                ligand_ag.positions,
                protein_ag.positions,
                box=self.universe.dimensions,
            )
            site_atom_indices = np.where(np.any(dist_mat <= float(cutoff), axis=0))[0]
            binding_site_ag = protein_ag[site_atom_indices]
        finally:
            self.universe.trajectory[first_frame]
        return binding_site_ag

    @staticmethod
    def _anchor_mask_to_selection(mask: str) -> str:
        token = mask.strip()
        if not token.startswith(':') or '@' not in token:
            raise ValueError(f'Invalid anchor mask: {mask!r}')
        resid, atom = token[1:].split('@', 1)
        resid = resid.strip()
        atom = atom.strip()
        if not resid or not atom:
            raise ValueError(f'Invalid anchor mask: {mask!r}')
        return f'protein and resid {resid} and name {atom}'

    def _get_protein_anchor_atoms(self):
        if len(self.protein_anchor_masks) != 3:
            logger.warning(
                f'Expected 3 protein anchors from YAML/sim config, got {len(self.protein_anchor_masks)}.'
            )
            return None

        atoms = []
        for mask in self.protein_anchor_masks:
            try:
                sel = self._anchor_mask_to_selection(mask)
            except ValueError as exc:
                logger.warning(f'Invalid anchor entry {mask!r}: {exc}')
                return None
            ag = self.universe.select_atoms(sel)
            if ag.n_atoms != 1:
                logger.warning(
                    f'Anchor selection {sel!r} matched {ag.n_atoms} atoms (expected 1).'
                )
                return None
            atoms.append(ag[0])

        return mda.AtomGroup(atoms)

    def _stable_distance_anchor_context(self) -> dict[str, Any] | None:
        if len(self.protein_anchor_masks) != 3:
            return None

        anchors = self._get_protein_anchor_atoms()
        if anchors is None or anchors.n_atoms != 3:
            return None

        def _nonempty_values(attr_name: str) -> set[str]:
            values = set()
            for atom in anchors:
                value = str(getattr(atom, attr_name, "")).strip()
                if value:
                    values.add(value)
            return values

        return {
            "anchors": anchors,
            "p1": anchors[0],
            "p2": anchors[1],
            "p3": anchors[2],
            "segids": _nonempty_values("segid"),
            "chainIDs": _nonempty_values("chainID"),
            "exclude_indices": {int(anchors[1].index), int(anchors[2].index)},
        }

    @staticmethod
    def _atom_group_from_atoms(
        atoms: Sequence[Any],
        fallback_group: AtomGroup,
    ) -> AtomGroup:
        if atoms:
            return mda.AtomGroup(list(atoms))
        return fallback_group[:0]

    def _filter_stable_candidates_to_anchor_context(
        self,
        candidates: AtomGroup,
        anchor_context: dict[str, Any] | None,
    ) -> AtomGroup:
        if anchor_context is None or candidates.n_atoms == 0:
            return candidates

        filtered = candidates
        for attr_name, context_key in (("segid", "segids"), ("chainID", "chainIDs")):
            allowed = {
                str(value).strip()
                for value in anchor_context.get(context_key, set())
                if str(value).strip()
            }
            if not allowed:
                continue
            matches = [
                atom
                for atom in filtered
                if str(getattr(atom, attr_name, "")).strip() in allowed
            ]
            if matches:
                filtered = mda.AtomGroup(matches)

        excluded = set(anchor_context.get("exclude_indices") or set())
        if excluded:
            filtered = self._atom_group_from_atoms(
                [atom for atom in filtered if int(atom.index) not in excluded],
                filtered,
            )
        return filtered

    def _stable_distance_protein_candidates(
        self,
        anchor_context: dict[str, Any] | None = None,
    ) -> AtomGroup:
        candidates = self.universe.select_atoms(
            'protein and not resname NMA ACE and name CA C N'
        )
        if candidates.n_atoms == 0:
            candidates = self.universe.select_atoms('protein and name CA C N')
        if candidates.n_atoms == 0:
            candidates = self._heavy_atoms_or_all(self.universe.select_atoms('protein'))
        candidates = self._filter_stable_candidates_to_anchor_context(
            candidates, anchor_context
        )
        return candidates

    def _stable_distance_ligand_candidates(
        self,
        ligand_atom_names: Sequence[str] | None = None,
    ) -> AtomGroup:
        ligand_ag = self.universe.select_atoms(f'resname {self.ligand}')
        if ligand_ag.n_atoms == 0:
            raise ValueError(f'No ligand atoms found for resname {self.ligand!r}.')

        ligand_ag = self._heavy_atoms_or_all(ligand_ag)
        requested = {
            str(name).strip()
            for name in ligand_atom_names or []
            if str(name).strip()
        }
        if not requested:
            return ligand_ag

        atoms = [atom for atom in ligand_ag if str(atom.name) in requested]
        if not atoms:
            logger.warning(
                'No ligand atoms matched the requested Boresch candidate names {}; '
                'using all ligand heavy atoms for stable-distance search.',
                sorted(requested),
            )
            return ligand_ag
        return mda.AtomGroup(atoms)

    @staticmethod
    def _atom_metadata(atom: Any) -> dict[str, Any]:
        try:
            atom_id = int(atom.id)
        except Exception:
            atom_id = None
        try:
            chain_id = str(atom.chainID).strip()
        except Exception:
            chain_id = ""
        segid = str(getattr(atom, "segid", "")).strip()
        resid = int(atom.resid)
        name = str(atom.name)
        return {
            "index": int(atom.index),
            "id": atom_id,
            "resid": resid,
            "resname": str(atom.resname),
            "name": name,
            "segid": segid,
            "chainID": chain_id,
            "mask": f":{resid}@{name}",
        }

    @staticmethod
    def _angle_matrix_degrees(
        protein_positions: np.ndarray,
        p2_position: np.ndarray,
        ligand_positions: np.ndarray,
    ) -> np.ndarray:
        p1_to_p2 = np.asarray(p2_position, dtype=float).reshape(1, 3) - np.asarray(
            protein_positions, dtype=float
        )
        p1_to_ligand = (
            np.asarray(ligand_positions, dtype=float).reshape(1, -1, 3)
            - np.asarray(protein_positions, dtype=float).reshape(-1, 1, 3)
        )
        dot = np.sum(p1_to_p2.reshape(-1, 1, 3) * p1_to_ligand, axis=2)
        norms = np.linalg.norm(p1_to_p2, axis=1).reshape(-1, 1) * np.linalg.norm(
            p1_to_ligand, axis=2
        )
        cosang = np.divide(
            dot,
            norms,
            out=np.full(dot.shape, np.nan, dtype=float),
            where=norms > 1.0e-8,
        )
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    def find_stable_boresch_distance(
        self,
        *,
        tail_fraction: float = 0.25,
        min_distance: float = 3.0,
        max_distance: float = 7.0,
        ligand_atom_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Pick a stable protein-ligand atom pair from the equilibration tail.

        Protein candidates follow BATTER's receptor-anchor atom class (backbone
        CA/C/N, with heavy-atom fallback). Ligand candidates can be restricted
        to names produced by the ligand-anchor candidate heuristic. The selected
        pair must have a mean distance in ``[min_distance, max_distance]`` over
        the trailing analysis window. When the original receptor anchors are
        available, the P2-P1-L1 angle must also satisfy the relaxed VMD first
        anchor tolerance.
        """
        if min_distance < 0 or max_distance <= min_distance:
            raise ValueError("min_distance must be >= 0 and below max_distance.")

        anchor_context = self._stable_distance_anchor_context()
        protein_candidates = self._stable_distance_protein_candidates(anchor_context)
        ligand_candidates = self._stable_distance_ligand_candidates(ligand_atom_names)
        if protein_candidates.n_atoms == 0:
            raise ValueError("No protein atoms available for stable-distance search.")
        if ligand_candidates.n_atoms == 0:
            raise ValueError("No ligand atoms available for stable-distance search.")

        n_frames = len(self.universe.trajectory)
        if n_frames == 0:
            raise ValueError(
                "No trajectory frames available for stable-distance search."
            )
        start_frame = self._trailing_analysis_start_frame(tail_fraction=tail_fraction)

        n_pairs = protein_candidates.n_atoms * ligand_candidates.n_atoms
        sum_dist = np.zeros(n_pairs, dtype=float)
        sum_sq_dist = np.zeros(n_pairs, dtype=float)
        sum_angle = np.zeros(n_pairs, dtype=float)
        sum_sq_angle = np.zeros(n_pairs, dtype=float)
        angle_counts = np.zeros(n_pairs, dtype=float)
        frame_indices: list[int] = []
        p2_atom = anchor_context.get("p2") if anchor_context is not None else None

        for ts in self.universe.trajectory[start_frame:n_frames]:
            dist_mat = distance_array(
                protein_candidates.positions,
                ligand_candidates.positions,
                box=self.universe.dimensions,
            )
            flat = dist_mat.reshape(-1)
            sum_dist += flat
            sum_sq_dist += flat * flat
            if p2_atom is not None:
                angle_mat = self._angle_matrix_degrees(
                    protein_candidates.positions,
                    np.asarray(p2_atom.position, dtype=float),
                    ligand_candidates.positions,
                )
                flat_angle = angle_mat.reshape(-1)
                finite_angle = np.isfinite(flat_angle)
                sum_angle[finite_angle] += flat_angle[finite_angle]
                sum_sq_angle[finite_angle] += (
                    flat_angle[finite_angle] * flat_angle[finite_angle]
                )
                angle_counts[finite_angle] += 1.0
            frame_indices.append(int(ts.frame))

        if not frame_indices:
            raise ValueError("No tail frames available for stable-distance search.")

        count = float(len(frame_indices))
        mean_dist = sum_dist / count
        var_dist = np.maximum((sum_sq_dist / count) - (mean_dist * mean_dist), 0.0)
        std_dist = np.sqrt(var_dist)
        finite = np.isfinite(mean_dist) & np.isfinite(std_dist)
        in_window = (
            finite
            & (mean_dist >= float(min_distance))
            & (mean_dist <= float(max_distance))
        )
        mean_angle = np.full(n_pairs, np.nan, dtype=float)
        std_angle = np.full(n_pairs, np.nan, dtype=float)
        angle_valid = np.ones(n_pairs, dtype=bool)
        if p2_atom is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_angle = np.divide(
                    sum_angle,
                    angle_counts,
                    out=np.full(n_pairs, np.nan, dtype=float),
                    where=angle_counts > 0.0,
                )
                var_angle = np.divide(
                    sum_sq_angle,
                    angle_counts,
                    out=np.full(n_pairs, np.nan, dtype=float),
                    where=angle_counts > 0.0,
                ) - (mean_angle * mean_angle)
            std_angle = np.sqrt(np.maximum(var_angle, 0.0))
            angle_valid = (
                np.isfinite(mean_angle)
                & (angle_counts == float(len(frame_indices)))
                & (
                    np.abs(mean_angle - _VMD_FIRST_ANCHOR_ANGLE_TARGET)
                    <= _VMD_FIRST_ANCHOR_ANGLE_TOLERANCE
                )
            )
        in_window &= angle_valid
        valid_indices = np.where(in_window)[0]
        if valid_indices.size == 0:
            angle_note = (
                " and a VMD-compatible P2-P1-L1 mean angle"
                if p2_atom is not None
                else ""
            )
            raise ValueError(
                "No protein-ligand candidate pair had a tail-window mean distance "
                f"between {min_distance:.2f} and {max_distance:.2f} Å{angle_note}."
            )

        mid_distance = (float(min_distance) + float(max_distance)) / 2.0
        best_flat = min(
            (int(idx) for idx in valid_indices),
            key=lambda idx: (
                float(std_dist[idx]),
                float(std_angle[idx]) if np.isfinite(std_angle[idx]) else 0.0,
                abs(float(mean_angle[idx]) - _VMD_FIRST_ANCHOR_ANGLE_TARGET)
                if np.isfinite(mean_angle[idx])
                else 0.0,
                abs(float(mean_dist[idx]) - mid_distance),
                int(idx // ligand_candidates.n_atoms),
                int(idx % ligand_candidates.n_atoms),
            ),
        )
        protein_idx = best_flat // ligand_candidates.n_atoms
        ligand_idx = best_flat % ligand_candidates.n_atoms
        protein_atom = protein_candidates[int(protein_idx)]
        ligand_atom = ligand_candidates[int(ligand_idx)]

        distances: list[float] = []
        vectors: list[np.ndarray] = []
        angles: list[float] = []
        for _ in self.universe.trajectory[start_frame:n_frames]:
            vector = np.asarray(
                ligand_atom.position - protein_atom.position, dtype=float
            )
            dist = distance_array(
                np.asarray(protein_atom.position, dtype=float).reshape(1, 3),
                np.asarray(ligand_atom.position, dtype=float).reshape(1, 3),
                box=self.universe.dimensions,
            )[0, 0]
            vectors.append(vector.copy())
            distances.append(float(dist))
            if p2_atom is not None:
                angle = self._angle_matrix_degrees(
                    np.asarray(protein_atom.position, dtype=float).reshape(1, 3),
                    np.asarray(p2_atom.position, dtype=float),
                    np.asarray(ligand_atom.position, dtype=float).reshape(1, 3),
                )[0, 0]
                angles.append(float(angle))

        vector_array = np.asarray(vectors, dtype=float)
        distance_array_values = np.asarray(distances, dtype=float)
        angle_array_values = np.asarray(angles, dtype=float)
        vector_mean = vector_array.mean(axis=0)
        vector_std = vector_array.std(axis=0)
        selected_mean = float(distance_array_values.mean())
        selected_std = float(distance_array_values.std())
        angle_record = None
        if p2_atom is not None and angle_array_values.size:
            angle_record = {
                "p2_anchor": self._atom_metadata(p2_atom),
                "target": _VMD_FIRST_ANCHOR_ANGLE_TARGET,
                "tolerance": _VMD_FIRST_ANCHOR_ANGLE_TOLERANCE,
                "mean": float(angle_array_values.mean()),
                "std": float(angle_array_values.std()),
                "min": float(angle_array_values.min()),
                "max": float(angle_array_values.max()),
                "last": float(angle_array_values[-1]),
            }

        self.results['stable_boresch_distance'] = distance_array_values
        self.results['stable_boresch_frame_indices'] = np.asarray(
            frame_indices, dtype=int
        )
        if angle_array_values.size:
            self.results['stable_boresch_angle'] = angle_array_values

        return {
            "schema_version": STABLE_BORESCH_DISTANCE_SCHEMA_VERSION,
            "source": "equil_analysis",
            "tail_fraction": float(tail_fraction),
            "analysis_start_frame": int(start_frame),
            "n_frames": int(len(frame_indices)),
            "criteria": {
                "protein_atom_names": ["CA", "C", "N"],
                "ligand_atom_names": [
                    str(name).strip()
                    for name in ligand_atom_names or []
                    if str(name).strip()
                ],
                "min_distance": float(min_distance),
                "max_distance": float(max_distance),
                "first_anchor_angle_target": _VMD_FIRST_ANCHOR_ANGLE_TARGET,
                "first_anchor_angle_tolerance": _VMD_FIRST_ANCHOR_ANGLE_TOLERANCE,
            },
            "protein": self._atom_metadata(protein_atom),
            "ligand": self._atom_metadata(ligand_atom),
            "distance": {
                "mean": selected_mean,
                "std": selected_std,
                "cv": selected_std / selected_mean if selected_mean else None,
                "min": float(distance_array_values.min()),
                "max": float(distance_array_values.max()),
                "last": float(distance_array_values[-1]),
            },
            "vector": {
                "mean": [float(x) for x in vector_mean],
                "std": [float(x) for x in vector_std],
            },
            "angle": angle_record,
            "frame_indices": [int(x) for x in frame_indices],
            "distances": [float(x) for x in distance_array_values],
            "angles": [float(x) for x in angle_array_values],
        }

    def _trailing_analysis_start_frame(self, tail_fraction: float = 0.25) -> int:
        if not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must be in the interval (0, 1].")
        n_frames = len(self.universe.trajectory)
        if n_frames <= 1:
            return 0
        start_frame = int(np.floor(n_frames * (1.0 - tail_fraction)))
        return min(max(start_frame, 0), n_frames - 1)

    def _ligand_dihedral(self, start_frame: int = 0):
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
        
        n_frames = len(self.universe.trajectory)
        if n_frames == 0:
            self.results['ligand_dihedrals'] = np.empty((0, len(ag_lists)))
            self.results['ligand_dihedral_frame_indices'] = np.array([], dtype=int)
            return
        start_frame = min(max(int(start_frame), 0), n_frames - 1)

        diheds = []
        frame_indices = []
        for ts in self.universe.trajectory[start_frame:n_frames]:
            dihed = []
            for ag in ag_lists:
                dihed.append(ag.dihedral.value())
            diheds.append(dihed)
            frame_indices.append(ts.frame)
        diheds = np.array(diheds)

        self.results['ligand_dihedrals'] = diheds
        self.results['ligand_dihedral_frame_indices'] = np.asarray(frame_indices, dtype=int)

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

    def plot_analysis(self, savefig=True, output_filename='simulation_analysis.png'):
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
            plt.savefig(self.workdir / output_filename)
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
    def find_representative_snapshot(
        self,
        savefig=True,
        output_filename='dihed_hist.png',
        tail_fraction: float = 0.25,
    ):
        """
        Find the representative snapshot based on tail-window mode dihedral values.
        """
        # convert to sin and cos values
        start_frame = self._trailing_analysis_start_frame(tail_fraction=tail_fraction)
        self._ligand_dihedral(start_frame=start_frame)
        dihed = self.results['ligand_dihedrals']
        frame_indices = np.asarray(
            self.results.get(
                'ligand_dihedral_frame_indices',
                np.arange(start_frame, start_frame + len(dihed)),
            ),
            dtype=int,
        )
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
        representative_local_index = np.argmin(np.sum(abs_diff, axis=1))
        representative_index = int(frame_indices[representative_local_index])
        
        # plot 
        fig, ax = plt.subplots(1, n_dihed, figsize=(20, 5), sharex=True, sharey=True,
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        ax = np.atleast_1d(ax)
        for i in range(n_dihed):
            ax[i].hist(dihed[:, i], bins=100, density=True, alpha=0.5, range=(-180, 180))
            ax[i].set_title(f"{i}")
            ax[i].vlines(dihed[representative_local_index, i], ymin=0, ymax=0.05,
                        color='r', linestyle='--', label='Representative')
        plt.tight_layout()
        if savefig:
            plt.savefig(self.workdir / output_filename)
        else:
            plt.show()
        plt.close(fig)

        self.results['representative_frame_index'] = representative_index
        self.results['representative_analysis_start_frame'] = start_frame
        return representative_index

    def dump_results(self, filename='equilibration_analysis_results.npz'):
        """
        Dump the results to a npz file
        """
        filepath = self.workdir / filename
        np.savez_compressed(filepath, **self.results)
        logger.debug(f'Simulation validation results saved to {filepath}')


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
