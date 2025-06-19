# Get restraints from simulations
import MDAnalysis as mda
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
from loguru import logger
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.lib.distances import (
    minimize_vectors,
    capped_distance,
    distance_array,
)

from openff.toolkit import Molecule as OFFMol

from MDAnalysis.transformations.base import TransformationBase
from MDAnalysis.transformations.nojump import NoJump

def get_central_atom_idx(rdmol: Chem.Mol) -> int:
    """
    Get the central atom in an rdkit Molecule.
    Parameters
    ----------
    rdmol : Chem.Mol
      RDKit Molecule to query
    Returns
    -------
    int
      Index of central atom in Molecule
    Note
    ----
    If there are equal likelihood centers, will return
    the first entry.
    """
    # TODO: switch to a manual conversion to avoid an OpenFF dependency
    offmol = OFFMol(rdmol, allow_undefined_stereo=True)
    nx_mol = offmol.to_networkx()

    if not nx.is_weakly_connected(nx_mol.to_directed()):
        errmsg = "A disconnected molecule was passed, cannot find the center"
        raise ValueError(errmsg)

    # Get a list of all shortest paths
    # Note: we call dict on shortest_path to support py3.10 which doesn't
    # support networkx 3.5
    shortest_paths = [
        path
        for node_paths in dict(nx.shortest_path(nx_mol)).values()
        for path in node_paths.values()
    ]

    # Get the longest of these paths (returns first instance)
    longest_path = max(shortest_paths, key=len)

    # Return the index of the central atom
    return longest_path[len(longest_path) // 2]


def get_heavy_atom_idxs(rdmol: Chem.Mol) -> list[int]:
    """
    Get idxs of heavy atoms in an RDKit Molecule
    Parameters
    ----------
    rmdol : Chem.Mol
    Returns
    -------
    list[int]
      A list of heavy atom idxs
    """
    idxs = [at.GetIdx() for at in rdmol.GetAtoms() if at.GetAtomicNum() > 1]
    return idxs

def _get_guest_atom_pool(
    rdmol: Chem.Mol,
    rmsf: npt.NDArray,
    rmsf_cutoff,
) -> tuple[Optional[set[int]], bool]:
    """
    Filter atoms based on rmsf & rings, defaulting to heavy atoms if
    there are not enough.
    Parameters
    ----------
    rdmol : Chem.Mol
      The RDKit Molecule to search through
    rmsf : npt.NDArray
      A 1-D array of RMSF values for each atom.
    rmsf_cutoff :
      The rmsf cutoff value for selecting atoms in A
    Returns
    -------
    atom_pool : Optional[set[int]]
      A pool of candidate atoms.
    ring_atoms_only : bool
      True if only ring atoms were selected.
    """
    # Get a list of all the aromatic rings
    # Note: no need to keep track of rings because we'll filter by
    # bonded terms after, so if we only keep rings then all the bonded
    # atoms should be within the same ring system.
    atom_pool: set[int] = set()
    ring_atoms_only: bool = True
    for ring in get_aromatic_rings(rdmol):
        max_rmsf = rmsf[list(ring)].max()
        if max_rmsf < rmsf_cutoff:
            atom_pool.update(ring)

    # if we don't have enough atoms just get all the heavy atoms
    if len(atom_pool) < 3:
        ring_atoms_only = False
        heavy_atoms = get_heavy_atom_idxs(rdmol)
        atom_pool = set(
            idx for idx in heavy_atoms
            if rmsf[idx] < rmsf_cutoff
        )
        if len(atom_pool) < 3:
            return None, False

    return atom_pool, ring_atoms_only


class Aligner(TransformationBase):
    """On-the-fly transformation to align a trajectory to minimise RMSD

    centers all coordinates onto origin
    rotates **entire universe** to minimise rmsd relative to **ref_ag**
    """
    ref_pos: npt.NDArray
    ref_idx: npt.NDArray
    weights: npt.NDArray

    def __init__(self, ref_ag: mda.AtomGroup):
        super().__init__()
        self.ref_idx = ref_ag.ix
        self.ref_pos = ref_ag.positions
        self.weights = np.asarray(ref_ag.masses, dtype=np.float64)
        self.weights /= np.mean(self.weights)  # normalise weights
        # remove COM shift from reference positions
        self.ref_pos -= np.average(self.ref_pos, axis=0, weights=self.weights)

    def _transform(self, ts):
        # todo: worry about first frame?  can skip if ts.frame == 0?
        mobile_pos = ts.positions[self.ref_idx]
        mobile_com = np.average(mobile_pos, axis=0, weights=self.weights)

        mobile_pos -= mobile_com

        # rotates mobile to best align with ref
        R, min_rmsd = rotation_matrix(mobile_pos, self.ref_pos,
                                      weights=self.weights)

        # apply the transformation onto **all** atoms
        ts.positions -= mobile_com
        ts.positions = np.dot(ts.positions, R.T)

        return ts


def get_local_rmsf(atomgroup: mda.AtomGroup):
    """
    Get the RMSF of an AtomGroup when aligned upon itself.
    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup
    Return
    ------
    rmsf
      ArrayQuantity of RMSF values.
    """
    # The no trajectory case
    if len(atomgroup.universe.trajectory) < 2:
        return np.zeros(atomgroup.n_atoms)

    # First let's copy our Universe
    copy_u = atomgroup.universe.copy()
    ag = copy_u.atoms[atomgroup.atoms.ix]

    # Reset the trajectory index, otherwise we'll get in trouble with nojump
    copy_u.trajectory[0]

    nojump = NoJump()
    align = Aligner(ag)

    copy_u.trajectory.add_transformations(nojump, align)

    rmsf = RMSF(ag)
    rmsf.run()
    return rmsf.results.rmsf

def find_guest_atom_candidates(
    universe: mda.Universe,
    rdmol: Chem.Mol,
    guest_idxs: list[int],
    rmsf_cutoff: 1,
) -> list[tuple[int, int, int]]:
    """
    Copy from https://github.com/OpenFreeEnergy/openfe/pull/1043/files

    Get a list of potential ligand atom choices for a Boresch restraint
    being applied to a given small molecule.

    Parameters
    ----------
    universe : mda.Universe
      An MDAnalysis Universe defining the system and its coordinates.
    rdmol : Chem.Mol
      An RDKit Molecule representing the small molecule ordered in
      the same way as it is listed in the topology.
    guest_idxs : list[int]
      The ligand indices in the topology.
    rmsf_cutoff : float
      The RMSF filter cut-off (in Angstroms)

    Returns
    -------
    angle_list : list[tuple[int]]
      A list of tuples for each valid G0, G1, G2 angle. If ``None``, no
      angles could be found.
    Raises
    ------
    ValueError
      If no suitable ligand atoms could be found.
    """
    ligand_ag = universe.atoms[guest_idxs]

    # 0. Get the ligand RMSF
    rmsf = get_local_rmsf(ligand_ag)
    universe.trajectory[-1]  # forward to the last frame

    # 1. Get the pool of atoms to work with
    atom_pool, rings_only = _get_guest_atom_pool(rdmol, rmsf, rmsf_cutoff)

    if atom_pool is None:
        # We don't have enough atoms so we raise an error
        errmsg = "No suitable ligand atoms were found for the restraint"
        raise ValueError(errmsg)

    # 2. Get the central atom
    center = get_central_atom_idx(rdmol)

    # 3. Sort the atom pool based on their distance from the center
    sorted_atom_pool = _sort_by_distance_from_atom(rdmol, center, atom_pool)

    # 4. Get a list of probable angles
    angles_list = []
    for atom in sorted_atom_pool:
        angles = _bonded_angles_from_pool(
            rdmol=rdmol,
            atom_idx=atom,
            atom_pool=sorted_atom_pool,
            aromatic_only=rings_only,
        )
        for angle in angles:
            # Check that the angle is at least not collinear
            angle_ag = ligand_ag.atoms[list(angle)]
            if not is_collinear(ligand_ag.positions, angle, universe.dimensions):
                angles_list.append(
                    (
                        angle_ag.atoms[0].ix,
                        angle_ag.atoms[1].ix,
                        angle_ag.atoms[2].ix
                    )
                )

    return angles_list

class Boresch_Fetcher:
    def __init__(self,
                 ligand_mol: rdkit.Chem.Mol,
                 universe: mda.Universe,
                 ligand_ag: mda.AtomGroup,
                 protein_anchor_selections: List[str],
    ):
        """
        Get optimal restraints from an equilibration simulation.
        Based on OpenFE Boresch protocol and RXRX (https://chemrxiv.org/engage/chemrxiv/article-details/67e3c860fa469535b990bfac)

        Parameters
        ----------
        ligand_mol : rdkit.Chem.Mol
            The ligand molecule in RDKit format.
        universe : mda.Universe
            The MDAnalysis universe containing the simulation data.
        ligand_ag : mda.AtomGroup
            The atom group representing the ligand in the universe.
        protein_anchor_selections : List[str]
            A list of three strings, each representing a selection for an anchor atom in the protein.
            Each string should select exactly one atom in the universe.
        """
        self.ligand_mol = ligand_mol
        self.universe = universe

        self.protein_anchor_selections = protein_anchor_selections
        if len(self.protein_anchor_selections) != 0:
            raise ValueError("protein_anchor_selections should be three strings, one for each anchor atom.")
        
        self.protein_anchor_atoms = [universe.select_atoms(sel) for sel in protein_anchor_selections]
        for i, ag in enumerate(self.protein_anchor_atoms):
            if len(ag) != 1:
                raise ValueError(f"Anchor selection {i}: {self.protein_anchor_selections[i]} should select exactly one atom, but selected {len(ag)} atoms.")

        self._get_ligand_anchor_candidates()

    def _get_ligand_anchor_candidates(self):

        # 1. get ligand_candidate_atoms
        # From RXRX protocol
        # The non-hydrogen atoms connected to at least two heavy atoms are
        # selected as candidate atoms for the ligand's restraint component
        
        mol = self.ligand_mol

        anchor_candidates = []
        for atom in mol.GetAtoms():
            # no H
            if atom.GetAtomicNum() == 1:
                continue
            heavy_neighbors = 0
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() != 1:
                    heavy_neighbors += 1
            if heavy_neighbors >= 2:
                anchor_candidates.append(atom.GetIdx())

        if len(anchor_candidates) == 0:
            logger.warning("No suitable ligand anchor candidates found. Use all ligand atoms as candidates.")
            anchor_candidates = list(range(mol.GetNumAtoms()))
        ligand_atom_candidates = self.ligand_ag[anchor_candidates]
        logger.info(f"Found {ligand_atom_candidates.n_atoms} ligand anchor candidates.")

        find_guest_atom_candidates(
            self.universe,
            self.ligand_mol,
            guest_idxs=ligand_atom_candidates.indices,
            rmsf_cutoff=1
        )
