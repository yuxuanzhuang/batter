"""Prepare complex systems (protein/ligand/membrane) for simulations."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import align
from loguru import logger

from batter._internal.templates import BUILD_FILES_DIR as build_files_orig
from batter.orchestrate.state_registry import register_phase_state
from batter.pipeline.payloads import StepPayload, SystemParams
from batter.pipeline.step import ExecResult, Step
from batter.systems.core import SimSystem
from batter.utils.builder_utils import find_anchor_atoms


# -----------------------
# Small helpers
# -----------------------
def _as_abs(p: str | Path | None, base: Path) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _ensure_pdb(lig_path: Path, out_dir: Path) -> Path:
    """
    Ensure a PDB exists for ligand file; if not PDB, convert via RDKit.
    Returns the path to a PDB file.
    """
    if lig_path.suffix.lower() == ".pdb":
        return lig_path

    try:
        from rdkit import Chem
    except Exception as e:
        raise RuntimeError(
            f"Ligand {lig_path} is not PDB; RDKit is required to convert SDF/MOL2 → PDB."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f"{lig_path.stem}.pdb"

    if lig_path.suffix.lower() == ".sdf":
        suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            raise ValueError(f"RDKit could not read any molecule from {lig_path}")
        Chem.MolToPDBFile(mols[0], str(out_pdb))
    elif lig_path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(lig_path), removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit could not read {lig_path}")
        Chem.MolToPDBFile(mol, str(out_pdb))
    elif lig_path.suffix.lower() == "pdb":
        _copy(lig_path, out_pdb)
    else:
        raise ValueError(f"Unsupported ligand format: {lig_path.suffix} for {lig_path}")
    return out_pdb


class _SystemPrepRunner:
    def __init__(self, system: SimSystem, yaml_dir: Path) -> None:
        self.system = system
        self.yaml_dir = yaml_dir

        self.output_dir = system.root
        self.ligands_folder = self.output_dir / "all-ligands"
        self.ligandff_folder = self.output_dir / "artifacts" / "ligands"
        self.ligandff_folder.mkdir(parents=True, exist_ok=True)

        # state
        self._system_name: str = ""
        self._protein_input: str = ""
        self._system_topology: str | None = None
        self._system_coordinate: str | None = None

        self.receptor_segment: str | None = None
        self.protein_align: str = "name CA and resid 60 to 250"
        self.receptor_ff: str = "protein.ff14SB"
        self.retain_lig_prot: bool = True
        self.ligand_ph: float = 7.4
        self.lipid_mol: List[str] = []
        self.membrane_simulation: bool = False
        self.lipid_ff: str = "lipid21"
        self.overwrite: bool = False
        self.verbose: bool = False

        self.ligand_dict: Dict[str, str] = {}
        self.unique_mol_names: List[str] = []
        self.system_dimensions = np.zeros(3)

        # alignment intermediates
        self._protein_aligned_pdb: str | None = None
        self._system_aligned_pdb: str | None = None
        self.translation: np.ndarray = np.zeros(3)
        self.mobile_coord: np.ndarray | None = None
        self.ref_coord: np.ndarray | None = None
        self.mobile_com: np.ndarray | None = None
        self.ref_com: np.ndarray | None = None

        # anchors
        self.anchor_atoms: List[str] = []
        self.ligand_anchor_atom: str | None = None
        self.l1_x = self.l1_y = self.l1_z = None
        self.l1_range = None
        self.p1 = self.p2 = self.p3 = None

    # properties used by your pasted functions
    @property
    def system_name(self) -> str:
        return self._system_name

    # legacy helper (resolve relative paths against YAML dir)
    def _convert_2_relative_path(self, p: str) -> str:
        ap = _as_abs(p, self.yaml_dir)
        if ap is None:
            raise ValueError("unexpected None path")
        return str(ap)

    @contextlib.contextmanager
    def _change_dir(self, path: Path):
        cwd = Path.cwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(cwd)

    # -----------------------
    # Core steps
    # -----------------------
    def _prepare_membrane(self):
        """
        Convert input lipid names to lipid21 set (PC/PA/OL for POPC) via lookup CSV.
        """
        logger.debug("Input: membrane system")

        # read charmmlipid2amber file
        charmm_csv_path = resources.files("batter") / "data/charmmlipid2amber.csv"
        charmm_amber_lipid_df = pd.read_csv(charmm_csv_path, header=1, sep=",")

        lipid_mol = list(self.lipid_mol)
        logger.debug(f"Converting lipid input: {lipid_mol}")
        amber_lipid_mol = charmm_amber_lipid_df.query("residue in @lipid_mol")[
            "replace"
        ]
        amber_lipid_mol = (
            amber_lipid_mol.apply(lambda x: x.split()[1]).unique().tolist()
        )

        # extend instead of replacing so that we can have both
        lipid_mol.extend(amber_lipid_mol)
        self.lipid_mol = lipid_mol
        logger.debug(f"New lipid_mol list: {self.lipid_mol}")

    def _get_alignment(self):
        """
        Prepare for the alignment of the protein and ligand to the system.
        """
        logger.debug("Getting the alignment of the protein and ligand to the system")

        # translate the cog of protein to the origin
        #
        u_prot = mda.Universe(self._protein_input)

        u_sys = mda.Universe(self._system_input_pdb, format="XPDB")
        cog_prot = u_sys.select_atoms("protein and name CA C N O").center_of_geometry()
        u_sys.atoms.positions -= cog_prot

        # get translation-rotation matrix
        mobile = u_prot.select_atoms(self.protein_align).select_atoms(
            "name CA and not resname NMA ACE"
        )
        ref = u_sys.select_atoms(self.protein_align).select_atoms(
            "name CA and not resname NMA ACE"
        )

        if mobile.n_atoms != ref.n_atoms:
            raise ValueError(
                f"Number of atoms in the alignment selection is different: protein_input: "
                f"{mobile.n_atoms} and system_input {ref.n_atoms} \n"
                f"The selection string is {self.protein_align} and name CA and not resname NMA ACE\n"
                f"protein selected resids: {mobile.residues.resids}\n"
                f"system selected resids: {ref.residues.resids}\n"
                "set `protein_align` to a selection string that has the same number of atoms in both files"
                "when running `create_system`."
            )
        mobile_com = mobile.center(weights=None)
        ref_com = ref.center(weights=None)
        mobile_coord = mobile.positions - mobile_com
        ref_coord = ref.positions - ref_com

        _ = align._fit_to(
            mobile_coordinates=mobile_coord,
            ref_coordinates=ref_coord,
            mobile_atoms=u_prot.atoms,
            mobile_com=mobile_com,
            ref_com=ref_com,
        )

        cog_prot = u_prot.select_atoms("protein and name CA C N O").center_of_geometry()
        u_prot.atoms.positions -= cog_prot
        u_prot.atoms.write(f"{self.ligands_folder}/protein_aligned.pdb")
        self._protein_aligned_pdb = f"{self.ligands_folder}/protein_aligned.pdb"
        u_sys.atoms.write(f"{self.ligands_folder}/system_aligned.pdb")
        self._system_aligned_pdb = f"{self.ligands_folder}/system_aligned.pdb"

        self.translation = cog_prot

        # store these for ligand alignment
        self.mobile_com = mobile_com
        self.ref_com = ref_com
        self.mobile_coord = mobile_coord
        self.ref_coord = ref_coord

    def _process_system(self):
        """
        Generate the protein, reference, and lipid (if applicable) files.
        We will align the protein_input to the system_topology because
        the system_topology is generated by dabble and may be shifted;
        we want to align the protein to the system so the membrane is
        properly positioned.
        """
        logger.debug("Processing the system")

        if not self._protein_aligned_pdb or not self._system_aligned_pdb:
            raise RuntimeError("Alignment not computed. Call _get_alignment() first.")

        u_prot = mda.Universe(self._protein_aligned_pdb)
        u_sys = mda.Universe(self._system_aligned_pdb, format="XPDB")
        try:
            u_sys.atoms.chainIDs
        except AttributeError:
            u_sys.add_TopologyAttr("chainIDs")

        memb_seg = u_sys.add_Segment(segid="MEMB")
        water_seg = u_sys.add_Segment(segid="WATR")

        # modify the chaininfo to be unique for each segment
        current_chain = 65
        u_prot.atoms.tempfactors = 0

        # read and validate the correct segments
        n_segments = len(u_sys.select_atoms("protein").segments)
        n_segment_name = np.unique(u_sys.select_atoms("protein").segids)
        if len(n_segment_name) != n_segments:
            logger.warning(
                f"Number of segments in the system is {n_segments} but the segment names are {n_segment_name}. "
                f"Setting all segments to 'A' for the protein. If you want to use different segments, "
                "modify the segments column in the system_topology file manually."
            )
            protein_seg = u_sys.add_Segment(segid="A")
            u_sys.select_atoms("protein").residues.segments = protein_seg

        for segment in u_sys.select_atoms("protein").segments:
            resid_seg = segment.residues.resids
            resid_seq = " ".join([str(resid) for resid in resid_seg])
            chain_id = segment.atoms.chainIDs[0]
            u_prot.select_atoms(
                f"resid {resid_seq} and chainID {chain_id} and protein"
            ).atoms.tempfactors = current_chain
            current_chain += 1
        u_prot.atoms.chainIDs = [
            chr(int(chain_nm)) for chain_nm in u_prot.atoms.tempfactors
        ]

        comp_2_combined = []

        if self.receptor_segment:
            protein_anchor = u_prot.select_atoms(
                f"segid {self.receptor_segment} and protein"
            )
            protein_anchor.atoms.chainIDs = "A"
            protein_anchor.atoms.tempfactors = 65
            other_protein = u_prot.select_atoms(
                f"not segid {self.receptor_segment} and protein"
            )
            comp_2_combined.append(protein_anchor)
            comp_2_combined.append(other_protein)
        else:
            comp_2_combined.append(u_prot.select_atoms("protein"))

        if self.membrane_simulation:
            membrane_ag = u_sys.select_atoms(f'resname {" ".join(self.lipid_mol)}')
            if len(membrane_ag) == 0:
                logger.warning(
                    "No membrane atoms found with resname {}. Available resnames are {}. "
                    "Please check the lipid_mol parameter.",
                    self.lipid_mol,
                    list(np.unique(u_sys.atoms.resnames)),
                )
            else:
                with open(f"{build_files_orig}/memb_opls2charmm.json", "r") as f:
                    MEMB_OPLS_2_CHARMM_DICT = json.load(f)
                if np.any(membrane_ag.names == "O1"):
                    if np.any(membrane_ag.residues.resnames != "POPC"):
                        raise ValueError(
                            f"Found OPLS lipid name {membrane_ag.residues.resnames}, only 'POPC' is supported."
                        )
                    # convert the lipid names to CHARMM names
                    membrane_ag.names = [
                        MEMB_OPLS_2_CHARMM_DICT.get(name, name)
                        for name in membrane_ag.names
                    ]
                    logger.info("Converting OPLS lipid names to CHARMM names.")
                membrane_ag.chainIDs = "M"
                membrane_ag.residues.segments = memb_seg
                logger.debug(f"Number of lipid molecules: {membrane_ag.n_residues}")
                comp_2_combined.append(membrane_ag)
        else:
            membrane_ag = u_sys.atoms[[]]  # empty selection

        # gather water (and ions) around protein/membrane
        water_ag = u_sys.select_atoms(
            "byres (((resname SPC and name O) or water) and around 15 (protein or group memb))",
            memb=membrane_ag,
        )
        logger.debug(f"Number of water molecules: {water_ag.n_residues}")
        ion_ag = u_sys.select_atoms(
            "byres (resname SOD POT CLA NA CL and around 5 (protein))"
        )
        logger.debug(f"Number of ion molecules: {ion_ag.n_residues}")
        # normalize ion names
        ion_ag.select_atoms("resname SOD").names = "Na+"
        ion_ag.select_atoms("resname SOD").residues.resnames = "Na+"
        ion_ag.select_atoms("resname NA").names = "Na+"
        ion_ag.select_atoms("resname NA").residues.resnames = "Na+"
        ion_ag.select_atoms("resname POT").names = "K+"
        ion_ag.select_atoms("resname POT").residues.resnames = "K+"
        ion_ag.select_atoms("resname CLA").names = "Cl-"
        ion_ag.select_atoms("resname CLA").residues.resnames = "Cl-"
        ion_ag.select_atoms("resname CL").names = "Cl-"
        ion_ag.select_atoms("resname CL").residues.resnames = "Cl-"

        water_ag = water_ag + ion_ag
        water_ag.chainIDs = "W"
        water_ag.residues.segments = water_seg
        if len(water_ag) == 0:
            logger.warning(
                "No water molecules found in the system. Available resnames are %s. "
                "Please check the system_topology and system_coordinate files.",
                np.unique(u_sys.atoms.resnames),
            )
        else:
            comp_2_combined.append(water_ag)

        u_merged = mda.Merge(*comp_2_combined)

        water = u_merged.select_atoms("water or resname SPC")
        if len(water) != 0:
            logger.debug(
                f"Number of water molecules in merged system: {water.n_residues}"
            )
            logger.debug(f"Water atom names: {water.residues[0].atoms.names}")

        # Normalize water O names for tleap
        water.select_atoms("name OW").names = "O"
        water.select_atoms("name OH2").names = "O"

        box_dim = np.zeros(6)
        if len(self.system_dimensions) == 3:
            box_dim[:3] = self.system_dimensions
            box_dim[3:] = 90.0
        elif len(self.system_dimensions) == 6:
            box_dim = self.system_dimensions
        else:
            raise ValueError(f"Invalid system_dimensions: {self.system_dimensions}")
        u_merged.dimensions = box_dim

        u_merged.atoms.write(f"{self.ligands_folder}/{self.system_name}.pdb")
        protein_ref = u_prot.select_atoms("protein")
        protein_ref.write(f"{self.ligands_folder}/reference.pdb")

    def _align_2_system(self, mobile_atoms):
        """
        Apply translation-only movement to bring ligand into system frame.
        """
        _ = align._fit_to(
            mobile_coordinates=self.mobile_coord,
            ref_coordinates=self.ref_coord,
            mobile_atoms=mobile_atoms,
            mobile_com=self.mobile_com,
            ref_com=self.ref_com,
        )

        mobile_atoms.positions -= self.translation

    def _prepare_all_ligands(self):
        """
        Prepare ligand ligands for the system from input ligand files (PDB/SDF/MOL2).
        """
        logger.debug("prepare ligands")
        new_ligand_dict: Dict[str, str] = {}
        # name order is deterministic
        for i, (name, ligand_path) in enumerate(sorted(self.ligand_dict.items())):
            name_up = name.upper()
            ligand_file = _ensure_pdb(Path(ligand_path), self.ligandff_folder)

            u = mda.Universe(str(ligand_file))
            try:
                u.atoms.chainIDs
            except AttributeError:
                u.add_TopologyAttr("chainIDs")
            lig_seg = u.add_Segment(segid="LIG")
            u.atoms.chainIDs = "L"
            u.atoms.residues.segments = lig_seg
            u.atoms.residues.resnames = "lig"

            logger.debug(f"Processing ligand {i}: {ligand_path}")
            self._align_2_system(u.atoms)
            out_ligand = f"{self.ligands_folder}/{name}.pdb"
            u.atoms.write(out_ligand)

            new_ligand_dict[name] = out_ligand
        self.ligand_dict = new_ligand_dict

    # -----------------------
    # Orchestrated entry
    # -----------------------
    def run(
        self,
        *,
        system_name: str,
        protein_input: str,
        ligand_paths: Dict[str, str],
        anchor_atoms: List[str],
        system_topology: str | None = None,
        ligand_anchor_atom: str | None = None,
        receptor_segment: str | None = None,
        system_coordinate: str | None = None,
        protein_align: str = "name CA and resid 60 to 250",
        receptor_ff: str = "protein.ff14SB",
        retain_lig_prot: bool = True,
        ligand_ph: float = 7.4,
        lipid_mol: List[str] = [],
        lipid_ff: str = "lipid21",
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        self._system_name = system_name
        self._protein_input = self._convert_2_relative_path(protein_input)
        self._system_topology = self._convert_2_relative_path(system_topology) if system_topology else None
        self._system_coordinate = (
            self._convert_2_relative_path(system_coordinate)
            if system_coordinate
            else None
        )

        self.ligand_dict = {
            k: self._convert_2_relative_path(v) for k, v in ligand_paths.items()
        }
        # prefer the provided keys for naming
        self.unique_mol_names = [k.upper() for k in ligand_paths.keys()]

        self.receptor_segment = receptor_segment
        self.protein_align = protein_align
        self.receptor_ff = receptor_ff
        self.retain_lig_prot = retain_lig_prot
        self.ligand_ph = ligand_ph
        self.overwrite = overwrite

        self.lipid_mol = lipid_mol or []
        self.membrane_simulation = bool(self.lipid_mol)
        self.lipid_ff = lipid_ff

        # sanity checks
        if not Path(self._protein_input).exists():
            raise FileNotFoundError(f"Protein input file not found: {protein_input}")
        for p in self.ligand_dict.values():
            if not Path(p).exists():
                raise FileNotFoundError(f"Ligand file not found: {p}")
        if self._system_coordinate and not Path(self._system_coordinate).exists():
            raise FileNotFoundError(
                f"System coordinate file not found: {system_coordinate}"
            )

        # Directories
        self.ligands_folder.mkdir(parents=True, exist_ok=True)

        # Box dimensions
        if self.membrane_simulation or self._system_topology is not None:
            u_sys = mda.Universe(self._system_topology, format="XPDB")
            if self._system_coordinate:
                with open(self._system_coordinate) as f:
                    lines = f.readlines()
                    box = np.array([float(x) for x in lines[-1].split()])
                self.system_dimensions = box
                u_sys.load_new(self._system_coordinate, format="INPCRD")
            else:
                try:
                    self.system_dimensions = u_sys.dimensions[:3]
                except TypeError:
                    if self.membrane_simulation:
                        raise ValueError(
                            "No box dimensions found in system_topology; required for membrane systems."
                        )
                    protein = u_sys.select_atoms("protein")
                    padding = 10.0
                    box_x = (
                        protein.positions[:, 0].max()
                        - protein.positions[:, 0].min()
                        + 2 * padding
                    )
                    box_y = (
                        protein.positions[:, 1].max()
                        - protein.positions[:, 1].min()
                        + 2 * padding
                    )
                    box_z = (
                        protein.positions[:, 2].max()
                        - protein.positions[:, 2].min()
                        + 2 * padding
                    )
                    self.system_dimensions = np.array([box_x, box_y, box_z])
                    logger.warning(
                        "No box dimensions in system_topology. Using default 10 Å padding around protein. "
                        f"Box dimensions: {self.system_dimensions}"
                    )
            u_sys.atoms.write(f"{self.ligands_folder}/system_input.pdb")
            self._system_input_pdb = f"{self.ligands_folder}/system_input.pdb"
        else:
            self._system_input_pdb = self._protein_input
        if (
            self.membrane_simulation
            and (u_sys.atoms.dimensions is None or not u_sys.atoms.dimensions.any())
            and self._system_coordinate is None
        ):
            raise ValueError(
                "No box dimensions found in system_topology or system_coordinate when lipid system is on."
            )


        # membrane remapping (if any)
        if self.membrane_simulation:
            self._prepare_membrane()

        # Align protein to system, save aligned files, compute translation
        self._get_alignment()

        # Build reference & docked PDBs
        self._process_system()

        # Make <ligand>.pdb for each ligand by translation-only
        self._prepare_all_ligands()

        # Anchors from first ligand + protein
        u_prot = mda.Universe(f"{self.output_dir}/all-ligands/reference.pdb")
        first_ligand_path = sorted(self.ligand_dict.values())[0]
        u_lig = mda.Universe(first_ligand_path)
        lig_sdf = str(Path(ligand_paths[self.unique_mol_names[0]]))

        l1_x, l1_y, l1_z, p1, p2, p3, l1_range = find_anchor_atoms(
            u_prot, u_lig, lig_sdf, anchor_atoms, ligand_anchor_atom
        )
        self.anchor_atoms = anchor_atoms
        self.ligand_anchor_atom = ligand_anchor_atom
        self.l1_x, self.l1_y, self.l1_z = l1_x, l1_y, l1_z
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.l1_range = l1_range

        # manifest for downstream steps
        manifest = {
            "system_name": self._system_name,
            "reference": str(self.ligands_folder / "reference.pdb"),
            "docked": str(self.ligands_folder / f"{self._system_name}.pdb"),
            "ligands": dict(self.ligand_dict),
            "anchors": {"p1": self.p1, "p2": self.p2, "p3": self.p3},
            "l1": {
                "x": self.l1_x,
                "y": self.l1_y,
                "z": self.l1_z,
                "range": self.l1_range,
            },
            "membrane": (
                {"lipid_mol": self.lipid_mol, "lipid_ff": self.lipid_ff}
                if self.membrane_simulation
                else None
            ),
        }
        (self.ligands_folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        logger.debug("System loaded and prepared.")
        return manifest


# -----------------------
# Handler entry point
# -----------------------
def system_prep(step: Step, system: SimSystem, params: Dict[str, Any]) -> ExecResult:
    """Prepare a system by aligning components and generating reference structures.

    Parameters
    ----------
    step : Step
        Pipeline metadata (unused).
    system : SimSystem
        Simulation system descriptor.
    params : dict
        Handler payload validated into :class:`StepPayload`.

    Returns
    -------
    ExecResult
        Paths to generated reference structures and a metadata dictionary with
        anchor and membrane information.
    """
    logger.info(f"[system_prep] Preparing system in {system.root}")
    payload = StepPayload.model_validate(params)
    sys_params = payload.sys_params or SystemParams()
    yaml_dir = Path(sys_params["yaml_dir"]).resolve()

    runner = _SystemPrepRunner(system, yaml_dir)
    manifest = runner.run(
        system_name=sys_params["system_name"],
        protein_input=sys_params["protein_input"],
        system_topology= sys_params.get("system_input", None),
        ligand_paths=sys_params["ligand_paths"],
        anchor_atoms=list(sys_params.get("anchor_atoms", [])),
        ligand_anchor_atom=sys_params.get("ligand_anchor_atom"),
        receptor_segment=sys_params.get("receptor_segment"),
        system_coordinate=sys_params.get("system_coordinate"),
        protein_align=sys_params.get("protein_align", "name CA and resid 60 to 250"),
        receptor_ff=sys_params.get("receptor_ff", "protein.ff14SB"),
        retain_lig_prot=bool(sys_params.get("retain_lig_prot", True)),
        ligand_ph=float(sys_params.get("ligand_ph", 7.4)),
        lipid_mol=list(sys_params.get("lipid_mol", [])),
        lipid_ff=sys_params.get("lipid_ff", "lipid21"),
        overwrite=bool(sys_params.get("overwrite", False)),
        verbose=bool(sys_params.get("verbose", False)),
    )

    outputs = [
        system.root / "all-ligands" / "reference.pdb",
        system.root / "all-ligands" / f"{sys_params['system_name']}.pdb",
    ]
    updates = {
        "p1": manifest["anchors"]["p1"],
        "p2": manifest["anchors"]["p2"],
        "p3": manifest["anchors"]["p3"],
        "l1_x": manifest["l1"]["x"],
        "l1_y": manifest["l1"]["y"],
        "l1_z": manifest["l1"]["z"],
        "l1_range": manifest["l1"]["range"],
        "lipid_mol": manifest["membrane"]["lipid_mol"] if manifest["membrane"] else [],
    }
    (manifest_dir := (system.root / "artifacts" / "config")).mkdir(
        parents=True, exist_ok=True
    )
    overrides_path = system.root / "artifacts" / "config" / "sim_overrides.json"
    overrides_path.write_text(json.dumps(updates, indent=2))

    marker_rel = overrides_path.relative_to(system.root).as_posix()
    register_phase_state(
        system.root,
        "system_prep",
        required=[[marker_rel]],
        success=[[marker_rel]],
    )

    logger.info(f"[system_prep] System preparation complete.")
    info = {"system_prep_ok": True, **manifest, "sim_updates": updates}
    return ExecResult(outputs, info)
