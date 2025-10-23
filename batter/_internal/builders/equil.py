from __future__ import annotations

from pathlib import Path
from loguru import logger
from typing import Any, Dict, Optional

from batter.config.simulation import SimulationConfig
from batter._internal.builders.base import BaseBuilder
from batter._internal.ops import build_complex, restraints, runfiles, box, amber, simprep, sim_files


class PrepareEquilBuilder(BaseBuilder):
    """
    Builder for the ABFE stage: `prepare_equil`.

    Responsibilities
    ----------------
    Writes under <work>/<ligand>/:
      - q_build_files/…        (aligned complex, anchors, reference artifacts)
      - q_amber_files/…        (rendered AMBER templates with sim params)
      - q_run_files/…          (run scripts/templates for the subsequent `equil` stage)
      - disang.rest, cv.in   (equil restraints)

    Notes
    -----
    Uses ligand-specific parameter directories (from the artifact index)
    and the resolved residue name provided by `prepare_equil_handler`.
    """

    stage = "prepare_equil"

    def __init__(
        self,
        ligand: str,
        residue_name: str,
        param_dir_dict: dict,
        sim_config: SimulationConfig,
        component_windows_dict: dict,
        working_dir: Path | str,
        system_root: Path | str,
        *,
        infe: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            ligand=ligand,
            sim_config=sim_config,
            component_windows=component_windows_dict,
            working_dir=working_dir,
            system_root=system_root,
            component="q",  # equilibrium component
            win=-1,
            residue_name=residue_name,
            param_dir_dict=param_dir_dict,
            extra=extra,
        )
        self.infe = infe
        logger.debug(f"[prepare_equil] ligand={ligand}")

    # ------------------------------------------------------------------
    # Core build hooks
    # ------------------------------------------------------------------

    def _build_complex(self) -> bool:
        """Align receptor–ligand complex and detect anchors."""
        logger.debug(f"[prepare_equil] Building complex for {self.ctx.ligand}")
        return build_complex.build_complex(self.ctx, infe=self.infe)

    def _create_amber_files(self) -> None:
        """Render AMBER templates for the system."""
        
        work = self.ctx.working_dir
        amber.write_amber_templates(
            out_dir=self.amber_dir,
            sim=self.ctx.sim,
            membrane=self.membrane_builder,
            production=False,
        )
        logger.debug(f"[prepare_equil] Created amber files for {self.ctx.ligand}")
    
    def _create_simulation_dir(self) -> None:
        """Create the simulation directory."""
        simprep.create_simulation_dir_eq(self.ctx)

    def _create_box(self) -> None:
        """Render AMBER templates and build solvated/ionized system."""

        box.create_box(self.ctx)
        logger.debug(f"[prepare_equil] Created box for {self.ctx.ligand}")

    def _restraints(self) -> None:
        """Write equilibrium restraints (disang.rest, cv.in)."""
        restraints.write_equil_restraints(self.ctx)
        logger.debug(f"[prepare_equil] Wrote restraints for {self.ctx.ligand}")

    def _sim_files(self) -> None:
        """Write equilibration input decks: mini.in, eqnvt.in, eqnpt*.in, etc."""
        sim_files.write_sim_files(self.ctx, infe=self.infe)
        logger.debug(f"[prepare_equil] Wrote sim files for {self.ctx.ligand}")

    def _run_files(self) -> None:
        """Emit run scripts for the next `equil` step."""
        runfiles.write_equil_run_files(self.ctx, stage=self.stage)
        logger.debug(f"[prepare_equil] Wrote run files for {self.ctx.ligand}")