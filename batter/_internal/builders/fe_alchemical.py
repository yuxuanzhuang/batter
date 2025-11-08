from __future__ import annotations
from loguru import logger
from pathlib import Path
from typing import Any, Dict, Optional

from batter._internal.builders.base import BaseBuilder
from .fe_registry import BUILD_COMPLEX_REGISTRY, CREATE_SIMULATION_REGISTRY, CREATE_BOX_REGISTRY, RESTRAINT_REGISTRY, SIM_FILES_REGISTRY
from batter._internal.ops import restraints, runfiles, box, amber


from batter.utils import (
    run_with_log,
    COMPONENTS_LAMBDA_DICT,
    COMPONENTS_FOLDER_DICT,
)

class AlchemicalFEBuilder(BaseBuilder):
    stage = "fe"

    def _build_complex(self) -> bool:
        """Dispatch component-specific build_complex"""
        comp = self.ctx.comp.lower()
        if comp not in BUILD_COMPLEX_REGISTRY:
            raise NotImplementedError(f"No build_complex registered for component '{comp}'")
        return BUILD_COMPLEX_REGISTRY[comp](self.ctx)

    def _create_amber_files(self) -> None:
        """Render AMBER templates for the system."""
        
        work = self.ctx.working_dir
        amber_dir = self.ctx.amber_dir
        amber.write_amber_templates(
            out_dir=amber_dir,
            sim=self.ctx.sim,
            membrane=self.ctx.sim.membrane_simulation,
            production=False,
        )
        logger.debug(f"[prepare_fe] Created amber files for {self.ctx.ligand}")
    
    def _create_simulation_dir(self) -> None:
        """Dispatch component-specific sim_files"""
        comp = self.ctx.comp.lower()
        if comp not in CREATE_SIMULATION_REGISTRY:
            raise NotImplementedError(f"No create_simulation registered for component '{comp}'")
        CREATE_SIMULATION_REGISTRY[comp](self.ctx)
        logger.debug(f"[prepare_fe] Created simulation files for {self.ctx.ligand}")
        
    def _create_box(self) -> None:
        """Render AMBER templates and build solvated/ionized system."""

        comp = self.ctx.comp.lower()
        if comp not in CREATE_BOX_REGISTRY:
            raise NotImplementedError(f"No create_box registered for component '{comp}'")
        CREATE_BOX_REGISTRY[comp](self.ctx)
        logger.debug(f"[prepare_fe] Created box for {self.ctx.ligand}")

    def _restraints(self) -> None:
        """Add restraints restraints as specified."""
        if self.ctx.win != -1 and COMPONENTS_LAMBDA_DICT[self.comp] == 'lambdas':
            return
        fn = RESTRAINT_REGISTRY.get(self.ctx.comp)
        fn(self, self.ctx)
        logger.debug(f"[{self.stage}] Added restraints for {self.ctx.ligand}")

    def _sim_files(self) -> None:
        """Create simulation input files."""
        fn = SIM_FILES_REGISTRY.get(self.ctx.comp)
        fn(self.ctx, self.component_windows)
        logger.debug(f"[{self.stage}] Created sim files for {self.ctx.ligand}")
        
    def _run_files(self) -> None:
        runfiles.write_fe_run_file(self.ctx, self.component_windows)