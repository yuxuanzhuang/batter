from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from batter._internal.builders.base import BaseBuilder
from batter.config.simulation import SimulationConfig

class AlchemicalFEBuilder(BaseBuilder):
    """
    Builder for alchemical free energy stages (infe, fe, etc.).
    """
    stage = "fe"

    def __init__(
        self,
        ligand: str,
        residue_name: str,
        param_dir_dict: Dict[str, str],
        sim_config: SimulationConfig,
        component: str,
        component_windows: Dict[str, Any],
        working_dir: Path | str,
        system_root: Path | str,
        win: int = -1,
        *,
        # legacy extras used by the old code
        molr: Optional[str] = None,
        poser: Optional[str] = None,
        infe: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            ligand=ligand,
            residue_name=residue_name,
            param_dir_dict=param_dir_dict,
            sim_config=sim_config,
            component=component,
            component_windows=component_windows,
            working_dir=working_dir,
            system_root=system_root,
            win=win,
            extra={**(extra or {}), "molr": molr, "poser": poser, "infe": infe},
        )
        # Legacy fields the old code reads directly
        self.molr = molr
        self.poser = poser
        self.infe = infe

    # ---- property aliases to match legacy names/paths ----
    # Legacy code uses these attribute names; keep them working by aliasing.

    @property
    def pose(self) -> str:
        return self.ctx.ligand

    @property
    def working_dir(self) -> Path:
        return self.ctx.working_dir

    @property
    def sim_config(self) -> SimulationConfig:
        return self.ctx.sim

    @property
    def component_windows_dict(self) -> Dict[str, Any]:
        return self.component_windows

    @property
    def comp(self) -> str:
        return self.ctx.comp

    @property
    def win(self) -> int:
        return self.ctx.win

    @property
    def other_mol(self) -> list[str]:
        # legacy code expects list; SimulationConfig already provides list
        return list(getattr(self.ctx.sim, "other_mol", []) or [])

    @property
    def lipid_mol(self) -> list[str]:
        return list(getattr(self.ctx.sim, "lipid_mol", []) or [])

    @property
    def membrane_builder(self) -> bool:
        # legacy checked a flag; mirror from SimulationConfig if present
        return bool(getattr(self.ctx.sim, "membrane_builder", False))

    # Directory names expected by legacy code:
    @property
    def build_file_folder(self) -> str:
        # e.g. "fe_build_files"
        return f"{self.comp}_build_files"

    @property
    def amber_files_folder(self) -> str:
        # e.g. "fe_amber_files"
        return f"{self.comp}_amber_files"

    @property
    def run_files_folder(self) -> str:
        # legacy code expects a run-files tree copied/templated by _create_run_files
        return f"{self.comp}_run_files"

    # ------------------ HOOKS ------------------
    # Below, paste the bodies of your legacy methods EXACTLY as-is,
    # with NO logic changes, only name references (self.pose, self.sim_config, etc.)
    # will resolve via the aliases above.

    def _build_complex(self) -> bool:
        # --- BEGIN: paste legacy FreeEnergyBuilder._build_complex body here ---
        # (Use the exact code you pasted; it will compile against the aliases/properties above.)
        raise NotImplementedError("Paste legacy _build_complex body here.")
        # --- END ---

    def _create_amber_files(self) -> None:
        # maps to legacy: _create_run_files
        # --- BEGIN: paste legacy FreeEnergyBuilder._create_run_files body here ---
        raise NotImplementedError("Paste legacy _create_run_files body here.")
        # --- END ---

    def _create_box(self) -> None:
        # maps to legacy: _create_simulation_dir
        # --- BEGIN: paste legacy FreeEnergyBuilder._create_simulation_dir body here ---
        raise NotImplementedError("Paste legacy _create_simulation_dir body here.")
        # --- END ---

    def _restraints(self) -> None:
        # --- BEGIN: paste legacy FreeEnergyBuilder._restraints body here ---
        raise NotImplementedError("Paste legacy _restraints body here.")
        # --- END ---

    def _sim_files(self) -> None:
        # --- BEGIN: paste legacy FreeEnergyBuilder._sim_files body here ---
        raise NotImplementedError("Paste legacy _sim_files body here.")
        # --- END ---

    def _run_files(self) -> None:
        # --- BEGIN: paste legacy FreeEnergyBuilder._run_files body here ---
        raise NotImplementedError("Paste legacy _run_files body here.")
        # --- END ---