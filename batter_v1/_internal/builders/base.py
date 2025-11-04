from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from loguru import logger

from batter.config.simulation import SimulationConfig, MEMBRANE_EXEMPT_COMPONENTS
from batter._internal.ops import io as io_ops
from batter._internal.ops import simprep as sim_ops
from batter._internal.builders.interfaces import BuildContext


class BaseBuilder(ABC):
    """
    Minimal, forward-only stage/component builder.

    Common responsibilities:
      - create/reset build & window directories
      - create or copy simulation directories
      - expose shared context (ligand, residue_name, sim_config, etc.)

    Stage-specific responsibilities (implemented in subclasses):
      - _build_complex(), _create_box(), _restraints(),
        _pre_sim_files(), _sim_files(), _run_files()
    """

    stage: Optional[str] = None

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
        infe: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        abs_working_dir = Path(working_dir).resolve()
        self.ctx = BuildContext(
            ligand=ligand,
            residue_name=residue_name,
            param_dir_dict=param_dir_dict,
            sim=sim_config,
            working_dir=abs_working_dir,
            system_root=Path(system_root),
            comp=component,
            win=win,
            extra=extra or {},
        )

        # whether to enable infe
        self.infe = infe
        self.component_windows = component_windows
        self.ctx.working_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"[{self.stage or 'builder'}] initialized for ligand={ligand}, "
            f"residue={residue_name}, workdir={abs_working_dir}"
        )

    # ---- folders
    @property
    def build_dir(self) -> Path:
        return self.ctx.build_dir

    @property
    def window_dir(self) -> Path:
        """
        Naming:
          - win == -1  →  <comp>-1   (FE equil/scaffold)
          - win >= 0   →  <comp><win>  (lambda window directories: z0, z1, ...)
        """
        return self.ctx.window_dir

    @property
    def amber_dir(self) -> Path:
        return self.ctx.amber_dir

    @property
    def comp(self) -> str:
        return self.ctx.comp
        
    @property
    def membrane_builder(self) -> bool:
        sim_flag = getattr(self.ctx.sim, "_membrane_simulation", False)
        return sim_flag and self.ctx.comp not in MEMBRANE_EXEMPT_COMPONENTS

    # ---- main template
    def build(self) -> "BaseBuilder":
        logger.debug(
            f"building {self.ctx.ligand} [{self.stage or 'stage'}] "
            f"(comp={self.ctx.comp}, win={self.ctx.win}, residue={self.ctx.residue_name})"
        )

        # 1) complex (anchors) at win == -1 only
        if self.ctx.win == -1:
            io_ops.reset_dir(self.build_dir)
            anchor_ok = self._build_complex()
            if not anchor_ok:
                raise ValueError(f"anchors not found for ligand={self.ctx.ligand}.")
            self._create_amber_files()

        # 2) create / copy simulation dir
        self.window_dir.mkdir(parents=True, exist_ok=True)
        if self.ctx.win == -1:
            self._create_simulation_dir()
        else:
            sim_ops.copy_simulation_dir(
                source=self.ctx.working_dir / f"{self.ctx.comp}-1",
                dest=self.window_dir,
                sim=self.ctx.sim,
            )

        # 3–7) delegate to subclass
        if self.ctx.win == -1:
            self._create_box()
        self._restraints()
        if self.ctx.win == -1:
            self._pre_sim_files()
        self._sim_files()
        self._run_files()

        return self

    # ---- hooks to be specialized by subclasses
    @abstractmethod
    def _build_complex(self) -> bool:
        """Return True if anchors found/placed; False to prune."""
        raise NotImplementedError

    @abstractmethod
    def _create_amber_files(self) -> None:
        """Create AMBER templates in build_dir."""
        raise NotImplementedError

    @abstractmethod
    def _create_simulation_dir(self) -> None:
        """Create simulation directory for this stage."""
        raise NotImplementedError

    @abstractmethod
    def _create_box(self) -> None:
        """Create solvated/ionized box, write full/vac systems."""
        raise NotImplementedError

    @abstractmethod
    def _restraints(self) -> None:
        """Write any stage-specific restraints."""
        raise NotImplementedError

    def _pre_sim_files(self) -> None:
        """Optional pre-sim transforms."""
        return

    @abstractmethod
    def _sim_files(self) -> None:
        """Write stage-specific MD input files."""
        raise NotImplementedError

    @abstractmethod
    def _run_files(self) -> None:
        """Emit run scripts for this stage."""
        raise NotImplementedError