"""Base class for stage-specific system builders."""

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
    """Minimal, forward-only stage/component builder.

    Responsibilities shared by all builders include creating/resetting
    directories, preparing simulation inputs, and exposing the build context
    (ligand identifiers, residue names, simulation config, etc.).
    Subclasses customize the workflow via the protected hook methods.
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
        """Initialize the builder with concrete ligand/system metadata.

        Parameters
        ----------
        ligand : str
            Ligand identifier (matches directory layout and artifact cache).
        residue_name : str
            Residue name to use when referencing the ligand in structures.
        param_dir_dict : dict[str, str]
            Mapping of artifact keys to parameter directories.
        sim_config : SimulationConfig
            Resolved simulation configuration for this workflow.
        component : str
            Legacy component code (e.g., ``"q"``, ``"z"``) used to name outputs.
        component_windows : dict[str, Any]
            Component-specific lambda/window metadata.
        working_dir : str or Path
            Root working directory for this ligand/system.
        system_root : str or Path
            Root path for reusable system-level data.
        win : int, default -1
            Window index currently being prepared; ``-1`` indicates scaffold/equil.
        infe : bool, default False
            Whether INFE (intermediate nonequilibrium) artifacts should be produced.
        extra : dict[str, Any], optional
            Optional opaque metadata forwarded to downstream helpers.
        """
        abs_working_dir = Path(working_dir).resolve()
        ctx_extra: Dict[str, Any] = dict(extra or {})
        ctx_extra.setdefault("infe", bool(infe))

        self.ctx = BuildContext(
            ligand=ligand,
            residue_name=residue_name,
            param_dir_dict=param_dir_dict,
            sim=sim_config,
            working_dir=abs_working_dir,
            system_root=Path(system_root),
            comp=component,
            win=win,
            extra=ctx_extra,
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
        """Path where intermediate build artifacts are written."""
        return self.ctx.build_dir

    @property
    def window_dir(self) -> Path:
        """Window-specific directory for the stage/component pair."""
        return self.ctx.window_dir

    @property
    def amber_dir(self) -> Path:
        """Directory containing rendered AMBER templates for this component."""
        return self.ctx.amber_dir

    @property
    def comp(self) -> str:
        """Return the component code backing this builder instance."""
        return self.ctx.comp
        
    @property
    def membrane_builder(self) -> bool:
        """Return True when membrane-specific handling is required.

        Returns
        -------
        bool
            ``True`` when the simulation config declares a membrane system and
            the current component is not listed in ``MEMBRANE_EXEMPT_COMPONENTS``.

        Raises
        ------
        AttributeError
            If the simulation config lacks the ``membrane_simulation`` flag.
        """
        if not hasattr(self.ctx.sim, "membrane_simulation"):
            raise AttributeError(
                "SimulationConfig is missing 'membrane_simulation'. "
                "Please add this field to the run configuration."
            )
        sim_flag = self.ctx.sim.membrane_simulation
        return sim_flag and self.ctx.comp not in MEMBRANE_EXEMPT_COMPONENTS

    # ---- main template
    def build(self) -> "BaseBuilder":
        """Execute the canonical build pipeline for the configured window.

        Returns
        -------
        BaseBuilder
            Self, to support fluent chaining if desired.
        """
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
        """Build/align the receptor–ligand complex and locate anchors.

        Returns
        -------
        bool
            ``True`` when anchors were found; ``False`` to signal pruning.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_amber_files(self) -> None:
        """Create AMBER templates in ``build_dir`` based on the simulation config."""
        raise NotImplementedError

    @abstractmethod
    def _create_simulation_dir(self) -> None:
        """Produce the stage-specific simulation directory structure."""
        raise NotImplementedError

    @abstractmethod
    def _create_box(self) -> None:
        """Create the solvated/ionized box and any full/vacuum system files."""
        raise NotImplementedError

    @abstractmethod
    def _restraints(self) -> None:
        """Write any stage-specific restraints (disang, cv, or related files)."""
        raise NotImplementedError

    def _pre_sim_files(self) -> None:
        """Optional transformation step before rendering simulation input files."""
        return

    @abstractmethod
    def _sim_files(self) -> None:
        """Write MD engine input files for this stage/window."""
        raise NotImplementedError

    @abstractmethod
    def _run_files(self) -> None:
        """Emit run scripts or job submission files for this stage/window."""
        raise NotImplementedError
