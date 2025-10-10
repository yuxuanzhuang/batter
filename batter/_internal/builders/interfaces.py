# batter/_internal/builders/interfaces.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Callable, Mapping, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from batter.config.simulation import SimulationConfig

# ---------------------------------------------------------------------------
# Enumerations (stringly-typed to stay friendly with existing code/CLI)
# ---------------------------------------------------------------------------

# Stages used by the high-level pipeline (ABFE etc.)
Stage = str  # e.g., "equil", "prepare_fe", "prepare_fe_windows", "fe_equil", "fe", "analyze"

# Component codes used by legacy builders (kept for compatibility)
# q: equil, e/v: alchemical, n/m: REST, x/o/z/s/y: other legacy modes
Component = str  # e.g., "q", "e", "v", "n", "m", "x", "o", "z", "s", "y"


# ---------------------------------------------------------------------------
# Builder Interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildContext:
    ligand: str
    residue_name: str
    param_dir_dict: Mapping[str, str]
    working_dir: Path
    system_root: Path
    comp: str
    win: int
    sim: SimulationConfig
    anchors: Tuple[str, str, str] | None = None
    lipid_mol: Tuple[str, ...] = ()
    other_mol: Tuple[str, ...] = ()
    extra: Mapping[str, Any] | None = None


@runtime_checkable
class ISystemBuilder(Protocol):
    """
    Minimal interface for all system builders.

    Notes
    -----
    - Concrete builders (Equilibration, FE prep, FE windows, etc.) should
      implement these methods.
    - The pipeline code depends only on this interface (not on concrete classes).
    """

    # Public attributes (read-only from pipeline POV)
    ctx: BuildContext
    stage: Stage
    component: Component
    win: int  # -1 for pre-window stages

    # ----- Lifecycle -----
    def build(self) -> ISystemBuilder:
        """
        Execute the full builder lifecycle inside its working directory:
        1) _build_complex
        2) _create_box
        3) _restraints
        4) _pre_sim_files
        5) _sim_files
        6) _run_files

        Returns
        -------
        ISystemBuilder
            Self (for chaining).
        """
        ...

    # ----- Hook methods (overridden by subclasses) -----
    def _build_complex(self) -> bool:
        """
        Prepare/align system files and locate anchors.
        Return False to signal the pipeline that anchors were not found.
        """
        ...

    def _create_box(self) -> None:
        """Create solvated/ionized box and write topology/coordinates (full/vac)."""
        ...

    def _restraints(self) -> None:
        """Write restraint definitions (e.g., disang files, cv.in)."""
        ...

    def _pre_sim_files(self) -> None:
        """Optional pre-processing before writing the final sim files."""
        ...

    def _sim_files(self) -> None:
        """Write AMBER (or other engine) input decks for this stage/window."""
        ...

    def _run_files(self) -> None:
        """Write job scripts (SLURM/local) for this stage/window."""
        ...


# ---------------------------------------------------------------------------
# Factory Interface
# ---------------------------------------------------------------------------

@runtime_checkable
class IBuilderFactory(Protocol):
    """
    Factory that returns a concrete builder for a (stage, component) pair.
    This mirrors your legacy `BuilderFactory.get_builder(...)` but typed
    against the interface, so downstream only sees `ISystemBuilder`.
    """

    def get_builder(
        self,
        *,
        stage: Stage,
        ligand: str,
        sim_config: SimulationConfig,
        component_windows_dict: Mapping[str, object],
        working_dir: Path | str,
        win: int = 0,
        component: Component = "q",
        # legacy knobs (optional):
        molr: Optional[str] = None,
        ligandr: Optional[str] = None,
        infe: bool = False,
    ) -> ISystemBuilder:
        ...


# ---------------------------------------------------------------------------
# Registry Type (optional helper)
# ---------------------------------------------------------------------------

# A registry maps (stage, component) to a callable that produces a builder.
BuilderCtor = Callable[
    [],
    ISystemBuilder,  # You can curry arguments in closures when registering
]

BuilderKey = Tuple[Stage, Component]
BuilderRegistry = Mapping[BuilderKey, BuilderCtor]