"""Utilities for configuring execution backends used by the orchestrator."""

from __future__ import annotations

from loguru import logger

from batter.exec.local import LocalBackend

MISSING_DEPENDENCY_MSG = (
    "Required optional dependency '{name}' not found. "
    "Install it to enable local execution handlers (e.g., `pip install {package}`)."
)


def register_local_handlers(backend: LocalBackend) -> None:
    """Register built-in pipeline handlers on the local backend.

    Parameters
    ----------
    backend : LocalBackend
        Backend instance that should receive the default handler mapping.
    """
    try:
        from batter.exec.handlers.system_prep import system_prep as _system_prep
        from batter.exec.handlers.system_prep_masfe import system_prep_masfe as _system_prep_masfe
        from batter.exec.handlers.param_ligands import param_ligands as _param_ligands
        from batter.exec.handlers.prepare_equil import prepare_equil_handler as _prepare_equil
        from batter.exec.handlers.equil import equil_handler as _equil
        from batter.exec.handlers.equil_analysis import equil_analysis_handler as _equil_analysis
        from batter.exec.handlers.prepare_fe import prepare_fe_handler as _prepare_fe
        from batter.exec.handlers.prepare_fe import prepare_fe_windows_handler as _prepare_fe_windows
        from batter.exec.handlers.fe import fe_equil_handler as _fe_equil
        from batter.exec.handlers.fe import fe_handler as _fe
        from batter.exec.handlers.fe_analysis import analyze_handler as _analyze
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise RuntimeError(
            MISSING_DEPENDENCY_MSG.format(
                name=missing,
                package="openff-toolkit"
                if "openff" in missing
                else missing,
            )
        ) from exc

    backend.register("system_prep", _system_prep)
    backend.register("system_prep_asfe", _system_prep_masfe)
    backend.register("param_ligands", _param_ligands)
    backend.register("prepare_equil", _prepare_equil)
    backend.register("equil", _equil)
    backend.register("equil_analysis", _equil_analysis)
    backend.register("prepare_fe", _prepare_fe)
    backend.register("prepare_fe_windows", _prepare_fe_windows)
    backend.register("fe_equil", _fe_equil)
    backend.register("fe", _fe)
    backend.register("analyze", _analyze)

    logger.debug("Registered LOCAL handlers: {}", list(backend._handlers.keys()))
