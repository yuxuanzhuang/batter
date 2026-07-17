"""Utilities for configuring execution backends used by the orchestrator."""

from __future__ import annotations

import importlib

from loguru import logger

from batter.exec.local import LocalBackend

MISSING_DEPENDENCY_MSG = (
    "Required optional dependency '{name}' not found. "
    "Install it to enable local execution handlers (e.g., `pip install {package}`). Note `kartograf` is only available with conda."
)


def _missing_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    missing = exc.name or "unknown"
    package = "openff-toolkit" if "openff" in missing else missing
    return RuntimeError(
        MISSING_DEPENDENCY_MSG.format(name=missing, package=package)
    )


def _lazy_import_handler(module_name: str, attr_name: str):
    """Return a handler that imports optional-dependency-heavy code on demand."""
    def _handler(step, system, params):
        try:
            module = importlib.import_module(module_name)
            handler = getattr(module, attr_name)
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error(exc) from exc
        return handler(step, system, params)

    return _handler


def register_local_handlers(backend: LocalBackend) -> None:
    """Register built-in pipeline handlers on the local backend.

    Parameters
    ----------
    backend : LocalBackend
        Backend instance that should receive the default handler mapping.

    Raises
    ------
    RuntimeError
        If optional handler dependencies (for example ``openff-toolkit``) are missing.
    """
    handler_specs = {
        "system_prep": ("batter.exec.handlers.system_prep", "system_prep"),
        "system_prep_asfe": (
            "batter.exec.handlers.system_prep_masfe",
            "system_prep_masfe",
        ),
        "param_ligands": ("batter.exec.handlers.param_ligands", "param_ligands"),
        "prepare_rbfe": ("batter.exec.handlers.prepare_rbfe", "prepare_rbfe_handler"),
        "prepare_equil": (
            "batter.exec.handlers.prepare_equil",
            "prepare_equil_handler",
        ),
        "equil": ("batter.exec.handlers.equil", "equil_handler"),
        "equil_analysis": (
            "batter.exec.handlers.equil_analysis",
            "equil_analysis_handler",
        ),
        "prepare_fe": ("batter.exec.handlers.prepare_fe", "prepare_fe_handler"),
        "pre_prepare_fe": ("batter.exec.handlers.prepare_fe", "prepare_fe_handler"),
        "prepare_fe_windows": (
            "batter.exec.handlers.prepare_fe",
            "prepare_fe_windows_handler",
        ),
        "fe_equil": ("batter.exec.handlers.fe", "fe_equil_handler"),
        "pre_fe_equil": ("batter.exec.handlers.fe", "fe_equil_handler"),
        "fe": ("batter.exec.handlers.fe", "fe_handler"),
        "analyze": ("batter.exec.handlers.fe_analysis", "analyze_handler"),
    }

    for step_name, (module_name, attr_name) in handler_specs.items():
        backend.register(step_name, _lazy_import_handler(module_name, attr_name))

    logger.debug("Registered LOCAL handlers: {}", list(backend._handlers.keys()))
