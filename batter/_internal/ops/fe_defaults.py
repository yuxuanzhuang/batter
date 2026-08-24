from __future__ import annotations


DEFAULT_FE_SEED_LAMBDA_STATES = 10
DEFAULT_FE_SEED_STEPS_PER_STATE = 10_000
DEFAULT_FE_WINDOW_EQUIL_TIME_PS = 50.0


def fe_window_equil_steps(timestep_ps: float) -> int:
    """Return the integral step count for the default FE window equilibration."""
    timestep_ps = float(timestep_ps)
    if timestep_ps <= 0:
        raise ValueError("timestep_ps must be positive")

    steps = DEFAULT_FE_WINDOW_EQUIL_TIME_PS / timestep_ps
    rounded_steps = round(steps)
    if abs(steps - rounded_steps) > 1e-9:
        raise ValueError(
            f"{DEFAULT_FE_WINDOW_EQUIL_TIME_PS} ps is not an integral number "
            f"of steps at dt={timestep_ps} ps"
        )
    return int(rounded_steps)
