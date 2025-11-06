"""CLI commands for FE Toolkit (fetkutils) helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from batter.utils.fekutils.fetkutils import (
    SimpleScheduleOpt,
    SimpleScheduleRead,
)


def _run_schedule_opt(
    nlam: int,
    data_dir: Path,
    temp: float,
    start: float,
    stop: float,
    ssc: bool,
    sym: bool,
    pso: bool,
    ar: bool,
    kl: bool,
    clean: bool,
    maxgap: float,
    spen: float,
    digits: int,
    verbose: bool,
):
    """Wrapper for SimpleScheduleOpt returning the info tuple."""
    return SimpleScheduleOpt(
        nlam,
        str(data_dir),
        temp,
        start,
        stop,
        ssc,
        sym,
        pso,
        ar,
        kl,
        clean,
        maxgap,
        spen,
        digits,
        verbose,
    )


def _run_schedule_read(
    read: str,
    data_dir: Path,
    temp: float,
    start: float,
    stop: float,
    ssc: bool,
    alpha0: float,
    alpha1: Optional[float],
    pso: bool,
    ar: bool,
    kl: bool,
    clean: bool,
    digits: int,
    verbose: bool,
):
    """Wrapper for SimpleScheduleRead returning the info tuple."""
    return SimpleScheduleRead(
        read,
        str(data_dir),
        temp,
        start,
        stop,
        ssc,
        alpha0,
        alpha1,
        pso,
        ar,
        kl,
        clean,
        digits,
        verbose,
    )


def _write_plot(
    info,
    interp,
    out_path: Path,
    label: str,
) -> None:
    """Generate the heatmap + path projection plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = 151
    ls = np.linspace(0.0, 1.0, m)
    mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            mat[i, j] = interp.InterpValueFromLambdas(ls[i], ls[j])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6.5, 2.9, forward=True)

    pad = 0.08
    fontsize = 8

    fig.suptitle(f"Predicted {label}", fontsize=fontsize)

    for lam in info.lams:
        ax1.plot([lam, lam], [0, 1], c="r", ls=":", lw=1)

    ax1.plot(info.midpts, info.vals, c="k", ls="-", lw=1)
    ax1.scatter(info.midpts, info.vals, c="k", s=3)

    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylabel(label, fontsize=fontsize, labelpad=pad)
    ax1.set_xlabel(r"${\lambda}$", fontsize=fontsize, labelpad=pad)
    ax1.tick_params(axis="both", which="major", labelsize=fontsize, pad=1.5)
    ax1.set_aspect("equal")

    im2 = ax2.imshow(
        mat,
        cmap="bwr",
        interpolation="nearest",
        vmin=0,
        vmax=min(2, np.amax(mat)),
        origin="lower",
        extent=[0, 1, 0, 1],
    )

    ax2.scatter(info.lams[:-1], info.lams[1:], c="k", s=3)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)

    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.xaxis.set_major_locator(MultipleLocator(0.2))
    ax2.set_xlabel(r"${\lambda_{1}}$", fontsize=fontsize, labelpad=pad)
    ax2.set_ylabel(r"${\lambda_{2}}$", fontsize=fontsize, labelpad=pad)
    for tick in cax2.get_yticklabels():
        tick.set_fontsize(fontsize)
    cax2.set_ylabel(label, fontsize=fontsize, labelpad=pad)
    cax2.tick_params(axis="both", which="major", labelsize=fontsize, pad=1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


@click.command("fek-schedule")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--opt", type=int, help="Optimize a schedule with N lambda points.")
@click.option(
    "--varlam-multiple",
    "varlam_multiple",
    type=int,
    default=4,
    show_default=True,
    help="Increase schedule size in multiples of this when --varlam-minval triggers.",
)
@click.option(
    "--varlam-minval",
    "varlam_minval",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum acceptable PSO/AR/KL; bump schedule size if optimisation result is below.",
)
@click.option(
    "--read",
    type=str,
    help="Read a schedule from FILE or interpret as count when used with --ssc.",
)
@click.option("--pso", is_flag=True, help="Optimise based on predicted phase-space overlap.")
@click.option("--ar", is_flag=True, help="Optimise based on predicted replica-exchange ratio.")
@click.option("--kl", is_flag=True, help="Optimise based on predicted KL divergence.")
@click.option("--ssc", is_flag=True, help="Restrict optimisation to SSC(alpha) schedules.")
@click.option("--alpha0", type=float, default=2.0, show_default=True, help="Alpha at lambda=0 for SSC schedules.")
@click.option("--alpha1", type=float, help="Alpha at lambda=1 for SSC schedules (defaults to alpha0).")
@click.option("-o", "--out", type=click.Path(dir_okay=False, path_type=Path), help="Write schedule to FILE.")
@click.option("--sym", is_flag=True, help="Force schedule symmetry around lambda = 0.5.")
@click.option("--maxgap", type=float, default=1.0, show_default=True, help="Maximum allowable gap between lambdas.")
@click.option("--spen", type=float, default=0.1, show_default=True, help="Small penalty threshold for optimisation.")
@click.option("--digits", type=int, default=5, show_default=True, help="Round schedule to this many decimal digits.")
@click.option("--start", type=float, default=0.0, show_default=True, help="Fraction to trim from simulation beginning.")
@click.option("--stop", type=float, default=1.0, show_default=True, help="Fraction to trim after this point.")
@click.option("--clean", is_flag=True, help="Apply smoothing to damp noise and distant overlaps.")
@click.option("-T", "--temp", type=float, default=298.0, show_default=True, help="Simulation temperature (K).")
@click.option("--plot", type=click.Path(dir_okay=False, path_type=Path), help="Write heatmap to IMAGE file.")
@click.option("--verbose/--quiet", default=False, show_default=False, help="Print diagnostic output from fetkutils.")
def fek_schedule(  # noqa: PLR0913
    directory: Path,
    opt: Optional[int],
    varlam_multiple: int,
    varlam_minval: float,
    read: Optional[str],
    pso: bool,
    ar: bool,
    kl: bool,
    ssc: bool,
    alpha0: float,
    alpha1: Optional[float],
    out: Optional[Path],
    sym: bool,
    maxgap: float,
    spen: float,
    digits: int,
    start: float,
    stop: float,
    clean: bool,
    temp: float,
    plot: Optional[Path],
    verbose: bool,
):
    """Optimise or analyse lambda schedules using the FE Toolkit utilities."""
    if opt is not None and read is not None:
        raise click.UsageError("--opt and --read are mutually exclusive.")

    metric_flags = sum(bool(flag) for flag in (pso, ar, kl))
    if metric_flags != 1:
        raise click.UsageError("Select exactly one of --pso, --ar, or --kl.")

    if opt is None and read is None:
        raise click.UsageError("Either --opt or --read must be provided.")

    info = None
    pedata = None
    interp = None

    if opt is not None:
        if varlam_minval > 0 and opt % varlam_multiple != 0:
            raise click.BadParameter(
                f"--opt ({opt}) must be divisible by --varlam-multiple ({varlam_multiple}).",
                param_hint="--opt",
            )

        nlam = opt
        if varlam_minval > 0:
            nlam = opt - varlam_multiple
            for _ in range(20):
                nlam += varlam_multiple
                info, pedata, interp = _run_schedule_opt(
                    nlam,
                    directory,
                    temp,
                    start,
                    stop,
                    ssc,
                    sym,
                    pso,
                    ar,
                    kl,
                    clean,
                    maxgap,
                    spen,
                    digits,
                    verbose,
                )
                if info.minval > varlam_minval:
                    break
            if info.minval < varlam_minval:
                click.echo(
                    f"[warning] Schedule with size {nlam} produced minval={info.minval:.3f}, below target {varlam_minval}",
                    err=True,
                )
        else:
            info, pedata, interp = _run_schedule_opt(
                nlam,
                directory,
                temp,
                start,
                stop,
                ssc,
                sym,
                pso,
                ar,
                kl,
                clean,
                maxgap,
                spen,
                digits,
                verbose,
            )
    else:
        resolved_read = read
        if ssc and read is not None:
            # treat as integer count if path-like value supplied
            try:
                int(read)
            except ValueError as exc:
                raise click.BadParameter("--read must be an integer when used with --ssc.") from exc
        elif read is None:
            raise click.UsageError("--read requires a value.")
        else:
            resolved_path = Path(read)
            if not resolved_path.exists():
                raise click.BadParameter(f"Schedule file not found: {resolved_path}", param_hint="--read")
            resolved_read = str(resolved_path)

        info, pedata, interp = _run_schedule_read(
            resolved_read,
            directory,
            temp,
            start,
            stop,
            ssc,
            alpha0,
            alpha1,
            pso,
            ar,
            kl,
            clean,
            digits,
            verbose,
        )

    assert info is not None and pedata is not None and interp is not None  # for type checkers

    click.echo(str(info))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        info.Save(str(out))
        click.echo(f"Wrote schedule to {out}")

    if plot is not None:
        label = "PSO"
        if ar:
            label = "AR"
        elif kl:
            label = "exp(-KL)"
        _write_plot(info, interp, plot, label)
        click.echo(f"Wrote plot to {plot}")
