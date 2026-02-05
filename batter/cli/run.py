"""Command-line interface for BATTER."""

from __future__ import annotations

from batter.cli.root import cli
from batter.cli import batch_cmds as _batch_cmds
from batter.cli import exec_cmds as _exec_cmds
from batter.cli import fe_cmds as _fe_cmds
from batter.cli import jobs_cmds as _jobs_cmds
from batter.cli import run_cmds as _run_cmds
from batter.cli.fek import fek_schedule

cli.add_command(fek_schedule)
