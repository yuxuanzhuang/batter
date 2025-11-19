from loguru import logger
import subprocess as sp
import os
import pickle
from functools import wraps
from pathlib import Path

import shlex
import signal

import contextlib
import joblib
from tqdm import tqdm
import re



antechamber = 'antechamber'
tleap = 'tleap'
cpptraj = 'cpptraj'
parmchk2 = 'parmchk2'
charmmlipid2amber = 'charmmlipid2amber.py'

_repo_root = Path(__file__).resolve().parents[2]
usalign = str(_repo_root / "batter" / "utils" / "USalign")

#obabel = '/home/groups/rondror/software/openbabel/bin/obabel'
obabel = 'obabel'
    
#vmd = '/home/groups/rondror/software/vmd-1.9.4/bin/vmd'
vmd = 'vmd'

DEC_FOLDER_DICT = {
    'dd': 'dd',
    'sdr': 'sdr',
    'exchange': 'sdr',
}

COMPONENTS_LAMBDA_DICT = {
    'v': 'lambdas',
    'e': 'lambdas',
    'w': 'lambdas',
    'f': 'lambdas',
    'x': 'lambdas',
    'o': 'lambdas',
    'z': 'lambdas',
    's': 'lambdas',
    'y': 'lambdas',
    'a': 'attach_rest',
    'l': 'attach_rest',
    't': 'attach_rest',
    'r': 'attach_rest',
    'c': 'attach_rest',
    'm': 'attach_rest',
    'n': 'attach_rest',
}

FEP_COMPONENTS = list(COMPONENTS_LAMBDA_DICT.keys())

COMPONENTS_FOLDER_DICT = {
    'v': 'sdr',
    'e': 'sdr',
    'w': 'sdr',
    'f': 'sdr',
    'x': 'sdr',
    'o': 'sdr',
    'z': 'sdr',
    's': 'sdr',
    'y': 'sdr',
    'a': 'rest',
    'l': 'rest',
    't': 'rest',
    'r': 'rest',
    'c': 'rest',
    'm': 'rest',
    'n': 'rest',
}

COMPONENTS_DICT = {
        'rest': ['a', 'l', 't', 'c', 'r', 'm', 'n'],
        'dd': ['e', 'v', 'f', 'w', 'x', 'o', 's', 'z', 'y'],
    }


def run_with_log(command, level='debug', working_dir=None,
                 error_match=None, timeout=None, shell=True, env=None):
    """
    Run a subprocess command and log its output using loguru logger.

    Parameters
    ----------
    command : str | list[str]
        The command to execute. If str and shell=False, it will be shlex.split().
    level : {'debug','info','warning','error','critical'}
        Log level for command output.
    working_dir : str | None
        Working directory for the command.
    error_match : str | None
        If provided, stdout/stderr are searched for this string; if found, error is raised.
    timeout : float | None
        Seconds before forcibly timing out the process.
    shell : bool
        Whether to execute via shell. Default True.
    env : dict | None
        Extra environment variables.

    Raises
    ------
    RuntimeError on failure, segfault, matched error string, or timeout.
    """
    if working_dir is None:
        working_dir = os.getcwd()

    log_methods = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }
    log = log_methods.get(level)
    if log is None:
        raise ValueError(f"Invalid log level: {level}")

    # Normalize command
    if isinstance(command, str) and not shell:
        cmd = shlex.split(command)
    else:
        cmd = command

    logger.debug(f"Running command: {command!r}")
    logger.debug(f"Working directory: {working_dir}")

    try:
        result = sp.run(
            cmd,
            shell=shell,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
            check=False,
            cwd=working_dir,
            timeout=timeout,
            env=(None if env is None else {**os.environ, **env}),
        )
    except sp.TimeoutExpired as e:
        logger.info(f"Command timed out after {timeout}s: {command!r}")
        # e.output / e.stderr may be None if the proc was killed mid-flight
        if e.output:
            log("Command output before timeout:")
            for line in e.output.splitlines():
                log(line)
        if e.stderr:
            log("Command error output before timeout:")
            for line in e.stderr.splitlines():
                log(line)
        raise RuntimeError(f"Command timed out after {timeout}s: {command!r}") from e

    # Log outputs regardless of success
    if result.stdout:
        log("Command output:")
        for line in result.stdout.splitlines():
            log(line)
    if result.stderr:
        log("Command errors:")
        for line in result.stderr.splitlines():
            log(line)

    # Optional content-based failure
    if error_match and (error_match in result.stdout or error_match in result.stderr):
        raise RuntimeError(
            f"Command {command!r} reported an error matching {error_match!r}."
        )

    # Check exit status
    rc = result.returncode
    if rc == 0:
        return result

    # If terminated by signal, returncode is negative
    if rc < 0:
        sig = -rc
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = f"SIG{sig}"
        raise RuntimeError(
            f"Command {command!r} died with signal {sig_name} ({sig})."
        )

    # Non-zero exit code (regular failure)
    raise RuntimeError(
        f"Command {command!r} failed with return code {rc}."
    )


def log_info(func):
    """
    A wrapper function that print out current working directory and function name
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Running function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def save_state(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._save_state()
        return result
    return wrapper

def fail_report_wrapper(func):
    """
    Function to report failure of a method
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f'Error in {func}: {e}')
            logger.error(f'all args: {args}; kwargs: {kwargs}')
            raise
    return wrapper


def builder_fail_report(func):
    """
    Decorator to report failure to build a system.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f'Error in building {self.pose}: {e}')
            raise
    return wrapper


def safe_directory(func):
    """Decorator to ensure function returns to the original directory if an error occurs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_dir = os.getcwd()  # Save the current directory
        try:
            return func(*args, **kwargs)
        except Exception as e:
            os.chdir(original_dir)  # Return to original directory on failure
            raise e  # Re-raise the exception
    return wrapper


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    Reference https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def components_under(root: Path) -> list[str]:
    """fe/<comp>/ must be a directory; component name = folder name."""
    fe_root = root / "fe"
    if not fe_root.exists():
        return []
    return sorted([p.name for p in fe_root.iterdir() if p.is_dir() and p.name in FEP_COMPONENTS])
