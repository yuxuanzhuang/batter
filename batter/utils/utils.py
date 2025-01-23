from loguru import logger
import subprocess as sp
import os

antechamber = 'antechamber'
tleap = 'tleap'
cpptraj = 'cpptraj'
parmchk2 = 'parmchk2'
charmmlipid2amber = 'charmmlipid2amber.py'

obabel = '/home/groups/rondror/software/openbabel/bin/obabel'
if not os.path.exists(obabel):
    obabel = 'obabel'
    
vmd = '/home/groups/rondror/software/vmd-1.9.4/bin/vmd'
if not os.path.exists(vmd):
    vmd = 'vmd'

COMPONENTS_LAMBDA_DICT = {
    'v': 'lambdas',
    'e': 'lambdas',
    'w': 'lambdas',
    'f': 'lambdas',
    'x': 'lambdas',
    'a': 'attach_rest',
    'l': 'attach_rest',
    't': 'attach_rest',
    'r': 'attach_rest',
    'c': 'attach_rest',
    'm': 'attach_rest',
    'n': 'attach_rest',
}

COMPONENTS_FOLDER_DICT = {
    'v': 'sdr',
    'e': 'sdr',
    'w': 'sdr',
    'f': 'sdr',
    'x': 'sdr',
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
        'dd': ['e', 'v', 'f', 'w', 'x'],
    }



def run_with_log(command, level='debug', working_dir=None,
                 error_match=None):
    """
    Run a subprocess command and log its output using loguru logger.

    Parameters
    ----------
    command : str
        The command to execute.
    level : str, optional
        The log level for logging the command output.
        Default is 'debug'.
    working_dir : str, optional
        The working directory for the command. Default is
        the current working directory.
    error_match : str, optional
        If provided, the command stdout and stderr 
        will be checked for this string.
        If the string is found,
        a subprocess.CalledProcessError will be
        raised. Default is None.

    Raises
    ------
    ValueError
        If an invalid log level is provided.
    subprocess.CalledProcessError
        If the command exits with a non-zero status.
    """
    if working_dir is None:
        working_dir = os.getcwd()
    # Map log level to loguru logger methods
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

    logger.debug(f"Running command: {command}")
    logger.debug(f"Working directory: {working_dir}")
    try:
        # Run the command and capture output
        result = sp.run(
            command,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
            check=True,
            cwd=working_dir
        )

        if error_match:
            if error_match in result.stdout or error_match in result.stderr:
                logger.info(f"Command failed with matched error: "
                            f"{error_match}")
                logger.info(f"Command output: {result.stdout}")
                logger.info(f"Command errors: {result.stderr}")
                raise sp.CalledProcessError(
                    returncode=1,
                    cmd=command,
                    output=result.stdout,
                    stderr=result.stderr
                )
                
        # Log stdout and stderr line by line
        if result.stdout:
            log("Command output:")
            for line in result.stdout.splitlines():
                log(line)

        if result.stderr:
            log("Command errors:")
            for line in result.stderr.splitlines():
                log(line)

    except sp.CalledProcessError as e:
        logger.info(f"Command failed with return code {e.returncode}")
        if e.stdout:
            log("Command output before failure:")
            for line in e.stdout.splitlines():
                log(line)
        if e.stderr:
            log("Command error output:")
            for line in e.stderr.splitlines():
                log(line)
        raise

def log_info(func):
    """
    A wrapper function that print out current working directory and function name
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Running function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper