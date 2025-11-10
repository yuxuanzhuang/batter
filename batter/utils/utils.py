import contextlib
import os
from functools import wraps

import joblib
from loguru import logger


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
