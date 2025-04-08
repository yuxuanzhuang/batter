import os
import subprocess
from datetime import datetime
from loguru import logger
import time

class SLURMJob:
    def __init__(self, filename, partition=None, jobname=None):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist")

        self.filename = filename
        self.path = os.path.dirname(filename)
        self.file_basename = os.path.basename(filename)
        self.partition = partition
        self.jobname = jobname
        self.jobid = None

    def submit(self,
               overwrite=False,
               requeue=False,
               other_env=None):
        """
        Submit the job to the SLURM queue.
        It will be tried three times in case of failure.
        """
        self.overwrite = overwrite
        self.other_env = other_env
        if requeue:
            for _ in range(3):
                try:
                    self._requeue()
                    break
                except RuntimeError as e:
                    logger.error(f"Failed to requeue job: {e}; retrying in 30 seconds")
                    time.sleep(30)
            else:
                raise RuntimeError("Failed to requeue job after 3 attempts.")
        for _ in range(3):
            try:
                self._submit()
                break
            except RuntimeError as e:
                logger.error(f"Failed to submit job: {e}; retrying in 30 seconds")
                time.sleep(30)
        else:
            raise RuntimeError("Failed to submit job after 3 attempts.")

    def _submit(self):

        # Prepare the environment: copy current environment and update OVERWRITE variable.
        env = os.environ.copy()
        env["OVERWRITE"] = "1" if self.overwrite else "0"
        if self.other_env:
            env.update(self.other_env)

        cmd = ["sbatch"]
        if self.partition:
            cmd.append(f"--partition={self.partition}")
        if self.jobname:
            cmd.append(f"--job-name={self.jobname}")
        cmd.append(self.file_basename)

        result = subprocess.run(
            cmd,
            cwd=self.path,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "sbatch command failed.\n"
                f"Exit code: {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        # Expected output: "Submitted batch job <jobid>"
        out = result.stdout.strip()
        parts = out.split()
        if len(parts) < 4 or parts[:3] != ["Submitted", "batch", "job"]:
            raise RuntimeError(f"Unexpected sbatch output: {out}")

        self.jobid = parts[3]  # The fourth token is the job ID
        logger.debug(f"JobID {self.jobid} submitted.")

    def _requeue(self):
        env = os.environ.copy()
        env["OVERWRITE"] = "1" if self.overwrite else "0"

        if not self.jobid:
            raise RuntimeError("No jobid. Have you submitted the job yet?")

        result = subprocess.run(
            ["scontrol", "requeue", self.jobid],
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            raise RuntimeError(
                "scontrol command failed.\n"
                f"Exit code: {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        logger.debug(f"JobID {self.jobid} requeued.")

    def check_status(self):
        if not self.jobid:
            raise RuntimeError("No jobid. Have you submitted the job yet?")

        result = subprocess.run(
            ["scontrol", "show", "jobid", "-dd", self.jobid],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "scontrol command failed.\n"
                f"Exit code: {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        out = result.stdout
        logger.debug(f"Check status output:\n{out}")

        job_state = None
        runtime_str = None

        for token in out.split():
            if token.startswith("JobState="):
                job_state = token.split("=")[1]
            elif token.startswith("RunTime="):
                runtime_str = token.split("=")[1]

        # Log the parsed info
        if job_state:
            logger.info(f"JobID {self.jobid} is in state {job_state}")
        if runtime_str:
            rt = datetime.strptime(runtime_str, "%H:%M:%S") - datetime(1900, 1, 1)
            logger.info(f"Job {self.jobid} has run for {rt}")

    def cancel_worker(self):
        if not self.jobid:
            logger.warning("No jobid available to cancel.")
            return

        result = subprocess.run(
            ["scancel", self.jobid],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "scancel command failed.\n"
                f"Exit code: {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        logger.debug(f"JobID {self.jobid} canceled.")

    def is_still_running(self):
        """
        Returns False if the job is in a final state (COMPLETED, FAILED, CANCELLED, etc.),
        or if the job no longer appears in scontrol (which usually means it's done).
        Returns True if it's still running or pending in the queue.
        """
        if not self.jobid:
            # If there's no job ID at all, treat that as "not running".
            return False

        # Query scontrol
        result = subprocess.run(
            ["scontrol", "show", "jobid", "-dd", self.jobid],
            capture_output=True,
            text=True
        )

        # If scontrol fails, it may mean the job ended so quickly
        # that it no longer shows up in the queue (or other errors).
        if result.returncode != 0:
            logger.debug(
                "scontrol did not find the job or returned an error. "
                "Assuming the job is finished."
            )
            return False

        out = result.stdout
        job_state = None
        for token in out.split():
            if token.startswith("JobState="):
                job_state = token.split("=")[1]
                break

        # If we can’t find JobState, assume it’s finished.
        if not job_state:
            return False

        # Slurm "final" states can include COMPLETED, CANCELLED, FAILED, TIMEOUT, etc.
        # Adjust this set as needed for your cluster’s usage.
        finished_states = {
            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
            "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"
        }
        if job_state in finished_states:
            return False
        else:
            return True