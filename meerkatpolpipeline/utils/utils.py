
from __future__ import annotations

import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from prefect.logging import get_run_logger


def add_timestamp_to_path(
    input_path: Path | str, timestamp: datetime | None = None
) -> Path:
    """Add a timestamp to a input path, where the timestamp is the
    current data and time. The time will be added to the name component
    before the file suffix. If the name component of the `input_path`
    has multiple suffixes than the timestamp will be added before the last.

    Args:
        input_path (Union[Path, str]): Path that should have a timestamp added
        timestamp: (datetime): The date-time to add. If None the current time is used. Defaults to None.
    Returns:
        Path: Updated path with a timestamp in the file name
    """
    input_path = Path(input_path)
    timestamp = timestamp if timestamp else datetime.now()

    time_str = timestamp.strftime("%Y%m%d-%H%M%S")
    new_name = f"{input_path.stem}-{time_str}{input_path.suffix}"
    output_path = input_path.with_name(new_name)

    return output_path

def execute_command(
    cmd: str | list[str],
    test: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a command (always as a list) with error handling.
    :param cmd: either a single string or a list of args
    :param test: if True, logs only and does not execute
    """
    logger = get_run_logger()

    # Normalize to list
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
        logger.info("Parsed command string into list: %s", cmd_list)
    else:
        cmd_list = cmd
        logger.info("Executing command list: %s", cmd_list)

    if test:
        logger.info("Test mode enabled; skipping command execution.")
        return None

    # If the first element is "bash", check the script exists
    if cmd_list and cmd_list[0] == "bash" and len(cmd_list) > 1:
        script_path = Path(cmd_list[1])
        if not script_path.exists():
            raise FileNotFoundError(
                f"Script not found at {script_path!r}; please verify the path."
            )

    try:
        result = subprocess.run(
            cmd_list,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (exit %d): %s", e.returncode, e.stderr)
        logger.info(e.stderr) #  or e.stdout or ""
        logger.info(e.returncode)
        raise

    return result

def check_create_symlink(symlink: Path, original_path: Path) -> Path:
    """
    check if 'symlink' exists, if not create it, linking to 'original_path'
    """
    logger = get_run_logger()
    if not symlink.exists():
        logger.info(f"Creating symlink at {symlink} pointing to {original_path}")
        symlink.symlink_to(original_path)
    else:
        logger.info(f"Symlink {symlink} already exists. Skipping symlink creation.")
    return symlink

def find_calibrated_ms(
        crosscal_base_dir: Path,
        preprocessed_ms: Path,
        look_in_subdirs: list = ["caracal", "casacrosscal"]
    ) -> Path | None:
    """
    If both crosscal steps are disabled, check for the existence of a calibrated
    measurement set in either the "caracal" or "casacrosscal" subdirectory

    Args:
        crosscal_base_dir (Path): Base directory where the crosscal directories are located.
        preprocessed_ms (Path): Path to the preprocessed measurement set.
        look_in_subdirs (list): List of subdirectories to look for the calibrated measurement set.
                                 Defaults to ["caracal", "casacrosscal"].
    Returns:
        Path | None: Path to the calibrated measurement set if found, otherwise None.
    """

    for subdir in look_in_subdirs:
        ms_path = crosscal_base_dir / subdir / (preprocessed_ms.stem + "-cal.ms")
        if ms_path.exists():
            return ms_path
    return None