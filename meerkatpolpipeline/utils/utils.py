
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

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
        timestamp: (Optional[datetime], optional): The date-time to add. If None the current time is used. Defaults to None.
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
    cmd: Union[str, list[str]],
    test: bool = False
) -> Optional[subprocess.CompletedProcess]:
    """
    Wrapper around subprocess.run with error handling.
    Supports both string commands (shell mode) and list commands.

    :param cmd: either a single string (will use shell=True)
                or a list of args (shell=False)
    :param test: if True, only logs the command without running it
    :returns: CompletedProcess if run; None if test mode
    """
    logger = get_run_logger()

    # If it's a string, we’ll run under a shell
    if isinstance(cmd, str):
        logger.info("Executing shell command: %s", cmd)
    else:
        # join for logging, but keep list for actual run
        logger.info("Executing command list: %s", " ".join(cmd))

    if test:
        logger.info("Test mode enabled; skipping execution.")
        return None

    # If we're calling bash, verify the script exists first
    # (handles both string and list forms)
    to_check = None
    if isinstance(cmd, str):
        parts = cmd.split()
        if parts and parts[0] == "bash" and len(parts) >= 2:
            to_check = parts[1]
    else:
        if cmd and cmd[0] == "bash" and len(cmd) >= 2:
            to_check = cmd[1]

    if to_check:
        script_path = Path(to_check)
        if not script_path.exists():
            raise FileNotFoundError(
                f"Script not found at {script_path!r}; "
                "please verify the path."
            )

    # Dispatch to subprocess.run
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (exit %d): %s", e.returncode, e.stderr)
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