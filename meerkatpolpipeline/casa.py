"""Utilities related to using casa tasks.
Shamelessly copied from the flint project.
https://github.com/flint-crew/flint/blob/main/flint/casa.py
"""

from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.sclient import singularity_wrapper


def args_to_casa_task_string(task: str, **kwargs) -> str:
    """Given a set of arguments, convert them to a string that can
    be used to run the corresponding CASA task that can be passed
    via ``casa -c`` for execution

    Args:
        task (str): The name of the task that will be executed

    Returns:
        str: The formatted string that will be given to CASA for execution
    """
    command = []
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            v = ",".join(rf"'{_v!s}'" for _v in v)
            arg = rf"{k}=({v})"
        elif isinstance(v, (str, Path)):
            arg = rf"{k}='{v!s}'"
        else:
            arg = rf"{k}={v}"
        command.append(arg)

    task_command = rf"casa -c {task}(" + ",".join(command) + r")"

    return task_command

@singularity_wrapper
def casa_command(task: str, **kwargs) -> str:
    """Construct and run a CASA task.

    Args:
        task (str): The name of the CASA task to run

        kwargs, e.g.
        casa_container (Path): Container with the CASA tooling
        ms (str): Path to the measurement set to transform
        output_ms (str): Path of the output measurement set produced by the transform

    Returns:
        str: The casa task string
    """
    logger = get_run_logger()

    task_str = args_to_casa_task_string(task, **kwargs)
    logger.info(f"CASA {task=} {task_str=}")

    return task_str
