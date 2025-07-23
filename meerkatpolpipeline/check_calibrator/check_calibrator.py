from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions


class CheckCalibratorOptions(BaseOptions):
    """A basic class to handle options for checking checking polcal with the meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Default False"""
    targetfield: str | None = None
    """name of targetfield"""

def split_polcal():
    pass

def go_wsclean_smallcubes():
    pass

def validate_calibrator_field():
    pass

def check_calibrator(check_calibrator_options: CheckCalibratorOptions, working_dir: Path) -> Path:

    split_polcal()

    go_wsclean_smallcubes()

    validate_calibrator_field()

    return
