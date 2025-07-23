from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.append('/home/osingae/OneDrive/postdoc/projects/MEERKAT_similarity_Bfields/meerkatpolpipeline')

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
