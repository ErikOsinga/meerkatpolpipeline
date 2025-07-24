from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from prefect.logging import get_run_logger

from meerkatpolpipeline.casa import casa_command
from meerkatpolpipeline.options import BaseOptions


class CheckCalibratorOptions(BaseOptions):
    """A basic class to handle options for checking checking polcal with the meerkatpolpipeline. """
    
    enable: bool
    """enable this step?"""
    crosscal_ms: Path | None = None
    """Path to cross-calibrated MS that contains the calibrators. If None, will be determined automatically"""

def split_polcal(
        cal_ms_path: Path,
        polcal_field: str,
        output_ms: Path | None = None,
        casa_container: Path | None = None,
        bind_dirs: list[Path] | None = None,
    ) -> Path:
    """
    Split the polarisation calibrator with default 16x channel averaging.
    """
    if output_ms is None:
        output_ms = cal_ms_path.with_name(cal_ms_path.stem + "-polcal.ms")

    logger = get_run_logger()

    logger.info(f"Splitting polarisation calibrator {polcal_field} from {cal_ms_path} to {output_ms}")

    casa_command(
        task="mstransform",
        vis=cal_ms_path,
        outputvis=output_ms,
        datacolumn="corrected",
        field=polcal_field,
        spw="",
        keepflags=True,
        usewtspectrum=False,
        hanning=False,
        container=casa_container,
        bind_dirs=bind_dirs,
    )

    return output_ms


def go_wsclean_smallcubes():
    pass

def validate_calibrator_field():
    pass

def check_calibrator(check_calibrator_options: CheckCalibratorOptions, working_dir: Path) -> Path:
    """Check the polcal calibrator field."""
    logger = get_run_logger()


    split_polcal(
        cal_ms_path=check_calibrator_options['crosscal_ms'],
        polcal_field=check_calibrator_options.targetfield or "polcal",
        output_ms=working_dir / "polcal.ms",
    )
    

    go_wsclean_smallcubes()

    validate_calibrator_field()

    return
