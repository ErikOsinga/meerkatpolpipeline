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
    targetfield: str | None = None
    """name of targetfield. IGNORED. This option is propagated to every step even though its not useful in this step."""    
    crosscal_ms: Path | None = None
    """Path to cross-calibrated MS that contains the calibrators. If None, will be determined automatically"""
    polcal_field: str | None = None
    """String containing the name of the polarisation calibrator field. If None, will be determined automatically"""

    
def split_polcal(
        cal_ms_path: Path,
        polcal_field: str,
        casa_container: Path,
        output_ms: Path | None = None,
        bind_dirs: list[Path] | None = None,
        chanbin: int = 16,
    ) -> Path:
    """
    Split the polarisation calibrator with default 16x channel averaging.
    """
    if output_ms is None:
        output_ms = cal_ms_path.with_name(cal_ms_path.stem + "-polcal.ms")

    logger = get_run_logger()

    if output_ms.exists():
        logger.info(f"Output MS {output_ms} already exists, skipping split.")
        return output_ms

    logger.info(f"Splitting polarisation calibrator {polcal_field} from {cal_ms_path} to {output_ms}")

    casa_command(
        task="mstransform",
        vis=cal_ms_path,
        outputvis=output_ms,
        datacolumn="corrected",
        field=polcal_field,
        spw="",
        chanaverage=True,
        chanbin=chanbin, # e.g. 16x averaging
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

def check_calibrator(
        check_calibrator_options: dict | CheckCalibratorOptions,
        working_dir: Path,
        casa_container: Path ,
        bind_dirs: list[Path],
    ) -> Path:
    """Check the polcal calibrator field.
    
    args:
        check_calibrator_options (dict | CheckCalibratorOptions): Dictionary storing CheckCalibratorOptions for the check_calibrator step.
        working_dir (Path): The working directory for the check_calibrator step
        casa_container (Path | None): Path to the container with the casa installation.
        bind_dirs (list[Path] | None): List of directories to bind to the container.
    
    Returns:
        Path: The path to the polcal measurement set after splitting.
    """
    logger = get_run_logger()

    split_polcal(
        cal_ms_path=check_calibrator_options['crosscal_ms'],
        polcal_field=check_calibrator_options['polcal_field'],
        output_ms=working_dir / "polcal.ms",
        casa_container=casa_container,
        bind_dirs=bind_dirs,
    )
    

    # go_wsclean_smallcubes()

    # validate_calibrator_field()

    return
