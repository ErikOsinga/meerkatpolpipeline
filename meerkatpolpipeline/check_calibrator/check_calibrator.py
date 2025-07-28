from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from prefect.logging import get_run_logger

from meerkatpolpipeline.casa import casa_command
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.wsclean.wsclean import (
    ImageSet,
    WSCleanOptions,
    create_wsclean_command,
    run_wsclean_command,
)


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
    logger.info(f"Splitting polarisation calibrator {polcal_field} from {cal_ms_path} to {output_ms}")

    if output_ms.exists():
        logger.info(f"Output MS {output_ms} already exists, skipping split.")
        return output_ms

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


def go_wsclean_smallcubes(ms: Path, working_dir: Path, lofar_container: Path) -> ImageSet:
    """
    Quick round of imaging the calibrator in IQU
    """

    logger = get_run_logger()

    # mkdir for wsclean output
    wsclean_output_dir = working_dir / "IQUimages"
    logger.info(f"Creating wsclean output directory at {wsclean_output_dir}")
    wsclean_output_dir.mkdir(exist_ok=True)

    hardcoded_options = {
        'data_column': 'DATA', # since we split the MS
        'no_update_model_required': True,
        'minuv_l': 10.0,
        'size': 1000,
        'weight': "briggs -0.5",
        'mgain': 0.9,
        'join_channels': True,
        'channels_out': 12,
        'no_mf_weighting': True,
        'parallel_gridding': 4,
        'auto_mask': 3.0,
        'auto_threshold': 1.5,
        'pol': 'i',
        'gridder': 'wgridder',
        'wgridder_accuracy': 0.0001,
        'abs_mem': 100,
        'nmiter': 8,
        'niter': 10000,
        'scale': '1.0arcsec',
    }

    # /check_calibrator/IQUimages/polcal-image-00*.fits"
    prefix = wsclean_output_dir / "polcal" 

    options = WSCleanOptions(**hardcoded_options)

    wsclean_command = create_wsclean_command(options, ms, prefix)

    run_wsclean_command(wsclean_command=wsclean_command,
                        container=lofar_container,
                        bind_dirs=[ms.parent,wsclean_output_dir]
    )


def validate_calibrator_field():
    pass

def check_calibrator(
        check_calibrator_options: dict | CheckCalibratorOptions,
        working_dir: Path,
        casa_container: Path,
        lofar_container: Path,
        bind_dirs: list[Path],
    ) -> Path:
    """Check the polcal calibrator field.
    
    args:
        check_calibrator_options (dict | CheckCalibratorOptions): Dictionary storing CheckCalibratorOptions for the check_calibrator step.
        working_dir (Path): The working directory for the check_calibrator step
        casa_container (Path | None): Path to the container with the casa installation.
        lofar_container (Path | None): Path to the container with the wsclean installation.
        bind_dirs (list[Path] | None): List of directories to bind to the container.
    
    Returns:
        Path: The path to the polcal measurement set after splitting.
    """
    logger = get_run_logger()

    polcal_ms = split_polcal(
        cal_ms_path=check_calibrator_options['crosscal_ms'],
        polcal_field=check_calibrator_options['polcal_field'],
        output_ms=working_dir / "polcal.ms",
        casa_container=casa_container,
        bind_dirs=bind_dirs,
    )

    go_wsclean_smallcubes(
        polcal_ms,
        working_dir,
        lofar_container= lofar_container,
    )

    # validate_calibrator_field()

    return
