from __future__ import annotations

from pathlib import Path

import numpy as np
from prefect.logging import get_run_logger

from meerkatpolpipeline.casa import casa_command
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.processfield import (
    determine_calibrator,
    determine_model,
    process_stokesI,
    process_stokesQU,
)
from meerkatpolpipeline.wsclean.wsclean import ImageSet, WSCleanOptions, run_wsclean


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
    no_fit_rm: bool = False
    """ disable the -fit-rm flag in wsclean, since its only available in the newest versions."""
    image_gaincal: bool = False
    """ also make an image of the gain calibrator? Default False"""
    image_primary: bool = False
    """ also make an image of the primary calibrator? Default False"""

    
def split_field(
        ms_path: Path,
        field: str,
        casa_container: Path,
        output_ms: Path | None = None,
        bind_dirs: list[Path] | None = None,
        chanbin: int = 16,
        timebin: str | None = None,
        datacolumn: str = "corrected",
    ) -> Path:
    """
    Split any field from a measurement set using mstransform.
    
    Can use e.g. to split (polarisation/gain/bandpass/flux) calibrator with default 16x channel averaging.
    or to split target field with no channel averaging (chanbin=1).

    Args:
        ms_path (Path): Path to the input measurement set.
        field (str): Field name or ID to split out.
        casa_container (Path): Path to the container with the casa installation.
        output_ms (Path | None): Path to the output measurement set. If None, will be created in the same directory as ms_path with suffix '-{field}.ms'.
        bind_dirs (list[Path] | None): List of directories to bind to the container.
        chanbin (int): Number of channels to average together. Default is 16.
        timebin (str | None): Time interval to average together, e.g. "16s". Default is None (no time averaging).
        datacolumn (str): Data column to use. Default is "corrected".

    Returns:
        output_ms (Path): Path to the output measurement set.
    

    """
    if output_ms is None:
        output_ms = ms_path.with_name(ms_path.stem + f"-{field}.ms")

    logger = get_run_logger()
    logger.info(f"Splitting {field} from {ms_path} to {output_ms}")

    if output_ms.exists():
        logger.info(f"Output MS {output_ms} already exists, skipping split.")
        return output_ms
    
    if chanbin == 1:
        logger.info("No channel binning requested, chanbin=1. Will not average channels.")
        chanaverage = False
    else:
        logger.info(f"Channel binning requested, chanbin={chanbin}. Will average channels.")
        chanaverage = True

    if timebin is not None:
        logger.info(f"Time binning requested, timebin={timebin}. Will average in time.")
        timeaverage = True
    else:
        logger.info("No time binning requested, timebin=None. Will not average in time.")
        timeaverage = False

    casa_command(
        task="mstransform",
        vis=ms_path,
        outputvis=output_ms,
        datacolumn=datacolumn,
        field=field,
        spw="",
        chanaverage=chanaverage,
        chanbin=chanbin, # e.g. 16x averaging
        timeaverage=timeaverage,
        timebin=timebin, # e.g. "16s"
        keepflags=True,
        usewtspectrum=False,
        hanning=False,
        container=casa_container,
        bind_dirs=bind_dirs,
    )

    return output_ms


def build_wsclean_options_for_calibrator(
    stokes: str,
    check_calibrator_options: dict
) -> WSCleanOptions:
    """
    Build default WSClean options for calibrators with your requested settings.
    stokes: 'i' or 'qu'
    """
    if stokes.lower() not in {"i", "qu"}:
        raise ValueError("stokes must be 'i' or 'qu'")

    base = {
        "no_update_model_required": True,
        "minuv_l": 10.0,
        "size": 1000,  # interpreted as -size 1000 1000 by command builder
        # "parallel_deconvolution": 1575,
        "reorder": True,
        "weight": "briggs -0.5",
        "parallel_reordering": 4,
        "data_column": "DATA",
        "join_channels": True,
        "channels_out": 12,
        "no_mf_weighting": True,
        "parallel_gridding": 4,
        "auto_mask": 3.0,
        "auto_threshold": 1.5,
        "gridder": "wgridder",
        "wgridder_accuracy": 0.0001,
        "abs_mem": 100,
        "nmiter": 8,
        "niter": 50000,
        "scale": "1.0arcsec",
    }

    if stokes.lower() == "i":
        base.update({
            "pol": "i",
            "mgain": 0.8,
        })
    else:
        base.update({
            "pol": "qu",
            "mgain": 0.7,
            "join_polarizations": True,
            "squared_channel_joining": True,
            "fit_rm": not bool(check_calibrator_options.get("no_fit_rm", False)),
        })

    return WSCleanOptions(**base)


def go_wsclean_smallcubes(
    ms: Path,
    working_dir: Path,
    check_calibrator_options: dict,
    lofar_container: Path,
    prefix: str
) -> tuple[ImageSet, ImageSet, ImageSet]:
    """
    Quick imaging in I+Q+U of calibrator fields.

    returns an ImageSet for each polarisation (I,Q,U)
    """
    # Stokes I (requested calibrator defaults)
    opts_I = build_wsclean_options_for_calibrator("i", check_calibrator_options)
    [imageset_I] = run_wsclean(
        ms=ms,
        working_dir=working_dir,
        lofar_container=lofar_container,
        prefix=prefix,
        options=opts_I,
        expected_pols=["i"],
    )

    # Stokes QU (requested calibrator defaults)
    opts_QU = build_wsclean_options_for_calibrator("qu", check_calibrator_options)
    imagesets_QU = run_wsclean(
        ms=ms,
        working_dir=working_dir,
        lofar_container=lofar_container,
        prefix=prefix,
        options=opts_QU,
        expected_pols=["q", "u"],
    )
    imageset_Q, imageset_U = imagesets_QU

    return imageset_I, imageset_Q, imageset_U


def regionfile_from_calibrator(src: str) -> Path:
    """
    Determine the region file for the calibrator source.
    """
    import meerkatpolpipeline
    path_to_files = Path(meerkatpolpipeline.__file__).parent / "files" 
    if src == "J1331+3030":
        regionfile = path_to_files / "3c286.reg"
    elif src == "J0521+1638":
        regionfile = path_to_files / "3c138.reg"
    
    assert regionfile.exists(), f"Region file for {src} not found at {regionfile}"
    
    return regionfile


def validate_calibrator_field(
        imageset_stokesI: ImageSet,
        imageset_stokesQ: ImageSet,
        imageset_stokesU: ImageSet,
        polcal_field: str,
        working_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the spectra from the calibrator field images to check how well it matches the expectations.
    """
    logger = get_run_logger()

    # determine regionfile from calibrator JXXXX+XX name
    region_file = regionfile_from_calibrator(polcal_field)

    # determine calibrator 3C name from regionfile name
    src = determine_calibrator(region_file)
    # get the correct models for this calibrator
    I_model, EVPA_model, polfrac_model = determine_model(src)
    models = {'i': I_model, 'evpa': EVPA_model, 'polfrac': polfrac_model, 'src': src}

    logger.info("Processing Stokes I images.")
    stokesI, stokesIpeak, frequencies = process_stokesI(
        imageset_stokesI=imageset_stokesI,
        region_file=region_file,
        models=models,
        logger=logger,
        unc=None,
        integrated=True,
        flagchan=None,
        plotmodel=True,
        plotdir= working_dir / "plots",
    )

    logger.info("Processing Stokes QU images.")

    process_stokesQU(
        imageset_stokesQ=imageset_stokesQ,
        imageset_stokesU=imageset_stokesU,
        stokesI_fluxdens=stokesI,
        region_file=region_file,
        models=models,
        logger=logger,
        unc=None,
        flagchan=None, # TODO
        plotmodel=True,
        plotdir= working_dir / "plots",
    )


    return stokesI, stokesIpeak, frequencies


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

    polcal_ms = split_field(
        ms_path=check_calibrator_options['crosscal_ms'],
        field=check_calibrator_options['polcal_field'],
        output_ms=working_dir / "polcal.ms",
        casa_container=casa_container,
        bind_dirs=bind_dirs,
    )

    imageset_stokesI, imageset_stokesQ, imageset_stokesU = go_wsclean_smallcubes(
        polcal_ms,
        working_dir,
        check_calibrator_options,
        lofar_container= lofar_container,
        prefix='polcal'
    )

    validate_calibrator_field(
        imageset_stokesI=imageset_stokesI,
        imageset_stokesQ=imageset_stokesQ,
        imageset_stokesU=imageset_stokesU,
        polcal_field=check_calibrator_options['polcal_field'],
        working_dir=working_dir,
    )

    logger.info(f"Calibrator checks completed. Please see the plots in {working_dir / 'plots'} for results.")

    return polcal_ms


def image_gaincal(
        check_calibrator_options: dict | CheckCalibratorOptions,
        gaincal_field: str,
        working_dir: Path,
        casa_container: Path,
        lofar_container: Path,
        bind_dirs: list[Path],
    ) -> None:
    """Image the gain calibrator field.
    
    args:
        check_calibrator_options (dict | CheckCalibratorOptions): Dictionary storing CheckCalibratorOptions for the check_calibrator step.
        gaincal_field (str): string denoting gaincal field
        working_dir (Path): The working directory for the check_calibrator step
        casa_container (Path | None): Path to the container with the casa installation.
        lofar_container (Path | None): Path to the container with the wsclean installation.
        bind_dirs (list[Path] | None): List of directories to bind to the container.
    
    Returns:
        None
    """
    logger = get_run_logger()


    working_dir = working_dir / "gaincal"
    working_dir.mkdir(exist_ok=True)

    logger.info(f"Gaincal imaging requested. Will image gain calibrator in {working_dir}.")

    gaincal_ms = split_field(
        ms_path=check_calibrator_options['crosscal_ms'],
        field=gaincal_field,
        output_ms=working_dir / "gaincal.ms",
        casa_container=casa_container,
        bind_dirs=bind_dirs + [working_dir],
    )

    imageset_stokesI, imageset_stokesQ, imageset_stokesU = go_wsclean_smallcubes(
        gaincal_ms,
        working_dir,
        check_calibrator_options,
        lofar_container=lofar_container,
        prefix='gaincal'
    )

    logger.info(f"Gaincal imaging completed. Please see the gaincal images in {working_dir} for results.")

    return

def image_primary(
        check_calibrator_options: dict | CheckCalibratorOptions,
        primary_field: str,
        working_dir: Path,
        casa_container: Path,
        lofar_container: Path,
        bind_dirs: list[Path],
    ) -> None:
    """Image the primary calibrator field.
    
    args:
        check_calibrator_options (dict | CheckCalibratorOptions): Dictionary storing CheckCalibratorOptions for the check_calibrator step.
        primary_field (str): string denoting primary calibrator field
        working_dir (Path): The working directory for the check_calibrator step
        casa_container (Path | None): Path to the container with the casa installation.
        lofar_container (Path | None): Path to the container with the wsclean installation.
        bind_dirs (list[Path] | None): List of directories to bind to the container.
    
    Returns:
        None
    """
    logger = get_run_logger()


    working_dir = working_dir / "primary"
    working_dir.mkdir(exist_ok=True)

    logger.info(f"Primary calibrator imaging requested. Will image primary calibrator in {working_dir}.")

    primary_ms = split_field(
        ms_path=check_calibrator_options['crosscal_ms'],
        field=primary_field,
        output_ms=working_dir / "primary.ms",
        casa_container=casa_container,
        bind_dirs=bind_dirs + [working_dir],
    )

    imageset_stokesI, imageset_stokesQ, imageset_stokesU = go_wsclean_smallcubes(
        primary_ms,
        working_dir,
        check_calibrator_options,
        lofar_container=lofar_container,
        prefix='primary'
    )

    logger.info(f"Primary calibrator imaging completed. Please see the primary calibrator images in {working_dir} for results.")

    return
