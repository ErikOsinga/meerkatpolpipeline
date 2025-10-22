from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from casacore.tables import table
from prefect.logging import get_run_logger

from meerkatpolpipeline.cube_imaging.pbcor import calculate_pb, pbcor_allchan
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.wsclean.wsclean import ImageSet, WSCleanOptions, run_wsclean


class CoarseCubeImagingOptions(BaseOptions):
    """A basic class to handle options for coarse cube imaging. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""    
    corrected_extracted_mses: list[Path] | None = None
    """list of Paths to extracted MSes that contains the corrected data. If None, will be determined automatically"""
    no_fit_rm: bool = False
    """ disable the -fit-rm flag in wsclean, since its only available in the newest versions."""
    also_image_for_mfs: bool = False
    """ also make MFS images in addition to the coarse cubes. This is a separate imaging step with no_mf_weighting"""
    run_pybdsf: bool = False
    """ also run pybdsf on the Stokes I MFS image to create a source catalogue. Requires also_image_for_mfs=True"""
    filter_pybdsf_cat_radius_deg: float | None = None
    """ filter the pybdsf source catalogue to only include sources within this radius (degrees) from the field centre. If None, no filtering is done."""

    # TODO: add size, scale, channels_out etc parameters for wsclean

class FineCubeImagingOptions(BaseOptions):
    """A basic class to handle options for fine cube imaging. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""    
    corrected_extracted_mses: list[Path] | None = None
    """list of Paths to extracted MSes that contains the corrected data. If None, will be determined automatically"""
    no_fit_rm: bool = False
    """ disable the -fit-rm flag in wsclean, since its only available in the newest versions."""
    chanwidth_MHz: float | None = 5
    """Channel width in MHz to aim for. Default 5 MHz"""
    startfreq_MHz: float | None = None
    """Start frequency in MHz for the fine channel cubes. If None, will use the lowest frequency in the MS."""
    endfreq_MHz: float | None = None
    """End frequency in MHz for the fine channel cubes. If None, will use the highest frequency in the MS."""
    # TODO: add size, scale, channels_out etc parameters for wsclean



def go_wsclean_cube_imaging_target(
    ms: Path | list[Path],
    working_dir: Path,
    lofar_container: Path,
    cube_imaging_options: dict | CoarseCubeImagingOptions | FineCubeImagingOptions,
    finecube: bool = False
) -> tuple[ImageSet, ImageSet, ImageSet]:
    """
    Image the TARGET field in I + Q + U using some default settings.
        
        stokes I uses multiscale; QU has multiscale disabled.
    
    Returns: (imageset_I, imageset_Q, imageset_U)
    """

    logger = get_run_logger()
    logger.info(f"Starting WSClean imaging for target field {cube_imaging_options['targetfield']} in {working_dir}")

    if not finecube:
        channels_out = 12 # sensible quick default
        channel_range = None # image whole band
        logger.info(f"Using default channels_out={channels_out} for coarse cubes.")

    else:
        channels_out, channel_range_start, channel_range_end = compute_chanout_from_chanwidth(ms, cube_imaging_options)
        
        logger.info(f"Computed channels_out={channels_out} for fine cube imaging, assuming a channel width of {cube_imaging_options['chanwidth_MHz']} MHz.")
        
        if channel_range_start is not None and channel_range_end is not None:
            # image only part of band
            logger.info(f"Using channel range {channel_range_start} to {channel_range_end} for fine cube imaging corresponding to frequencies {cube_imaging_options['startfreq_MHz']} MHz to {cube_imaging_options['endfreq_MHz']} MHz.")
            channel_range = [channel_range_start, channel_range_end]
        else:
            # image whole band
            channel_range = None

    # Common parameters between I and QU
    common = dict(
        no_update_model_required=True,
        minuv_l=10.0,
        size=3150,
        parallel_deconvolution=1575,
        reorder=True,
        weight="briggs 0",
        parallel_reordering=4,
        data_column="DATA", # _subtracted_ddcor should only contain DATA column
        join_channels=True,
        channels_out=channels_out, # channels out computed above
        channel_range=channel_range,
        no_mf_weighting=True,
        parallel_gridding=4,
        auto_mask=3.0,
        auto_threshold=1.5,
        gridder="wgridder",
        wgridder_accuracy=0.0001,
        mem=80,
        nmiter=6,
        niter=75000,
        scale="2.5arcsec",
        taper_gaussian="10.0arcsec",
        apply_primary_beam=False,
    )

    # >>> Primary beam correction can only be performed on Stokes I, polarizations (XX,YY) or when imaging all four polarizations.

    # ----- Stokes I (multiscale ON) -----
    opts_I = WSCleanOptions(
        **common,
        pol="i",
        mgain=0.8,
        multiscale=True,
        multiscale_scale_bias=0.75,
        multiscale_max_scales=9,
    )

    [imageset_I] = run_wsclean(
        ms=ms,
        working_dir=working_dir,
        lofar_container=lofar_container,
        prefix=cube_imaging_options['targetfield']+'_stokesI',
        options=opts_I,
        expected_pols=["i"],
        open_files_limit=None, # can change this if 'error writing to temporary file' appears in WSClean during reorder step
    )

    imageset_I = pbcor_cubes_target(imageset_I, working_dir / "pbcor_images", pol='i')

    # ----- Stokes QU (multiscale OFF) -----
    # Build fresh options to avoid inheriting multiscale from I.
    opts_QU = WSCleanOptions(
        **common,
        pol="qu",
        mgain=0.7,
        join_polarizations=True,
        squared_channel_joining=True,
        fit_rm=not cube_imaging_options['no_fit_rm'],
        # Explicitly disable/remove multiscale-related flags
        multiscale=False,
        multiscale_scale_bias=None,
        multiscale_max_scales=None,
    )

    imagesets_QU = run_wsclean(
        ms=ms,
        working_dir=working_dir,
        lofar_container=lofar_container,
        prefix=cube_imaging_options['targetfield']+'_stokesQU',
        options=opts_QU,
        expected_pols=["q", "u"],
        open_files_limit=9000, # require many open files for fine cube imaging with 2 stokes
    )
    imageset_Q, imageset_U = imagesets_QU

    imageset_Q = pbcor_cubes_target(imageset_Q, working_dir / "pbcor_images", pol='q')
    imageset_U = pbcor_cubes_target(imageset_U, working_dir / "pbcor_images", pol='u')


    if cube_imaging_options['also_image_for_mfs']:
        working_dir_mfs = working_dir / "MFSimaging"
        working_dir_mfs.mkdir(parents=True, exist_ok=True)

        logger.info(f"Also making MFS images for target field {cube_imaging_options['targetfield']} in {working_dir_mfs}")

        # MFS imaging run is exactly the same but with optimal weighting for MFS
        opts_I_mfs = opts_I.with_options(no_mf_weighting=False)

        [imageset_I_mfs] = run_wsclean(
            ms=ms,
            working_dir=working_dir_mfs,
            lofar_container=lofar_container,
            prefix=cube_imaging_options['targetfield']+'_stokesI',
            options=opts_I_mfs,
            expected_pols=["i"],
        )

        # can put pbcor images in same directory as coarse cubes
        imageset_I_mfs = pbcor_cubes_target(imageset_I_mfs, working_dir / "pbcor_images", pol='i')

    else:
        imageset_I_mfs = None

    return imageset_I, imageset_Q, imageset_U, imageset_I_mfs


def pbcor_cubes_target(imset: ImageSet, outdir_pbcor_images: Path, pol: str) -> ImageSet:
    """
    Do PB correction for cubes made in go_wsclean_cube_imaging_target.
    """
    outdir_pbcor_images.mkdir(parents=True, exist_ok=True)

    path_split = str(imset.image[0]).split("-0000-")
    assert len(path_split) == 2, f"Cannot parse image name {imset.image[0]}. Expected '-0000-' in the name."

    if pol.lower() == 'i':
        globstr = path_split[0] + "-*image.fits"
    elif pol.lower() == 'q':
        globstr = path_split[0] + "-*Q-image.fits"
    elif pol.lower() == 'u':
        globstr = path_split[0] + "-*U-image.fits"

    # TODO: make script aware of Meerkat Band (eg. L or UHF)
    print("TODO: make script aware of Meerkat Band (eg. L or UHF)")
    calculate_pb(globstr, band='L', outdir=outdir_pbcor_images)

    globstr_pbcor = f"{outdir_pbcor_images}/*-I-pb_model.fits"

    all_corrected, all_pbcor = pbcor_allchan(globstr, globstr_pbcor, verbose=False)

    imset = imset.with_options(image_pbcor=all_corrected, pbcor_model_images=all_pbcor)

    return imset


def _normalize_ms_list(ms: Path | list[Path]) -> list[Path]:
    if isinstance(ms, (str, Path)):
        return [Path(ms)]
    return [Path(p) for p in ms]


def _freq_bounds_from_ms(ms_path: Path) -> tuple[float, float]:
    """
    Return (fmin_MHz, fmax_MHz) from SPECTRAL_WINDOW::CHAN_FREQ (Hz) for a single MS.
    """
    spw_path = str(ms_path / "SPECTRAL_WINDOW")
    with table(spw_path) as t:
        freqs_Hz = t.getcol("CHAN_FREQ")  # shape: (n_spw, n_chan) or (n_row, n_chan)
    freqs_Hz = np.asarray(freqs_Hz, dtype=np.float64).ravel()
    fmin_MHz = float(np.min(freqs_Hz) * u.Hz.to(u.MHz))
    fmax_MHz = float(np.max(freqs_Hz) * u.Hz.to(u.MHz))
    if fmin_MHz > fmax_MHz:
        fmin_MHz, fmax_MHz = fmax_MHz, fmin_MHz
    return fmin_MHz, fmax_MHz


def _load_ms_chanwidth_MHz(ms_path: Path) -> float:
    """
    Load channel width in measurement set

    Returns:
        float: channel width in MHz
    """

    spw_path = str(ms_path / "SPECTRAL_WINDOW")
    with table(spw_path) as t:
        if "CHAN_WIDTH" in t.colnames():
            widths_Hz = np.asarray(t.getcol("CHAN_WIDTH"), dtype=np.float64).ravel()
            assert (widths_Hz == widths_Hz[0]).all(), "Found non-equal channel width in MS. Should not happen after facetselfcal."
        else:
            raise ValueError(f"Could not obtain channel width from {ms_path}")

    return widths_Hz[0]/1e6


def _global_bounds_MHz(ms_list: list[Path]) -> tuple[float, float]:
    fmins, fmaxs = zip(*[_freq_bounds_from_ms(p) for p in ms_list])
    return float(min(fmins)), float(max(fmaxs))


def _resolve_bounds_MHz(ms: Path | list[Path], opts: dict) -> tuple[float, float]:
    """
    Obtain start and end frequency from MS or options, clamped to MS extent.

    Args:
        ms: Path or list of Paths to Measurement Sets.
        opts: Cube imaging options with optional startfreq_MHz and endfreq_MHz.
    
    Returns:
        (start_MHz, end_MHz, fmin_ms, fmax_ms)
    """
    ms_list = _normalize_ms_list(ms)

    # Start/end from options or MS
    if opts['startfreq_MHz'] is not None and opts['endfreq_MHz'] is not None:
        start_MHz = float(opts['startfreq_MHz'])
        end_MHz = float(opts['endfreq_MHz'])
    else:
        fmin_ms, fmax_ms = _global_bounds_MHz(ms_list)
        start_MHz = fmin_ms
        end_MHz = fmax_ms

    if start_MHz == end_MHz:
        raise ValueError("startfreq_MHz equals endfreq_MHz; bandwidth is zero.")
    if start_MHz > end_MHz:
        raise ValueError(f"{start_MHz=} is bigger than {end_MHz=}; invalid frequency range.")

    # Clamp to MS extent
    fmin_ms, fmax_ms = _global_bounds_MHz(ms_list)
    start_MHz = max(start_MHz, fmin_ms)
    end_MHz = min(end_MHz, fmax_ms)

    if end_MHz <= start_MHz:
        raise ValueError("After clamping, endfreq_MHz <= startfreq_MHz; no usable bandwidth.")

    return start_MHz, end_MHz, fmin_ms, fmax_ms


def compute_chanout_from_chanwidth(
    ms: Path | list[Path],
    cube_imaging_options: dict
) -> tuple[int, int | None, int | None]:
    """
    Compute integer channels_out for WSClean so the per-channel width is
    as close as possible to, but not exceeding, chanwidth_MHz.
    If exact division is impossible, we err smaller (more channels).

    Example behaviour:

        - any band, bandwidth = 31 MHz, any channel width in MS, target width = 5 MHz 
            -> channels_out = 7 (width=4.42857 MHz), channel_range_start = None, channel_range_end = None
        
        - Lband = 907-1670 MHz, 0.84 MHz channel width in MS, 'startfreq_MHz' = 1400, 'endfreq_MHz' = 1600 
            -> channels_out = 40 (width=5 MHz), channel_range_start = 589 (1399 MHz), channel_range_end = 829 (1600.1 MHz)

    args:
        ms: Path or list of Paths to Measurement Sets.
        cube_imaging_options: dict with at least 'chanwidth_MHz', optional
            'startfreq_MHz' and 'endfreq_MHz'.

    Returns:
        int: number of channels between 'startfreq_MHz' and 'endfreq_MHz' with a channel width of chanwidth_MHz
        int: channel_range_start for WSclean (or None)
        int: channel_range_end for WSclean (or None)
    """

    target_width_MHz = cube_imaging_options['chanwidth_MHz']
    if target_width_MHz <= 0:
        raise ValueError("chanwidth_MHz must be positive.")

    start_MHz, end_MHz, freqmin_ms, freqmax_ms = _resolve_bounds_MHz(ms, cube_imaging_options)
    bandwidth_MHz = (end_MHz - start_MHz)

    # Floor ensures resulting width <= target (unless bandwidth < target).
    channels_out = int(np.floor(bandwidth_MHz / target_width_MHz))
    if channels_out < 1:
        raise ValueError(f"Computed channels_out < 1 for {start_MHz=}, {end_MHz=} and {target_width_MHz=}.")

    # Numerical safety: enforce width <= target if channels_out > 1
    resulting_width = bandwidth_MHz / channels_out
    if resulting_width > target_width_MHz * (1.0 + 1e-12):
        channels_out += 1
    resulting_width = bandwidth_MHz / channels_out

    # Now also compute "channel-range", i.e. the start and end channel in WSclean
    # this depends on the channel width in the measurement set
    ms_chanwidth_MHz = _load_ms_chanwidth_MHz(_normalize_ms_list(ms)[0])
    # since WSclean first computes which channel range to cover, and then splits that range into "channels_out" channels for imaging.

    if start_MHz == freqmin_ms and end_MHz == freqmax_ms:
        channel_range_start = None
        channel_range_end = None
    else:
        channel_range_start = int((start_MHz - freqmin_ms) / ms_chanwidth_MHz)
        channel_range_end = int((end_MHz - freqmin_ms) / ms_chanwidth_MHz)+1

    return int(channels_out), channel_range_start, channel_range_end