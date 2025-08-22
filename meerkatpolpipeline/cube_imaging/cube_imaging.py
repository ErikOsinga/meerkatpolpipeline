from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.processfield import (
    determine_calibrator,
    determine_model,
    process_stokesI,
    process_stokesQU,
)
from meerkatpolpipeline.wsclean.wsclean import ImageSet, WSCleanOptions, run_wsclean


class SmallCubeImagingOptions(BaseOptions):
    """A basic class to handle options for cube imaging. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""    
    corrected_extracted_mses: list[Path] | None = None
    """list of Paths to extracted MSes that contains the corrected data. If None, will be determined automatically"""
    no_fit_rm: bool = False
    """ disable the -fit-rm flag in wsclean, since its only available in the newest versions."""


def go_wsclean_smallcubes_target(
    ms: Path,
    working_dir: Path,
    lofar_container: Path,
    cube_imaging_options: SmallCubeImagingOptions
) -> tuple[ImageSet, ImageSet, ImageSet]:
    """
    Image the TARGET field in I + Q + U using some default settings.
        
        stokes I uses multiscale; QU has multiscale disabled.
    
    Returns: (imageset_I, imageset_Q, imageset_U)
    """

    # Common parameters between I and QU
    common = dict(
        no_update_model_required=True,
        minuv_l=10.0,
        size=3150,
        parallel_deconvolution=1575,
        reorder=True,
        weight="briggs 0",
        parallel_reordering=4,
        data_column="CORRECTED_DATA",
        join_channels=True,
        channels_out=12,
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
        apply_primary_beam=True,
    )

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
    )

    # ----- Stokes QU (multiscale OFF) -----
    # Build fresh options to avoid inheriting multiscale from I.
    opts_QU = WSCleanOptions(
        **common,
        pol="qu",
        mgain=0.7,
        join_polarizations=True,
        squared_channel_joining=True,
        fit_rm=True,
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
    )
    imageset_Q, imageset_U = imagesets_QU

    return imageset_I, imageset_Q, imageset_U