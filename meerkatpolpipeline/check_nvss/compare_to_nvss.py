from __future__ import annotations

from pathlib import Path

from meerkatpolpipeline.check_nvss.target_vs_nvss import _compare_to_nvss
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.wsclean.wsclean import ImageSet


class CompareNVSSOptions(BaseOptions):
    """A basic class to handle NVSS comparison options for meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Required parameter"""
    nvss_dir: Path
    """Path to directory containing NVSS catalog and FITS files. Required."""
    targetfield: str | None = None
    """name of targetfield. Propagated to all steps."""
    flux_extraction_regions: Path | None = None
    """Path to a ds9 region file containing flux extraction regions. If not given, will attempt to find regions from bright NVSS sources."""
    nvss_max_regions: None | int = 10
    """Maximum number of regions to extract fluxes from. Used when flux_extraction_regions is None"""
    nvss_cutout_size: float = 500.0
    """Size of NVSS cutout in arcseconds. Default 500.0"""


def compare_to_nvss(
        compare_nvss_options: dict | CompareNVSSOptions,
        working_dir: Path,
        imageset_I: ImageSet,
        imageset_Q: ImageSet,
        imageset_U: ImageSet,
    ) -> None:
    """Compare sources in the field to NVSS catalogue
    
    args:
        compare_nvss_options (dict | CompareNVSSOptions): Dictionary storing CompareNVSSOptions for this step
        working_dir (Path): The working directory for the compare_to_nvss step
        imageset_I/Q/U (ImageSet): The ImageSet containing the target field coarse cubes for Stokes I, Q and U respectively.
    
    Returns:
        None: plots are made in the working_dir
    """

    _compare_to_nvss(
        ds9reg = compare_nvss_options['flux_extraction_regions'],
        flag_chans = compare_nvss_options['flag_chans'],
        ifiles = imageset_I.image_pbcor,
        qfiles = imageset_Q.image_pbcor,
        ufiles = imageset_U.image_pbcor,
        pb_files = imageset_I.pbcor_model_images,
        output_dir = working_dir,
        comparenvssdirect = True,
        nvss_size = compare_nvss_options['nvss_cutout_size'],
        nvss_dir = compare_nvss_options['nvss_dir'],
        comparetable = None, # TODO: implement comparison to a table instead of NVSS fits files?
        comparetable_idx = None,
        chan_unc_center = None,
        output_dir_data = working_dir / "data",
        flag_by_noise=None,
    )

    return None