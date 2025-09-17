from __future__ import annotations

from pathlib import Path

from meerkatpolpipeline.check_nvss.target_vs_nvss import _compare_to_nvss
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.wsclean.wsclean import ImageSet


class CompareNVSSOptions(BaseOptions):
    """A basic class to handle NVSS comparison options for meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Required parameter"""
    targetfield: str | None = None
    """name of targetfield. Propagated to all steps."""


class PlaceholderArgs:
    pass

def create_args_compare_nvss(
        compare_nvss_options: dict | CompareNVSSOptions,
        working_dir: Path,
        imageset_I: ImageSet,
        imageset_Q: ImageSet,
        imageset_U: ImageSet,
    ) -> PlaceholderArgs:
    """
    Create args for the _compare_to_nvss function
    """

    args = PlaceholderArgs()
    
    print("TODO: probably want to change target_vs_nvss.py to take an ImageSet instead of globs / optionally instead of globs")

    args.i_glob = "./small_cube_imaging/IQUimages/*stokesI-00*-image.pbcor.fits"
    args.q_glob = "./small_cube_imaging/IQUimages/*stokesQU-00*-Q-image.pbcor.fits"
    args.pbcor_glob  = "./small_cube_imaging/pbcor_images/*fits"
    # args.ds9reg = Path('/net/rijn9/data2/osinga/meerkatBfields/Abell754/Lband_combined/source11.reg')
    args.ds9reg = Path('/net/rijn9/data2/osinga/meerkatBfields/Abell754/Lband_combined/veck_sources_in_field.reg')
    args.output_dir = Path("./small_cube_imaging/nvss_comparison/")

    args.chan_unc_center = 1.25e-5
    args.nvss_size = 500.0

    args.flag_chans = "[]"
    args.flag_by_noise = None
    args.flag_by_noise_factor = 2

    args.comparetable = None
    args.comparetable_idx = None
    args.comparenvssdirect = True
    # only possible if users have access to this data.. 
    args.nvss_dir = "/net/rijn9/data2/osinga/meerkatBfields/NVSS/NVSS_IQUp_catalog/"

    args.output_dir_data = Path("./small_cube_imaging/nvss_comparison/data")

    return args

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

    args = create_args_compare_nvss(
        compare_nvss_options=compare_nvss_options,
        working_dir=working_dir,
        imageset_I=imageset_I,
        imageset_Q=imageset_Q,
        imageset_U=imageset_U,
    )

    _compare_to_nvss(args)