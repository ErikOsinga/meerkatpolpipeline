"""Simple interface into wsclean

Shamelessly stolen from the Flint project and adapted to my needs

"""
from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.sclient import singularity_wrapper


class ImageSet(BaseOptions):
    """A structure to represent the images and auxiliary products produced by
    wsclean"""

    prefix: str
    """Prefix of the images and other output products. This should correspond to the -name argument from wsclean"""
    image: list[Path]
    """Images produced. """
    psf: list[Path] | None = None
    """References to the PSFs produced by wsclean. """
    dirty: list[Path] | None = None
    """Dirty images. """
    model: list[Path] | None = None
    """Model images.  """
    residual: list[Path] | None = None
    """Residual images."""
    source_list: Path | None = None
    """Path to a source list that accompanies the image data"""
    image_pbcor: list[Path] | None = None
    """Primary beam corrected images, if primary beam correction was applied."""


class WSCleanOptions(BaseOptions):
    """A basic container to handle WSClean options. These attributes should
    conform to the same option name in the calling signature of wsclean

    Basic support for environment variables is available. Should a value start
    with `$` it is assumed to be a environment variable, it is will be looked up.
    Some basic attempts to determine if it is a path is made.

    Should the `temp_dir` options be specified then all images will be
    created in this location, and then moved over to the same parent directory
    as the imaged MS. This is done by setting the wsclean `-name` argument.
    """

    abs_mem: int = 400
    """Memory (GB) wsclean should try to limit itself to"""
    local_rms_window: int | None = None
    """Size of the window used to estimate rms noise"""
    size: int = 3000
    """Image size, only a single dimension is required. Note that this means images will be squares. """
    local_rms: bool = False
    """Whether a local rms map is computed"""
    force_mask_rounds: int | None = None
    """Round of force masked derivation"""
    auto_mask: float | None = 3.5
    """How deep the construct clean mask is during each cycle"""
    auto_threshold: float | None = 0.5
    """How deep to clean once initial clean threshold reached"""
    threshold: float | None = None
    """Threshold in Jy to stop cleaning"""
    channels_out: int = 12
    """Number of output channels"""
    mgain: float = 0.7
    """Major cycle gain"""
    nmiter: int = 15
    """Maximum number of major cycles to perform"""
    niter: int = 750000
    """Maximum number of minor cycles"""
    multiscale: bool = False
    """Enable multiscale deconvolution"""
    multiscale_scale_bias: float | None = None
    """Multiscale bias term"""
    multiscale_gain: float | None = None
    """Size of step made in the subminor loop of multi-scale. Default currently 0.2, but shows sign of instability. A value of 0.1 might be more stable."""
    multiscale_scales: tuple[int, ...] | None = None
    """Scales used for multi-scale deconvolution"""
    multiscale_max_scales: int | None = None
    """Maximum number of scales to use in the multiscale deconvolution. If None, then WSclean default is 10."""    
    fit_spectral_pol: int | None = None
    """Number of spectral terms to include during sub-band subtraction"""
    weight: str = "briggs -0.5"
    """Robustness of the weighting used"""
    data_column: str = "CORRECTED_DATA"
    """Which column in the MS to image"""
    scale: str = "2.5asec"
    """Pixel scale size"""
    gridder: str | None = "wgridder"
    """Use the wgridder kernel in wsclean (instead of the default w-stacking method)"""
    nwlayers: int | None = None
    """Number of w-layers to use if the gridder mode is w-stacking"""
    wgridder_accuracy: float | None = None
    """The accuracy requested of the wgridder (should it be used), compared as the RMS error when compred to a DFT"""
    join_channels: bool = True
    """Collapse the sub-band images down to an MFS image when peak-finding"""
    squared_channel_joining: bool = False
    """Use with -join-channels to perform peak finding in the sum of squared values over
    channels, instead of the normal sum. This is useful for imaging QU polarizations
    with non-zero rotation measures, for which the normal sum is insensitive.
    """
    join_polarizations: bool = False
    """Perform deconvolution by searching for peaks in the sum of squares of the polarizations,
    but subtract components from the individual images. Only possible when imaging two or four Stokes
    or linear parameters. Default: off.
    """
    minuv_l: float | None = None
    """The minimum lambda length that the visibility data needs to meet for it to be selected for imaging"""
    minuvw_m: float | None = None
    """A (u,v) selection command, where any baselines shorter than this will be ignored during imaging"""
    maxw: float | None = None
    """A percentage specifying the maximum w-term to be gridded, relative to the max w-term being considered"""
    no_update_model_required: bool = False
    """Will instruct wsclean not to create the MODEL_DATA column"""
    no_small_inversion: bool = False
    """Disables an optimisation of wsclean's w-gridder mode. This might improve accuracy of the w-gridder. """
    beam_fitting_size: float | None = 1.25
    """Use a fitting box the size of <factor> times the theoretical beam size for fitting a Gaussian to the PSF."""
    fits_mask: Path | None = None
    """Path to a FITS file that encodes a cleaning mask"""
    deconvolution_channels: int | None = None
    """The channels out will be averaged down to this many sub-band images during deconvolution"""
    parallel_deconvolution: int | None = None
    """If not none, then this is the number of sub-regions wsclean will attempt to divide and clean"""
    parallel_gridding: int | None = None
    """If not none, then this is the number of channel images that will be gridded in parallel"""
    temp_dir: str | Path | None = None
    """The path to a temporary directory where files will be written. """
    pol: str = "i"
    """The polarisation to be imaged"""
    save_source_list: bool = False
    """Saves the found clean components as a BBS/DP3 text sky model"""
    channel_range: tuple[int, int] | None = None
    """Image a channel range between a lower (inclusive) and upper (exclusive) bound"""
    no_reorder: bool = False
    """If True turn off the reordering of the MS at the beginning of wsclean"""
    no_mf_weighting: bool = False
    """Opposite of -mf-weighting; can be used to turn off MF weighting in -join-channels mode. Suggested for channel science"""
    fit_rm: bool = False
    """Fit a rotation measure to the data during deconvlution. Available since WSClean 3.7"""
    reorder: bool = True
    """If True, reorder the MS at the beginning of wsclean. This is the default behaviour"""
    parallel_reordering: int | None = 4
    """Default 4 in WSClean as well"""
    mem: int | None = None
    """Memory percentage to use for wsclean. If None, then the value of abs_mem will be used or all memory."""
    taper_gaussian: str | None = None
    """A Gaussian taper to apply to the MS before imaging. This is a string that will be passed to wsclean as the -taper-gaussian argument."""
    apply_primary_beam: bool = False
    """Apply the primary beam correction to the images."""



class WSCleanCommand(BaseOptions):
    """Simple container for a wsclean command."""

    cmd: str
    """The constructed wsclean command that would be executed."""
    options: WSCleanOptions
    """The set of wslean options used for imaging"""
    ms: Path | list[Path]
    """The measurement sets that have been included in the wsclean command. """
    image_prefix_str: str | None = None
    """The prefix of the images that will be created"""
    cleanup: bool = True
    """Will clean up the dirty images/psfs/residuals/models when the imaging has completed"""
    image_set: ImageSet | None = None
    """The set of images produced by wsclean"""


def create_wsclean_command(
        options: WSCleanOptions,
        ms: list[Path] | Path,
        prefix: str | None = None
    ) -> WSCleanCommand:
    """
    Construct a wsclean command from the given options and measurement set.
    """
    logger = get_run_logger()
    opt_dict = vars(options)
    cmd_parts: list[str] = ["wsclean"]

    # Build flags from options
    for name, val in opt_dict.items():
        if val is None:
            continue
        flag = f"-{name.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cmd_parts.append(flag)
        elif name == "size":
            # always output two dimensions
            dims = (val, val) if isinstance(val, int) else val
            cmd_parts.extend([flag] + [str(d) for d in dims])
        elif isinstance(val, (list, tuple)):
            cmd_parts.extend([flag, ",".join(str(v) for v in val)])
        else:
            cmd_parts.extend([flag, str(val)])

    # Handle output prefix
    if prefix:
        cmd_parts.extend(["-name", prefix])

    # Append the measurement set(s)
    if isinstance(ms, Path):
        cmd_parts.append(str(ms))
    elif isinstance(ms, (list, np.ndarray)):
        cmd_parts.extend([str(m) for m in ms])

    # Join with line continuations for readability
    cmd_str = " \
  ".join(cmd_parts)

    logger.info("Constructed wsclean command:\n%s", cmd_str)
    return WSCleanCommand(
        cmd=cmd_str,
        options=options,
        ms=ms,
        image_prefix_str=prefix,
    )

@singularity_wrapper
def run_wsclean_command(wsclean_command: WSCleanCommand, **kwargs) -> str:
    """Run a wsclean command using singularity wrapper

    Args:
        wsclean_command: The result of a wsclean command construction, containing the command and options
        **kwargs: Additional keyword arguments that will be passed to the singularity_wrapper

    Returns:
        str: the command that was executed
    """
    logger = get_run_logger()

    logger.info(f"WSclean command {wsclean_command.cmd}")

    return wsclean_command.cmd


def get_imset_from_prefix(
        prefix: str | Path,
        pol: str = "i",
        validate: bool = True,
        chanout: int | None = None,
        pbcor_done: bool = False
    ) -> ImageSet:
    """
    Get an ImageSet from a given prefix, i.e. wsclean '-name' argument or globstring pattern.

    args:
        prefix (str | Path): The prefix or glob string to match files against.
        pol (str): The polarisation to extract. Defaults to 'i' for Stokes I.
        validate (bool): Check whether all 'chanout' images are present? Defaults to True.
        chanout (int | None): Number of channels out, used for validation. If None, validate must be False
    
    returns:
        ImageSet: An ImageSet object containing the matched files.
    raises:
        ValueError: If the prefix is invalid or if the polarisation is not supported.

    """
    
    # Helper to glob all matches
    def glob_all(pattern: str) -> list[Path]:
        return sorted(glob(pattern))

    # Find files for a given product and polarization
    def find_pol_files(kind: str) -> list[Path]:
        if pol in ("q", "u"):
            pattern = f"{prefix}*-{pol.upper()}-{kind}*.fits"
            return sorted(glob(pattern))
        # Stokes I: exclude Q/U
        all_files = glob_all(f"{prefix}*-{kind}*.fits")
        return [p for p in all_files if "-Q-" not in p and "-U-" not in p]

    images = find_pol_files("image")
    dirty = find_pol_files("dirty")
    model = find_pol_files("model")
    residual = find_pol_files("residual")
    psf = find_pol_files("psf")

    # Source list file (e.g., txt or model list)
    sl = glob_all(f"{prefix}*source*txt")
    source_list = sl[0] if sl else None

    imset = ImageSet(
        prefix=prefix,
        image=images,
        psf=psf,
        dirty=dirty,
        model=model,
        residual=residual,
        source_list=source_list,
    )

    if validate:
        if chanout is None:
            raise ValueError("chanout must be specified if validate is True")
        validate_imset(imset, chanout, pbcor_done=pbcor_done)

    return imset

def get_wsclean_output(
        wsclean_command: WSCleanCommand,
        pol: str = 'i',
        validate: bool =  True
    ) -> ImageSet:
    """Parse a wsclean command, extract the prefix, and gather output files into an ImageSet.
    
    Assumes WSClean commands are constructed with the `-name` argument with absolute paths

    args:
        wsclean_command (WSCleanCommand): The command that was run
        pol (str): The polarisation to extract. Defaults to 'i' for Stokes I.

    """
    cmd = wsclean_command.cmd
    parts = cmd.split()
    try:
        idx = parts.index("-name")
        prefix = parts[idx + 1]
    except ValueError:
        raise ValueError("wsclean command missing '-name' argument for prefix")


    # Validate polarization
    pol = pol.lower()
    if pol not in ("i", "q", "u"):
        raise ValueError("pol must be 'i', 'q', or 'u'")

    chanout = wsclean_command.options.channels_out

    imset = get_imset_from_prefix(
        prefix,
        pol,
        validate,
        chanout,
        pbcor_done=wsclean_command.options.apply_primary_beam
    )

    return imset

def validate_imset(imset: ImageSet, nchan: int, pbcor_done: bool = False) -> None:
    """
    Validate that for each product (image, dirty, model, residual),
    there are nchan + 1 files (including MFS).
    """
    expected = nchan + 1
    products = {
        "image": imset.image,
        "dirty": imset.dirty,
        "model": imset.model,
        "residual": imset.residual,
    }

    # if pbcor was done, these images will have 2x expected number of files
    can_be_pbcor = ["image", "model", "residual"] 

    for kind, files in products.items():
        count = len(files) if files is not None else 0

        if kind in can_be_pbcor and pbcor_done:
            # If pbcor was done, expect 2x the number of files per channel
            count //= 2 + 1 # plus one for MFS

        if count != expected:
            raise ValueError(
                f"Expected {expected} '{kind}' files, but found {count}."
            )
    return


def run_wsclean(
    ms: Path | list[Path],
    working_dir: Path,
    lofar_container: Path,
    prefix: str,
    options: WSCleanOptions,
    expected_pols: list[str],
) -> list[ImageSet]:
    """
    Run WSClean once with the given options and return ImageSet(s) for expected_pols.

    expected_pols: e.g. ['i'] or ['q','u']
    """
    logger = get_run_logger()

    # Ensure output dir exists
    wsclean_output_dir = working_dir / "IQUimages"
    if not wsclean_output_dir.exists():
        logger.info(f"Creating wsclean output directory at {wsclean_output_dir}")
        wsclean_output_dir.mkdir(parents=True, exist_ok=True)

    # Prefix lives inside IQUimages
    full_prefix = str(wsclean_output_dir / prefix)

    # Build command
    wsclean_cmd = create_wsclean_command(options, ms, full_prefix)

    # Probe outputs
    pre = [get_wsclean_output(wsclean_cmd, pol=p.lower(), validate=False) for p in expected_pols]
    need_run = any(len(imgset.image) == 0 for imgset in pre)

    if need_run:
        logger.info(f"Running WSClean for pol={options.pol} imaging")
       
        # currently assume all MSes at least live in the same parent directory
        if not isinstance(ms, list):
            ms = [ms]

        run_wsclean_command(
            wsclean_command=wsclean_cmd,
            container=lofar_container,
            bind_dirs=[ms[0].parent, wsclean_output_dir],
        )
        post = [get_wsclean_output(wsclean_cmd, pol=p.lower(), validate=True) for p in expected_pols]
        return post
    else:
        logger.info(f"WSClean output for pol(s) {expected_pols} already exists, skipping run.")
        return pre
