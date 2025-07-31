"""Simple interface into facetselfcal, based on wsclean/wsclean.py

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from prefect.logging import get_run_logger

from meerkatpolpipeline.measurementset import check_ms_timesteps
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.sclient import singularity_wrapper
from meerkatpolpipeline.wsclean.wsclean import ImageSet, get_imset_from_prefix


class SelfCalOptions(BaseOptions):
    """
    A container to handle selfcal options from the input config.yaml file
    """
    enable: bool
    """enable this step? Required parameter"""
    facetselfcal_directory: Path
    """directory to lofar_facet_selfcal. Required"""
    targetfield: str | None = None
    """name of targetfield. Propagated to all steps."""
    selfcal_clip_chan_start: int = 20
    """clip channels from MS before this number. Keep in mind that this is after cross-cal averaging"""
    selfcal_clip_total_nchan: int = 894
    """number of channels to use from the input MS (0 means till the end). Keep in mind that this is after cross-cal averaging"""
    imsize: int = 8192
    """number of pixels on one side, in square imaging. 'Radio images are square because radio imagers make square images.' - RvW """
    pixelsize: float | None = None
    """pixel size in arcseconds. If None, determined automatically."""
    

class FacetselfcalOptions(BaseOptions):
    """A container to handle facetselfcal.py options. 
    Attributes correspond exactly to the CLI flags of facetselfcal.py."""

    imagename: str = "image"
    """Input image prefix. Facetselfcal default is 'image' """

    msinnchan: int | None = None
    """Before averaging, only take this number of input channels. The default is None."""

    msinstartchan: int | None = 0
    """Before averaging, start channel for --msinnchan. The default is 0."""

    forwidefield: bool = False
    """Keep solutions such that they can be used for widefield imaging/screens."""

    noarchive: bool = False
    """Do not archive the data"""

    solint_list: list[str] | None = None
    """Solution interval corresponding to solution types (in same order as soltype-list input)"""

    soltype_list: list[str] | None = None
    """"List with solution types. Possible input: 'complexgain', 'scalarcomplexgain', 'scalaramplitude', 'amplitudeonly', 'phaseonly', 'fulljones', 'rotation', 'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude', 'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase', 'faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase' , 'tec', 'tecandphase', 'scalarphase', 'scalarphasediff', 'scalarphasediffFR', 'phaseonly_phmin', 'rotation_phmin', 'tec_phmin', 'tecandphase_phmin', 'scalarphase_phmin', 'scalarphase_slope', 'phaseonly_slope'. The default is [tecandphase,tecandphase,scalarcomplexgain]."""

    nchan_list: list[int] | None = None
    """Number of channels corresponding to solution types (in same order as soltype-list input)"""

    soltypecycles_list: list[int] | None = None
    """Selfcalcycle where step from soltype-list starts. """

    smoothnessconstraint_list: list[float] | None = None
    """List with frequency smoothness values in MHz (in same order as soltype-list input)"""

    imsize: int | None = None
    """Image size in pixels (single dimension, square images)"""

    pixelsize: float | None = None
    """Pixels size in arcsec. Typically, 3.0 for LBA and 1.5 for HBA for the Dutch stations (these are also the default values). For LOFAR ILT the defaults are 0.04 and 0.08 for HBA and LBA, repspectively. For MeerKAT the defaults are 1.8,1.0, and 0.5 for UHF, L, and S-band, repspectively.'"""

    niter: int | None = None
    """Number of cleaning iterations. Computed automatically if None"""

    channelsout: int | None = None
    """Number of channels out during imaging. Facetselfcal default is 6"""

    uvminim: float | None = None
    """Inner uv-cut for imaging in lambda. The default is 80."""

    fitspectralpol: int | None = None
    """Polynomial order for fitting spectral index"""

    paralleldeconvolution: int | None = None
    """Parallel-deconvolution size for WSCLean (see WSClean documentation). The default is 0 which means the parallel deconvolution value is determined automatically in facetselfcal. For large images, values around 1000-2000 usually work well. For any value below zero the option is turned off."""

    parallelgridding: int | None = None
    """Parallel-gridding for WSClean (see WSClean documentation). The default is 0 which means it is set automatically"""

    beamcor: str = 'auto' # facetselfcal will complain if its not at auto
    """Correct the visibilities for beam in the phase center. Possible values are 'no' 'yes' 'auto'. This is a LOFAR specific option. (default is auto, auto means the LOFASR beam is taken out in the curent phase center, tolerance for that is 10 arcsec). """

    start: int | None = None
    """Start selfcal cycle at this iteration number. The default is 0."""

    stop: int | None = 4
    """Stop selfcal cycle at this iteration number. The default is 4."""

    multiscale: bool = False
    """Enable multiscale cleaning"""

    multiscale_start: int | None = None
    """Starting scale for multiscale cleaning"""

    useaoflagger: bool = False
    """Enable AOFlagger for RFI flagging"""

    aoflagger_strategy: str | None = None
    """Path to AOFlagger strategy file"""

    stopafterpreapply: bool = False
    """Stop execution after the preapply step"""

    dde: bool = False
    """Enable direction-dependent effects calibration"""

    facetdirections: str | None = None
    """ASCII csv file containing facet directions. File needs at least two columns with decimal degree RA and Dec. Default is None."""

    normamps_list: list[str | None] | None = None
    """List with amplitude normalization options. Possible input: 'normamps', 'normslope', 'normamps_per_ant, 'normslope+normamps', 'normslope+normamps_per_ant', or None. The default is [normamps,normamps,normamps,etc]. Only has an effect if the corresponding soltype outputs and amplitude000 table (and is not fulljones)."""

    remove_outside_center: bool = False
    """Subtract sources that are outside the central parts of the FoV, square box is used in the phase center with sizes of 3.0, 2.0, 1.5 degr for MeerKAT UHF, L, and S-band, repspectively. In case you want something else set --remove-outside-center-box. In case of a --DDE solve the solution closest to the box center is applied."""

    remove_outside_center_box: str | float | None = None
    """float [deg] or User defined box DS9 region file to subtract sources that are outside this part of the image, see also --remove-outside-center. If "keepall" is set then no subtract is done and everything is kept, this is mainly useful if you are already working on box-extracted data. If number is given a boxsize of this size (degr) will be used in the phase center. In case of a --DDE solve the solution closest to the box center is applied (unless "keepall" is set)."""


class FacetselfcalCommand(BaseOptions):
    """Simple container for a facetselfcal command."""

    cmd: str
    """The constructed facetselfcal command that would be executed."""
    options: FacetselfcalOptions
    """The set of facetselfcal options used for imaging"""
    ms: list[Path] | Path
    """The measurement set(s) that have been included in the facetselfcal command. """
    image_prefix_str: str | None = None
    """The prefix of the images that will be created"""
    cleanup: bool = True
    """Will clean up the dirty images/psfs/residuals/models when the imaging has completed"""
    image_set: ImageSet | None = None
    """The set of images produced by facetselfcal"""


def create_facetselfcal_command(
        options: FacetselfcalOptions,
        ms: list[Path] | np.ndarray[Path] | Path,
        facetselfcal_directory: Path,
        prefix: str | None = None
    ) -> FacetselfcalCommand:
    """
    Construct a facetselfcal command from the given options and measurement set(s).
    """
    logger = get_run_logger()
    opt_dict = vars(options)
    cmd_parts: list[str] = ["python", f"{str(facetselfcal_directory)}/facetselfcal.py"]

    # Build flags from options
    for name, val in opt_dict.items():
        if val is None:
            continue
        flag = f"--{name.replace('_', '-')}" # all facetselfcal arguments are '--argument'
        if isinstance(val, bool):
            if val:
                cmd_parts.append(flag)
        elif name == "size":
            # always output two dimensions
            dims = (val, val) if isinstance(val, int) else val
            cmd_parts.extend([flag] + [str(d) for d in dims])
        elif isinstance(val, list):
            # facetselfcal expects lists literally as strings
            cmd_parts.extend([flag, str(val)])
        elif isinstance(val, tuple):
            cmd_parts.extend([flag, ",".join(str(v) for v in val)])
        else:
            cmd_parts.extend([flag, str(val)])

    # Handle output prefix
    if prefix:
        cmd_parts.extend(["-name", prefix])

    if isinstance(ms, Path):
        # Append the measurement set
        cmd_parts.append(str(ms))
    elif isinstance(ms, (list,np.ndarray)):
        # append the measurement sets
        cmd_parts.extend([str(ms_i) for ms_i in ms])
    else:
        raise ValueError(f"{ms} is of type {type(ms)}. Expected list or Path")

    # Join with line continuations for readability
    cmd_str = " \
  ".join(cmd_parts)

    logger.info("Constructed facetselfcal command:\n%s", cmd_str)
    return FacetselfcalCommand(
        cmd=cmd_str,
        options=options,
        ms=ms,
        image_prefix_str=prefix,
    )

@singularity_wrapper
def run_facetselfcal_command(facetselfcal_command: FacetselfcalCommand, **kwargs) -> str:
    """Run a facetselfcal command using singularity wrapper

    Note that all arguments should be given as kwargs to not confuse singularity wrapper

    Args:
        facetselfcal_command: The result of a facetselfcal command construction, containing the command and options
        **kwargs: Additional keyword arguments that will be passed to the singularity_wrapper

    Returns:
        str: the command that was executed
    """
    logger = get_run_logger()

    logger.info(f"facetselfcal command {facetselfcal_command.cmd}")

    return facetselfcal_command.cmd


def get_facetselfcal_output(
        facetselfcal_command: FacetselfcalCommand,
        pol: str = 'i',
        validate: bool =  True
    ) -> ImageSet:
    """Parse a facetselfcal command, extract the prefix, and gather output files into an ImageSet.
    
    Assumes facetselfcal commands are constructed with the `-name` argument with absolute paths

    args:
        facetselfcal_command (FacetselfcalCommand): The command that was run
        pol (str): The polarisation to extract. Defaults to 'i' for Stokes I.

    """
    cmd = facetselfcal_command.cmd
    parts = cmd.split()
    try:
        idx = parts.index("-name")
        prefix = parts[idx + 1]
    except ValueError:
        raise ValueError("facetselfcal command missing '-name' argument for prefix")


    # Validate polarization
    pol = pol.lower()
    if pol not in ("i", "q", "u"):
        raise ValueError("pol must be 'i', 'q', or 'u'")

    chanout = facetselfcal_command.options.channels_out

    imset = get_imset_from_prefix(prefix, pol, validate, chanout)

    return imset

def get_options_facetselfcal_preprocess(selfcal_options: SelfCalOptions):
    """
    Hardcoded set of options for preprocessing a measurement set before facetselfcal
        given some user input SelfcalOptions (for clipping channels)
    """

    opt_dict = {
        "noarchive": True,
        "msinstartchan": selfcal_options['selfcal_clip_chan_start'],
        "msinnchan": selfcal_options['selfcal_clip_total_nchan'] ,
        "stopafterpreapply": True,
        "useaoflagger": True,
        "aoflagger_strategy": "default_StokesQUV.lua", # do we need to give full path?
        "imsize" : 8192 # not used, but required by facetselfcal...
    }

    facetselfcal_options = FacetselfcalOptions(**opt_dict)
    return facetselfcal_options

def do_facetselfcal_preprocess(
        selfcal_options: SelfCalOptions,
        ms: Path,
        workdir: Path,
        lofar_container: Path
    ) -> Path | list[Path]:
    """Run the facetselfcal preprocess step.

    This does additional channel clipping and aoflagging with the default_StokesQUV.lua strategy
    and it splits the measurement set into X parts, because DP3 doesnt handle unequal time axes very well.

    Returns:
        Path or list[Path] to the facetselfcal preprocessed MSes, with irregular timeaxis split.

    """
    logger = get_run_logger()

    logger.info(f"Starting facetselfcal preprocess step in {workdir}")

    # Check if preprocess was already done by a previous run.
    preprocessed_msdir = workdir / "split_measurements"
    all_preprocessed_mses = np.array(sorted(preprocessed_msdir.glob("*.ms")))
    if len(all_preprocessed_mses) > 0:
        logger.info(f"Found {len(all_preprocessed_mses)} existing preprocessed MSes in {preprocessed_msdir}")
        logger.info("Assuming facetselfcal preprocess step already done. Not repeating.")
    
        # check how many have at least 20 timesteps (required by facetselfcal)
        bad_ntimes = check_ms_timesteps(all_preprocessed_mses, ntimes_cutoff=20)
        logger.info(f"Out of the {len(all_preprocessed_mses)} mses, {np.sum(bad_ntimes)} have less than {20} timesteps and not be used.")

        return all_preprocessed_mses[~bad_ntimes]

    # Otherwise, build and start preprocess command
    facetselfcal_options = get_options_facetselfcal_preprocess(selfcal_options)

    # note the difference between selfcal_options (from user via .yaml file) and facetselfcal_options (hardcoded mostly)
    facetselfcal_cmd = create_facetselfcal_command(facetselfcal_options, ms, selfcal_options['facetselfcal_directory'])

    # all arguments should be given as kwargs to not confuse singularity wrapper
    run_facetselfcal_command(
        facetselfcal_command=facetselfcal_cmd,
        container=lofar_container,
        bind_dirs=[
            selfcal_options['facetselfcal_directory'],
            ms.parent,
        ],
        options = ["--pwd", str(workdir)] # execute command in selfcal workdir
    )

    all_preprocessed_mses = np.array(sorted(preprocessed_msdir.glob("*.ms")))
    if len(all_preprocessed_mses) == 0:
        raise ValueError(f"Found no preprocessed mses at the expected location: {preprocessed_msdir}. Something went wrong?")

    logger.info(f"Measurement set has been split into {len(all_preprocessed_mses)}, can be found in {preprocessed_msdir}")
    
    # check how many have at least 20 timesteps (required by facetselfcal)
    bad_ntimes = check_ms_timesteps(all_preprocessed_mses, ntimes_cutoff=20)
    logger.info(f"Out of the {len(all_preprocessed_mses)} mses, {np.sum(bad_ntimes)} have less than {20} timesteps and not be used.")
    all_preprocessed_mses = all_preprocessed_mses[~bad_ntimes]

    # facetselfcal by default makes a copy of the MS, but it should also have produced
    # timesplit MSes in the 'split_measurements' subdirectory. 
    # so we can remove the copy of the MS
    facetselfcal_ms_copy = ms.with_name(ms.name.replace('.ms','.ms.copy'))
    logger.info(f"Removing facetselfcal ms.copy {facetselfcal_ms_copy}")
    facetselfcal_ms_copy.unlink()

    return all_preprocessed_mses


def get_options_facetselfcal_DI(selfcal_options: SelfCalOptions):
    """
    Hardcoded set of options for DIcal with facetselfcal
        given some user input SelfcalOptions 
    
        TODO: can extend user input SelfcalOptions with things like solint-list, soltype-list, channelsout etc.
              to expose facetselfcal parameters to the user via the .yaml file
              analogous to imsize, pixelsize
    """

    opt_dict = {
        "imsize" : selfcal_options['imsize'],
        "pixelsize": selfcal_options['pixelsize'],
        "noarchive": True,
        "forwidefield": True,
        "solint_list": ['1min'],
        "soltype_list": ['scalarphase'],
        "nchan_list": [1],
        "soltypecycles_list": [0],
        "smoothnessconstraint_list": [100.],
        "niter": 75000,
        "channelsout": 12,
        "uvminim": 10,
        "fitspectralpol": 9,
        "paralleldeconvolution": 1200,
        "parallelgridding": 4,
        "start": 0,
        "stop": 3,
        "multiscale": True,
        "multiscale_start": 0
    }

    facetselfcal_options = FacetselfcalOptions(**opt_dict)
    return facetselfcal_options


def do_facetselfcal_DI(
        selfcal_options: SelfCalOptions,
        all_preprocessed_mses: list[Path] | Path,
        workdir: Path,
        lofar_container: Path
    ) -> Path:
    """Run the facetselfcal Direction Independent (DI) self-calibration step.

    Args:
        selfcal_options       : dict(SelfCalOptions) : user input via yaml file
        all_preprocessed_mses : list                 : output from do_facetselfcal_preprocess()
        workdir               : Path                 : where to run facetselfcal_DI
        lofar_container       : Path                 : lofar software

    Returns:
        Path: new Path to the facetselfcal calibrated MS

    """
    logger = get_run_logger()

    logger.info(f"Starting facetselfcal DI step in {workdir}.")

    # Check if DI step was already done by a previous run.
    print("TODO: check if already done")
    # preprocessed_msdir = workdir / "split_measurements"
    # all_preprocessed_mses = list(sorted(preprocessed_msdir.glob("*.ms")))
    # if len(all_preprocessed_mses) > 0:
    #     logger.info(f"Found {len(all_preprocessed_mses)} existing preprocessed MSes in {preprocessed_msdir}")
    #     logger.info("Assuming facetselfcal preprocess step already done. Not repeating.")
    #     return all_preprocessed_mses

    # Otherwise, build and start DI command
    facetselfcal_options = get_options_facetselfcal_DI(selfcal_options)

    # note the difference between selfcal_options (from user via .yaml file) and facetselfcal_options (hardcoded mostly)
    facetselfcal_cmd = create_facetselfcal_command(
        options=facetselfcal_options,
        ms=all_preprocessed_mses,
        facetselfcal_directory=selfcal_options['facetselfcal_directory']
    )

    if isinstance(all_preprocessed_mses, (list,np.ndarray)):
        msdir = all_preprocessed_mses[0].parent
    else:
        msdir = all_preprocessed_mses.parent

    # all arguments should be given as kwargs to not confuse singularity wrapper
    run_facetselfcal_command(
        facetselfcal_command=facetselfcal_cmd,
        container=lofar_container,
        bind_dirs=[
            selfcal_options['facetselfcal_directory'],
            msdir,
        ],
        options = ["--pwd", str(workdir)] # execute command in selfcal workdir
    )

    print("TODO: collect results, check if ran succesfully.")