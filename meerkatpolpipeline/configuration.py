"""Loading a strategy file. Inspired from flint
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from meerkatpolpipeline.caracal._caracal import CrossCalOptions
from meerkatpolpipeline.check_calibrator.check_calibrator import CheckCalibratorOptions
from meerkatpolpipeline.check_nvss.compare_to_nvss import CompareNVSSOptions
from meerkatpolpipeline.cube_imaging.cube_imaging import (
    CoarseCubeImagingOptions,
    FineCubeImagingOptions,
)
from meerkatpolpipeline.download.download import DownloadOptions
from meerkatpolpipeline.logging import logger
from meerkatpolpipeline.rmsynth.rmsynth1d import RMSynth1Doptions
from meerkatpolpipeline.rmsynth.validate_rmsynth import ValidateRMsynth1dOptions
from meerkatpolpipeline.scienceplots.science_rms1d import ScienceRMSynth1DOptions
from meerkatpolpipeline.selfcal._facetselfcal import SelfCalOptions
from meerkatpolpipeline.utils.utils import add_timestamp_to_path
from meerkatpolpipeline.validation.validate_field import ValidateFieldOptions

# Known headers must **always** be present in the strategy file
KNOWN_HEADERS = (
    "defaults",
     "version",
     "targetfield",
     "lofar_container",
     "casa_container",
     )
# Optional headers are not required, but if present must be in the correct format
OPTIONAL_HEADERS = (
    "casa_additional_bind",
)
# Known options are optional, but if present must be in the correct format
KNOWN_OPERATIONS = (
    "download_preprocess",
    "crosscal",
    "check_calibrator",
    "selfcal",
    "coarse_cube_imaging",
    "compare_to_nvss",
    "validation",
    "grid_freq_axis",
    "fine_cube_imaging",
    "rmsynth1d",
    "validate_rmsynth1d",
    "rmsynth3d",
    "validate_rmsynth3d",
    "science_plots_rms1d",
    "science_plots_rms3d",
)
FORMAT_VERSION = 0.1
STRATEGY_OPTIONS_MAPPING = {
    "download_preprocess": DownloadOptions,
    "crosscal": CrossCalOptions,
    "check_calibrator": CheckCalibratorOptions,
    "selfcal": SelfCalOptions,
    "coarse_cube_imaging": CoarseCubeImagingOptions,
    "compare_to_nvss": CompareNVSSOptions,
    "validation": ValidateFieldOptions,
    "fine_cube_imaging": FineCubeImagingOptions,
    "rmsynth1d": RMSynth1Doptions,
    "validate_rmsynth1d": ValidateRMsynth1dOptions,
    "science_plots_rms1d": ScienceRMSynth1DOptions,
    # TODO
}

class Strategy(dict):
    """Base representation for handling a loaded strategy"""
    pass

def load_and_copy_strategy(
    reduction_strategy: Path,
    output_path: Path | None = None, 
) -> Strategy | None:
    """Load a strategy file and copy a timestamped version into the output directory
    that would contain the science processing.

    Args:
        reduction_strategy (Optional[Path], optional): Location of the strategy file. Defaults to None.
        output_path (Path): Where the strategy file should be copied to (e.g. where the data would be processed)

    Returns:
        Union[Strategy, None]: The loaded strategy file
    """
    return (
        load_strategy_yaml(
            input_yaml=copy_and_timestamp_strategy_file(
                output_dir=output_path,
                input_yaml=reduction_strategy,
            ),
            verify=True,
        )
        if reduction_strategy
        else None
    )

def load_strategy_yaml(input_yaml: Path, verify: bool = True) -> Strategy:
    """Load in a configuration file, which
    will be used to form the strategy for data reduction

    Args:
        input_yaml (Path): The imaging strategy to use
        verify (bool, optional): Apply some basic checks to ensure a correctly formed strategy. Defaults to True.

    Returns:
        Strategy: The parameters of the imaging and self-calibration to use.
    """

    logger.info(f"Loading {input_yaml} file. ")

    with open(input_yaml) as in_file:
        input_strategy = Strategy(yaml.load(in_file, Loader=yaml.Loader))

    if verify:
        verify_configuration(input_strategy=input_strategy)

    return input_strategy

def verify_configuration(input_strategy: Strategy, raise_on_error: bool = True) -> bool:
    """Perform basic checks on the configuration file

    Args:
        input_strategy (Strategy): The loaded configuration file structure
        raise_on_error (bool, optional): Whether to raise an error should an issue in thew config file be found. Defaults to True.

    Raises:
        ValueError: Whether structure is valid

    Returns:
        bool: Config file is not valid. Raised only if `raise_on_error` is `True`
    """

    errors: list[str] = []

    for known_header in KNOWN_HEADERS:
        if known_header not in input_strategy.keys():
            errors.append(
                f"Required section header {known_header} missing from input configuration."
            )

    if "version" in input_strategy.keys():
        if input_strategy["version"] != FORMAT_VERSION:
            errors.append(
                f"Version mismatch. Expected {FORMAT_VERSION}, got {input_strategy['version']}"
            )

    # make sure the main components of the file are there
    unknown_headers = [
        header
        for header in input_strategy.keys()
        if header not in KNOWN_HEADERS and header not in KNOWN_OPERATIONS and header not in OPTIONAL_HEADERS
    ]

    if unknown_headers:
        errors.append(f"{unknown_headers=} found. Supported headers: {KNOWN_HEADERS} or {OPTIONAL_HEADERS}. Known operations: {KNOWN_OPERATIONS}")

    for operation in KNOWN_OPERATIONS:
        if operation in input_strategy.keys():

            try:
                options = get_options_from_strategy(
                    strategy=input_strategy, operation=operation
                )
                try:
                    _ = STRATEGY_OPTIONS_MAPPING[operation](**options)
                except TypeError as typeerror:
                    errors.append(
                        f"{operation=} incorrectly formed. {typeerror} "
                    )
            except Exception as exception:
                errors.append(f"{exception}")

            if operation == "caracal":
                # double check that either all the calibrators are set
                # or that "auto_determine_obsconf" is set
                try:
                    if not options["auto_determine_obsconf"]:
                        assert options["obsconf_target"] is not None, "obsconf_target should be set if 'auto_determine_obsconf is False"
                        assert options["obsconf_gcal"] is not None, "obsconf_gcal should be set if 'auto_determine_obsconf is False"
                        assert options["obsconf_xcal"] is not None, "obsconf_xcal should be set if 'auto_determine_obsconf is False"
                        assert options["obsconf_bpcal"] is not None, "obsconf_bpcal should be set if 'auto_determine_obsconf is False"
                        assert options["obsconf_fcal"] is not None, "obsconf_fcal should be set if 'auto_determine_obsconf is False"
                except Exception as exception:
                    errors.append(f"{exception}")

            if operation == "coarse_cube_imaging":
                # double check that if run_pybdsf is set, also_image_for_mfs is set
                try:
                    if options["run_pybdsf"]:
                        assert options["also_image_for_mfs"], "Error in coarse_cube_imaging settings: If run_pybdsf is True, also_image_for_mfs must be True as well."
                except Exception as exception:
                    errors.append(f"{exception}")
    

    valid_config = len(errors) == 0
    if not valid_config:
        for error in errors:
            logger.warning(error)

        if raise_on_error:
            raise ValueError("Configuration file not valid. Please see warnings above.")

    return valid_config

def get_options_from_strategy(
    strategy: Strategy | None | Path,
    operation: str,
) -> dict[Any, Any]:
    f"""Extract a set of options from a strategy file to use in a pipeline
    run. If the mode exists in the default section, these are used as a base.

    Args:
        strategy (Union[Strategy,None,Path]): A loaded instance of a strategy file. If `None` is provided then an empty dictionary is returned. If `Path` attempt to load the strategy file.
        operation (Optional[str], optional): Get options related to a specific operation. Defaults to None. Allowed values in {KNOWN_OPERATIONS}

    Raises:
        ValueError: An unrecongised value for `round`.
        AssertError: An unrecongised value for `round`.

    Returns:
        Dict[Any, Any]: Options specific to the requested set
    """

    if strategy is None:
        return {}
    elif isinstance(strategy, Path):
        strategy = load_strategy_yaml(input_yaml=strategy)

    # Some sanity checks
    assert isinstance(strategy, (Strategy, dict)), (
        f"Unknown input strategy type {type(strategy)}"
    )
    if operation not in KNOWN_OPERATIONS:
        raise ValueError(
            f"{operation=} is not recognised. Known operations are {KNOWN_OPERATIONS}"
        )

    # step one, get the user supplied defaults for this operation (e.g. 'default')
    options = dict(**strategy["defaults"][operation]) if operation in strategy["defaults"] else {}

    # step two, add a "targetfield" key to every step
    options.update(targetfield=strategy["targetfield"])

    logger.debug(f"Defaults for {operation=}, {options=}")


    # A default empty dict
    update_options = {}

    assert operation in strategy, f"{operation=} not in {strategy.keys()}"

    operation_scope = strategy.get(operation)

    if operation_scope:
        # print(f"operation specific options: {operation_scope}")
        update_options = dict(**operation_scope)

    if update_options:
        logger.debug(f"Updating options with {update_options=}")
        options.update(update_options)

    # Finally, pass it by the class in case there are any parameters that the user didnt specify
    options = dict(STRATEGY_OPTIONS_MAPPING[operation](**options))

    return options

def copy_and_timestamp_strategy_file(output_dir: Path, input_yaml: Path) -> Path:
    """Timestamp and copy the input strategy file to an
    output directory

    Args:
        output_dir (Path): Output directory the file will be copied to
        input_yaml (Path): The file to copy

    Returns:
        Path: Copied and timestamped file path
    """
    stamped_reduction_strategy = (
        output_dir / add_timestamp_to_path(input_path=input_yaml).name
    )
    logger.info(f"Copying {input_yaml.absolute()} to {stamped_reduction_strategy}")
    shutil.copyfile(input_yaml.absolute(), stamped_reduction_strategy)

    return Path(stamped_reduction_strategy)

def log_enabled_operations(strategy: Strategy) -> list:
    operations = []
    for key in strategy.keys():
        if key not in KNOWN_HEADERS and key not in OPTIONAL_HEADERS:
            if strategy[key]['enable']:
                operations.append(key)

    logger.info(f"Will perform the following operations: {operations}")
    return operations