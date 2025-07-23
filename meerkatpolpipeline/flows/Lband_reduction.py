"""A prefect based pipeline that:
- will perform meerKAT L-band data processing
- given an input strategy file
"""

from __future__ import annotations

from pathlib import Path

from configargparse import ArgumentParser
from prefect import flow, task  #, tags, unmapped
from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal import _caracal
from meerkatpolpipeline.check_calibrator import check_calibrator
from meerkatpolpipeline.configuration import (
    Strategy,
    get_options_from_strategy,
    load_and_copy_strategy,
    log_enabled_operations,
)
from meerkatpolpipeline.download.download import download_and_extract

# from meerkatpolpipeline.logging import logger

# TODO: submit prefect tasks instead of running sequentially?
# TODO: look into logging to prefect dashboard with custom logger


def check_caracal_run():
    print("TODO: verify that caracal completed succesfully, and return the -cal.ms path location")


@flow(name="MeerKAT L-band pipeline", log_prints=True)
def process_science_fields(
    strategy: Strategy,
    working_dir: Path
) -> None:
    """
    Flow that will execute all the enabled steps as tasks. 

    Each task will be done in a subdirectory
    """

    logger = get_run_logger()

    enabled_operations = log_enabled_operations(strategy)

    ########## step 1: download & clip channels ##########
    if "download" in enabled_operations:
        # create subdirectory 'download'
        download_workdir = working_dir / "download"
        download_workdir.mkdir(exist_ok=True) # runs can be repeated

        download_options = get_options_from_strategy(strategy, operation="download")
        
        # download and extract
        task_start_download = task(download_and_extract, name="download_and_extract")
        task_start_download(download_options, working_dir=download_workdir)

        # do parang correction

        # clip if requested


        # TODO: add clipping, which should be checked on re-run, even if the MS is downloaded
    else:
        logger.warning("Download step is disabled, skipping download and clipping of channels.")


    ########## step 2: cross-calibration with either casa or caracal ##########
    if "caracal" in enabled_operations and "casacrosscal" in enabled_operations:
        logger.warning("Both caracal and casacrosscal are enabled. This is not supported, please choose one of them.")
        raise ValueError("Both caracal and casacrosscal are enabled. This is not supported, please choose one of them.")
    
    if "caracal" in enabled_operations:
        caracal_options = get_options_from_strategy(strategy, operation="caracal")

        caracal_workdir = caracal_options['msdir'] # caracal will always work in the msdir directory
        # caracal_workdir.mkdir(exist_ok=True) # runs can be repeated

        _caracal.start_caracal(caracal_options, working_dir=caracal_workdir)
    else:
        logger.info("Caracal step is disabled, skipping caracal cross-calibration.")

    # TODO: optionally second step could also be casa script. 
    if "casacrosscal" in enabled_operations:
        print("TODO")

    elif "caracal" not in enabled_operations:
        logger.warning("Both caracal and casacrosscal are disabled. Skipping cross-calibration")

    else:
        logger.info("Casacrosscal step is disabled, skipping casacrosscal cross-calibration.")
        

    ########## step 3: check polarisation calibrator ##########
    if "check_calibrator" in enabled_operations:
        check_calibrator_workdir = working_dir / "check_calibrator"
        check_calibrator_workdir.mkdir(exist_ok=True)

        check_calibrator_options = get_options_from_strategy(strategy, operation="check_calibrator")

        # need location of calibrator measurement set to split the polcal


        check_calibrator(check_calibrator_options, working_dir=check_calibrator_workdir)
    else:
        logger.warning("Check calibrator step is disabled, skipping checking of polarisation calibrator.")


    ########## step 4: facetselfcal ##########
    # DI
    # DD
    # Extract
    # then DI or DD on extracted field as well

    ########## step 5: IQUV cube image 12 channel ##########
    # do I image separate from QUV
    

    ########## step 6: prelim. check of IQUV cubes vs NVSS ##########


    ########## step 7: Resample the frequency axis if requested (required for L-band + UHF imaging) ##########


    ########## step 8: Many-channel IQU imaging ##########


    ########## step 9: RM synthesis 1D ##########


    ########## step 10: Verify RMSynth1D ##########


    ########## step 11: RM synthesis 3D ##########


    ########## step 12: Verify RMSynth3D ##########


    ########## step 13: Science plots ##########

    
def setup_run(
    strategy_path: Path,
    working_dir: Path
) -> None:
    
    # load strategy and copy timestamp to working dir
    strategy = load_and_copy_strategy(strategy_path, working_dir)

    # determine target from strategy file, in the caracal option
    # note: this means 'targetfield' should be set even if caracal is disabled.
    # TODO: this could be improved by setting the targetfield as a separate input in the config file?
    target = strategy['targetfield']

    print(f"Starting pipeline for {target=}")

    # when testing without prefect
    # process_science_fields(
    #     strategy=strategy,
    #     working_dir=working_dir
    # )

    process_science_fields.with_options(
        name=f"MeerKAT L-band pipeline - {target}"
        # , task_runner=dask_task_runner
    )(
        strategy=strategy,
        working_dir=working_dir
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="MeerKAT L-band data processing")

    parser.add_argument(
        "--cli-config-file", type=str, help="Path to strategy configuration file"
    )
    parser.add_argument(
        "--working-dir", type=str, default="./", help="Path to main working directory. Default ./"
    )
    
    return parser


def cli() -> None:

    parser = get_parser()

    args = parser.parse_args()

    setup_run(
        strategy_path=Path(args.cli_config_file),
        working_dir=Path(args.working_dir)
    )


if __name__ == "__main__":
    cli()