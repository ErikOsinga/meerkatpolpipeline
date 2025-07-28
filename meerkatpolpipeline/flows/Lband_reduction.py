"""A prefect based pipeline that:
- will perform meerKAT data processing
- given an input strategy file
"""

from __future__ import annotations

from pathlib import Path

from configargparse import ArgumentParser
from prefect import flow, task  #, tags, unmapped
from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal import _caracal
from meerkatpolpipeline.casacrosscal import casacrosscal
from meerkatpolpipeline.check_calibrator.check_calibrator import check_calibrator
from meerkatpolpipeline.configuration import (
    Strategy,
    get_options_from_strategy,
    load_and_copy_strategy,
    log_enabled_operations,
)
from meerkatpolpipeline.download.clipping import copy_and_clip_ms
from meerkatpolpipeline.download.download import download_and_extract
from meerkatpolpipeline.measurementset import load_field_intents_csv, msoverview_summary
from meerkatpolpipeline.sclient import run_singularity_command
from meerkatpolpipeline.utils.utils import find_calibrated_ms

# from meerkatpolpipeline.logging import logger

# TODO: submit prefect tasks instead of running sequentially?
# TODO: look into logging to prefect dashboard with custom logger


def check_caracal_run():
    print("TODO: verify that caracal completed succesfully, and return the -cal.ms path location")


@flow(name="MeerKAT pipeline", log_prints=True)
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

    lofar_container = Path(strategy['lofar_container'])
    casa_container = Path(strategy['casa_container'])

    # check for additional bind directory
    if 'casa_additional_bind' in strategy:
        casa_additional_bind = strategy['casa_additional_bind']
        if isinstance(casa_additional_bind, str):
            casa_additional_bind = [casa_additional_bind]
        casa_additional_bind = [Path(path) for path in casa_additional_bind]
    else:
        casa_additional_bind = []

    ########## step 1: download & clip channels ##########
    download_options = get_options_from_strategy(strategy, operation="download_preprocess")
    download_workdir = working_dir / "download"
    if "download_preprocess" in enabled_operations:
        # create subdirectory 'download'
        download_workdir.mkdir(exist_ok=True) # runs can be repeated

        #### 1.1 download and extract
        task_start_download = task(download_and_extract, name="download_and_preprocess")
        ms_path = task_start_download(download_options, working_dir=download_workdir)

        # get MS summary
        task_msoverview_summary = task(msoverview_summary, name="msoverview_summary")
        ms_summary = task_msoverview_summary(
            binds=[str(ms_path.parent)],
            container=lofar_container,
            ms=ms_path,
            output_to_file= download_workdir / "msoverview_summary.txt",
        )

        #### 1.2 parang correction

        # grab the script from the meerkatpolpipeline package
        from meerkatpolpipeline.download import download  # cant import casa scripts
        parang_script = Path(download.__file__).parent / "go_correct_parang.py"
                    
        cmd_parang = f"""python {parang_script} \
             --running-inside-sing \
             --test-already-done \
            {ms_path}
        """

        task_parang_correction = task(run_singularity_command, name="run_parang_correction")
        task_parang_correction(
            lofar_container,
            cmd_parang,
            bind_dirs=[ms_path.parent,parang_script.parent],
            max_retries=1
        )

        #### 1.3 copy CORRECTED_DATA over to a new MS with only DATA column including clip if requested
        preprocessed_ms = ms_path.parent / f"{ms_path.stem}_preprocessed.ms"

        if preprocessed_ms.exists():
            logger.info(f"Preprocessed MS already exists at {preprocessed_ms}, skipping clipping and copying.")

        else:
            logger.info(f"Preprocessed MS does not exist at {preprocessed_ms}, will copy and potentially clip channels from {ms_path}.")
            
            task_copy_and_clip_ms = task(copy_and_clip_ms, name="copy_and_clip_ms")
            task_copy_and_clip_ms(
                ms_path=ms_path,
                output_ms=preprocessed_ms,
                ms_summary=ms_summary, 
                clip_assumed_nchan=download_options['clip_assumed_nchan'],
                clip_chan_start=download_options['clip_chan_start'],
                clip_chan_end=download_options['clip_chan_end'],
                casa_container=casa_container,
                bind_dirs = [ms_path.parent, preprocessed_ms.parent] + casa_additional_bind
            )

        logger.info("Download and preprocessing step completed.")
        logger.info(f"Preprocessed MS can be found at {preprocessed_ms}")

        # TODO: clean up .tar.gz file if user requests?

    else:
        logger.warning("Download step is disabled, skipping download and preprocessing.")
        preprocessed_ms = ms_path.parent / f"{ms_path.stem}_preprocessed.ms"
        logger.info(f"Assuming preprocessed MS is available at {preprocessed_ms}. If not, please enable the download step in the strategy file.")


    ########## step 2: cross-calibration with either casa or caracal ##########
    crosscal_base_dir = working_dir / "crosscal" # /caracal or /casacrosscal
    crosscal_base_dir.mkdir(exist_ok=True) # runs can be repeated

    # will be written either by caracal step, or CASA (TODO)
    field_intents_csv = crosscal_base_dir / "field_intents.csv"

    if "crosscal" in enabled_operations:
        logger.info("Cross-calibration step is enabled, starting cross-calibration.")

        crosscal_options = get_options_from_strategy(strategy, operation="crosscal")


        # get MS summary, optionally with scan intents if user wants auto determined calibrators
        task_msoverview_summary = task(msoverview_summary, name="msoverview_preprocessed")
        ms_summary = task_msoverview_summary(
            binds=[str(preprocessed_ms.parent)],
            container=lofar_container,
            ms=preprocessed_ms,
            output_to_file= crosscal_base_dir / "msoverview_summary.txt",
            get_intents=crosscal_options["auto_determine_obsconf"]
        )
        if not crosscal_options['auto_determine_obsconf']:
            # then the calibrators should be set by the user. Write to an intent file
            logger.info(f"Auto-determination of obsconf is disabled, writing user-defined field intents to {field_intents_csv}.")
            _caracal.write_crosscal_csv(crosscal_options, output_path=field_intents_csv)

        logger.info(f"{ms_summary=}")
        

        # Then do specifically caracal or casa crosscal
        
        ############ 2. option 1: caracal cross-calibration step ############
        if crosscal_options['which'] == 'caracal':
            logger.info("Caracal cross-calibration step is enabled, starting caracal cross-calibration.")
            crosscal_dir = crosscal_base_dir / 'caracal'

            # set up tasks
            task_cleanup_caracal = task(_caracal.cleanup_caracal_run, name="cleanup_caracal_run")
            task_caracal_crosscal = task(_caracal.do_caracal_crosscal, name="caracal_crosscal")

            # Check if caracal was already done but maybe files were not moved from /download to /caracal
            calibrated_cal_ms = find_calibrated_ms(
                crosscal_base_dir.parent,
                preprocessed_ms,
                look_in_subdirs=[Path('download')],
                suffix="-cal.ms"
            )
            calibrated_target_ms = find_calibrated_ms(
                crosscal_base_dir.parent,
                preprocessed_ms,
                look_in_subdirs=[Path('download')],
                suffix=f"-{strategy['targetfield']}-corr.ms"
            )
            if calibrated_cal_ms is not None and calibrated_target_ms is not None:
                # if both are found, we assume the caracal run was already done
                logger.info(f"Found already calibrated target MS at {calibrated_target_ms}. Moving to /crosscal/caracal/ directory.")
                
                # move calibrated MS & caracal output from "$workdir/download/" to "$workdir/crosscal/caracal/" 
                calibrated_cal_ms, calibrated_target_ms = task_cleanup_caracal(
                    caracal_rundir=calibrated_target_ms.parent,
                    preprocessed_ms_name=preprocessed_ms.stem,
                    calibrated_cal_ms=calibrated_cal_ms,
                    calibrated_target_ms=calibrated_target_ms,
                    output_dir = (crosscal_base_dir / 'caracal')
                )


            else:
                # Do the actual caracal run.

                # note: this task also tests whether the calibrated MS exists in the crosscal_base_dir/caracal_crosscal directory
                calibrated_cal_ms, calibrated_target_ms = task_caracal_crosscal(
                    crosscal_options,
                    preprocessed_ms,
                    crosscal_base_dir,
                    ms_summary,
                    lofar_container # only required if user overwrites input MS.
                )

                if calibrated_cal_ms is None or calibrated_target_ms is None:
                    raise ValueError(
                        "Caracal cross-calibration did not return valid calibrated MS paths. Please check the caracal logs."
                    )
            
                logger.info(f"Caracal cross-calibration completed. Calibrated MS can be found at {calibrated_target_ms} and {calibrated_cal_ms}.")
                # move calibrated MS & caracal output from "$workdir/download/" to "$workdir/crosscal/caracal/" 
                calibrated_cal_ms, calibrated_target_ms = task_cleanup_caracal(
                    caracal_rundir=calibrated_target_ms.parent,
                    preprocessed_ms_name=preprocessed_ms.stem,
                    calibrated_cal_ms=calibrated_cal_ms,
                    calibrated_target_ms=calibrated_target_ms,
                    output_dir = (crosscal_base_dir / 'caracal')
                )


        ############ 2. option 2: casa cross-calibration step ############
        elif crosscal_options['which'] == 'casacrosscal':
            logger.info("Casa crosscal step is enabled, starting casa cross-calibration.")
            print("TODO")
            task_casa_crosscal = task(casacrosscal.do_casa_crosscal, name="casa_crosscal")
            crosscal_dir = task_casa_crosscal(crosscal_options, preprocessed_ms, crosscal_base_dir, ms_summary)

        ############ 2. option 3: oopsie, user has made a mistake ############
        else:
            logger.error(f"Invalid crosscal option '{crosscal_options['which']}'. Expected 'caracal' or 'casacrosscal'.")
            raise ValueError(f"Invalid crosscal option '{crosscal_options['which']}'. Expected 'caracal' or 'casacrosscal'.")

    else:
        logger.warning("Crosscal is disabled. Skipping cross-calibration.")
        crosscal_dir = None
        

    # check if we are proceeding with a caracal-cal.ms or a casacrosscal-cal.ms based on user options
    if crosscal_dir is None:
        logger.warning(f"No cross-calibration step was performed, checking for calibrated MS in {crosscal_base_dir} subdirectories")
        
        calibrated_cal_ms = find_calibrated_ms(
            crosscal_base_dir.parent,
            preprocessed_ms,
            suffix="-cal.ms"
        )
        calibrated_target_ms = find_calibrated_ms(
            crosscal_base_dir.parent,
            preprocessed_ms,
            suffix=f"-{strategy['targetfield']}-corr.ms"
        )
    
    
        if calibrated_cal_ms is None or calibrated_target_ms is None:
            raise ValueError(
                f"No calibrated target/cal measurement set found in {crosscal_base_dir}. Please enable either caracal or casacrosscal step in the strategy file."
            )

        # set crosscal dir wherever the calibrated MS was found
        crosscal_dir = calibrated_target_ms.parent

    ########## step 3: check polarisation calibrator ##########
    if "check_calibrator" in enabled_operations:
        check_calibrator_workdir = working_dir / "check_calibrator"
        check_calibrator_workdir.mkdir(exist_ok=True)

        check_calibrator_options = get_options_from_strategy(strategy, operation="check_calibrator")

        # get polcal field from field intents
        field_intents_dict = load_field_intents_csv(field_intents_csv)
        _, polcal_field = _caracal.obtain_by_intent(field_intents_dict, "polcal")
        
        # check for user overwrite
        if check_calibrator_options['crosscal_ms'] is None:
            check_calibrator_options['crosscal_ms'] = calibrated_cal_ms
        if check_calibrator_options['polcal_field'] is None:
            check_calibrator_options['polcal_field'] = polcal_field

        # split calibrator, make images, and validation plots
        task_check_calibrator = task(check_calibrator, name="check_calibrator")
        task_check_calibrator(
            check_calibrator_options,
            working_dir=check_calibrator_workdir,
            casa_container=casa_container,
            bind_dirs = [check_calibrator_options['crosscal_ms'].parent] + casa_additional_bind
        )

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
    working_dir: Path,
    append_to_flowname: str = ""
) -> None:
    
    # load strategy and copy timestamp to working dir
    strategy = load_and_copy_strategy(strategy_path, working_dir)

    # determine target from strategy file, in the caracal option
    target = strategy['targetfield']

    print(f"Starting pipeline for {target=}")

    # when testing without prefect
    # process_science_fields(
    #     strategy=strategy,
    #     working_dir=working_dir
    # )

    process_science_fields.with_options(
        name=f"MeerKAT pipeline - {target} {append_to_flowname}"
        # , task_runner=dask_task_runner
    )(
        strategy=strategy,
        working_dir=working_dir.resolve() # resolve in case relative path
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="MeerKAT L-band data processing")

    parser.add_argument(
        "--cli-config-file", type=str, help="Path to strategy configuration file"
    )
    parser.add_argument(
        "--working-dir", type=str, default="./", help="Path to main working directory. Default ./"
    )
    parser.add_argument(
        "--append-to-flowname", type=str, default="", help="String to attach to the flow name. Default ''"
    )

    return parser


def cli() -> None:

    parser = get_parser()

    args = parser.parse_args()

    setup_run(
        strategy_path=Path(args.cli_config_file),
        working_dir=Path(args.working_dir),
        append_to_flowname=args.append_to_flowname
    )


if __name__ == "__main__":
    cli()