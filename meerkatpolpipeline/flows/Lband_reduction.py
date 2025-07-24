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
from meerkatpolpipeline.download.clipping import copy_and_clip_ms
from meerkatpolpipeline.download.download import download_and_extract
from meerkatpolpipeline.measurementset import msoverview_summary
from meerkatpolpipeline.sclient import run_singularity_command

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
        task_start_download = task(download_and_extract, name="download_and_extract")
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
    if "caracal" in enabled_operations and "casacrosscal" in enabled_operations:
        logger.warning("Both caracal and casacrosscal are enabled. This is not supported, please choose one of them.")
        raise ValueError("Both caracal and casacrosscal are enabled. This is not supported, please choose one of them.")
    
    if "caracal" in enabled_operations:
        caracal_options = get_options_from_strategy(strategy, operation="caracal")

        if caracal_options['msdir'] is None:
            logger.info(f"Caracal msdir is not set. Will run caracal in {working_dir / 'caracal'}")
            caracal_options['msdir'] = working_dir / "caracal"
        if caracal_options['dataid'] is None:
            logger.info(f"Caracal dataid is not set. Will assume ms name from download+preprocess step: {preprocessed_ms.name}")
            caracal_options['dataid'] = preprocessed_ms.stem # use stem to avoid .ms extension

        caracal_workdir = caracal_options['msdir'] # caracal will always work in the msdir directory
        caracal_workdir.mkdir(exist_ok=True) # runs can be repeated
        
        # symlink the preprocessed MS to the caracal workdir
        preprocessed_ms_symlink = caracal_workdir / preprocessed_ms.name
        if not preprocessed_ms_symlink.exists():
            logger.info(f"Creating symlink for preprocessed MS at {preprocessed_ms_symlink}")
            preprocessed_ms_symlink.symlink_to(preprocessed_ms)
        else:
            logger.info(f"Symlink for preprocessed MS already exists at {preprocessed_ms_symlink}, skipping symlink creation.")
        

        # get MS summary, optionally with scan intents if user wants auto determined calibrators
        task_msoverview_summary = task(msoverview_summary, name="msoverview_summary")
        ms_summary = task_msoverview_summary(
            binds=[str(preprocessed_ms_symlink.parent)],
            container=lofar_container,
            ms=preprocessed_ms_symlink,
            output_to_file= caracal_workdir / "msoverview_summary.txt",
            get_intents=caracal_options["auto_determine_obsconf"]
        )
        logger.info(f"Starting caracal with {preprocessed_ms_symlink} in {caracal_workdir}")
        logger.info(f"{ms_summary=}")

        # start caracal
        _caracal.start_caracal(caracal_options, working_dir=caracal_workdir, ms_summary=ms_summary)
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