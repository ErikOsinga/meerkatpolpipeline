from __future__ import annotations

import shutil
from pathlib import Path

from casacore.tables import table
from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal._caracal import CrossCalOptions, obtain_by_intent
from meerkatpolpipeline.check_calibrator.check_calibrator import split_field
from meerkatpolpipeline.sclient import singularity_wrapper


def obtain_all_calibrators(ms_summary: dict) -> list[str]:
    """
    Obtain unique list of all calibrators from ms_summary
    """

    fcal_id, fcal = obtain_by_intent(ms_summary['field_intents'], 'fluxcal')
    bpcal_id, bpcal = obtain_by_intent(ms_summary['field_intents'], 'bpcal')
    gcal_id, gcal = obtain_by_intent(ms_summary['field_intents'], 'gaincal')
    xcal_id, xcal = obtain_by_intent(ms_summary['field_intents'], 'polcal')

    calibrators = list(set([fcal, bpcal, gcal, xcal]))

    return calibrators, fcal, bpcal, gcal, xcal


@singularity_wrapper
def run_casa_script(cmd_casa: str, **kwargs) -> str:
    """Run the casa crosscal script using singularity wrapper

    Note that all arguments should be given as kwargs to not confuse singularity wrapper

    Args:
        cmd_casa: casa script command
        **kwargs: Additional keyword arguments that will be passed to the singularity_wrapper

    Returns:
        str: the command that was executed
    """
    logger = get_run_logger()

    logger.info(f"casa crosscal script command {cmd_casa}")

    return cmd_casa


@singularity_wrapper
def run_aoflagger(cmd_aoflagger: str, **kwargs) -> str:
    """Run an aoflagger command using singularity wrapper

    Args:
        cmd_aoflagger: aoflagger command to run
        **kwargs: Additional keyword arguments that will be passed to the singularity_wrapper

    Returns:
        str: the command that was executed
    """
    logger = get_run_logger()

    logger.info(f"aoflagger command {cmd_aoflagger}")

    return cmd_aoflagger


def do_casa_crosscal(
        crosscal_options: CrossCalOptions | dict,
        preprocessed_ms: Path,
        crosscal_dir: Path,
        ms_summary: dict,
        casa_container: Path,
        lofar_container: Path,
        bind_dirs: list[Path] = [],
    ) -> tuple[Path, Path]:
    """Run the CASA cross-calibration step.
    
    Args:
        crosscal_options: Options for the cross-calibration step
        preprocessed_ms: Path to the preprocessed measurement set
        crosscal_dir: Directory where the cross-calibrated measurement sets will be stored
        ms_summary: Summary of the measurement set, including field intents
        casa_container: Path to the CASA singularity container
        lofar_container: Path to the LOFAR singularity container (for aoflagger)
        bind_dirs: Directories to bind into the singularity containers
    Returns:
        Tuple[Path, Path]: Paths to the calibrator and target measurement sets
    
    """
    logger = get_run_logger()

    assert "casacrosscal" in crosscal_dir.name, "crosscal_dir must contain 'casacrosscal' in its name"

        
    logger.info(f"Starting casa crosscal in {crosscal_dir} with options: {crosscal_options}")


    # check if calibrated target MS already exists.
    if (crosscal_dir / (preprocessed_ms.stem + "-target-corr.ms")).exists():
        logger.info(f"Calibrated target MS {crosscal_dir / (preprocessed_ms.stem + '-target-corr.ms')} already exists. Skipping casa crosscal step.")
        cal_ms = crosscal_dir / (preprocessed_ms.stem + "-cal.ms")
        calibrated_target_ms = crosscal_dir / (preprocessed_ms.stem + "-target-corr.ms")
        return cal_ms, calibrated_target_ms

    # otherwise, continue with casa crosscal step. First, determine calibrators
    if not crosscal_options['auto_determine_obsconf']:
        logger.info("Using user-supplied calibrators for caracal reduction")

        fcal = crosscal_options['obsconf_fcal']
        bpcal = crosscal_options['obsconf_bpcal']
        gcal = crosscal_options['obsconf_gcal']
        xcal = crosscal_options['obsconf_xcal']
        calibrators = list(set([fcal, bpcal, gcal, xcal]))
    else:
        logger.info("Auto-determining calibrators from MS field intents for caracal reduction")
        calibrators, fcal, bpcal, gcal, xcal = obtain_all_calibrators(ms_summary)

    if len(calibrators) == 0:
        raise ValueError("No calibrators found in ms_summary. Please check your input data.")

    logger.info(f"Using calibrators: {calibrators}, with fcal: {fcal}, bpcal: {bpcal}, gcal: {gcal}, xcal: {xcal}")

    if crosscal_options['cal_ms_for_casa'] is not None:
        logger.info("Using user-supplied calibrator MS for caracal reduction")
        cal_ms = crosscal_options['cal_ms_for_casa']
        if not cal_ms.exists():
            raise FileNotFoundError(f"User-supplied calibrator MS {cal_ms} does not exist.")

    else:
        logger.info(f"Splitting off calibrator and target into separate MS in {crosscal_dir}")

        # Split calibrator, will be skipped if already exists
        cal_ms = split_field(
            ms_path=preprocessed_ms,
            field=','.join(calibrators), # casa stringlist
            casa_container=casa_container,
            output_ms=crosscal_dir / (preprocessed_ms.stem + "-cal.ms"),
            bind_dirs=bind_dirs,
            chanbin=1,
            datacolumn="DATA"
        )

    targetfield = crosscal_options['targetfield']

    # Split target, will be skipped if already exists
    target_ms = split_field(
        ms_path=preprocessed_ms,
        field=targetfield,
        casa_container=casa_container,
        output_ms=crosscal_dir / (preprocessed_ms.stem + "-target.ms"),
        bind_dirs=bind_dirs,
        chanbin=1,
        datacolumn="DATA"
    )

    logger.info(f"Running aoflagger on calibrator measurementset {cal_ms}")

    from meerkatpolpipeline.casacrosscal import (
        cal_J0408,  # cant import casa scripts
    )
    casa_script = Path(cal_J0408.__file__).parent / "casa_script_crosscal.py"
    aoflagger_strategy = Path(cal_J0408.__file__).parent / "default_StokesQUV.lua"

    # cant run aoflagger inside the casa script because it requires a different container
    binding_dir_aof = [cal_ms.parent, aoflagger_strategy.parent, crosscal_dir]

    # initial flagging on raw data
    cal_fields = ','.join(calibrators)  # casa stringlist

    cmd_aoflagger = f"""aoflagger -strategy {aoflagger_strategy} \
        --fields {cal_fields} \
        {cal_ms}
    """

    logger.info("Running aoflagger")

    run_aoflagger(
        cmd_aoflagger=cmd_aoflagger,
        container=lofar_container,
        bind_dirs=binding_dir_aof,
        options=["--pwd", str(crosscal_dir)]  # execute command in crosscal/casacrosscal workdir
    )

    leakcal = fcal  # use fcal for leakcal
    logger.info(f"Defaulting to fcal as leakage calibrator: {fcal}")
    
    # TODO: consider if we want to use only one leakage calibrator if multiple are present?
    # if ',' in leakcal:
    #     leakcal = leakcal.split(',')[0]
    #     logger.info(f"Found two input flux calibrators {fcal=}. Using only the first one as leakage cal: {leakcal}")

    # run casa crosscal script
    cmd_casa = f"""casa --nologger --nogui -c {casa_script} \
        --calms {cal_ms} \
        --targetms {target_ms} \
        --fcal {fcal} \
        --bpcal {bpcal} \
        --gcal {gcal} \
        --xcal {xcal} \
        --leakcal {leakcal} \
    """

    logger.info("Running casa crosscal script.")

    # run the casa crosscal script
    # all arguments should be given as kwargs to not confuse singularity wrapper
    run_casa_script(
        cmd_casa=cmd_casa,
        container=casa_container,
        bind_dirs=bind_dirs+[casa_script.parent],
        options = ["--pwd", str(crosscal_dir)] # execute command in crosscal/casacrosscal workdir
    )
    
    logger.info(f"Copying CORRECTED_DATA to DATA column and averaging to {crosscal_options['timebin_casa']} bin width in time and averaging a factor {crosscal_options['freqbin_casa']} in freq.")
    
    # split + average corrected target to a new MS.
    calibrated_target_ms = split_field(
        ms_path=target_ms,
        field=targetfield,
        casa_container=casa_container,
        output_ms=crosscal_dir / (preprocessed_ms.stem + "-target-corr.ms"),
        bind_dirs=bind_dirs,
        chanbin=crosscal_options['freqbin_casa'],
        timebin=crosscal_options['timebin_casa'],
        datacolumn="corrected"
    )

    # remove the uncalibrated target MS to save space
    if target_ms.exists():
        logger.info(f"Removing un-averaged target MS {target_ms} to save space.")
        table(str(target_ms)).unlock()
        shutil.rmtree(target_ms)
    else:
        raise FileNotFoundError(f"Expected (un)calibrated target MS {target_ms} to exist, but it does not.")
    
    logger.info(f"Casa crosscal completed, calibrated target MS saved at {calibrated_target_ms}")

    return cal_ms, calibrated_target_ms