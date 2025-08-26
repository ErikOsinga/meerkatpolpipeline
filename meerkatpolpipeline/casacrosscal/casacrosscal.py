from __future__ import annotations

from pathlib import Path

from casacore.tables import table
from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal._caracal import CrossCalOptions, obtain_by_intent
from meerkatpolpipeline.check_calibrator.check_calibrator import split_calibrator
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


    logger.info(f"Splitting off calibrator and target into separate MS in {crosscal_dir}")

    # Split calibrator, will be skipped if already exists
    cal_ms = split_calibrator(
        cal_ms_path=preprocessed_ms,
        cal_field=','.join(calibrators), # casa stringlist
        casa_container=casa_container,
        output_ms=crosscal_dir / (preprocessed_ms.stem + "-cal.ms"),
        bind_dirs=bind_dirs,
        chanbin=1,
        datacolumn="DATA"
    )

    targetfield = crosscal_options['targetfield']

    # Split target, will be skipped if already exists
    target_ms = split_calibrator(
        cal_ms_path=preprocessed_ms,
        cal_field=targetfield,
        casa_container=casa_container,
        output_ms=crosscal_dir / (preprocessed_ms.stem + "-target-corr.ms"),
        bind_dirs=bind_dirs,
        chanbin=1,
        datacolumn="DATA"
    )


    # Check if the target MS is already calibrated
    tb = table(str(target_ms), readonly=True)
    if 'CORRECTED_DATA' in tb.colnames():
        tb.close()
        logger.info(f"Target MS {target_ms} already contains CORRECTED_DATA column, assuming it is already calibrated. Skipping crosscal step.")
        return cal_ms, target_ms
    
    else:
        tb.close()
    
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

        # run casa crosscal script
        cmd_casa = f"""casa --nologger --nogui -c '{casa_script}' \
            --calms {cal_ms} \
            --targetms {target_ms} \
            --fcal {fcal} \
            --bpcal {bpcal} \
            --gcal {gcal} \
            --xcal {xcal} \
            --leakcal {fcal} \
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
       
        logger.info(f"Casa crosscal completed, target MS saved at {target_ms}")

        return cal_ms, target_ms