from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal._caracal import CrossCalOptions, obtain_by_intent
from meerkatpolpipeline.check_calibrator.check_calibrator import split_calibrator
from meerkatpolpipeline.sclient import singularity_wrapper
from meerkatpolpipeline.utils.utils import (
    find_calibrated_ms,
)


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
        crosscal_options: CrossCalOptions,
        preprocessed_ms: Path,
        crosscal_dir: Path,
        ms_summary: dict,
        casa_container: Path,
        lofar_container: Path,
        bind_dirs: list[Path] = [],
    ) -> Path:
    """Run the CASA cross-calibration step."""
    logger = get_run_logger()

    assert "casacrosscal" in crosscal_dir.name, "crosscal_dir must contain 'casacrosscal' in its name"
      
    # Check if casacrosscal was already done by a previous run
    print("TODO: check if casacrosscal already done by a previous run. Should look for target MS")
    # calibrated_ms = find_calibrated_ms(crosscal_dir.parent, preprocessed_ms, look_in_subdirs=['casacrosscal'])
    # if calibrated_ms is not None:
    #     logger.info(f"Casa cross-calibration already done, found calibrated MS at {calibrated_ms}. Skipping caracal step.")
    #     return calibrated_ms

    if False:
        pass

    else:
        
        logger.info(f"Starting casa crosscal in {crosscal_dir} with options: {crosscal_options}")

        logger.info(f"Splitting off calibrator and target into separate MS in {crosscal_dir}")


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

        # Split calibrator
        cal_ms = split_calibrator(
            cal_ms_path=preprocessed_ms,
            cal_field=','.join(calibrators), # casa stringlist
            casa_container=casa_container,
            output_ms=crosscal_dir / (preprocessed_ms.stem + "-cal.ms"),
            bind_dirs=bind_dirs,
            chanbin=1,
            datacolumn="DATA"
        )

        _, targetfield = obtain_by_intent(ms_summary['field_intents'], 'target')

        # Split target
        target_ms = split_calibrator(
            cal_ms_path=preprocessed_ms,
            cal_field=targetfield,
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
        binding_dir = f"{cal_ms.parent},{aoflagger_strategy.parent}"


        # initial flagging on raw data
        cal_fields = ','.join(calibrators)  # casa stringlist

        cmd_aoflagger = f"""aoflagger -strategy {aoflagger_strategy} \
            --fields {cal_fields} \
            {cal_ms}
        """


        run_aoflagger(
            cmd_aoflagger=cmd_aoflagger,
            container=lofar_container,
            bind_dirs=[binding_dir],
            options=["--pwd", str(crosscal_dir)]  # execute command in crosscal/casacrosscal workdir
        )

        # run casa crosscal script
        cmd_casa = f"""casa -c {casa_script} \
            --calms {cal_ms} \
            --targetms {target_ms} \
            --fcal {fcal} \
            --bpcal {bpcal} \
            --gcal {gcal} \
            --xcal {xcal} \
            --leakcal {fcal} \
            --aoflagger_strategy {aoflagger_strategy} \
            --flocs_simg {lofar_container}
        """

        # run the casa crosscal script
        # all arguments should be given as kwargs to not confuse singularity wrapper
        run_casa_script(
            cmd_casa=cmd_casa,
            container=casa_container,
            bind_dirs=bind_dirs,
            options = ["--pwd", str(crosscal_dir)] # execute command in crosscal/casacrosscal workdir
        )

        # if completed succesfully, rename the target MS to have suffix {strategy['targetfield']}-corr.ms
        calibrated_target_ms = target_ms.stem + f"-{crosscal_options.targetfield}-corr.ms"
        target_ms.rename(crosscal_dir / calibrated_target_ms)
        
        logger.info(f"Casa crosscal completed, target MS saved at {crosscal_dir / calibrated_target_ms}")

        return calibrated_target_ms

# to prepare target and calibrators for casa crosscal,
# see /net/rijn9/data2/osinga/meerkatBfields/Abell754/test_annalisa_script/copy_and_split_ms.py