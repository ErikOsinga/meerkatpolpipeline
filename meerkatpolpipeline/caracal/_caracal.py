from __future__ import annotations

import csv
import glob
import shutil
from pathlib import Path

from casacore.tables import table
from prefect.logging import get_run_logger

from meerkatpolpipeline.measurementset import msoverview_summary
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import (
    add_timestamp_to_path,
    # check_create_symlink,
    execute_command,
    find_calibrated_ms,
)
from meerkatpolpipeline.utils.yaml import yaml

CARACAL_SINGULARITIES = {
    "stimela_aoflagger_1.2.0.sif",
    "stimela_ragavi_1.7.3.sif",
    "stimela_meqtrees_1.7.2.sif",
    "stimela_msutils_1.6.9.sif",
    "stimela_owlcat_1.6.6.sif",
    "stimela_msutils_1.4.6.sif",
    "stimela_casa_1.7.1.sif"
}

class CrossCalOptions(BaseOptions):
    """A basic class to handle caracal options for meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Required parameter"""
    which: str
    """which cross-calibration to use, either 'caracal' or 'casacrosscal'. Required parameter"""
    targetfield: str | None = None
    """name of targetfield"""
    caracal_template_strategy: Path | None = None
    """path to the template polcal-strategy.yml for caracal. A copy will be made with updated parameters"""
    msdir: Path | None = None
    """Path to the directory where uncalibrated MS is stored"""
    dataid: str | None = None
    """Name of the measurement set without .ms extension"""
    prefix: str = "caracalpipelinerun"
    """prefix"""
    backend: str = "singularity"
    """backend for caracal, either singularity or docker"""
    caracal_files: str | None = None
    """where input files for caracal are stored, e.g. flagging strategy"""
    auto_determine_obsconf: bool = False
    """Automatically determine which calibrators are which and which is the target? Default False"""
    obsconf_target: str | None = None
    """Target"""
    obsconf_gcal: str | None = None
    """Gaincal"""
    obsconf_xcal: str | None = None
    """Pol cal"""
    obsconf_bpcal: str | None = None
    """BP calibrator, should be unpolarised"""
    obsconf_fcal: str | None = None
    """Flux calibrator, should be unpolarised"""
    obsconf_refant: str = "m024"
    """reference antenna"""
    test: bool = False
    """create the caracal command but dont run it, for testing purposes only"""

class CaracalConfigFile(BaseOptions):
    """A class to hold values for the caracal polcal.yaml file, see the template caracal file
    
    Can be extended to many more parameters
    """

    prefix: str 
    """e.g. caracalpipelinerun"""
    msdir: Path
    """e.g. /data2/osinga/meerkatBfields/Abell754/caracal_reduction_2023-03-25/"""
    input: Path 
    """e.g. /data2/osinga/meerkatBfields/caracal_files/"""
    dataid: str
    """e.g. ['Abell754_default_uncalibrated-2023-03-25'] """
    target: str
    """e.g. A754"""
    fcal: str
    """e.g. J0408-6545 """
    bpcal: str
    """e.g. J0408-6545"""
    gcal: str
    """e.g. J1008+0730"""
    xcal: str
    """e.g. J1331+3030"""
    refant: str
    """e.g. m024"""

def obtain_by_intent(mapping: dict, intent: str) -> tuple[int, str]:
    """return field id (int) and field_name (str) from a mapping dict
    
    for either:

        intent = 'target'
        intent = 'polcal'
        intent = 'bpcal'
        intent = 'fluxcal'
        intent = 'gaincal'

    """
    intent_to_map = {
        "target": "TARGET",
        "polcal": "CALIBRATE_POL",
        "bpcal": "CALIBRATE_BANDPASS",
        "fluxcal": "CALIBRATE_FLUX",
        "gaincal": "CALIBRATE_PHASE",
    }

    if intent not in intent_to_map.keys():
        raise ValueError(f"{intent=} This is not one of {intent_to_map.keys()}")

    for fid, value in mapping.items():
        fieldname, intents = value
        if intent_to_map[intent] in intents:
            return fid, fieldname
    raise ValueError(f"Did not find {intent=} automatically. Looked in {mapping}.")

def determine_calibrators(caracal_options: CrossCalOptions, ms_summary: dict) -> CrossCalOptions:
    """Determine calibrators automatically or from user input"""

    logger = get_run_logger()

    if caracal_options["auto_determine_obsconf"]:
        logger.info("Automatically determinining calibrators from MS...")

        if ms_summary is None:
            raise ValueError("ms_summary must be provided if 'auto_determine_obsconf' is True")
        
        fcal_id, fcal = obtain_by_intent(ms_summary['field_intents'], 'fluxcal')
        bpcal_id, bpcal = obtain_by_intent(ms_summary['field_intents'], 'bpcal')
        gcal_id, gcal = obtain_by_intent(ms_summary['field_intents'], 'gaincal')
        xcal_id, xcal = obtain_by_intent(ms_summary['field_intents'], 'polcal')
        
        # update the caracal options with the automatically determined calibrators
        update_caracal_options = {
            "obsconf_fcal": fcal,
            "obsconf_bpcal": bpcal,
            "obsconf_gcal": gcal,
            "obsconf_xcal": xcal
        }

        logger.info("Automatically determined the following calibrators, please verify")
        logger.info(update_caracal_options)

        caracal_options.update(**update_caracal_options)

    else:
        logger.info("Using user-supplied calibrators for caracal reduction")
        logger.info(f"{caracal_options['obsconf_fcal']=}")
        logger.info(f"{caracal_options['obsconf_bpcal']=}")
        logger.info(f"{caracal_options['obsconf_gcal']=}")
        logger.info(f"{caracal_options['obsconf_xcal']=}")

        if caracal_options['obsconf_fcal'] is None:
            raise ValueError("obsconf_fcal must be set if 'auto_determine_obsconf' is False")
        if caracal_options['obsconf_bpcal'] is None:
            raise ValueError("obsconf_bpcal must be set if 'auto_determine_obsconf' is False")
        if caracal_options['obsconf_gcal'] is None:
            raise ValueError("obsconf_gcal must be set if 'auto_determine_obsconf' is False")
        if caracal_options['obsconf_xcal'] is None:
            raise ValueError("obsconf_xcal must be set if 'auto_determine_obsconf' is False")

    return caracal_options


def get_field_ids(ms_path: str, fieldnames: list[str]) -> dict[str, int]:
    """
    Retrieve unique field IDs for a list of fieldnames from a Measurement Set.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set.
    fieldnames : list of str
        List of fieldnames to search for.

    Returns
    -------
    dict
        Dictionary mapping each fieldname to its unique field ID.

    Raises
    ------
    ValueError
        If any fieldname is not found or appears multiple times in the FIELD subtable.
    """
    ms_path = str(ms_path)
    
    field_table_path = ms_path.rstrip("/") + "/FIELD"
    with table(field_table_path) as t:
        names = t.getcol("NAME")

    result = {}
    for fname_str in fieldnames:
        # fieldname can be a casa stringlist, split it
        for fname in fname_str.split(","):
            matches = [i for i, n in enumerate(names) if n == fname]
            if len(matches) == 0:
                raise ValueError(f"Fieldname '{fname}' not found in {ms_path}.")
            if len(matches) > 1:
                raise ValueError(f"Fieldname '{fname}' is not unique in {ms_path} (found IDs {matches}).")
            result[fname] = matches[0]

    return result


def write_crosscal_csv(
    crosscal_options: dict[str, str | None],
    ms: Path,
    output_path: str
) -> Path:
    """
    Write a CSV file with columns: field_id, field_name, intent_string,
    based on the entries in crosscal_options.

    This function is used when 'auto_determine_obsconf' is False,
    and the user has supplied the calibrators manually.
    The output CSV will contain the calibrators and their intents

    The MS is required to get the field IDs.

    Only options with non‑None values are written.
    """
    # map each option key to its intent string
    intent_map: dict[str, str] = {
        "obsconf_target":    "TARGET",
        "obsconf_xcal":      "CALIBRATE_POL",
        "obsconf_bpcal":     "CALIBRATE_BANDPASS",
        "obsconf_fcal":      "CALIBRATE_FLUX",
        "obsconf_gcal":      "CALIBRATE_PHASE",
    }

    all_fieldnames = {} # e.g. obsconf_target: "Coma1", obsconf_xcal: "J0408+6545" etc
    for key, intent in intent_map.items():
        fieldname = crosscal_options.get(key)
        if fieldname is not None:
            all_fieldnames[key] = fieldname

    # retrieve field IDs
    ids_dict = get_field_ids(ms, all_fieldnames.values())

    with open(output_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["field_id", "field_name", "intent_string"]
        )
        writer.writeheader()

        for key, fieldname in all_fieldnames.items():
            intent = intent_map.get(key)
            if intent is None:
                raise ValueError(f"Intent for key '{key}' not found in intent_map. Please check the crosscal_options.")
            
            for fname in fieldname.split(','): # fieldname can be a casa stringlist
                # get the field ID for this fieldname
                field_id = ids_dict.get(fname)

                if field_id is None:
                    raise ValueError(f"Field name '{fname}' not found in the MS '{ms}'. Please check the field names.")
                
                # write each field ID with the corresponding field name and intent
                writer.writerow({
                    "field_id": field_id,  
                    "field_name": fname,
                    "intent_string": intent
                })

    return output_path

def _update_caracal_template_with_options(caracal_template: dict, caracal_config_file_options: CaracalConfigFile) -> dict:
    """Update the caracal template dict with the user-supplied caracal config options"""
        
    caracal_template.update(dict(caracal_config_file_options))

    return caracal_template

def write_and_timestamp_caracal_strategy(output_yaml: Path, caracal_options: dict) -> Path:
    """Write the updated yaml dict to a timestamped file 
    
    Args:
        output_yaml (Path): Output file path options will be written to (.yaml)
        caracal_options (dict): The caracal options

    Returns:
        Path: Copied and timestamped file path
    """
    logger = get_run_logger()

    with open(output_yaml, 'w') as out_file:
        # ruamel.yaml should preserve the order of the keys
        yaml.dump(caracal_options, out_file)
        # cant save dump with Path objects, even if we add representer

    output_dir = output_yaml.parent

    stamped_caracal_strategy = (
        output_dir / add_timestamp_to_path(input_path=output_yaml).name
    )
    logger.info(f"Copying {output_yaml.absolute()} to {stamped_caracal_strategy}")
    shutil.copyfile(output_yaml.absolute(), stamped_caracal_strategy)

    return Path(stamped_caracal_strategy)

def edit_caracal_template(caracal_options: CrossCalOptions, working_dir: Path, meerkat_band: str) -> Path:
    """Take the base template for a caracal strategy and update MS path, calibrators etc"""

    # load the template yaml
    caracal_template = caracal_options["caracal_template_strategy"]

    with open(caracal_template) as in_file:
        caracal_template_yaml = yaml.load(in_file) # dict

    # map the input user options to the caracal names
    caracal_config_options = {
        "general": {
           "prefix": caracal_options['prefix'],
           "msdir": caracal_options['msdir'],
           "input": caracal_options['caracal_files'],
        },
        "getdata": {
            # caracal requires a list
            "dataid": [caracal_options['dataid']],

        },
        "obsconf": {
            # caracal requires lists
            "target": [caracal_options['targetfield']],
            # TODO: extend to multiple calibrators have it always be a list
            "fcal": [caracal_options['obsconf_fcal']],
            "bpcal": [caracal_options['obsconf_bpcal']],
            "gcal": [caracal_options['obsconf_gcal']],
            "xcal": [caracal_options['obsconf_xcal']],
            "refant": caracal_options['obsconf_refant'],
        },
        "crosscal":  caracal_template_yaml['crosscal']
    }

    # make sure to update meerkat_band in the ['crosscal']['set_model'] options
    # very important as the 'meerkat_band' parameter defaults to 'L'
    caracal_config_options['crosscal']['set_model']['meerkat_band'] = meerkat_band

    # # put them in the class holder for a caracal config file
    # caracal_config_file_options = CaracalConfigFile(**caracal_config_options)
    ## TODO: update class holder or write function to check validity of arguments
    caracal_config_file_options = caracal_config_options

    # update the template yaml with the user options
    final_caracal_options = _update_caracal_template_with_options(caracal_template_yaml, caracal_config_file_options)

    # save the yaml file with timestamp
    output_yaml_path = working_dir/ "caracal_polcal.latest.yaml"
    final_caracal_yaml_path = write_and_timestamp_caracal_strategy(output_yaml_path, final_caracal_options)

    return final_caracal_yaml_path


def cleanup_caracal_run(
        caracal_rundir: Path,
        preprocessed_ms_name: str,
        calibrated_cal_ms: Path,
        calibrated_target_ms: Path,
        output_dir: Path,
        cleanup_sing: bool = True
    ) -> tuple[Path, Path]:
    """
    Move outputs from a caracal run to another directory.

    Useful function because caracal always runs in the location of the measurement set
    which is in the /download/ directory to allow re-running the pipeline easily.
    But it's better to have the outputs in the /crosscal/carcal directory.

    Args:
        caracal_rundir (Path): Directory where the caracal run was executed.
        preprocessed_ms_name (str): name (stem) of the preprocessed measurement set, used to find various output files
        calibrated_cal_ms (Path): Path to the calibrated MS with calibrators.
        calibrated_target_ms (Path): Path to the calibrated MS with target.
        output_dir (Path): Directory to move the outputs to.
        cleanup_sing (bool): If True, will remove the caracal (stimela) singularity images from the caracal_rundir
    Returns:
        tuple[Path, Path]: new Paths to the caracal calibrated MSes with calibrators and target respectively
    """
    logger = get_run_logger()

    if caracal_rundir == output_dir:
        logger.info(f"caracal_rundir {caracal_rundir} is the same as output_dir {output_dir}. Nothing to move.")
        return calibrated_cal_ms, calibrated_target_ms

    # move the caracal run directory to the output directory
    if not output_dir.exists():
        logger.info(f"Creating output directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Attempting to move caracal run from {caracal_rundir} to {output_dir}")

    # check if all files are present
    n_errors = 0
    files_to_move = []
    
    # most caracal output is in the /output/ directory
    caracal_output = caracal_rundir / "output"
    if not caracal_output.exists():
        msg = f"Caracal output directory {caracal_output} does not exist, nothing to move."
        logger.error(msg)
        n_errors += 1
    else:
        files_to_move.append(caracal_output)

    # grab various .json and .txt files from the caracal rundir
    caracal_txtfiles = glob.glob(
        str(caracal_rundir / f"{preprocessed_ms_name}*.txt")
    )
    if len(caracal_txtfiles) == 0:
        msg = f"No .txt files found in {caracal_rundir} with name {preprocessed_ms_name}*.txt"
        logger.error(msg)
        n_errors += 1
    else:
        files_to_move.extend([Path (c) for c in caracal_txtfiles])

    # usually there are 3 txt and 3 json files: one for input ms, and one for calibrated target and calibrators ms.
    caracal_json = glob.glob(
        str(caracal_rundir / f"{preprocessed_ms_name}*.json")
    )
    if len(caracal_json) == 0:
        msg = f"No .json files found in {caracal_rundir} with name {preprocessed_ms_name}*.json"
        logger.error(msg)
        n_errors += 1
    else:
        files_to_move.extend([Path (c) for c in caracal_json])
    
    # grab the elevation plot
    caracal_elevation = glob.glob(
        str(caracal_rundir / f"{preprocessed_ms_name}*elevation-tracks.png")
    )
    if len(caracal_elevation) == 0:
        msg = f"No elevation plot found in {caracal_rundir} with name {preprocessed_ms_name}*elevation-tracks.png"
        logger.error(msg)
        n_errors += 1
    else:
        files_to_move.extend([Path (c) for c in caracal_elevation])

    # grab the calibrated MSes and their flagversions
    if calibrated_cal_ms.parent != caracal_rundir:
        msg = f"calibrated_cal_ms must be in the caracal run directory: {caracal_rundir}. Instead found it at {calibrated_cal_ms}"
        logger.error(msg)
        n_errors += 1
    else:
        files_to_move.append(calibrated_cal_ms)
        files_to_move.append(calibrated_cal_ms.with_suffix(".ms.flagversions"))
    
    if calibrated_target_ms.parent != caracal_rundir:
        msg = f"calibrated_target_ms must be in the caracal run directory: {caracal_rundir}. Instead found it at {calibrated_target_ms}"
        logger.error(msg)
        n_errors += 1
    else:
        files_to_move.append(calibrated_target_ms)
        files_to_move.append(calibrated_target_ms.with_suffix(".ms.flagversions"))
    
    # grab caracal config file
    caracal_config_file = caracal_rundir / "caracal_polcal.latest.yaml"
    if not caracal_config_file.exists():
        msg = f"Caracal config file {caracal_config_file} does not exist, nothing to move."
        logger.error(msg)
        n_errors += 1
    else:
        # including any timestamped config file
        caracal_configfiles = glob.glob(str(caracal_config_file.parent / caracal_config_file.stem) + "*yaml")
        files_to_move.extend([Path(c) for c in caracal_configfiles])
    
    if n_errors > 0:
        raise ValueError(f"{n_errors} Errors encountered while checking caracal run directory {caracal_rundir}. Please see logs")

    else:
        logger.info(f"Moving caracal run from {caracal_rundir} to {output_dir}")
        # move the files to the output directory
        for file in files_to_move:
            if not file.exists():
                msg = f"File {file} does not exist. This should not happen"
                logger.error(msg)
                raise ValueError(msg)
            logger.debug(f"Moving file {file} to {output_dir} / {file.name}")
            file.rename(output_dir / file.name)


    if cleanup_sing:
        logger.info(f"Cleaning up caracal run directory {caracal_rundir} by removing singularity images")
        # remove the singularity images
        for singularity in CARACAL_SINGULARITIES:
            # currently hardcoded stimela versions, probably need to udate this
            singularity_path = caracal_rundir / singularity
            if singularity_path.exists():
                logger.info(f"Removing singularity image {singularity_path}")
                singularity_path.unlink()
            else:
                logger.info(f"Singularity image {singularity_path} does not exist, skipping removal.")

    # return the new paths to the calibrated MSes
    cal_ms = output_dir / calibrated_cal_ms.name
    target_ms = output_dir / calibrated_target_ms.name

    return cal_ms, target_ms


def start_caracal(
        caracal_options: CrossCalOptions,
        working_dir: Path,
        ms_summary: dict | None = None,
    ) -> Path:
    """
    Start a caracal reduction run using the caracal options and working directory.
    Args:
        caracal_options (CrossCalOptions): Options for the caracal run.
        working_dir (Path): Directory to work in.
        ms_summary (dict | None): Summary of the measurement set, if available.
                                  Required if caracal_options['auto_determine_obsconf'] is True.
        test (bool): for testing mode, if True, will not execute the command, but only log it.
    Returns:
        Path: Path where the run was executed
    """
    logger = get_run_logger()

    # determine calibrators from MS summary (automatically) or from user input
    caracal_options = determine_calibrators(caracal_options, ms_summary)
    
    # write the caracal config file with user options
    caracal_config_file = edit_caracal_template(caracal_options, working_dir, ms_summary['meerkat_band'])

    with open(working_dir / "go_caracal.sh", "w") as file:
        # TODO: let users supply conda env command?
        # conda activate caracalfork
        file.write("source /net/lofar4/data2/osinga/software/miniconda/installation/bin/activate caracalfork\n")

        # Makes sure cache is not saved in homedir (no space)
        file.write(f"export APPTAINER_CACHEDIR={working_dir}/temporary-apptainer-cache")
        file.write("\n")
        file.write("\n")
        # caracal always runs in the current working dir
        file.write(f"cd {working_dir}\n")
        file.write(f"caracal -ct singularity -c {caracal_config_file}")
   
    logger.info("Starting caracal")
    # grab logs by doing execute_command
    cmd = f"bash {working_dir / 'go_caracal.sh'}"
    execute_command(cmd, test=caracal_options['test'])

    return working_dir

def do_caracal_crosscal(
        crosscal_options: CrossCalOptions,
        preprocessed_ms: Path,
        crosscal_base_dir: Path,
        ms_summary: dict,
        lofar_container: Path | None = None
    ) -> tuple[Path,Path]:
    """Run the caracal cross-calibration step.
    
    lofar_container is only required if user overwrites the input MS to caracal
    in that case we need it to do msoverview on the MS to get the summary.

    Returns:
        tuple[Path, Path]: new Paths to the caracal calibrated MSes with calibrators and target respectively

    """
    logger = get_run_logger()

    # field_intents_csv = crosscal_base_dir / "field_intents.csv"

    # Check if caracal was already done by a previous run.
    calibrated_cal_ms = find_calibrated_ms(
        crosscal_base_dir,
        preprocessed_ms,
        look_in_subdirs=['caracal'],
        suffix="-cal.ms"
    )
    calibrated_target_ms = find_calibrated_ms(
        crosscal_base_dir,
        preprocessed_ms,
        look_in_subdirs=['caracal'],
        suffix=f"-{crosscal_options['targetfield']}-corr.ms"
    )

    if calibrated_cal_ms is not None and calibrated_target_ms is not None:
        logger.info(f"Caracal cross-calibration already done, found calibrated MSes at {calibrated_cal_ms} and {calibrated_target_ms}. Skipping caracal step.")
        return calibrated_cal_ms, calibrated_target_ms
    
    elif calibrated_cal_ms is not None or calibrated_target_ms is not None:
        raise ValueError("Found one of the calibrated MSes, but not both. This is not expected. "
                         "Please check the caracal run directory for errors. ")

    ########## The actual caracal step ###########
    else: 
        caracal_workdir = crosscal_base_dir / "caracal"
        caracal_workdir.mkdir(exist_ok=True) # runs can be repeated

        if crosscal_options['msdir'] is None:
            logger.info(f"Caracal msdir is not set. Will run caracal in {crosscal_base_dir / 'caracal'}")
            # crosscal_options['msdir'] = crosscal_base_dir / "caracal" # disabled because symlinking and caracal dont agree
            crosscal_options['msdir'] = preprocessed_ms.parent # caracal will always work in the msdir directory
        if crosscal_options['dataid'] is None:
            logger.info(f"Caracal dataid is not set. Will assume ms name from download+preprocess step: {preprocessed_ms.name}")

            ## symlink the preprocessed MS to the caracal workdir
            ## we do this because caracal writes all files to the parent of the MS

            # preprocessed_ms_symlink = caracal_workdir / preprocessed_ms.name
            # preprocessed_ms_symlink = check_create_symlink(preprocessed_ms_symlink, preprocessed_ms)

            ## symlinks dont work because caracal only binds the msdir
            ## so let's just run caracal in the download/ directory on the preprocessed MS
            ## and use the cleanup_caracal_run() to check if succesful and move all the output over to the caracal directory.
            
            preprocessed_ms_symlink = preprocessed_ms # no symlink

            crosscal_options['dataid'] = preprocessed_ms.stem # use stem to avoid .ms extension
        else:
            # user overwrites the dataid, probably also the 'msdir', but not strictly required.
            preprocessed_ms_symlink = crosscal_options['msdir'] / (crosscal_options['dataid'] + ".ms")
            logger.info(f"Using user-supplied dataid {crosscal_options['dataid']} for caracal, will assume preprocessed data is in {preprocessed_ms_symlink}")

            # in that case we have to recompute the ms summary
            ms_summary = msoverview_summary(
                binds=[str(preprocessed_ms_symlink.parent)],
                container=lofar_container,
                ms=preprocessed_ms_symlink,
                output_to_file= crosscal_base_dir / "msoverview_summary.txt",
                get_intents=crosscal_options["auto_determine_obsconf"]
            )


        caracal_workdir = crosscal_options['msdir'] # caracal will always work in the msdir directory
        caracal_workdir.mkdir(exist_ok=True) # runs can be repeated
        
        logger.info(f"Starting caracal with {preprocessed_ms_symlink} in {caracal_workdir}")

        # start caracal
        start_caracal(crosscal_options, working_dir=caracal_workdir, ms_summary=ms_summary)

        # Check if caracal has now completed succesfully
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
            suffix=f"-{crosscal_options['targetfield']}-corr.ms"
        )

        return calibrated_cal_ms, calibrated_target_ms