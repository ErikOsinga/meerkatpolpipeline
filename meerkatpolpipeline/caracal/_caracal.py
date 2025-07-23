from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.append('/home/osingae/OneDrive/postdoc/projects/MEERKAT_similarity_Bfields/meerkatpolpipeline')

from meerkatpolpipeline.logging import logger
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils import add_timestamp_to_path


class CaracalOptions(BaseOptions):
    """A basic class to handle caracal options for meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Default False"""
    targetfield: str | None = None
    """name of targetfield"""
    caracal_template_strategy: Path | None = None
    """path to the template polcal-strategy.yml for caracal. A copy will be made with updated parameters"""
    msdir: Path | None = None
    """Path to the directory where MS is stored"""
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

def auto_determine_calibrators(caracal_options: CaracalOptions) -> CaracalOptions:
    """Automatically determine calibrators"""

    logger.warning("TODO: automatically determine from MS. Currently hardcoded")
    
    fcal = 'J0408-6545'
    bpcal = 'J0408-6545'
    gcal = 'J1008+0730'
    xcal = 'J1331+3030'

    update_caracal_options = {
        "obsconf_fcal": fcal,
        "obsconf_bpcal": bpcal,
        "obsconf_gcal": gcal,
        "obsconf_xcal": xcal
    }

    logger.info("Automatically determined the following calibrators, please verify")
    logger.info(update_caracal_options)

    caracal_options.update(**update_caracal_options)

    return caracal_options

def _update_caracal_template_with_options(caracal_template: dict, caracal_config_file_options: CaracalConfigFile) -> dict:
    """Update the caracal template dict with the user-supplied caracal config options"""
    
    print("TODO: actually update")
    logger.warning("TODO: actually update")
    
    updated_caracal_template = caracal_template
    return updated_caracal_template

def write_and_timestamp_caracal_strategy(output_yaml: Path, caracal_options: dict) -> Path:
    """Write the updated yaml dict to a timestamped file 
    
    Args:
        output_yaml (Path): Output file path options will be written to (.yaml)
        caracal_options (dict): The caracal options

    Returns:
        Path: Copied and timestamped file path
    """

    print("TODO: write yaml to file with same ordering. Probably some regex instead.")
    with open(output_yaml, 'w') as out_file:
        yaml.dump(caracal_options, out_file)

    output_dir = output_yaml.parent

    stamped_caracal_strategy = (
        output_dir / add_timestamp_to_path(input_path=output_yaml).name
    )
    logger.info(f"Copying {output_yaml.absolute()} to {stamped_caracal_strategy}")
    shutil.copyfile(output_yaml.absolute(), stamped_caracal_strategy)

    return Path(stamped_caracal_strategy)

def edit_caracal_template(caracal_options: CaracalOptions, working_dir: Path) -> Path:
    """Take the base template for a caracal strategy and update MS path, calibrators etc"""

    # First check whether the user supplied calibrators or wants them auto determined
    if caracal_options["auto_determine_obsconf"]:
        caracal_options = auto_determine_calibrators(caracal_options)

    # map the input user options to the caracal names
    caracal_config_options = {
       "prefix": caracal_options['prefix'],
       "msdir": caracal_options['msdir'],
       "input": caracal_options['caracal_files'],
       "dataid": caracal_options['dataid'],
       "target": caracal_options['targetfield'],
       "fcal": caracal_options['obsconf_fcal'],
       "bpcal": caracal_options['obsconf_bpcal'],
       "gcal": caracal_options['obsconf_gcal'],
       "xcal": caracal_options['obsconf_xcal'],
       "refant": caracal_options['obsconf_refant'],
    }
    # put them in the class holder for a caracal config file
    caracal_config_file_options = CaracalConfigFile(**caracal_config_options)

    # load the template yaml
    caracal_template = caracal_options["caracal_template_strategy"]

    with open(caracal_template) as in_file:
        caracal_template_yaml = yaml.load(in_file, Loader=yaml.Loader) # dict

    # update the template yaml with the user options
    final_caracal_options = _update_caracal_template_with_options(caracal_template_yaml, caracal_config_file_options)

    output_yaml_path = working_dir/ "caracal_polcal.latest.yaml"
    final_caracal_yaml_path = write_and_timestamp_caracal_strategy(output_yaml_path, final_caracal_options)

    return final_caracal_yaml_path


def start_caracal(caracal_options: CaracalOptions, working_dir: Path) -> None:
    
    caracal_config_file = edit_caracal_template(caracal_options, working_dir)

    with open(working_dir / "go_caracal.sh", "w") as file:
        # TODO: let users supply conda env?
        file.write("startconda\n")
        file.write("conda activate caracalfork\n")

        # Makes sure cache is not saved in homedir (no space)
        file.write(f"export APPTAINER_CACHEDIR={working_dir}/temporary-apptainer-cache")
        file.write("\n")
        file.write("\n")
        file.write(f"caracal -ct singularity -c {caracal_config_file}")

    logger.info("Starting caracal")
    # TODO check where caracal runs, it might be the parent directory actually..
    os.system(f"bash {working_dir / 'go_caracal.sh'}")

