"""A prefect based pipeline that:
- will perform meerKAT L-band data processing
- given an input strategy file
"""

from __future__ import annotations

import sys
from pathlib import Path

from configargparse import ArgumentParser
from prefect import flow  #, tags, unmapped

sys.path.append('/home/osingae/OneDrive/postdoc/projects/MEERKAT_similarity_Bfields/meerkatpolpipeline')

from meerkatpolpipeline.caracal import _caracal
from meerkatpolpipeline.check_calibrator import check_calibrator
from meerkatpolpipeline.configuration import (
    Strategy,
    get_options_from_strategy,
    load_and_copy_strategy,
    log_enabled_operations,
)
from meerkatpolpipeline.download.download import start_download
from meerkatpolpipeline.logging import logger


def check_caracal_run():
    print("TODO: verify that caracal completed succesfully, and return the -cal.ms path location")





# @flow(name="MeerKAT L-band pipeline")
def process_science_fields(
    strategy: Strategy,
    working_dir: Path
) -> None:
    """
    Flint flow that will execute all the enabled steps as tasks. 

    Each task will be done in a subdirectory
    """

    print("TODO")

    enabled_operations = log_enabled_operations(strategy)

    # first step, download & clip channels
    if "download" in enabled_operations:
        # create subdirectory 'download'
        download_workdir = working_dir / "download"
        download_workdir.mkdir(exist_ok=True) # runs can be repeated

        download_options = get_options_from_strategy(strategy, operation="download")
        
        start_download(download_options, working_dir=download_workdir)

        # TODO: add clipping, which should be checked on re-run, even if the MS is downloaded

    # second step, caracal for cross-calibration
    if "caracal" in enabled_operations:
        caracal_options = get_options_from_strategy(strategy, operation="caracal")

        caracal_workdir = caracal_options['msdir'] # caracal will always work in the msdir directory
        # caracal_workdir.mkdir(exist_ok=True) # runs can be repeated

        _caracal.start_caracal(caracal_options, working_dir=caracal_workdir)

    # TODO: optionally second step could also be casa script. 
    if "casacrosscal" in enabled_operations:
        print("TODO")
    
    if "check_calibrator" in enabled_operations:
        check_calibrator_workdir = working_dir / "check_calibrator"
        check_calibrator_workdir.mkdir(exist_ok=True)

        check_calibrator_options = get_options_from_strategy(strategy, operation="check_calibrator")

        # need location of calibrator measurement set to split the polcal


        check_calibrator(check_calibrator_options, working_dir=check_calibrator_workdir)




    
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

    logger.info(f"Starting pipeline for {target=}")

    process_science_fields(
        strategy=strategy,
        working_dir=working_dir
    )

    # process_science_fields.with_options(
    #     name=f"MeerKAT L-band pipeline - {target}"
    #     # , task_runner=dask_task_runner
    # )(
    #     strategy=strategy,
    # )


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