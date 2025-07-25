from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal._caracal import CrossCalOptions
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import (
    add_timestamp_to_path,
    check_create_symlink,
    find_calibrated_ms,
)


def do_casa_crosscal(
        crosscal_options: CrossCalOptions,
        preprocessed_ms: Path,
        crosscal_base_dir: Path,
        ms_summary: dict
    ) -> Path:
    """Run the CASA cross-calibration step."""
    logger = get_run_logger()

    print("TODO")
      
    # Check if casacrosscal was already done by a previous run
    calibrated_ms = find_calibrated_ms(crosscal_base_dir, preprocessed_ms, look_in_subdirs=['casacrosscal'])
    if calibrated_ms is not None:
        logger.info(f"Casa cross-calibration already done, found calibrated MS at {calibrated_ms}. Skipping caracal step.")
        casacrosscal_dir = calibrated_ms.parent
        return casacrosscal_dir

    else:

        casacrosscal_dir = crosscal_base_dir / "casacrosscal"
        
        # to be consistent with caracal we add a symlink to the crosscal directory
        preprocessed_ms_symlink = casacrosscal_dir / preprocessed_ms.name
        preprocessed_ms_symlink = check_create_symlink(preprocessed_ms_symlink, preprocessed_ms)
        
        logger.info(f"Starting casa crosscal in {casacrosscal_dir} with options: {crosscal_options}")

        print("TODO: implement casa script for cross-calibration")

        return "TODO"

# to prepare target and calibrators for casa crosscal,
# see /net/rijn9/data2/osinga/meerkatBfields/Abell754/test_annalisa_script/copy_and_split_ms.py