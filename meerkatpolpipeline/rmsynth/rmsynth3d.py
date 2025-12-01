from __future__ import annotations

from pathlib import Path

import numpy as np
from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.rmsynth.rmsynth1d import (
    check_pipeline_complete,
    create_config_from_template,
    find_last_log_file,
)
from meerkatpolpipeline.utils.utils import execute_command


class RMSynth3Doptions(BaseOptions):
    """A basic class to handle options for 3D RM synthesis. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""    
    rmsynth_3d_config_template: Path
    """Path to a template config file for the 3D RM synthesis step."""
    module_directory: Path
    """Path to POSSUM_Polarimetry_Pipeline pipeline/modules/ directory for this step."""
    working_directory: Path | None = None
    """Optional: Path to working directory for this step. If 'null' will assign a default working directory."""
    overwrite: bool = False
    """Overwrite existing output files? Default False, in which case it will skip this step if output already exists"""


def write_noise_file(out_path: Path, rms: np.ndarray) -> None:
    # Each line: a single float value; preserve scientific notation if needed
    with out_path.open("w") as f:
        for v in rms:
            if np.isfinite(v):
                f.write(f"{v:.8g}\n")
            else:
                f.write("nan\n")

def run_rmsynth3d(rmsynth3d_options: dict | RMSynth3Doptions,
                  stokesI_cube_path: Path,
                  rmsynth3d_workdir: Path,
                  rms_qu_average: list[float] | None = None
) -> Path:
    """
    Run 3D rm synthesis using the POSSUM_Polarimetry_Pipeline setup
    
    args:
        rmsynth3d_options (dict | RMSynth3Doptions): Dictionary storing RMSynth3Doptions for this step
        stokesI_cube_path (Path): Path to the Stokes I cube file for this targetfield.
        rmsynth3d_workdir (Path): The working directory for the rmsynth3d step

    returns:
        Path to the folder with the output RMSynth 3D files

    """
    logger = get_run_logger()

    if not rmsynth3d_options['overwrite']:
        # TODO: should check for all expected output files, not just one
        expected_output_file = rmsynth3d_workdir / f"{rmsynth3d_options['targetfield']}.pipeline_3d_PhiPeakPIfit_rm2.fits"
        if expected_output_file.exists():
            logger.info(f"RMSynth 3D output file {expected_output_file} already exists and overwrite is False. Skipping RMSynth 3D step.")
            return rmsynth3d_workdir

    module_dir = Path(rmsynth3d_options["module_directory"]).expanduser().resolve()
    if not module_dir.exists():
        raise FileNotFoundError(f"Module directory not found: {module_dir}")

    if not stokesI_cube_path.exists():
        raise FileNotFoundError(f"Stokes I cube not found: {stokesI_cube_path}")

    # Set working directory, checking for user overwrite
    if rmsynth3d_options['working_directory'] is None:
        rmsynth3d_options['working_directory'] = rmsynth3d_workdir.expanduser().resolve()
    else:
        rmsynth3d_workdir = Path(rmsynth3d_options['working_directory']).expanduser().resolve()

    if rms_qu_average is not None:
        # write .dat file with 1 line of rms value per channel
        noise_file_path = rmsynth3d_workdir / f"{rmsynth3d_options['targetfield']}.noise.dat"
        logger.info(f"Writing noise file for RMSynth 3D from provided Q/U rms per channel. Writing to {noise_file_path}")
        write_noise_file(noise_file_path, np.array(rms_qu_average))

    # Create config file from template
    config_path = create_config_from_template(
        rmsynth_options=rmsynth3d_options,
        template_option_key="rmsynth_3d_config_template",
        stokesI_cube_path=stokesI_cube_path,
        output_path=rmsynth3d_workdir / "rmsynth_3d_config.ini",
        snr_option_key=None,  # or just omit (default)
    )
    logger.info(f"Created RMSynth 3D config file at {config_path}")

    # Run the POSSUM pipeline rmsynth_3d.py script with this config
    rmsynth3d_script = module_dir.parent / "pipeline.py"
    if not rmsynth3d_script.exists():
        raise FileNotFoundError(f"pipeline.py script not found, tried searching at {rmsynth3d_script}")


    cmd = ["python3", str(rmsynth3d_script), str(config_path)]
    result = execute_command(cmd)

    if result.returncode != 0:
        logger.error(f"RMSynth 3D failed with stderr:\n{result.stderr}")
        raise RuntimeError("RMSynth 3D execution failed.")
    
    logfile = find_last_log_file(rmsynth3d_workdir)
    if logfile is None:
        logger.error("No log file found after RMSynth 3D execution.")
        raise RuntimeError("RMSynth 3D execution failed: no log file found.")
    
    status = check_pipeline_complete(logfile)
    if status != "Completed":
        logger.error(f"RMSynth 3D pipeline did not complete successfully. Check log file at {logfile}")
        raise RuntimeError("RMSynth 3D execution failed: pipeline did not complete successfully.")
    
    logger.info("RMSynth 3D completed successfully.")

    return rmsynth3d_workdir