from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Any

from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import execute_command


class RMSynth1Doptions(BaseOptions):
    """A basic class to handle options for 1D RM synthesis. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""    
    rmsynth_1d_config_template: Path
    """Path to a template config file for the 1D RM synthesis step."""
    working_directory: Path
    """Path to working directory for this step."""
    module_directory: Path
    """Path to POSSUM_Polarimetry_Pipeline pipeline/modules/ directory for this step."""
    catalog_file: Path
    """Path to input catalog file for this step."""
    snr: int = 20
    """SNR threshold in Stokes I for doing RM synthesis."""


def _coerce_to_str(v: Any) -> str:
    """Coerce common types (Path, numbers, bool) to ini-safe strings."""
    if isinstance(v, Path):
        return str(v.expanduser().resolve())
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def create_config_from_template(
    rmsynth1d_options: dict | RMSynth1Doptions,
    stokesI_cube_path: Path,
    output_path: Path,
) -> Path:
    """
    Read a POSSUM pipeline .ini template and overwrite only the [DEFAULT] keys
    provided by the user.

    Parameters
    ----------
    rmsynth1d_options : dict created from RMSynth1Doptions 
        Must contain:
          - 'targetfield': str
          - 'rmsynth_1d_config_template': Path-like to the template .ini
          - Any other keys intended for [DEFAULT] (e.g., data_file, working_directory, etc.)
    
    stokesI_cube_path: Path to the Stokes I cube file for this targetfield.
    
    output_path: Path for the output .ini
    
    Returns
    -------
    output_path : Path
        Path to the written .ini

    Notes
    -----
    - Comments and exact formatting from the template are not preserved by ConfigParser.
      Keys and section order are preserved.
    - Only the [DEFAULT] section is updated; all other sections are copied verbatim
      (subject to ConfigParser reserialization).
    """

    template_path = Path(rmsynth1d_options["rmsynth_1d_config_template"]).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Load template
    config = ConfigParser(interpolation=None)  # leave %-style literals untouched as strings
    # Preserve key case in output? ConfigParser lowercases by default. We'll accept lowercase keys.
    with template_path.open("r", encoding="utf-8") as f:
        config.read_file(f)

    # Ensure DEFAULT section exists
    if not config.defaults():
        raise ValueError("Template INI file is missing [DEFAULT] section.")

    # Build the set of default keys to write: every user-specified key except the special ones
    special_keys = {"rmsynth_1d_config_template", "enable"}
    # also do logging_directory = working_directory
    rmsynth1d_options['logging_directory'] = rmsynth1d_options['working_directory']
    for k, v in rmsynth1d_options.items():
        if k in special_keys:
            continue
        # We always allow overriding targetfield and any other defaults
        config["DEFAULT"][k] = _coerce_to_str(v)

    # also do data_file pointing to stokes I cube
    config["DEFAULT"]["data_file"] = str(stokesI_cube_path.resolve())

    # then write SNR version of catalogue file
    config["DEFAULT"]["catalog_file_snr"] = config["DEFAULT"]["catalog_file"].replace(".fits",f".{rmsynth1d_options['snr']:d}sig.fits")

    # Write out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        config.write(f)

    return output_path


def run_rmsynth1d(rmsynth1d_options: dict | RMSynth1Doptions, stokesI_cube_path: Path, rmsynth1d_workdir: Path, ) -> None:
    """
    Run 1D rm synthesis using the POSSUM_Polarimetry_Pipeline setup
    """
    logger = get_run_logger()

    module_dir = Path(rmsynth1d_options["module_directory"]).expanduser().resolve()
    if not module_dir.exists():
        raise FileNotFoundError(f"Module directory not found: {module_dir}")

    if not stokesI_cube_path.exists():
        raise FileNotFoundError(f"Stokes I cube not found: {stokesI_cube_path}")

    # Create config file from template
    config_path = create_config_from_template(
        rmsynth1d_options,
        stokesI_cube_path,
        output_path=rmsynth1d_workdir / "rmsynth_1d_config.ini",
    )
    logger.info(f"Created RMSynth 1D config file at {config_path}")

    # Run the POSSUM pipeline rmsynth_1d.py script with this config
    rmsynth1d_script = module_dir.parent / "pipeline.py"
    if not rmsynth1d_script.exists():
        raise FileNotFoundError(f"pipeline.py script not found, tried searching at {rmsynth1d_script}")


    cmd = ["python3", str(rmsynth1d_script), str(config_path)]
    result = execute_command(cmd)

    if result.returncode != 0:
        logger.error(f"RMSynth 1D failed with stderr:\n{result.stderr}")
        raise RuntimeError("RMSynth 1D execution failed.")
    
    logger.info("RMSynth 1D completed successfully.")

    return None