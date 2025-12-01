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
    module_directory: Path
    """Path to POSSUM_Polarimetry_Pipeline pipeline/modules/ directory for this step."""
    catalog_file: Path | None = None
    """Path to input catalog file for this step. If null, will attempt to use the PYBDSF catalog from previous step."""
    working_directory: Path | None = None
    """Optional: Path to working directory for this step. If 'null' will assign a default working directory."""
    snr: int = 20
    """SNR threshold in Stokes I for doing RM synthesis. Default 20"""
    overwrite: bool = False
    """Overwrite existing output files? Default False, in which case it will skip this step if output already exists"""
    hutschenreuter_map: Path | None = None
    """Path to Hutschenreuter Galactic RM map FITS file. If provided, will do Galactic RM correction using this map."""


def _coerce_to_str(v: Any) -> str:
    """Coerce common types (Path, numbers, bool) to ini-safe strings."""
    if isinstance(v, Path):
        return str(v.expanduser().resolve())
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def find_last_log_file(logs_directory: Path) -> Path | None:
    log_files = sorted(logs_directory.glob("*.log"))
    if not log_files:
        return None
    return log_files[-1]


def check_pipeline_complete(log_file_path):
    with open(log_file_path) as file:
        log_contents = file.read()
    
    if "Pipeline stopping due to errors" in log_contents:
        return "Failed"

    if "Pipeline complete." in log_contents:
        return "Completed"
    else:
        return "Failed"

def create_config_from_template(
    rmsynth_options: dict,  # or dict | RMSynth1Doptions | RMSynth3Doptions
    template_option_key: str,
    stokesI_cube_path: Path,
    output_path: Path,
    snr_option_key: str | None = None,
) -> Path:
    """
    Read a POSSUM pipeline .ini template and overwrite only the [DEFAULT] keys
    provided by the user.

    Parameters
    ----------
    rmsynth_options : dict-like
        Must contain:
          - 'targetfield': str
          - template_option_key: Path-like to the template .ini
          - 'working_directory': str or Path (for logging_directory)
          - Any other keys intended for [DEFAULT] (e.g., data_file, working_directory, etc.)
          - If snr_option_key is not None: that key must be present and integer-like.
    template_option_key : str
        Key in rmsynth_options that points to the template path
        (e.g. 'rmsynth_3d_config_template' or 'rmsynth_1d_config_template').
    stokesI_cube_path : Path
        Path to the Stokes I cube file for this targetfield.
    output_path : Path
        Path for the output .ini
    snr_option_key : str or None, optional
        If provided, also create 'catalog_file_snr' in the [DEFAULT] section based on:
        catalog_file -> catalog_file_snr = "<base>.<snr>sig.fits"

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

    # Resolve template path from the provided key
    template_path = Path(rmsynth_options[template_option_key]).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Load template
    config = ConfigParser(interpolation=None)  # leave %-style literals untouched as strings
    with template_path.open("r", encoding="utf-8") as f:
        config.read_file(f)

    # Ensure DEFAULT section exists
    if not config.defaults():
        raise ValueError("Template INI file is missing [DEFAULT] section.")

    # Build the set of default keys to write: every user-specified key except the special ones
    special_keys = {template_option_key, "enable"}

    # logging_directory = working_directory
    rmsynth_options["logging_directory"] = rmsynth_options["working_directory"]

    for k, v in rmsynth_options.items():
        if k in special_keys:
            continue
        # We always allow overriding targetfield and any other defaults
        config["DEFAULT"][k] = _coerce_to_str(v)

    # Ensure data_file points to Stokes I cube
    config["DEFAULT"]["data_file"] = str(stokesI_cube_path.resolve())

    # Optional: SNR-based catalogue file (1D-style behaviour)
    if snr_option_key is not None:
        snr_value = int(rmsynth_options[snr_option_key])
        catalog_file = config["DEFAULT"]["catalog_file"]
        config["DEFAULT"]["catalog_file_snr"] = catalog_file.replace(
            ".fits", f".{snr_value:d}sig.fits"
        )

    # Write out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        config.write(f)

    return output_path


def run_rmsynth1d(rmsynth1d_options: dict | RMSynth1Doptions, stokesI_cube_path: Path, rmsynth1d_workdir: Path) -> tuple[Path, Path, Path]:
    """
    Run 1D rm synthesis using the POSSUM_Polarimetry_Pipeline setup
    
    args:
        rmsynth1d_options (dict | RMSynth1Doptions): Dictionary storing RMSynth1Doptions for this step
        stokesI_cube_path (Path): Path to the Stokes I cube file for this targetfield.
        rmsynth1d_workdir (Path): The working directory for the rmsynth1d step

    returns:
        catalog: Path to the output RMSynth 1D catalog FITS file
        fdf: Path to the output RMSynth 1D FDF FITS file
        spectra: Path to the output RMSynth 1D spectra FITS file

    """
    logger = get_run_logger()

    # this has to change if the template ever changes
    catalog = rmsynth1d_workdir / f"{rmsynth1d_options['targetfield']}.rmsynth1d.catalog.fits"
    fdf = rmsynth1d_workdir / f"{rmsynth1d_options['targetfield']}.rmsynth1d.FDF.fits"
    spectra = rmsynth1d_workdir / f"{rmsynth1d_options['targetfield']}.rmsynth1d.spectra.fits"

    if catalog.exists() and fdf.exists() and spectra.exists() and (not rmsynth1d_options['overwrite']):
        logger.info("RMSynth 1D output files already exist and overwrite is False; skipping this step.")
        return catalog, fdf, spectra


    module_dir = Path(rmsynth1d_options["module_directory"]).expanduser().resolve()
    if not module_dir.exists():
        raise FileNotFoundError(f"Module directory not found: {module_dir}")

    if not stokesI_cube_path.exists():
        raise FileNotFoundError(f"Stokes I cube not found: {stokesI_cube_path}")

    # Set working directory, checking for user overwrite
    if rmsynth1d_options['working_directory'] is None:
        rmsynth1d_options['working_directory'] = rmsynth1d_workdir.expanduser().resolve()
    else:
        rmsynth1d_workdir = Path(rmsynth1d_options['working_directory']).expanduser().resolve()

    # Create config file from template
    config_path = create_config_from_template(
        rmsynth_options=rmsynth1d_options,
        template_option_key="rmsynth_1d_config_template",
        stokesI_cube_path=stokesI_cube_path,
        output_path=rmsynth1d_workdir / "rmsynth_1d_config.ini",
        snr_option_key="snr",
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
    
    logfile = find_last_log_file(rmsynth1d_workdir)
    if logfile is None:
        logger.error("No log file found after RMSynth 1D execution.")
        raise RuntimeError("RMSynth 1D execution failed: no log file found.")
    
    status = check_pipeline_complete(logfile)
    if status != "Completed":
        logger.error(f"RMSynth 1D pipeline did not complete successfully. Check log file at {logfile}")
        raise RuntimeError("RMSynth 1D execution failed: pipeline did not complete successfully.")
    
    logger.info("RMSynth 1D completed successfully.")

    return catalog, fdf, spectra