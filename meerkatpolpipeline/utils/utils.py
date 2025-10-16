
from __future__ import annotations

import argparse
import shlex
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from prefect.logging import get_run_logger

# Suppress FITSFixedWarning from astropy when opening FITS files
warnings.filterwarnings('ignore', category=FITSFixedWarning)

class PrintLogger:
    """Custom logger that prints to stdout."""
    def info(self, msg):
        print(msg)
    def warning(self, msg):
        print("WARNING:", msg)
    def error(self, msg):
        print("ERROR:", msg)


def str2bool(v: str) -> bool:
    """Convert a string representation of truth to a boolean value."""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean type expected.')

def parse_comma_separated(value: str) -> list[str]:
    return value.split(',')

def read_fits_data_and_frequency(filename: Path) -> tuple[np.ndarray, np.ndarray, WCS]:
    """Read a FITS file, return the image data, frequency from CRVAL3 in Hz and wcs"""
    with fits.open(filename) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
        frequency = header['CRVAL3']  # Frequency in Hz from CRVAL3
        assert 'FREQ' in header['CTYPE3'], f"CRVAL3 should be frequency axis, instead found {header['CTYPE3']}"

    # remove freq and stokes axis
    if len(data.shape) == 4:
        data = data[0,0]
    if len(data.shape) == 3:
        data = data[0]

    return data, frequency, wcs


def convert_units(data: np.ndarray, fitsimage: Path | fits.HDUList) -> np.ndarray:
    """
    Convert the units of 'data' array which is assumed to be Jy/beam 
    to Jy/pix using the beam information given in the header of 'fitsimage'.
    """
    if isinstance(fitsimage, str) or isinstance(fitsimage, Path):
        with fits.open(fitsimage) as hdul:
            header = hdul[0].header 
    else:
        hdul = fitsimage
        header = hdul[0].header 

    if header['BUNIT'].upper() == 'JY/BEAM':
        bmaj = header['BMIN'] * u.deg
        bmin = header['BMAJ'] * u.deg
        # pix_size = abs(header['CDELT2']) * u.deg  # assume square pix size

        beammaj = bmaj / (2. * (2. * np.log(2.)) ** 0.5)  # Convert to sigma
        beammin = bmin / (2. * (2. * np.log(2.)) ** 0.5)  # Convert to sigma
        pix_area = abs(header['CDELT1'] * header['CDELT2']) * u.deg ** 2
        beam_area = 2. * np.pi * 1.0 * beammaj * beammin  # beam area in steradians
        beam2pix = beam_area / pix_area  # beam area in pixels
    else:
        raise ValueError(f"UNITS ARE NOT Jy/beam. PLEASE CHECK HEADER. Found {header['BUNIT']}")
    
    data = data / beam2pix  # convert to Jy/pix
    return data


def add_timestamp_to_path(
    input_path: Path | str, timestamp: datetime | None = None
) -> Path:
    """Add a timestamp to a input path, where the timestamp is the
    current data and time. The time will be added to the name component
    before the file suffix. If the name component of the `input_path`
    has multiple suffixes than the timestamp will be added before the last.

    Args:
        input_path (Union[Path, str]): Path that should have a timestamp added
        timestamp: (datetime): The date-time to add. If None the current time is used. Defaults to None.
    Returns:
        Path: Updated path with a timestamp in the file name
    """
    input_path = Path(input_path)
    timestamp = timestamp if timestamp else datetime.now()

    time_str = timestamp.strftime("%Y%m%d-%H%M%S")
    new_name = f"{input_path.stem}-{time_str}{input_path.suffix}"
    output_path = input_path.with_name(new_name)

    return output_path

def execute_command(
    cmd: str | list[str],
    test: bool = False,
    logfile: Path | None = None
) -> subprocess.CompletedProcess:
    """
    Run a command (always as a list) with error handling.
    :param cmd: either a single string or a list of args
    :param test: if True, logs only and does not execute
    """
    logger = get_run_logger()

    # Normalize to list
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
        logger.info("Parsed command string into list: %s", cmd_list)
    else:
        cmd_list = cmd
        logger.info("Executing command list: %s", cmd_list)

    if test:
        logger.info("Test mode enabled; skipping command execution.")
        return None

    # If the first element is "bash", check the script exists
    if cmd_list and cmd_list[0] == "bash" and len(cmd_list) > 1:
        script_path = Path(cmd_list[1])
        if not script_path.exists():
            raise FileNotFoundError(
                f"Script not found at {script_path!r}; please verify the path."
            )
    
    # Open logfile if requested
    log_handle = None
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(logfile, "w", encoding="utf-8")

    try:
        # Run command and stream output line by line
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        for line in process.stdout:
            print(line, end="")  # live terminal output
            output_lines.append(line)
            if log_handle:
                log_handle.write(line)
                log_handle.flush()

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd_list)

        result = subprocess.CompletedProcess(cmd_list, process.returncode, "".join(output_lines), "")
        return result

    except subprocess.CalledProcessError as e:
        logger.error("Command failed (exit %d)", e.returncode)
        raise

    finally:
        if log_handle:
            log_handle.close()

def check_create_symlink(symlink: Path, original_path: Path) -> Path:
    """
    check if 'symlink' exists, if not create it, linking to 'original_path'
    """
    logger = get_run_logger()
    if not symlink.exists():
        logger.info(f"Creating symlink at {symlink} pointing to {original_path}")
        symlink.symlink_to(original_path)
    else:
        logger.info(f"Symlink {symlink} already exists. Skipping symlink creation.")
    return symlink

def find_calibrated_ms(
        crosscal_base_dir: Path,
        preprocessed_ms: Path,
        look_in_subdirs: list = ["caracal", "casacrosscal"],
        suffix: str = "-cal.ms"
    ) -> Path | None:
    """
    If both crosscal steps are disabled, check for the existence of a calibrated
    measurement set in either the "caracal" or "casacrosscal" subdirectory

    Args:
        crosscal_base_dir (Path): Base directory where the crosscal directories are located.
        preprocessed_ms (Path): Path to the preprocessed measurement set.
        look_in_subdirs (list): List of subdirectories to look for the calibrated measurement set.
                                 Defaults to ["caracal", "casacrosscal"].
        suffix (str): Suffix to append to the preprocessed measurement set name when searching.
                       : "-cal.ms" is the default for the caracal split MS with the (corrected) calibrators
                       : "-{target}-corr.ms" is the default for the caracal split MS with the corrected target 
    Returns:
        Path | None: Path to the calibrated measurement set if found, otherwise None.
    """
    # check if suffix ends with .ms otherwise append .ms
    if not suffix.endswith(".ms"):
        suffix += ".ms"

    for subdir in look_in_subdirs:
        ms_path = crosscal_base_dir / subdir / (preprocessed_ms.stem + suffix)
        if ms_path.exists():
            return ms_path
    return None


def make_utf8(inp):
    """
    Convert input to utf8 instead of bytes
    :param inp: string input
    :return: input in utf-8 format
    """

    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def remove_one_copy_from_filename(paths: np.ndarray[Path]) -> np.ndarray[Path]:
    """Remove one '.copy' extension from each PosixPath in the array.
    Useful because facetselfcal with --start != 0 requires inputs without the .copy to resume a run.
    otherwise it will just create another .copy file again.
    """
    new_paths = []
    for p in paths:
        name = p.name
        if name.endswith(".copy"):
            name = name[:-5]  # remove the last ".copy"
        new_paths.append(p.with_name(name))
    return np.array(new_paths, dtype=object)