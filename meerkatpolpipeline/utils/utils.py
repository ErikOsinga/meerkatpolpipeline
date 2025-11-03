
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
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


def convert_units(data: np.ndarray, fitsimage: Path | fits.HDUlist) -> np.ndarray:
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
        look_in_subdirs (list): list of subdirectories to look for the calibrated measurement set.
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

    # caracal renames as follows
    if "+" in suffix:
        suffix = suffix.replace('+','_p_')
    if "-" in suffix:
        suffix = suffix.replace('-','_m_')

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


def _read_ds9_regions(regions_path: str) -> list[str]:
    """
    Read a DS9 .reg file and return all non-empty lines as a list.
    Keeps comments and global directives so they can be preserved in output.
    """
    with open(regions_path) as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    return lines


def _is_fk5_context(lines: list[str]) -> bool:
    """
    Return True if the file declares FK5 coordinates (typical for PyBDSF regions).
    """
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("fk5"):
            return True
        # DS9 allows 'global' lines; continue scanning until a coord system appears.
        if s and not (s.startswith("#") or s.startswith("global")):
            # If the first non-comment, non-global line starts with fk5( or shape,
            # we will assume FK5 was omitted and default to False here.
            break
    return False


def _extract_fk5_center(line: str) -> SkyCoord | None:
    """
    Extract the central coordinate for common DS9 shapes written in FK5.
    Assumes decimal degrees if numeric; otherwise lets SkyCoord parse.
    Supported shapes: circle, ellipse, box, point, text.
    For polygons, estimate center as the mean of vertices.

    Returns None if the line does not contain a recognizable FK5 region.
    """
    ln = line.strip()
    if not ln or ln.startswith("#") or ln.lower().startswith("global") or ln.lower().startswith("fk5"):
        return None

    # DS9 region line examples (FK5):
    # circle(ra,dec,r)
    # ellipse(ra,dec,major,minor,pa)
    # box(ra,dec,width,height,pa)
    # point(ra,dec)
    # text(ra,dec) # text={...}
    # polygon(ra1,dec1, ra2,dec2, ..., ran,decn)

    m = re.match(r"([a-zA-Z]+)\s*\(([^)]+)\)", ln)
    if not m:
        return None

    shape = m.group(1).lower()
    args_str = m.group(2)

    # Split args by comma while respecting potential whitespace
    parts = [p.strip() for p in args_str.split(",") if p.strip()]
    if len(parts) < 2:
        return None

    def _to_coord(ra_str: str, dec_str: str) -> SkyCoord:
        # Try numeric degrees first; fall back to SkyCoord parsing if needed
        try:
            ra = float(ra_str)
            dec = float(dec_str)
            return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="fk5")
        except ValueError:
            return SkyCoord(ra_str + " " + dec_str, frame="fk5")

    if shape in {"circle", "ellipse", "box", "point", "text"}:
        ra_s, dec_s = parts[0], parts[1]
        return _to_coord(ra_s, dec_s)

    if shape == "polygon":
        # Expect even number of coordinates: ra1,dec1, ra2,dec2, ...
        if len(parts) < 4 or len(parts) % 2 != 0:
            return None
        ras: list[float] = []
        decs: list[float] = []
        # Attempt numeric parse for centroid; if any non-numeric, fall back to first pair
        numeric_ok = True
        for i in range(0, len(parts), 2):
            try:
                ras.append(float(parts[i]))
                decs.append(float(parts[i + 1]))
            except ValueError:
                numeric_ok = False
                break
        if numeric_ok:
            return SkyCoord(ra=(sum(ras) / len(ras)) * u.deg, dec=(sum(decs) / len(decs)) * u.deg, frame="fk5")
        else:
            # Fallback: use first vertex as approximate center
            return _to_coord(parts[0], parts[1])

    # Unsupported shape
    return None


def _filter_regions_within_radius(
    regions_path: str,
    center_coord: SkyCoord,
    radius_deg: float
) -> list[str]:
    """
    Load a DS9 .reg file and return only the lines whose region centers
    lie within the given radius of center_coord. Preserves header/global lines.
    """
    lines = _read_ds9_regions(regions_path)
    out_lines: list[str] = []

    # Always keep DS9 header and global lines
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or s.lower().startswith("global"):
            out_lines.append(ln)
        elif s.lower().startswith("fk5"):
            out_lines.append(ln)
        else:
            # Defer filtering to second pass
            pass

    fk5 = _is_fk5_context(lines)
    if not fk5:
        # If FK5 not declared, prepend it to be explicit (most PyBDSF regions are FK5)
        out_lines.insert(0, "fk5")

    # Now filter shape lines
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or s.lower().startswith("global") or s.lower().startswith("fk5"):
            continue
        coord = _extract_fk5_center(ln)
        if coord is None:
            # If we cannot parse, keep line conservatively or drop?
            # Choose conservative: keep only if clearly not a region shape.
            # Here we drop unparseable region shapes to avoid false positives.
            continue
        sep = center_coord.separation(coord)
        if sep <= radius_deg * u.deg:
            out_lines.append(ln)

    return out_lines


def filter_sources_within_radius(
    sourcelist_path: str,
    center_coord: SkyCoord,
    radius_deg: float = 0.5,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    output_path: str | None = None,
    regions_path: str | None = None,
    regions_output_path: str | None = None,
) -> tuple[Table, list[str] | None]:
    """
    Filter a PYBDSF source list (.fits catalogue) to sources within radius of center_coord.
    Optionally filter a corresponding DS9 .reg file to the same footprint.

    Parameters
    ----------
    sourcelist_path : str
        Path to the input PYBDSF source list (.fits).
    center_coord : SkyCoord
        Central coordinate to search around.
    radius_deg : float, optional
        Search radius in degrees (default: 0.5).
    ra_col : str, optional
        Column name for Right Ascension in degrees (default: 'RA').
    dec_col : str, optional
        Column name for Declination in degrees (default: 'DEC').
    output_path : str, optional
        If provided, writes the filtered catalogue to this path.
    regions_path : str, optional
        If provided, path to a DS9 .reg file (e.g., PyBDSF regions) to filter by radius.
        Lines are assumed to be FK5 coordinates (decimal degrees typical for PyBDSF).
    regions_output_path : str, optional
        If provided and regions_path is given, writes the filtered .reg to this path.

    Returns
    -------
    filtered_table : astropy.table.Table
        Catalogue rows within the radius.
    filtered_regions : list[str] or None
        Filtered DS9 region lines if regions_path is provided; otherwise None.
    """
    # Load and validate catalogue
    table = Table.read(sourcelist_path)
    if ra_col not in table.colnames or dec_col not in table.colnames:
        raise KeyError(f"Input catalogue must contain '{ra_col}' and '{dec_col}' columns.")

    source_coords = SkyCoord(ra=table[ra_col].value * u.deg, dec=table[dec_col].value * u.deg)
    separations = center_coord.separation(source_coords)
    mask = separations <= radius_deg * u.deg
    filtered_table = table[mask]

    if output_path:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        filtered_table.write(output_path, overwrite=True)

    filtered_regions: list[str] | None = None
    if regions_path:
        filtered_regions = _filter_regions_within_radius(regions_path, center_coord, radius_deg)
        if regions_output_path:
            Path(regions_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(regions_output_path, "w") as f:
                for ln in filtered_regions:
                    f.write(ln + "\n")

    return filtered_table, filtered_regions


def get_fits_image_center(image_path: str | Path) -> SkyCoord:
    """
    Returns the celestial coordinate of the image centre from a FITS file.

    Parameters
    ----------
    image_path : str
        Path to the FITS image.

    Returns
    -------
    center_coord : astropy.coordinates.SkyCoord
        The sky coordinate of the image centre.
    """
    # Open FITS and extract header
    with fits.open(str(image_path)) as hdul:
        header = hdul[0].header

    # Build celestial WCS (important for redundant axes compliance)
    wcs = WCS(header).celestial

    # Determine the pixel coordinates of the image centre
    nx = header.get("NAXIS1")
    ny = header.get("NAXIS2")

    if nx is None or ny is None:
        raise ValueError("FITS header missing NAXIS1/NAXIS2 keywords.")

    x_center = nx / 2.0
    y_center = ny / 2.0

    # Convert pixel coordinates to world (RA, Dec)
    ra_deg, dec_deg = wcs.wcs_pix2world(x_center, y_center, 0)
    center_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

    return center_coord


def find_rms(data: np.ndarray, mask_threshold: float = 1e-7) -> float:
    """
    Iteratively estimate the RMS noise level of an array by sigma-clipping.
        Inspired by DDFacet findrms.py

    This function removes values below a specified absolute threshold, then
    iteratively computes the standard deviation (RMS) while excluding outliers
    beyond a fixed multiple (3sigma by default) of the current estimate around the median.
    Iteration stops when the relative change in RMS is below 10%.

    Parameters
    ----------
    data : np.ndarray
        Input array containing pixel or intensity values.
    mask_threshold : float, optional
        Absolute value below which data points are ignored. Default is 1e-7.

    Returns
    -------
    float
        The estimated RMS value after iterative clipping.

    Notes
    -----
    - The iteration is capped at 10 steps.
    - This approach is robust against bright sources that could bias the RMS.
    - Convergence is defined as a relative RMS change < 0.1 (10%).
    """
    # Mask out low-amplitude pixels
    valid_data = data[np.abs(data) > mask_threshold]
    if valid_data.size == 0:
        return np.nan
        # raise ValueError("No valid data points above mask_threshold.")

    # Initial RMS estimate
    rms_previous = np.std(valid_data)
    convergence_threshold = 0.1  # 10% change allowed
    sigma_cut = 3.0  # 3-sigma clipping window
    median_value = np.median(valid_data)

    for _ in range(10):
        within_clip = np.abs(valid_data - median_value) < rms_previous * sigma_cut
        clipped_data = valid_data[within_clip]

        rms_new = np.std(clipped_data)
        relative_change = np.abs((rms_new - rms_previous) / rms_previous)

        if relative_change < convergence_threshold:
            break
        rms_previous = rms_new

    return rms_new


def find_pybdsf_filtered_cats(cube_imaging_workdir: Path) -> tuple[Path, Path]:
    """
    Attempt to find filtered PyBDSF catalog files in the given cube imaging workdir.

    returns:
        tuple of Paths: (sourcelist_fits_filtered, sourcelist_reg_filtered)
    """
    # Assuming pybdsf results in these files
    sourcelist_fits = cube_imaging_workdir / 'sourcelist.srl.fits'
    sourcelist_reg = cube_imaging_workdir / 'sourcelist.srl.reg'
    rmsmap = cube_imaging_workdir / 'rms_map.fits'

    assert sourcelist_fits.exists(), f"Could not find PyBDSF sourcelist at {sourcelist_fits}"
    assert sourcelist_reg.exists(), f"Could not find PyBDSF region file at {sourcelist_reg}"
    assert rmsmap.exists(), f"Could not find PyBDSF rms map at {rmsmap}"

    # attempt to find filtered sourcelists
    sourcelist_fits_filtered = list(cube_imaging_workdir.glob('sourcelist_filtered_within_*_deg.fits'))
    sourcelist_reg_filtered = list(cube_imaging_workdir.glob('sourcelist_filtered_within_*_deg.reg'))

    if len(sourcelist_fits_filtered) == 1:
        sourcelist_fits_filtered = sourcelist_fits_filtered[0]
    else:
        raise ValueError(f"Could not uniquely identify filtered PyBDSF sourcelist in {cube_imaging_workdir}. Found: {sourcelist_fits_filtered}")
    
    if len(sourcelist_reg_filtered) == 1:
        sourcelist_reg_filtered = sourcelist_reg_filtered[0]
    else:
        raise ValueError(f"Could not uniquely identify filtered PyBDSF region file in {cube_imaging_workdir}. Found: {sourcelist_reg_filtered}")
    
    return sourcelist_fits_filtered, sourcelist_reg_filtered


def _wrap_angle_deg(chi_deg: np.ndarray) -> np.ndarray:
    """
    Wrap polarization angle to [-90, 90) deg to avoid 180-deg jumps.
    """
    return ((np.asarray(chi_deg) + 90.0) % 180.0) - 90.0


def _get_option(opts, key, default=None):
    """Access option whether `opts` is a dict or a dataclass-like object."""
    try:
        return opts[key]
    except Exception:
        return getattr(opts, key, default)


