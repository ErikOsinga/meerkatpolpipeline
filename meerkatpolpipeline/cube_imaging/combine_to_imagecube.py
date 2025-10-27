#!/usr/bin/env python3
"""
combine_to_imagecube.py

Combine per-channel FITS images into a single frequency cube.

Importable API:
- parse_flag_chans(text: str) -> list[int]
- find_channel_number(filename: str) -> int
- normalize_file_input(file_input: FileInput) -> list[Path]
- build_cube_from_files(files: list[Path], nchan: int, width_mhz: float,
                        reference_chan0: Path, flag_chans: list[int]) -> tuple[np.ndarray, fits.Header]
- write_cube(cube: np.ndarray, header: fits.Header, output_path: Path, overwrite: bool = True) -> Path
- combine_to_cube(file_input: FileInput, reference_chan0: Path, output: Path,
                  nchan: int = 170, width_mhz: float = 5.0, flag_chans: list[int] | None = None,
                  overwrite: bool = True) -> Path

CLI:
    python combine_to_imagecube.py "I_*_imaging-0*-image.pbcor.fits" I_imaging-0000-image.pbcor.fits out.fits
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits

__all__ = [
    "parse_flag_chans",
    "find_channel_number",
    "normalize_file_input",
    "build_cube_from_files",
    "write_cube",
    "combine_to_cube",
    "parse_args",
    "main",
]

# Accept either a glob string or explicit list/tuple of Paths
FileInput = str | list[Path] | tuple[Path, ...]


def parse_flag_chans(text: str) -> list[int]:
    """
    Parse a string like "[5,6,44-101,128-168]" into a sorted, unique list of channel ints.
    Intended for CLI use but also reusable programmatically.
    """
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise argparse.ArgumentTypeError("flag-chans must be in brackets, e.g. [5,6,44-101]")

    items = s[1:-1].split(",")
    chans: set[int] = set()
    for item in items:
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid range '{item}'. Must be integers like 44-101.")
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid range '{item}'. Start must be <= end.")
            chans.update(range(start, end + 1))
        else:
            try:
                chans.add(int(item))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid channel '{item}'. Must be an integer.")
    return sorted(chans)


def find_channel_number(filename: str) -> int:
    """
    Extract the 4-digit channel number from a filename.
    Recognizes '_imaging-XXXX-Q-image', '_imaging-XXXX-U-image', or '_imaging-XXXX-image'.
    """
    if "-Q-image" in filename:
        pattern = r"-(\d{4})-Q-image"
    elif "-U-image" in filename:
        pattern = r"-(\d{4})-U-image"
    else:
        pattern = r"-(\d{4})-image"

    m = re.search(pattern, filename)
    if not m:
        raise ValueError(f"Cannot find channel number in '{filename}'")
    return int(m.group(1))


def normalize_file_input(file_input: FileInput) -> list[Path]:
    """Normalize input into a concrete, sorted list of Paths."""
    if isinstance(file_input, str):
        files = sorted(Path(p) for p in glob.glob(file_input))
    else:
        files = sorted(Path(p) for p in file_input)
    if not files:
        raise FileNotFoundError("No files matched / provided.")
    return files


def build_cube_from_files(
    files: list[Path],
    nchan: int,
    width_mhz: float,
    reference_chan0: Path,
    flag_chans: list[int],
    logger: logging.Logger = logging,
) -> tuple[np.ndarray, fits.Header]:
    """
    Build a (1, nchan, ny, nx) cube with channels from files, reference frequency from reference_chan0,
    and flagged channels set to NaN.
    """
    if not files:
        raise RuntimeError("No files provided to combine.")

    # Map channel index -> filename (ensure uniqueness)
    chan_to_file: dict[int, Path] = {}
    for fn in files:
        try:
            ch = find_channel_number(fn.name)
        except IndexError as e:
            logger.error("Error parsing channel number from filename '%s': %s", fn, e)
            raise ValueError(f"Cannot find channel number in '{fn}'") from e
        if ch < 0 or ch >= nchan:
            raise ValueError(f"Channel {ch} in '{fn}' out of expected range 0-{nchan-1}")
        if ch in chan_to_file:
            raise ValueError(f"Duplicate file for channel {ch}: {chan_to_file[ch]} and {fn}")
        chan_to_file[ch] = fn

    # Determine spatial size and dtype from the first channel present
    first_ch = min(chan_to_file)
    with fits.open(chan_to_file[first_ch]) as hdul0:
        data0 = hdul0[0].data
        if data0 is None or data0.ndim != 2:
            raise ValueError(f"{chan_to_file[first_ch]}: expected 2D image in primary HDU.")
        header0 = hdul0[0].header.copy()
        ny, nx = data0.shape
        dtype = data0.dtype

    # Initialize cube
    cube = np.full((1, nchan, ny, nx), np.nan, dtype=dtype)

    # Fill available channels
    for ch, fn in chan_to_file.items():
        with fits.open(fn) as hdul:
            img = hdul[0].data
            if img is None or img.shape != (ny, nx):
                raise ValueError(f"{fn}: shape mismatch; expected {(ny, nx)}, got {None if img is None else img.shape}")
            cube[0, ch] = img

    # Flag requested channels
    for ch in flag_chans:
        if 0 <= ch < nchan:
            cube[0, ch, :, :] = np.nan
        else:
            logger.warning("Flagged channel %d out of range (0-%d); ignored", ch, nchan - 1)

    # Frequency axis keywords
    hdr = header0
    hdr["NAXIS"] = 4
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nchan
    hdr["NAXIS4"] = 1
    hdr["CTYPE3"] = "FREQ"
    hdr["CUNIT3"] = "Hz"
    hdr["CDELT3"] = (width_mhz * u.MHz).to(u.Hz).value

    # Set reference frequency from channel 0 image
    with fits.open(reference_chan0) as hch0:
        if "CRVAL3" not in hch0[0].header:
            raise KeyError(f"{reference_chan0}: missing CRVAL3 for reference frequency.")
        ref_freq = float(hch0[0].header["CRVAL3"])

    hdr["CRVAL3"] = ref_freq
    hdr["CRPIX3"] = 1

    # Add a simple STOKES axis (I) for completeness
    hdr["CTYPE4"] = "STOKES"
    hdr["CUNIT4"] = ""
    hdr["CRVAL4"] = 1.0  # 1 = I
    hdr["CRPIX4"] = 1.0
    hdr["CDELT4"] = 1.0

    return cube, hdr


def write_cube(cube: np.ndarray, header: fits.Header, output_path: Path, overwrite: bool = True, logger=None) -> Path:
    """Write the data cube and header to a FITS file and return the output Path."""
    output_path = Path(output_path)
    hdu = fits.PrimaryHDU(data=cube, header=header)
    fits.HDUList([hdu]).writeto(output_path, overwrite=overwrite)
    if logger is not None:
        logger.info("Written cube: %s", output_path)
    return output_path


def combine_to_cube(
    file_input: FileInput,
    reference_chan0: Path,
    output: Path,
    nchan: int = 170,
    width_mhz: float = 5.0,
    flag_chans: list[int] | None = None,
    overwrite: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """
    High-level API: combine images into a cube.

    file_input: glob string or explicit list/tuple of Paths.
    reference_chan0: FITS image whose CRVAL3 defines the cube's reference frequency (channel 0).
    output: destination FITS path.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logging.basicConfig(level=getattr(logging, "INFO"), format="%(levelname)s: %(message)s")
        logger = logging

    if output.exists() and not overwrite:
        # Skip processing if output exists and overwrite is False
        logger.info(f"Output {output} exists and overwrite is False; skipping.")
        return output

    files = normalize_file_input(file_input)
    cube, header = build_cube_from_files(
        files=files,
        nchan=nchan,
        width_mhz=width_mhz,
        reference_chan0=Path(reference_chan0),
        flag_chans=flag_chans or [],
        logger=logger
    )
    return write_cube(cube, header, output, overwrite=overwrite, logger=logger)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine Stokes I images into a single cube.")
    p.add_argument("input_glob", help="Glob pattern matching the input FITS images")
    p.add_argument("input_chan0_image", help="FITS image for channel 0 (provides CRVAL3)")
    p.add_argument("output", help="Path to the output FITS cube")
    p.add_argument("--nchan", type=int, default=170, help="Number of channels in the final cube (default: 170)")
    p.add_argument("--width", type=float, default=5.0, help="Channel width in MHz (default: 5.0)")
    p.add_argument(
        "--flag-chans",
        dest="flag_chans",
        type=parse_flag_chans,
        default=[],
        help='Channels to flag, e.g. "[5,6,44-101,128-168]"',
    )
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(levelname)s: %(message)s")
    

    out = combine_to_cube(
        file_input=args.input_glob,
        reference_chan0=Path(args.input_chan0_image),
        output=Path(args.output),
        nchan=args.nchan,
        width_mhz=args.width,
        flag_chans=args.flag_chans,
        overwrite=args.overwrite,
        logger = logging,
    )
    logging.info("Done: %s", out)


if __name__ == "__main__":
    main()
