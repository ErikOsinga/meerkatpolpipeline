#!/usr/bin/env python3
"""
beam_info_vs_freq.py

Plot restoring beam axes (BMAJ & BMIN) vs frequency for Stokes I and Q images.

Importable API:
- generate_beam_plots(i_input: str, q_input: str, output_dir: str, ...)
- extract_beam_info(pattern: str) -> BeamData
- plot_beam_axes(...) -> list[pathlib.Path]

CLI:
python beam_info_vs_freq.py --i_input './IQU_combined/*I_imaging-0*-image.pbcor.fits' \
                                --q_input './IQU_combined/*QU_imaging-0*-Q-image.pbcor.fits' \
                                --output_dir './plots'
"""
from __future__ import annotations

import argparse
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits

__all__ = [
    "BeamData",
    "extract_beam_info",
    "extract_beam_info_from_files",
    "plot_beam_axes",
    "generate_beam_plots",
    "parse_args",
    "main",
]


@dataclass(frozen=True)
class BeamData:
    """Container for frequency and beam axes.

    Attributes
    ----------
    freq_hz : np.ndarray
        Frequencies in Hz.
    bmaj_deg : np.ndarray
        Beam major axis in degrees.
    bmin_deg : np.ndarray
        Beam minor axis in degrees.
    files : tuple[str, ...]
        The file list corresponding to the measurements (sorted by frequency).
    """
    freq_hz: np.ndarray
    bmaj_deg: np.ndarray
    bmin_deg: np.ndarray
    files: tuple[str, ...]


def _to_beam_data(file_input: str | list[Path]) -> BeamData:
    """Normalize user input into BeamData.

    Accepts either:
    - a glob pattern (str), or
    - a concrete list/tuple of Path objects.
    """
    if isinstance(file_input, str):
        return extract_beam_info(file_input)

    # assume an iterable of Path-like objects
    files = [str(p) for p in file_input]
    if not files:
        raise FileNotFoundError("No files provided (empty list/tuple).")
    return extract_beam_info_from_files(files)

def extract_beam_info(pattern: str) -> BeamData:
    """Read FITS files matching the glob pattern and extract frequency, BMAJ, BMIN.

    Parameters
    ----------
    pattern : str
        Glob pattern for FITS files.

    Returns
    -------
    BeamData

    Raises
    ------
    FileNotFoundError
        If no files matched the pattern.
    KeyError
        If required header keywords are missing.
    """
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        raise FileNotFoundError(f"No FITS files matched pattern: {pattern}")
    return extract_beam_info_from_files(file_list)


def extract_beam_info_from_files(files: Sequence[str]) -> BeamData:
    """Extract frequency (Hz), BMAJ (deg), and BMIN (deg) from a sequence of FITS files.

    Notes
    -----
    - Expects spectral axis reference value in 'CRVAL3' (Hz).
    - Expects 'BMAJ' and 'BMIN' (deg) in headers.

    Returns
    -------
    BeamData
        Sorted by frequency ascending.

    Raises
    ------
    KeyError
        If required header keywords are missing in any file.
    """
    freqs: list[float] = []
    bmaj_vals: list[float] = []
    bmin_vals: list[float] = []
    for fn in files:
        header = fits.getheader(fn, memmap=True)

        freq = header.get("CRVAL3")
        if freq is None:
            raise KeyError(f"CRVAL3 not found in header of {fn}")

        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")
        if bmaj is None or bmin is None:
            raise KeyError(f"BMAJ/BMIN missing in header of {fn}")

        freqs.append(float(freq))
        bmaj_vals.append(float(bmaj))
        bmin_vals.append(float(bmin))

    freq_arr = np.asarray(freqs, dtype=float)
    bmaj_arr = np.asarray(bmaj_vals, dtype=float)
    bmin_arr = np.asarray(bmin_vals, dtype=float)

    # Sort by frequency for consistent plotting
    order = np.argsort(freq_arr)
    sorted_files = tuple(str(files[i]) for i in order)

    return BeamData(
        freq_hz=freq_arr[order],
        bmaj_deg=bmaj_arr[order],
        bmin_deg=bmin_arr[order],
        files=sorted_files,
    )


def plot_beam_axes(
    data: BeamData,
    stokes: str,
    output_dir: str | Path,
    *,
    yline_arcsec: float | None = 15.0,
    ylim_arcsec: tuple[float, float] | None = (-5.0, 30.0),
    show: bool = False,
    zoom_suffix: str = ".zoom",
) -> list[Path]:
    """Plot BMAJ and BMIN vs frequency for a given Stokes parameter and save figures.

    Parameters
    ----------
    data : BeamData
        Data to plot (Hz and deg).
    stokes : str
        Stokes parameter, e.g., 'I' or 'Q'.
    output_dir : str or Path
        Directory where figures will be written (created if absent).
    yline_arcsec : float or None, default 15.0
        Draw a horizontal reference line at this value (arcsec). Pass None to disable.
    ylim_arcsec : tuple[float, float] or None, default (-5, 30)
        Y-limits for a second "zoom" figure. Pass None to skip the zoom figure.
    show : bool, default False
        If True, show the plots interactively. False is recommended for library usage.
    zoom_suffix : str, default ".zoom"
        Suffix inserted before the extension for the zoomed figure.

    Returns
    -------
    list[pathlib.Path]
        Paths of the saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert with astropy.units for clarity and unit-safety
    freqs_ghz = (data.freq_hz * u.Hz).to(u.GHz).value
    bmaj_arcsec = (data.bmaj_deg * u.deg).to(u.arcsec).value
    bmin_arcsec = (data.bmin_deg * u.deg).to(u.arcsec).value

    saved: list[Path] = []

    # Full-range figure
    fig, ax = plt.subplots()
    ax.plot(freqs_ghz, bmaj_arcsec, marker="o", linestyle="-", label="BMAJ [arcsec]")
    ax.plot(freqs_ghz, bmin_arcsec, marker="s", linestyle="--", label="BMIN [arcsec]")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Beam axis [arcsec]")
    ax.set_title(f"Stokes {stokes}: BMAJ & BMIN vs Frequency")
    ax.grid(True)

    if yline_arcsec is not None:
        ax.axhline(yline_arcsec, label=f"{yline_arcsec} arcsec", ls="dashed", color="k")

    ax.legend(loc="best")

    out_full = output_dir / f"{stokes}_beam_info_vs_freq.png"
    fig.savefig(out_full, dpi=150, bbox_inches="tight")
    saved.append(out_full)

    if show:
        plt.show()
    plt.close(fig)

    # Zoom figure (optional)
    if ylim_arcsec is not None:
        fig2, ax2 = plt.subplots()
        ax2.plot(freqs_ghz, bmaj_arcsec, marker="o", linestyle="-", label="BMAJ [arcsec]")
        ax2.plot(freqs_ghz, bmin_arcsec, marker="s", linestyle="--", label="BMIN [arcsec]")
        ax2.set_xlabel("Frequency [GHz]")
        ax2.set_ylabel("Beam axis [arcsec]")
        ax2.set_title(f"Stokes {stokes}: BMAJ & BMIN vs Frequency (Zoom)")
        ax2.grid(True)
        if yline_arcsec is not None:
            ax2.axhline(yline_arcsec, label=f"{yline_arcsec} arcsec", ls="dashed", color="k")
        ax2.legend(loc="best")
        ax2.set_ylim(*ylim_arcsec)

        out_zoom = output_dir / f"{stokes}_beam_info_vs_freq{zoom_suffix}.png"
        fig2.savefig(out_zoom, dpi=150, bbox_inches="tight")
        saved.append(out_zoom)

        if show:
            plt.show()
        plt.close(fig2)

    logging.info("Saved %d figure(s) for Stokes %s to %s", len(saved), stokes, output_dir)
    return saved


def generate_beam_plots(
    i_input: str | list[Path],
    q_input: str | list[Path],
    output_dir: str | Path,
    *,
    yline_arcsec: float | None = 15.0,
    ylim_arcsec: tuple[float, float] | None = (-5.0, 30.0),
    show: bool = False,
) -> dict[str, list[Path]]:
    """High-level API to generate beam plots for Stokes I and Q.

    Args:
        i_input : str or list[Path]
            Glob pattern for Stokes I FITS files
            OR a list of Path objects.
        q_input : str
            Glob pattern for Stokes Q FITS files
            OR a list of Path objects
        output_dir : str or Path
            Directory where the beam plots will be saved.
        yline_arcsec : float or None, default 15.0.
            Draw a horizontal reference line at this value (arcsec). Pass None to disable.
        ylim_arcsec : tuple[float, float] or None, default (-5.0, 30.0).
            Y-limits (arcsec) for the zoomed figure; ignored if None.
        show : bool, default False

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Mapping from stokes label to list of saved figure paths.

    e.g. 
        {'I': [PosixPath('tests/I_beam_info_vs_freq.png')],
        'Q': [PosixPath('tests/Q_beam_info_vs_freq.png')]}



    """
    data_i = _to_beam_data(i_input)
    data_q = _to_beam_data(q_input)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Path]] = {}
    results["I"] = plot_beam_axes(
        data_i, "I", output_dir, yline_arcsec=yline_arcsec, ylim_arcsec=ylim_arcsec, show=show
    )
    results["Q"] = plot_beam_axes(
        data_q, "Q", output_dir, yline_arcsec=yline_arcsec, ylim_arcsec=ylim_arcsec, show=show
    )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot restoring beam axes (BMAJ & BMIN) vs frequency for Stokes I and Q images"
    )
    parser.add_argument(
        "--q_input",
        required=True,
        help="Glob pattern for Stokes Q FITS files, e.g. './IQU_combined/*QU_imaging-0*-Q-image.pbcor.fits'",
    )
    parser.add_argument(
        "--i_input",
        required=True,
        help="Glob pattern for Stokes I FITS files, e.g. './IQU_combined/*I_imaging-0*-image.pbcor.fits'",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the beam plots will be saved",
    )
    parser.add_argument(
        "--no-zoom",
        action="store_true",
        help="Disable the zoomed figure with custom y-limits",
    )
    parser.add_argument(
        "--yline_arcsec",
        type=float,
        default=15.0,
        help="Draw a horizontal reference line at this value (arcsec). Use a negative number to disable.",
    )
    parser.add_argument(
        "--ylim_arcsec",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=(-5.0, 30.0),
        help="Y-limits (arcsec) for the zoomed figure; ignored if --no-zoom is set",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s: %(message)s",
    )

    yline = None if (args.yline_arcsec is not None and args.yline_arcsec < 0) else args.yline_arcsec
    ylim = None if args.no_zoom else tuple(args.ylim_arcsec)

    outdir = Path(args.output_dir)

    results = generate_beam_plots(
        i_input=args.i_input,
        q_input=args.q_input,
        output_dir=outdir,
        yline_arcsec=yline,
        ylim_arcsec=ylim,
        show=args.show,
    )

    for stokes, paths in results.items():
        for p in paths:
            logging.info("Stokes %s figure saved: %s", stokes, p)


if __name__ == "__main__":
    main()