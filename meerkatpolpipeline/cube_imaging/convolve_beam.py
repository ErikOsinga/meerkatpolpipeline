#!/usr/bin/env python3

"""
# Convolve fits image(s) to a target beam.
#
# Written in python3 by Andrea Botteon (andrea.botteon@inaf.it) and based on https://github.com/mhardcastle/ddf-pipeline/blob/offsets_compact/scripts/convolve.py
# 
# Adapted by Erik Osinga:

Convolve FITS image(s) to a target beam.

- Importable API:
    - BeamParams
    - flatten_to_2d(hdulist: fits.HDUList) -> fits.PrimaryHDU
    - read_beam_from_template(template: Path) -> BeamParams
    - convolve_to_beam(infile: Path, beam: BeamParams, outfile: Path | None = None) -> Path
    - convolve_images(inputs: list[Path] | Path, beam: BeamParams | None = None,
                      template: Path | None = None, output_dir: Path | None = None,
                      suffix_mode: str = "beam", overwrite: bool = True) -> list[Path]
    - main(argv: list[str] | None = None) -> None

- CLI example:
    python convolve_to_target_beam.py -beam 15.0 12.0 0.0 myimage.fits
    python convolve_to_target_beam.py -template target.fits image1.fits image2.fits
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
import scipy
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
from radio_beam import Beam

__all__ = [
    "BeamParams",
    "flatten_to_2d",
    "read_beam_from_template",
    "convolve_to_beam",
    "convolve_images",
    "parse_args",
    "main",
]


@dataclass(frozen=True)
class BeamParams:
    """Target restoring beam (arcsec, arcsec, deg)."""
    bmaj_arcsec: float
    bmin_arcsec: float
    bpa_deg: float

    def to_radio_beam(self) -> Beam:
        return Beam(
            major=self.bmaj_arcsec * u.arcsec,
            minor=self.bmin_arcsec * u.arcsec,
            pa=self.bpa_deg * u.deg,
        )

    def suffix(self) -> str:
        # Used for filenames when suffix_mode == "beam"
        return f"{self.bmaj_arcsec:.1f}x{self.bmin_arcsec:.1f}"


def flatten_to_2d(hdulist: fits.HDUList) -> fits.PrimaryHDU:
    """
    Flatten a FITS cube to a 2D image (taking the first plane along extra axes).
    Returns a new PrimaryHDU with a celestial WCS header.

    Notes
    -----
    - Uses WCS(header).celestial to remain compliant with redundant axes handling.
    - Copies common metadata: EQUINOX, EPOCH, BMAJ, BMIN, BPA, RESTFRQ, TELESCOP, OBSERVER.
    """
    h0 = hdulist[0]
    header = h0.header
    data = h0.data
    naxis = header.get("NAXIS", 0)

    if naxis < 2:
        raise ValueError(f"Cannot make 2D map: found {naxis} axis/axes.")

    if naxis == 2:
        # Already 2D; return a lightweight PrimaryHDU copy
        return fits.PrimaryHDU(header=header, data=data)

    # Build a purely celestial WCS
    w = WCS(header)
    wc = w.celestial  # IMPORTANT: comply with redundant axes as requested

    w2 = WCS(naxis=2)
    w2.wcs.crpix[0] = wc.wcs.crpix[0]
    w2.wcs.crpix[1] = wc.wcs.crpix[1]
    w2.wcs.cdelt = wc.wcs.cdelt[0:2]
    w2.wcs.crval = wc.wcs.crval[0:2]
    w2.wcs.ctype[0] = wc.wcs.ctype[0]
    w2.wcs.ctype[1] = wc.wcs.ctype[1]

    new_header = w2.to_header()
    new_header["NAXIS"] = 2

    for k in ("EQUINOX", "EPOCH", "BMAJ", "BMIN", "BPA", "RESTFRQ", "TELESCOP", "OBSERVER"):
        val = header.get(k)
        if val is not None:
            new_header[k] = val

    # Take the first index along any additional axes > 2
    # Build a selection tuple like data[0, 0, :, :] for NAXIS=4
    selection: list[object] = []
    for ax in range(naxis, 0, -1):
        if ax <= 2:
            selection.append(np.s_[:])
        else:
            selection.append(0)
    selection = tuple(selection)

    new_data = data[selection]
    return fits.PrimaryHDU(header=new_header, data=new_data)


def read_beam_from_template(template: Path) -> BeamParams:
    """Read target beam from template FITS header."""
    with fits.open(template, memmap=False) as t:
        bmaj_deg = t[0].header.get("BMAJ")
        bmin_deg = t[0].header.get("BMIN")
        bpa_deg = t[0].header.get("BPA")
        if bmaj_deg is None or bmin_deg is None or bpa_deg is None:
            raise KeyError(f"Template {template} missing BMAJ/BMIN/BPA.")
        return BeamParams(
            bmaj_arcsec=float(bmaj_deg) * 3600.0,
            bmin_arcsec=float(bmin_deg) * 3600.0,
            bpa_deg=float(bpa_deg),
        )


def convolve_to_beam(infile: Path, beam: BeamParams, outfile: Path | None = None,
                     overwrite: bool = True) -> Path:
    """
    Convolve a FITS image to the target beam. Returns the output Path.

    - If impossible (target smaller than current, or deconvolution fails), raises ValueError.
    - Output is saved as float32 (BITPIX=-32) to reduce size.
    """
    infile = Path(infile)
    if outfile is None:
        outfile = infile.with_suffix("")  # strip ".fits"
        outfile = outfile.with_name(f"{outfile.name}_{beam.suffix()}").with_suffix(".fits")

    with fits.open(infile, memmap=False) as hdul:
        # Prepare data and headers
        hdu2d = flatten_to_2d(hdul)
        old_beam = Beam.from_fits_header(hdul[0].header)

        if (old_beam.major <= 0 * u.deg) or (old_beam.minor <= 0 * u.deg):
            raise ValueError(f"{infile}: existing beam is zero.")

        target_beam = beam.to_radio_beam()

        # Try deconvolution (if impossible, radio_beam will raise a ValueError)
        try:
            dcb = target_beam.deconvolve(old_beam)
        except Exception as e:
            raise ValueError(f"{infile}: cannot convolve to a smaller/narrower beam; {e}")

        # Pixel scale (absolute value in arcsec/pixel; FITS CDELT can be negative)
        cdelt2 = hdul[0].header.get("CDELT2")
        if cdelt2 is None:
            # Try CD2_2 if CDELT2 not present
            cd2_2 = hdul[0].header.get("CD2_2")
            if cd2_2 is None:
                raise KeyError(f"{infile}: missing CDELT2/CD2_2 for pixel scale.")
            pixscale_arcsec = abs(float(cd2_2)) * 3600.0
        else:
            pixscale_arcsec = abs(float(cdelt2)) * 3600.0

        # Build convolution kernel
        kernel = dcb.as_kernel(pixscale_arcsec * u.arcsec)

        # FFT wrappers: use scipy FFT with all cores
        def _fftn(a):
            return scipy.fft.fftn(a, workers=-1)

        def _ifftn(a):
            return scipy.fft.ifftn(a, workers=-1)

        # Convolution (allow_huge for big images)
        smoothed = convolve_fft(
            hdu2d.data,
            kernel,
            allow_huge=True,
            fftn=_fftn,
            ifftn=_ifftn,
            boundary="fill",
            fill_value=0.0,
            preserve_nan=True,
        )

        # Flux rescaling by beam area ratio (Jy/beam images)
        rr = target_beam.sr / old_beam.sr
        smoothed = smoothed * rr.to_value()

        # Update header with target beam
        out_header = hdul[0].header.copy()
        out_header.update(target_beam.to_header_keywords())

    # Write output as float32
    out_hdu = fits.PrimaryHDU(np.asarray(smoothed, dtype=np.float32), header=out_header)
    out_hdu.writeto(outfile, overwrite=overwrite)
    return outfile


def convolve_images(
    inputs: list[Path] | Path,
    beam: BeamParams | None = None,
    template: Path | None = None,
    output_dir: Path | None = None,
    suffix_mode: str = "beam",
    overwrite: bool = True,
) -> list[Path]:
    """
    High-level API to convolve one or many images.

    Parameters
    ----------
    inputs : list[Path] | Path
        Single file or list of files.
    beam : BeamParams | None
        Target beam. Required unless template is provided.
    template : Path | None
        FITS file from which to read target beam (overrides beam if given).
    output_dir : Path | None
        If set, outputs are written to this directory. Otherwise next to inputs.
    suffix_mode : str
        "beam" -> append "_<bmaj>x<bmin>.fits";
        "keep" -> keep original name, only change directory.
    overwrite : bool
        Allow overwriting existing files.

    Returns
    -------
    list[Path]
        Paths of created files.
    """
    files: list[Path] = [inputs] if isinstance(inputs, Path) else list(inputs)
    if not files:
        raise FileNotFoundError("No input files provided.")

    if template is not None:
        target = read_beam_from_template(Path(template))
    else:
        if beam is None:
            raise ValueError("Provide either 'beam' or 'template'.")
        target = beam

    results: list[Path] = []
    for infile in files:
        infile = Path(infile)
        if not infile.exists():
            raise FileNotFoundError(f"Input not found: {infile}")

        if output_dir is None:
            if suffix_mode == "beam":
                outfile = infile.with_suffix("")
                outfile = outfile.with_name(f"{outfile.name}_{target.suffix()}").with_suffix(".fits")
            else:
                outfile = infile  # will overwrite in place if allowed
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if suffix_mode == "beam":
                outfile = output_dir / f"{infile.stem}_{target.suffix()}.fits"
            else:
                outfile = output_dir / infile.name

        try:
            out = convolve_to_beam(infile, target, outfile=outfile, overwrite=overwrite)
            results.append(out)
            logging.info("Created: %s", out)
        except Exception as e:
            logging.error("Failed: %s (%s)", infile, e)
            raise

    return results



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convolve FITS image(s) to a target beam.")
    p.add_argument(
        "-beam",
        nargs=3,
        type=float,
        metavar=("BMAJ_ARCSEC", "BMIN_ARCSEC", "BPA_DEG"),
        help="Target beam parameters (arcsec arcsec deg).",
    )
    p.add_argument(
        "-template",
        type=str,
        default=None,
        help="Template FITS whose BMAJ/BMIN/BPA define the target beam.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write outputs here (default: alongside inputs).",
    )
    p.add_argument(
        "--suffix_mode",
        choices=["beam", "keep"],
        default="beam",
        help='Filename mode: "beam" appends "_<bmaj>x<bmin>", "keep" preserves name.',
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    p.add_argument(
        "infiles",
        nargs="+",
        help="Input FITS file(s).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(levelname)s: %(message)s")

    beam: BeamParams | None = None
    if args.beam is not None:
        bmaj, bmin, bpa = args.beam
        beam = BeamParams(float(bmaj), float(bmin), float(bpa))

    template = Path(args.template) if args.template is not None else None
    inputs = [Path(s) for s in args.infiles]
    outdir = Path(args.output_dir) if args.output_dir is not None else None

    if template is None and beam is None:
        logging.error("You must provide either -beam or -template.")
        raise SystemExit(2)

    if template is not None:
        target = read_beam_from_template(template)
    else:
        target = beam  # type: ignore[assignment]

    logging.info(
        "Target beam: %.2f arcsec x %.2f arcsec, PA=%.2f deg",
        target.bmaj_arcsec,
        target.bmin_arcsec,
        target.bpa_deg,
    )

    created = convolve_images(
        inputs=inputs,
        beam=beam,
        template=template,
        output_dir=outdir,
        suffix_mode=args.suffix_mode,
        overwrite=args.overwrite,
    )

    for p in created:
        logging.info("Wrote %s", p)


if __name__ == "__main__":
    main()