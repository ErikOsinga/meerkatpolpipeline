#!/usr/bin/env python3
"""
nvss_cutout.py

Command-line tool and library to generate NVSS Stokes I/Q/U/P cutouts given RA/Dec.

Usage (CLI):
    python nvss_cutout.py --ra RA --dec DEC [--size_arcsec SIZE] [--nvss_dir DIR]
                          [--outfile PNG] [--outfile_fits FITS_BASE]

Importable:
    from nvss_cutout import get_nvss_cutouts, write_nvss_cutouts
    cutouts = get_nvss_cutouts(ra, dec, size_arcsec, nvss_dir)
    write_nvss_cutouts(cutouts, 'output_base')  # writes .I.fits, .p.fits, .Q.fits, .U.fits
"""
from __future__ import annotations

import argparse
import os
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS

# ignore ONLY the “invalid value encountered in divide” RuntimeWarning
warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in divide',
    category=RuntimeWarning,
    module='astropy\\.units\\.quantity'
)

# In the NVSS directory we expect this catalog filename
CATALOG_NAME: str = "p_I_1arcmin_I.cat"

# Internal cache for catalog and coordinates
_catalog: Table | None = None
_coords_p: SkyCoord | None = None

# NVSS synthesized beam parameters
BMAJ: float = 1.2500E-02
BMIN: float = 1.2500E-02
BPA: float = 0.0


def load_catalog(nvss_dir: str) -> tuple[Table, SkyCoord]:
    """
    Load the NVSS polarization catalog for fast tile lookup.

    Returns:
        tuple[Table, SkyCoord]: (Astropy Table with at least mosaic_p, radeg_p, decdeg_p,
                                 SkyCoord array of polarized positions)
    """
    global _catalog, _coords_p
    if _catalog is None or _coords_p is None:
        cat_path = os.path.join(nvss_dir, CATALOG_NAME)
        _catalog = Table.read(cat_path, format='ascii')
        _coords_p = SkyCoord(
            ra=_catalog['radeg_p'] * u.deg,
            dec=_catalog['decdeg_p'] * u.deg
        )
    return _catalog, _coords_p


def find_tile_file(stokes: str, ra: float, dec: float, nvss_dir: str) -> str:
    """
    Locate the FITS file for any Stokes plane using the nearest polarization catalog entry.

    Args:
        stokes: One of "I", "P", "Q", "U".
        ra: Right Ascension in decimal degrees.
        dec: Declination in decimal degrees.
        nvss_dir: Base NVSS directory path.

    Returns:
        str: Absolute path to the matching FITS file.
    """
    catalog, coords_p = load_catalog(nvss_dir)
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    idx, sep2d, _ = target.match_to_catalog_sky(coords_p)  # noqa: F841 (sep2d unused but informative)
    mosaic = catalog['mosaic_p'][idx]
    # File extension: 'p' for P, uppercase for I/Q/U
    ext = 'p' if stokes == 'P' else stokes
    filename = f"{mosaic}.{ext}.fits"
    fn = os.path.join(nvss_dir, f"stokes{stokes}", filename)
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Expected file {fn} for Stokes {stokes} not found")
    return fn


def load_cutout_array(
    stokes: str,
    ra: float,
    dec: float,
    size_arcsec: float,
    nvss_dir: str
) -> tuple[np.ndarray, WCS]:
    """
    Return a 2D cutout array and its WCS for the given Stokes plane.

    Args:
        stokes: One of "I", "P", "Q", "U".
        ra: Right Ascension in decimal degrees.
        dec: Declination in decimal degrees.
        size_arcsec: Size of the square cutout in arcseconds.
        nvss_dir: Base NVSS directory path.

    Returns:
        tuple[np.ndarray, WCS]: (cutout data, celestial WCS of the cutout)
    """
    fn = find_tile_file(stokes, ra, dec, nvss_dir)
    with fits.open(fn) as hdulist:
        data = np.squeeze(hdulist[0].data)
        header = hdulist[0].header
        wcs = WCS(header).celestial  # use celestial to be compliant with redundant axes
        # World -> pixel (origin=0 matches numpy indexing)
        x, y = wcs.all_world2pix(ra, dec, 0)
        # Pixel scale in arcsec/pixel (use astropy units for clarity)
        pix_scale_arcsec: float = (abs(wcs.wcs.cdelt[0]) * u.deg).to(u.arcsec).value
        size_pix = max(int(size_arcsec / pix_scale_arcsec), 1)
        # Be robust near edges: allow partial cutout and fill outside with NaN
        cut = Cutout2D(
            data,
            position=(x, y),
            size=(size_pix, size_pix),
            wcs=wcs,
            mode='partial',
            fill_value=np.nan
        )
        return cut.data, cut.wcs


def get_nvss_cutouts(
    ra: float,
    dec: float,
    size_arcsec: float,
    nvss_dir: str
) -> dict[str, tuple[np.ndarray, WCS]]:
    """
    Return a dict of cutout arrays and WCS objects for Stokes I, P, Q, and U.

    Args:
        ra: Right Ascension in decimal degrees.
        dec: Declination in decimal degrees.
        size_arcsec: Size of the square cutout in arcseconds.
        nvss_dir: Base NVSS directory path.

    Returns:
        dict[str, tuple[np.ndarray, WCS]]: {'I': (data, wcs), 'P': (data, wcs), 'Q': ..., 'U': ...}
    """
    cutouts: dict[str, tuple[np.ndarray, WCS]] = {}
    for st in ['I', 'P', 'Q', 'U']:
        cutouts[st] = load_cutout_array(st, ra, dec, size_arcsec, nvss_dir)
    return cutouts


def write_nvss_cutouts(cutouts: dict[str, tuple[np.ndarray, WCS]], outfile_fits_base: str) -> None:
    """
    Write separate FITS files for each Stokes plane from a cutouts dict.

    Args:
        cutouts: {'I': (data, wcs), 'P': (data, wcs), 'Q': (data, wcs), 'U': (data, wcs)}
        outfile_fits_base: Base filename (with or without .fits)
    """
    base, ext = os.path.splitext(outfile_fits_base)
    base_name = base if ext.lower() == '.fits' else outfile_fits_base
    for st in ['I', 'P', 'Q', 'U']:
        data_s, wcs_s = cutouts[st]
        ext_char = 'p' if st == 'P' else st
        outfn = f"{base_name}.{ext_char}.fits"
        
        # Ensure data has 3 axes: RA, Dec, Freq
        if data_s.ndim == 2:
            data_s = data_s[np.newaxis, :, :]  # Add a frequency axis

        hdu = fits.PrimaryHDU(data=data_s, header=wcs_s.to_header())
        hdu.header['BUNIT'] = 'Jy/beam'
        hdu.header['BMAJ'] = BMAJ
        hdu.header['BMIN'] = BMIN
        hdu.header['BPA'] = BPA
        # add NVSS defaults
        hdu.header['CTYPE3'] = 'FREQ'
        hdu.header['CRVAL3'] = 1.40000000000E+09
        hdu.header['CDELT3'] = 1.000000000E+08
        hdu.header['CRPIX3'] = 1.000000000E+00
        hdu.header['CROTA3'] = 0.000000000E+00
        # set correct number of WCSAXES
        hdu.header['WCSAXES'] = 3

        hdu.writeto(outfn, overwrite=True)
        print(f"Saved Stokes {st} FITS to {outfn}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NVSS Stokes I/Q/U/P cutouts around a given RA/DEC."
    )
    parser.add_argument('--ra', type=float, required=True, help='Right Ascension in decimal degrees')
    parser.add_argument('--dec', type=float, required=True, help='Declination in decimal degrees')
    parser.add_argument('--nvss_dir', type=str, required=True, help='NVSS data directory')
    parser.add_argument('--size_arcsec', type=float, default=500.0, help='Cutout size in arcseconds')
    parser.add_argument('--outfile', type=str, default='nvss_cutout.png', help='Output PNG filename')
    parser.add_argument('--outfile_fits', type=str, default=None, help='Optional base for FITS outputs')
    args = parser.parse_args()

    cutouts = get_nvss_cutouts(args.ra, args.dec, args.size_arcsec, args.nvss_dir)

    # Plot
    stokes_list: list[str] = ['I', 'P', 'Q', 'U']
    fig = plt.figure(figsize=(4 * len(stokes_list), 4))
    for i, st in enumerate(stokes_list):
        data, wcs = cutouts[st]
        ax = fig.add_subplot(1, len(stokes_list), i + 1, projection=wcs)
        im = ax.imshow(data, origin='lower')
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Intensity [Jy/beam]')
        ax.set_title(f'Stokes {st}')
        ax.coords.grid(True, color='white', linestyle='--')
    fig.tight_layout()
    fig.savefig(args.outfile, bbox_inches='tight', dpi=150)
    print(f"Saved cutout plot to {args.outfile}")

    if args.outfile_fits:
        write_nvss_cutouts(cutouts, args.outfile_fits)


if __name__ == '__main__':
    main()
