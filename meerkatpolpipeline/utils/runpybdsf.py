from __future__ import annotations

import argparse
from pathlib import Path

import bdsf
from astropy import units as u
from astropy.io import fits

from meerkatpolpipeline.utils.utils import PrintLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run PyBDSF on a Stokes I FITS image to extract source catalogues.'
    )
    # required positional
    parser.add_argument(
        'filename',
        type=Path,
        help='Path to the input FITS file (Stokes I image).'
    )
    # adaptive_rms_box flag, default True
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--adaptive-rms-box',
        dest='adaptive_rms_box',
        action='store_true',
        help='Enable adaptive rms box (default).'
    )
    group.add_argument(
        '--no-adaptive-rms-box',
        dest='adaptive_rms_box',
        action='store_false',
        help='Disable adaptive rms box.'
    )
    parser.set_defaults(adaptive_rms_box=True)
    # rms_box_bright tuple
    parser.add_argument(
        '--rms-box-bright',
        nargs=2,
        type=int,
        default=[60, 15],
        metavar=('NP', 'NB'),
        help='Dimensions for rms_box_bright as two ints: NP NB (default: 60 15).'
    )
    parser.add_argument(
        '--outdir',
        nargs=1,
        type=Path,
        default=Path("./pybdsf/"),
        help='output dir'
    )    
    return parser.parse_args()


def _runpybdsf(
    outdir: Path,
    filename: Path,
    adaptive_rms_box: bool,
    rms_box_bright: tuple[int] = [60, 15],
    verbose: bool = False,
    logger=None,
) -> tuple[Path, Path, Path]:
    """
    run PyBDSF on a Stokes I FITS image to extract source catalogues.
    Args:
        outdir (str): Output directory for catalogs and RMS map.
        filename (str): Path to the input FITS file (Stokes I image).
        adaptive_rms_box (bool): Whether to use adaptive rms box.
        rms_box_bright (tuple): Tuple of two ints for rms_box_bright parameter.
    Returns:
        tuple: (sourcelist_fits, sourcelist_reg, rmsmap) paths.
    """
    if logger is None:
        logger = PrintLogger()

    outdir.mkdir(exist_ok=True)
    # --- prepare output paths ---
    sourcelist_fits = outdir / 'sourcelist.srl.fits'
    sourcelist_reg = outdir / 'sourcelist.srl.reg'
    rmsmap = outdir / 'rms_map.fits'

    if sourcelist_fits.exists() and sourcelist_reg.exists() and rmsmap.exists():
        logger.info(f"Output files already exist in {outdir}, skipping PyBDSF run.")
        return sourcelist_fits, sourcelist_reg, rmsmap
    else:
        logger.info(f"PYBDSF output files will be written to {outdir}.")

    # --- open FITS and compute beam major axis in arcsec ---
    with fits.open(str(filename)) as hdul:
        head = hdul[0].header
        bmaj = (head['BMAJ'] * u.deg).to(u.arcsec).value
    
    if verbose:
        logger.info(f'Beam major axis: {bmaj:.2f} arcsec')
    if bmaj == 0:
        raise ValueError(f"Invalid value for {bmaj=} in {filename}")

    # --- run PyBDSF ---
    if verbose:
        logger.info(f"====> Running PYBDSF on the image {filename} to get source locations and RMS map.")

    img = bdsf.process_image(
        filename,
        adaptive_rms_box=adaptive_rms_box,
        rms_box=(150, 50), # not used if adaptive_rms_box=True
        rms_box_bright=tuple(rms_box_bright),
        polarisation_do=False,
        rms_map=True,
        thresh_pix=5.0,
        thresh_isl=3.0,
    )
    
    img.write_catalog(outfile=str(sourcelist_fits),
                      catalog_type='srl',
                      format='fits',
                      clobber=True)
    img.write_catalog(outfile=str(sourcelist_reg),
                      catalog_type='srl',
                      format='ds9',
                      clobber=True)

    img.export_image(img_type='rms', outfile=str(rmsmap))

    logger.info(f'PYBDSF Catalogs written to {outdir}')
    
    return sourcelist_fits, sourcelist_reg, rmsmap


def main():
    args = parse_args()

    _runpybdsf(args.outdir, args.filename, args.adaptive_rms_box, args.rms_box_bright)

if __name__ == '__main__':
    # run from command line
    main()
