from __future__ import annotations

from pathlib import Path

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table


def correct_galactic_rm_hutschenreuter(results: Path, rmmap_huts: Path) -> Path:
    """
    Remove the Galactic contribution to Faraday rotation, using the map from Hutschenreuter2022.

    Returns a path to the new table ".corrected.fits" with three new columns:
        'grm_huts' : Galactic RM from Hutschenreuter map
        'rrm_huts' : Residual RM after subtracting Galactic RM
        'rrm_huts_err' : Error on residual RM, adding in quadrature the measurement error and the Galactic RM uncertainty
    """
    outpath = results.with_name(results.stem + '.corrected.fits')

    results: Table = Table.read(str(results))

    ra = results['ra']
    dec = results['dec']
    sc = SkyCoord(ra,dec,unit=(u.deg,u.deg),frame='icrs')
    
    nest = False # usng RING ordering, not NESTED pixel ordering.    
    nside = 512 # Hutschenreuter map nside

    hdu = fits.open(rmmap_huts)
    # Array with value of every pixel. Faraday depth 
    rmmap = hp.read_map(hdu[1],field=0,dtype=float,nest=nest)
    # Faraday depth scatter uncertainty
    rmunc = hp.read_map(hdu[1],field=1,dtype=float,nest=nest)    

    # Need galactic longitude and latitude.
    sc_galactic = sc.icrs.galactic
    
    # Find for every coordinate, the correction value
    rmcorrection = np.zeros(len(results))
    rmcorrection_err = np.zeros(len(results))
    for i, coord in enumerate(sc_galactic):
        gal_l = coord.l.value #lon 
        gal_b = coord.b.value #lat
          
        # get pixel index of galactic coordinates lon,lat. 
        x = hp.ang2pix(nside,gal_l, gal_b, nest, lonlat=True)
        rmcorrection[i]     = rmmap[x]
        rmcorrection_err[i] = rmunc[x] 
    
    results['grm_huts'] = rmcorrection
    results['rrm_huts'] = results['rm'] - rmcorrection
    # Add error in quadrature. Only a good assumption if both errors are independent and Gaussian
    results['rrm_huts_err'] = np.sqrt(results['rm_err']**2 + rmcorrection_err**2)

    results.write(str(outpath), overwrite=True)
    
    return outpath