"""

Modified from Joppe Swart by Erik Osinga

This script calculates the primary beam corrections for each channel.

"""
from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path

import numpy as np
from astropy.io import fits

# assumes pip install katbeam
from katbeam import JimBeam


def showbeam(beam,freqMHz,pol,beam_extend, i, size, hdul, outdir='./'):
    """
    showbeam: This function makes the primary beam per channel using JimBeam
    INPUTS:
        beam: Standard beam package from JimBeam.
        sourcename: Name of the data block of interest.
        freqMHz: Frequency in MHz where we want the beam.
        pol: Polarization for which we want the beam.
        beam_extend: Width of the image.
        i: Index for the file name.
        hdul: template hdul
    OUTPUTS:
        fits file containing the primary beam.
    """
    # print(f"Calculating primary beam for Stokes {pol} at {freqMHz} MHz.")

    margin=np.linspace(-beam_extend/2.,beam_extend/2.,size)
    x,y=np.meshgrid(margin,margin) # distance from pointing centre in deg
    if pol=='H':
        beampixels=beam.HH(x,y,freqMHz)
    elif pol=='V':
        beampixels=beam.VV(x,y,freqMHz)
    elif pol =="I":
        beampixels=beam.I(x,y,freqMHz)


    # Save output
    outfile = f"{outdir}/{i:04d}-{pol}-pb_model.fits"
    # print(f"Saving output file {outfile}")
    hdulshape = hdul[0].data.shape
    # Make sure its same shape
    beampixels = np.reshape(beampixels, hdulshape)
    hdul[0].data = beampixels
    hdul.writeto(outfile, overwrite=True)
    return beampixels

def calculate_pb(globstr, band='L', outdir='./', verbose=False):
    """
    calculate_pb: Calculate the primary beam per channel
    INPUTS:
        globstr -- str -- filename of the channel images
        band    -- str -- "L" or "UHF"

    NOTE THAT DIFFERENT CHANNELS-OUT WILL REQUIRE RE-RUNNING THIS SCRIPT
    """
    if band == "L":
        beammodel=JimBeam('MKAT-AA-L-JIM-2020') # model for L-band
    elif band == "UHF":
        beammodel = JimBeam('MKAT-AA-UHF-JIM-2020') # model for UHF band
    else:
        raise ValueError(f"Band {band} beam not implemented")

    filenames = sorted(glob(globstr))

    # Get image parameters from first file 
    with fits.open(filenames[0]) as hdul1:
        I_hdr = hdul1[0].header
        size = I_hdr['NAXIS1']# assumes square
        scale = np.abs(I_hdr['CDELT1']) # assumes square
        nchannels = len(filenames)
        beam_extend = size*scale # width of image in degrees


        if verbose:
            print(f"Calculating the primary beam for {nchannels} channels")
            print("Using parameters")
            print(f"{size=} pixels")
            print(f"{scale=} deg = {scale*3600} arcsec")
            print(f"image width = {beam_extend:.2f} deg")
            print(f"{nchannels=}")

        for i, i_filename in enumerate(filenames):
            if "MFS" in i_filename:
                if verbose:
                    print(f"Skipping MFS image {i_filename}")
                continue

            # print(f'Calculating the primary beam corrections on channel {i}')
            I_hdr = fits.getheader(i_filename)
            # also use hdul as template for writing PB images (same freq)
            with fits.open(i_filename) as hdul:

                if "FREQ" in I_hdr["CTYPE3"]:
                    freq = I_hdr['CRVAL3']/1e6 #convert to MHz

                    if band == "L":
                        if not (900 <= freq <= 1670):
                            raise ValueError(f"Frequency {freq} MHz outside L-band range")
                    elif band == "UHF":
                        if not (580 <= freq <= 1015):
                            raise ValueError(f"Frequency {freq} MHz outside UHF-band range")

                else:
                    raise ValueError("please update script or set 3rd axis as freq")

                if not os.path.exists(f"{i:04d}-I-pb_model.fits"):

                    # Shape (1,1,8192,8192)
                    showbeam(beammodel,freq,'I',beam_extend, i, size, hdul, outdir)

    if verbose:
        print('primary beam corrections calculated, apply the corrections with the apply script')
        print(f"Saved in {outdir}/*-I-pb_model.fits")

def extract_last_four_digits(filepath: str) -> str:
    """
    Extract the last occurrence of a four-digit sequence from the filename.

    Parameters
    ----------
    filepath : str
        Path to the file (e.g. '.../image-0001-0002.fits').

    Returns
    -------
    str
        The last four-digit channel number found in the filename.

    Raises
    ------
    ValueError
        If no four-digit sequence is found.
    """
    filename = os.path.basename(filepath)
    # Find all non-overlapping 4-digit sequences
    matches = re.findall(r'\d{4}', filename)
    if not matches:
        raise ValueError(f"No four-digit channel number found in '{filepath}'")
    return matches[-1]

def pbcor_allchan(globstr_original: str, globstr_pbcor: str, verbose: bool = False) -> tuple[list[Path], list[Path]]:
    """
    Call pbcor many times
    """
    original_files = sorted(glob(globstr_original))
    pbcor_files = sorted(glob(globstr_pbcor))

    if verbose:
        print(f"original_files={original_files[:10]}")
        print(f"original_files={pbcor_files[:10]}")

    all_corrected = []
    all_pbcor = []
    for i, (original, pbcor) in enumerate(zip(original_files,pbcor_files)):
        if "MFS" in original:
            if verbose:
                print(f"Skipping MFS image {original}")
            continue

        # Make sure we are correcting the same channel number
        channum_original = extract_last_four_digits(original)
        channum_pbcor = extract_last_four_digits(pbcor)
        assert channum_original == channum_pbcor, f"Found channel {channum_original} vs pbcor channel {channum_pbcor}"

        pbcorrected_file = do_pbcor(original, pbcor, verbose=verbose)
        all_corrected.append(Path(pbcorrected_file))
        all_pbcor.append(Path(pbcor))

    return all_corrected, all_pbcor


def do_pbcor(original: str, pbcor: str, verbose=False) -> str:
    """
    Given an original fits image and a pbcor image, create a new image
    .pbcor.fits that is original/pbcor. I.e. creates pbcorrected image.

    This corrects stokes I
    """

    if verbose:
        print (f"Making pb corrected images for {original}")
        print(f"Using {pbcor} for the primary beam")

    with fits.open(original) as original_hdu:
        size = np.shape(original_hdu[0].data)[-1]

        if verbose:
            print(f"Found image size to be {size}")

        outfile = original.replace(".fits",".pbcor.fits")

        with fits.open(pbcor) as pbcorhdu:
            # Make sure we are correcting at the same frequency
            freq_original = original_hdu[0].header['CRVAL3']
            freq_pbcor = pbcorhdu[0].header['CRVAL3']
            if not np.allclose(freq_original, freq_pbcor):
                raise ValueError(f"Found freq {freq_original} but pbcor freq {freq_pbcor}. Make sure you are giving the corresponding pbcor images.")

            pbarraydata = pbcorhdu[0].data
            original_hdu[0].data /= pbarraydata

            if verbose:
                print(f"Writing to {outfile}")
            original_hdu.writeto(outfile, overwrite=True)

    return outfile



if __name__ == '__main__':


    # path to channel images
    imagedir = "/path/to/IQU_combined/"
    globstr = f"{imagedir}/Abell754-combined-I_imaging-0*image.fits"
    # path to save pb models
    outdir = './pbcor170chan/'

    if not os.path.exists(f"{outdir}"):
        print(f"mkdir {outdir}")
        os.system(f"mkdir -p {outdir}")

    calculate_pb(globstr, band='L', outdir=outdir)

    globstr_pbcor = f"{outdir}/*-I-pb_model.fits"

    pbcor_allchan(globstr, globstr_pbcor, verbose=False)