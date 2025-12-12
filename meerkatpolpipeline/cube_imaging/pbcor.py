"""

Modified from Joppe Swart by Erik Osinga

This script calculates the primary beam corrections for each channel.

"""
from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import numpy as np
from astropy.io import fits

# assumes pip install katbeam
from katbeam import JimBeam


def calculate_and_write_primary_beam(beam, freqMHz, pol, beam_extend, i, size, hdul, outfile=None):
    """
    calculate_and_write_primary_beam: This function makes the primary beam per channel using JimBeam
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
    if outfile is not None:
        # print(f"Saving output file {outfile}")
        hdulshape = hdul[0].data.shape
        # Make sure its same shape
        beampixels = np.reshape(beampixels, hdulshape)
        hdul[0].data = beampixels
        hdul.writeto(outfile, overwrite=True)

    return beampixels

def count_n_channels(filenames: list[Path]) -> int:
    """Count number of channels in a list of filenames from a globstr"""
    nchannels = 0
    for f in filenames:
        if "MFS" in f.name:
            continue
        else:
            nchannels += 1
    return nchannels


def calculate_pb(globstr: str, band: str ='L', outdir: Path = Path('./'), verbose: bool = False) -> None:
    """
    calculate_pb: Calculate the primary beam per channel
    INPUTS:
        globstr -- str -- filename of the channel images
        band    -- str -- "L" or "UHF"
        outdir  -- path -- directory to save the pb modelss

    NOTE THAT DIFFERENT CHANNELS-OUT WILL REQUIRE RE-RUNNING THIS SCRIPT BECAUSE CHANNEL NUMBERS WILL BE DIFFERENT
    """
    if band.upper() == "L":
        beammodel=JimBeam('MKAT-AA-L-JIM-2020') # model for L-band
    elif band.upper() == "UHF":
        beammodel = JimBeam('MKAT-AA-UHF-JIM-2020') # model for UHF band
    else:
        raise ValueError(f"Band {band} beam not implemented")

    filenames = sorted(glob(globstr)) # sorted makes sure channels are sorted and MFS is last one
    filenames = [Path(f) for f in filenames]
    nchannels = count_n_channels(filenames)

    if nchannels == 0:
        raise ValueError(f"Did not find any files with glob {globstr}")


    # Get image parameters from first file 
    with fits.open(filenames[0]) as hdul1:
        I_hdr = hdul1[0].header
        size = I_hdr['NAXIS1']# assumes square
        scale = np.abs(I_hdr['CDELT1']) # assumes square
        beam_extend = size*scale # width of image in degrees

        if verbose:
            print(f"Calculating the primary beam for {nchannels} channels")
            print("Using parameters")
            print(f"{size=} pixels")
            print(f"{scale=} deg = {scale*3600} arcsec")
            print(f"image width = {beam_extend:.2f} deg")
            print(f"{nchannels=}")

    for i, i_filename in enumerate(filenames):
        if "MFS" in i_filename.name:
            assert i == nchannels, f"MFS image should be last in the list. Please check list of files: {filenames}"

            i="MFS" # string for MFS image

            outfile = outdir / "MFS-I-pb_model.fits"

        else:

            outfile = outdir / f"{i:04d}-I-pb_model.fits"

        # print(f'Calculating the primary beam corrections on channel {i}')
        I_hdr = fits.getheader(str(i_filename))
        # also use hdul as template for writing PB images (same freq)
        with fits.open(str(i_filename)) as hdul:

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

            
            calculate_and_write_primary_beam(beammodel,freq,'I',beam_extend, i, size, hdul, outfile)

    if verbose:
        print('primary beam corrections calculated, apply the corrections with the apply script')
        print(f"Saved in {outdir}/*-I-pb_model.fits")

    return None


def extract_last_four_digits(filepath: Path) -> str:
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
    filename = filepath.name
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

    original_files = [Path(f) for f in original_files]
    pbcor_files = [Path(f) for f in pbcor_files]
    if len(original_files) != len(pbcor_files):
        raise ValueError(f"Found {len(original_files)} original files but {len(pbcor_files)} pbcor files. Make sure you have the corresponding pbcor images.")

    if verbose:
        print(f"original_files={original_files[:10]}")
        print(f"original_files={pbcor_files[:10]}")

    all_corrected = []
    all_pbcor = []
    for i, (original, pbcor) in enumerate(zip(original_files,pbcor_files)):
        if "MFS" in original.name:
            # Make sure we are correcting at the same frequency
            assert "MFS" in pbcor.name, f"Expected MFS in pbcor filename {pbcor} for original {original}"
        else:
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

        outfile = original.with_suffix(".pbcor.fits")

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
    # example usage

    # path to channel images
    imagedir = "/path/to/IQU_combined/"
    globstr = f"{imagedir}/Abell754-combined-I_imaging-0*image.fits"
    # path to save pb models
    outdir = Path('./pbcor170chan/')

    outdir.mkdirs(exist_ok=True)

    calculate_pb(globstr, band='L', outdir=outdir)

    globstr_pbcor = f"{outdir}/*-I-pb_model.fits"

    pbcor_allchan(globstr, globstr_pbcor, verbose=False)