from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from configargparse import ArgumentParser
from prefect.logging import get_run_logger
from regions import Regions

from meerkatpolpipeline.utils.source_models import (
    EVPA_model_3c138,
    EVPA_model_3c286,
    polfrac_model_3c138,
    polfrac_model_3c286,
    stokesI_model_3c138,
    stokesI_model_3c286,
)
from meerkatpolpipeline.utils.utils import (
    convert_units,
    parse_comma_separated,
    read_fits_data_and_frequency,
    str2bool,
)
from meerkatpolpipeline.wsclean.wsclean import ImageSet, get_imset_from_prefix

KNOWN_CALIBRATORS = [
    '3c286', # polcal
    '3c138', # polcal
    'j0408-6545' # fluxcal
]


def determine_calibrator(region_file: Path) -> str:
    """
    Determine which calibrator is being processed based on the region file.
    
    Args:
        region_file (Path): Path to the region file, either 3c286.reg or 3c138.reg
    
    Returns:
        str: The name of the calibrator.
    """
    # Extract the name from the region file
    name = region_file.stem.split('_')[0]
    
    # Check if the name matches known calibrators
    if name.lower() in KNOWN_CALIBRATORS:
        return name.lower()
    else:
        raise ValueError(f"Unknown calibrator: {name}. Expected one of {KNOWN_CALIBRATORS}.")

def determine_model(src: str) -> tuple[Callable,Callable,Callable]:
    """
    return the model functions for the given source.
    Args:
        src (str): The name of the source, e.g. '3c286', '3c138', 'j0408-6545'.
    Returns:
        tuple[Callable, Callable, Callable]: A tuple of three functions for the model.
            1. Stokes I model: total intensity                (Jy) given input freq in Hz
            2. EVPA model    : electric field position angle  (degrees) given input freq in Hz
            3. polfrac_model : polarisation fraction          (unitless) given input freq in Hz
    """

    if src == "3c286":
        I_model    = stokesI_model_3c286
        EVPA_model = EVPA_model_3c286
        polfrac_model    = polfrac_model_3c286
    elif src == "3c138":
        I_model    = stokesI_model_3c138
        EVPA_model = EVPA_model_3c138
        polfrac_model    = polfrac_model_3c138
    elif src == "j0408-6545":
        print("TODO")
        # I_model    = stokesI_model_j0408
        # # J0408-6545 is unpolarised
        # EVPA_model = lambda freq: np.ones(len(freq))*np.nan # undefined
        # P_model = lambda freq: np.zeros(len(freq)) # zero polfrac
    else:
        raise ValueError(f"Unknown source '{src}'; must be 3c286 or 3c138 if -plotmodel is enabled")
    
    return I_model, EVPA_model, polfrac_model


def calculate_flux_and_peak_flux(filename: Path, region_file: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate total flux and peak flux for a source in a FITS file using a DS9 region file.
    
    Args:
        filename (Path): Path to the FITS file containing the image data.
        region_file (Path): Path to the DS9 region file defining the source region(s)
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - total_fluxes: Total flux density in Jy for each region in the region file.
            - peak_fluxes: Peak flux density in Jy/beam for each region in the region file.
            - freq: Frequency in Hz from the FITS file.

    """
    data, freq, wcs = read_fits_data_and_frequency(filename)
    
    # Load the DS9 region file
    regions = Regions.read(region_file)
    
    total_fluxes = []
    peak_fluxes = []

    for r in regions:
        # Convert region to pixel coordinates
        rpix = r.to_pixel(wcs.celestial)
        rmask = rpix.to_mask()

        # Mask the data and isolate the source
        masked_data = rmask.cutout(data)
        if masked_data is None:
            continue  # If the region is outside the image
        ellipsemask = np.array(rmask.data, dtype=bool)
        masked_data[np.invert(ellipsemask)] = 0  # Set outside of ellipse to 0
        
        # Calculate peak_flux in Jy/beam
        peak_flux = np.nanmax(masked_data)

        # Convert units from Jy/beam to Jy/pix
        masked_data = convert_units(masked_data, filename)

        # Calculate total flux in Jy
        total_flux = np.sum(masked_data)

        total_fluxes.append(total_flux)
        peak_fluxes.append(peak_flux)

    return np.array(total_fluxes), np.array(peak_fluxes), freq


def plot_total_intensity_spectrum(
    frequencieslist: np.ndarray,
    intensitieslist: np.ndarray,
    fit: bool = False,
    unc: np.ndarray | float | None = None,
    show: bool = True,
    label: str = 'data',
    marker: str = 'o',
    ylabel: str | None = None,
    plotdir: Path | None = None,
    verbose: bool = False,
) -> None:
    """Plot total intensity spectrum.
    
    # frequencieslist = (nchannels,)
    # intensitieslist = (nchannels,nregions)
    """
    for i in range(np.shape(intensitieslist)[1]):
        intensities = intensitieslist[:, i]
        # make sure values make sense
        mask = (intensities > 0) & (frequencieslist > 0)
        if (len(mask) - np.sum(mask)) > 0:
            if verbose:
                print(f"    Region {i}: Masking {len(mask) - np.sum(mask)} channels")
        frequencies = frequencieslist[mask]
        intensities = intensities[mask]

        if unc is not None:
            if not isinstance(unc, float):
                uc = unc[:, i]  # fractional RMS added
                uc = uc[mask]
            else:
                uc = unc  # uniform input user RMS

            plt.errorbar(frequencies, intensities, yerr=uc, ls='none', c=f'C{i}')

        plt.scatter(frequencies, intensities, label=label, c=f'C{i}', marker=marker)
        if fit and len(intensities) > 3:  # need at least 3 channels to fit
            # Perform log-log fitting: log(S) = alpha * log(nu)
            log_frequencies = np.log10(frequencies)
            log_intensities = np.log10(intensities)
            # Fit a line to the log-log data
            alpha, intercept = np.polyfit(log_frequencies, log_intensities, 1)
            # Print the best-fit alpha
            if verbose:
                print(f"    Region {i}: Best-fit alpha: {alpha:.2f}")
            # Generate best-fit model
            fit_intensities = 10 ** (alpha * log_frequencies + intercept)
            # Plot the best-fit model
            plt.plot(
                frequencies,
                fit_intensities,
                label=f'Fit (alpha={alpha:.2f})',
                color=f'C{i}'
            )

    plt.loglog()
    plt.xlim(0.9 * np.nanmin(frequencieslist), 1.1 * np.nanmax(frequencieslist))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(ylabel)
    plt.title('Total Intensity Spectrum (Stokes I)')
    plt.legend()
    if plotdir is not None:
        plt.savefig(plotdir / "stokesI_spectrum.png")
    if show:
        plt.show()
    plt.close()
    return

def process_stokesI(
    imageset_stokesI: ImageSet,
    region_file: Path,
    models: dict | None,
    logger: Callable,
    plotmodel: bool = True,
    plotdir: Path | None = None,
    unc: float | None = None,
    integrated: bool = True,
    flagchan: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process Stokes I images to extract flux density and compare with model.
    """

    # total flux density
    stokesI = []
    # peak flux
    stokesIpeak = []
    # freqs
    frequencies = []

    # calculate fluxes for each image in the imageset
    for i_filename in imageset_stokesI.image: # should be sorted by default
        # skip MFS image
        if "MFS" in i_filename.name:
            continue

        i_fluxes, i_peaks, freq_i = calculate_flux_and_peak_flux(i_filename, region_file)

        stokesI.append(i_fluxes)
        stokesIpeak.append(i_peaks)
        frequencies.append(freq_i)

    # convert to numpy arrays
    stokesI = np.array(stokesI) # shape (nfreq, n_regions)
    stokesIpeak = np.array(stokesIpeak) # shape (nfreq, n_regions)
    frequencies = np.array(frequencies) # shape (nfreq,)

    if integrated: 
        # Want to plot total flux generally
        logger.info("Plotting integrated flux density")
        stokesIplot = stokesI
        ylabel = "Total flux density [Jy]"
    else:
        # if user wants to plot peak flux
        logger.warning("Plotting peak flux density. WARNING, ONLY CONSISTENT IF BEAM SIZE IS THE SAME ACROSS ALL CHANNELS. OR IF THE SOURCE IS UNRESOLVED")

        stokesIplot = stokesIpeak
        ylabel = 'Peak Intensity  [Jy/beam]'
    
    if flagchan is not None:
        flagchan = [int(fc) for fc in flagchan]
        logger.info(f"Flagging channels {flagchan}")
        # mask = np.ones(len(frequencies),dtype='bool')
        # mask[args.flagchan] = False
        frequencies[flagchan] = np.nan
        stokesIplot[flagchan, :] = np.nan
    
    if plotmodel:
        plt.plot(frequencies, models['i'](frequencies), label=f"{models['src']} model", color='black', linestyle='--')

    plot_total_intensity_spectrum(
        frequencieslist=frequencies,
        intensitieslist=stokesIplot,
        fit=True,
        unc=unc,
        show=False,
        label='Stokes I',
        marker='o',
        ylabel=ylabel,
        plotdir=plotdir,
        verbose=True,
    )
    




def process_stokesQU(
    imageset_stokesQ: ImageSet,
    imageset_stokesU: ImageSet,
):
    """
    Process Stokes Q and U images to extract polarisation properties.
    """

    print("TODO")


def processfield(
    globstr_stokesI: str,
    region_file: Path,
    globstr_stokesQU: str | None = None,
    read_spectra_list: list[str] | None = None,
    flagchan: list[int] | None = None,
    plotmodel: bool = True,
    plotdir: Path | None = None,
    unc: float | None = None,
    integrated: bool = True,
    logger: Callable | None = None
) -> None:
    """
    Process a field of images to extract flux densities and compare with models.
    Args:
        globstr_stokesI (str): Glob string for Stokes I images.
        region_file (Path): Path to the region file (e.g., 3c286.reg).
        globstr_stokesQU (str, optional): Glob string for Stokes Q and U images. Defaults to None.
        read_spectra_list (list[str], optional): TODO: List of spectra files to read and plot. Defaults to None.
        integrated (bool, optional): Whether to use integrated flux density (True) or peak flux density (False). Defaults to True.
        logger (Callable, optional): Logger function. If None, uses Prefect's run logger.
    Returns:
        None: This function does not return anything, it processes the images and logs the results.
    """

    if logger is None:
        # assume we are running in a Prefect flow
        logger = get_run_logger()

    if plotmodel:
        # determine calibrator from regionfile name
        src = determine_calibrator(region_file)
        # get the correct models for this calibrator
        I_model, EVPA_model, polfrac_model = determine_model(src)
        models = {'i': I_model, 'evpa': EVPA_model, 'polfrac': polfrac_model, 'src': src}
    else:
        models = None

    if read_spectra_list is not None:
        print("TODO")
        return "TODO"

    imageset_stokesI = get_imset_from_prefix(globstr_stokesI, pol='i', validate=False)
    if len(imageset_stokesI.image) == 0:
        msg = f"No Stokes I images found in {globstr_stokesI}."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Processing Stokes I images in {globstr_stokesI}")
    process_stokesI(
        imageset_stokesI=imageset_stokesI,
        region_file=region_file,
        models=models,
        logger=logger,
        unc=unc,
        integrated=integrated,
        flagchan=flagchan,
        plotmodel=plotmodel,
        plotdir=plotdir,
    )

    if globstr_stokesQU is not None:
        # Process Stokes Q and U images as well
        logger.info(f"Processing Stokes Q and U images in {globstr_stokesQU}")
        if not integrated: 
            logger.warning(f"User requested peak flux {integrated=} but setting integrated=True for QU due to QU behaviour.")
            integrated = True

        imageset_stokesQ = get_imset_from_prefix(globstr_stokesQU, pol='q', validate=False)
        imageset_stokesU = get_imset_from_prefix(globstr_stokesQU, pol='u', validate=False)
        if len(imageset_stokesQ.image) == 0 and len(imageset_stokesU.image) == 0:
            msg = f"No Stokes Q or U images found in {globstr_stokesQU}."
            logger.error(msg)
            raise ValueError(msg)
        
        process_stokesQU(
            imageset_stokesQ=imageset_stokesQ,
            imageset_stokesU=imageset_stokesU
        )


def get_parser() -> ArgumentParser:

    descStr = """
    Process a set of images + region file to produce spectra and (optionally) compare to models.

    Function is smart enough to handle WSClean inconsistent naming conventions having -Q- and -U- but not -I- 
    
    Requires glob string for stokes I to plot total intensity spectrum.
    and optionally glob string for stokes Q and U images (can be the same), in which case it will plot polarisation intensities as well.

    Can also generally plot a bunch of spectra from a region file. In that case -plotmodel should be False. If -plotmodel True is given
    only KNOWN_CALIBRATORS are allowed as region file names, i.e. 3c286.reg, 3c138.reg, j0408-6545.reg
    """

    parser = ArgumentParser(description="")

    # required arguments
    parser.add_argument("globstr_stokesI", help="(str) Glob string for total intensity images e.g. './3c286-00*-image.fits'", type=str)
    parser.add_argument("region_file", help="(Path) Region file (aperture) for flux extraction e.g. './3c286.reg'", type=Path)

    # optional arguments
    parser.add_argument("-globstr_stokesQU", help="(str) Glob string for polint images. If given will do polint comparison. e.g. './3c286-00*-image.fits'", default=None, type=str)
    parser.add_argument('-unc',    help='(float) Estimate for 1sigma uncertainty in Jy', default=None, type=float)
    parser.add_argument('-savespectra',    help='(Path) Filename for saving the output spectra', default=None, type=Path)
    parser.add_argument('-plotmodel',    help='(str2bool) Whether to plot the model. Default True', default=True, type=str2bool)
    parser.add_argument('-plotdir',    help='(Path) where to save the plots. Default None', default=None, type=Path)
    parser.add_argument('-pbcor',    help='(str2bool) Whether to (try to) use .pbcor.fits instead', default=False, type=str2bool)
    parser.add_argument('-integrated',    help='(str2bool) Whether plot integrated flux density (True) or peak flux (False). Default True.',default=True, type=str2bool)
    parser.add_argument('-flagchan', type=parse_comma_separated, help="Comma-separated list channel numbers to flag (start from 0) (e.g. '0,5,6') ",default=None)
    
    # In case 'read_spectra_list' argument is given, the globstr, region_file and savespectra arguments are ignored
    # this can be used to plot spectra from files that were already created
    parser.add_argument('-read_spectra_list', type=parse_comma_separated, help="Comma-separated list of spectra to read and plot (e.g. 'file1.npy,file2.npy') ",default=None)
    parser.add_argument('-labels', type=parse_comma_separated, help="Comma-separated list of labels to plot (e.g. 'file1,file2') ",default=None)
    parser.add_argument('-xbrightest', type=int, help="Optional: plot only X brightest sources",default=None)
    parser.add_argument('-addfluxunc', type=float, help="Optional: Add flux uncertainty (fraction)",default=None)

    return parser


def cli() -> None:

    parser = get_parser()

    args = parser.parse_args()

    import logging as logger

    processfield(
        globstr_stokesI=args.globstr_stokesI,
        region_file=args.region_file,
        globstr_stokesQU=args.globstr_stokesQU,
        read_spectra_list=args.read_spectra_list,
        flagchan=args.flagchan,
        plotmodel=args.plotmodel,
        plotdir=args.plotdir,
        unc=args.unc,
        integrated=args.integrated,
        logger=logger
    )

if __name__ == "__main__":
    cli()