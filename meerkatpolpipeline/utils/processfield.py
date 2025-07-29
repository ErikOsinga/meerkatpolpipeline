from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
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
    plt.tight_layout()
    if plotdir is not None:
        plt.savefig(plotdir / "stokesI_spectrum.png")
    if show:
        plt.show()
    plt.close()
    return


def get_fluxes(imageset: ImageSet, region_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate flux densities and peak fluxes for a set of Stokes I/Q/U images.
    """
    # total flux density
    totalflux = []
    # peak flux
    peakflux = []
    # freqs
    frequencies = []

    # calculate fluxes for each image in the imageset
    for filename in imageset.image: # should be sorted by default
        # skip MFS image
        if "MFS" in filename.name:
            continue

        fluxes, peaks, freq = calculate_flux_and_peak_flux(filename, region_file)

        totalflux.append(fluxes)
        peakflux.append(peaks)
        frequencies.append(freq)

    # convert to numpy arrays
    totalflux = np.array(totalflux) # shape (nfreq, n_regions)
    peakflux = np.array(peakflux) # shape (nfreq, n_regions)
    frequencies = np.array(frequencies) # shape (nfreq,)

    return totalflux, peakflux, frequencies


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

    Returns:
           stokesI, stokesIpeak, frequencies
           (np.array, np.ndarray, np.ndarray):
    shapes (nfreq, n_regions), (nfreq, n_regions), (nfreq,)
    """

    stokesI, stokesIpeak, frequencies = get_fluxes(imageset_stokesI, region_file)

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

    if not plotdir.exists():
        plotdir.mkdir()

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
    
    return stokesI, stokesIpeak, frequencies


def plot_stokesQU_spectra(
    frequencies: np.ndarray,
    Qvals: np.ndarray,
    Uvals: np.ndarray,
    unc: float | None = None,
    plotdir: Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot the Stokes Q and U spectra 

    Assuming Qvals and Uvals are of shape (nfreq, nregions),
    """

    plt.figure()

    for i in range(np.shape(Qvals)[1]):
        plt.plot(frequencies, Qvals[:, i], label='Stokes Q', marker='o')
        plt.plot(frequencies, Uvals[:, i], label='Stokes U', marker='o')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux density [Jy]')
    plt.title('Stokes Q and U Spectra')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    if plotdir is not None:
        plt.savefig(plotdir / "stokesQU_spectra.png")
    plt.close()
    return


def compute_polint_polfrac(
    i_values: np.ndarray,
    q_values: np.ndarray,
    u_values: np.ndarray,
    frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the polarisation fraction as a function of lambda^2 from I, Q and U values.
    
    assumes frequencies in Hz and i_values and q_values in the same unit as u_values.

    returns
        lambda_squared       -- np.ndarray -- wavelength^2 value per frequency
        polarised_intensity  -- np.ndarray -- polint  value per frequency/wavelength^2
        pol_fractions        -- np.ndarray -- polfrac value per frequency/wavelength^2
    
    """
    polarised_intensity = np.sqrt(q_values**2 + u_values**2)
    pol_fractions = polarised_intensity / i_values
    lambda_squared = (c.value / frequencies)**2
    
    return lambda_squared, polarised_intensity, pol_fractions


def plot_polfrac_vs_lambdasq(
    lambdasq_obs: np.ndarray,
    polfrac_obs: np.ndarray,
    lambdasq_fit: np.ndarray | None = None,
    polfrac_fit: np.ndarray | None = None,
    src: str = "",
    plotdir: Path | None = None,
    show: bool = False
) -> None:
    """
    Plot polarisation fraction vs lambdasq

    Assuming polfrac_obs is of shape (nfreq, nregions),
    """
    nregions = polfrac_obs.shape[1]
    has_model = lambdasq_fit is not None and polfrac_fit is not None

    # Create figure and axes
    if has_model:
        fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    else:
        fig, ax_top = plt.subplots(figsize=(8, 4))
        ax_bottom = None

    # Top panel: observed data and model (if available)
    for i in range(nregions):
        ax_top.plot(lambdasq_obs, polfrac_obs[:, i], "o-", label=f"Observed region {i+1}")
    if has_model:
        ax_top.plot(lambdasq_fit, polfrac_fit, label=f"{src} Model", color="k", ls="--")

    ax_top.set_ylabel("Polarisation fraction")
    ax_top.set_title("Polarisation Fraction vs Lambda$^2$")
    ax_top.legend()

    # Bottom panel: residuals (data - model)
    if has_model and ax_bottom is not None:
        residuals = polfrac_obs - polfrac_fit[:, None]
        for i in range(nregions):
            ax_bottom.plot(lambdasq_obs, residuals[:, i], "o-", label=f"Residual region {i+1}")
        ax_bottom.set_xlabel("Lambda$^2$ (m$^2$)")
        ax_bottom.set_ylabel("Residual (data - model)")
        ax_bottom.axhline(0, color="gray", lw=1, ls=":")
        ax_bottom.legend()
    else:
        ax_top.set_xlabel("Lambda$^2$ (m$^2$)")

    plt.tight_layout()

    if show:
        plt.show()
    if plotdir is not None:
        filename = "polfrac_vs_lambdasq.png" if not has_model else "polfrac_vs_lambdasq_with_residuals.png"
        fig.savefig(plotdir / filename)

    plt.close(fig)
    return


def compute_polarisation_angle_spectrum(
    q_values: np.ndarray,
    u_values:np.ndarray,
    frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the polarisation angle as a function of lambda^2 from Q and U values.
    
    assumes frequencies in Hz and q_values in the same unit as u_values.
    
    """
    angles = 0.5 * np.arctan2(u_values, q_values) # polarisation angle in rad
    angles = np.degrees(angles) # in deg
    lambda_squared = (c.value / frequencies)**2
    return lambda_squared, angles


def plot_evpa_vs_lambdasq(
    lambdasq_obs: np.ndarray,
    evpa_obs: np.ndarray,
    lambdasq_fit: np.ndarray | None = None,
    evpa_fit: np.ndarray | None = None,
    src: str = "",
    plotdir: Path | None = None,
    fig_suffix: str = "",
    show: bool = False
) -> np.ndarray | None:
    """
    Plot EVPA vs lambda^2

    Assuming evpa_obs is of shape (nfreq, nregions).
    If a model is provided, a two-panel plot is created with residuals (data - model) below.
    and the residuals are returned. Note that for South-Africa we expect negative RM explaining the residuals.
    """
    nregions = evpa_obs.shape[1]
    has_model = lambdasq_fit is not None and evpa_fit is not None

    # Create figure and axes
    if has_model:
        fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    else:
        fig, ax_top = plt.subplots(figsize=(8, 4))
        ax_bottom = None

    # Top panel: observed EVPA (and model if available)
    for i in range(nregions):
        ax_top.plot(lambdasq_obs, evpa_obs[:, i], 'o-', label=f"Observed region {i+1}")
    if has_model:
        ax_top.plot(lambdasq_fit, evpa_fit, label=f"{src} EVPA model", color='k', ls='--')

    ax_top.set_ylabel("EVPA (deg)")
    if "corrected" in fig_suffix:
        ax_top.set_title("Polarisation Angle Spectrum (corrected for data-derived ionosphere)")
    else:
        ax_top.set_title("Polarisation Angle Spectrum")
    ax_top.legend()

    # Bottom panel: residuals (data - model)
    if has_model and ax_bottom is not None:
        residuals = evpa_obs - evpa_fit[:, None]
        for i in range(nregions):
            ax_bottom.plot(lambdasq_obs, residuals[:, i], 'o-', label=f"Residual region {i+1}")
        ax_bottom.axhline(0, color='gray', ls=':')
        ax_bottom.set_xlabel("Lambda$^2$ (m$^2$)")
        ax_bottom.set_ylabel("Residual (data - model)")
        ax_bottom.legend()
    else:
        ax_top.set_xlabel("Lambda$^2$ (m$^2$)")
        residuals = None

    plt.tight_layout()

    if show:
        plt.show()
    if plotdir is not None:
        filename = "evpa_vs_lambdasq" if not has_model else "evpa_vs_lambdasq_with_residuals"
        filename += fig_suffix
        filename += ".png"
        fig.savefig(plotdir / filename)

    plt.close(fig)
    return residuals


def process_stokesQU(
    imageset_stokesQ: ImageSet,
    imageset_stokesU: ImageSet,
    stokesI_fluxdens: np.ndarray,
    region_file: Path,
    models: dict | None,
    logger: Callable,
    plotmodel: bool = True,
    plotdir: Path | None = None,
    unc: float | None = None,
    flagchan: list[int] | None = None,
) -> float:
    """
    Process Stokes Q and U images to extract polarisation properties.

    Returns:
           rm_iono_rad_m2: float or None -- inferred RM that explains offset between data and model.
    """
    if not plotdir.exists():
        logger.info(f"Creating {plotdir}")
        plotdir.mkdir()

    # measure Q and U, integrated flux only
    Qvals, _, freqs_Q = get_fluxes(imageset_stokesQ, region_file)
    Uvals, _, freqs_U = get_fluxes(imageset_stokesU, region_file)

    assert (freqs_Q == freqs_U).all(), "Frequencies for Stokes Q and U must match"

    # I flux density should be input by user, and match Q and U exactly.
    # can be calculated with process_stokesI()
    Ivals = stokesI_fluxdens

    # 1) plot raw Q/U spectra
    plot_stokesQU_spectra(
        frequencies=freqs_Q,
        Qvals=Qvals,
        Uvals=Uvals,
        unc=unc,
        plotdir=plotdir,
        show=False
    )

    # 2) plot polarisation fraction
    lambdasq_obs, polint_obs, polfrac_obs  = compute_polint_polfrac(Ivals, Qvals, Uvals, freqs_Q)
    if models is not None and plotmodel:
        # compute the model
        lambdasq_fit, polfrac_fit = models['polfrac'](freqs_Q)
        src = models['src']
    else:
        lambdasq_fit, polfrac_fit, src = None, None, None

    plot_polfrac_vs_lambdasq(
        lambdasq_obs,
        polfrac_obs,
        lambdasq_fit,
        polfrac_fit,
        src,
        plotdir
    )

    # 3) plot polarisation angle (EVPA)
    lambdasq_obs, evpa_obs = compute_polarisation_angle_spectrum(Qvals, Uvals, freqs_Q)
    if models is not None and plotmodel:
        # compute the model
        lambdasq_fit, evpa_fit = models['evpa'](freqs_Q)
    else:
        lambdasq_fit, evpa_fit = None, None

    evpa_residuals = plot_evpa_vs_lambdasq(
        lambdasq_obs,
        evpa_obs,
        lambdasq_fit,
        evpa_fit,
        plotdir
    )

    # 4) Infer ionospheric RM from EVPA residuals
    # analytical least squares fit
    rm_iono_deg_m2 = np.sum(lambdasq_obs * evpa_residuals) / np.sum(lambdasq_obs**2)
    # convert to rad/m^2:
    rm_iono_rad_m2 = np.deg2rad(rm_iono_deg_m2)
    logger.info(f"Inferred ionospheric RM from EVPA residuals = {rm_iono_rad_m2:.3f} rad/m^2")

    if rm_iono_rad_m2 > 0:
        logger.warning("For South-Africa we expect a negative ionospheric RM contribution. Something has gone wrong?")


    # 5) Build a 'corrected' EVPA curve
    # i.e. PA_obs - RM_iono*lambda^2 should be equal to the model
    evpa_obs_corrected = evpa_obs - (rm_iono_deg_m2*lambdasq_obs)[:, None] # make sure its same shape
    print(f"{evpa_obs=}")
    print(f"{evpa_obs_corrected=}")
    print(f"{(rm_iono_deg_m2*lambdasq_obs)[:, None]=}")
    plot_evpa_vs_lambdasq(
        lambdasq_obs,
        evpa_obs_corrected,
        lambdasq_fit,
        evpa_fit,
        plotdir,
        fig_suffix=f"_corrected_{rm_iono_rad_m2:.1f}_radm2"
    )

    return rm_iono_rad_m2


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
    stokesI_fluxdens, stokesIpeak, frequencies = process_stokesI(
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
        
        rm_iono_rad_m2 = process_stokesQU(
            imageset_stokesQ=imageset_stokesQ,
            imageset_stokesU=imageset_stokesU,
            stokesI_fluxdens=stokesI_fluxdens,
            region_file=region_file,
            models=models,
            logger=logger,
            unc=unc,
            integrated=integrated,
            flagchan=flagchan,
            plotmodel=plotmodel,
            plotdir=plotdir,
        )

        rm_iono_rad_m2


def get_parser() -> ArgumentParser:

    descStr = """
    Process a set of images + region file to produce spectra and (optionally) compare to models.

    Function is smart enough to handle WSClean inconsistent naming conventions having -Q- and -U- but not -I- 
    
    Requires glob string for stokes I to plot total intensity spectrum.
    and optionally glob string for stokes Q and U images (can be the same), in which case it will plot polarisation intensities as well.

    Can also generally plot a bunch of spectra from a region file. In that case -plotmodel should be False. If -plotmodel True is given
    only KNOWN_CALIBRATORS are allowed as region file names, i.e. 3c286.reg, 3c138.reg, j0408-6545.reg
    """

    parser = ArgumentParser(description=descStr)

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