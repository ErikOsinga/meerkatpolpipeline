from __future__ import annotations

import argparse
import ast
import glob
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from prefect.logging import get_run_logger
from regions import Regions
from uncertainties import unumpy as unp

from meerkatpolpipeline.check_nvss.nvss_cutout import (
    get_nvss_cutouts,
    write_nvss_cutouts,
)
from meerkatpolpipeline.utils.processfield import calculate_flux_and_peak_flux


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute integrated flux across I/Q/U FITS images with uncertainties, table & NVSS comparison"
    )
    # required args
    parser.add_argument("--i_glob", required=True, help="Glob for I images", type=str)
    parser.add_argument("--q_glob", required=True, help="Glob for Q images", type=str)
    parser.add_argument("--pbcor_glob", required=True, help="Glob for primary beam correction FITS", type=str)
    parser.add_argument("--ds9reg", required=True, help="DS9 region file defining the source", type=Path)
    parser.add_argument("--output_dir", required=True, help="Directory to save plots", type=Path)
    
    # optional args
    parser.add_argument("--chan_unc_center", default=None, help="Channel uncertainty at field centre", type=float)
    parser.add_argument("--nvss_size", type=float, default=500.0, help="NVSS cutout size in arcsec")
    # optional args related to flagging channels
    parser.add_argument("--flag-chans", default="[]", help="List of channel indices to omit, e.g. [4,5,6]", type=str)
    parser.add_argument("--flag-by-noise", default=None, help="Table with rms noise created by analyse_noise_propertis.py", type=Path)
    parser.add_argument("--flag-by-noise-factor", default=2, help="How many times median noise is acceptable", type=float)

    # optional args related to comparing NVSS or a fits table
    parser.add_argument("--comparetable", default=None, help="FITS table for comparison (optional)")
    parser.add_argument("--comparetable_idx", default=None, help="Row index in comparison table (optional)", type=int)
    parser.add_argument("--comparenvssdirect", action='store_true', help="Enable direct NVSS comparison at 1.4 GHz")
    parser.add_argument('--nvss_dir', type=str, default=None, help='NVSS data directory. Required if comparenvssdirect is set.')

    # optional args related to saving nvss processed data
    parser.add_argument("--output_dir_data", default=None, help="Directory to save .npz data", type=Path)
    
    args = parser.parse_args()

    if args.comparenvssdirect and args.nvss_dir is None:
        raise ValueError("If --comparenvssdirect is set, --nvss_dir must be provided.")

    return args


def parse_region_center(regfile):
    with open(regfile) as f:
        for line in f:
            if 'circle' in line.lower():
                vals = line.split('(')[1].split(')')[0].split(',')
                return float(vals[0]), float(vals[1])
    raise ValueError(f"No circle region found in {regfile}")


def collect_files(glob_stokesI: str, glob_stokesQ: str = None) -> list[str] | tuple[list[str], list[str], list[str]]:
    """Collect stokes IQU files from globs"""
    ifiles = sorted(glob.glob(glob_stokesI))
    if glob_stokesQ is not None:
        qfiles = sorted(glob.glob(glob_stokesQ))
        ufiles = [q.replace('-Q-image', '-U-image') for q in qfiles]
        return ifiles, qfiles, ufiles
    return ifiles


def get_channel_frequencies(q_files: list) -> np.ndarray:
    freqs = []
    for fname in q_files:
        with fits.open(fname) as hdul:
            hdr = hdul[0].header
            if 'CRVAL3' in hdr:
                freqs.append(hdr['CRVAL3'])
            elif 'RESTFRQ' in hdr:
                freqs.append(hdr['RESTFRQ'])
            else:
                raise KeyError(f'Missing frequency keyword in {fname}')
    return np.array(freqs)


def get_nvss_fluxes(args, prefix):
    # TODO: make this work for multiple regions in ds9reg file
    ra, dec = parse_region_center(args.ds9reg)
    cutouts = get_nvss_cutouts(ra, dec, args.nvss_size, args.nvss_dir)

    nvss_dir = args.output_dir / 'nvsscutouts'
    nvss_dir.mkdir(parents=True, exist_ok=True)
    base = nvss_dir / f"{prefix}_nvsscutout.fits"

    write_nvss_cutouts(cutouts, base)
    ifn = base.with_suffix(".I.fits")
    qfn = base.with_suffix(".Q.fits")
    ufn = base.with_suffix(".U.fits")
    pfn = base.with_suffix(".p.fits")

    fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(ifn, args.ds9reg)
    fi = fluxes[0] # get first region
    fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(qfn, args.ds9reg)
    fq = fluxes[0] # get first region
    fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(ufn, args.ds9reg)
    fu = fluxes[0] # get first region
    fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(pfn, args.ds9reg)
    fp = fluxes[0] # get first region
    nvss_fluxes = {'freq': 1.4e9, 'flux_I': fi, 'flux_Q': fq, 'flux_U': fu, 'flux_P': fp}
    # TODO: add uncertainties
    
    return nvss_fluxes


def compute_uncertainty_pbcor(unc0, pb_files, ra, dec):
    """
    Given an uncertainty estimate in the centre of the field, unc0
    compute the uncertainty at the given ra,dec position using the
    primary beam correction files.
    """
    corrected = []
    sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    for fname in pb_files:
        with fits.open(fname) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr).celestial
            xpix, ypix = w.world_to_pixel(sky)
            pb = hdul[0].data[0,0,int(np.round(ypix)), int(np.round(xpix))]
            corrected.append(unc0 / pb)
    return np.array(corrected)


def compute_fluxes(ifiles, qfiles, ufiles, region_file):
    """Compute integrated fluxes for I, Q, U files in the given region"""

    flux_I, flux_Q, flux_U = [], [], []
    beams_I, beams_Q, beams_U = [], [], []
    for i_f, q_f, u_f in zip(ifiles, qfiles, ufiles):
        fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(i_f, region_file)
        fI = fluxes[0]  # get first region
        bI = Nbeams[0]
        fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(q_f, region_file)
        fQ = fluxes[0]  # get first region
        bQ = Nbeams[0]
        fluxes, peaks, freq, Nbeams = calculate_flux_and_peak_flux(u_f, region_file)
        fU = fluxes[0]  # get first region
        bU = Nbeams[0]

        flux_I.append(fI)
        beams_I.append(bI)
        flux_Q.append(fQ)
        beams_Q.append(bQ)
        flux_U.append(fU)
        beams_U.append(bU)

    return (np.array(flux_I), np.array(flux_Q), np.array(flux_U),
            np.array(beams_I), np.array(beams_Q), np.array(beams_U))


def save_data(prefix, output_dir_data, freqs, flux_I, flux_Q, flux_U,
              unc_I, unc_Q, unc_U, beams_I, beams_Q, beams_U):
    """Save data to .npz file"""
    output_dir_data.mkdir(parents=True, exist_ok=True)
    outpath = output_dir_data / f"{prefix}_integratedflux.npz"
    np.savez(outpath,
             freqs=freqs,
             flux_I=flux_I, flux_Q=flux_Q, flux_U=flux_U,
             unc_I=unc_I, unc_Q=unc_Q, unc_U=unc_U,
             beams_I=beams_I, beams_Q=beams_Q, beams_U=beams_U)
    print(f"Data saved to {outpath}")
    return outpath


def plot_flux_vs_nvss(prefix, output_dir,
              freqs, flux_I, flux_Q, flux_U,
              unc_I, unc_Q, unc_U,
              lam2, channels,
              polint, polint_err,
              polang, polang_err,
              comp=None, nvss=None):
    """
    Plot fluxes vs frequency and wavelength^2, comparing to NVSS and table if given.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    # Stokes I
    ax1.errorbar(freqs, flux_I, yerr=unc_I, fmt='o', linestyle='none', label='measured')
    if comp and np.isfinite(comp['reffreq_I']) and np.isfinite(comp['stokesI']):
        ax1.errorbar(comp['reffreq_I'], comp['stokesI'],
                     yerr=comp['stokesI_err'], fmt='D', mfc='none', mec='r', label='table')
    if nvss:
        ax1.errorbar(nvss['freq'], nvss['flux_I'], fmt='x', label='NVSS')
    ax1.set_ylabel('Integrated flux [Jy]')
    ax1.set_title('Stokes I')
    ax1.grid(True)
    ax1_top = ax1.twiny(); ax1_top.set_xlim(ax1.get_xlim()); ax1_top.set_xticks(freqs); ax1_top.set_xticklabels(channels); ax1_top.set_xlabel('Channel #')
    ax1.legend()

    # Polarised intensity
    ax2.errorbar(lam2, polint, yerr=polint_err, fmt='o', linestyle='none', label='measured')
    if comp and np.isfinite(comp['lam2_pol']) and np.isfinite(comp['polint']):
        ax2.errorbar(comp['lam2_pol'], comp['polint'],
                     yerr=comp['polint_err'], fmt='D', mfc='none', mec='r', label='table')
    if nvss:
        lam2_nvss = (c.value / 1.4e9)**2
        ax2.errorbar(lam2_nvss, nvss['flux_P'], fmt='x', label='NVSS')
    ax2.set_ylabel('Integrated flux [Jy]'); ax2.set_title('Polarised intensity'); ax2.grid(True)
    ax2_top = ax2.twiny(); ax2_top.set_xlim(ax2.get_xlim()); ax2_top.set_xticks(lam2); ax2_top.set_xticklabels(channels); ax2_top.set_xlabel('Channel #')
    ax2.legend()

    # Stokes Q & U
    ax3.errorbar(freqs, flux_Q, yerr=unc_Q, fmt='s', linestyle='none', label='Q')
    ax3.errorbar(freqs, flux_U, yerr=unc_U, fmt='^', linestyle='none', label='U')
    if nvss:
        ax3.errorbar(nvss['freq'], nvss['flux_Q'], fmt='x', label='NVSS Q')
        ax3.errorbar(nvss['freq'], nvss['flux_U'], fmt='x', label='NVSS U')
    ax3.set_ylabel('Integrated flux [Jy]'); ax3.set_title('Stokes Q & U'); ax3.grid(True)
    ax3_top = ax3.twiny(); ax3_top.set_xlim(ax3.get_xlim()); ax3_top.set_xticks(freqs); ax3_top.set_xticklabels(channels); ax3_top.set_xlabel('Channel #')
    ax3.set_xlabel('Frequency [Hz]'); ax3.legend()

    # Polarisation angle (degrees)
    polang_deg = np.degrees(polang)
    polang_err_deg = np.degrees(polang_err)
    ax4.errorbar(lam2, polang_deg, yerr=polang_err_deg, fmt='s', linestyle='none', label='measured')
    rad_unw = np.unwrap(polang)
    RM_fit, chi0_fit = np.polyfit(lam2, rad_unw, 1)
    fitdeg = np.degrees(RM_fit * lam2 + chi0_fit)
    ax4.plot(lam2, fitdeg, color='k', ls='--', label=f"fit: RM={RM_fit:.1f} rad/m², chi0={chi0_fit:.2f} rad")
    if comp and np.isfinite(comp['rm']):
        chi0_comp = np.mean(rad_unw - comp['rm'] * lam2)
        tabdeg = np.degrees(comp['rm'] * lam2 + chi0_comp)
        ax4.plot(lam2, tabdeg, color='r', ls=':', label=f"table: RM={comp['rm']:.1f} rad/m², chi0={chi0_comp:.2f} rad")
    if nvss:
        nvss_angle = np.degrees(0.5 * np.arctan2(nvss['flux_U'], nvss['flux_Q']))
        ax4.scatter((c.value / 1.4e9)**2, nvss_angle, marker='x', label='NVSS angle')
    ax4.set_ylabel('Polarisation angle [deg]'); ax4.set_title('Polarisation angle'); ax4.set_ylim(-110, 110); ax4.grid(True)
    ax4_top = ax4.twiny(); ax4_top.set_xlim(ax4.get_xlim()); ax4_top.set_xticks(lam2); ax4_top.set_xticklabels(channels); ax4_top.set_xlabel('Channel #')
    ax4.set_xlabel('Wavelength² [m²]'); ax4.legend()

    fig.tight_layout()
    out = output_dir / f"{prefix}_integratedflux.png"
    fig.savefig(out)
    print(f"Plot saved to {out}")
    plt.show()
    plt.close(fig)
    return out


def compare_to_nvss(args):
    prefix = args.ds9reg.stem

    flag_chans = ast.literal_eval(args.flag_chans)

    # collect fits lists
    ifiles, qfiles, ufiles = collect_files(args.i_glob, args.q_glob)
    pb_files = sorted(glob.glob(args.pbcor_glob))
    
    assert len(pb_files) == len(qfiles) == len(ifiles), f"pbcor files count must match Q and I files count. Instead {len(pb_files)=} and {len(qfiles)=} and {len(ifiles)=}"

    if args.comparetable and args.comparetable_idx is not None:
        # comp = compare_to_table()
        print("TODO: implement comparison to table")
    else:
        comp = None

    if args.comparenvssdirect:
        nvss_fluxes = get_nvss_fluxes(args, prefix)

    else:
        nvss_fluxes = None

    # channel properties
    freqs = get_channel_frequencies(qfiles)
    channels = np.arange(len(freqs))
    lam2 = (c.value / freqs)**2

    # compute integrated fluxes for first region in ds9reg
    flux_I, flux_Q, flux_U, beams_I, beams_Q, beams_U = compute_fluxes(ifiles, qfiles, ufiles, args.ds9reg)

    # TODO: make it simply do all regions in a ds9reg file.

    if args.chan_unc_center is not None:
        # TODO: make this work for multiple regions in ds9reg file
        ra, dec = parse_region_center(args.ds9reg)
        unc0 = args.chan_unc_center
        unc_I = compute_uncertainty_pbcor(unc0, pb_files, ra, dec)
        unc_Q = compute_uncertainty_pbcor(unc0, pb_files, ra, dec)
        unc_U = compute_uncertainty_pbcor(unc0, pb_files, ra, dec)

        # TODO: add uncertainty from number of beams

    else:
        unc_I = np.zeros(len(qfiles))
        unc_Q = unc_U = unc_I

    # propagate uncertainties
    I_u = unp.uarray(flux_I, unc_I)
    Q_u = unp.uarray(flux_Q, unc_Q)
    U_u = unp.uarray(flux_U, unc_U)
    P_u = unp.sqrt(Q_u**2 + U_u**2)
    psi_u = 0.5 * unp.arctan2(U_u, Q_u)

    polint = unp.nominal_values(P_u)
    polint_err = unp.std_devs(P_u)
    polang = unp.nominal_values(psi_u)
    polang_err = unp.std_devs(psi_u)

    # save data
    if args.output_dir_data is not None:
        save_data(prefix, args.output_dir_data,
                freqs, flux_I, flux_Q, flux_U,
                unc_I, unc_Q, unc_U,
                beams_I, beams_Q, beams_U)

    # apply channel flags and plot
    mask = ~np.isin(channels, flag_chans) if flag_chans else np.ones_like(channels, bool)
    print(f"Found {np.sum(~mask)}/{len(mask)} flagged channels manually")

    # flag NaNs
    nan_mask = np.isnan(flux_I) | np.isnan(flux_Q) | np.isnan(flux_U)
    print(f"Found {np.sum(nan_mask)}/{len(mask)} flagged channels because NaN")
    # flag crazy-high/low values
    threshold = 1e3
    high_mask = (np.abs(flux_I) > threshold) | (np.abs(flux_Q) > threshold) | (np.abs(flux_U) > threshold)
    print(f"Found {np.sum(high_mask)}/{len(mask)} flagged channels because crazy-high values")

    # if user gives noise per channel
    if args.flag_by_noise is not None:
        raise NotImplementedError("Flag by noise not yet implemented")
        factor = args.flag_by_noise_factor
        tbl = Table.read(args.flag_by_noise)
        mediannoise = np.nanmedian(tbl['noise_U'])

        mask_by_noise = tbl['noise_U'] > factor * mediannoise
        mask_indices = tbl['channel'][mask_by_noise]
        print(f"Masking {len(mask_indices)} channels because higher than {factor:.1f}x median noise")
        mask_for_noise = np.ones_like(channels, bool)
        mask_for_noise[mask_indices] = False 

        # combine all, now True is keep and False is mask
        mask &= ~(nan_mask | high_mask | ~mask_for_noise)

    else:

        # combine all. so that data[mask] returns all good data.
        mask &= ~(nan_mask | high_mask)

    print(f"Final number of good channels: {np.sum(mask)}/{len(mask)}")

    plot_flux_vs_nvss(prefix, args.output_dir,
              freqs[mask], flux_I[mask], flux_Q[mask], flux_U[mask],
              unc_I[mask], unc_Q[mask], unc_U[mask],
              lam2[mask], channels[mask],
              polint[mask], polint_err[mask], polang[mask], polang_err[mask],
              comp, nvss_fluxes)

if __name__ == "__main__":
    args = parse_args()

    compare_to_nvss(args)