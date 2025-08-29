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
from astropy.wcs import WCS
from regions import Regions
from uncertainties import unumpy as unp

from meerkatpolpipeline.check_nvss.nvss_cutout import (
    get_nvss_cutouts,
    write_nvss_cutouts,
)
from meerkatpolpipeline.utils.processfield import calculate_flux_and_peak_flux


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - i_glob, q_glob, pbcor_glob: globs for Stokes I, Q, and PB-correction FITS
        - ds9reg: DS9 region file path
        - output_dir: output directory for plots
        - chan_unc_center: per-channel uncertainty at field center (optional)
        - nvss_size: NVSS cutout size in arcsec
        - flag_chans: string representing python list of channel indices to omit
        - flag_by_noise: table path for channel noise (Not implemented)
        - flag_by_noise_factor: factor above median noise to flag
        - comparetable, comparetable_idx: comparison table inputs (not implemented)
        - comparenvssdirect: whether to compare directly to NVSS
        - nvss_dir: NVSS data directory (required if comparenvssdirect)
        - output_dir_data: directory to save .npz data (optional)
    """
    parser = argparse.ArgumentParser(
        description="Compute integrated flux across I/Q/U FITS images with uncertainties, table & NVSS comparison."
    )
    # required args
    parser.add_argument("--i_glob", required=True, help="Glob for I images", type=str)
    parser.add_argument("--q_glob", required=True, help="Glob for Q images", type=str)
    parser.add_argument(
        "--pbcor_glob", required=True, help="Glob for primary beam correction FITS", type=str
    )
    parser.add_argument(
        "--ds9reg", required=True, help="DS9 region file defining the source", type=Path
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save plots", type=Path
    )

    # optional args
    parser.add_argument(
        "--chan_unc_center", default=None, help="Channel uncertainty at field centre", type=float
    )
    parser.add_argument(
        "--nvss_size", type=float, default=500.0, help="NVSS cutout size in arcsec"
    )
    # optional args related to flagging channels
    parser.add_argument(
        "--flag-chans",
        dest="flag_chans",
        default="[]",
        help="List of channel indices to omit, e.g. [4,5,6]",
        type=str,
    )
    parser.add_argument(
        "--flag-by-noise",
        dest="flag_by_noise",
        default=None,
        help="Table with rms noise created by analyse_noise_propertis.py",
        type=Path,
    )
    parser.add_argument(
        "--flag-by-noise-factor",
        dest="flag_by_noise_factor",
        default=2.0,
        help="How many times median noise is acceptable",
        type=float,
    )

    # optional args related to comparing NVSS or a fits table
    parser.add_argument(
        "--comparetable", default=None, help="FITS table for comparison (optional)"
    )
    parser.add_argument(
        "--comparetable_idx",
        default=None,
        help="Row index in comparison table (optional)",
        type=int,
    )
    parser.add_argument(
        "--comparenvssdirect",
        action="store_true",
        help="Enable direct NVSS comparison at 1.4 GHz",
    )
    parser.add_argument(
        "--nvss_dir",
        type=str,
        default=None,
        help="NVSS data directory. Required if comparenvssdirect is set.",
    )

    # optional args related to saving nvss processed data
    parser.add_argument(
        "--output_dir_data", default=None, help="Directory to save .npz data", type=Path
    )

    args = parser.parse_args()

    if args.comparenvssdirect and args.nvss_dir is None:
        raise ValueError("If --comparenvssdirect is set, --nvss_dir must be provided.")

    return args


def parse_region_center(regfile: Path) -> tuple[float, float]:
    """
    Return RA, Dec (deg) for the first region in a DS9 region file.

    Parameters
    ----------
    regfile : Path
        Path to DS9 region file.

    Returns
    -------
    tuple[float, float]
        (ra_deg, dec_deg) of the first region's center.

    Notes
    -----
    Uses `regions.Regions.read` when possible. Falls back to a
    lightweight circle parser if the file cannot be parsed by `regions`.
    """
    try:
        regs = Regions.read(str(regfile))
        if len(regs) == 0:
            raise ValueError(f"No regions found in {regfile}")
        center = regs[0].center
        sky = center.to_skycoord() if hasattr(center, "to_skycoord") else center
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        # fallback: minimal parser for lines containing "circle"
        with open(regfile) as f:
            for line in f:
                if "circle" in line.lower():
                    vals = line.split("(")[1].split(")")[0].split(",")
                    return float(vals[0]), float(vals[1])
    raise ValueError(f"No circle region found in {regfile}")


def collect_files(glob_stokesI: str, glob_stokesQ: str | None = None) -> list[str] | tuple[list[str], list[str], list[str]]:
    """
    Collect Stokes I (and optionally Q/U) file lists from globs.

    Parameters
    ----------
    glob_stokesI : str
        Glob pattern for Stokes I images.
    glob_stokesQ : str | None
        Glob pattern for Stokes Q images. If provided, U files are inferred
        by replacing '-Q-image' with '-U-image'.

    Returns
    -------
    list[str] | tuple[list[str], list[str], list[str]]
        If only I is requested: list of I files.
        If Q provided: (I_files, Q_files, U_files).
    """
    ifiles = sorted(glob.glob(glob_stokesI))
    if glob_stokesQ is None:
        return ifiles
    qfiles = sorted(glob.glob(glob_stokesQ))
    ufiles = [q.replace("-Q-image", "-U-image") for q in qfiles]
    return ifiles, qfiles, ufiles


def get_channel_frequencies(q_files: list[str]) -> np.ndarray:
    """
    Extract per-channel frequencies (Hz) from FITS headers.

    Parameters
    ----------
    q_files : list[str]
        List of Stokes Q FITS files, one per channel.

    Returns
    -------
    np.ndarray
        Array of frequencies in Hz.

    Raises
    ------
    KeyError
        If neither CRVAL3 nor RESTFRQ is found in a header.
    """
    freqs: list[float] = []
    for fname in q_files:
        with fits.open(fname) as hdul:
            hdr = hdul[0].header
            if "CRVAL3" in hdr:
                freqs.append(float(hdr["CRVAL3"]))
            elif "RESTFRQ" in hdr:
                freqs.append(float(hdr["RESTFRQ"]))
            else:
                raise KeyError(f"Missing frequency keyword in {fname}")
    return np.asarray(freqs, dtype=float)


def get_nvss_fluxes(args: argparse.Namespace, prefix: str) -> dict[str, float]:
    """
    Produce NVSS I/Q/U/p cutouts and compute integrated fluxes in the DS9 region.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed args (uses ds9reg, nvss_size, nvss_dir, output_dir).
    prefix : str
        Base prefix for output files.

    Returns
    -------
    dict[str, float]
        {'freq': 1.4e9, 'flux_I': fi, 'flux_Q': fq, 'flux_U': fu, 'flux_P': fp}

    Notes
    -----
    Currently assumes a single region in ds9reg.
    """
    ra, dec = parse_region_center(args.ds9reg)
    cutouts = get_nvss_cutouts(ra, dec, args.nvss_size, args.nvss_dir)

    nvss_dir = args.output_dir / "nvsscutouts"
    nvss_dir.mkdir(parents=True, exist_ok=True)
    base = nvss_dir / f"{prefix}_nvsscutout.fits"

    write_nvss_cutouts(cutouts, base)
    ifn = base.with_suffix(".I.fits")
    qfn = base.with_suffix(".Q.fits")
    ufn = base.with_suffix(".U.fits")
    pfn = base.with_suffix(".p.fits")

    def _first_flux(fpath: Path) -> float:
        fluxes, _, _, nbeams = calculate_flux_and_peak_flux(fpath, args.ds9reg)
        _ = nbeams  # available if needed later
        return float(fluxes[0])

    fi = _first_flux(ifn)
    fq = _first_flux(qfn)
    fu = _first_flux(ufn)
    fp = _first_flux(pfn)

    # TODO: add uncertainties from NVSS maps (rms + beam)
    return {"freq": 1.4e9, "flux_I": fi, "flux_Q": fq, "flux_U": fu, "flux_P": fp}


def compute_uncertainty_pbcor(unc0: float, pb_files: list[str], ra: float, dec: float) -> np.ndarray:
    """
    Scale a central-channel uncertainty by primary-beam gain at a sky position.

    Parameters
    ----------
    unc0 : float
        Uncertainty at field center for a single channel (same units as desired output).
    pb_files : list[str]
        List of primary-beam correction FITS files (one per channel).
    ra, dec : float
        Sky position in deg.

    Returns
    -------
    np.ndarray
        Array of per-channel uncertainties after PB correction.
    """
    corrected: list[float] = []
    sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    for fname in pb_files:
        with fits.open(fname) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr).celestial  # use celestial WCS to be robust to redundant axes
            xpix, ypix = w.world_to_pixel(sky)
            xi, yi = int(np.round(xpix)), int(np.round(ypix))
            pb = float(hdul[0].data[0, 0, yi, xi])
            corrected.append(unc0 / pb if pb != 0.0 else np.inf)
    return np.asarray(corrected, dtype=float)


def _first_region_flux_and_beams(fpath: str, region_file: Path) -> tuple[float, float]:
    """
    Wrapper: compute integrated flux and beams for first region only.

    Returns
    -------
    tuple[float, float]
        (flux, nbeams) for the first region.
    """
    fluxes, _, _, nbeams = calculate_flux_and_peak_flux(fpath, region_file)
    return float(fluxes[0]), float(nbeams[0])


def compute_fluxes(ifiles: list[str], qfiles: list[str], ufiles: list[str], region_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute integrated fluxes and beam counts for I, Q, U over channels.

    Parameters
    ----------
    ifiles, qfiles, ufiles : list[str]
        File lists (aligned per-channel).
    region_file : Path
        DS9 region file.

    Returns
    -------
    tuple of np.ndarray
        (flux_I, flux_Q, flux_U, beams_I, beams_Q, beams_U), each 1D arrays.
    """
    flux_I: list[float] = []
    flux_Q: list[float] = []
    flux_U: list[float] = []
    beams_I: list[float] = []
    beams_Q: list[float] = []
    beams_U: list[float] = []

    for i_f, q_f, u_f in zip(ifiles, qfiles, ufiles):
        fI, bI = _first_region_flux_and_beams(i_f, region_file)
        fQ, bQ = _first_region_flux_and_beams(q_f, region_file)
        fU, bU = _first_region_flux_and_beams(u_f, region_file)

        flux_I.append(fI)
        beams_I.append(bI)
        flux_Q.append(fQ)
        beams_Q.append(bQ)
        flux_U.append(fU)
        beams_U.append(bU)

    return (
        np.asarray(flux_I, dtype=float),
        np.asarray(flux_Q, dtype=float),
        np.asarray(flux_U, dtype=float),
        np.asarray(beams_I, dtype=float),
        np.asarray(beams_Q, dtype=float),
        np.asarray(beams_U, dtype=float),
    )


def save_data(
    prefix: str,
    output_dir_data: Path,
    freqs: np.ndarray,
    flux_I: np.ndarray,
    flux_Q: np.ndarray,
    flux_U: np.ndarray,
    unc_I: np.ndarray,
    unc_Q: np.ndarray,
    unc_U: np.ndarray,
    beams_I: np.ndarray,
    beams_Q: np.ndarray,
    beams_U: np.ndarray,
) -> Path:
    """
    Save channel-wise results to a compressed .npz.

    Returns
    -------
    Path
        Path to the saved .npz file.
    """
    output_dir_data.mkdir(parents=True, exist_ok=True)
    outpath = output_dir_data / f"{prefix}_integratedflux.npz"
    np.savez(
        outpath,
        freqs=freqs,
        flux_I=flux_I,
        flux_Q=flux_Q,
        flux_U=flux_U,
        unc_I=unc_I,
        unc_Q=unc_Q,
        unc_U=unc_U,
        beams_I=beams_I,
        beams_Q=beams_Q,
        beams_U=beams_U,
    )
    print(f"Data saved to {outpath}")
    return outpath


def plot_flux_vs_nvss(
    prefix: str,
    output_dir: Path,
    freqs: np.ndarray,
    flux_I: np.ndarray,
    flux_Q: np.ndarray,
    flux_U: np.ndarray,
    unc_I: np.ndarray,
    unc_Q: np.ndarray,
    unc_U: np.ndarray,
    lam2: np.ndarray,
    channels: np.ndarray,
    polint: np.ndarray,
    polint_err: np.ndarray,
    polang: np.ndarray,
    polang_err: np.ndarray,
    comp: dict | None = None,
    nvss: dict | None = None,
) -> Path:
    """
    Plot Stokes I/Q/U, P, and polarisation angle vs frequency or lambda^2.

    Parameters
    ----------
    prefix : str
        Output prefix for plot filename.
    output_dir : Path
        Directory to save the plot.
    freqs, lam2, channels : np.ndarray
        Frequency (Hz), lambda^2 (m^2), channel indices.
    flux_I, flux_Q, flux_U : np.ndarray
        Integrated fluxes (Jy).
    unc_I, unc_Q, unc_U : np.ndarray
        Uncertainties (Jy).
    polint, polint_err : np.ndarray
        Polarised intensity and uncertainty (Jy).
    polang, polang_err : np.ndarray
        Polarisation angle and uncertainty (rad).
    comp : dict | None
        Optional comparison dictionary with keys:
        reffreq_I, stokesI, stokesI_err, lam2_pol, polint, polint_err, rm.
    nvss : dict | None
        Optional NVSS dictionary from `get_nvss_fluxes`.

    Returns
    -------
    Path
        Path to saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    # Stokes I
    ax1.errorbar(freqs, flux_I, yerr=unc_I, fmt="o", linestyle="none", label="measured")
    if comp and np.isfinite(comp.get("reffreq_I", np.nan)) and np.isfinite(comp.get("stokesI", np.nan)):
        ax1.errorbar(
            comp["reffreq_I"], comp["stokesI"], yerr=comp["stokesI_err"], fmt="D", mfc="none", mec="r", label="table"
        )
    if nvss:
        ax1.errorbar(nvss["freq"], nvss["flux_I"], fmt="x", label="NVSS")
    ax1.set_ylabel("Integrated flux [Jy]")
    ax1.set_title("Stokes I")
    ax1.grid(True)
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(freqs)
    ax1_top.set_xticklabels(channels)
    ax1_top.set_xlabel("Channel #")
    ax1.legend()

    # Polarised intensity
    ax2.errorbar(lam2, polint, yerr=polint_err, fmt="o", linestyle="none", label="measured")
    if comp and np.isfinite(comp.get("lam2_pol", np.nan)) and np.isfinite(comp.get("polint", np.nan)):
        ax2.errorbar(
            comp["lam2_pol"], comp["polint"], yerr=comp["polint_err"], fmt="D", mfc="none", mec="r", label="table"
        )
    if nvss:
        lam2_nvss = (c.value / 1.4e9) ** 2
        ax2.errorbar(lam2_nvss, nvss["flux_P"], fmt="x", label="NVSS")
    ax2.set_ylabel("Integrated flux [Jy]")
    ax2.set_title("Polarised intensity")
    ax2.grid(True)
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(lam2)
    ax2_top.set_xticklabels(channels)
    ax2_top.set_xlabel("Channel #")
    ax2.legend()

    # Stokes Q & U
    ax3.errorbar(freqs, flux_Q, yerr=unc_Q, fmt="s", linestyle="none", label="Q")
    ax3.errorbar(freqs, flux_U, yerr=unc_U, fmt="^", linestyle="none", label="U")
    if nvss:
        ax3.errorbar(nvss["freq"], nvss["flux_Q"], fmt="x", label="NVSS Q")
        ax3.errorbar(nvss["freq"], nvss["flux_U"], fmt="x", label="NVSS U")
    ax3.set_ylabel("Integrated flux [Jy]")
    ax3.set_title("Stokes Q & U")
    ax3.grid(True)
    ax3_top = ax3.twiny()
    ax3_top.set_xlim(ax3.get_xlim())
    ax3_top.set_xticks(freqs)
    ax3_top.set_xticklabels(channels)
    ax3_top.set_xlabel("Channel #")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.legend()

    # Polarisation angle (degrees) and simple RM fit
    polang_deg = np.degrees(polang)
    polang_err_deg = np.degrees(polang_err)
    ax4.errorbar(lam2, polang_deg, yerr=polang_err_deg, fmt="s", linestyle="none", label="measured")
    rad_unw = np.unwrap(polang)
    RM_fit, chi0_fit = np.polyfit(lam2, rad_unw, 1)
    fitdeg = np.degrees(RM_fit * lam2 + chi0_fit)
    ax4.plot(lam2, fitdeg, color="k", ls="--", label=f"fit: RM={RM_fit:.1f} rad/m^2, chi0={chi0_fit:.2f} rad")
    if comp and np.isfinite(comp.get("rm", np.nan)):
        chi0_comp = float(np.mean(rad_unw - comp["rm"] * lam2))
        tabdeg = np.degrees(comp["rm"] * lam2 + chi0_comp)
        ax4.plot(lam2, tabdeg, color="r", ls=":", label=f"table: RM={comp['rm']:.1f} rad/m^2, chi0={chi0_comp:.2f} rad")
    if nvss:
        nvss_angle = np.degrees(0.5 * np.arctan2(nvss["flux_U"], nvss["flux_Q"]))
        ax4.scatter((c.value / 1.4e9) ** 2, nvss_angle, marker="x", label="NVSS angle")
    ax4.set_ylabel("Polarisation angle [deg]")
    ax4.set_title("Polarisation angle")
    ax4.set_ylim(-110, 110)
    ax4.grid(True)
    ax4_top = ax4.twiny()
    ax4_top.set_xlim(ax4.get_xlim())
    ax4_top.set_xticks(lam2)
    ax4_top.set_xticklabels(channels)
    ax4_top.set_xlabel("Channel #")
    ax4.set_xlabel("Wavelength^2 [m^2]")
    ax4.legend()

    fig.tight_layout()
    out = output_dir / f"{prefix}_integratedflux.png"
    fig.savefig(out)
    print(f"Plot saved to {out}")
    # plt.show()
    plt.close(fig)
    return out


def compare_to_nvss(args: argparse.Namespace) -> None:
    """
    Main pipeline:
    - Collect files
    - Optionally compute NVSS comparison fluxes
    - Compute per-channel I/Q/U fluxes in region
    - Compute uncertainties (from PB correction if central channel unc provided)
    - Propagate uncertainties to P and angle
    - Flag channels and plot

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    prefix = args.ds9reg.stem

    # manual channel flags
    try:
        flag_chans: list[int] = ast.literal_eval(args.flag_chans)
        if not isinstance(flag_chans, list):
            raise ValueError
    except Exception:
        raise ValueError("--flag-chans must be a Python list literal like [4,5,6]")

    # collect fits lists
    ifiles, qfiles, ufiles = collect_files(args.i_glob, args.q_glob)
    pb_files = sorted(glob.glob(args.pbcor_glob))

    if not (len(pb_files) == len(qfiles) == len(ifiles)):
        raise AssertionError(
            f"pbcor files count must match Q and I files count. Instead len(pb)={len(pb_files)}, len(Q)={len(qfiles)}, len(I)={len(ifiles)}"
        )

    comp = None
    if args.comparetable and args.comparetable_idx is not None:
        # Placeholder for future comparison-table support.
        print("TODO: implement comparison to table")

    nvss_fluxes = get_nvss_fluxes(args, prefix) if args.comparenvssdirect else None

    # channel properties
    freqs = get_channel_frequencies(qfiles)
    channels = np.arange(len(freqs))
    lam2 = (c.value / freqs) ** 2

    # integrated fluxes (first region)
    flux_I, flux_Q, flux_U, beams_I, beams_Q, beams_U = compute_fluxes(
        ifiles, qfiles, ufiles, args.ds9reg
    )

    # uncertainties
    if args.chan_unc_center is not None:
        ra, dec = parse_region_center(args.ds9reg)
        unc0 = float(args.chan_unc_center)
        unc_I = compute_uncertainty_pbcor(unc0, pb_files, ra, dec)
        unc_Q = compute_uncertainty_pbcor(unc0, pb_files, ra, dec)
        unc_U = compute_uncertainty_pbcor(unc0, pb_files, ra, dec)
        # TODO: include additional terms (e.g. number-of-beams, calibration)
    else:
        unc_I = np.zeros(len(qfiles), dtype=float)
        unc_Q = unc_I.copy()
        unc_U = unc_I.copy()

    # propagate to P and psi
    I_u = unp.uarray(flux_I, unc_I)  # noqa: F841
    Q_u = unp.uarray(flux_Q, unc_Q)
    U_u = unp.uarray(flux_U, unc_U)
    P_u = unp.sqrt(Q_u**2 + U_u**2)
    psi_u = 0.5 * unp.arctan2(U_u, Q_u)

    polint = unp.nominal_values(P_u)
    polint_err = unp.std_devs(P_u)
    polang = unp.nominal_values(psi_u)
    polang_err = unp.std_devs(psi_u)

    # save data if requested
    if args.output_dir_data is not None:
        save_data(
            prefix,
            args.output_dir_data,
            freqs,
            flux_I,
            flux_Q,
            flux_U,
            unc_I,
            unc_Q,
            unc_U,
            beams_I,
            beams_Q,
            beams_U,
        )

    # build masks
    mask = ~np.isin(channels, flag_chans) if len(flag_chans) > 0 else np.ones_like(channels, dtype=bool)
    print(f"Found {np.sum(~mask)}/{len(mask)} flagged channels manually")

    nan_mask = np.isnan(flux_I) | np.isnan(flux_Q) | np.isnan(flux_U)
    print(f"Found {np.sum(nan_mask)}/{len(mask)} flagged channels because NaN")

    threshold = 1e3
    high_mask = (np.abs(flux_I) > threshold) | (np.abs(flux_Q) > threshold) | (np.abs(flux_U) > threshold)
    print(f"Found {np.sum(high_mask)}/{len(mask)} flagged channels because crazy-high values")

    if args.flag_by_noise is not None:
        raise NotImplementedError("Flag by noise not yet implemented")
        # Example (when implemented):
        # factor = float(args.flag_by_noise_factor)
        # tbl = Table.read(args.flag_by_noise)
        # mediannoise = np.nanmedian(tbl['noise_U'])
        # mask_by_noise = tbl['noise_U'] > factor * mediannoise
        # mask_indices = tbl['channel'][mask_by_noise]
        # print(f"Masking {len(mask_indices)} channels because higher than {factor:.1f}x median noise")
        # mask_for_noise = np.ones_like(channels, bool)
        # mask_for_noise[mask_indices] = False
        # mask &= ~(nan_mask | high_mask | ~mask_for_noise)
    else:
        mask &= ~(nan_mask | high_mask)

    print(f"Final number of good channels: {np.sum(mask)}/{len(mask)}")

    # plot
    plot_flux_vs_nvss(
        prefix,
        args.output_dir,
        freqs[mask],
        flux_I[mask],
        flux_Q[mask],
        flux_U[mask],
        unc_I[mask],
        unc_Q[mask],
        unc_U[mask],
        lam2[mask],
        channels[mask],
        polint[mask],
        polint_err[mask],
        polang[mask],
        polang_err[mask],
        comp,
        nvss_fluxes,
    )


def main() -> None:
    """
    Entry point for CLI execution.
    """
    args = parse_args()
    compare_to_nvss(args)


if __name__ == "__main__":
    main()