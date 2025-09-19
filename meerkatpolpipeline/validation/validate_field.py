#!/usr/bin/env python3
"""
Make spectral-summary plots (I, Q, U; P, p; Q vs U; insets) for the brightest 10 sources.

Usage
-----
python make_spectral_summary.py \
  --imageset-I /path/to/pickled_or_importable_imageset_I.pyobj \
  --imageset-Q /path/to/pickled_or_importable_imageset_Q.pyobj \
  --imageset-U /path/to/pickled_or_importable_imageset_U.pyobj \
  --regions /path/to/regions.reg \
  --catalog /path/to/pybdsf_table.fits \
  --output-prefix /path/to/out/summary

Notes
-----
- If you already have three ImageSet objects in scope, import this file as a module
  and call `plot_top_n_source_spectra(...)` directly.
- Frequencies are read from FITS headers. Common keyword patterns are tried
  (FREQ, CRVAL3 with CTYPE3 like 'FREQ', or WSCN3FRQ). All are assumed to be in Hz.
- For RA/DEC to pixel work (e.g., insets), WCS(header).celestial is used.
- The DS9 region i is asserted to correspond to catalog row i (centers matched).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from regions import Regions

from meerkatpolpipeline.check_nvss.target_vs_nvss import compute_fluxes
from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import PrintLogger
from meerkatpolpipeline.wsclean.wsclean import ImageSet

# -------------------- I/O & helpers --------------------

class ValidateFieldOptions(BaseOptions):
    """A basic class to handle validation plots options for meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Required parameter"""
    targetfield: str | None = None
    """name of targetfield. Propagated to all steps."""

def read_table(catalog_path: Path) -> Table:
    return Table.read(str(catalog_path))


def read_regions(ds9_path: Path) -> list[SkyCoord]:
    regs = Regions.read(str(ds9_path))
    centers = []
    for r in regs:
        # Most DS9/regions primitives expose a SkyCoord .center (FK5)
        if hasattr(r, "center") and isinstance(r.center, SkyCoord):
            centers.append(r.center.icrs)
        else:
            raise ValueError("Region without a SkyCoord center encountered.")
    return centers


def get_freq_hz_from_header(hdr: fits.Header) -> float | None:
    """
    Try several common patterns to obtain frequency (Hz) from a WSClean image.
    Returns None if not found.
    """
    # Direct
    if "FREQ" in hdr:
        return float(hdr["FREQ"])
    # Axis-based (CRVALn with CTYPE 'FREQ' or 'Frequency')
    for ax in (3, 4):
        ctype = hdr.get(f"CTYPE{ax}", "").upper()
        if "FREQ" in ctype or "FREQUENCY" in ctype:
            cunit = hdr.get(f"CUNIT{ax}", "Hz").lower()
            val = hdr.get(f"CRVAL{ax}")
            if val is not None:
                nu = float(val)
                if cunit in ("hz",):
                    return nu
                if cunit in ("khz",):
                    return nu * 1e3
                if cunit in ("mhz",):
                    return nu * 1e6
                if cunit in ("ghz",):
                    return nu * 1e9
                # fall back assuming Hz
                return nu
    # WSClean sometimes writes a helper keyword
    if "WSCN3FRQ" in hdr:
        return float(hdr["WSCN3FRQ"])
    return None


def sort_files_by_frequency(files: list[Path]) -> tuple[list[Path], list[float], list[int]]:
    """
    Returns channel files sorted by ascending frequency, dropping MFS or files without a frequency.
    Also returns the sorted frequencies and the indices kept.
    """
    freq_info = []
    for i, p in enumerate(files):
        with fits.open(p) as hdul:
            hdr = hdul[0].header
            # Heuristic: ignore MFS products
            if "MFS" in p.name.upper():
                continue
            nu = get_freq_hz_from_header(hdr)
            if nu is not None:
                freq_info.append((i, p, nu))
    if not freq_info:
        raise RuntimeError("Could not find any per-channel images with frequency keywords.")
    freq_info.sort(key=lambda t: t[2])
    kept_idx = [i for i, _, _ in freq_info]
    kept_files = [p for _, p, _ in freq_info]
    freqs = [nu for _, _, nu in freq_info]
    return kept_files, freqs, kept_idx


def first_mfs_file(files: list[Path]) -> Path | None:
    for p in files[::-1]:
        if "MFS" in p.name.upper():
            return p
    return None


def assert_region_matches_catalog(
    centers: list[SkyCoord], table: Table, tol_arcsec: float = 1.0
) -> None:
    if len(centers) < len(table):
        # PYBDSF often lists many sources. We only need that indices align up to len(centers)
        pass
    n = min(len(centers), len(table))
    for i in range(n):
        ra = float(table["RA"][i])
        dec = float(table["DEC"][i])
        cat = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        sep = centers[i].separation(cat).to(u.arcsec).value
        assert sep <= tol_arcsec, (
            f"Region index {i} center does not match catalog RA,DEC within "
            f"{tol_arcsec} arcsec (sep = {sep:.2f} arcsec)."
        )


def format_frequency_axes_ghz(axes: list[plt.Axes]) -> None:
    """
    Apply log x-scale in GHz with fixed-point tick labels to a list of axes.
    """
    for ax in axes:
        ax.set_xscale("log")

        # Major ticks: keep default LogFormatter
        ax.xaxis.set_major_formatter(mticker.LogFormatter(base=10))

        # Minor ticks: use same formatter but shrink font size
        ax.xaxis.set_minor_formatter(mticker.LogFormatter(base=10, labelOnlyBase=False)) # subs=[2,3,5]

        for tick in ax.xaxis.get_minor_ticks():
            tick.label1.set_fontsize(9)   # smaller size for minor ticks

        ax.set_xlabel("Frequency (GHz)")


# -------------------- fluxes & spectra --------------------

@dataclass
class SpectralSeries:
    nu_hz: np.ndarray
    I: np.ndarray  # noqa: E741
    Q: np.ndarray
    U: np.ndarray

    @property
    def P(self) -> np.ndarray:
        return np.sqrt(self.Q**2 + self.U**2)

    @property
    def pfrac(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            pf = np.where(self.I > 0, self.P / self.I, np.nan)
        return pf


def gather_flux_series(
    imageset_I: ImageSet,
    imageset_Q: ImageSet,
    imageset_U: ImageSet,
    ds9reg: Path,
    region_index: int,
    logger: callable,
) -> SpectralSeries:
    # Sort per-channel files by frequency and apply the same channel subset to I, Q, U.
    ifiles_all = list(imageset_I.image_pbcor)
    qfiles_all = list(imageset_Q.image_pbcor)
    ufiles_all = list(imageset_U.image_pbcor)

    ifiles, nu_I, kept = sort_files_by_frequency(ifiles_all)

    # Align Q and U by identical kept indices
    qfiles = [qfiles_all[i] for i in kept]
    ufiles = [ufiles_all[i] for i in kept]

    region_object = Regions.read(ds9reg)

    # Integrated fluxes, compute_fluxes is assumed to return integrated in Jy)
    logger.info(f"Gathering fluxes for {region_index=}")
    flux_I, flux_Q, flux_U, _, _, _ = compute_fluxes(
        ifiles, qfiles, ufiles, region_object, region_index
    )

    nu = np.asarray(nu_I, dtype=float)
    I = np.asarray(flux_I, dtype=float)  # noqa: E741
    Q = np.asarray(flux_Q, dtype=float)
    U = np.asarray(flux_U, dtype=float)

    # Remove NaNs consistently
    m = np.isfinite(nu) & np.isfinite(I) & np.isfinite(Q) & np.isfinite(U)
    return SpectralSeries(nu_hz=nu[m], I=I[m], Q=Q[m], U=U[m])


def fit_power_law_alpha(nu_hz: np.ndarray, S: np.ndarray) -> tuple[float, float]:
    """
    Fit S(nu) = S0 * (nu/nu0)^alpha in log10 space.
    Returns (alpha, S0_at_nu0) with nu0 = median frequency.
    """
    m = (S > 0) & np.isfinite(S) & np.isfinite(nu_hz)
    nu = nu_hz[m]
    s = S[m]
    if len(s) < 2:
        return np.nan, np.nan
    nu0 = np.median(nu)
    x = np.log10(nu / nu0)
    y = np.log10(s)
    coeffs = np.polyfit(x, y, 1)  # y = a*x + b
    alpha = float(coeffs[0])
    S0 = float(10 ** coeffs[1])
    return alpha, S0


def compute_fractional_residuals(series: SpectralSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (nu_hz_valid, S_valid, frac_residuals) where
    frac_residuals = (S - S_fit) / S_fit   with S_fit from the per-source power-law fit.
    """
    nu = series.nu_hz
    S  = series.I
    m  = np.isfinite(nu) & np.isfinite(S) & (S > 0)
    if np.count_nonzero(m) < 2:
        return nu[m], S[m], np.array([])
    alpha, S0 = fit_power_law_alpha(nu[m], S[m])
    if not np.isfinite(alpha) or not np.isfinite(S0):
        return nu[m], S[m], np.array([])
    nu0 = np.median(nu[m])
    Sfit = S0 * (nu[m] / nu0) ** alpha
    frac = (S[m] - Sfit) / Sfit
    return nu[m], S[m], frac


# -------------------- plotting --------------------

def add_inset_cutouts(
    ax_grid: dict[str, plt.Axes],
    imageset_I: ImageSet,
    imageset_Q: ImageSet,
    imageset_U: ImageSet,
    center: SkyCoord,
    cutout_size_pix: int = 40,
) -> None:
    """Optional insets: Stokes I (MFS) and Polarized Intensity (from MFS Q,U)."""

    VMIN_HARDCODED = -1e-4

    im_I = first_mfs_file(list(imageset_I.image_pbcor))
    im_Q = first_mfs_file(list(imageset_Q.image_pbcor))
    im_U = first_mfs_file(list(imageset_U.image_pbcor))

    if im_I is None:
        return

    with fits.open(im_I) as hdulI:
        dataI = hdulI[0].data.squeeze()
        wcsI = WCS(hdulI[0].header).celestial
        cutI = Cutout2D(dataI, position=center, wcs=wcsI, size=(cutout_size_pix, cutout_size_pix))
        ax = ax_grid["inset_I"]
        norm = ImageNormalize(vmin=VMIN_HARDCODED, vmax=np.nanmax(cutI.data), stretch=AsinhStretch())
        im = ax.imshow(cutI.data, origin="lower", norm=norm)  # noqa: F841
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("[Jy/beam]", fontsize=8)
        ax.set_title("Stokes I", fontsize=8)
        ax.scatter([cutout_size_pix / 2], [cutout_size_pix / 2], marker="+", s=30, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    if im_Q is not None and im_U is not None:
        with fits.open(im_Q) as hdulQ, fits.open(im_U) as hdulU:
            dataQ = hdulQ[0].data.squeeze()
            dataU = hdulU[0].data.squeeze()
            wcs = WCS(hdulQ[0].header).celestial
            cutQ = Cutout2D(dataQ, position=center, wcs=wcs, size=(cutout_size_pix, cutout_size_pix))
            cutU = Cutout2D(dataU, position=center, wcs=wcs, size=(cutout_size_pix, cutout_size_pix))
            Pimg = np.sqrt(cutQ.data**2 + cutU.data**2)
            ax = ax_grid["inset_P"]
            norm = ImageNormalize(vmin=VMIN_HARDCODED, vmax=np.nanmax(Pimg.data), stretch=AsinhStretch())
            im = ax.imshow(Pimg, origin="lower", norm=norm)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("[Jy/beam]", fontsize=8)
            ax.set_title("Polarized Intensity", fontsize=8)
            ax.scatter([cutout_size_pix / 2], [cutout_size_pix / 2], marker="+", s=30, color='red')
            ax.set_xticks([])
            ax.set_yticks([])


def make_summary_figure(
    name: str,
    series: SpectralSeries,
    center: SkyCoord,
    imageset_I: ImageSet,
    imageset_Q: ImageSet,
    imageset_U: ImageSet,
    cutout_size_pix: int,
) -> plt.Figure:
    """
    Make a 3x3 summary figure for one source.
    """
    fig = plt.figure(figsize=(12.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax_I = fig.add_subplot(gs[0, 0])
    # gs[0,1] is where V would be in the reference figure;
    ax_inset_P = fig.add_subplot(gs[0, 1])  
    ax_inset_I = fig.add_subplot(gs[0, 2])
    ax_Q = fig.add_subplot(gs[1, 0])
    ax_U = fig.add_subplot(gs[1, 1])
    ax_qu = fig.add_subplot(gs[1, 2])
    ax_P = fig.add_subplot(gs[2, 0])
    ax_pfrac = fig.add_subplot(gs[2, 1])
    # gs[2,2] (Leakage in the example) is intentionally left unused

    # Frequencies and labels
    nu = series.nu_hz
    ghz = nu / 1e9

    # Stokes I with power-law fit
    ax_I.scatter(ghz, series.I, s=10)
    ax_I.set_yscale("log")
    ax_I.set_xlabel("Frequency (GHz)")
    ax_I.set_ylabel("Flux (Jy)")
    ax_I.set_title("Stokes I", fontsize=9)
    ax_I.set_xscale("log")




    alpha, S0 = fit_power_law_alpha(nu, series.I)
    if np.isfinite(alpha):
        nu0 = np.median(nu)
        xx = np.linspace(nu.min(), nu.max(), 200)
        yy = S0 * (xx / nu0) ** alpha
        ax_I.plot(xx/1e9, yy, lw=1)
        txt = f"I( nu0 ) = {S0:.3g} Jy\nnu0 = {nu0/1e9:.3f} GHz\nalpha(fit) = {alpha:.2f}"
        ax_I.text(0.02, 0.02, txt, transform=ax_I.transAxes, fontsize=8, va="bottom")

    # Q, U
    ax_Q.scatter(ghz, series.Q, s=10)
    ax_Q.set_xscale("log")
    ax_Q.set_xlabel("Frequency (GHz)")
    ax_Q.set_ylabel("Flux (Jy/beam)")
    ax_Q.set_title("Stokes Q", fontsize=9)

    ax_U.scatter(ghz, series.U, s=10)
    ax_U.set_xscale("log")
    ax_U.set_xlabel("Frequency (GHz)")
    ax_U.set_ylabel("Flux (Jy/beam)")
    ax_U.set_title("Stokes U", fontsize=9)

    # Q vs U, color by frequency
    sc = ax_qu.scatter(series.Q, series.U, c=ghz, s=10, cmap='rainbow')
    ax_qu.set_xlabel("Flux (Jy/beam)")
    ax_qu.set_ylabel("Flux (Jy/beam)")
    ax_qu.set_title("Stokes Q vs U", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax_qu)
    cbar.set_label("Frequency (GHz)")

    # Polarized intensity P and fraction p
    P = series.P
    pfrac = series.pfrac

    ax_P.scatter(ghz, P, s=10)
    ax_P.set_xscale("log")
    ax_P.set_xlabel("Frequency (GHz)")
    ax_P.set_ylabel("Flux (Jy/beam)")
    ax_P.set_title("Polarized intensity", fontsize=9)

    ax_pfrac.scatter(ghz, pfrac, s=10)
    ax_pfrac.set_xscale("log")
    ax_pfrac.set_xlabel("Frequency (GHz)")
    ax_pfrac.set_ylabel("P/I (scalar)")
    ax_pfrac.set_title("Polarized fraction", fontsize=9)

    # Apply log-x GHz formatting to all frequency-x panels at once
    # i.e. no scientific notation but simple scalar formatting. Requires all plots to be in GHz!
    format_frequency_axes_ghz([ax_I, ax_Q, ax_U, ax_P, ax_pfrac])

    # Insets (I MFS and P from Q,U MFS) with center marker
    add_inset_cutouts(
        {"inset_I": ax_inset_I, "inset_P": ax_inset_P},
        imageset_I, imageset_Q, imageset_U, center=center,
        cutout_size_pix=cutout_size_pix
    )

    fig.suptitle(name, y=0.98, fontsize=11)
    return fig


def plot_all_I_spectra(
    named_series: list[tuple[str, SpectralSeries]],
    output_path: Path,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=(9.0, 8.0), constrained_layout=True)
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # --- Top: Stokes I spectra ---
    for (name, series, idx) in named_series:
        nu_ghz = series.nu_hz / 1e9
        m = np.isfinite(nu_ghz) & np.isfinite(series.I) & (series.I > 0)
        if not np.any(m):
            continue
        ax_top.scatter(nu_ghz[m], series.I[m], s=10, label=name)
        ax_top.plot(nu_ghz[m], series.I[m], lw=0.8, alpha=0.6)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Flux (Jy)")
    ax_top.set_title("Top sources: Stokes I spectra")
    ax_top.legend(fontsize=8, ncols=2, loc="best")

    # Plot a reference line with a spectral index of -0.7 at the mean flux density
    mean_flux = np.mean([series.I[m].mean() for _, series in named_series if np.any(m)])
    if np.isfinite(mean_flux):
        nu0 = np.median([series.nu_hz[m].mean()/1e9 for _, series in named_series if np.any(m)])
        xx = np.linspace(nu_ghz.min()*0.9, nu_ghz.max()*1.1, 200)  # GHz range for the reference line
        yy = mean_flux * (xx / nu0) ** -0.7
        ax_top.plot(xx, yy, color="black", linestyle="--", label="Reference: α = -0.7")

    # --- Bottom: fractional residuals per source ---
    for (name, series, idx) in named_series:
        nu_hz, _, frac = compute_fractional_residuals(series)
        if frac.size == 0:
            continue
        ax_bot.scatter(nu_hz / 1e9, frac, s=10, label=name, alpha=0.8)

    ax_bot.set_xscale("log")
    ax_bot.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_bot.set_xlabel("Frequency (GHz)")
    ax_bot.set_ylabel("(S - S_fit) / S_fit")

    # format axis ticks to 2 decimal places and scalar values instead of scientific notation
    format_frequency_axes_ghz([ax_bot])

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

# -------------------- top-N driver --------------------

def plot_top_n_source_spectra(
    imageset_I: ImageSet,
    imageset_Q: ImageSet,
    imageset_U: ImageSet,
    ds9_regions: Path,
    catalog_path: Path,
    output_prefix: Path,
    top_n: int = 10,
    logger: callable | None = None,
) -> list[Path]:
    """
    Plot top N source spectra summaries, by Total_flux from a PYBDSF catalog, using regions for flux extraction.
    """
    named_series: list[tuple[str, SpectralSeries, int]] = []

    if logger is None:
        logger = PrintLogger()

    # Inputs
    table = read_table(catalog_path)
    centers = read_regions(ds9_regions)
    regions = Regions.read(str(ds9_regions))

    # For wcs to pixels using WCS of one MFS I image
    im_I = first_mfs_file(list(imageset_I.image_pbcor))
    with fits.open(im_I) as hdulI:
        wcsI = WCS(hdulI[0].header).celestial
        pixscale_deg = np.abs(wcsI.proj_plane_pixel_scales()[0].to(u.deg).value)

    # Sanity check mapping (region i <-> row i)
    assert_region_matches_catalog(centers, table, tol_arcsec=1.5)

    # Brightest N by Total_flux
    if "Total_flux" not in table.colnames:
        raise KeyError("Catalog must contain 'Total_flux' column.")
    order = np.argsort(np.asarray(table["Total_flux"]))[::-1]
    keep = order[: min(top_n, len(order))]

    out_paths: list[Path] = []
    for rank, idx in enumerate(keep, start=1):
        center = SkyCoord(
            ra=float(table["RA"][idx]) * u.deg,
            dec=float(table["DEC"][idx]) * u.deg,
            frame="icrs",
        )

        series = gather_flux_series(
            imageset_I=imageset_I,
            imageset_Q=imageset_Q,
            imageset_U=imageset_U,
            ds9reg=ds9_regions,
            region_index=int(idx),
            logger=logger,
        )

        name = f"Rank {rank} | index {int(idx)} | Total_flux = {1000*table['Total_flux'][idx]:.3f} mJy"
        named_series.append((name, series, int(idx)))

        region = regions[int(idx)]
        # Major axis in degrees
        if hasattr(region, "width") and hasattr(region, "height"):
            major_deg = max(region.width.to(u.deg).value, region.height.to(u.deg).value)
        else:
            raise ValueError("Region has no width/height; cannot scale cutout size.")
        cutout_size_pix = int(np.ceil(1.5 * major_deg / pixscale_deg))

        fig = make_summary_figure(
            name=name,
            series=series,
            center=center,
            imageset_I=imageset_I,
            imageset_Q=imageset_Q,
            imageset_U=imageset_U,
            cutout_size_pix=cutout_size_pix,
        )

        out_png = Path(f"{output_prefix}_rank{rank:02d}_idx{int(idx)}.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        out_paths.append(out_png)

    # --- one combined plot for all Stokes I spectra ---
    combined_png = Path(f"{output_prefix}_spectral-summary-top{len(named_series)}.png")
    plot_all_I_spectra(named_series, combined_png)
    out_paths.append(combined_png)

    # Residual sum vs major axis
    # Rebuild the regions list in the *same* order as named_series:
    all_regions = Regions.read(str(ds9_regions))
    regions_in_order = regions_in_order = [all_regions[i] for _, _, i in named_series]
    
    res_png = Path(f"{output_prefix}_residual-sum_vs_majoraxis_top{len(named_series)}.png")
    plot_sum_residuals_vs_major(named_series, regions_in_order, res_png)
    out_paths.append(res_png)

    return out_paths


def plot_sum_residuals_vs_major(
    named_series: list[tuple[str, SpectralSeries]],
    regions: list,  # Regions from regions.read(...)
    output_path: Path,
) -> Path:
    """
    For each source, compute sum over channels of |(S - S_fit)/S_fit| and plot vs region major axis.
    Major axis is max(width, height); units on x-axis are arcsec.
    """
    import matplotlib.pyplot as plt
    from astropy import units as u

    majors_arcsec: list[float] = []
    sums_abs_frac: list[float] = []
    labels: list[str] = []

    for (name, series), reg in zip(named_series, regions):
        # Major axis (arcsec)
        if not (hasattr(reg, "width") and hasattr(reg, "height")):
            continue
        major = max(reg.width.to(u.arcsec).value, reg.height.to(u.arcsec).value)
        nu_hz, _, frac = compute_fractional_residuals(series)
        if frac.size == 0:
            continue
        majors_arcsec.append(float(major))
        sums_abs_frac.append(float(np.nansum(np.abs(frac))))
        labels.append(name)

    if not majors_arcsec:
        # Nothing to plot
        return output_path

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    sc = ax.scatter(majors_arcsec, sums_abs_frac, s=25)

    ax.set_xlabel("Region major axis (arcsec)")
    ax.set_ylabel("Sum |(S - S_fit) / S_fit|")
    ax.set_title("Spectral fit residual sum vs source size")

    # Optional: annotate a few largest outliers for quick triage
    try:
        idx = np.argsort(sums_abs_frac)[-3:]
        for i in idx:
            ax.annotate(labels[i], (majors_arcsec[i], sums_abs_frac[i]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    except Exception:
        pass

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# -------------------- CLI --------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create spectral-summary plots for bright sources.")
    p.add_argument("--regions", type=Path, required=True, help="DS9 .reg file (FK5).")
    p.add_argument("--catalog", type=Path, required=True, help="PYBDSF catalog FITS table.")
    p.add_argument("--output-prefix", type=Path, required=True, help="Prefix for output PNGs.")
    p.add_argument("--top-n", type=int, default=10, help="Number of top sources to plot.")
    # For typical usage you pass three importable imagesets; if they are already in scope,
    # import this as a module and call plot_top_n_source_spectra(...) directly.
    p.add_argument("--imageset-I", type=str, required=False, help="Python import path to an ImageSet named 'imageset_I'")
    p.add_argument("--imageset-Q", type=str, required=False, help="Python import path to an ImageSet named 'imageset_Q'")
    p.add_argument("--imageset-U", type=str, required=False, help="Python import path to an ImageSet named 'imageset_U'")
    return p.parse_args()


def _import_imageset(import_path: str, var_name: str) -> ImageSet:
    """
    Import a module and fetch an ImageSet object from a variable by name.
    This avoids serializing pickles on the CLI. Example:
      --imageset-I mypkg.myimagesets:imageset_I
    """
    if ":" not in import_path:
        raise ValueError(f"Expected 'module.path:varname', got {import_path}")
    mod_path, obj_name = import_path.split(":", 1)
    mod = __import__(mod_path, fromlist=[obj_name])
    obj = getattr(mod, obj_name)
    if not isinstance(obj, ImageSet):
        raise TypeError(f"{var_name} is not an ImageSet.")
    return obj


if __name__ == "__main__":
    args = _parse_args()

    # Load ImageSets either from import paths or raise instructive error
    if args.imageset_I and args.imageset_Q and args.imageset_U:
        imageset_I = _import_imageset(args.imageset_I, "imageset_I")
        imageset_Q = _import_imageset(args.imageset_Q, "imageset_Q")
        imageset_U = _import_imageset(args.imageset_U, "imageset_U")
    else:
        raise SystemExit(
            "Please provide --imageset-I, --imageset-Q, --imageset-U as 'module.path:varname'. "
            "Alternatively, import this module and call plot_top_n_source_spectra(...) from Python."
        )

    outputs = plot_top_n_source_spectra(
        imageset_I=imageset_I,
        imageset_Q=imageset_Q,
        imageset_U=imageset_U,
        ds9_regions=args.regions,
        catalog_path=args.catalog,
        output_prefix=args.output_prefix,
        top_n=args.top_n,
    )
    print("Wrote:")
    for p in outputs:
        print(p)
