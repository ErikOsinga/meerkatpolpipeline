from __future__ import annotations  # noqa: I001

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import (
    AsinhStretch,
    AsymmetricPercentileInterval,
    ImageNormalize,
    LinearStretch,
    ZScaleInterval,
)
from astropy.wcs import WCS
from matplotlib.colors import TwoSlopeNorm

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import PrintLogger, _get_option
from meerkatpolpipeline.scienceplots.science_utils import runningstatistics, RunningStatisticsResult


# Set plotting style. Avoid requiring an external LaTeX installation; use mathtext with serif fonts.
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

RM_COL_OPTIONS = ["RM", "RM_rad_m2", "rm", "RM_obs", "RM_obs_rad_m2"]
RM_ERR_COL_OPTIONS = ["e_RM", "RM_ERR", "RM_err", "dRM", "ERR_RM", "RM_err"]
RA_COL_OPTIONS = ["RA", "ra", "RA_deg", "RAJ2000", "optRA"]
DEC_COL_OPTIONS = ["DEC", "Dec", "dec", "DEC_deg", "DEJ2000", "optDec"]

class ScienceRMSynth1DOptions(BaseOptions):
    """A basic class to handle options for science plots after 1D RM synthesis. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""
    snr_threshold: float = 7.0
    """SNR threshold in polarized intensity for making science plots. Default 7.0"""
    z_cluster: float | None = None
    """Redshift of the cluster."""
    bubble_scaling_factor: float = 1e0
    """Scaling factor (division) for bubble sizes in RM bubble plot. Default 1."""
    bubble_scale_function: str = "linear"
    """Scaling function for bubble sizes in RM bubble plot. Default 'linear', also supports 'quadratic'."""
    grm_correction: str | None = None
    """Type of Galactic RM correction applied, if any. Options: None/null, 'hutschenreuter', 'annulus_method'."""
    running_scatter_method: str = "iqr"
    """Method for running scatter calculation. Options: 'std', 'iqr', 'mad'."""
    running_scatter_window_arcmin: float | None = None
    """Window size in arcmin for running scatter vs radius plot. If null, uses fixed-count window."""
    running_scatter_Npoints: int | None = 20
    """Number of points in fixed-count window for running scatter vs radius plot. If null, uses window size."""
    running_scatter_nbootstrap: int | None = None
    """Number of bootstrap resamples for running scatter uncertainty estimation. If null, no bootstrap."""


def _find_col(tbl: Table, candidates):
    """Return first matching column name in `tbl` from a list of `candidates`."""
    for name in candidates:
        if name in tbl.colnames:
            return name
    return None

def _as_quantity(col):
    """Ensure an astropy Quantity for RM-like columns if units are encoded in the name."""
    # If the column is already a Quantity, return it as-is
    try:
        import astropy.units as u
        if hasattr(col, "unit") and (col.unit is not None):
            return col
        # Heuristic: treat as dimensionless number with RM units attached
        return col * (u.rad / u.m**2)
    except Exception:
        return col


def _resolve_rm_columns(tab: Table, science_options):
    """
    Decide which RM columns to use based on `grm_correction`.
    Returns: (rm_col, err_col, title_suffix, file_suffix)
    """
    grm = _get_option(science_options, "grm_correction", None)
    if grm is None:
        rm_col  = _find_col(tab, RM_COL_OPTIONS)
        err_col = _find_col(tab, RM_ERR_COL_OPTIONS)
        if rm_col is None:
            raise KeyError(
                f"Could not find RM column. Tried: {RM_COL_OPTIONS}."
            )
        return rm_col, err_col, "", ""  # no suffixes
    
    grm_l = str(grm).lower()
    if grm_l == "hutschenreuter":
        if "rrm_huts" not in tab.colnames:
            raise KeyError("grm_correction='hutschenreuter' but column 'rrm_huts' not found.")
        rm_col = "rrm_huts"
        err_col = "rrm_huts_err" if "rrm_huts_err" in tab.colnames else None
        return rm_col, err_col, " (Hutschenreuter GRM corrected)", "_grm-hutschenreuter"
    else:
        # Anything else: not implemented (for now)
        raise NotImplementedError(f"grm_correction={grm!r} is not implemented.")


def _scale_scatter_method_to_sigma(method: str) -> float:
    """
    Return the scaling from a robust statistic to sigma, assuming a Gaussian distribution.
    """
    if method == 'iqr':
        scale_by = 1.349
    elif method == 'mad':
        scale_by = 1.4826
    elif method == 'std':
        scale_by = 1.0
    return scale_by


def plot_rm_vs_radius(
    science_options: dict | ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    center_coord: SkyCoord,
    plot_dir: Path,
    logger=None,
):
    """
    Plot RM vs radius, both in degrees and kpc (if redshift is provided).
    """

    if logger is None:
        logger = PrintLogger()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Read catalogue
    tab = Table.read(str(rms1d_catalog))

    # apply SNR threshold
    tab = tab[tab["SNR_PI"] > science_options['snr_threshold']]
    logger.info(f"Selected {len(tab)} sources with SNR_PI > {science_options['snr_threshold']}")

    # Detect columns
    ra_col  = _find_col(tab, RA_COL_OPTIONS)
    dec_col = _find_col(tab, DEC_COL_OPTIONS)

    rm_col, err_col, title_suffix, file_suffix = _resolve_rm_columns(tab, science_options)

    if ra_col is None or dec_col is None:
        raise KeyError(
            "Could not find RA/Dec columns. Tried: "
            f"{RA_COL_OPTIONS} and {DEC_COL_OPTIONS}"
        )

    # Sky positions and separations
    coords = SkyCoord(tab[ra_col] * u.deg, tab[dec_col] * u.deg, frame="icrs")
    sep = coords.separation(center_coord)
    r_arcmin = sep.to(u.arcmin).value

    # RM and optional uncertainties
    RM = _as_quantity(tab[rm_col]).to(u.rad / u.m**2).value
    RM_err = None
    if err_col is not None:
        try:
            RM_err = _as_quantity(tab[err_col]).to(u.rad / u.m**2).value
        except Exception:
            RM_err = np.asarray(tab[err_col])

    # Cosmology conversion (if z given)
    z = science_options['z_cluster']
    have_kpc_axis = (z is not None)
    if have_kpc_axis:
        DA = Planck18.angular_diameter_distance(float(z)).to(u.kpc)  # kpc
        kpc_per_arcmin = (1.0 * u.arcmin).to(u.rad) * DA
        kpc_per_arcmin = kpc_per_arcmin.value

        def arcmin_to_kpc(x):
            return x * kpc_per_arcmin

        def kpc_to_arcmin(x):
            return x / kpc_per_arcmin


    fig, ax = plt.subplots(figsize=(5.0, 3.8))  # compact

    # Scatter (with errorbars if available)
    if RM_err is not None:
        ax.errorbar(
            r_arcmin, RM, yerr=RM_err, fmt="o", ms=3.5, lw=1.0, elinewidth=0.8,
            capsize=1.8, alpha=0.9,
        )
    else:
        ax.plot(r_arcmin, RM, "o", ms=3.5, alpha=0.9)

    ax.set_xlabel(r"Radius to centre ($\mathrm{arcmin}$)")
    if title_suffix == "":
        ax.set_ylabel(r"$\mathrm{RM}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")
    else:
        ax.set_title(title_suffix)
        ax.set_ylabel(r"$\mathrm{RM}_{\mathrm{corr}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")

    # Secondary x-axis in kpc if z is provided
    if have_kpc_axis:
        secax = ax.secondary_xaxis("top", functions=(arcmin_to_kpc, kpc_to_arcmin))
        secax.set_xlabel(
            r"Radius to centre ($\mathrm{kpc}$)"
            + f"  [Planck18, $z={float(z):.3f}$]"
        )

    # Light grid, tight layout
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)
    fig.tight_layout()

    # Filenames
    field = getattr(science_options, "targetfield", None) or "field"
    base = f"rm_vs_radius_{field}"
    if have_kpc_axis:
        base += f"_z{float(z):.3f}"
    base += file_suffix  # add '' or '_grm-hutschenreuter'

    png_path = plot_dir / f"{base}.png"
    pdf_path = pdf_dir / f"{base}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved RM vs radius plot: {png_path.name} and {pdf_path.name}")

    return {"png": png_path, "pdf": pdf_path}


def plot_rm_bubble_map(
    science_options: dict | ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    center_coord: SkyCoord,
    plot_dir: Path,
    logger=None,
):
    """
    Bubble map of RM at source sky positions (RA/Dec).
    Color: blue (RM<0) to red (RM>0); size ~ (|RM|^p) * bubble_scaling_factor.
    Saves PNG to `plot_dir` and PDF to `plot_dir/pdfs`.
    """
    if logger is None:
        logger = PrintLogger()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Read catalogue
    tab = Table.read(str(rms1d_catalog))

    # Apply SNR cut
    snr_thr = float(_get_option(science_options, "snr_threshold", 7.0))
    if "SNR_PI" not in tab.colnames:
        raise KeyError("Required column 'SNR_PI' not found for SNR filtering.")
    tab = tab[tab["SNR_PI"] > snr_thr]
    logger.info(f"Selected {len(tab)} sources with SNR_PI > {snr_thr}")

    # Detect columns
    ra_col  = _find_col(tab, RA_COL_OPTIONS)
    dec_col = _find_col(tab, DEC_COL_OPTIONS)

    rm_col, err_col, title_suffix, file_suffix = _resolve_rm_columns(tab, science_options)


    if ra_col is None or dec_col is None:
        raise KeyError(
            "Could not find RA/Dec columns. Tried: "
            "RA/ra/RA_deg/RAJ2000/optRA and DEC/Dec/dec/DEC_deg/DEJ2000/optDec."
        )
    if rm_col is None:
        raise KeyError(
            "Could not find RM column. Tried: RM/RM_rad_m2/rm/RM_obs/RM_obs_rad_m2."
        )

    # Coordinates
    ra  = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)

    # RM (+ optional uncertainty, not drawn but may be useful later)
    RM = _as_quantity(tab[rm_col]).to(u.rad / u.m**2).value
    if err_col is not None:
        try:
            RM_err = _as_quantity(tab[err_col]).to(u.rad / u.m**2).value
        except Exception:
            RM_err = np.asarray(tab[err_col])
    else:
        RM_err = None  # noqa: F841

    # Bubble sizes
    scale_func = str(_get_option(science_options, "bubble_scale_function")).lower()
    power = 1 if scale_func == "linear" else 2
    scaling = float(_get_option(science_options, "bubble_scaling_factor"))
    sizes = (np.abs(RM) ** power) / scaling
    # ensure a visible minimum size
    sizes = np.where(np.isfinite(sizes), sizes, 0.0)
    # sizes = np.maximum(sizes, 10.0)

    # Color mapping: blue (neg) -> white (0) -> red (pos)
    finite_rm = RM[np.isfinite(RM)]
    if finite_rm.size == 0:
        vabs = 1.0
    else:
        vabs = np.nanmax(np.abs(finite_rm))
        if not np.isfinite(vabs) or vabs == 0:
            vabs = 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=+vabs)
    cmap = "coolwarm"

    # Figure
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    sc = ax.scatter(
        ra, dec,
        c=RM, s=sizes,
        cmap=cmap, norm=norm,
        linewidths=0.4, edgecolors="k", alpha=0.9,
    )

    # Mark cluster centre
    ax.plot(center_coord.ra.deg, center_coord.dec.deg, marker="*", ms=10, mec="k", mfc="none", lw=1.0)

    # Axes and labels
    ax.set_xlabel(r"$\mathrm{RA}\;(\deg)$")
    ax.set_ylabel(r"$\mathrm{Dec}\;(\deg)$")
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)

    # Astronomical convention: RA increases to the left
    ax.invert_xaxis()

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    if title_suffix == "":
        cbar.set_label(r"$\mathrm{RM}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")
    else:
        cbar.set_label(r"$\mathrm{RM}_{\mathrm{corr}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")


    # Title 
    field = _get_option(science_options, "targetfield", "field")
    z = _get_option(science_options, "z_cluster", None)
    if z is not None:
        ax.set_title(f"{field} — RM bubble map (z={float(z):.3f}){title_suffix}")
    else:
        ax.set_title(f"{field} — RM bubble map{title_suffix}")

    fig.tight_layout()

    # Outputs
    base = f"rm_bubble_map_{field}{file_suffix}"

    png_path = plot_dir / f"{base}.png"
    pdf_path = pdf_dir / f"{base}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved RM bubble map: {png_path.name} and {pdf_path.name}")

    return {"png": png_path, "pdf": pdf_path}


def plot_rm_bubble_map_on_stokesI(
    science_options: dict | ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    stokesI_MFS: Path,
    center_coord: SkyCoord,
    plot_dir: Path,
    logger=None,
):
    """
    Bubble map of RM at source sky positions (RA/Dec) overlaid on a Stokes-I MFS image.
    Color: blue (RM<0) to red (RM>0); bubble area ~ (|RM|^p) * bubble_scaling_factor.
    Saves PNG to `plot_dir` and PDF to `plot_dir/pdfs`.
    """
    if logger is None:
        logger = PrintLogger()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # --- Read catalogue and apply cuts
    tab = Table.read(str(rms1d_catalog))

    snr_thr = float(_get_option(science_options, "snr_threshold", 7.0))
    if "SNR_PI" not in tab.colnames:
        raise KeyError("Required column 'SNR_PI' not found for SNR filtering.")
    tab = tab[tab["SNR_PI"] > snr_thr]
    logger.info(f"Selected {len(tab)} sources with SNR_PI > {snr_thr}")

    # RA/Dec detection
    ra_col  = _find_col(tab, RA_COL_OPTIONS)
    dec_col = _find_col(tab, DEC_COL_OPTIONS)
    if ra_col is None or dec_col is None:
        raise KeyError(
            "Could not find RA/Dec columns. Tried: "
            "RA/ra/RA_deg/RAJ2000/optRA and DEC/Dec/dec/DEC_deg/DEJ2000/optDec."
        )

    # RM columns (GRM correction aware)
    rm_col, err_col, title_suffix, file_suffix = _resolve_rm_columns(tab, science_options)

    # Values
    ra  = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)
    RM  = _as_quantity(tab[rm_col]).to(u.rad / u.m**2).value

    # Bubble sizes
    scale_func = str(_get_option(science_options, "bubble_scale_function", "linear")).lower()
    power = 1 if scale_func == "linear" else 2
    scaling = float(_get_option(science_options, "bubble_scaling_factor", 1e4))
    sizes = (np.abs(RM) ** power) / scaling
    sizes = np.where(np.isfinite(sizes), sizes, 0.0)
    # sizes = np.maximum(sizes, 10.0)

    # Color normalization for RM
    finite_rm = RM[np.isfinite(RM)]
    vabs = np.nanmax(np.abs(finite_rm)) if finite_rm.size else 1.0
    if not np.isfinite(vabs) or vabs == 0:
        vabs = 1.0
    rm_norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=+vabs)
    rm_cmap = "coolwarm"

    # --- Read Stokes-I MFS image & WCS
    stokesI_MFS = Path(stokesI_MFS)
    if not stokesI_MFS.exists():
        raise FileNotFoundError(f"Stokes I MFS file not found: {stokesI_MFS}")

    with fits.open(stokesI_MFS) as hdul:
        # find first HDU with image data
        ihdu = None
        for h in hdul:
            if (h.data is not None) and (h.data.size > 0):
                ihdu = h
                break
        if ihdu is None:
            raise ValueError(f"No image data found in {stokesI_MFS}")

        data = ihdu.data
        hdr = ihdu.header

    # Squeeze and pick a 2D plane if needed (take the first along extra axes)
    data = np.squeeze(data)
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError("Image is not 2D after squeezing.")

    # Celestial WCS
    wcs = WCS(hdr).celestial

    # Display scaling: asinh with robust percentile cuts
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")
    
    # Robust percentiles for stretch
    # p_lo, p_hi = AsymmetricPercentileInterval(1, 99.5).get_limits(data[finite])
    # norm = ImageNormalize(vmin=p_lo, vmax=p_hi, stretch=AsinhStretch(a=0.02))

    # DS9 default contrast is ~0.25
    zint = ZScaleInterval(contrast=0.25)
    vmin, vmax = zint.get_limits(data[finite])
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    # --- Plot
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = plt.subplot(111, projection=wcs)
    im = ax.imshow(data, origin="lower", cmap="gray", norm=norm)  # noqa: F841

    # Scatter bubbles at world coords
    sc = ax.scatter(
        ra, dec,
        c=RM, s=sizes,
        cmap=rm_cmap, norm=rm_norm,
        linewidths=0.4, edgecolors="k", alpha=0.9,
        transform=ax.get_transform("world"),
    )

    # Mark cluster centre
    ax.plot(
        center_coord.ra.deg, center_coord.dec.deg,
        marker="*", ms=10, mec="k", mfc="none", lw=1.0,
        transform=ax.get_transform("world"),
    )

    # Axes & grid (WCSAxes handles labels/units)
    ax.grid(color="white", alpha=0.2, ls=":", lw=0.8)

    # Colorbar for RM
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.045)
    if title_suffix:
        cbar.set_label(r"$\mathrm{RM}_{\mathrm{corr}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")
    else:
        cbar.set_label(r"$\mathrm{RM}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")

    # Title
    field = _get_option(science_options, "targetfield", "field")
    z = _get_option(science_options, "z_cluster", None)
    if z is not None:
        ax.set_title(f"{field} — RM bubble map on Stokes I (z={float(z):.3f}){title_suffix}")
    else:
        ax.set_title(f"{field} — RM bubble map on Stokes I{title_suffix}")

    fig.tight_layout()

    # Outputs
    base = f"rm_bubble_on_stokesI_{field}{file_suffix}"
    png_path = plot_dir / f"{base}.png"
    pdf_path = pdf_dir / f"{base}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved RM bubble map on Stokes I: {png_path.name} and {pdf_path.name}")

    return {"png": png_path, "pdf": pdf_path}


def running_scatter_vs_radius(
    science_options: dict | ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    center_coord: SkyCoord,
    plot_dir: Path,
    logger=None,
) -> None:
    """
    Plot running scatter of RM vs radius.

    Uses the chosen robust statistic ('std'|'iqr'|'mad') per window and converts it to an
    equivalent Gaussian sigma via `_scale_scatter_method_to_sigma`. If bootstrap
    is requested in `running_scatter_nbootstrap`, error bars are derived from the
    16th-84th percentiles of the bootstrap distribution (per bin). If measurement
    errors are available in the catalogue, they are propagated via
    sigma_intr^2 = sigma_est^2 - Yerrcor_sq (clipped at 0).
    """
    if logger is None:
        logger = PrintLogger()

    # Skip if neither window specification is given
    if _get_option(science_options, 'running_scatter_window_arcmin') is None and \
       _get_option(science_options, 'running_scatter_Npoints') is None:
        logger.info("Both 'running_scatter_window_arcmin' and 'running_scatter_Npoints' are None/Null; skipping running scatter plot.")
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # --- Read catalogue and prepare inputs
    tab = Table.read(str(rms1d_catalog))

    # Apply SNR cut
    snr_thr = float(_get_option(science_options, "snr_threshold", 7.0))
    if "SNR_PI" not in tab.colnames:
        raise KeyError("Required column 'SNR_PI' not found for SNR filtering.")
    tab = tab[tab["SNR_PI"] > snr_thr]
    logger.info(f"Selected {len(tab)} sources with SNR_PI > {snr_thr}")

    # RA/Dec columns
    ra_col  = _find_col(tab, RA_COL_OPTIONS)
    dec_col = _find_col(tab, DEC_COL_OPTIONS)
    if ra_col is None or dec_col is None:
        raise KeyError(
            "Could not find RA/Dec columns. Tried: "
            f"{RA_COL_OPTIONS} and {DEC_COL_OPTIONS}"
        )

    # RM columns (GRM-aware)
    rm_col, err_col, title_suffix, file_suffix = _resolve_rm_columns(tab, science_options)

    # Positions and radii [arcmin]
    coords = SkyCoord(tab[ra_col] * u.deg, tab[dec_col] * u.deg, frame="icrs")
    sep = coords.separation(center_coord)
    r_arcmin = sep.to(u.arcmin).value

    # RM values and (optional) uncertainties
    RM = _as_quantity(tab[rm_col]).to(u.rad / u.m**2).value
    if err_col is not None:
        try:
            RM_err = _as_quantity(tab[err_col]).to(u.rad / u.m**2).value
        except Exception:
            RM_err = np.asarray(tab[err_col])
    else:
        RM_err = None

    # --- Run running statistics
    xwidth = _get_option(science_options, "running_scatter_window_arcmin")
    nboot  = _get_option(science_options, "running_scatter_nbootstrap")
    Mfix   = _get_option(science_options, "running_scatter_Npoints")

    rs: RunningStatisticsResult = runningstatistics(
        seqX=r_arcmin,
        seqY=RM,
        xwidth=xwidth,
        seqYerr=RM_err,
        nbootstrap=nboot,
        redshifts=None,
        M=Mfix,
    )

    # Helper: summarize bootstrap/object arrays -> (median, low, high)
    def _summarize(arr):
        """
        Accepts either a float array (shape [nbin]) or an object array where
        each element is a 1D bootstrap array. Returns (y, yerr_low, yerr_up).
        """
        arr = np.asarray(arr, dtype=object) if isinstance(arr, (list, tuple)) or getattr(arr, "dtype", None) is None else arr
        if getattr(arr, "dtype", None) is object:
            y = np.empty(len(arr), float)
            ylo = np.empty(len(arr), float)
            yhi = np.empty(len(arr), float)
            for i, samp in enumerate(arr):
                samp = np.asarray(samp, float)
                if samp.size == 0 or not np.all(np.isfinite(samp)):
                    y[i] = np.nan
                    ylo[i] = np.nan
                    yhi[i] = np.nan
                else:
                    y[i]  = np.nanmedian(samp)
                    p16   = np.nanpercentile(samp, 16.0)
                    p84   = np.nanpercentile(samp, 84.0)
                    ylo[i] = y[i] - p16
                    yhi[i] = p84 - y[i]
            return y, ylo, yhi
        else:
            # No bootstrap -> no vertical error bars from the statistic
            return np.asarray(arr, float), None, None

    # Choose base scatter statistic
    method = str(_get_option(science_options, "running_scatter_method", "iqr")).lower()
    if method == "std":
        base_stat, elo, ehi = _summarize(rs.stds_y)
    elif method == "iqr":
        base_stat, elo, ehi = _summarize(rs.iqrs_y)
    elif method == "mad":
        base_stat, elo, ehi = _summarize(rs.MADs_y)
    else:
        raise ValueError(f"Unknown running_scatter_method={method!r} (expected 'std'|'iqr'|'mad').")

    # Convert chosen statistic to Gaussian sigma
    scale_by = _scale_scatter_method_to_sigma(method)
    sigma = base_stat / scale_by
    if elo is not None and ehi is not None:
        elo = elo / scale_by
        ehi = ehi / scale_by

    # Correct for measurement error if available: sigma_intr^2 = sigma_est^2 - Yerrcor_sq

    logger.warning("TODO: implement correction also for 'extrinsic' scatter")
    if getattr(rs, "Yerrcor_sq", None) is not None:
        yerrcor = np.asarray(rs.Yerrcor_sq, float)
        if yerrcor.shape == sigma.shape:
            sigma2 = np.clip(sigma**2 - yerrcor, a_min=0.0, a_max=None)
            sigma = np.sqrt(sigma2)
            # Propagate bootstrap error bars through the same subtraction (approximation):
            if elo is not None and ehi is not None:
                # Finite-difference style propagation on the +/− bars
                sigma_plus  = np.sqrt(np.clip((sigma + ehi)**2 - yerrcor, 0.0, None))
                sigma_minus = np.sqrt(np.clip((np.maximum(sigma - elo, 0.0))**2 - yerrcor, 0.0, None))
                ehi = np.abs(sigma_plus - sigma)
                elo = np.abs(sigma - sigma_minus)

    # X positions for plotting
    x = np.asarray(rs.medians_x, float)
    z = _get_option(science_options, "z_cluster", None)
    have_kpc_axis = (z is not None)
    if have_kpc_axis:
        DA = Planck18.angular_diameter_distance(float(z)).to(u.kpc)  # kpc
        kpc_per_arcmin = (1.0 * u.arcmin).to(u.rad) * DA
        kpc_per_arcmin = kpc_per_arcmin.value

        def arcmin_to_kpc(xv):
            return xv * kpc_per_arcmin

        def kpc_to_arcmin(xv):
            return xv / kpc_per_arcmin

    # --- Plot
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    if elo is not None and ehi is not None:
        ax.errorbar(
            x, sigma, yerr=[elo, ehi], fmt="o-", ms=3.5, lw=1.1, elinewidth=0.9,
            capsize=2.0, alpha=0.95,
        )
    else:
        ax.plot(x, sigma, "o-", ms=3.5, lw=1.1, alpha=0.95)

    ax.set_xlabel(r"Radius to centre ($\mathrm{arcmin}$)")
    # y-label reflects GRM-corrected case
    if file_suffix:
        ax.set_ylabel(r"$\sigma_{\mathrm{RM,corr}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")
    else:
        ax.set_ylabel(r"$\sigma_{\mathrm{RM}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")

    # Secondary kpc axis if redshift given
    if have_kpc_axis:
        secax = ax.secondary_xaxis("top", functions=(arcmin_to_kpc, kpc_to_arcmin))
        secax.set_xlabel(
            r"Radius to centre ($\mathrm{kpc}$)"
            + f"  [Planck18, $z={float(z):.3f}$]"
        )

    ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)
    fig.tight_layout()

    # Filenames
    field = _get_option(science_options, "targetfield", "field")
    base = f"running_scatter_vs_radius_{field}_method-{method}"
    if have_kpc_axis:
        base += f"_z{float(z):.3f}"
    base += file_suffix  # '' or '_grm-hutschenreuter'

    png_path = plot_dir / f"{base}.png"
    pdf_path = pdf_dir / f"{base}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved running scatter vs radius plot: {png_path.name} and {pdf_path.name}")

    return None



def generate_science_plots(
    science_options: ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    rms1d_fdf: Path,
    rms1d_spectra: Path,
    center_coord: SkyCoord,
    stokesI_MFS: Path,
    output_dir: Path,
    logger=None,
) -> None:
    """
    Create ALL the science plots after 1D RM synthesis.
    """
    if logger is None:
        logger = PrintLogger()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot RM vs radius
    plot_rm_vs_radius(
        science_options,
        rms1d_catalog,
        center_coord,
        plot_dir=output_dir,
        logger=logger,
    )

    # Plot RM bubble map
    plot_rm_bubble_map(
        science_options,
        rms1d_catalog,
        center_coord,
        plot_dir=output_dir,
        logger=logger,
    )

    # Plot RM bubble on stokes I image
    plot_rm_bubble_map_on_stokesI(
        science_options,
        rms1d_catalog,
        stokesI_MFS=stokesI_MFS,
        center_coord=center_coord,
        plot_dir=output_dir,
        logger=logger,
    )

    # Plot running scatter vs radius
    running_scatter_vs_radius(
        science_options,
        rms1d_catalog,
        center_coord,
        plot_dir=output_dir,
        logger=logger,
    )