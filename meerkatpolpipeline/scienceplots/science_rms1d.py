from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.colors import TwoSlopeNorm

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import PrintLogger, _get_option

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
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    # RM (+ optional uncertainty, not drawn but may be useful later)
    RM = _as_quantity(tab[rm_col]).to(u.rad / u.m**2).value
    if err_col is not None:
        try:
            RM_err = _as_quantity(tab[err_col]).to(u.rad / u.m**2).value
        except Exception:
            RM_err = np.asarray(tab[err_col])
    else:
        RM_err = None

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


def generate_science_plots(
    science_options: ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    rms1d_fdf: Path,
    rms1d_spectra: Path,
    center_coord: SkyCoord,
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