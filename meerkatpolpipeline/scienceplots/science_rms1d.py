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
from contextlib import contextmanager
from astropy.wcs import WCS
from matplotlib.colors import TwoSlopeNorm
from astropy.nddata import Cutout2D
from matplotlib.patches import Circle
from astropy.visualization.wcsaxes import SphericalCircle

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import PrintLogger, _get_option
from meerkatpolpipeline.scienceplots.science_utils import runningstatistics, RunningStatisticsResult
from meerkatpolpipeline.validation.rms_vs_freq import compute_rms_from_imagelist
from meerkatpolpipeline.check_racs.target_vs_racs import get_beam_from_header


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
RM_ERR_COL_OPTIONS = ["e_RM", "RM_ERR", "rm_err", "dRM", "ERR_RM", "RM_err"]
RA_COL_OPTIONS = ["RA", "ra", "RA_deg", "RAJ2000", "optRA"]
DEC_COL_OPTIONS = ["DEC", "Dec", "dec", "DEC_deg", "DEJ2000", "optDec"]
KPC_AXIS_COLOR = "#1f77b4" # mpl default blue
# muted orange ("#e07b39") for more contrast,
# teal ("#008b8b") more subtle

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
    running_scatter_binwidth_visualisation: str | None = None
    """How to visualise the running-bin width: 'upperpanel' (x-whiskers), 'lowerpanel' (r_min/r_max panel), or None."""
    mfs_image_vmin: float | None = None
    """Lower display limit for MFS image. If None, use percentile (99.5%) with arcsinh scaling."""
    mfs_image_vmax: float | None = None
    """Upper display limit for MFS image. If None, use percentile (99.5%) with arcsinh scaling."""
    mfs_image_width_deg: float | None = None
    """If set, crop the MFS image to a square of this width (deg) centered on `center_coord`."""
    cluster_r500_deg: float | None = None
    """If set, draw a yellow dashed circle of this radius (deg) centered on `center_coord`, labeled $R_{\\mathrm{500}}$."""
    presentation: bool = False
    """If True, use presentation styling (transparent figure bg, optional custom style)."""
    figstyle: Path | None = None
    """Optional Matplotlib style file to load when presentation=True."""


@contextmanager
def _presentation_mode(science_options, logger=None):
    """
    Temporarily enable presentation styling:
      - optional plt.style.use(path)
      - savefig.facecolor transparent (outside axes)
    Reverts automatically on exit.
    """
    if logger is None:
        logger = PrintLogger()

    use_presentation = bool(_get_option(science_options, "presentation", False))
    style_path = _get_option(science_options, "figstyle", None)

    if not use_presentation:
        yield
        return

    rc_overrides = {
        "savefig.facecolor": (1.0, 0.0, 0.0, 0.0),  # transparent figure bg
        # keep ticks/labels visible regardless of style
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }

    with mpl.rc_context(rc=rc_overrides):
        if style_path is not None:
            try:
                with plt.style.context(str(style_path)):
                    yield
                    return
            except Exception as e:
                logger.warning(f"Could not load figstyle '{style_path}': {e}. Continuing without it.")
        yield


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


def _draw_r500_circle_plain(ax, center_coord: SkyCoord, r500_deg: float):
    """Draw R500 on a plain RA/Dec axes (no WCS projection)."""
    circ = Circle(
        (center_coord.ra.deg, center_coord.dec.deg),
        radius=float(r500_deg),
        fill=False, ls="--", lw=1.2, ec="yellow", alpha=0.9,
        zorder=3,
    )
    ax.add_patch(circ)
    # label near the top of the circle
    ax.text(
        center_coord.ra.deg,
        center_coord.dec.deg + float(r500_deg),
        r"$R_{\mathrm{500}}$",
        color="yellow", ha="center", va="bottom", fontsize=10, zorder=4,
    )


def _draw_r500_circle_wcs(ax, center_coord: SkyCoord, r500_deg: float):
    """Draw R500 on a WCSAxes projection using a spherical circle."""
    circ = SphericalCircle(
        center=center_coord,
        radius=float(r500_deg) * u.deg,
        edgecolor="yellow", facecolor="none",
        linestyle="--", linewidth=1.2, alpha=0.9,
        transform=ax.get_transform("world"),
        zorder=3,
    )
    ax.add_patch(circ)
    # label at 'top' point (roughly dec + r)
    ax.text(
        center_coord.ra.deg,
        center_coord.dec.deg + float(r500_deg),
        r"$R_{\mathrm{500}}$",
        color="yellow", ha="center", va="bottom", fontsize=10,
        transform=ax.get_transform("world"),
        zorder=4,
    )



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

    with _presentation_mode(science_options, logger=logger):
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
            if science_options['presentation']:
                secax.tick_params(axis="x", color="C1", labelcolor="C1")


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

        if science_options['presentation']:
            # apparently cant style labels and ticks in different colors, so have to hardcode
            ax.tick_params(axis="x", color="black", labelcolor="white")  # lines black, labels white
            ax.tick_params(axis="y", color="black", labelcolor="white")  # lines black, labels white


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
    with _presentation_mode(science_options, logger=logger):
        fig, ax = plt.subplots(figsize=(5.0, 4.2))

        sc = ax.scatter(
            ra, dec,
            c=RM, s=sizes,
            cmap=cmap, norm=norm,
            linewidths=0.4, edgecolors="k", alpha=0.9,
        )

        # Mark cluster centre
        ax.plot(center_coord.ra.deg, center_coord.dec.deg, marker="*", ms=10, mec="k", mfc="none", lw=1.0)

        # Optional R500 overlay
        r500_deg = _get_option(science_options, "cluster_r500_deg", None)
        if r500_deg is not None:
            try:
                _draw_r500_circle_plain(ax, center_coord, float(r500_deg))
            except Exception as e:
                logger.warning(f"Could not draw R500 circle (plain): {e}")


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

        if science_options['presentation']:
            # apparently cant style labels and ticks in different colors, so have to hardcode
            ax.tick_params(axis="x", color="black", labelcolor="white")  # lines black, labels white
            ax.tick_params(axis="y", color="black", labelcolor="white")  # lines black, labels white


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
    
    # Squeeze to 2D (take first plane if necessary)
    data = np.squeeze(data)
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError("Image is not 2D after squeezing.")

    # Celestial WCS
    wcs = WCS(hdr).celestial

    # Optional cutout around the centre
    width_deg = _get_option(science_options, "mfs_image_width_deg", None)
    if width_deg is not None:
        try:
            size = (width_deg * u.deg, width_deg * u.deg)
        except Exception:
            # be robust if user passes a Quantity already
            size = (float(width_deg) * u.deg, float(width_deg) * u.deg)
        cut = Cutout2D(
            data, position=center_coord, size=size, wcs=wcs, mode="trim"
        )
        data = cut.data
        wcs = cut.wcs

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
    with _presentation_mode(science_options, logger=logger):
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

        # Optional R500 overlay (WCS)
        r500_deg = _get_option(science_options, "cluster_r500_deg", None)
        if r500_deg is not None:
            try:
                _draw_r500_circle_wcs(ax, center_coord, float(r500_deg))
            except Exception as e:
                logger.warning(f"Could not draw R500 circle (WCS): {e}")


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
            ax.set_title(f"{field} — RM bubble map on Stokes I (z={float(z):.3f})\n{title_suffix}")
        else:
            ax.set_title(f"{field} — RM bubble map on Stokes I\n{title_suffix}")

        ax.set_xlabel(r"$\mathrm{RA}$")
        ax.set_ylabel(r"$\mathrm{Dec}$")

        if science_options['presentation']:
            # apparently cant style labels and ticks in different colors, so have to hardcode
            ax.tick_params(axis="x", color="black", labelcolor="white")  # lines black, labels white
            ax.tick_params(axis="y", color="black", labelcolor="white")  # lines black, labels white

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

    logger.info(f"Made {len(rs.medians_x)} running bins from {len(RM)} sources")

    # --- Choose base statistic array (per-bin), possibly bootstrap object arrays
    method = str(_get_option(science_options, "running_scatter_method", "iqr")).lower()
    if method == "std":
        stat_arr = rs.stds_y
    elif method == "iqr":
        stat_arr = rs.iqrs_y
    elif method == "mad":
        stat_arr = rs.MADs_y
    else:
        raise ValueError(f"Unknown running_scatter_method={method!r} (expected 'std'|'iqr'|'mad').")

    # Convert chosen statistic to Gaussian sigma using the appropriate scale factor
    scale_by = _scale_scatter_method_to_sigma(method)

    # Prepare measurement-error variance per bin (None if unavailable or mismatched)
    yerrcor_sq = getattr(rs, "Yerrcor_sq", None)
    if yerrcor_sq is not None:
        yerrcor_sq = np.asarray(yerrcor_sq, float)
    if (yerrcor_sq is not None) and (yerrcor_sq.shape != np.asarray(rs.medians_x).shape):
        logger.warning("Yerrcor_sq shape does not match number of bins; skipping measurement-error correction.")
        yerrcor_sq = None

    logger.warning("TODO: implement correction also for 'extrinsic' scatter.")

    # Apply measurement-error correction **before** summarising bootstraps, then summarise.
    # For bootstrap object arrays: correct each bootstrap sample: sigma = sqrt(max((stat/scale_by)^2 - yerrcor_sq, 0))
    # For plain arrays: same correction, but no vertical error bars available.
    def _summarize_and_correct_sigma(stat_arr, scale_by, yerrcor_sq):
        arr = np.asarray(stat_arr, dtype=object) if (
            isinstance(stat_arr, (list, tuple)) or getattr(stat_arr, "dtype", None) is None
        ) else stat_arr

        # Bootstrap/object case
        if isinstance(getattr(arr, "dtype", None), object):
            
            # convert to 1sigma, and float array
            arr = np.asarray(arr, dtype=float) / scale_by

            # correct each bootstrap-sampled bin for measurement error
            sig_samp = np.sqrt( np.clip (arr**2 - yerrcor_sq[:,np.newaxis], a_min=0.0, a_max=None)) # shape (nbin, nboot)

            # TODO: extrinsic scatter correction here in future
            # .... (calculate in script somewhere or allow user override?)

            # summarize per-bin bootstrap distributions
            med = np.nanmedian(sig_samp, axis=1)
            p16 = np.nanpercentile(sig_samp, 16.0, axis=1)
            p84 = np.nanpercentile(sig_samp, 84.0, axis=1)
            
            # return in standard format
            y   = med
            ylo = med - p16
            yhi = p84 - med

            return y, ylo, yhi
        
        else:
            # Non-bootstrap (plain float array) case. No vertical errorbars
            sigma = np.asarray(arr, float) / scale_by
            if yerrcor_sq is not None:
                sigma = np.sqrt(np.clip(sigma**2 - yerrcor_sq, 0.0, None))
            # TODO: extrinsic scatter correction here in future
            return sigma, None, None

    sigma, elo, ehi = _summarize_and_correct_sigma(stat_arr, scale_by, yerrcor_sq)

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

    # --- Compute asymmetric bin-width errors for x-whiskers
    left_bounds   = np.asarray(rs.left_bounds,    float)
    window_widths = np.asarray(rs.window_widths,  float)
    x             = np.asarray(rs.medians_x,      float)

    right_bounds  = left_bounds + window_widths
    xerr_left  = np.clip(x - left_bounds, 0.0, None)
    xerr_right = np.clip(right_bounds - x, 0.0, None)
    xerr = np.vstack([xerr_left, xerr_right])  # shape (2, N)

    # Do we need a second panel for Npoints?
    xwidth = _get_option(science_options, "running_scatter_window_arcmin")

    # --- Plotting
    # --- Compute bin edges and x-error (arcmin)
    left_bounds   = np.asarray(rs.left_bounds,    float)
    window_widths = np.asarray(rs.window_widths,  float)
    x             = np.asarray(rs.medians_x,      float)

    right_bounds  = left_bounds + window_widths
    xerr_left  = np.clip(x - left_bounds, 0.0, None)
    xerr_right = np.clip(right_bounds - x, 0.0, None)
    xerr = np.vstack([xerr_left, xerr_right])  # (2, N)

    # Decide visualization mode
    binvis = _get_option(science_options, "running_scatter_binwidth_visualisation", None)
    binvis = None if binvis is None else str(binvis).lower()

    # When to show a lower panel?
    # - If user asked 'lowerpanel' and we have fixed-count windows (Mfix set, xwidth None) -> show rmin/rmax there
    # - If fixed-width windows (xwidth set) -> show N per bin (same as before), regardless of binvis
    want_lowerpanel = False
    show_rminmax_panel = False
    if xwidth is not None:
        want_lowerpanel = True  # varying N: show N panel
    elif Mfix is not None and binvis == "lowerpanel":
        want_lowerpanel = True
        show_rminmax_panel = True

    # x-whiskers only for 'upperpanel'; otherwise no xerr
    use_xerr = (binvis == "upperpanel")
    xerr_to_use = xerr if use_xerr else None

    with _presentation_mode(science_options, logger=logger):
        # Figure layout
        if want_lowerpanel:
            fig, (ax, ax2) = plt.subplots(
                2, 1, sharex=True, figsize=(5.6, 5.6), height_ratios=[2.6, 1.0]
            )
        else:
            fig, ax = plt.subplots(figsize=(5.2, 6.9))
            ax2 = None

        # --- Top panel: sigma vs radius (with optional y-errors and optional x-whiskers)
        if elo is not None and ehi is not None:
            ax.errorbar(
                x, sigma,
                xerr=xerr_to_use, yerr=[elo, ehi],
                fmt="o", ms=3.5, lw=1.1, elinewidth=0.9,
                capsize=2.0,
                alpha=0.95, color='k'
            )
        else:
            ax.errorbar(
                x, sigma,
                xerr=xerr_to_use,
                fmt="o", ms=3.5, lw=1.1, elinewidth=0.9,
                capsize=2.0 if use_xerr else 0.0,
                alpha=0.95, color='k'
            )

        ax.set_xlabel(r"Radius to centre ($\mathrm{arcmin}$)")
        if file_suffix:
            ax.set_ylabel(r"$\sigma_{\mathrm{RM,corr}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")
        else:
            ax.set_ylabel(r"$\sigma_{\mathrm{RM}}\;(\mathrm{rad}\,\mathrm{m}^{-2})$")

        # Secondary kpc axis on top (colored as you set elsewhere)
        if have_kpc_axis:
            secax = ax.secondary_xaxis("top", functions=(arcmin_to_kpc, kpc_to_arcmin))
            kpc_color = "C1"
            # kpc_color = "#1f77b4"
            secax.set_xlabel(
                r"Radius to centre ($\mathrm{kpc}$)"
                + f"  [Planck18, $z={float(z):.3f}$]",
                color=kpc_color,
            )
            secax.tick_params(axis="x", colors=kpc_color)
            secax.spines["top"].set_color(kpc_color)
            secax.xaxis.label.set_color(kpc_color)

        ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)

        if science_options['presentation']:
            # apparently cant style labels and ticks in different colors, so have to hardcode
            ax.tick_params(axis="x", color="black", labelcolor="white")  # lines black, labels white
            ax.tick_params(axis="y", color="black", labelcolor="white")  # lines black, labels white

        # --- Lower panel content
        if want_lowerpanel and ax2 is not None:
            if science_options['presentation']:
                # apparently cant style labels and ticks in different colors, so have to hardcode
                ax2.tick_params(axis="x", color="black", labelcolor="white")

            if show_rminmax_panel:
                # Show r_min and r_max of each running bin
                rmin_arcmin = left_bounds
                rmax_arcmin = right_bounds
                if have_kpc_axis:
                    rmin = arcmin_to_kpc(rmin_arcmin)
                    rmax = arcmin_to_kpc(rmax_arcmin)
                    ylab = r"$r_{\min},\,r_{\max}\;(\mathrm{kpc})$"
                else:
                    rmin = rmin_arcmin
                    rmax = rmax_arcmin
                    ylab = r"$r_{\min},\,r_{\max}\;(\mathrm{arcmin})$"

                ax2.plot(x, rmin, "-", ms=2.8, lw=1.0, label=r"$r_{\min}$",color='C0')
                ax2.plot(x, rmax, "-", ms=2.8, lw=1.0, label=r"$r_{\max}$",color='C1')
                ax2.plot(x, arcmin_to_kpc(x), "o-", ms=2.8, lw=1.0, label=r"$r_{\mathrm{median}}$",color='k')
                ax2.set_ylabel(ylab)
                ax2.set_xlabel(r"Radius to centre ($\mathrm{arcmin}$)")
                ax2.grid(alpha=0.2, linestyle=":", linewidth=0.8)
                ax2.legend(frameon=False, loc="best")
            else:
                # Fixed-width: show N per bin (as before)
                Npoints = np.asarray(rs.Npoints, int)
                ax2.plot(x, Npoints, "s-", ms=3.0, lw=1.0)
                ax2.set_ylabel(r"$N_{\mathrm{bin}}$")
                ax2.set_xlabel(r"Radius to centre ($\mathrm{arcmin}$)")
                ax2.grid(alpha=0.2, linestyle=":", linewidth=0.8)

        fig.tight_layout()

        # Filenames
        field = _get_option(science_options, "targetfield", "field")
        method = str(_get_option(science_options, "running_scatter_method", "iqr")).lower()
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

    return


def plot_mfs_image_publication(
    science_options: dict | ScienceRMSynth1DOptions,
    stokesI_MFS: Path,
    center_coord: SkyCoord,
    plot_dir: Path,
    logger=None,
) -> None:
    """
    Render a publication-grade Stokes-I MFS image using WCS, with arcsinh scaling and
    the 'inferno' colormap. If both mfs_image_vmin and mfs_image_vmax are None, use
    robust percentile limits (1–99.5%) with an arcsinh stretch. If provided, use the
    user-specified vmin/vmax (still arcsinh).
    """
    if logger is None:
        logger = PrintLogger()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    stokesI_MFS = Path(stokesI_MFS)
    if not stokesI_MFS.exists():
        raise FileNotFoundError(f"Stokes I MFS file not found: {stokesI_MFS}")

    # Read FITS & find first image HDU
    with fits.open(stokesI_MFS) as hdul:
        ihdu = None
        for h in hdul:
            if (h.data is not None) and (h.data.size > 0):
                ihdu = h
                break
        if ihdu is None:
            raise ValueError(f"No image data found in {stokesI_MFS}")
        data = ihdu.data
        hdr = ihdu.header

    # Squeeze to 2D (take first plane if necessary)
    data = np.squeeze(data)
    while data.ndim > 2:
        data = data[0]
    if data.ndim != 2:
        raise ValueError("Image is not 2D after squeezing.")

    # Celestial WCS
    wcs = WCS(hdr).celestial

    # Optional cutout around the centre
    width_deg = _get_option(science_options, "mfs_image_width_deg", None)
    if width_deg is not None:
        try:
            size = (width_deg * u.deg, width_deg * u.deg)
        except Exception:
            # be robust if user passes a Quantity already
            size = (float(width_deg) * u.deg, float(width_deg) * u.deg)
        cut = Cutout2D(
            data, position=center_coord, size=size, wcs=wcs, mode="trim"
        )
        data = cut.data
        wcs = cut.wcs


    # Determine display normalization
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")

    user_vmin = _get_option(science_options, "mfs_image_vmin", None)
    user_vmax = _get_option(science_options, "mfs_image_vmax", None)

    if (user_vmin is None) and (user_vmax is None):
        # Robust percentile limits + arcsinh stretch
        p_lo, p_hi = AsymmetricPercentileInterval(1, 99.5).get_limits(data[finite])
        norm = ImageNormalize(vmin=p_lo, vmax=p_hi, stretch=AsinhStretch(a=0.02))
        logger.info(f"Using robust percentiles for MFS image display: vmin={p_lo:.3e}, vmax={p_hi:.3e}")
    else:
        # Use provided bounds (if one side is None, infer that side from percentiles)
        if (user_vmin is None) or (user_vmax is None):
            p_lo, p_hi = AsymmetricPercentileInterval(1, 99.5).get_limits(data[finite])
            if user_vmin is None:
                user_vmin = float(p_lo)
            if user_vmax is None:
                user_vmax = float(p_hi)
        norm = ImageNormalize(vmin=float(user_vmin), vmax=float(user_vmax), stretch=AsinhStretch(a=0.02))

    # Prepare labels
    field = _get_option(science_options, "targetfield", "field")
    bunit = hdr.get("BUNIT", "").strip()
    cbar_label = f"Stokes I ({bunit})" if bunit else "Stokes I"

    # Plot
    with _presentation_mode(science_options, logger=logger):
        fig = plt.figure(figsize=(5.4, 4.6))
        ax = plt.subplot(111, projection=wcs)
        im = ax.imshow(data, origin="lower", cmap="inferno", norm=norm)  # noqa: F841

        # Optional R500 overlay (WCS)
        r500_deg = _get_option(science_options, "cluster_r500_deg", None)
        if r500_deg is not None:
            try:
                _draw_r500_circle_wcs(ax, center_coord, float(r500_deg))
            except Exception as e:
                logger.warning(f"Could not draw R500 circle (WCS): {e}")


        # Axis cosmetics (publication-leaning)
        ax.grid(alpha=0.15, linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$\mathrm{RA}$")
        ax.set_ylabel(r"$\mathrm{Dec}$")

        # Colorbar
        cbar = fig.colorbar(ax.images[0], ax=ax, pad=0.01, fraction=0.046)
        cbar.set_label(cbar_label)

        # Title
        ax.set_title(f"{field} — Stokes I (MFS)")

        if science_options['presentation']:
            # apparently cant style labels and ticks in different colors, so have to hardcode
            ax.tick_params(axis="x", color="black", labelcolor="white")  # lines black, labels white
            ax.tick_params(axis="y", color="black", labelcolor="white")  # lines black, labels white

        fig.tight_layout()

        # Save
        base = f"mfs_image_{field}"
        png_path = plot_dir / f"{base}.png"
        pdf_path = pdf_dir / f"{base}.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved MFS image: {png_path.name} and {pdf_path.name}")

    return {"png": png_path, "pdf": pdf_path}


def plot_summary_text_panel(
    science_options: dict | ScienceRMSynth1DOptions,
    rms1d_catalog: Path,
    center_coord: SkyCoord,
    plot_dir: Path,
    logger=None,
    stokesI_MFS: Path | None = None,
):
    """
    Render a publication-grade text summary:
      - Target name
      - R500 (deg; also arcmin if available)
      - Redshift z
      - # of RMs within R500 (after SNR cut and GRM selection)
      - RM surface density [per deg^2], assuming a 0.5 deg radius catalogue footprint
      - std(RM) within R500
      - iqr(RM) within R500
    """
    if logger is None:
        logger = PrintLogger()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Read and filter catalogue
    tab = Table.read(str(rms1d_catalog))

    snr_thr = float(_get_option(science_options, "snr_threshold", 7.0))
    if "SNR_PI" not in tab.colnames:
        raise KeyError("Required column 'SNR_PI' not found for SNR filtering.")
    tab = tab[tab["SNR_PI"] > snr_thr]

    # Columns and RM choice (GRM aware)
    ra_col  = _find_col(tab, RA_COL_OPTIONS)
    dec_col = _find_col(tab, DEC_COL_OPTIONS)
    if ra_col is None or dec_col is None:
        raise KeyError(f"Could not find RA/Dec columns. Tried: {RA_COL_OPTIONS} / {DEC_COL_OPTIONS}")
    rm_col, err_col, title_suffix, file_suffix = _resolve_rm_columns(tab, science_options)

    # Coordinates and separations
    coords = SkyCoord(tab[ra_col] * u.deg, tab[dec_col] * u.deg, frame="icrs")
    sep_deg = coords.separation(center_coord).to(u.deg).value

    # RM values (in rad m^-2)
    RM = _as_quantity(tab[rm_col]).to(u.rad / u.m**2).value

    # Inputs
    field = _get_option(science_options, "targetfield", "field")
    z     = _get_option(science_options, "z_cluster", None)
    r500d = _get_option(science_options, "cluster_r500_deg", None)

    # Within R500 (if available)
    if r500d is not None:
        mask_r500 = np.isfinite(sep_deg) & (sep_deg <= float(r500d))
        RM_in = RM[mask_r500]
        n_in = int(np.sum(mask_r500))
        if np.sum(np.isfinite(RM_in)) >= 2:
            rm_std = float(np.nanstd(RM_in, ddof=1))
        else:
            rm_std = np.nan
        if np.sum(np.isfinite(RM_in)) >= 1:
            q25, q75 = np.nanpercentile(RM_in, [25.0, 75.0])
            rm_iqr = float(q75 - q25)
        else:
            rm_iqr = np.nan
    else:
        n_in = None
        rm_std = np.nan
        rm_iqr = np.nan

    # RM surface density [deg^-2] assuming 0.5 deg radius coverage
    # (use all SNR-selected rows)
    n_total = int(len(tab))
    area_deg2 = np.pi * (0.5 ** 2)
    rm_per_deg2 = n_total / area_deg2 if area_deg2 > 0 else np.nan

    # Compose lines (mathtext where helpful)
    lines = []
    lines.append(f"Target: {field}")
    if r500d is not None:
        r500_arcmin = 60.0 * float(r500d)
        lines.append(f"$R_{{\\mathrm{{500}}}}$: {float(r500d):.3f} deg  ({r500_arcmin:.1f} arcmin)")
    else:
        lines.append(r"$R_{\mathrm{500}}$: n/a")
    if stokesI_MFS is not None:
        stokesI_rms = compute_rms_from_imagelist([stokesI_MFS])[0]
        lines.append(f"Stokes I image RMS approx: {stokesI_rms*1e6:.2e} muJy/beam")
        with fits.open(stokesI_MFS) as hdul:
            hdr = hdul[0].header
        bmaj, bmin, bpa = get_beam_from_header(hdr)
        lines.append(f"Beam: {bmaj*3600:.2f}\" x {bmin*3600:.2f}\" @ {bpa:.1f} deg")

    if z is not None:
        lines.append(f"Redshift: $z={float(z):.3f}$")
    else:
        lines.append("Redshift: n/a")
    if n_in is not None:
        lines.append(f"\# RMs within $1\\times R_{{\\mathrm{{500}}}}$: {n_in:d}")
    else:
        lines.append(r"\# RMs within $1\times R_{\mathrm{500}}$: n/a")

    lines.append(f"RM density (per deg$^2$) [footprint $r=0.5$ deg]: {rm_per_deg2:.2f}")
    if np.isfinite(rm_std):
        lines.append(r"$\mathrm{std}(\mathrm{RM})$ within $R_{\mathrm{500}}$: " + f"{rm_std:.2f} rad m$^{{-2}}$")
    else:
        lines.append(r"$\mathrm{std}(\mathrm{RM})$ within $R_{\mathrm{500}}$: n/a")
    if np.isfinite(rm_iqr):
        lines.append(r"$\mathrm{IQR}(\mathrm{RM})$ within $R_{\mathrm{500}}$: " + f"{rm_iqr:.2f} rad m$^{{-2}}$")
    else:
        lines.append(r"$\mathrm{IQR}(\mathrm{RM})$ within $R_{\mathrm{500}}$: n/a")

    # Render: white axis, black text (independent of presentation theme)
    with _presentation_mode(science_options, logger=logger):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.set_facecolor("white")
        for side in ("top", "right", "left", "bottom"):
            if side in ax.spines:
                ax.spines[side].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # Left-aligned text block
        y = 0.90
        for ln in lines:
            ax.text(0.03, y, ln, ha="left", va="top", fontsize=12, color="black")
            y -= 0.10

        # Title (include GRM note if any)
        title_suffix = "" if file_suffix == "" else " — GRM corrected"
        ax.set_title(f"{field} — Summary{title_suffix}", color="black", fontsize=12)

        fig.tight_layout()

        base = f"summary_text_{field}{file_suffix}"
        png_path = plot_dir / f"{base}.png"
        pdf_path = pdf_dir / f"{base}.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved summary text panel: {png_path.name} and {pdf_path.name}")
    return {"png": png_path, "pdf": pdf_path}


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

    # generate MFS image 
    plot_mfs_image_publication(
        science_options,
        stokesI_MFS=stokesI_MFS,
        center_coord=center_coord,
        plot_dir=output_dir,
        logger=logger,
    )

    # Summary text panel
    plot_summary_text_panel(
        science_options,
        rms1d_catalog,
        center_coord,
        stokesI_MFS=stokesI_MFS,
        plot_dir=output_dir,
        logger=logger,
    )