"""
RM-synthesis validation plotting.

Requirements/assumptions:
- Load tables:
    - catalog: astropy Table with columns: 'Source_id', 'ra', 'dec', 'rm', 'rm_err',
      'SNR_PI', 'S_Code', 'polint', 'polint_err', 'IFitStat', 'fracpol', 'Maj' (deg)
    - fdf_tbl: astropy Table with columns: 'Source_id', 'phi_fdf' (array), 'fdf' (array),
      'phi_rmsf' (array), 'rmsf' (array). Plot absolute amplitude for FDF and RMSF.
    - spectra: polspectra.polarizationspectra from polspectra.from_FITS, table-like with
      row-wise arrays: 'freq' (Hz), 'stokesI', 'stokesI_error', 'stokesQ', 'stokesQ_error',
      'stokesU', 'stokesU_error', plus 'Source_id'.

- Insets:
    You must pass ImageSet objects for I, Q, U (from meerkatpolpipeline.wsclean.wsclean).
    Insets are produced by:
        from meerkatpolpipeline.validation.validate_field import add_inset_cutouts

- Units & axes:
    * Stokes I vs frequency: x in GHz (log), y in mJy (log).
      (Convert I from Jy -> mJy.)
    * Stokes Q and U vs lambda^2 (m^2), linear axes, include error bars if provided.
      (Q/U kept in Jy.)
    * RMSF vs phi and FDF vs phi: x in rad/m^2, y = abs(amplitude), label:
        "amplitude [Jy/beam/RMSF]"
    * Cutout size in pixels = (Maj * 2.0 deg) / pixel_scale_deg.


To use: call make_rm_validation_plots(...) from a driver (flow).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from polspectra import from_FITS as polspectra_from_FITS
from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.validation.validate_field import (
    add_inset_cutouts,
    first_mfs_file,
)
from meerkatpolpipeline.wsclean.wsclean import ImageSet


class ValidateRMsynth1dOptions(BaseOptions):
    """A basic class to handle options for validation plots for 1D RM synthesis. """
    
    enable: bool
    """enable this step?"""
    targetfield: str | None = None
    """name of targetfield. This option is propagated to every step."""
    snr_threshold: float = 7.0
    """SNR threshold in polarized intensity for making validation plots. Default 7.0"""


# ---------------------- File helpers ----------------------

def _pixel_scale_deg_from_wcs(fits_path: Path) -> float:
    with fits.open(fits_path) as hdul:
        wcs = WCS(hdul[0].header).celestial
        # proj_plane_pixel_scales returns Quantity in angle/pixel
        # Take mean of x/y to be robust to non-square pixels
        scales = wcs.proj_plane_pixel_scales().to(u.deg).value
        return float(np.mean(scales))


# ---------------------- Data access -----------------------

def _row_by_source_id(tab: Table, source_id) -> int:
    idx = np.where(tab["Source_id"] == source_id)[0]
    if idx.size != 1:
        raise KeyError(f"Expected exactly one row for Source_id={source_id}, found {idx.size}.")
    return int(idx[0])


def _row_by_source_id_spectra(spectra, source_id) -> int:
    # polspectra object acts like an astropy Table for column access
    sid = np.asarray(spectra["Source_id"])
    idx = np.where(sid == source_id)[0]
    if idx.size != 1:
        raise KeyError(f"Expected exactly one spectra row for Source_id={source_id}, found {idx.size}.")
    return int(idx[0])


# ---------------------- Plot builders ---------------------

def _build_figure() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    3x3 layout:
      [0,0] Stokes I vs nu(GHz)    [0,1] RMSF vs phi       [0,2] FDF vs phi
      [1,0] Q vs lambda^2          [1,1] U vs lambda^2     [1,2] Summary text
      [2,0] Inset I (image)        [2,1] Inset P (image)   [2,2] (unused)
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 1.0], width_ratios=[1.0, 1.0, 1.0])

    ax_I = fig.add_subplot(gs[0, 0])
    ax_rmsf = fig.add_subplot(gs[0, 1])
    ax_fdf = fig.add_subplot(gs[0, 2])

    ax_Q = fig.add_subplot(gs[1, 0])
    ax_U = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")

    ax_inset_I = fig.add_subplot(gs[2, 0])
    ax_inset_P = fig.add_subplot(gs[2, 1])

    axes = {
        "I": ax_I,
        "RMSF": ax_rmsf,
        "FDF": ax_fdf,
        "Q": ax_Q,
        "U": ax_U,
        "TEXT": ax_text,
        "inset_I": ax_inset_I,
        "inset_P": ax_inset_P,
    }
    return fig, axes


def _plot_stokes_I(ax: plt.Axes, freq_hz: np.ndarray, I_jy: np.ndarray) -> None:
    # Convert to GHz and mJy, log-log
    nu_ghz = np.asarray(freq_hz, dtype=float) / 1e9
    I_mJy = 1e3 * np.asarray(I_jy, dtype=float)

    m = np.isfinite(nu_ghz) & np.isfinite(I_mJy) & (I_mJy > 0)
    if not np.any(m):
        ax.text(0.5, 0.5, "No valid I data", transform=ax.transAxes, ha="center")
        return

    ax.scatter(nu_ghz[m], I_mJy[m], s=12)
    ax.plot(nu_ghz[m], I_mJy[m], lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Stokes I [mJy]")
    ax.set_title("Stokes I vs frequency")


def _plot_Q_U(axQ: plt.Axes, axU: plt.Axes,
              freq_hz: np.ndarray,
              Q_jy: np.ndarray, U_jy: np.ndarray,
              Qerr: np.ndarray | None, Uerr: np.ndarray | None) -> None:
    # lambda^2 in m^2
    nu = np.asarray(freq_hz, dtype=float)
    lam = (c.to(u.m / u.s).value / nu)  # meters
    lam2 = lam**2

    for ax, arr, err, label in (
        (axQ, Q_jy, Qerr, "Stokes Q"),
        (axU, U_jy, Uerr, "Stokes U"),
    ):
        y = np.asarray(arr, dtype=float)
        m = np.isfinite(lam2) & np.isfinite(y)
        if not np.any(m):
            ax.text(0.5, 0.5, f"No valid {label}", transform=ax.transAxes, ha="center")
            continue
        if err is not None:
            e = np.asarray(err, dtype=float)
            if e.shape == y.shape:
                ax.errorbar(lam2[m], y[m], yerr=e[m], fmt="o", ms=3, lw=0.8, alpha=0.9)
            else:
                ax.plot(lam2[m], y[m], "o-", ms=3, lw=0.8, alpha=0.9)
        else:
            ax.plot(lam2[m], y[m], "o-", ms=3, lw=0.8, alpha=0.9)
        ax.set_xlabel("lambda^2 [m^2]")
        ax.set_ylabel("Flux [Jy]")
        ax.set_title(f"{label} vs lambda^2")


def _plot_rmsf(ax: plt.Axes, phi_rmsf: np.ndarray, rmsf: np.ndarray) -> None:
    # Magnitude; label per requirement
    x = np.asarray(phi_rmsf, dtype=float)
    y = np.abs(np.asarray(rmsf, dtype=float))
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        ax.text(0.5, 0.5, "No RMSF", transform=ax.transAxes, ha="center")
        return
    ax.plot(x[m], y[m], lw=1.0)
    ax.set_xlabel("phi [rad/m^2]")
    ax.set_ylabel("amplitude [Jy/beam/RMSF]")
    ax.set_title("RMSF")


def _plot_fdf(ax: plt.Axes, phi_fdf: np.ndarray, fdf: np.ndarray) -> None:
    x = np.asarray(phi_fdf, dtype=float)
    y = np.abs(np.asarray(fdf, dtype=float))
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        ax.text(0.5, 0.5, "No FDF", transform=ax.transAxes, ha="center")
        return
    ax.plot(x[m], y[m], lw=1.0)
    ax.set_xlabel("phi [rad/m^2]")
    ax.set_ylabel("amplitude [Jy/beam/RMSF]")
    ax.set_title("FDF")


def _plot_summary_text(ax: plt.Axes, cat_row: Table.Row) -> None:
    # Show required fields; any missing -> "N/A"
    def get(name: str) -> str:
        return str(cat_row[name]) if name in cat_row.colnames or name in cat_row.keys() else "N/A"

    lines = [
        f"Source_id: {get('Source_id')}",
        f"rm: {get('rm')} +/- {get('rm_err')} rad/m^2",
        f"SNR_PI: {get('SNR_PI')}",
        f"S_Code: {get('S_Code')}",
        f"polint: {get('polint')} +/- {get('polint_err')} Jy",
        f"IFitStat: {get('IFitStat')}",
        f"fracpol: {get('fracpol')}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)


def _compute_cutout_size_pixels(maj_deg: float, imageset_I: ImageSet) -> int:
    mfs_I = first_mfs_file(list(imageset_I.image_pbcor))
    if mfs_I is None:
        # Fallback: assume 2 arcsec pixels if no MFS found (conservative)
        pix_deg = (2.0 / 3600.0)
    else:
        pix_deg = _pixel_scale_deg_from_wcs(mfs_I)
    size_pix = int(np.ceil((maj_deg * 2.0) / pix_deg))
    # Keep a sane min/max to avoid huge tiles
    return int(np.clip(size_pix, 24, 512))


# ---------------------- Public API ------------------------

def make_rm_validation_plots(
    validation_rmsynth1d_options: dict | ValidateRMsynth1dOptions,
    imageset_I: ImageSet,
    imageset_Q: ImageSet,
    imageset_U: ImageSet,
    rms1d_catalog: Path,
    rms1d_fdf: Path,
    rms1d_spectra: Path,
    plot_dir: Path,
) -> list[Path]:
    """
    Build one validation plot per source with SNR_PI >= snr_threshold.

    Returns list of written PNG paths.
    """

    logger = get_run_logger()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    catalog = Table.read(str(rms1d_catalog))
    fdf_tbl = Table.read(str(rms1d_fdf))
    spectra = polspectra_from_FITS(str(rms1d_spectra))

    out_paths: list[Path] = []

    snr_threshold = validation_rmsynth1d_options['snr_threshold']

    # Filter sources
    if "SNR_PI" not in catalog.colnames:
        raise KeyError("Catalog must contain 'SNR_PI' column.")
    sel = np.where(np.asarray(catalog["SNR_PI"], dtype=float) >= float(snr_threshold))[0]


    for i in sel:
        logger.info(f"Making RM-synthesis validation plot for Source {i} out of {len(sel)} with SNR_PI >= {snr_threshold}")

        cat_row = catalog[i]
        source_id = cat_row["Source_id"]

        # Cross-match rows
        j_fdf = _row_by_source_id(fdf_tbl, source_id)
        j_spc = _row_by_source_id_spectra(spectra, source_id)

        # Extract arrays
        freq_hz = np.asarray(spectra["freq"][j_spc], dtype=float)
        I_jy = np.asarray(spectra["stokesI"][j_spc], dtype=float)
        Q_jy = np.asarray(spectra["stokesQ"][j_spc], dtype=float)
        U_jy = np.asarray(spectra["stokesU"][j_spc], dtype=float)

        Ierr = spectra["stokesI_error"][j_spc] if "stokesI_error" in spectra.columns else None
        Qerr = spectra["stokesQ_error"][j_spc] if "stokesQ_error" in spectra.columns else None
        Uerr = spectra["stokesU_error"][j_spc] if "stokesU_error" in spectra.columns else None
        Ierr = np.asarray(Ierr, dtype=float) if Ierr is not None else None
        Qerr = np.asarray(Qerr, dtype=float) if Qerr is not None else None
        Uerr = np.asarray(Uerr, dtype=float) if Uerr is not None else None

        phi_fdf = np.asarray(fdf_tbl["phi_fdf"][j_fdf], dtype=float)
        fdf = np.asarray(fdf_tbl["fdf"][j_fdf], dtype=float)
        phi_rmsf = np.asarray(fdf_tbl["phi_rmsf"][j_fdf], dtype=float)
        rmsf = np.asarray(fdf_tbl["rmsf"][j_fdf], dtype=float)

        # Build figure
        fig, axes = _build_figure()

        # Panels
        _plot_stokes_I(axes["I"], freq_hz=freq_hz, I_jy=I_jy)
        _plot_Q_U(axes["Q"], axes["U"], freq_hz=freq_hz,
                  Q_jy=Q_jy, U_jy=U_jy, Qerr=Qerr, Uerr=Uerr)
        _plot_rmsf(axes["RMSF"], phi_rmsf=phi_rmsf, rmsf=rmsf)
        _plot_fdf(axes["FDF"], phi_fdf=phi_fdf, fdf=fdf)
        _plot_summary_text(axes["TEXT"], cat_row=cat_row)

        # Insets (I and P), using Maj*2 scaling (in degrees -> pixels via I-MFS WCS)
        center = SkyCoord(ra=float(cat_row["ra"]) * u.deg, dec=float(cat_row["dec"]) * u.deg, frame="icrs")
        maj_deg = float(cat_row["Maj"])  # already degrees
        cutout_size_pix = _compute_cutout_size_pixels(maj_deg, imageset_I)

        add_inset_cutouts(
            {"inset_I": axes["inset_I"], "inset_P": axes["inset_P"]},
            imageset_I=imageset_I,
            imageset_Q=imageset_Q,
            imageset_U=imageset_U,
            center=center,
            cutout_size_pix=cutout_size_pix,
        )

        fig.suptitle(f"Source_id: {source_id}", fontsize=12)
        out_path = plot_dir / f"rm_validation_source{source_id}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths
