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
from RMutils.util_misc import (
    powerlaw_poly5,  # assumes fit_function='log' in RMsynth .ini file
)

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import _wrap_angle_deg
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
    chan_for_i_cutout: int | None = None
    """Optionally, use a channel for plotting the stokes I cutout. Useful when MFS image corrupted by bad channels"""
    chan_for_qu_cutout: int | None = None
    """Optionally, use a channel for plotting the stokes P cutout (Q^2+U^2)^0.5. Useful when MFS image corrupted by bad channels. 7 is a good default for 12 channel imaging in L-band."""


# ---------------------- File helpers ----------------------

def _pixel_scale_deg_from_wcs(fits_path: Path) -> float:
    with fits.open(fits_path) as hdul:
        wcs = WCS(hdul[0].header).celestial
        # proj_plane_pixel_scales returns Quantity in angle/pixel
        scales = wcs.proj_plane_pixel_scales() # should be 2-element list
        assert scales[0] == scales[1], "Non-square pixels not supported."
        return float(scales[0].to(u.deg).value)


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
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
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
    ax_chi = fig.add_subplot(gs[2, 2])


    axes = {
        "I": ax_I,
        "RMSF": ax_rmsf,
        "FDF": ax_fdf,
        "Q": ax_Q,
        "U": ax_U,
        "TEXT": ax_text,
        "inset_I": ax_inset_I,
        "inset_P": ax_inset_P,
         "CHI": ax_chi,
    }
    return fig, axes


def _plot_stokes_I(
    ax: plt.Axes,
    freq_hz: np.ndarray,
    I_jy: np.ndarray,
    Ierr: np.ndarray,
    *,
    model_params: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Plot observed stokes I spectrum and (optional) overplot model.

    model_params: optional tuple (I_curvature, spectral_index, stokesI_Jy, reffreq_Hz)

    See RM-tools powerlaw_poly5 for model definition. Assumes fit_type='log' in RMsynth .ini file.

    """
    # Convert to GHz and mJy, log-log
    nu_ghz = np.asarray(freq_hz, dtype=float) / 1e9
    I_mJy = 1e3 * np.asarray(I_jy, dtype=float)
    e_mJy = 1e3 * Ierr

    m = np.isfinite(nu_ghz) & np.isfinite(I_mJy) & (I_mJy > 0)
    if not np.any(m):
        ax.text(0.5, 0.5, "No valid I data", transform=ax.transAxes, ha="center")
        return

    # ax.scatter(nu_ghz[m], I_mJy[m], s=12)
    ax.errorbar(nu_ghz[m], I_mJy[m], yerr=e_mJy[m], fmt="o", ms=3, lw=0.8, alpha=0.9)
    # ax.plot(nu_ghz[m], I_mJy[m], lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Stokes I [mJy]")
    ax.set_title("Stokes I vs frequency")
    ax.grid()

    # overlay best-fit spectrum if available
    if (model_params is not None):
        I_curv, alpha, stokesI_Jy, reffreq_Hz = model_params
        p = [I_curv, alpha, stokesI_Jy]
        model = powerlaw_poly5(p)  # callable
        y_model_Jy = model(freq_hz / float(reffreq_Hz))
        y_model_mJy = 1e3 * np.asarray(y_model_Jy, dtype=float)
        mm = np.isfinite(nu_ghz) & np.isfinite(y_model_mJy) & (y_model_mJy > 0)
        if np.any(mm):
            ax.plot(nu_ghz[mm], y_model_mJy[mm], lw=1.2, alpha=0.9, label="best-fit I model")
            ax.legend(fontsize=8, loc="best")


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
        ax.grid()


def _plot_rmsf(ax: plt.Axes, phi_rmsf: np.ndarray, rmsf: np.ndarray) -> None:
    x = phi_rmsf
    y = np.abs(rmsf)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        ax.text(0.5, 0.5, "No RMSF", transform=ax.transAxes, ha="center")
        return
    ax.plot(x[m], y[m], lw=1.0)
    ax.set_xlabel("phi [rad/m^2]")
    ax.set_ylabel("amplitude [Jy/beam/RMSF]")
    ax.set_title("RMSF")


def _plot_fdf(ax: plt.Axes, phi_fdf: np.ndarray, fdf: np.ndarray) -> None:
    x = phi_fdf
    y = np.abs(fdf)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        ax.text(0.5, 0.5, "No FDF", transform=ax.transAxes, ha="center")
        return
    ax.plot(x[m], y[m], lw=1.0)
    ax.set_xlabel("phi [rad/m^2]")
    ax.set_ylabel("amplitude [Jy/beam/RMSF]")
    fdf_peak_loc = x[m][np.argmax(y[m])]
    ax.set_title(f"FDF, peak at {fdf_peak_loc:.1f} rad/m^2")


def _plot_summary_text(ax: plt.Axes, cat_row: Table.Row) -> None:
    # Show required fields; any missing -> "N/A"
    def get(name: str) -> str:
        if name in ['Source_ID', 'S_Code', 'IFitStat']:
            # raw string
            return str(cat_row[name]) if name in cat_row.colnames or name in cat_row.keys() else "N/A"
        else:
            # float with 2 decimal places
            return str(f"{cat_row[name]:.2f}") if name in cat_row.colnames or name in cat_row.keys() else "N/A"

    lines = [
        f"Source_id: {get('Source_id')}",
        f"Stokes I: {get('stokesI')*1e3} mJy",
        f"RM: {get('rm')} +/- {get('rm_err')} rad/m^2",
        f"SNR_PI: {get('SNR_PI')}",
        f"S_Code: {get('S_Code')}",
        f"polint: {get('polint')*1e3} +/- {get('polint_err')*1e3} mJy",
        f"IFitStat: {get('IFitStat')}",
        f"fracpol: {get('fracpol')}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)


def _plot_pol_angle(ax: plt.Axes,
                    freq_hz: np.ndarray,
                    Q_jy: np.ndarray,
                    U_jy: np.ndarray,
                    derot_polangle_deg: float,
                    derot_polangle_err_deg: float,
                    rm_rad_m2: float,
                    rm_err_rad_m2: float) -> None:
    """
    Plot pol angle (0.5 * arctan2(U, Q) in degrees) vs lambda^2,
    and overlay best-fit line: chi_fit = derot_polangle + rm * lambda^2,
    with a 1-sigma shaded band from independent errors:
      sigma_chi(x)^2 = sigma_derot^2 + (x^2) * sigma_rm^2
    """
    nu = np.asarray(freq_hz, dtype=float)
    Q = np.asarray(Q_jy, dtype=float)
    U = np.asarray(U_jy, dtype=float)

    m = np.isfinite(nu) & np.isfinite(Q) & np.isfinite(U) & (nu > 0)
    if not np.any(m):
        ax.text(0.5, 0.5, "No valid Q/U data", transform=ax.transAxes, ha="center")
        return

    lam2 = (c.to(u.m / u.s).value / nu[m])**2  # m^2
    chi_deg = np.degrees(0.5 * np.arctan2(U[m], Q[m]))
    chi_deg = _wrap_angle_deg(chi_deg)

    # Data points
    ax.scatter(lam2, chi_deg, s=12, alpha=0.9, label="data")

    # Best-fit line and 1-sigma band
    # Convert RM and its error to deg/m^2
    rm_deg = rm_rad_m2 * (180.0 / np.pi)
    rm_err_deg = rm_err_rad_m2 * (180.0 / np.pi)

    x = np.linspace(lam2.min(), lam2.max(), 200)
    chi_fit = derot_polangle_deg + rm_deg * x

    # 1-sigma band assuming independence of intercept and slope
    sigma = np.sqrt(derot_polangle_err_deg**2 + (x**2) * (rm_err_deg**2))

    ax.plot(x, chi_fit, lw=1.2, label="best fit")
    ax.fill_between(x, chi_fit - sigma, chi_fit + sigma, alpha=0.25, label="1-sigma")

    ax.set_xlabel("lambda^2 [m^2]")
    ax.set_ylabel("pol angle [deg]")
    ax.set_title("Pol angle vs lambda^2")
    ax.legend(fontsize=8, loc="best")


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

    counter = 1
    for i in sel:
        logger.info(f"Plotting RM-synthesis validation for Source index {i}, number {counter} out of {len(sel)} with SNR_PI >= {snr_threshold}")
        counter += 1

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
        fdf = np.asarray(fdf_tbl["fdf"][j_fdf], dtype=complex)
        phi_rmsf = np.asarray(fdf_tbl["phi_rmsf"][j_fdf], dtype=float)
        rmsf = np.asarray(fdf_tbl["rmsf"][j_fdf], dtype=complex)

        # Build figure
        fig, axes = _build_figure()

        # Panels
        model_params = (
            float(cat_row["I_curvature"]),
            float(cat_row["spectral_index"]),
            float(cat_row["stokesI"]),       # Jy
            float(cat_row["reffreq_pol"]),   # Hz
        )
        # plot stokes I and RM-synthesis best-fit model
        _plot_stokes_I(
            axes["I"],
            freq_hz=freq_hz,
            I_jy=I_jy,
            Ierr=Ierr,
            model_params=model_params
        )
        _plot_Q_U(axes["Q"], axes["U"], freq_hz=freq_hz,
                  Q_jy=Q_jy, U_jy=U_jy, Qerr=Qerr, Uerr=Uerr)
        _plot_rmsf(axes["RMSF"], phi_rmsf=phi_rmsf, rmsf=rmsf)
        _plot_fdf(axes["FDF"], phi_fdf=phi_fdf, fdf=fdf)
        _plot_summary_text(axes["TEXT"], cat_row=cat_row)

        _plot_pol_angle(
            axes["CHI"],
            freq_hz=freq_hz,
            Q_jy=Q_jy,
            U_jy=U_jy,
            derot_polangle_deg=float(cat_row["derot_polangle"]),
            derot_polangle_err_deg=float(cat_row["derot_polangle_err"]),
            rm_rad_m2=float(cat_row["rm"]),
            rm_err_rad_m2=float(cat_row["rm_err"]),
        )

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
            validation_options=validation_rmsynth1d_options,
        )

        fig.suptitle(f"Source_id: {source_id}", fontsize=12)
        out_path = plot_dir / f"rm_validation_source{source_id}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths
