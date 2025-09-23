from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from regions import Regions

from meerkatpolpipeline.utils.processfield import calculate_flux_and_peak_flux

# ------------------------------ WCS / FITS I/O ------------------------------ #

def load_primary_image_2d(fpath: Path) -> tuple[np.ndarray, fits.Header, WCS]:
    """
    Load the first available 2D slice of a FITS image and its celestial WCS.
    Crashes if WCS is invalid or missing (per user request).

    Returns
    -------
    data2d : np.ndarray
    header : fits.Header
    wcs_cel : WCS  (WCS(header).celestial)
    """
    with fits.open(fpath) as hdul:
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None:
            raise ValueError(f"No image data found in FITS: {fpath}")

        data = np.asarray(hdu.data)
        data = np.squeeze(data)
        if data.ndim > 2:
            # Take the first plane along non-spatial axes
            data = data.reshape((-1, *data.shape[-2:]))[0]
        if data.ndim != 2:
            raise ValueError(f"Could not obtain a 2D image from FITS: {fpath}")

        header = hdu.header.copy()

    # WCS must be valid; use celestial to comply with redundant axes
    try:
        wcs_cel = WCS(header).celestial
    except Exception as e:
        raise RuntimeError(f"Invalid or missing celestial WCS in {fpath}: {e}")

    return data, header, wcs_cel


def write_fits_like(out_path: Path, data: np.ndarray, header: fits.Header) -> None:
    """
    Write a FITS file with given data and header.
    Overwrites if exists.
    """
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(out_path, overwrite=True)


# ------------------------------ Beam handling ------------------------------- #

def get_beam_from_header(header: fits.Header) -> tuple[float, float, float]:
    """
    Extract (BMAJ, BMIN, BPA) from header in degrees and degrees.
    Returns
    -------
    bmaj_deg, bmin_deg, bpa_deg
    """
    for key in ("BMAJ", "BMIN", "BPA"):
        if key not in header:
            raise ValueError(f"Header missing {key}; cannot determine beam")
    bmaj = float(header["BMAJ"])  # degrees
    bmin = float(header["BMIN"])  # degrees
    bpa = float(header["BPA"])    # degrees, PA of major axis, North->East
    if not np.isfinite(bmaj) or not np.isfinite(bmin) or bmaj <= 0.0 or bmin <= 0.0:
        raise ValueError("Invalid beam in header (non-finite or non-positive BMAJ/BMIN)")
    return bmaj, bmin, bpa


def fwhm_to_sigma(fwhm_deg: float) -> float:
    """ Convert FWHM in degrees to Gaussian sigma in degrees. """
    return float(fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def rotation_matrix(theta_rad: float) -> np.ndarray:
    """ 2x2 rotation matrix for angle theta (radians), CCW. """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def beam_cov_sky_deg2(bmaj_deg: float, bmin_deg: float, bpa_deg: float) -> np.ndarray:
    """
    Beam covariance matrix in sky coordinates (deg^2) in the (RA, Dec) basis.

    BPA is measured from North to East (radio convention).
    The angle from the RA axis (x) is phi = 90 deg - BPA.
    """
    sigma_maj = fwhm_to_sigma(bmaj_deg)
    sigma_min = fwhm_to_sigma(bmin_deg)
    phi = np.deg2rad(90.0 - float(bpa_deg))  # from x-axis
    R = rotation_matrix(phi)
    D = np.diag([sigma_maj**2, sigma_min**2])
    return R @ D @ R.T


def cd_matrix_deg_per_pix(wcs: WCS) -> np.ndarray:
    """
    Get the 2x2 matrix mapping pixel -> degrees for the celestial WCS.
    Uses CD if present, else PC*CDELT.
    """
    hdr = wcs.to_header()
    if all(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        CD = np.array([[hdr["CD1_1"], hdr["CD1_2"]],
                       [hdr["CD2_1"], hdr["CD2_2"]]], dtype=float)
        return CD
    # Fall back to PC and CDELT
    if "CDELT1" not in hdr or "CDELT2" not in hdr:
        raise ValueError("WCS header missing CD or CDELT keywords")
    cdelt = np.diag([hdr["CDELT1"], hdr["CDELT2"]])
    if all(k in hdr for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
        PC = np.array([[hdr["PC1_1"], hdr["PC1_2"]],
                       [hdr["PC2_1"], hdr["PC2_2"]]], dtype=float)
    else:
        PC = np.eye(2, dtype=float)
    return PC @ cdelt


def beam_cov_in_pixels(header: fits.Header, wcs: WCS) -> np.ndarray:
    """
    Compute beam covariance matrix in pixel units (pixels^2).
    Σ_pixel = S^{-1} Σ_sky S^{-T}, where S maps pixel -> deg.
    """
    bmaj, bmin, bpa = get_beam_from_header(header)
    Sigma_sky = beam_cov_sky_deg2(bmaj, bmin, bpa)  # deg^2
    S = cd_matrix_deg_per_pix(wcs)                  # deg / pix
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        raise ValueError("CD/PC*CDELT matrix is singular; cannot invert")
    Sigma_pix = S_inv @ Sigma_sky @ S_inv.T
    return Sigma_pix


def kernel_from_covariance_pix(Sigma_kernel_pix: np.ndarray) -> Gaussian2DKernel:
    """
    Build a Gaussian2DKernel from a covariance matrix in pixel units.
    """
    # Eigen decomposition; ensure symmetric
    Sigma = 0.5 * (Sigma_kernel_pix + Sigma_kernel_pix.T)
    vals, vecs = np.linalg.eigh(Sigma)
    if np.any(vals <= 0.0) or not np.all(np.isfinite(vals)):
        raise ValueError("Computed kernel covariance is not positive definite; "
                         "target beam may be smaller than source or invalid")
    # Order by descending variance so that eigvecs[:,0] is major axis
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    sigma_x = np.sqrt(vals[0])
    sigma_y = np.sqrt(vals[1])
    # theta: angle of major-axis eigenvector relative to +x
    vx = vecs[:, 0]
    theta = np.arctan2(vx[1], vx[0])
    return Gaussian2DKernel(x_stddev=sigma_x, y_stddev=sigma_y, theta=theta, x_size=None, y_size=None)


def convolve_image_to_target_beam(
    data: np.ndarray,
    header: fits.Header,
    wcs: WCS,
    target_header: fits.Header,
    target_wcs: WCS,
) -> tuple[np.ndarray, fits.Header]:
    """
    Convolve an image to the target beam using covariance algebra:
    Σ_kernel_pix = Σ_target_pix - Σ_source_pix

    Returns
    -------
    convolved_data, new_header
    """
    Sigma_src_pix = beam_cov_in_pixels(header, wcs)
    Sigma_tgt_pix = beam_cov_in_pixels(target_header, target_wcs)

    Sigma_kernel_pix = Sigma_tgt_pix - Sigma_src_pix
    kernel = kernel_from_covariance_pix(Sigma_kernel_pix)

    out = convolve_fft(
        data,
        kernel.array,
        normalize_kernel=True,
        nan_treatment="interpolate",
        boundary="fill",
        fill_value=np.nan,
        allow_huge=True,
    )

    # Update header beam keywords to target beam values
    new_header = header.copy()
    for k in ("BMAJ", "BMIN", "BPA"):
        new_header[k] = target_header[k]

    return out, new_header


# --------------------------- Flux / Regions / Plots -------------------------- #

def compute_fluxes_and_nbeams(fits_path: Path, ds9reg: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Use the pipeline helper to compute integrated flux and Nbeams per region.
    """
    fluxes, _, _, nbeams = calculate_flux_and_peak_flux(fits_path, ds9reg)
    fluxes = np.asarray(fluxes, dtype=float)
    nbeams = np.asarray(nbeams, dtype=float)
    return fluxes, nbeams


def estimate_sigma_from_rms(rms_jy_per_beam: float, nbeams: np.ndarray) -> np.ndarray:
    """
    sigma_flux = rms * sqrt(Nbeams)
    """
    nbeams = np.asarray(nbeams, dtype=float)
    sig = float(rms_jy_per_beam) * np.sqrt(np.clip(nbeams, 0.0, np.inf))
    # For non-finite nbeams, set NaN
    sig[~np.isfinite(nbeams)] = np.nan
    return sig


def scatter_with_unity(
    x: np.ndarray,
    y: np.ndarray,
    xerr: Optional[np.ndarray],
    yerr: Optional[np.ndarray],
    out_png: Path,
    title: str,
    subtitle: Optional[str],
) -> None:
    """
    Make a scatter plot comparing two flux sets, adding a 1:1 line.
    Excludes non-finite points (assumed pre-filtered).
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if xerr is not None and yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", linestyle="none", alpha=0.8)
    else:
        ax.plot(x, y, "o", alpha=0.8)

    # Axes limits
    finite_all = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_all):
        ax.set_title(title)
        ax.text(0.5, 0.5, "No finite points to plot", transform=ax.transAxes, ha="center")
    else:
        xmin = np.nanmin(x[finite_all])
        xmax = np.nanmax(x[finite_all])
        ymin = np.nanmin(y[finite_all])
        ymax = np.nanmax(y[finite_all])
        lo = min(xmin, ymin)
        hi = max(xmax, ymax)
        pad = 0.05 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 1.0
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        # 1:1
        grid = np.linspace(lo - pad, hi + pad, 64)
        ax.plot(grid, grid, "--", lw=1.2, label="y = x")

    ax.set_xlabel("Input flux [Jy]")
    ax.set_ylabel("SPICE-RACS flux [Jy]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    if subtitle:
        ax.text(0.02, 0.98, subtitle, ha="left", va="top", transform=ax.transAxes)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def overlays_panel(
    img1_fits: Path,
    img2_fits: Path,
    ds9reg: Path,
    out_png: Path,
    cutout_size_arcmin: Optional[float] = None,
) -> None:
    """
    Two-panel figure showing both images with the regions overlaid.
    If cutout_size_arcmin provided and regions are present, shows a single
    representative region (first region) cutout for both images, centered on the same sky position.
    """
    regs = Regions.read(str(ds9reg))
    region_first = regs[0] if len(regs) > 0 else None

    def _load_for_display(fpath: Path) -> tuple[np.ndarray, Optional[WCS]]:
        data, hdr, _w = load_primary_image_2d(fpath)
        try:
            w = WCS(hdr).celestial
        except Exception:
            w = None
        return data, w

    data1, w1 = _load_for_display(img1_fits)
    data2, w2 = _load_for_display(img2_fits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    axes = [ax1, ax2]
    datas = [data1, data2]
    wcss = [w1, w2]
    titles = ["Input image", "SPICE-RACS image"]

    # Optionally, determine a common sky center for cutouts
    sky_center = None
    if region_first is not None and hasattr(region_first, "center"):
        ctr = region_first.center
        sky = ctr.to_skycoord() if hasattr(ctr, "to_skycoord") else ctr
        sky_center = sky

    for ax, data, w, tt in zip(axes, datas, wcss, titles):
        if w is not None and cutout_size_arcmin is not None and sky_center is not None:
            size = (cutout_size_arcmin * u.arcmin, cutout_size_arcmin * u.arcmin)
            try:
                co = Cutout2D(data=data, position=sky_center, size=size, wcs=w, mode="trim")
                data_show = co.data
                w_show = co.wcs
                ax = plt.subplot(1, 2, 1 if tt == titles[0] else 2, projection=w_show)
            except Exception:
                data_show = data
                w_show = w
                ax = plt.subplot(1, 2, 1 if tt == titles[0] else 2, projection=w_show) if w_show is not None else ax
        else:
            data_show = data
            w_show = w
            ax = plt.subplot(1, 2, 1 if tt == titles[0] else 2, projection=w_show) if w_show is not None else ax

        norm = ImageNormalize(data_show, interval=ZScaleInterval())
        ax.imshow(data_show, origin="lower", norm=norm, cmap="gray")
        ax.set_title(tt)

        if w_show is not None:
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
        else:
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")

        # Overlay all regions
        try:
            for r in regs:
                rr = r
                if hasattr(r, "to_pixel") and w_show is not None:
                    rr = r.to_pixel(w_show)
                rr.plot(ax=ax, lw=1.0)
        except Exception as e:
            ax.text(0.02, 0.02, f"Region overlay failed: {e}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------- Main pipeline ------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare integrated fluxes in DS9 regions between an input MFS image and a SPICE-RACS image. "
                    "The higher-resolution image is convolved to the lower resolution using Gaussian covariance algebra."
    )
    p.add_argument("--image_input", required=True, type=Path, help="Path to Stokes I MFS FITS image (reference).")
    p.add_argument("--image_spiceracs", required=True, type=Path, help="Path to SPICE-RACS Stokes I FITS image.")
    p.add_argument("--ds9reg", required=True, type=Path, help="DS9 region file with one or more source regions.")
    p.add_argument("--output_dir", required=True, type=Path, help="Directory to save CSV and plots.")

    p.add_argument("--rms_input", required=False, type=float, default=None,
                   help="Global rms of input image [Jy/beam] for sigma estimates.")
    p.add_argument("--rms_spiceracs", required=False, type=float, default=None,
                   help="Global rms of SPICE-RACS image [Jy/beam] for sigma estimates.")

    p.add_argument("--show_overlays", action="store_true",
                   help="If set, produce a two-panel overlay figure with DS9 regions.")
    p.add_argument("--cutout_size_arcmin", type=float, default=None,
                   help="If overlays are requested, show cutouts of this size (arcmin) centered on the first region.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both images
    data_in, hdr_in, wcs_in = load_primary_image_2d(args.image_input)
    data_sp, hdr_sp, wcs_sp = load_primary_image_2d(args.image_spiceracs)

    # Decide which is lower resolution by beam area (deg^2)
    bmaj_in, bmin_in, _ = get_beam_from_header(hdr_in)
    bmaj_sp, bmin_sp, _ = get_beam_from_header(hdr_sp)
    area_in = np.pi * bmaj_in * bmin_in / (4.0 * np.log(2.0))
    area_sp = np.pi * bmaj_sp * bmin_sp / (4.0 * np.log(2.0))

    # Convolve higher-res to lower-res
    temp_dir = out_dir / "tmp_convolution"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_for_flux = args.image_input
    spiceracs_for_flux = args.image_spiceracs

    if area_in < area_sp:
        # Input has higher resolution -> convolve input to SPICE-RACS beam
        conv_data, conv_hdr = convolve_image_to_target_beam(
            data_in, hdr_in, wcs_in, hdr_sp, wcs_sp
        )
        conv_path = temp_dir / f"{args.image_input.stem}_to_{args.image_spiceracs.stem}_beam.fits"
        write_fits_like(conv_path, conv_data.astype(np.float32), conv_hdr)
        input_for_flux = conv_path
        print(f"Convolved INPUT -> target SPICE-RACS beam: {conv_path}")
    elif area_sp < area_in:
        # SPICE-RACS has higher resolution -> convolve SPICE-RACS to INPUT beam
        conv_data, conv_hdr = convolve_image_to_target_beam(
            data_sp, hdr_sp, wcs_sp, hdr_in, wcs_in
        )
        conv_path = temp_dir / f"{args.image_spiceracs.stem}_to_{args.image_input.stem}_beam.fits"
        write_fits_like(conv_path, conv_data.astype(np.float32), conv_hdr)
        spiceracs_for_flux = conv_path
        print(f"Convolved SPICE-RACS -> target INPUT beam: {conv_path}")
    else:
        print("Beams appear equal in area; no convolution applied.")

    # Compute region fluxes and Nbeams
    flux_in, nbeams_in = compute_fluxes_and_nbeams(input_for_flux, args.ds9reg)
    flux_sp, nbeams_sp = compute_fluxes_and_nbeams(spiceracs_for_flux, args.ds9reg)

    # Uncertainties if provided
    sigma_in = estimate_sigma_from_rms(args.rms_input, nbeams_in) if args.rms_input is not None else None
    sigma_sp = estimate_sigma_from_rms(args.rms_spiceracs, nbeams_sp) if args.rms_spiceracs is not None else None

    # Build table
    num = max(len(flux_in), len(flux_sp))
    idx = np.arange(num, dtype=int)

    # Pad to same length if required (should not be necessary if helper returns per-region arrays)
    def _safe(a: np.ndarray, n: int) -> np.ndarray:
        return a if len(a) == n else np.pad(a, (0, max(0, n - len(a))), constant_values=np.nan)

    flux_in = _safe(flux_in, num)
    flux_sp = _safe(flux_sp, num)
    nbeams_in = _safe(nbeams_in, num)
    nbeams_sp = _safe(nbeams_sp, num)
    if sigma_in is not None:
        sigma_in = _safe(sigma_in, num)
    if sigma_sp is not None:
        sigma_sp = _safe(sigma_sp, num)

    ratio = flux_sp / flux_in
    diff = flux_sp - flux_in
    diff_pct = 100.0 * diff / flux_in

    # Save CSV
    csv_path = out_dir / "target_vs_spiceracs.csv"
    cols = [
        "region_index",
        "flux_input_Jy",
        "flux_spiceracs_Jy",
        "nbeams_input",
        "nbeams_spiceracs",
        "ratio_spiceracs_over_input",
        "diff_Jy",
        "diff_percent",
    ]
    if sigma_in is not None:
        cols += ["sigma_input_Jy"]
    if sigma_sp is not None:
        cols += ["sigma_spiceracs_Jy"]

    lines = [",".join(cols)]
    for i in range(num):
        row = [
            str(i + 1),
            f"{flux_in[i]:.8g}",
            f"{flux_sp[i]:.8g}",
            f"{nbeams_in[i]:.6g}",
            f"{nbeams_sp[i]:.6g}",
            f"{ratio[i]:.6g}",
            f"{diff[i]:.8g}",
            f"{diff_pct[i]:.6g}",
        ]
        if sigma_in is not None:
            row.append(f"{sigma_in[i]:.8g}")
        if sigma_sp is not None:
            row.append(f"{sigma_sp[i]:.8g}")
        lines.append(",".join(row))

    csv_path.write_text("\n".join(lines))
    print(f"Wrote {csv_path}")

    # Plot: exclude non-finite points
    finite_mask = np.isfinite(flux_in) & np.isfinite(flux_sp)
    x = flux_in[finite_mask]
    y = flux_sp[finite_mask]
    xerr = sigma_in[finite_mask] if sigma_in is not None else None
    yerr = sigma_sp[finite_mask] if sigma_sp is not None else None

    if x.size == 0:
        print("No finite flux pairs to plot; skipping scatter.")
    else:
        med_ratio = np.nanmedian(y / x)
        med_pct = 100.0 * (med_ratio - 1.0)
        title = "SPICE-RACS vs Input flux"
        subtitle = f"Median ratio = {med_ratio:.3f} ({med_pct:+.2f} percent)"
        png_path = out_dir / "target_vs_spiceracs.png"
        scatter_with_unity(x, y, xerr, yerr, png_path, title, subtitle)
        print(f"Wrote {png_path}")

    # Optional overlays
    if args.show_overlays:
        ov_png = out_dir / "target_vs_spiceracs_overlays.png"
        overlays_panel(Path(input_for_flux), Path(spiceracs_for_flux), args.ds9reg, ov_png, args.cutout_size_arcmin)
        print(f"Wrote {ov_png}")


if __name__ == "__main__":
    main()
