#!/usr/bin/env python3
"""
Flag image frequencies by interpolating a flag-percentage curve.

Given:
- centers_mhz: 1-D array of bin centers in MHz (from your earlier script)
- avg_flag_pct: 1-D array of average flag percentages per bin (same length)
- image_freqs: 1-D array of arbitrary frequencies to test (floats in MHz or astropy Quantity)
- threshold_pct: float, e.g. 20.0 means "flag if interpolated >= 20%"

Returns:
- flag_mask: boolean array, True where image_freqs should be flagged
- interp_pct: float array of interpolated (or filled) flag percentages at image_freqs

Notes on handling gaps and edges:
- Bins with no data often produce NaNs in avg_flag_pct. By default we *skip*
  those bins when forming the interpolation (linear over the finite points).
- Outside the min..max(centers_mhz), behavior is controlled by `outside`:
    * "nan": assign NaN (will not flag unless you set treat_nan_as_flag=True)
    * "extend": use the nearest edge value (left/right)
    * "false": treat as 0.0%
- You can force NaNs to be flagged by setting treat_nan_as_flag=True.

Example:
    flag_mask, interp = flag_image_freqs(
        centers_mhz, avg_flag_pct, image_freqs,
        threshold_pct=20.0, outside="extend"
    )
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from astropy import units as u

FloatArray = Union[np.ndarray, list, tuple]


def _to_mhz(x: FloatArray | u.Quantity) -> np.ndarray:
    """Return values as float64 MHz, accepting bare floats or Quantity."""
    if isinstance(x, u.Quantity):
        return x.to_value(u.MHz, equivalencies=u.spectral())
    x = np.asarray(x, dtype=np.float64)
    return x


def interpolate_flag_curve(
    centers_mhz: FloatArray,
    avg_flag_pct: FloatArray,
    query_mhz: FloatArray,
    *,
    outside: Literal["nan", "extend", "false"] = "nan",
) -> np.ndarray:
    """
    Interpolate flag-percentage curve to arbitrary query frequencies (MHz).

    Linear interpolation over finite points only. Behavior outside the covered
    range is controlled by `outside` (see module docstring).
    """
    x = np.asarray(centers_mhz, dtype=np.float64)
    y = np.asarray(avg_flag_pct, dtype=np.float64)
    q = np.asarray(query_mhz, dtype=np.float64)

    # Keep only finite points for building the interpolant
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        # Not enough information to interpolate; return all-NaN
        return np.full(q.shape, np.nan, dtype=np.float64)

    xs = x[finite]
    ys = y[finite]

    # Ensure strictly increasing x for numpy.interp
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    if outside == "nan":
        left = np.nan
        right = np.nan
    elif outside == "extend":
        # Nearest-edge extension
        left = float(ys[0])
        right = float(ys[-1])
    elif outside == "false":
        # Treat outside as 0% flagged
        left = 0.0
        right = 0.0
    else:
        raise ValueError('outside must be one of "nan", "extend", "false"')

    interp = np.interp(q, xs, ys, left=left, right=right)

    # np.interp cannot produce NaN inside the covered range; if the original
    # y had internal gaps (NaNs) we already removed them and interpolated
    # linearly across those gaps. If you prefer to *not* bridge wide gaps,
    # you could add logic to invalidate interpolation where spacing is large.

    return interp


def flag_image_freqs(
    centers_mhz: FloatArray | u.Quantity,
    avg_flag_pct: FloatArray | u.Quantity,
    image_freqs: FloatArray | u.Quantity,
    *,
    threshold_pct: float = 20.0,
    outside: Literal["nan", "extend", "false"] = "nan",
    treat_nan_as_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute boolean flags for image_freqs by interpolating avg_flag_pct.

    Args:
        centers_mhz: bin centers (MHz) as floats or Quantity
        avg_flag_pct: average flag percentages (0..100)
        image_freqs: query freqs (floats in MHz or Quantity convertible to MHz)
        threshold_pct: flag if interpolated >= threshold_pct
        outside: how to treat frequencies outside centers range
        treat_nan_as_flag: if True, NaNs in the interpolated values are treated as flagged

    Returns:
        (flag_mask, interp_pct)
    """
    centers_mhz = _to_mhz(centers_mhz)
    image_mhz = _to_mhz(image_freqs)

    # Allow avg_flag_pct as plain floats; if Quantity, strip units
    if isinstance(avg_flag_pct, u.Quantity):
        avg_flag_pct = avg_flag_pct.to_value(u.percent)

    interp_pct = interpolate_flag_curve(
        centers_mhz=centers_mhz,
        avg_flag_pct=np.asarray(avg_flag_pct, dtype=np.float64),
        query_mhz=image_mhz,
        outside=outside,
    )

    if treat_nan_as_flag:
        flag_mask = np.where(np.isnan(interp_pct), True, interp_pct >= threshold_pct)
    else:
        flag_mask = np.where(np.isnan(interp_pct), False, interp_pct >= threshold_pct)

    return flag_mask, interp_pct


# ------------------------------
# Optional demo / sanity check
# ------------------------------

def _demo() -> None:
    # Fake inputs for quick testing
    centers = np.array([900, 950, 1000, 1050, 1100], dtype=np.float64)  # MHz
    avg_pct = np.array([5, 10, 40, np.nan, 30], dtype=np.float64)       # %
    image_freqs = np.array([880, 930, 975, 1005, 1080, 1120], dtype=np.float64)

    threshold = 20.0

    mask, interp = flag_image_freqs(
        centers, avg_pct, image_freqs,
        threshold_pct=threshold,
        outside="extend",          # try "nan" or "false" as well
        treat_nan_as_flag=False,
    )
    print("centers MHz:", centers)
    print("avg_flag_pct %:", avg_pct)
    print('=======================')
    print("image_freqs MHz:", image_freqs)
    print("interp %:", np.round(interp, 2))
    print(f"threshold {threshold} %")
    print("flag mask:", mask)


if __name__ == "__main__":
    _demo()
