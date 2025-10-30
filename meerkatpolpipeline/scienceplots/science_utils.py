from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass
class RunningStatisticsResult:
    """Container for runningstatistics outputs
    
    All statistics can either be float arrays (point estimates per bin) or object arrays (bootstrap distributions per bin)
    """

    left_bounds: np.ndarray
    """left bound of running bin"""
    medians_x: np.ndarray
    """median in x of running bin"""
    medians_y: np.ndarray
    """median in y of running bin"""
    stds_y: np.ndarray
    """std y of running bin"""
    err_low: np.ndarray
    """lower 1sigma uncertainty in median(y) of running bin"""
    err_up: np.ndarray
    """upper 1sigma uncertainty in median(y) of running bin"""
    iqrs_y: np.ndarray
    """IQR of y in running bin"""
    MADs_y: np.ndarray
    """MAD of y in running bin"""
    window_widths: np.ndarray
    """width of running bin in x units"""
    Npoints: np.ndarray
    """number of points in running bin"""
    Yerrcor_sq: np.ndarray | None  # None if seqYerr is None
    """If seqYerr is provided, Yerrcor_sq = sum(err^2)/(N-1) is reported per window"""
    means_y: np.ndarray
    """mean of y in running bin"""
    sterr_y: np.ndarray
    """standard error of the mean of y in running bin"""


def runningstatistics(
    seqX: ArrayLike,
    seqY: ArrayLike,
    xwidth: float | None,
    seqYerr: ArrayLike | None = None,
    nbootstrap: int | None = None,
    redshifts: ArrayLike | None = None,
    M: int | None = None,
) -> RunningStatisticsResult:
    """
    Running statistics in a sliding window that moves right 1 point at a time (correlated windows).

    Can have either:
     - equal width sliding window that moves to the right by 1 point at a time. 
        Set xwidth to a float.
     - OR a fixed number of points in the sliding window, that moves right by 1 point at a time.
        if xwidth=None and M!=None. (set to an int)


    Args:
        seqX: Sequence of X values (e.g., radius) to sort and slide over.
        seqY: Sequence of Y values (e.g., measurements) to compute statistics on.
        xwidth: Width of the sliding window in X units. If None, M must be provided.
        seqYerr: Optional sequence of errors on Y values. If provided, Yerrcor_sq is computed.
        nbootstrap: Optional number of bootstrap resamples for uncertainty estimation.
                    If None, no bootstrap is performed.
        redshifts: Optional sequence of redshifts corresponding to each (X,Y) pair.
                   Used for redshift correction of scatter-like quantities.
        M: Optional fixed number of points in the sliding window. If None, xwidth must be provided.


    Notes
    -----
    - Lamee+2016 (Fig. 11) prescription is used for error on the median.
    - If redshifts is provided, scatter-like quantities are multiplied by
      (1 + median(z_window))**2 (cf. Osinga+2024 eq. 6).
    - If seqYerr is provided, Yerrcor_sq = sum(err^2)/(N-1) is reported
      per window; otherwise it is None in the result.
    """

    # --- sort by X (e.g. radius) ---
    order = np.argsort(seqX)
    x = np.asarray(seqX)[order]
    y = np.asarray(seqY)[order]

    yerr = np.asarray(seqYerr)[order] if seqYerr is not None else None
    z = np.asarray(redshifts)[order] if redshifts is not None else None

    # --- collectors ---
    left_bounds: list[float] = []
    med_x: list[float] = []
    med_y: list[np.ndarray | float] = []
    std_y: list[np.ndarray | float] = []
    e_low: list[float] = []
    e_up: list[float] = []
    iqr_y: list[np.ndarray | float] = []
    mad_y: list[np.ndarray | float] = []
    widths: list[float] = []
    n_pts: list[int] = []
    yerrcor_sq: list[float] = []
    mean_y: list[np.ndarray | float] = []
    ste_y: list[np.ndarray | float] = []

    # --- helpers (Lamee+2016) ---
    def err_lamee_low(arr: np.ndarray) -> float:
        med = np.median(arr)
        return np.abs(med - np.percentile(arr, 16)) / np.sqrt(len(arr))

    def err_lamee_up(arr: np.ndarray) -> float:
        med = np.median(arr)
        return np.abs(med - np.percentile(arr, 84)) / np.sqrt(len(arr))

    n = len(x)
    for i in range(n):
        # --- window bounds ---
        if xwidth is not None:
            left = x[i]
            right = x[i] + xwidth
        elif M is not None and (i + M) <= n:
            left = x[i]
            right = x[i + M - 1]
        else:
            # Not enough points for the fixed-count window at the tail
            continue

        idx = np.where((x >= left) & (x <= right))[0]
        if idx.size <= 1:
            continue  # need at least 2 points for scatter

        xw = x[idx]
        yw = y[idx]
        yw_err = yerr[idx] if yerr is not None else None
        zw = z[idx] if z is not None else None

        # --- invariants (no bootstrap) ---
        left_bounds.append(left)
        med_x.append(np.median(xw))
        e_low.append(err_lamee_low(yw))
        e_up.append(err_lamee_up(yw))
        widths.append(np.max(xw) - np.min(xw))
        n_pts.append(idx.size)

        if yw_err is not None and idx.size > 1:
            yerrcor_sq.append(np.sum(yw_err ** 2) / (idx.size - 1))

        # redshift correction factor for scatter-like quantities
        rz_corr = (1.0 + np.median(zw)) ** 2 if zw is not None else 1.0

        # --- point estimates OR bootstrap distributions ---
        if nbootstrap is None:
            med_y.append(np.median(yw))
            std_y.append(np.std(yw, ddof=1) * rz_corr)
            iqr_y.append(stats.iqr(yw) * rz_corr)
            mad_y.append(stats.median_abs_deviation(yw) * rz_corr)
            mean_y.append(np.mean(yw))
            ste_y.append(np.std(yw, ddof=1) * rz_corr / np.sqrt(idx.size))
        else:
            # bootstrap within the window
            reps = np.asarray(
                [np.random.default_rng().integers(0, idx.size, idx.size) for _ in range(nbootstrap)]
            )
            yw_boot = yw[reps]  # shape: (nbootstrap, idx.size)

            med_y.append(np.median(yw_boot, axis=1))
            std_y.append(np.std(yw_boot, axis=1, ddof=1) * rz_corr)
            iqr_y.append(stats.iqr(yw_boot, axis=1) * rz_corr)
            mad_y.append(stats.median_abs_deviation(yw_boot, axis=1) * rz_corr)
            mean_y.append(np.mean(yw_boot, axis=1))
            ste_y.append(np.std(yw_boot, axis=1, ddof=1) * rz_corr / np.sqrt(idx.size))

    # When seqYerr is None, we set Yerrcor_sq to None
    yerrcor_sq_arr = np.array(yerrcor_sq) if yerr is not None else None

    return RunningStatisticsResult(
        left_bounds=np.asarray(left_bounds),
        medians_x=np.asarray(med_x),
        medians_y=np.asarray(med_y, dtype=object) if nbootstrap is not None else np.asarray(med_y),
        stds_y=np.asarray(std_y, dtype=object) if nbootstrap is not None else np.asarray(std_y),
        err_low=np.asarray(e_low),
        err_up=np.asarray(e_up),
        iqrs_y=np.asarray(iqr_y, dtype=object) if nbootstrap is not None else np.asarray(iqr_y),
        MADs_y=np.asarray(mad_y, dtype=object) if nbootstrap is not None else np.asarray(mad_y),
        window_widths=np.asarray(widths),
        Npoints=np.asarray(n_pts),
        Yerrcor_sq=yerrcor_sq_arr,
        means_y=np.asarray(mean_y, dtype=object) if nbootstrap is not None else np.asarray(mean_y),
        sterr_y=np.asarray(ste_y, dtype=object) if nbootstrap is not None else np.asarray(ste_y),
    )
