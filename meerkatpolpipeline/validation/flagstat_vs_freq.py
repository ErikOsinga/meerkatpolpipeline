#!/usr/bin/env python3
"""
Compute and plot flagging percentage vs. frequency from one or more Measurement Sets.

- Uses casacore.tables to read MAIN, DATA_DESCRIPTION, and SPECTRAL_WINDOW.
- Aggregates flags across all times and correlations, honoring FLAG_ROW.
- Bins by a common global frequency grid (default bin width: 50 MHz) computed over all inputs.
- Prints tables per MS and for the average (sum flagged_count and total_count across MS).
- Saves per-MS plots in --figdir, and the average plot as flag_vs_freq_avg.png in --figdir.
- CSV output:
    * If one MS and --out is a file path -> write that CSV.
    * If multiple MS and --out is provided -> treated as a directory; per-MS CSVs plus an average CSV are written there.

Example:
    python ms_flag_vs_freq_multi.py my1.ms my2.ms \
        --bin-width-mhz 50 --figdir ./plots_flagstat/ --out ./csv_out/

    or with glob:

    python /data2/osinga/meerkatBfields/flag_vs_freq.py \
    ./selfcal/DDcal/Abell754_*copy_chunk*.ms.copy.copy.subtracted_ddcor \
    --figdir ./plots_flagstat/ --out ./csv_out/

Requirements:
    - casacore.tables
    - numpy
    - astropy
    - matplotlib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from casacore.tables import table


def read_spw_info(ms_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Read per-SPW channel frequencies and widths for a single MS.

    Returns:
        dict: {spw_id: (chan_freq_hz, chan_width_hz)}, arrays are float64 1-D
    """
    t_spw = table(str(ms_path / "SPECTRAL_WINDOW"), ack=False, readonly=True)
    try:
        chan_freq = t_spw.getcol("CHAN_FREQ")
        chan_width = t_spw.getcol("CHAN_WIDTH")
    finally:
        t_spw.close()

    spw_info: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for spw_id in range(chan_freq.shape[0]):
        freqs = np.array(chan_freq[spw_id], dtype=np.float64)
        widths = np.array(chan_width[spw_id], dtype=np.float64)
        spw_info[spw_id] = (freqs, widths)
    return spw_info


def read_ddi_to_spw(ms_path: Path) -> np.ndarray:
    """Map DATA_DESC_ID -> SPECTRAL_WINDOW_ID for a single MS."""
    t_dd = table(str(ms_path / "DATA_DESCRIPTION"), ack=False, readonly=True)
    try:
        spw_ids = t_dd.getcol("SPECTRAL_WINDOW_ID").astype(np.int64)
    finally:
        t_dd.close()
    return spw_ids


def global_freq_edges(all_spw_infos: list[dict[int, tuple[np.ndarray, np.ndarray]]],
                      bin_width_mhz: float) -> np.ndarray:
    """
    Build global frequency bin edges covering ALL provided MS.

    Returns:
        np.ndarray of bin edges in Hz
    """
    all_min = np.inf
    all_max = -np.inf
    for spw_info in all_spw_infos:
        for freqs_hz, widths_hz in spw_info.values():
            if freqs_hz.size == 0:
                continue
            half = np.abs(widths_hz) / 2.0
            all_min = min(all_min, np.min(freqs_hz - half))
            all_max = max(all_max, np.max(freqs_hz + half))

    if not np.isfinite(all_min) or not np.isfinite(all_max) or all_max <= all_min:
        raise ValueError("Could not determine global frequency range from the provided MS list.")

    bw = (bin_width_mhz * u.MHz).to_value(u.Hz)
    n_bins = int(np.ceil((all_max - all_min) / bw))
    edges = all_min + np.arange(n_bins + 1, dtype=np.float64) * bw
    return edges


def accumulate_for_ddi(ms_path: Path,
                       ddi: int,
                       spw_freqs_hz: np.ndarray,
                       edges_hz: np.ndarray,
                       chunk_rows: int,
                       accum_flagged: np.ndarray,
                       accum_total: np.ndarray) -> None:
    """
    Process one DATA_DESC_ID selection in chunks, updating accumulators.
    """
    # Precompute bin index per channel
    chan_bin_idx = np.digitize(spw_freqs_hz, edges_hz) - 1
    valid = (chan_bin_idx >= 0) & (chan_bin_idx < accum_flagged.size)
    if not np.any(valid):
        return
    valid_idx = np.where(valid)[0]
    chan_bin_idx = chan_bin_idx[valid]

    t_main_sel = table(str(ms_path), ack=False, readonly=True).query(
        f"DATA_DESC_ID=={ddi}", style="python"
    )
    try:
        nrows = t_main_sel.nrows()
        if nrows == 0:
            return

        start = 0
        while start < nrows:
            stop = min(start + max(chunk_rows, 1), nrows)
            flag_block = np.asarray(t_main_sel.getcol("FLAG", startrow=start, nrow=stop - start), dtype=bool)
            flag_row = np.asarray(t_main_sel.getcol("FLAG_ROW", startrow=start, nrow=stop - start), dtype=bool)
            combined = np.logical_or(flag_block, flag_row[:, None, None])
            combined = combined[:, valid_idx, :]

            flagged_per_chan = combined.sum(axis=(0, 2)).astype(np.int64)
            total_per_chan = np.int64(combined.shape[0] * combined.shape[2])

            np.add.at(accum_flagged, chan_bin_idx, flagged_per_chan)
            np.add.at(accum_total, chan_bin_idx, total_per_chan)

            start = stop
    finally:
        t_main_sel.close()


def compute_flag_vs_freq_single(ms_path: Path,
                                edges_hz: np.ndarray,
                                chunk_rows: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-MS flag stats on a pre-defined global frequency binning.

    Returns:
        centers_mhz, flag_percent, counts (n_bins x 2) where counts[:,0]=flagged, counts[:,1]=total
    """
    spw_info = read_spw_info(ms_path)
    ddi_to_spw = read_ddi_to_spw(ms_path)

    n_bins = edges_hz.size - 1
    accum_flagged = np.zeros(n_bins, dtype=np.int64)
    accum_total = np.zeros(n_bins, dtype=np.int64)

    for ddi, spw_id in enumerate(ddi_to_spw.tolist()):
        freqs_hz, _ = spw_info.get(spw_id, (np.array([], dtype=np.float64), np.array([], dtype=np.float64)))
        if freqs_hz.size == 0:
            continue
        accumulate_for_ddi(
            ms_path=ms_path,
            ddi=ddi,
            spw_freqs_hz=freqs_hz,
            edges_hz=edges_hz,
            chunk_rows=chunk_rows,
            accum_flagged=accum_flagged,
            accum_total=accum_total,
        )

    centers_hz = 0.5 * (edges_hz[:-1] + edges_hz[1:])
    centers_mhz = (centers_hz * u.Hz).to_value(u.MHz)
    with np.errstate(invalid="ignore", divide="ignore"):
        flag_percent = np.where(accum_total > 0,
                                (accum_flagged / accum_total) * 100.0,
                                np.nan)
    counts = np.vstack([accum_flagged, accum_total]).T
    return centers_mhz, flag_percent, counts


def write_csv(out_path: Path,
              centers_mhz: np.ndarray,
              flag_percent: np.ndarray,
              counts: np.ndarray) -> None:
    """Save results to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "freq_mhz_center,flag_percent,flagged_count,total_count"
    data = np.column_stack([centers_mhz, flag_percent, counts])
    np.savetxt(str(out_path), data, delimiter=",", header=header, comments="", fmt=["%.6f", "%.6f", "%d", "%d"])
    print(f"Wrote: {out_path}")


def plot_flag_vs_freq(centers_mhz: np.ndarray,
                      flag_percent: np.ndarray,
                      out_path: Path,
                      title: str,
                      image_freqs_MHz: np.ndarray | None = None
) -> None:
    """Plot and save flagging percentage vs. frequency."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(centers_mhz, flag_percent, marker="o", linestyle="-", markersize=3)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Flagged fraction (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # overlay center frequencies at which we have imaged, if given.
    if image_freqs_MHz is not None:
            for i, freq in enumerate(image_freqs_MHz):
                plt.axvline(freq, color="black", linestyle="--", alpha=0.7)
                plt.text(freq, 50, f"{i:02d}", color="black", fontsize=8, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def print_table(label: str,
                centers_mhz: np.ndarray,
                flag_percent: np.ndarray,
                counts: np.ndarray) -> None:
    """Print a compact table (only bins with data)."""
    print(f"\n=== {label} ===")
    print("freq_mhz_center, flag_percent, flagged_count, total_count")
    has_data = counts[:, 1] > 0
    for f, p, (fc, tc) in zip(centers_mhz[has_data], flag_percent[has_data], counts[has_data]):
        print(f"{f:.3f}, {p:.3f}, {int(fc)}, {int(tc)}")


def compute_flagstat_vs_freq(ms_paths: list[Path], bin_width_mhz: float, chunk_rows: int = 4096) -> tuple[list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute flagging percentage vs. frequency from one or more Measurement Sets.

    args:
        ms_paths: list of MS paths
        bin_width_mhz: frequency bin width in MHz
        chunk_rows: row chunk size when reading FLAG/FLAG_ROW

    returns:
        list of tuples per MS: (ms_path, centers_mhz, flag_percent, counts)
        and the average (centers_mhz, avg_flag_percent, sum_counts)
    """
    # 1) Build global binning across all MS
    all_spw_infos = [read_spw_info(ms) for ms in ms_paths]
    edges_hz = global_freq_edges(all_spw_infos, bin_width_mhz)
    centers_mhz = (0.5 * (edges_hz[:-1] + edges_hz[1:]) * u.Hz).to_value(u.MHz)

    # 2) Per-MS computation
    per_ms_results = []
    for ms in ms_paths:
        c_mhz, flag_pct, counts = compute_flag_vs_freq_single(ms, edges_hz, chunk_rows)
        per_ms_results.append((ms, c_mhz, flag_pct, counts))

    # 3) Average across all MS (sum counts, then fraction)
    sum_counts = np.zeros((edges_hz.size - 1, 2), dtype=np.int64)
    for _, _, _, counts in per_ms_results:
        sum_counts += counts
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_flag_pct = np.where(sum_counts[:, 1] > 0,
                                (sum_counts[:, 0] / sum_counts[:, 1]) * 100.0,
                                np.nan)

    return per_ms_results, centers_mhz, avg_flag_pct, sum_counts

def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    ms_paths = [Path(p).resolve() for p in args.ms]
    figdir = Path(args.figdir).resolve()

    per_ms_results, (centers_mhz, avg_flag_pct, sum_counts) = compute_flagstat_vs_freq(
        ms_paths=ms_paths,
        bin_width_mhz=args.bin_width_mhz,
        chunk_rows=args.chunk_rows,
    )

    # 4) Printing
    for ms, c_mhz, flag_pct, counts in per_ms_results:
        print_table(label=ms.name, centers_mhz=c_mhz, flag_percent=flag_pct, counts=counts)
    print_table(label="AVERAGE (all input MS)", centers_mhz=centers_mhz, flag_percent=avg_flag_pct, counts=sum_counts)

    # 5) Plotting per MS
    for ms, c_mhz, flag_pct, _ in per_ms_results:
        out_path = figdir / f"{ms.name}_flag_vs_freq.png"
        plot_flag_vs_freq(c_mhz, flag_pct, out_path, f"Flagging vs Frequency: {ms.name}")

    # 6) Plotting average
    avg_out = figdir / "flag_vs_freq_avg.png"
    plot_flag_vs_freq(centers_mhz, avg_flag_pct, avg_out, "Flagging vs Frequency: AVERAGE")

    # 7) CSV output logic
    if args.out:
        out_path = Path(args.out).resolve()
        if len(ms_paths) == 1:
            # Single MS: --out is a file path
            ms, c_mhz, flag_pct, counts = per_ms_results[0]
            write_csv(out_path, c_mhz, flag_pct, counts)
        else:
            # Multiple MS: --out is treated as a directory
            out_dir = out_path
            out_dir.mkdir(parents=True, exist_ok=True)
            # Per-MS CSVs
            for ms, c_mhz, flag_pct, counts in per_ms_results:
                write_csv(out_dir / f"{ms.name}_flag_vs_freq.csv", c_mhz, flag_pct, counts)
            # Average CSV
            write_csv(out_dir / "flag_vs_freq_avg.csv", centers_mhz, avg_flag_pct, sum_counts)

    return 0


def parse_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and plot flagging percentage vs. frequency from one or more Measurement Sets."
    )
    p.add_argument("ms", nargs="+", help="One or more Measurement Sets (.ms directories)")
    p.add_argument("--bin-width-mhz", type=float, default=10.0,
                   help="Frequency bin width in MHz (default: 10)")
    p.add_argument("--figdir", type=str, default="./plots_flagstat/",
                   help="Directory to save figures (default: ./plots_flagstat/)")
    p.add_argument("--out", type=str, default=None,
                   help="CSV output. If one MS, this is a file path. If multiple MS, this is treated as a directory.")
    p.add_argument("--chunk-rows", type=int, default=4096,
                   help="Row chunk size when reading FLAG/FLAG_ROW (default: 4096)")
    return p.parse_args(argv)


if __name__ == "__main__":
    # CLI
    sys.exit(main())
