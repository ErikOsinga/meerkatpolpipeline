#!/usr/bin/env python3
"""
remove_missing_baseline_timesteps.py

Identify MS timesteps whose number of cross-correlation baselines is not equal
to the uniform maximum (e.g., 2016 for MeerKAT with 64 ants), report how many,
and write a new MS that excludes those timesteps (suffix: .fixed.ms).

Requirements:
- python-casacore (casacore.tables)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from casacore.tables import table, taql


class PrintLogger:
    """Custom logger that prints to stdout."""
    def info(self, msg):
        print(msg)
    def warning(self, msg):
        print("WARNING:", msg)
    def error(self, msg):
        print("ERROR:", msg)


def compute_uniform_count(ms: Path) -> tuple[int, int]:
    """
    Return (ntimes_total, nmax, nmin) where 
        ntimes_total is the number of timesteps in the MS, and;
        nmax is the maximum baseline count per TIME across the MS
        nmin is the minimum baseline count per TIME across the MS

    """

    # Total number of distinct TIME bins considered
    # q_total = (
    #     f"SELECT COUNT() AS ntimes FROM ("
    #     f"  SELECT TIME FROM {ms} WHERE {where} GROUP BY TIME"
    #     f")"
    # )  ## this syntax doesnt work, implemented as below.

    q_total = (
        f"SELECT TIME, gcount(*) as nbaselines from {ms} groupby TIME"
    )

    t_total = taql(q_total)
    timecol = t_total.getcol("TIME")
    nbaselines = t_total.getcol("nbaselines")

    ntimes_total = len(timecol)
    nmax = np.max(nbaselines)
    nmin = np.min(nbaselines)
    t_total.close()

    return ntimes_total, nmax, nmin


def materialize_good_times(ms: Path, nmax: int, good_tab: Path) -> int:
    """
    Create a small casacore table with one column TIME containing all 'good' timesteps,
    i.e., those whose baseline count equals nmax. Returns number of good timesteps.
    """
    # Write good times to a tiny table on disk to avoid floating-point equality issues.
    q_good = (
        f"SELECT TIME, gcount(*) as nbaselines from {ms} groupby TIME having gcount(*)=={nmax} giving '{good_tab}'"
    )
    taql(q_good)

    t_good = table(str(good_tab), readonly=True, ack=False)
    ngood = t_good.nrows()
    t_good.close()

    return int(ngood)


def materialize_bad_times(ms: Path, nmax: int, bad_tab: Path) -> int:
    """
    Create a small table of bad timesteps and return how many there are.
    """
    q_bad = (
        f"SELECT TIME, gcount(*) as nbaselines from {ms} groupby TIME having gcount(*)!={nmax} giving '{bad_tab}'"
    )
    taql(q_bad)

    t_bad = table(str(bad_tab), readonly=True, ack=False)
    nbad = t_bad.nrows()
    t_bad.close()

    return int(nbad)


def write_fixed_ms(ms: Path, good_tab: Path, out_ms: Path) -> None:
    """
    Join the original MS with the table of good timesteps and write a cleaned MS.
    """
    # Ensure destination does not exist
    if out_ms.exists():
        raise RuntimeError(f"Output MS already exists: {out_ms}")
    
    q_join = (
        f"SELECT FROM {ms} WHERE TIME IN (SELECT TIME FROM '{good_tab}') "
        f"GIVING '{out_ms}' AS PLAIN"
    )

    out_ms = taql(q_join)
    out_ms.close()
    
    return 

def preview_bad_examples(bad_tab: Path, limit: int = 10) -> list[tuple[float, int]]:
    """
    Return up to `limit` example rows from the bad-times table as (TIME, n) tuples.
    """
    t = table(str(bad_tab), readonly=True, ack=False)
    n = min(limit, t.nrows())
    if n == 0:
        t.close()
        return []
    times = t.getcol("TIME", 0, n)
    counts = t.getcol("nbaselines", 0, n)
    t.close()

    return [(float(times[i]), int(counts[i])) for i in range(n)]


def remove_missing_baseline_timesteps(ms: Path, out_ms: Path | None, keep_temp: bool = False, show_bad: int = 0, logger = None) -> Path:
    """
    Main function to remove timesteps with missing baselines from an MS.

    Parameters
    ----------
    ms : Path
        Path to input MeasurementSet.
    out_ms : Path
        Path to output MeasurementSet (will be created).
    keep_temp : bool, optional
        Whether to keep temporary helper tables, by default False.
    show_bad : int, optional
        Number of bad timestamps to print as examples, by default 0.

    RETURNS:
    -------
    Path to the output MeasurementSet with uniform number of baselines. (same as input if no changes made).
    """
    if logger is None:
        logger = PrintLogger()
    
    if not ms.exists():
        raise FileNotFoundError(f"MS not found: {ms}")
    
    if out_ms is None:
        out_ms = ms.with_suffix(".fixed.ms")

    # Temp tables (same directory as output)
    base_dir = out_ms.parent
    good_tab = base_dir / "good_times.tmp.tab"
    bad_tab = base_dir / "bad_times.tmp.tab"

    # Clean any stale temp tables
    for tmp in (good_tab, bad_tab):
        if tmp.exists():
            shutil.rmtree(tmp)

    # 1) Determine the uniform baseline count (maximum per-TIME)
    ntimes_total, nmax, nmin = compute_uniform_count(ms)

    # 2) Materialize good and bad timesteps
    ngood = materialize_good_times(ms, nmax, good_tab)
    nbad = materialize_bad_times(ms, nmax, bad_tab)

    logger.info(f"Total TIME bins considered: {ntimes_total}")
    logger.info(f"Uniform baseline count (max per TIME): {nmax}")
    logger.info(f"Good TIME bins: {ngood}")
    logger.info(f"Bad  TIME bins: {nbad}")

    if nbad > 0 and show_bad > 0:
        examples = preview_bad_examples(bad_tab, limit=show_bad)
        if examples:
            logger.info("Examples of bad timesteps (TIME [s], baseline_count):")
            for tval, cnt in examples:
                logger.info(f"  {tval:.3f}  {cnt}")

    if nbad == 0:
        logger.info("No bad timesteps detected. Not writing a new MS.")
        if not keep_temp:
            for tmp in (good_tab, bad_tab):
                if tmp.exists():
                    shutil.rmtree(tmp)
        return ms
    
    if ngood < nbad:
        logger.warning(f"More bad timesteps ({nbad}) than good ({ngood}); check selection.")
        logger.warning("Not writing fixed MS")
        return ms

    # 3) Write fixed MS
    logger.info(f"Writing fixed MS to: {out_ms}")
    write_fixed_ms(ms, good_tab, out_ms)

    # 4) Cleanup
    if not keep_temp:
        for tmp in (good_tab, bad_tab):
            if tmp.exists():
                shutil.rmtree(tmp)

    # 5) Sanity check: confirm uniformity on the output
    ntimes_fixed, nmax_fixed, nmin_fixed = compute_uniform_count(out_ms)

    logger.info(f"Fixed MS TIME bins: {ntimes_fixed}  |  per-TIME baselines min/max: {nmin_fixed}/{nmax_fixed}")

    if nmin_fixed != nmax_fixed:
        logger.warning("per-TIME baseline counts are still non-uniform! Something went wrong.")
        raise ValueError("Fixed MS has non-uniform per-TIME baseline counts.")
    else:
        logger.info("Success: per-TIME baseline counts are uniform.")
    
    return out_ms


def main() -> None:
    p = argparse.ArgumentParser(description="Remove timesteps with missing baselines and write a fixed MS.")
    p.add_argument("ms", help="Path to input MeasurementSet", type=Path)
    p.add_argument("--out", default=None, help="Output MS path (default: <ms>.fixed.ms)")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary helper tables")
    p.add_argument("--show-bad", type=int, default=8, help="Print up to this many bad timestamps (default: 8)")
    args = p.parse_args()

    remove_missing_baseline_timesteps(args.ms, args.out, args.keep_temp, args.show_bad)

if __name__ == "__main__":
    main()