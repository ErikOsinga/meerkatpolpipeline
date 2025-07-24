#!/usr/bin/env python3
"""
ms_field_scan_intents.py

Extracts, for each field in a MeasurementSet, the first associated
scan intent (from STATE/OBS_MODE) and either prints it or writes to CSV.

for more info see https://www.aoc.nrao.edu/~sbhatnag/misc/msselection/ScanintentBasedSelection.html#x20-320008

Usage:
    python ms_field_scan_intents.py /path/to/your.ms [--outfile_csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from casacore.tables import table

# 3c138 and 3c286, often have intent = "UNKNOWN"
POL_CALIBRATORS = ["J0521+1638", "J1331+3030"]

def get_scan_intents(msfile: Path) -> list[str]:
    """
    Read the STATE subtable's OBS_MODE column.

    Parameters
    ----------
    msfile : Path
        Path to the MeasurementSet directory.

    Returns
    -------
    List[str]
        List of intent strings, indexed by state row number.
    """
    state_tbl = table(str(msfile / "STATE"), ack=False)
    try:
        raw_modes = state_tbl.getcol("OBS_MODE")
    finally:
        state_tbl.close()

    modes: list[str] = []
    for entry in raw_modes:
        if isinstance(entry, (bytes, bytearray)):
            modes.append(entry.decode("utf-8"))
        else:
            modes.append(str(entry))
    return modes


def get_main_ids(msfile: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read STATE_ID and FIELD_ID columns from the main MS table.

    Returns
    -------
    state_ids : np.ndarray[int]
    field_ids : np.ndarray[int]
    """
    main_tbl = table(str(msfile), ack=False)
    try:
        state_ids = main_tbl.getcol("STATE_ID")
        field_ids = main_tbl.getcol("FIELD_ID")
    finally:
        main_tbl.close()
    return state_ids, field_ids


def get_field_names(msfile: Path) -> list[str]:
    """
    Read field names from FIELD/NAME column.

    Returns
    -------
    names : List[str]
    """
    field_tbl = table(str(msfile / "FIELD"), ack=False)
    try:
        raw_names = field_tbl.getcol("NAME")
    finally:
        field_tbl.close()

    names: list[str] = []
    for entry in raw_names:
        if isinstance(entry, (bytes, bytearray)):
            names.append(entry.decode("utf-8"))
        else:
            names.append(str(entry))
    return names


def map_fields_to_intents(
    msfile: Path
) -> list[tuple[int, str, str]]:
    """
    For each unique field ID, find the first row in the main table
    and report (field_id, field_name, intent).

    Returns
    -------
    List of tuples: (field_id, field_name, intent)
    """
    intents = get_scan_intents(msfile)
    state_ids, field_ids = get_main_ids(msfile)
    field_names = get_field_names(msfile)

    unique_fids = np.unique(field_ids)
    result = []
    for fid in unique_fids:
        rows = np.where(field_ids == fid)[0]
        if rows.size == 0:
            continue
        first_row = rows[0]
        state_idx = int(state_ids[first_row])
        intent = intents[state_idx]
        name = field_names[fid]

        if name in POL_CALIBRATORS:
            if "CALIBRATE_POL" not in intent:
                intent += (",CALIBRATE_POL")

        result.append((int(fid), name, intent))
        
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or export first scan intent for each MS field"
    )
    parser.add_argument(
        "msfile",
        type=Path,
        help="Path to the MeasurementSet directory"
    )
    parser.add_argument(
        "--outfile_csv",
        type=Path,
        help="If given, write results to this CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    msfile = args.msfile

    mapping = map_fields_to_intents(msfile)

    if args.outfile_csv:
        with open(args.outfile_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["field_id", "field_name", "intent_string"])
            for fid, name, intent in mapping:
                writer.writerow([fid, name, intent])
        print(f"Written CSV to {args.outfile_csv}")
    else:
        for fid, name, intent in mapping:
            print(f"fid={fid}, field_name={name}, intent={intent}")


if __name__ == "__main__":
    main()

