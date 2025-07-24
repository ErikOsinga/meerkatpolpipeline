"""Utilities related to measurement sets
Shamelessly adapted from the flint project.

"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from meerkatpolpipeline.utils import execute_command


def msoverview_summary(
        binds: list[Path],
        container: Path,
        ms: Path,
        output_to_file: Path = "./file.txt",
        get_intents: bool = False,
    ) -> dict[str, Any]:
    """
    Run msoverview on a measurement set within a Singularity container and parse its summary.

    Parameters:
        binds: List of Paths to bind into the container (e.g. ["/data2", "/net/rijn9/"]).
        container: Path to the lofar Singularity .sif container.
        ms: Path to the input MeasurementSet.
        output_to_file: Path where the raw msoverview output will be written.
        get_intents: If True, will also retrieve the intents of the MeasurementSet (slower)

    Returns:
        A dict with keys:
          - 'spwID' (int)
          - 'nchan' (int)
          - 'Frame' (str)
          - 'Ch0MHz' (float)
          - 'ChanWid(kHz)' (float)
          - 'TotBW(kHz)' (float)
          - 'CtrFreq(MHz)' (float)
          - 'fields' (List[str])
          - 'field_intents' (dict[int, tuple[str, str]]) if get_intents is True
    """
    # build and run the command
    bind_str = ",".join(binds)
    cmd = [
        "singularity", "exec",
        "-B", bind_str,
        container,
        "msoverview", f"in={ms}"
    ]

    result = execute_command(cmd)
    output = result.stdout

    # write raw output to file
    with open(output_to_file, "w") as f:
        f.write(output)

    # prepare summary dict
    mssummary: dict[str, Any] = {}

    # parse spectral window line
    spw_re = re.compile(
        r"^\s*(\d+)\s+\S+\s+(\d+)\s+(\S+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    )
    for line in output.splitlines():
        m = spw_re.match(line)
        if not m:
            continue
        mssummary['spwID']        = int(m.group(1))
        mssummary['nchan']       = int(m.group(2))
        mssummary['Frame']        = m.group(3)
        mssummary['Ch0MHz']       = float(m.group(4))
        mssummary['ChanWid(kHz)'] = float(m.group(5))
        mssummary['TotBW(kHz)']   = float(m.group(6))
        mssummary['CtrFreq(MHz)'] = float(m.group(7))
        break

    # parse field names
    fields: list[str] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Fields:"):
            # skip the header line that follows
            for ln in lines[i+2:]:
                if not ln.strip() or ln.strip().startswith("Spectral Windows"):
                    break
                if re.match(r"^\s*\d+", ln):
                    parts = ln.split()
                    if len(parts) >= 3:
                        fields.append(parts[2])
            break

    mssummary['fields'] = fields

    if get_intents:
        # get field intents
        field_intents = get_field_intents(
            binds=binds,
            container=container,
            ms=ms,
            output_to_file=Path(output_to_file.parent, "field_intents.csv")
        )
        if field_intents is None:
            raise ValueError("Failed to retrieve field intents.")
        mssummary['field_intents'] = field_intents

    return mssummary

def load_field_intents(csv_path: Path) -> dict[int, tuple[str, str]]:
    """
    Load field intents from a CSV with headers:
      field_id, field_name, intent_string

    Returns a dict mapping:
      field_id (int) → (field_name, intent_string)
    """
    mapping: dict[int, tuple[str, str]] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = int(row["field_id"])
            mapping[fid] = (row["field_name"], row["intent_string"])
    return mapping

def get_field_intents(
        binds: list[Path],
        container: Path,
        ms: Path,
        output_to_file: Path
    ) -> dict[str, tuple[int, str]] | None:
    """
    Get the field intents from a MeasurementSet using casacore

    Parameters:
        binds: List of Paths to bind into the container (e.g. ["/data2", "/net/rijn9/"]).
        container: Path to the lofar Singularity .sif container.
        ms: Path to the input MeasurementSet.
        output_to_file: Path where the field intents will be written

    Returns:
        dict: keys are fieldnames strings; values are (field_id: int, intent: string) tuples.
    """

    # build and run the command
    bind_str = ",".join(binds)
    cmd = [
        "singularity", "exec",
        "-B", bind_str,
        container,
        "python", f"get_field_intents.py {ms} --outfile_csv {output_to_file}",
    ]

    execute_command(cmd)
    
    # read the mapping from csv
    field_intents = load_field_intents(output_to_file)

    return field_intents