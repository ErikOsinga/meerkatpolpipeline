"""Utilities related to measurement sets
Shamelessly adapted from the flint project.

"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def msoverview_summary(binds: List[Path], container: str, ms: str, output_to_file: str = "./file.txt") -> Dict[str, Any]:
    """
    Run msoverview on a measurement set within a Singularity container and parse its summary.

    Parameters:
        binds: List of directories to bind into the container (e.g. ["/data2", "/net/rijn9/"]).
        container: Path to the Singularity .sif container.
        ms: Path to the input MeasurementSet.
        output_to_file: Path where the raw msoverview output will be written.

    Returns:
        A dict with keys:
          - 'spwID' (int)
          - '#Chans' (int)
          - 'Frame' (str)
          - 'Ch0MHz' (float)
          - 'ChanWid(kHz)' (float)
          - 'TotBW(kHz)' (float)
          - 'CtrFreq(MHz)' (float)
          - 'fields' (List[str])
    """
    # build and run the command
    bind_str = ",".join(binds)
    cmd = [
        "singularity", "exec",
        "-B", bind_str,
        container,
        "msoverview", f"in={ms}"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=True)
    output = result.stdout

    # write raw output to file
    with open(output_to_file, "w") as f:
        f.write(output)

    # prepare summary dict
    mssummary: Dict[str, Any] = {}

    # parse spectral window line
    spw_re = re.compile(
        r"^\s*(\d+)\s+\S+\s+(\d+)\s+(\S+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    )
    for line in output.splitlines():
        m = spw_re.match(line)
        if not m:
            continue
        mssummary['spwID']        = int(m.group(1))
        mssummary['#Chans']       = int(m.group(2))
        mssummary['Frame']        = m.group(3)
        mssummary['Ch0MHz']       = float(m.group(4))
        mssummary['ChanWid(kHz)'] = float(m.group(5))
        mssummary['TotBW(kHz)']   = float(m.group(6))
        mssummary['CtrFreq(MHz)'] = float(m.group(7))
        break

    # parse field names
    fields: List[str] = []
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
    return mssummary
