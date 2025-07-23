#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

from casacore.tables import table

"""
Script to call Benjamin Hugo's correct_parang.py script for all fields in an MS.

See also:

https://github.com/bennahugo/LunaticPolarimetry/blob/master/correct_parang.py
"""

def get_field_indices(msfile):
    """
    Return a list of integer field IDs for the measurement set.
    """
    field_table = os.path.join(msfile, 'FIELD')
    t = table(field_table, ack=False)
    try:
        count = t.nrows()
    finally:
        t.close()
    return list(range(count))


def run_correct_parang(msfile, field_id, container, binds, inside_sing):
    """
    Construct and run the correct_parang.py command using the integer field ID.
    """
    script = os.path.join(os.path.dirname(__file__), 'correct_parang.py')

    base_cmd = [
        sys.executable,       # e.g. /usr/bin/python3
        script,               # full path to correct_parang.py
        '-f', str(field_id),
        '--noparang',
        '--applyantidiag',
        msfile
    ]
    if inside_sing:
        cmd = base_cmd
    else:
        binds_str = ",".join(binds)
        cmd = [
            'singularity', 'exec',
            '-B', binds_str,
            container
        ] + base_cmd

    print(f"Running for field ID {field_id}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, shell=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run correct_parang.py for each field ID in an MS, optionally inside Singularity"
    )
    parser.add_argument(
        'msfile',
        help="Path to the measurement set directory"
    )
    parser.add_argument(
        '--container',
        default='/net/achterrijn/data1/sweijen/software/containers/lofar_sksp_rijnX.sif',
        help="Path to the Singularity container (default: %(default)s)"
    )
    parser.add_argument(
        '--binds',
        nargs='+',
        default=[
            '/tmp',
            '/dev/shm',
            '/data1',
            '/data2',
            '/net/lofar4',
            '/net/lofar7',
            '/net/voorrijn/',
            '/net/rijn9/'
        ],
        help="Directories to bind into the container (space-separated)"
    )
    parser.add_argument(
        '--running-inside-sing',
        dest='running_inside_sing',
        action='store_true',
        help="If set, skip Singularity exec and run inside the container directly"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    field_ids = get_field_indices(args.msfile)
    if not field_ids:
        print(f"No fields found in {args.msfile}")
        return
    for fid in field_ids:
        run_correct_parang(
            msfile=args.msfile,
            field_id=fid,
            container=args.container,
            binds=args.binds,
            inside_sing=args.running_inside_sing
        )

    #  dp3 doesnt like multiple fields
    # print("Copying corrected data column to new MS")
    # run_dp3copy(
    #     msfile=args.msfile,
    #     container=args.container,
    #     binds=args.binds,
    #     inside_sing=args.running_inside_sing
    # )

if __name__ == '__main__':
    main()
