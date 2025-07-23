#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
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

def get_receptor_angle(vis):
    print(f"Obtaining receptor angle for {vis=}")
    tb = table(vis + '/FEED', readonly=True) # false for changing
    angles = tb.getcol('RECEPTOR_ANGLE')
    # tb.putcol('RECEPTOR_ANGLE', np.zeros_like(angles))
    tb.close()
    return angles

def set_receptor_angle(vis, angle_rad):
    print(f"Setting receptor angle to {angle_rad} radians for {vis=}")
    tb = table(vis + '/FEED', readonly=False) # false for changing
    angles = tb.getcol('RECEPTOR_ANGLE')
    tb.putcol('RECEPTOR_ANGLE', angle_rad*np.ones_like(angles))
    tb.close()
    return 

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

    parser.add_argument(
        '--test-already-done',
        dest='test_already_done',
        action='store_true',
        help="If set, check if RECEPTOR ANGLE has been changed from -90 to 0. If so, assume parang correction is already done."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    field_ids = get_field_indices(args.msfile)
    if not field_ids:
        print(f"No fields found in {args.msfile}")
        return
    
    if args.test_already_done:
        # meerKAT MSes come out with RECEPTOR_ANGLE=-90, which implies parang needs to be corrected
        print("Checking RECEPTOR ANGLE value in the measurement set. If zero, assuming parang correction is already done.")
        angles = get_receptor_angle(args.msfile)
        if (np.degrees(angles) == -90).all():
            print("RECEPTOR ANGLE is -90 for all fields, assuming parang correction is not done yet.")
        elif (np.degrees(angles) == 0).all():
            print("RECEPTOR ANGLE is 0 for all fields, assuming parang correction is already done.")
            return
        else:
            error = "RECEPTOR ANGLE has mixed values. Something went wrong? Please check the MS."
            print(error)
            print(f"Found receptor angles in degrees: {np.unique(np.degrees(angles))}")
            raise ValueError(error)


    for fid in field_ids:
        run_correct_parang(
            msfile=args.msfile,
            field_id=fid,
            container=args.container,
            binds=args.binds,
            inside_sing=args.running_inside_sing
        )
    print("All fields processed. Rotated visibilities will be in the CORRECTED_DATA column.")

    # if completed succesfully, set the RECEPTOR_ANGLE to 0
    print("Setting RECEPTOR_ANGLE to 0 for all fields")
    set_receptor_angle(
        vis=args.msfile,
        angle_rad=0.0
    )

if __name__ == '__main__':
    main()
