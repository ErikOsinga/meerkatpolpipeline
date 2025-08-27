"""

TEST CASA SCRIPT. Can be run, for example, with

casa -c test_casa_script_args.py \
  --calms ms_with_parangcorr-cal.ms \
  --scan-xcal "12,22"

"""

from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Full-polarisation calibration script for MeerKAT L-band'
    )
    # required arguments
    parser.add_argument(
        '--calms', required=True,
        help="Path to calibration MS (e.g. './A754-...-cal.ms')"
    )
    parser.add_argument(
        '--fcal', required=True,
        help="str denoting fcal (e.g. 'J0408-6545' or 'J1939-6342,J0408-6545')"
    )
    parser.add_argument(
        '--bpcal', required=True,
        help="str denoting bpcal (e.g. 'J0408-6545')"
    )
    parser.add_argument(
        '--gcal', required=True,
        help="str denoting gcal (e.g. 'J0408-6545')"
    )
    parser.add_argument(
        '--xcal', required=True,
        help="str denoting xcal (e.g. 'J1331+3030')"
    )

    # optional arguments
    parser.add_argument(
        '--targetms', default=None,
        help="Path to target MS (default: derived from --calms by replacing '-cal.ms' with '-target.ms')"
    )
    parser.add_argument(
        '--leakcal', default=None,
        help="Leakage calibrator. If None, defaults to fcal."
    )    
    parser.add_argument(
        '--refant', default='m024',
        help="Reference antenna (default: m024; m002 also works well)"
    )
    parser.add_argument(
        '--scan-xcal', default='',type=str,
        help="Which scans to use for polcal, should be casa-compliant stringList, e.g. '12,22' or just '12', or empty '' for all scans"
    )
    return parser.parse_args()


def main():
    args = parse_args()


    print("Running casa script with arguments:")
    print(args)

if __name__ == '__main__':
    main()
