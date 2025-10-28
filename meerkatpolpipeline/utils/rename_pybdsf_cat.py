#!/usr/bin/env python3

"""
Script to rename specific columns in a PYBDSF source catalogue to be compliant with Polarimetry Pipeline
"""
from __future__ import annotations

import argparse
from pathlib import Path

from astropy.table import Table


def rename_columns(table: Table | str | Path) -> Table:
    """
    Rename columns according to a predefined mapping.

    Parameters
    ----------
    table : Table
        Input catalog table.

    Returns
    -------
    Table
        Table with renamed columns.
    """
    mapping = {
        "RA": "ra",
        "E_RA": "e_ra",
        "DEC": "dec",
        "E_DEC": "e_dec",
        "Total_flux": "col_flux_int",
        "E_total_flux": "col_e_flux_int",
        "Isl_rms": "col_rms_image",
    }

    if isinstance(table, (str, Path)):
        table = Table.read(str(table))

    for old_name, new_name in mapping.items():
        if old_name in table.colnames:
            table.rename_column(old_name, new_name)
    return table


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes 'input_cat' and 'output_cat'.
    """
    parser = argparse.ArgumentParser(
        description="Rename columns in a PYBDSF source catalogue."
    )
    parser.add_argument(
        "input_cat",
        help="Path to the input PYBDSF source catalogue (.fits)."
    )
    parser.add_argument(
        "output_cat",
        help="Path to save the modified catalogue (.fits)."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point: read, process, and write the catalogue.
    """
    args = parse_args()
    catalog = Table.read(args.input_cat)
    catalog = rename_columns(catalog)
    catalog.write(
        args.output_cat,
        format="fits",
        overwrite=True
    )
    print(f"Written to file {args.output_cat}")


if __name__ == "__main__":
    main()
