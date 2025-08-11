"""Utilities related to measurement sets
Shamelessly adapted from the flint project.

"""
from __future__ import annotations

import csv
import multiprocessing
import re
from pathlib import Path
from typing import Any

import astropy.units as units
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from casacore.tables import table
from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal import field_intents
from meerkatpolpipeline.sclient import singularity_wrapper
from meerkatpolpipeline.utils.utils import execute_command, make_utf8


def check_ms_timesteps(mslist: list[Path], ntimes_cutoff: int = 20) -> np.ndarray[bool]:
    """
    Facetselfcal will complain if MSes have less than 20 timesteps, use this function to check.
    
    returns array with True/False per ms in mslist, True if less timesteps in MS than ntimes_cutoff
    """
    times_below_cutoff = np.zeros_like(mslist, dtype='bool')
    for i, ms in enumerate(mslist):
        t = table(str(ms), ack=False)
        times = np.unique(t.getcol('TIME'))

        if len(times) <= ntimes_cutoff:
            times_below_cutoff[i] = True

        t.close()
    
    return times_below_cutoff


def determine_meerkat_band_from_ch0(Ch0MHz: float):
    """
    Based on the first channel frequency in the MS, return the MeerKAT band

    either 'L' or 'UHF'
    """
    if Ch0MHz < 855:
        return 'UHF'
    elif Ch0MHz < 1713:
        return 'L'
    raise ValueError(f"{Ch0MHz} not automatically mapped to a MeerKAT band.")


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
          - 'meerkat_band' (str), either "L" or "UHF" depending on the frequency range in the MS
          - 'field_intents' (dict[int, tuple[str, str]]) if get_intents is True
    """
    # build and run the command
    bind_str = ",".join([str(b) for b in binds])
    cmd = [
        "singularity", "exec",
        "-B", bind_str,
        str(container), # cant handle posixpath
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

    mssummary['meerkat_band'] = determine_meerkat_band_from_ch0(mssummary['Ch0MHz'])

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


def load_field_intents_csv(csv_path: Path) -> dict[int, tuple[str, str]]:
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

    # get mapping from casacore
    mapping = field_intents.map_fields_to_intents(ms)
    # write to csv
    field_intents.write_to_csv(mapping, output_to_file)
    # read the mapping from csv to return a dict
    field_intents_dict = load_field_intents_csv(output_to_file)

    return field_intents_dict


def find_closest_ddsol(h5, ms): # 
    """TAKEN FROM facetselfcal.py
    find closest direction in multidir h5 files to the phasecenter of the ms
    """
    t2 = table(ms + '::FIELD', ack=False)
    phasedir = t2.getcol('PHASE_DIR').squeeze()
    t2.close()
    c1 = SkyCoord(phasedir[0] * units.radian,  phasedir[1] * units.radian, frame='icrs')
    H5 = tables.open_file(h5)
    distance = 1e9 # just a big number
    for direction_id, direction in enumerate (H5.root.sol000.source[:]):
        ra, dec = direction[1]
        c2 = SkyCoord(ra * units.radian,  dec * units.radian, frame='icrs')
        angsep = c1.separation(c2).to(units.degree)
        print(direction[0], angsep.value, '[degree]')
        if angsep.value < distance:
            distance = angsep.value
            dirname = direction[0]
    H5.close()
    return dirname


def copy_corrdata_to_data_dp3(
        msin: Path | list[Path],
        msout_dir: Path,
        lofar_container: Path,
        msin_datacolumn: str = "CORRECTED_DATA"
    ) -> list[Path]:
    """
    copy the CORRECTED_DATA column of a (list of) MS to a new directory with the same MS.name but with only a DATA column 

    returns the copied MSes as list[Path]
    """

    if isinstance(msin, Path):
        msin = [msin]
    elif isinstance(msin, (list,np.ndarray)):
        assert isinstance(msin[0], Path), f"if {msin=} is a list, expect it to be of type list[Path]"
    else:
        raise TypeError(f"{msin=} expected to be of type list[Path] or Path")

    copied_mses = []
    for ms in msin:
        assert msout_dir != ms.parent, f"Output directory {msout_dir} should be different than parent directory of {msin=}"
        
        cmd = f"DP3 msin={msin} msin.datacolumn={msin_datacolumn} setps=[], msout={msout_dir / msin.name}"
        
        run_DP3_command(
            dp3_command=cmd,
            container=lofar_container,
            bind_dirs=[
                msin.parent,
                msout_dir,
            ],
        )

        copied_mses.append(msout_dir / msin.name)

    return copied_mses


@singularity_wrapper
def run_DP3_command(dp3_command: str, **kwargs) -> str:
    """Run a DP3 command using singularity wrapper

    Note that all arguments should be given as kwargs to not confuse singularity wrapper

    Args:
        dp3_command: a DP3 command as a string
        **kwargs: Additional keyword arguments that will be passed to the singularity_wrapper

    Returns:
        str: the command that was executed
    """
    logger = get_run_logger()

    logger.info(f"DP3 command {dp3_command}")

    return dp3_command


def fulljonesparmdb(h5):
    """TAKEN FROM facetselfcal.py
      Checks if a given h5parm has a fulljones solution table as sol000.

    Args:
        h5 (str): path to the h5parm.
    Returns:
        fulljones (bool): whether the sol000 contains fulljones solutions.
    """
    H=tables.open_file(h5) 
    try:
        pol_p = H.root.sol000.phase000.pol[:]
        pol_a = H.root.sol000.amplitude000.pol[:]
        if len(pol_p) == 4 and len(pol_a) == 4:
            fulljones = True
        else:
            fulljones = False
    except:  # noqa: E722
        fulljones = False
    H.close()
    return fulljones


def applycal(ms, inparmdblist, msincol='DATA',msoutcol='CORRECTED_DATA', \
             msout='.', dysco=True, modeldatacolumns=[], invert=True, direction=None, find_closestdir=False):
    """TAKEN FROM facetselfcal.py
    Apply an H5parm to a Measurement Set.

    Args:
        ms (str): path to a Measurement Set to apply solutions to.
        inparmdblist (list): list of H5parms to apply.
        msincol (str): input column to apply solutions to.
        msoutcol (str): output column to store corrected data in.
        msout (str): name of the output Measurement Set.
        dysco (bool): Dysco compress the output Measurement Set.
        modeldatacolumns (list): Model data columns list, if len(modeldatacolumns) > 1 we have a DDE solve
    Returns:
        None
    """
    if find_closestdir and direction is not None:
       print('Wrong input, you cannot use find_closestdir and set a direction')
       raise Exception('Wrong input, you cannot use find_closestdir and set a direction')
    
    
    
    if len(modeldatacolumns) > 1:      
        return  
    # to allow both a list or a single file (string)
    if not isinstance(inparmdblist, list):
        inparmdblist = [inparmdblist]

    cmd = 'DP3 numthreads=' + str(np.min([multiprocessing.cpu_count(),8])) + ' msin=' + ms
    cmd += ' msout=' + msout + ' '
    cmd += 'msin.datacolumn=' + msincol + ' '
    if msout == '.':
        cmd += 'msout.datacolumn=' + msoutcol + ' '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '
    count = 0
    for parmdb in inparmdblist:
        if find_closestdir:
           direction = make_utf8(find_closest_ddsol(parmdb,ms))
           print('Applying direction:', direction)
        if fulljonesparmdb(parmdb):
            cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
            cmd += 'ac' + str(count) + '.type=applycal '
            cmd += 'ac' + str(count) + '.correction=fulljones '
            cmd += 'ac' + str(count) + '.soltab=[amplitude000,phase000] '
            if not invert:
                cmd += 'ac' + str(count) + '.invert=False '  
            if direction is not None:
                if direction.startswith('MODEL_DATA'): # because then the direction name in the h5 contains bracket strings
                   cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                else:
                   cmd += 'ac' + str(count) + '.direction=' + direction + ' ' 
            count = count + 1
        else:  
            H=tables.open_file(parmdb) 

            if not invert: # so corrupt, rotation comes first in a rotation+diagonal apply
                try:
                    # phase = H.root.sol000.rotation000.val[:]  # note that rotation comes before amplitude&phase for a corrupt (important if the solve was a rotation+diagonal one)
                    cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                    cmd += 'ac' + str(count) + '.type=applycal '  
                    cmd += 'ac' + str(count) + '.correction=rotation000 '
                    cmd += 'ac' + str(count) + '.invert=False '                 
                    if direction is not None:
                        if direction.startswith('MODEL_DATA'): # because then the direction name in the h5 contains bracket strings
                           cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                        else:
                           cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                    count = count + 1        
                except:  # noqa: E722
                    pass

            try:
                # phase = H.root.sol000.phase000.val[:]
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '
                cmd += 'ac' + str(count) + '.correction=phase000 '
                if not invert:
                    cmd += 'ac' + str(count) + '.invert=False '                 
                if direction is not None:
                    if direction.startswith('MODEL_DATA'): # because then the direction name in the h5 contains bracket strings
                       cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                    else:
                       cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                count = count + 1    
            except:  # noqa: E722
                pass

            try:
                # phase = H.root.sol000.amplitude000.val[:]
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '  
                cmd += 'ac' + str(count) + '.correction=amplitude000 '
                if not invert:
                    cmd += 'ac' + str(count) + '.invert=False '                 
                if direction is not None:
                    if direction.startswith('MODEL_DATA'): # because then the direction name in the h5 contains bracket strings
                       cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                    else:
                       cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                count = count + 1        
            except:  # noqa: E722
                pass


            try:
                # phase = H.root.sol000.tec000.val[:]
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '
                cmd += 'ac' + str(count) + '.correction=tec000 '
                if not invert:
                    cmd += 'ac' + str(count) + '.invert=False '                 
                if direction is not None:
                    if direction.startswith('MODEL_DATA'): # because then the direction name in the h5 contains bracket strings
                       cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                    else:
                       cmd += 'ac' + str(count) + '.direction=' + direction + ' '                                     
                count = count + 1
            except:  # noqa: E722
                pass

            if invert: # so applycal, rotation comes last in a rotation+diagonal apply
                try:
                    # phase = H.root.sol000.rotation000.val[:]  # note that rotation comes after amplitude&phase for an applycal (important if the solve was a rotation+diagonal one)
                    cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                    cmd += 'ac' + str(count) + '.type=applycal '  
                    cmd += 'ac' + str(count) + '.correction=rotation000 '
                    cmd += 'ac' + str(count) + '.invert=True '   # by default True but set here as a reminder because order matters for rotation+diagonal in this DP3 step depending on invert=True/False              
                    if direction is not None:
                        if direction.startswith('MODEL_DATA'): # because then the direction name in the h5 contains bracket strings
                           cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                        else:
                           cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                    count = count + 1        
                except:  # noqa: E722
                    pass

            H.close()

    if count < 1:
        print('Something went wrong, cannot build the applycal command. H5 file is valid?')
        raise Exception('Something went wrong, cannot build the applycal command. H5 file is valid?')
    # build the steps command
    cmd += 'steps=['
    for i in range(count):
        cmd += 'ac' + str(i)
        if i < count - 1:  # to avoid last comma in the steps list
            cmd += ','
    cmd += ']'

    print('DP3 applycal:', cmd)
    execute_command(cmd,log=True)
    return
