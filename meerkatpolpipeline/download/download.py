from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils.utils import execute_command

# from meerkatpolpipeline.logging import logger

class DownloadOptions(BaseOptions):
    """A basic class to handle download options. """
    
    enable: bool
    """enable this step? Default False"""
    targetfield: str | None = None
    """name of targetfield"""
    link: str | None = None
    """string containing to MeerKAT direct download url"""
    output_name: Path | None = None
    """Path to output name of download, e.g. target_uncalibrated.ms.tar.gz"""
    ms_name: Path | None = None
    """Path to output name of untarred ms, e.g. target_uncalibrated.ms"""
    tries: int | str = "inf"
    """amount of tries in wget call, integer or 'inf' for infinite tries."""
    waitretry_seconds: int = 2
    """amount of seconds to wait between downloads."""
    clip_assumed_nchan: int = 4096
    """double-check whether this is the number of channels before any clipping"""
    clip_chan_start: int = 163
    """clip channels from MS before this number"""
    clip_chan_end: int = 3885
    """clip channels from MS after this number"""

def download_and_extract(downloadoptions: DownloadOptions, working_dir: Path, test: bool = False) -> Path:
    """
    Download MS from MeerKAT direct download link and extract it if necessary.
    If the MS is already downloaded, it will return the path to the MS.
    Args:
        downloadoptions (DownloadOptions): Options for the download.
        working_dir (Path): Directory to work in.
        test (bool, optional): If True, will not execute the command, but only log it. Defaults to False.
    Returns:
        ms_path (Path): path to measurementset
        or 
        cmd (str): command that would be executed, if test=True
    """
    logger = get_run_logger()

    output_path = working_dir / downloadoptions['output_name']
    ms_path = working_dir / downloadoptions['ms_name']

    # Check if ms_path is already an existing directory
    if ms_path.exists():
        logger.info(f"The output MS '{ms_path}' already exists. Assuming MS is already downloaded")
        return ms_path
    
    # check if we're downloading a .tar.gz file or .ms file directly
    link_is_tar = False
    if ".tar.gz" in downloadoptions['link']:
        logger.info("Downloading a .tar.gz file, will extract it after download.")
        link_is_tar = True

    if output_path.exists():
        logger.info(f"The output path '{output_path}' already exists. Assuming MS is already downloaded")

    else: # download MS from link
        cmd = f"wget --tries {downloadoptions['tries']} --waitretry={downloadoptions['waitretry_seconds']} -c -O {output_path} {downloadoptions['link']}"
        
        execute_command(cmd, test=test)

        if test:
            return cmd
    
    if link_is_tar:
        logger.info(f"Extracting {output_path} to {ms_path}")
        # extract the tar.gz file
        cmd = f"tar -xzf {output_path} -C {working_dir}"
        execute_command(cmd)
        # grab the MS name from the download link
        unpacked_ms_name = working_dir / downloadoptions['link'].name.split('.tar.gz')[0]
        # check whether we can find it
        assert unpacked_ms_name.exists(), f"Unpacked MS {unpacked_ms_name} does not exist, please check the download link and the tar.gz file."
        # rename the MS to the expected name
        unpacked_ms_name.rename(ms_path)

    return ms_path


