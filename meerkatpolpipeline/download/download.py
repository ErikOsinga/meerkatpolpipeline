from __future__ import annotations

import os
from pathlib import Path

from meerkatpolpipeline.logging import logger
from meerkatpolpipeline.options import BaseOptions


class DownloadOptions(BaseOptions):
    """A basic class to handle download options. """
    
    enable: bool
    """enable this step? Default False"""
    targetfield: str | None = None
    """name of targetfield"""
    link: Path | None = None
    """Path to MeerKAT direct download link"""
    output_name: Path | None = None
    """Path to output name, e.g. target_uncalibrated.ms.tar.gz"""
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

def start_download(downloadoptions: DownloadOptions, working_dir: Path, test: bool = False) -> str:

    output_path = working_dir / downloadoptions.output_name.name

    cmd = f"wget --tries {downloadoptions.tries} --waitretry={downloadoptions.waitretry_seconds} -c -O {output_path} {downloadoptions.link}"

    # todo: capture command?
    if not test:
        logger.info("Starting download command:")
        logger.info(cmd)
        os.system(cmd)
    else:
        logger.info("Created download command:")
        logger.info(cmd)
        logger.info("Not executing as test=True")

    return cmd


