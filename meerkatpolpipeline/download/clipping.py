from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.casa import casa_command


def get_clip_channels_default(nchan):
    """return the channel clipping recommended by SARAO if wasnt done during download"""
    if nchan == 4096:
        spw="0:163~3885"
    else:
        spw=""
    return spw

def construct_spw_string(
    clip_chan_start: int = 163,
    clip_chan_end: int = 3885,
) -> str:
    """
    Construct a string for the spw parameter in mstransform.
    Assuming MEERKAT MSes which have one spectral window only. 

    by default, it will return "0:163~3885".

    if clip_chan_start and clip_chan_end are both None, it returns an empty string
    """
    if clip_chan_end is None and clip_chan_start is None:
        return ""
    return f"0:{clip_chan_start}~{clip_chan_end}"

def copy_and_clip_ms(
        ms_path: Path,
        ms_summary: dict,
        clip_assumed_nchan: int = 4096,
        clip_chan_start: int = 163,
        clip_chan_end: int = 3885,
        output_ms: Path | None = None,
        casa_container: Path | None = None,
        bind_dirs: list[Path] | None = None,
    ) -> Path:
    """
    copy measurement set and clip channels from it using casa task mstransform.

    Args:
        ms_path (Path): Path to the input measurement set.
        ms_summary (dict): Summary of the measurement set, containing 'nchan'.
        clip_assumed_nchan (int): The number of channels assumed for clipping.
        clip_chan_start (int): Start channel for clipping.
        clip_chan_end (int): End channel for clipping.
        output_ms (Path | None): Path for the output clipped measurement set. If None, it will be created with "_clipped" suffix.
        casa_container (Path | None): Path to the CASA container.
        bind_dirs (list[Path] | None): Directories to bind into the container.
    Returns:
        output_ms (Path): Path to the output clipped measurement set.
    """
    logger = get_run_logger()

    if output_ms is None:
        output_ms = ms_path.with_name(f"{ms_path.stem}_clipped.ms")
    
    # check if the MS has the expected number of channels
    if ms_summary['nchan'] != clip_assumed_nchan:
        raise ValueError(f"MS {ms_path} has {ms_summary['nchan']} channels, expected {clip_assumed_nchan} channels.")

    spw = construct_spw_string(
        clip_chan_start=clip_chan_start,
        clip_chan_end=clip_chan_end,
    )

    # clip MS using casa task mstransform
    # this function is wrapped in a decorator that runs it inside a container
    casa_command(
        task="mstransform",
        vis=ms_path,
        outputvis=output_ms,
        datacolumn="corrected",
        spw=spw,
        keepflags=True,
        usewtspectrum=False,
        container=casa_container,
        bind_dirs=bind_dirs,
    )

    logger.info(f"Clipped MS saved to {output_ms} with only a DATA column")

    return output_ms

