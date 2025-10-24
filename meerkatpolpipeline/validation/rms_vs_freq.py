#!/usr/bin/env python3
"""
Compute and plot rms vs. frequency / channel from one or more images
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from meerkatpolpipeline.cube_imaging.combine_to_imagecube import find_channel_number
from meerkatpolpipeline.utils.utils import find_rms


def compute_rms_from_imagelist(imagelist: list[Path]) -> np.ndarray:
    """Compute RMS values from a list of images.

    Args:
        imagelist (list[Path]): List of image file paths.

    Returns:
        np.ndarray: Array of RMS values corresponding to each image.
    """
    rms_values = []
    for image_path in imagelist:
        data = fits.getdata(image_path)
        rms = find_rms(data)
        rms_values.append(rms)
    return np.array(rms_values)


def plot_rms_vs_channel_from_imlist(imagelist: list[Path], rmsvalues: np.ndarray, output_dir: Path, output_prefix: str) -> None:
    """
    Plot RMS vs channel number from a list of images and their RMS values.
    """

    channels = [find_channel_number(img.stem) for img in imagelist]
    channels = np.array(channels)

    fig, ax = plt.subplots()
    ax.plot(channels, rmsvalues, marker="o", linestyle="-", label="RMS")
    ax.set_xlabel("Channel number")
    ax.set_ylabel("RMS value [Jy/beam]")
    ax.grid(True)
    ax.set_yscale("log")
    ax.legend(loc="best")

    out_full = output_dir / f"{output_prefix}_rms_vs_channel.png"
    fig.savefig(out_full, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return