#!/usr/bin/env python3
"""
Compute and plot rms vs. frequency / channel from one or more images
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from meerkatpolpipeline.check_nvss.target_vs_nvss import get_channel_frequencies
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
        try:
            rms = find_rms(data)
        except Exception as e:
            raise RuntimeError(f"Error computing RMS for image {image_path}: {e}")
        rms_values.append(rms)
    return np.array(rms_values)


def plot_rms_vs_channel_from_imlist(imagelist: list[Path], rmsvalues: np.ndarray, output_dir: Path, output_prefix: str) -> np.ndarray:
    """
    Plot RMS vs channel number from a list of images and their RMS values.
    Includes a secondary top axis showing frequency in MHz.

    Args:
        imagelist (list[Path]): List of image file paths.
        rmsvalues (np.ndarray): Array of RMS values corresponding to each image.
        output_dir (Path): Directory to save the output plot.
        output_prefix (str): Prefix for the output plot filename.
    Returns:
        np.ndarray: Array of frequencies in Hz corresponding to each channel.
    """
    channels = np.array([find_channel_number(img.stem) for img in imagelist])
    frequencies_Hz = get_channel_frequencies(imagelist)
    frequencies_MHz = frequencies_Hz / 1e6

    fig, ax = plt.subplots()
    ax.plot(channels, rmsvalues, marker="o", linestyle="-", label="RMS")
    ax.set_xlabel("Channel number")
    ax.set_ylabel("RMS value [Jy/beam]")
    ax.grid(True)
    ax.set_yscale("log")
    ax.legend(loc="best")

    # Define mapping functions between channel number and frequency (MHz)
    def channel_to_freq(ch):
        return np.interp(ch, channels, frequencies_MHz)

    def freq_to_channel(freq):
        return np.interp(freq, frequencies_MHz, channels)

    # Add secondary x-axis on top showing frequency in MHz
    ax_freq = ax.secondary_xaxis('top', functions=(channel_to_freq, freq_to_channel))
    ax_freq.set_xlabel("Frequency [MHz]")

    out_full = output_dir / f"{output_prefix}_rms_vs_channel.png"
    fig.savefig(out_full, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return frequencies_Hz