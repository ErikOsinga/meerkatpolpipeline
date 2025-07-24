from __future__ import annotations

from pathlib import Path

from prefect.logging import get_run_logger

from meerkatpolpipeline.options import BaseOptions
from meerkatpolpipeline.utils import add_timestamp_to_path


class CasaCrosscalOptions(BaseOptions):
    """A basic class to handle casa crosscal options for meerkatpolpipeline. """
    
    enable: bool
    """enable this step? Default False"""
    targetfield: str | None = None
    """name of targetfield"""
    msdir: Path | None = None
    """Path to the uncalibrated MS"""
    auto_determine_obsconf: bool = False
    """Automatically determine which calibrators are which and which is the target? Default False"""
    obsconf_target: str | None = None
    """Target"""
    obsconf_gcal: str | None = None
    """Gaincal"""
    obsconf_xcal: str | None = None
    """Pol cal"""
    obsconf_bpcal: str | None = None
    """BP calibrator, should be unpolarised"""
    obsconf_fcal: str | None = None
    """Flux calibrator, should be unpolarised"""
    obsconf_refant: str = "m024"
    """reference antenna"""
    test: bool = False
    """create the casa crosscal script but dont run it, for testing purposes only"""