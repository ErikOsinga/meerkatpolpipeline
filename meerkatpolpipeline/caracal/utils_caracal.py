from __future__ import annotations

from meerkatpolpipeline.configuration import Strategy


def determine_obsconf(strategy: Strategy) -> None:
    """Determine calibrators according to the strategy file
    either hardcoded as input by the user, or from the MS if "auto_determine_obsconf"
    """
    return "TODO"
