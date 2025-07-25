"""wrapper around yaml to make it deal with path objects

other scripts can then

from meerkatpolpipeline.utils.yaml import yaml
"""
from __future__ import annotations

from pathlib import PurePath

from ruamel.yaml import YAML


def _path_representer(dumper, data):
    """
    Represent any PurePath (PosixPath, WindowsPath, etc) as a plain YAML string.
    """
    # use the standard YAML str tag
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

# create a single YAML() instance that can be used throughout the project
# typ="rt" gives you RoundTripLoader & RoundTripDumper under the hood
yaml = YAML(typ="rt")

# register for all Path‑like classes
yaml.representer.add_multi_representer(PurePath, _path_representer)

# now expose the full API
__all__ = ["yaml"]

