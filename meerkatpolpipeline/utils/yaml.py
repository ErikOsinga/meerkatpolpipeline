"""wrapper around yaml to make it deal with path objects

other scripts can then

from meerkatpolpipeline.utils.yaml import yaml
"""
from __future__ import annotations

from pathlib import PurePath

import yaml as _yaml


def _path_representer(dumper, data):
    # represent Path() objects as their string form
    return dumper.represent_str(str(data))

# register once, globally, on import
_yaml.add_multi_representer(PurePath, _path_representer)

# now expose the real yaml API under the name `yaml`
yaml = _yaml

print(f"hi from the utils script, {yaml}")

__all__ = ["yaml"]
