"""Loading of strategy parameters from a JSON config file.

Keeping the tunable parameters of the detection strategies in a JSON file lets them be adjusted per device without changing code.

Resolution order for the config path:

1. an explicit path passed to :func:`load_config`,
2. ``config.json`` in the current working directory.

Strategies then fall back to their built-in
default, should a parameter be missing (either because the config file is missing or because the strategy section is incomplete).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULT_FILENAME = "config.json"

ENV_VAR = "PATTERN_FINDER_CONFIG"


def _resolve_path(path):
    if path is not None:
        return path
    return DEFAULT_FILENAME


def load_config(path: str = None) -> Dict[str, Any]:
    """Return the full config mapping, or an empty dict if no file is found.

    Raises
    ------
    ValueError
        If the file exists but does not contain valid JSON.
    """
    resolved = _resolve_path(path)
    if not os.path.isfile(resolved):
        return {}
    with open(resolved, "r") as handle:
        try:
            return json.load(handle)
        except ValueError as exc:
            raise ValueError(
                "Invalid JSON in config file {}: {}".format(resolved, exc)
            )


def strategy_params(name: str, path: str = None) -> Dict[str, Any]:
    """Return the parameter dict for one strategy (empty if not configured)."""
    return load_config(path).get(name, {})
