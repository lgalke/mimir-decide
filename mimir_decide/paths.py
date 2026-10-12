"""Where data and caches live.

The data root is the folder in the environment variable MIMIR_DECIDE_DATA, or ~/mimir-decide-data when the variable
is not set. Configs refer to it with the placeholder {DATA_ROOT}, for example `output_dir: "{DATA_ROOT}/mixture-v0"`.
On a machine whose home folder is not persistent, set the variable to a persistent folder (for example in ~/.bashrc).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ENV = "MIMIR_DECIDE_DATA"
DEFAULT = "~/mimir-decide-data"


def data_root() -> Path:
    return Path(os.environ.get(ENV) or DEFAULT).expanduser()


def resolve(value):
    """Replace {DATA_ROOT} in every string of a (nested) config value."""
    if isinstance(value, str):
        return value.replace("{DATA_ROOT}", str(data_root()))
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v) for v in value]
    return value


def load_config(path) -> dict:
    with open(path) as f:
        return resolve(yaml.safe_load(f))
