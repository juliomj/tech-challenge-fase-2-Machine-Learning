"""Load project configuration from a YAML file."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path.cwd()
config_path = PROJECT_ROOT / "configs" / "config.yaml"
with config_path.open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
