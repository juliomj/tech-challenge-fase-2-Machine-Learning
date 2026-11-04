"""Load project configuration from a YAML file and the optional ``.env``."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

# src/purchase_propensity/utils/config.py -> raiz do repositorio, independente
# do diretorio de onde o comando foi executado.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
