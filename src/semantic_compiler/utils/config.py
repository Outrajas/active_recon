import yaml
from pathlib import Path

def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
