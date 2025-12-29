"""Simple config loader for clemgame features."""

import json
from pathlib import Path


def load_config():
    """Load clemgame configuration from clemgame_config.json.

    Returns:
        Dict with configuration settings.
    """
    config_path = Path(__file__).parent / "agency_config.json"

    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)


    else:
        print("Configuration file not found.")
