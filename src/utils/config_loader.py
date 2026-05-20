import os

import yaml


def load_config(config_path: str) -> dict:
    """
    Load YAML config file.

    Args:
        config_path (str): Path to yaml file

    Returns:
        dict: Config dictionary
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(cfg, save_path):
    with open(save_path, "w") as f:
        yaml.dump(cfg, f)
