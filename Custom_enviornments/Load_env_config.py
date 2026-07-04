from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("Config.yaml")


def load_env_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_text = config_file.read()

    try:
        import yaml

        config = yaml.safe_load(config_text) or {}
    except ImportError:
        config = _load_simple_yaml(config_text)

    config["OBS_PLAYER_SIZE"] = int(config["OBS_PLAYER_SIZE"])
    config["OBS_ENEMY_SIZE"] = int(config["OBS_ENEMY_SIZE"])
    config["MAX_ENEMIES"] = int(config["MAX_ENEMIES"])
    config["OBS_SIZE"] = int(config.get("OBS_SIZE", _calculate_obs_size(config)))
    config["MAX_EPISODE_STEPS"] = int(config["MAX_EPISODE_STEPS"])

    if config["OBS_SIZE"] != _calculate_obs_size(config):
        raise ValueError("OBS_SIZE must equal OBS_PLAYER_SIZE + OBS_ENEMY_SIZE * MAX_ENEMIES.")

    return config


def _calculate_obs_size(config):
    return config["OBS_PLAYER_SIZE"] + config["OBS_ENEMY_SIZE"] * config["MAX_ENEMIES"]


def _load_simple_yaml(config_text):
    config = {}
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_yaml_value(value.strip())
    return config


def _parse_yaml_value(value):
    value = value.strip('"').strip("'")
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
