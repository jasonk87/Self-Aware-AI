import json
import os

# Logger levels in order of severity
LOGGER_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

# Cache config to avoid repeated disk reads
_config_cache = None

def get_logger_level(config_path="config.json"):
    global _config_cache
    if _config_cache is None:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    _config_cache = json.load(f)
                except Exception:
                    _config_cache = {}
        else:
            _config_cache = {}
    return (_config_cache.get("logger_level") or _config_cache.get("logging_level") or "INFO").upper()

def should_log(level: str, config_path="config.json") -> bool:
    """Return True if the message should be logged at the current config level."""
    level = level.upper()
    config_level = get_logger_level(config_path)
    try:
        return LOGGER_LEVELS.index(level) >= LOGGER_LEVELS.index(config_level)
    except ValueError:
        return True  # If unknown level, log it

# Example usage:
# if should_log("DEBUG"): print("debug message")
