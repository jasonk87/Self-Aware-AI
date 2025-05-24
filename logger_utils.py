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

def log(level: str, message: str, config_path="config.json"):
    """
    Logs a message to the console if the given log level is appropriate.

    The log level is determined by the 'logger_level' or 'logging_level'
    setting in the config file (defaults to "INFO").

    Args:
        level (str): The severity level of the message (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        message (str): The message to log.
        config_path (str, optional): Path to the configuration file. Defaults to "config.json".
    """
    if should_log(level, config_path):
        print(f"[{level.upper()}] {message}")

# Example usage:
# from logger_utils import log
# log("INFO", "This is an informational message.")
# log("DEBUG", "This is a debug message.")
