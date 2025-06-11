import logging
import logging.handlers
import json
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_FILE_NAME = "app.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

def setup_logging():
    # Read log level from config.json
    config_path = "config.json"
    log_level_str = DEFAULT_LOG_LEVEL
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                log_level_str = config_data.get("logger_level", DEFAULT_LOG_LEVEL).upper()
        except (json.JSONDecodeError, IOError) as e:
            # Use basic print here as logging might not be set up yet for this failure
            print(f"[ERROR] Failed to read logger_level from {config_path}: {e}. Using default {DEFAULT_LOG_LEVEL}.", file=sys.stderr)

    numeric_log_level = getattr(logging, log_level_str, logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG, handlers will filter

    # Clear existing handlers to avoid duplication if setup_logging is called multiple times
    # (though it should ideally be called once)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for console
    console_handler.setLevel(numeric_log_level)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Rotating)
    # Ensure meta directory exists for the log file
    meta_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta")
    if not os.path.exists(meta_dir):
        try:
            os.makedirs(meta_dir, exist_ok=True)
        except OSError as e:
            print(f"[ERROR] Could not create meta directory for logs: {meta_dir}. Error: {e}", file=sys.stderr)
            # Fallback to current directory if meta creation fails for some reason
            meta_dir = os.path.dirname(os.path.abspath(__file__))


    log_file_path = os.path.join(meta_dir, LOG_FILE_NAME)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Log all DEBUG level and above to the file
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(process)d - %(threadName)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file handler setup fails, log to console about it
        print(f"[ERROR] Failed to set up file logging to {log_file_path}: {e}", file=sys.stderr)


if __name__ == '__main__':
    # Example of how to use it:
    # Call this once at the beginning of your main application script.
    setup_logging()

    # Example usage in other modules:
    logger = logging.getLogger(__name__) # Get a logger specific to the current module
    logger.debug("This is a debug message from logging_config main.")
    logger.info("This is an info message from logging_config main.")
    logger.warning("This is a warning message from logging_config main.")
    logger.error("This is an error message from logging_config main.")
    logger.critical("This is a critical message from logging_config main.")

    # Example of a logger from a different "module"
    other_module_logger = logging.getLogger("another.module")
    other_module_logger.info("Info message from another.module")

    print(f"Logging configured. Check console and '{LOG_FILE_NAME}' in the 'meta' directory (or current if meta creation failed).")
