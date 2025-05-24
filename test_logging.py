# test_logging.py
import json
import os
# Option 1: Modify logger_utils directly (if worker can ensure it's re-imported or cache is handled)
import logger_utils 
from logger_utils import log, LOGGER_LEVELS, get_logger_level # Ensure get_logger_level is imported

def set_config_log_level(level: str):
    print(f"Attempting to set log level to: {level}")
    config_path = "config.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: config.json is not valid JSON. Initializing with empty config.")
                config = {} # Initialize to empty dict if JSON is invalid
    
    original_level = config.get("logger_level")
    config["logger_level"] = level
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    # Crucial: Invalidate logger_utils's cache
    logger_utils._config_cache = None 
    # Fetch the level via get_logger_level to confirm it's updated after cache invalidation
    current_effective_level = get_logger_level(config_path)
    print(f"Config logger_level set to {level}. Cache invalidated. Current effective level: {current_effective_level}")
    return original_level

def run_logs():
    print("Running log tests...")
    for level_to_log in LOGGER_LEVELS: # LOGGER_LEVELS is ["DEBUG", "INFO", "WARNING", "ERROR"]
        # The log function itself now handles the should_log check and formatting.
        log(level_to_log, f"This is a test message at {level_to_log} level.")

if __name__ == "__main__":
    original_config_level = None
    config_data = {}
    config_path = "config.json" # Define config_path for __main__

    # Ensure config.json exists, if not, create a default one for the test
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found. Creating a default for testing.")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"logger_level": "INFO"}, f, indent=4) # Default to INFO if not present

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError:
                print(f"ERROR: Could not decode {config_path}. Test cannot reliably store/restore original level.")
                config_data = {"logger_level": "INFO"} # Fallback
        original_config_level = config_data.get("logger_level", "INFO") # Default to INFO if key is missing
    else:
        # This case should not be reached if the above check creates the file
        print(f"ERROR: {config_path} still does not exist. Test cannot run properly.")
        original_config_level = "INFO" # Fallback

    try:
        test_levels_map = {
            "DEBUG": ["DEBUG", "INFO", "WARNING", "ERROR"],
            "INFO": ["INFO", "WARNING", "ERROR"],
            "WARNING": ["WARNING", "ERROR"],
            "ERROR": ["ERROR"],
            "CRITICAL": ["ERROR"], # CRITICAL is not in LOGGER_LEVELS, so it will default to True in should_log
                                 # and effectively behave like ERROR for filtering, but the tag will be CRITICAL.
                                 # For testing purposes, we expect ERROR level messages to show if level is CRITICAL.
                                 # Or, more accurately, if we log("CRITICAL", ...), it should appear if the config_level is ERROR or lower.
                                 # Let's adjust this to test what actually happens with an unknown level.
                                 # For now, assuming CRITICAL messages are always shown as per should_log's ValueError case.
                                 # This means CRITICAL will show up if config is DEBUG, INFO, WARNING, or ERROR.
        }

        # Add a test for a custom/unknown level to see if it defaults to logging
        # For this test, we'll set the config to INFO and then try to log a "SPECIAL" message
        # It should appear because unknown levels default to True in should_log
        
        for target_level_config, expected_outputs in test_levels_map.items():
            print(f"\n--- Setting config to: {target_level_config} ---")
            set_config_log_level(target_level_config) 
            
            effective_level = get_logger_level(config_path) # Use the function from logger_utils
            if effective_level != target_level_config:
                 # If target_level_config was "CRITICAL", effective_level might be different
                 # because "CRITICAL" is not in LOGGER_LEVELS.
                 # get_logger_level() defaults to "INFO" if the level in config is not in LOGGER_LEVELS.
                 # So, this check needs to be smarter or the test adjusted.
                 # For now, we'll print a warning if they don't match but proceed.
                 print(f"Warning: Config level set to {target_level_config}, but effective level for filtering is {effective_level} (due to predefined LOGGER_LEVELS).")
            else:
                print(f"Effective log level for filtering is correctly set to: {effective_level}")
            
            run_logs() # This will call log(level, message) for DEBUG, INFO, WARNING, ERROR
            
            # Test logging with the target_level_config itself, especially if it's "CRITICAL"
            if target_level_config not in LOGGER_LEVELS:
                 log(target_level_config, f"This is a test message at an unknown {target_level_config} level.")

            print(f"--- End of test for config level: {target_level_config} ---")
            # Expected output based on filtering by `effective_level`
            # If target_level_config is "CRITICAL", effective_level becomes "INFO" by default in get_logger_level
            # So, we need to calculate expected outputs based on the `effective_level` that `should_log` will use.
            
            current_filtering_level = effective_level
            if target_level_config not in LOGGER_LEVELS : # If we set an unknown level like CRITICAL
                # `get_logger_level` defaults to INFO if the level in config isn't in LOGGER_LEVELS
                # `should_log` with an unknown log level (e.g. log("SPECIAL", ...)) defaults to True
                # `should_log` with a known log level (e.g. log("DEBUG",...)) compares against the config_level (which defaulted to INFO)
                if target_level_config == "CRITICAL": # Special case for this test
                    print(f"Expected to see for config {target_level_config} (filters like {current_filtering_level}, plus CRITICAL): {expected_outputs + ['CRITICAL']}")
            else:
                 print(f"Expected to see for config {target_level_config}: {expected_outputs}")


        # Test for unknown log level against INFO config
        print(f"\n--- Setting config to: INFO for unknown log level test ---")
        set_config_log_level("INFO")
        effective_level = get_logger_level(config_path)
        print(f"Effective log level for filtering is: {effective_level}")
        log("SPECIAL", "This is a test message at SPECIAL level.")
        print(f"--- End of test for unknown log level ---")
        print(f"Expected to see: ['SPECIAL'] (as unknown levels default to log)")


    finally:
        if original_config_level is not None:
            print(f"\nRestoring original log level in config.json to: {original_config_level}")
            config_data["logger_level"] = original_config_level
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
            logger_utils._config_cache = None # Invalidate cache again after restoring
            print("Original log level restored in config.json.")
        else:
            print("Could not restore original log level (was not found initially or config was invalid).")

    print("\nTest script finished.")
