# Self-Aware/notifier_module.py
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Callable, List # Added List and Callable

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_NOTIFIER_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (NotifierClassFallbackLog) {message}")

class Notifier:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 prompt_manager_instance: Optional[Any] = None): # PromptManager for triggering reflection

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_NOTIFIER_FALLBACK_LOGGER
        self.prompt_manager = prompt_manager_instance

        # Initialize paths from config or use defaults from your original script
        self.meta_dir_path = self.config.get("meta_dir", "meta")
        self.changelog_file_path = self.config.get("changelog_file", os.path.join(self.meta_dir_path, "changelog.json"))
        self.version_file_path = self.config.get("version_file", os.path.join(self.meta_dir_path, "version.json"))
        self.last_update_flag_file_path = self.config.get("last_update_flag_file", os.path.join(self.meta_dir_path, "last_update.json"))

        # Ensure meta directory exists when a Notifier instance is created
        self._ensure_meta_dir()

    def _ensure_meta_dir(self):
        """Ensures the meta directory (self.meta_dir_path) exists."""
        try:
            os.makedirs(self.meta_dir_path, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(Notifier class): Could not create meta directory {self.meta_dir_path}: {e}")

    def _log_internal_message(self, level: str, message: str): # Replaces global _log_notifier_message
        """Uses the instance logger, prefixing with class context."""
        self.logger(level, f"(Notifier class): {message}")


    def init_versioning(self):
        """Initializes changelog and version files if they don't exist."""
        self._ensure_meta_dir() # Ensures self.meta_dir_path is created
        if not os.path.exists(self.changelog_file_path):
            try:
                with open(self.changelog_file_path, 'w', encoding='utf-8') as f: json.dump([], f, indent=2)
                self._log_internal_message("INFO", f"Initialized changelog file: {self.changelog_file_path}")
            except IOError as e:
                self._log_internal_message("ERROR", f"Could not initialize changelog file {self.changelog_file_path}: {e}")

        if not os.path.exists(self.version_file_path):
            try:
                with open(self.version_file_path, 'w', encoding='utf-8') as f: json.dump({"version": "0.1.0"}, f, indent=2)
                self._log_internal_message("INFO", f"Initialized version file: {self.version_file_path}")
            except IOError as e:
                 self._log_internal_message("ERROR", f"Could not initialize version file {self.version_file_path}: {e}")


    def get_current_version(self) -> str:
        """Reads the current version from the version file."""
        try:
            with open(self.version_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)["version"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self._log_internal_message("WARNING", f"Could not read version file ({self.version_file_path}): {e}. Returning default '0.0.0'.")
            self._ensure_meta_dir() # Ensure meta dir exists before trying to write default
            try: # Attempt to create a default version file
                with open(self.version_file_path, 'w', encoding='utf-8') as f_default:
                    json.dump({"version": "0.0.0"}, f_default, indent=2)
                self._log_internal_message("INFO", f"Created default version file at {self.version_file_path}")
            except IOError as e_create:
                self._log_internal_message("ERROR", f"Could not create default version file at {self.version_file_path}: {e_create}")
            return "0.0.0"

    def bump_version(self) -> str:
        """Increments the patch version and saves it."""
        version_str = self.get_current_version()
        try:
            major, minor, patch = map(int, version_str.split("."))
            patch += 1
            new_version_str = f"{major}.{minor}.{patch}"
        except ValueError: # Handle non-SemVer format if encountered
            self._log_internal_message("WARNING", f"Version '{version_str}' not standard SemVer. Appending '.bump'.")
            new_version_str = f"{version_str}.bump" # Or handle error differently
            
        try:
            with open(self.version_file_path, 'w', encoding='utf-8') as f:
                json.dump({"version": new_version_str}, f, indent=2)
            self._log_internal_message("INFO", f"Version bumped to {new_version_str}")
            return new_version_str
        except IOError as e:
            self._log_internal_message("ERROR", f"Could not write new version to {self.version_file_path}: {e}")
            return version_str # Return old version on failure

    def log_update(self, summary: str, goals_completed_texts: Optional[List[str]] = None, approved_by: str = "user"):
        """Logs an update to the changelog and sets the last update flag."""
        self._ensure_meta_dir() # Ensure meta_dir exists
        new_version = self.bump_version() # Use instance method
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entry = {
            "version":   new_version,
            "timestamp": timestamp,
            "summary":   summary,
            "goals_completed_texts": goals_completed_texts if goals_completed_texts is not None else [],
            "approved_by": approved_by
        }

        changelog: List[Dict[str, Any]] = [] # Ensure type
        if os.path.exists(self.changelog_file_path):
            try:
                with open(self.changelog_file_path, "r", encoding="utf-8") as f_cl:
                    content = f_cl.read()
                    if content.strip():
                        loaded_cl = json.loads(content)
                        if isinstance(loaded_cl, list):
                            changelog = loaded_cl
                        else: # Reinitialize if not a list
                            self._log_internal_message("WARNING", f"Changelog file {self.changelog_file_path} not a list. Reinitializing.")
                            changelog = []
            except json.JSONDecodeError:
                self._log_internal_message("WARNING", f"Could not decode {self.changelog_file_path}. Reinitializing.")
                changelog = []
            except Exception as e_cl_load:
                self._log_internal_message("ERROR", f"Failed to load {self.changelog_file_path}: {e_cl_load}. Reinitializing.")
                changelog = []
        
        changelog.append(entry)

        try:
            with open(self.changelog_file_path, "w", encoding="utf-8") as f_cl_write:
                json.dump(changelog, f_cl_write, indent=2)
        except IOError as e_cl_write_io:
            self._log_internal_message("ERROR", f"Could not write to {self.changelog_file_path}: {e_cl_write_io}")

        try:
            with open(self.last_update_flag_file_path, "w", encoding="utf-8") as flag_f:
                json.dump(entry, flag_f, indent=2) # Save the latest entry as the flag
        except IOError as e_flag_io:
            self._log_internal_message("ERROR", f"Could not write update flag to {self.last_update_flag_file_path}: {e_flag_io}")

        # Announce via the instance logger (which points to ai_core's log_background_message)
        self.logger("INFO", f"(Notifier class): System Update Rolled Out -> v{new_version}: {summary}") # Use self.logger

        # Trigger reflection using the passed prompt_manager_instance
        if self.prompt_manager and hasattr(self.prompt_manager, 'auto_update_prompt'):
            try:
                self._log_internal_message("INFO", "Update logged. Triggering auto-reflection cycle via PromptManager instance.")
                self.prompt_manager.auto_update_prompt()
            except Exception as e_reflect_trigger:
                self._log_internal_message("ERROR", f"Error triggering reflection after update via PromptManager: {e_reflect_trigger}")
        elif self.prompt_manager: # Instance exists but lacks method
             self._log_internal_message("WARNING", "PromptManager instance available but auto_update_prompt method missing.")
        else: # No instance provided
            self._log_internal_message("WARNING", "PromptManager instance not available to Notifier. Cannot trigger reflection.")

# --- End of Notifier Class ---

if __name__ == "__main__":
    print("--- Testing Notifier Class (Standalone) ---")

    # Mock config and logger for testing
    test_notifier_config = {
        "meta_dir": "meta_notifier_test", # Test-specific meta directory
        # File names will default to standard if not in config
    }
    def main_test_logger_notifier(level, message): print(f"[{level.upper()}] (Notifier_Test) {message}")

    class MockPromptManagerForNotifier:
        def __init__(self, logger): self.logger = logger
        def auto_update_prompt(self):
            self.logger("INFO", "(MockPromptManagerForNotifier) auto_update_prompt called successfully!")

    # Clean up old test directory and files if they exist
    test_meta_dir = test_notifier_config["meta_dir"]
    if os.path.exists(test_meta_dir):
        import shutil
        shutil.rmtree(test_meta_dir)
    # os.makedirs(test_meta_dir, exist_ok=True) # _ensure_meta_dir in __init__ will create it

    mock_pm_instance_notifier = MockPromptManagerForNotifier(main_test_logger_notifier)
    
    notifier_instance_main = Notifier(
        config=test_notifier_config,
        logger_func=main_test_logger_notifier,
        prompt_manager_instance=mock_pm_instance_notifier
    )

    print("\n1. Initializing versioning...")
    notifier_instance_main.init_versioning() # Creates files in meta_notifier_test

    print(f"\n2. Current version: {notifier_instance_main.get_current_version()}")

    print("\n3. Logging an update...")
    notifier_instance_main.log_update("Test summary of changes.", ["Goal A completed", "Goal B done"], approved_by="TestUser")
    
    print(f"\n4. Version after update: {notifier_instance_main.get_current_version()}")

    print(f"\n5. Checking changelog ({notifier_instance_main.changelog_file_path}):")
    if os.path.exists(notifier_instance_main.changelog_file_path):
        with open(notifier_instance_main.changelog_file_path, "r") as f:
            changelog_data = json.load(f)
            if changelog_data and isinstance(changelog_data, list):
                print(json.dumps(changelog_data[-1], indent=2)) # Print last entry
            else:
                print("   Changelog empty or malformed.")
    else:
        print(f"   {notifier_instance_main.changelog_file_path} not found.")

    print(f"\n6. Checking last update flag ({notifier_instance_main.last_update_flag_file_path}):")
    if os.path.exists(notifier_instance_main.last_update_flag_file_path):
        with open(notifier_instance_main.last_update_flag_file_path, "r") as f_flag:
            flag_data = json.load(f_flag)
            print(json.dumps(flag_data, indent=2))
    else:
        print(f"   {notifier_instance_main.last_update_flag_file_path} not found.")

    print("\n--- Notifier Class Test Complete ---")