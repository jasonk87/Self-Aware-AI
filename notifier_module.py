# Self-Aware/notifier_module.py
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Callable, List, Union
from logger_utils import should_log # Added import

def get_current_version() -> str:
    """Get the current version from version.json"""
    try:
        meta_dir = os.path.abspath("meta")
        version_file = os.path.join(meta_dir, "version.json")
        if os.path.exists(version_file):
            with open(version_file, "r", encoding="utf-8") as f:
                ver_data = json.load(f)
                return ver_data.get("version", "0.0.0")
    except Exception as e:
        if should_log("ERROR"): print(f"[ERROR] Failed to read version: {e}")
    return "0.0.0"

def log_update(summary: str, goals: Union[List[Dict[str, Any]], None] = None, approved_by: Optional[str] = None) -> bool:
    """Module-level wrapper for Notifier.log_update."""
    try:
        notifier = Notifier()
        return notifier.log_update(summary, goals, approved_by)
    except Exception as e:
        if should_log("ERROR"): print(f"[ERROR] Module-level log_update failed: {str(e)}")
        return False

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_NOTIFIER_FALLBACK_LOGGER = lambda level, message: (print(f"[{level.upper()}] (NotifierClassFallbackLog) {message}") if should_log(level.upper()) else None)

class Notifier:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None,
                 prompt_manager_instance: Optional[Any] = None):

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_NOTIFIER_FALLBACK_LOGGER
        self.query_llm = query_llm_func if query_llm_func else lambda *args, **kwargs: "[Error: LLM query not available]"
        self.prompt_manager = prompt_manager_instance

        # Initialize paths from config or use defaults
        self.meta_dir_path = os.path.abspath(self.config.get("meta_dir", "meta"))
        self.changelog_file_path = os.path.join(self.meta_dir_path, "changelog.json")
        self.version_file_path = os.path.join(self.meta_dir_path, "version.json")
        self.last_update_flag_file_path = os.path.join(self.meta_dir_path, "last_update.json")

        # Ensure meta directory exists when a Notifier instance is created
        os.makedirs(self.meta_dir_path, exist_ok=True)

    def log_update(self, summary: str, goals: Union[List[Dict[str, Any]], None] = None, approved_by: Optional[str] = None) -> bool:
        """Log a system update/change with optional goals affected."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "summary": summary,
                "approved_by": approved_by or "system",
                "related_goals": goals or []
            }

            # Load existing changelog
            changelog = []
            if os.path.exists(self.changelog_file_path):
                try:
                    with open(self.changelog_file_path, 'r', encoding='utf-8') as f:
                        changelog = json.load(f)
                except json.JSONDecodeError:
                    self.logger("WARNING", "(Notifier) Changelog file corrupted, creating new")

            # Add new entry and save
            changelog.append(entry)
            os.makedirs(os.path.dirname(self.changelog_file_path), exist_ok=True)
            with open(self.changelog_file_path, 'w', encoding='utf-8') as f:
                json.dump(changelog, f, indent=2)

            # Also update last_update.json for quick reference
            with open(self.last_update_flag_file_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f, indent=2)

            self.logger("INFO", f"(Notifier) Logged update: {summary[:100]}...")
            return True

        except Exception as e:
            self.logger("ERROR", f"(Notifier) Failed to log update: {str(e)}")
            return False

    def get_last_update(self) -> Optional[Dict[str, Any]]:
        """Get the most recent update entry."""
        try:
            if os.path.exists(self.last_update_flag_file_path):
                with open(self.last_update_flag_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger("ERROR", f"(Notifier) Failed to read last update: {str(e)}")
        return None

# --- End of Notifier Class ---

if __name__ == "__main__":
    if should_log("INFO"): print("--- Testing Notifier Class (Standalone) ---")

    # Mock config and logger for testing
    test_notifier_config = {
        "meta_dir": "meta_notifier_test", # Test-specific meta directory
        # File names will default to standard if not in config
    }
    def main_test_logger_notifier(level, message):
        if should_log(level.upper()): print(f"[{level.upper()}] (Notifier_Test) {message}")

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

    if should_log("INFO"): print("\n1. Initializing versioning...")
    notifier_instance_main.init_versioning() # Creates files in meta_notifier_test

    if should_log("INFO"): print(f"\n2. Current version: {notifier_instance_main.get_current_version()}")

    if should_log("INFO"): print("\n3. Logging an update...")
    notifier_instance_main.log_update("Test summary of changes.", ["Goal A completed", "Goal B done"], approved_by="TestUser")
    
    if should_log("INFO"): print(f"\n4. Version after update: {notifier_instance_main.get_current_version()}")

    if should_log("INFO"): print(f"\n5. Checking changelog ({notifier_instance_main.changelog_file_path}):")
    if os.path.exists(notifier_instance_main.changelog_file_path):
        with open(notifier_instance_main.changelog_file_path, "r") as f:
            changelog_data = json.load(f)
            if changelog_data and isinstance(changelog_data, list):
                if should_log("DEBUG"): print(json.dumps(changelog_data[-1], indent=2)) # Print last entry
            else:
                if should_log("WARNING"): print("   Changelog empty or malformed.")
    else:
        if should_log("WARNING"): print(f"   {notifier_instance_main.changelog_file_path} not found.")

    if should_log("INFO"): print(f"\n6. Checking last update flag ({notifier_instance_main.last_update_flag_file_path}):")
    if os.path.exists(notifier_instance_main.last_update_flag_file_path):
        with open(notifier_instance_main.last_update_flag_file_path, "r") as f_flag:
            flag_data = json.load(f_flag)
            if should_log("DEBUG"): print(json.dumps(flag_data, indent=2))
    else:
        if should_log("WARNING"): print(f"   {notifier_instance_main.last_update_flag_file_path} not found.")

    if should_log("INFO"): print("\n--- Notifier Class Test Complete ---")