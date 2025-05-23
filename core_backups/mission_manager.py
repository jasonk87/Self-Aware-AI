# Self-Aware/mission_manager.py
import os
import json
import traceback
from typing import Dict, Any, Optional, Callable

# Fallback logger for standalone testing or if no logger is provided to the class
_standalone_logger = lambda level, message: print(f"[{level.upper()}] (MissionManager_standalone_log) {message}")

class MissionManager:
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, 
                 logger_func: Optional[Callable[[str, str], None]] = None):
        self.config = config if config else {}
        self.logger = logger_func if logger_func else _standalone_logger

        # Use config for paths, with sensible defaults
        self.meta_dir = self.config.get("meta_dir", "meta")
        self.mission_file_path = self.config.get("mission_file", os.path.join(self.meta_dir, "mission.json"))
        
        # Default mission can also be part of config or loaded from a default file if more complex
        self.default_mission_data = self.config.get("default_mission", {
            "identity_statement": "I am Weebo (Worker for Evolving and Executing Bot Operations), an AI assistant. My core purpose is to continuously evolve into the world's most effective and helpful AI assistant.",
            "core_directive": "In every operational cycle, I must strive to learn, improve a tool or process, refine a thought, successfully complete a task, or strategically prioritize a new goal to advance my core purpose.",
            "values": ["Effectiveness", "Helpfulness", "Adaptability", "Continuous Learning", "Robustness", "Clarity", "Precision"],
            "current_focus_areas": ["Improving tool success rate", "Generating more impactful suggestions", "Efficient goal decomposition", "Strengthening self-correction capabilities"],
            "version": "1.0.0" # Version of the mission structure itself
        })
        
        self._ensure_meta_dir() # Call it during initialization

    def _ensure_meta_dir(self):
        """Ensures the meta directory exists."""
        try:
            os.makedirs(self.meta_dir, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(MissionManager class) Could not create meta directory {self.meta_dir}: {e}")

    def load_mission(self) -> Dict[str, Any]:
        """
        Loads the mission statement from the configured mission file.
        If the file doesn't exist or is invalid, it creates/repairs it with default values.
        """
        self._ensure_meta_dir() # Ensure dir exists before trying to load
        mission_data_to_return = self.default_mission_data.copy()

        if not os.path.exists(self.mission_file_path):
            self.logger("INFO", f"(MissionManager class): Mission file {self.mission_file_path} not found. Creating with default mission.")
            self.save_mission(self.default_mission_data) # Use instance method to save
            return self.default_mission_data.copy()
        
        try:
            with open(self.mission_file_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict):
                self.logger("WARNING", f"(MissionManager class): Mission file {self.mission_file_path} does not contain a valid JSON object. Reinitializing.")
                self.save_mission(self.default_mission_data)
                return self.default_mission_data.copy()

            updated_mission_data = self.default_mission_data.copy()
            updated_mission_data.update(loaded_data)

            if updated_mission_data.get("version") != self.default_mission_data.get("version"):
                self.logger("INFO", f"(MissionManager class): Mission structure version mismatch (file: {updated_mission_data.get('version')}, default: {self.default_mission_data.get('version')}). Ensuring all keys present.")
                if loaded_data != updated_mission_data:
                    self.save_mission(updated_mission_data)
            
            return updated_mission_data

        except json.JSONDecodeError as e_json:
            self.logger("ERROR", f"(MissionManager class): Error decoding JSON from {self.mission_file_path}: {e_json}. Reinitializing with default mission.")
            self.save_mission(self.default_mission_data)
            return self.default_mission_data.copy()
        except IOError as e_io:
            self.logger("ERROR", f"(MissionManager class): IOError when loading mission file {self.mission_file_path}: {e_io}. Returning default.")
            return self.default_mission_data.copy()
        except Exception as e_unexp:
            self.logger("CRITICAL", f"(MissionManager class): Unexpected error loading mission from {self.mission_file_path}: {e_unexp}\n{traceback.format_exc()}. Returning default.")
            return self.default_mission_data.copy()

    def save_mission(self, mission_data: Dict[str, Any]):
        """Saves the mission statement to the configured mission file."""
        self._ensure_meta_dir()
        try:
            with open(self.mission_file_path, "w", encoding="utf-8") as f:
                json.dump(mission_data, f, indent=2)
            self.logger("INFO", f"(MissionManager class): Mission statement saved to {self.mission_file_path}.")
        except IOError as e:
            self.logger("ERROR", f"(MissionManager class): Could not save mission to {self.mission_file_path}: {e}")
        except Exception as e_unexp:
            self.logger("CRITICAL", f"(MissionManager class): Unexpected error saving mission to {self.mission_file_path}: {e_unexp}\n{traceback.format_exc()}.")

    def get_mission_statement_for_prompt(self) -> str:
        """
        Constructs a string representation of the mission for use in system prompts.
        """
        mission = self.load_mission() # This ensures we always work with a valid mission object
        
        identity = mission.get('identity_statement', self.default_mission_data['identity_statement'])
        directive = mission.get('core_directive', self.default_mission_data['core_directive'])
        values_list = mission.get('values', self.default_mission_data['values'])
        focus_list = mission.get('current_focus_areas', self.default_mission_data['current_focus_areas'])

        values_str = ", ".join(values_list) if isinstance(values_list, list) else "Effectiveness, Helpfulness"
        focus_str = ", ".join(focus_list) if isinstance(focus_list, list) else "General Improvement"

        return (
            f"### MISSION & IDENTITY ###\n"
            f"{identity}\n"
            f"Core Directive: {directive}\n"
            f"Guiding Values: {values_str}.\n"
            f"Current Areas of Focus for Self-Improvement: {focus_str}.\n"
            f"Always consider your mission and these focus areas in your planning, execution, and reflection.\n"
            f"-------------------------"
        )

    def get_mission(self) -> Dict[str, Any]: # Added for compatibility if AICore tries to call .get_mission()
        """Returns the current mission data dictionary."""
        return self.load_mission()

if __name__ == "__main__":
    print("--- Testing MissionManager Class (Standalone) ---")
    
    # Mock config for testing
    test_config = {
        "meta_dir": "meta_mm_test", # Use a test-specific meta directory
        "mission_file": os.path.join("meta_mm_test", "mission_test.json"),
        "default_mission": { # Example of overriding default mission via config
            "identity_statement": "Test AI Identity",
            "core_directive": "Test Core Directive for MissionManager class.",
            "values": ["TestValue1", "TestValue2"],
            "current_focus_areas": ["Testing MissionManager", "Standalone Execution"],
            "version": "1.0.1-test"
        }
    }
    
    # Simple print logger for testing
    def main_test_logger(level, message):
        print(f"[{level}] (MM_Test) {message}")

    # Clean up old test file if it exists
    test_mission_file_path = test_config["mission_file"]
    if os.path.exists(test_mission_file_path):
        try:
            os.remove(test_mission_file_path)
            main_test_logger("INFO", f"Removed old test mission file: {test_mission_file_path}")
        except OSError as e:
            main_test_logger("WARNING", f"Could not remove old test file {test_mission_file_path}: {e}")
    if os.path.exists(test_config["meta_dir"]) and not os.listdir(test_config["meta_dir"]): # remove meta if empty
        try: os.rmdir(test_config["meta_dir"])
        except: pass


    mm_instance = MissionManager(config=test_config, logger_func=main_test_logger)

    print("\n1. Initial load_mission() (should create default from config):")
    initial_mission = mm_instance.load_mission()
    print(f"   Loaded mission (first time):\n{json.dumps(initial_mission, indent=2)}")
    if initial_mission["identity_statement"] != "Test AI Identity":
        print("   ERROR: Initial mission did not match config default.")

    print("\n2. get_mission_statement_for_prompt():")
    prompt_statement = mm_instance.get_mission_statement_for_prompt()
    print(f"   Prompt statement:\n{prompt_statement}")
    if "Test AI Identity" not in prompt_statement:
        print("   ERROR: Prompt statement missing correct identity.")

    print("\n3. Modifying and saving mission:")
    current_mission = mm_instance.load_mission()
    current_mission["current_focus_areas"] = ["New Focus Test 1", "New Focus Test 2"]
    current_mission["values"].append("NewTestValue")
    mm_instance.save_mission(current_mission)

    reloaded_mission = mm_instance.load_mission()
    print(f"   Reloaded mission after save:\n{json.dumps(reloaded_mission, indent=2)}")
    if "New Focus Test 1" not in reloaded_mission["current_focus_areas"] or \
       "NewTestValue" not in reloaded_mission["values"]:
        print("   ERROR: Mission modification not saved or loaded correctly.")
    else:
        print("   Mission modification successful.")

    print("\n4. Testing resilience (deleting file and reloading):")
    if os.path.exists(test_mission_file_path):
        os.remove(test_mission_file_path)
        print(f"   Deleted test mission file: {test_mission_file_path}")
    
    resilient_mission = mm_instance.load_mission() # Should recreate from config's default_mission
    print(f"   Mission loaded after deletion:\n{json.dumps(resilient_mission, indent=2)}")
    if resilient_mission["identity_statement"] == "Test AI Identity":
        print("   Resilience test PASSED: Mission recreated from config default.")
    else:
        print("   Resilience test FAILED: Default mission not correctly restored from config.")

    print("\n--- MissionManager Class Test Complete ---")
    print(f"Review test artifacts in '{test_config['meta_dir']}' directory.")