"""Fallback implementation of MissionManager for when main implementation can't be loaded."""
import os
import json
from typing import Dict, Any, Optional, Callable

class MissionManager_FB:
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, 
                 logger_func: Optional[Callable[[str, str], None]] = None):
        self.config = config if config else {}
        self.logger = logger_func if logger_func else lambda lvl, msg: print(f"[{lvl}] (FallbackMissionManager): {msg}")

        # Use config for paths, with sensible defaults
        self.meta_dir = self.config.get("meta_dir", "meta")
        self.mission_file_path = self.config.get("mission_file", os.path.join(self.meta_dir, "mission.json"))
        
        # Default mission can also be part of config or loaded from a default file if more complex
        self.default_mission_data = self.config.get("default_mission", {
            "identity_statement": "Weebo (Worker for Evolving and Executing Bot Operations)",
            "core_directive": "Operate in fallback mode while core functionality is restored.",
            "values": ["Effectiveness", "Adaptability", "Robustness"],
            "current_focus_areas": ["Restoring core functionality", "Maintaining basic operations"],
            "version": "1.0.0-fallback"
        })
        
        self._ensure_meta_dir()

    def _ensure_meta_dir(self):
        """Ensures the meta directory exists."""
        try:
            os.makedirs(self.meta_dir, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(MissionManager_FB) Could not create meta directory {self.meta_dir}: {e}")

    def load_mission(self) -> Dict[str, Any]:
        """
        Loads the mission statement from the configured mission file.
        If the file doesn't exist or is invalid, it creates/repairs it with default values.
        """
        self._ensure_meta_dir()
        try:
            if os.path.exists(self.mission_file_path):
                with open(self.mission_file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        return loaded_data
        except Exception as e:
            self.logger("ERROR", f"(MissionManager_FB) Error loading mission file: {e}")
        
        # If we get here, either file doesn't exist or had issues
        self.save_mission(self.default_mission_data)
        return self.default_mission_data.copy()

    def save_mission(self, mission_data: Dict[str, Any]) -> bool:
        """Saves the mission data to the configured file."""
        self._ensure_meta_dir()
        try:
            with open(self.mission_file_path, 'w', encoding='utf-8') as f:
                json.dump(mission_data, f, indent=2)
            return True
        except Exception as e:
            self.logger("ERROR", f"(MissionManager_FB) Error saving mission file: {e}")
            return False

    def get_mission_statement_for_prompt(self) -> str:
        """Returns a formatted string of the mission for use in prompts."""
        mission = self.load_mission()
        return (
            f"### MISSION & IDENTITY ###\n"
            f"{mission['identity_statement']}\n"
            f"Core Directive: {mission['core_directive']}\n"
            f"Values: {', '.join(mission['values'])}\n"
            f"Current Focus Areas: {', '.join(mission['current_focus_areas'])}\n"
            f"-------------------------"
        )

    def get_mission(self) -> Dict[str, Any]:
        """Alias for load_mission() for compatibility."""
        return self.load_mission()
