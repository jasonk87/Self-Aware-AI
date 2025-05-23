"""Fallback implementation of PromptManager for when main implementation can't be loaded."""
import os
import json
from typing import Dict, Any, Optional, Callable, Union

class PromptManager_FB:
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None,
                 mission_manager_instance: Optional[Any] = None):
        self.config = config if config else {}
        self.logger = logger_func if logger_func else lambda lvl, msg: print(f"[{lvl}] (FallbackPromptManager): {msg}")
        self.query_llm = query_llm_func if query_llm_func else lambda *a, **k: "[Error: No LLM query function provided to PromptManager_FB]"
        self.mission_manager = mission_manager_instance
        
        self.prompts_base_path = self.config.get("prompts_base_path", self.config.get("prompt_files_dir", "prompts"))
        
        # Basic fallback prompts that must be available
        self.operational_prompt_text = self._load_default_operational_prompt()

    def _load_default_operational_prompt(self) -> str:
        """Loads or creates a basic operational prompt."""
        try:
            os.makedirs(self.prompts_base_path, exist_ok=True)
            default_prompt_path = os.path.join(self.prompts_base_path, "default_system_prompt.txt")
            
            if os.path.exists(default_prompt_path):
                with open(default_prompt_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                default_content = (
                    "You are Weebo (Worker for Evolving and Executing Bot Operations), an AI assistant "
                    "currently operating in fallback mode. Your mission is to provide basic assistance "
                    "while core functionality is being restored. Maintain clear communication and focus "
                    "on robustness.\n\nInteract with users respectfully and clearly state any limitations "
                    "due to fallback mode."
                )
                with open(default_prompt_path, 'w', encoding='utf-8') as f:
                    f.write(default_content)
                return default_content
        except Exception as e:
            self.logger("ERROR", f"Error loading/creating default prompt: {e}")
            return "[Fallback System Prompt: Error loading default.]"

    def get_full_system_prompt(self) -> str:
        """Returns the complete system prompt including mission and operational instructions."""
        mission_statement = (
            self.mission_manager.get_mission_statement_for_prompt()
            if self.mission_manager and hasattr(self.mission_manager, 'get_mission_statement_for_prompt')
            else "[Mission Statement Unavailable in Fallback Mode]"
        )
        return f"{mission_statement}\n\n{self.operational_prompt_text}"

    def get_operational_prompt_text(self) -> str:
        """Returns the current operational instructions."""
        return self.operational_prompt_text

    def update_operational_prompt_text(self, new_text: str) -> None:
        """Updates the operational instructions portion of the system prompt."""
        if not new_text.strip():
            self.logger("WARNING", "Attempted to update operational prompt with empty text.")
            return
        
        self.operational_prompt_text = new_text
        try:
            os.makedirs(self.prompts_base_path, exist_ok=True)
            with open(os.path.join(self.prompts_base_path, "default_system_prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(new_text)
        except Exception as e:
            self.logger("ERROR", f"Failed to save updated operational prompt: {e}")

    def auto_update_prompt(self) -> bool:
        """Simulates the auto-update functionality."""
        self.logger("WARNING", "auto_update_prompt called in fallback mode - not implemented")
        return False

    def get_template_content(self, template_name: str, sub_component: Optional[str] = None) -> Optional[str]:
        """Gets the content of a prompt template."""
        template_path = os.path.join(self.prompts_base_path, f"{template_name}.txt")
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self.logger("ERROR", f"Error reading template {template_name}: {e}")
        
        if template_name == "planner_decide_action":
            return (
                "Based on the following context:\n{{user_input}}\n\n"
                "Decide the next action to take. Focus on basic operations "
                "and clearly state limitations of fallback mode.\n\n"
                "Respond in JSON format with 'action' and 'details' fields."
            )
        return None
        
    def get_filled_template(self, template_name: str, 
                          dynamic_content: Dict[str, Any], 
                          sub_component: Optional[str] = None) -> str:
        """Gets and fills a template with provided content."""
        template = self.get_template_content(template_name, sub_component)
        if not template:
            return f"[Error: Template '{template_name}' not found in fallback mode]"
        
        try:
            # Simple placeholder replacement
            result = template
            for key, value in dynamic_content.items():
                placeholder = "{{" + key + "}}"
                result = result.replace(placeholder, str(value))
            return result
        except Exception as e:
            self.logger("ERROR", f"Error filling template {template_name}: {e}")
            return f"[Error: Failed to fill template '{template_name}' in fallback mode]"
