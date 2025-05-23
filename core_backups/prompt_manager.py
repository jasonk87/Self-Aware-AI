# Self-Aware/prompt_manager.py
import os
import sys
import json
import time
import traceback # For detailed error logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union # Added Callable

# --- Fallback definitions for when ai_core or mission_manager are not available during direct module testing ---
# These fallbacks are primarily for the __main__ block or if the module is run standalone.
# When instantiated by AICore, proper instances/functions should be passed.

_ai_core_fallback_logger = lambda level, message: print(f"[{level.upper()}] (prompt_manager_standalone_log) {message}")
_query_llm_internal_fallback = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=300: \
    f"[Error: Fallback query_llm_internal called. ai_core.query_llm_internal not available to PromptManager instance. Prompt: {prompt_text[:100]}...]"

_mission_manager_get_mission_statement_fallback = lambda: "### MISSION & IDENTITY ###\n[CRITICAL ERROR: Mission Manager instance not available to PromptManager. Operating with basic instructions only.]\n-------------------------"
_mission_manager_load_mission_fallback = lambda: {"current_focus_areas": ["Fallback: General system stability and task completion."]}


class PromptManager:
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, 
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None, # For auto_update_prompt
                 mission_manager_instance: Optional[Any] = None): # Instance of MissionManager class

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _ai_core_fallback_logger
        self.query_llm = query_llm_func if query_llm_func else _query_llm_internal_fallback
        self.mission_manager = mission_manager_instance

        # Configuration for paths and defaults
        self.meta_dir = self.config.get("meta_dir", "meta")
        self.prompts_base_path = self.config.get("prompts_base_path", "prompts") # For structured prompts

        # Specific file paths used by PromptManager
        self.system_prompt_operational_part_file = os.path.join(self.meta_dir, "system_prompt.txt")
        self.goal_file_path = os.path.join(self.meta_dir, self.config.get("goal_file", "goals.json"))
        self.suggestion_file_path = os.path.join(self.meta_dir, self.config.get("suggestion_file", "suggestions.json"))
        self.changelog_file_path = os.path.join(self.meta_dir, self.config.get("changelog_file", "changelog.json"))
        self.tool_registry_file_path = os.path.join(self.meta_dir, self.config.get("tool_registry_file", "tool_registry.json"))
        self.last_reflection_file_path = os.path.join(self.meta_dir, self.config.get("last_reflection_file", "last_reflection.json"))
        self.conversation_review_log_file_path = os.path.join(self.meta_dir, self.config.get("conversation_review_log_file", "conversation_review_log.json"))

        self.default_base_operational_prompt = self.config.get("default_base_operational_prompt", 
            ("You are a highly capable AI developer assistant. Your primary functions are: "
             "1. Decompose complex user goals into actionable subtasks, considering thread context (THREAD_ID). "
             "2. Select existing tools or generate new Python tools (must use argparse, have a main(), no input()) to execute subtasks. "
             "3. Analyze errors from tool execution/generation and formulate corrective sub-goals, referencing failure categories and THREAD_ID. "
             "4. Continuously reflect on performance (using goal evaluations, tool success rates, mission focus, conversation patterns) to improve strategies and self-correct these operational instructions. "
             "5. Generate relevant suggestions for new goals based on project context, mission focus, recent activity/evaluations, and conversation insights. "
             "Prioritize clarity, efficiency, robust error handling, precise execution, and adherence to your mission and values in all operations. "
             "Strictly use THREAD_ID for contextual understanding and linking related tasks.")
        )
        self._ensure_meta_dir()

    def _ensure_meta_dir(self):
        try:
            os.makedirs(self.meta_dir, exist_ok=True)
            os.makedirs(self.prompts_base_path, exist_ok=True) # Ensure prompts directory also exists
        except OSError as e:
            self.logger("ERROR", f"(PromptManager class) Could not create meta or prompts directory: {e}")

    def _format_timestamp(self, ts_input: Any) -> str:
        """Helper to format timestamps consistently for prompts."""
        if not ts_input: return "Unknown Time"
        try:
            dt_obj: Optional[datetime] = None
            if isinstance(ts_input, str):
                if ts_input.endswith('Z'):
                    if sys.version_info < (3, 11): ts_input = ts_input[:-1] + '+00:00'
                dt_obj = datetime.fromisoformat(ts_input)
            elif isinstance(ts_input, (int, float)):
                dt_obj = datetime.fromtimestamp(ts_input, tz=timezone.utc)
            
            if dt_obj:
                if dt_obj.tzinfo: dt_obj = dt_obj.astimezone(None)
                return dt_obj.strftime("%Y-%m-%d %H:%M")
            return str(ts_input)
        except Exception:
            return str(ts_input)

    def _get_mission_statement_for_prompt(self) -> str:
        if self.mission_manager and hasattr(self.mission_manager, 'get_mission_statement_for_prompt'):
            try:
                return self.mission_manager.get_mission_statement_for_prompt()
            except Exception as e:
                self.logger("ERROR", f"(PromptManager class) Error getting mission statement from MissionManager instance: {e}")
                return _mission_manager_get_mission_statement_fallback()
        # Fallback if no mission_manager instance or it lacks the method
        return _mission_manager_get_mission_statement_fallback()

    def _load_mission_data_for_reflection(self) -> dict:
        if self.mission_manager and hasattr(self.mission_manager, 'load_mission'):
            try:
                return self.mission_manager.load_mission()
            except Exception as e:
                self.logger("ERROR", f"(PromptManager class) Error loading mission data from MissionManager instance for reflection: {e}")
                return _mission_manager_load_mission_fallback()
        return _mission_manager_load_mission_fallback()

    def get_operational_prompt_text(self) -> str:
        """Loads only the operational part of the system prompt."""
        self._ensure_meta_dir()
        base_prompt_text = self.default_base_operational_prompt

        if not os.path.exists(self.system_prompt_operational_part_file):
            self.logger("INFO", f"(PromptManager class): Operational prompt file {self.system_prompt_operational_part_file} not found. Creating with default.")
            try:
                with open(self.system_prompt_operational_part_file, "w", encoding="utf-8") as f:
                    f.write(self.default_base_operational_prompt)
            except IOError as e:
                self.logger("ERROR", f"(PromptManager class): Could not write default operational prompt to {self.system_prompt_operational_part_file}: {e}")
        else:
            try:
                with open(self.system_prompt_operational_part_file, "r", encoding="utf-8") as f:
                    loaded_base_text = f.read().strip()
                if not loaded_base_text:
                    self.logger("WARNING", f"(PromptManager class): Operational prompt file {self.system_prompt_operational_part_file} is empty. Using default and attempting to rewrite.")
                    try:
                        with open(self.system_prompt_operational_part_file, "w", encoding="utf-8") as f_w: f_w.write(self.default_base_operational_prompt)
                    except IOError: 
                        self.logger("ERROR", f"(PromptManager class): Could not rewrite empty prompt file {self.system_prompt_operational_part_file} with default.")
                else:
                    base_prompt_text = loaded_base_text
            except UnicodeDecodeError: 
                self.logger("WARNING", f"(PromptManager class): Failed to decode {self.system_prompt_operational_part_file} as UTF-8. Attempting fallback to 'cp1252'.")
                try:
                    with open(self.system_prompt_operational_part_file, "r", encoding="cp1252") as f_cp:
                        content_cp1252 = f_cp.read().strip()
                    base_prompt_text = content_cp1252 if content_cp1252 else self.default_base_operational_prompt
                    with open(self.system_prompt_operational_part_file, "w", encoding="utf-8") as f_utf8_write: 
                        f_utf8_write.write(base_prompt_text)
                    self.logger("INFO", f"(PromptManager class): Successfully read {self.system_prompt_operational_part_file} with 'cp1252' and re-saved as UTF-8.")
                except Exception as e_fallback_read:
                    self.logger("ERROR", f"(PromptManager class): Could not read/resave {self.system_prompt_operational_part_file} with fallback: {e_fallback_read}. Using default.")
                    base_prompt_text = self.default_base_operational_prompt
            except Exception as e_load_base:
                self.logger("ERROR", f"(PromptManager class): Error loading base operational prompt from {self.system_prompt_operational_part_file}: {e_load_base}. Using default.")
                base_prompt_text = self.default_base_operational_prompt
        return base_prompt_text

    def get_full_system_prompt(self) -> str:
        """
        Constructs the full system prompt by prepending the mission statement
        to the operational instructions.
        """
        mission_prompt_part = self._get_mission_statement_for_prompt()
        operational_prompt_part = self.get_operational_prompt_text()
        
        final_prompt = f"{mission_prompt_part}\n\n### OPERATIONAL INSTRUCTIONS ###\n{operational_prompt_part}"
        return final_prompt

    def update_operational_prompt_text(self, new_operational_text: str):
        """Updates only the operational part of the system prompt."""
        self._ensure_meta_dir()
        try:
            with open(self.system_prompt_operational_part_file, "w", encoding="utf-8") as f:
                f.write(new_operational_text.strip()) 
            self.logger("INFO", "(PromptManager class): Operational system prompt updated successfully.")
        except IOError as e:
            self.logger("ERROR", f"(PromptManager class): Could not update operational prompt file {self.system_prompt_operational_part_file}: {e}")

    def _load_json_for_reflection(self, path: str, default_value: Optional[Union[List, Dict]] = None) -> Any:
        actual_default = [] if default_value is None else default_value
        if not os.path.exists(path): return actual_default
        try:
            with open(path, "r", encoding="utf-8") as f: 
                content = f.read()
                if not content.strip(): return actual_default
                return json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            self.logger("WARNING", f"(PromptManager class): Error loading {path} for reflection: {e}. Using default.")
            return actual_default
        except Exception as e_json_unexp: 
            self.logger("ERROR", f"(PromptManager class): Unexpected error loading JSON {path} for reflection: {e_json_unexp}\n{traceback.format_exc()}. Using default.")
            return actual_default

    def auto_update_prompt(self):
        """Performs deep reflection to update the operational system prompt."""
        self.logger("INFO", "(PromptManager class): Starting auto-reflection for system prompt update...")
        self._ensure_meta_dir()

        if not self.query_llm:
            self.logger("ERROR", "(PromptManager class): LLM query function not available. Cannot perform reflection.")
            return

        goals_data = self._load_json_for_reflection(self.goal_file_path, [])
        suggs_data = self._load_json_for_reflection(self.suggestion_file_path, [])
        changelog_data = self._load_json_for_reflection(self.changelog_file_path, [])
        tool_registry_data = self._load_json_for_reflection(self.tool_registry_file_path, [])
        last_reflection_time_val = self._load_json_for_reflection(self.last_reflection_file_path, {}).get("timestamp")
        
        mission_obj = self._load_mission_data_for_reflection()
        mission_focus_str = ", ".join(mission_obj.get("current_focus_areas", ["N/A"]))

        conversation_snippets_data = self._load_json_for_reflection(self.conversation_review_log_file_path, [])
        recent_conversation_summary_parts = []
        if conversation_snippets_data and isinstance(conversation_snippets_data, list):
            for snippet_entry in conversation_snippets_data[-3:]:
                if isinstance(snippet_entry, dict) and "conversation_snippet" in snippet_entry:
                    thread_id_snip = snippet_entry.get("thread_id", "N/A")[-6:]
                    turns_text = []
                    for turn in snippet_entry.get("conversation_snippet", [])[-4:]:
                        if isinstance(turn, dict):
                             turns_text.append(f"    {turn.get('speaker', '?')}: {turn.get('text', '')[:120]}...")
                    if turns_text:
                        timestamp_snip = snippet_entry.get("timestamp", "N/A")
                        recent_conversation_summary_parts.append(f"  - Snippet from Thread ..{thread_id_snip} (Logged: {self._format_timestamp(timestamp_snip)}):\n" + "\n".join(turns_text))
        
        conversation_context_for_reflection_str = "\n### Recent Conversation Interaction Patterns (Consider for improving dialogue flow, context retention, or information delivery): ###\n" + \
                                                  ("\n".join(recent_conversation_summary_parts) if recent_conversation_summary_parts else "  (No recent conversation snippets logged for review.)")

        total_goals = len(goals_data)
        status_counts: Dict[str, int] = {}
        recent_evaluated_goals_summary = []
        recent_terminal_goals = sorted(
            [g for g in goals_data if isinstance(g, dict) and g.get("status") in ["completed", "executed_with_errors", "build_failed", "failed_max_retries", "failed_correction_unclear"]],
            key=lambda x: x.get("history", [{}])[-1].get("timestamp", x.get("created_at", "1970-01-01T00:00:00Z")), reverse=True
        )[:5] 

        for g_term in recent_terminal_goals:
            eval_score = g_term.get("evaluation", {}).get("final_score", "N/E") 
            fail_cat_term = g_term.get("failure_category", "N/A")
            goal_text_preview = g_term.get('goal', g_term.get('description', ''))[:40] # Check for 'description' too
            summary_entry = f"  - GID ..{g_term.get('goal_id','N/A')[-6:]} (St: {g_term.get('status')}, Score: {eval_score}, FailCat: {fail_cat_term}): '{goal_text_preview}...'"
            recent_evaluated_goals_summary.append(summary_entry)
        
        for g_dict_item in goals_data:
            if isinstance(g_dict_item, dict): status_counts[g_dict_item.get("status","unknown")] = status_counts.get(g_dict_item.get("status","unknown"), 0) + 1
        
        sugg_status_counts: Dict[str, int] = {}
        for s_dict_item in suggs_data:
            if isinstance(s_dict_item, dict): sugg_status_counts[s_dict_item.get("status","pending")] = sugg_status_counts.get(s_dict_item.get("status","pending"),0) + 1

        recent_changelog_entries_summary = [f"  - v{entry.get('version','N/A')}: {entry.get('summary','N/A')[:70]}..." for entry in changelog_data[-3:] if isinstance(entry, dict)]

        total_tools, tools_with_failures_count, successful_tool_runs, total_tool_runs = 0, 0, 0, 0
        if isinstance(tool_registry_data, list):
            total_tools = len([t for t in tool_registry_data if isinstance(t,dict)])
            tools_with_failures_count = len([t for t in tool_registry_data if isinstance(t,dict) and t.get("failure_count", 0) > 0])
            successful_tool_runs = sum(t.get("success_count",0) for t in tool_registry_data if isinstance(t,dict))
            total_tool_runs = sum(t.get("total_runs",0) for t in tool_registry_data if isinstance(t,dict))
        avg_tool_success_rate_str = f"{(successful_tool_runs / total_tool_runs * 100 if total_tool_runs > 0 else 0):.1f}%"
        
        seconds_since_last_reflection_str = f"{int(time.time() - (last_reflection_time_val or time.time()))}s"

        operational_prompt_to_reflect_on = self.get_operational_prompt_text()
        if not operational_prompt_to_reflect_on.strip(): 
            operational_prompt_to_reflect_on = self.default_base_operational_prompt

        reflection_context_parts = [
            "### Current Operational Instructions (for self-correction - DO NOT include the prepended Mission Statement in your response): ###",
            f"\"\"\"{operational_prompt_to_reflect_on}\"\"\"",
            "\n### Current Mission Focus Areas (for context): ###", mission_focus_str,
            "\n### Recent Performance & System Summary: ###",
            f"- Goals: Total={total_goals}. Statuses: {json.dumps(status_counts)}",
            f"- Recently Evaluated/Terminal Goals (last {len(recent_evaluated_goals_summary)}):\n" + ("\n".join(recent_evaluated_goals_summary) if recent_evaluated_goals_summary else "  (No recent terminal goals with evaluation data.)"),
            f"- Suggestions: Total={len(suggs_data)}. Statuses: {json.dumps(sugg_status_counts)}",
            f"- Tools: Registered={total_tools}, With Failures={tools_with_failures_count}, Overall Success Rate={avg_tool_success_rate_str} ({successful_tool_runs}/{total_tool_runs} runs)",
            f"- Recent Releases (last {len(recent_changelog_entries_summary)}):\n" + ("\n".join(recent_changelog_entries_summary) if recent_changelog_entries_summary else "  (None)"),
            conversation_context_for_reflection_str,
            f"- Time Since Last Reflection: {seconds_since_last_reflection_str}",
            "\n### Task for Self-Correction of Operational Instructions: ###",
            "Based on the current OPERATIONAL INSTRUCTIONS and the PERFORMANCE SUMMARY (especially evaluated goals, failure categories, tool success rates, mission focus areas, AND RECENT CONVERSATION PATTERNS), rewrite ONLY THE OPERATIONAL INSTRUCTIONS to improve overall effectiveness. Your response MUST be *only* the new text for these operational instructions.",
            # ... (rest of your detailed instructions for the LLM, unchanged) ...
            "Key areas for improvement based on data:",
            "1. **Error Reduction & Self-Correction:** If specific 'failure_category' or low 'evaluation.final_score' trends are noted, suggest changes to mitigate these. How can instructions for error analysis or self-correction goal formulation be more precise?",
            "2. **Tool Management:** If tool success rate is low or specific tools often fail, how can instructions for tool generation (argparse, main(), no input()), selection, or refinement be improved? Should there be more emphasis on testing new tools?",
            "3. **Goal Processing & Context:** How can instructions better emphasize THREAD_ID usage for context, or improve goal decomposition strategies?",
            "4. **Suggestion Quality:** How can operational instructions guide the AI to generate more impactful and mission-aligned suggestions, considering recent performance and conversation topics?",
            "5. **Conversational Ability (NEW FOCUS):** If 'Recent Conversation Interaction Patterns' show issues like context loss, poor information delivery for queries, or failure to use tools for current information, suggest changes to address these dialogue weaknesses.",
            "6. **Mission Alignment:** Ensure the new operational instructions implicitly support the current mission focus areas.",
            "The new operational instructions should be concise, clear, and maintain the persona of a highly capable AI developer assistant. They will be placed after the standard mission statement.",
            "\nRESPOND *ONLY* WITH THE FULL TEXT OF THE NEW OPERATIONAL INSTRUCTIONS. DO NOT INCLUDE PREambles, THE MISSION STATEMENT, OR ANY OTHER EXPLANATIONS."
        ]
        reflection_prompt_for_llm = "\n".join(reflection_context_parts)
        
        reflection_system_prompt_override = "You are a meta-level AI assistant. Your task is to analyze the provided operational instructions and performance data of an AI system, and then rewrite *only* the operational instructions to enhance its performance and self-improvement capabilities, focusing on data-driven insights. Output only the revised operational instructions text."
        
        new_prompt_candidate_text = self.query_llm( # Use the passed-in query_llm function
            prompt_text=reflection_prompt_for_llm, 
            system_prompt_override=reflection_system_prompt_override, # Assuming query_llm_func can take this
            raw_output=True, 
            timeout=self.config.get("reflection_llm_timeout", 720) # Configurable timeout
        )

        if new_prompt_candidate_text and not new_prompt_candidate_text.startswith("[Error:"):
            new_operational_prompt_clean = new_prompt_candidate_text.strip()
            
            if not new_operational_prompt_clean:
                self.logger("WARNING", "(PromptManager class): Reflection generated an empty new operational prompt. Old prompt retained.")
            elif "### MISSION & IDENTITY ###" in new_operational_prompt_clean or \
                 "mission statement" in new_operational_prompt_clean.lower() or \
                 "### OPERATIONAL INSTRUCTIONS ###" in new_operational_prompt_clean:
                 self.logger("WARNING", "(PromptManager class): Reflection incorrectly included mission-like text or section headers. Old prompt retained. LLM Output (first 200 chars): " + new_operational_prompt_clean[:200] + "...")
            elif len(new_operational_prompt_clean) < 150:
                 self.logger("WARNING", f"(PromptManager class): Reflection generated a very short new operational prompt ({len(new_operational_prompt_clean)} chars). Old prompt retained. LLM: {new_operational_prompt_clean[:100]}...")
            elif new_operational_prompt_clean.lower() == operational_prompt_to_reflect_on.strip().lower():
                self.logger("INFO", "(PromptManager class): Reflection resulted in no change to operational system prompt.")
                try: 
                    with open(self.last_reflection_file_path, "w", encoding="utf-8") as f_reflect: json.dump({"timestamp": time.time()}, f_reflect)
                except IOError as e_reflect_io: self.logger("ERROR", f"(PromptManager class): Could not save last reflection timestamp (no change): {e_reflect_io}")
            else:
                self.update_operational_prompt_text(new_operational_prompt_clean)
                try:
                    with open(self.last_reflection_file_path, "w", encoding="utf-8") as f_reflect: json.dump({"timestamp": time.time()}, f_reflect)
                    self.logger("INFO", "(PromptManager class): Operational prompt auto-updated based on reflection. Review changes in meta/system_prompt.txt.")
                except IOError as e_reflect_io: 
                    self.logger("ERROR", f"(PromptManager class): Could not save last reflection timestamp after update: {e_reflect_io}")
        else:
            self.logger("ERROR", f"(PromptManager class): Failed to get new operational prompt from LLM during reflection. LLM Error/Response: {new_prompt_candidate_text}")

    # --- Methods for structured prompt templates ---
    def get_template_content(self, template_name: str, sub_component: Optional[str] = None) -> Optional[str]:
        """
        Loads the content of a specific prompt template file.
        Example: template_name='planner_decide_action', sub_component='planner'
        Looks for 'prompts/planner/planner_decide_action.txt' or 'prompts/planner_decide_action.txt'
        """
        filename = f"{template_name}.txt"
        search_paths = []
        if sub_component:
            search_paths.append(os.path.join(self.prompts_base_path, sub_component, filename))
        search_paths.append(os.path.join(self.prompts_base_path, filename))

        for filepath in search_paths:
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception as e:
                    self.logger("ERROR", f"(PromptManager class) Error reading template {filepath}: {e}")
                    return None
        self.logger("WARNING", f"(PromptManager class) Template '{template_name}' not found in paths: {search_paths}")
        return None

    def render_prompt_with_dynamic_content(self, template_name: str, dynamic_content: Dict[str, Any], sub_component: Optional[str] = None) -> str:
        """
        Loads a template and fills it with dynamic content.
        Uses simple {{key}} replacement.
        """
        template_string = self.get_template_content(template_name, sub_component)
        if template_string is None:
            self.logger("ERROR", f"(PromptManager class) Cannot render prompt: Template '{template_name}' (sub: {sub_component}) not found or failed to load.")
            # Fallback to a very basic prompt structure if template loading fails
            if template_name == "planner_decide_action": # Specific fallback for planner
                 return f"Fallback PromptManager: Error loading template '{template_name}'.\nBased on context: {str(dynamic_content)[:200]}..., what is the best next_action and its details? Respond ONLY with a valid JSON object."
            return f"[ERROR: Template '{template_name}' not found for rendering]"

        # Simple {{key}} replacement
        # For more complex templating, consider libraries like Jinja2
        rendered_prompt = template_string
        for key, value in dynamic_content.items():
            placeholder = "{{" + key + "}}"
            rendered_prompt = rendered_prompt.replace(placeholder, str(value))
        
        # Check for any remaining unreplaced placeholders (optional, for debugging)
        # import re
        # if re.search(r"\{\{[^}]+\}\}", rendered_prompt):
        #     self.logger("WARNING", f"Unreplaced placeholders found in rendered prompt for template '{template_name}'.")
            
        return rendered_prompt


if __name__ == "__main__":
    print("--- Testing PromptManager Class (Standalone) ---")
    
    # Mock dependencies for standalone testing
    mock_config_pm_test = {
        "meta_dir": "meta_pm_test",
        "prompts_base_path": "prompts_pm_test",
        "goal_file": "goals_test.json", # Ensure these are relative to meta_dir
        "suggestion_file": "suggestions_test.json",
        "changelog_file": "changelog_test.json",
        "tool_registry_file": "tool_registry_test.json",
        "last_reflection_file": "last_reflection_test.json",
        "conversation_review_log_file": "conv_review_test.json",
        "reflection_llm_timeout": 60 # Shorter for test
    }
    
    # Create dummy directories and files needed for the test
    os.makedirs(mock_config_pm_test["meta_dir"], exist_ok=True)
    os.makedirs(mock_config_pm_test["prompts_base_path"], exist_ok=True)

    # Create a dummy operational prompt file
    dummy_op_prompt_path = os.path.join(mock_config_pm_test["meta_dir"], "system_prompt.txt")
    with open(dummy_op_prompt_path, "w", encoding="utf-8") as f:
        f.write("This is the default operational prompt for testing the PromptManager class.")

    # Create a dummy structured prompt template file
    dummy_template_dir = os.path.join(mock_config_pm_test["prompts_base_path"], "planner")
    os.makedirs(dummy_template_dir, exist_ok=True)
    dummy_template_path = os.path.join(dummy_template_dir, "planner_decide_action.txt")
    with open(dummy_template_path, "w", encoding="utf-8") as f:
        f.write("User Input: {{user_input}}\nAvailable Tools: {{tool_capabilities_summary}}\nDecide next action.")


    def mock_logger_pm(level, message): print(f"[{level.upper()}] (MockLoggerForPMTest) {message}")
    
    def mock_query_llm_pm(prompt_text, system_prompt_override=None, raw_output=False, timeout=300):
        mock_logger_pm("INFO", f"Mock LLM called for PM test. Prompt starts: {prompt_text[:100]}...")
        if "### Task for Self-Correction" in prompt_text: # Reflection prompt
            return "This is a new operational instruction from mock LLM after reflection."
        return "[Mock LLM Response: Action processed]"

    class MockMissionManagerForPMTest:
        def get_mission_statement_for_prompt(self): return "### MOCK MISSION ###\nThis is a test mission statement.\n---------------------"
        def load_mission(self): return {"current_focus_areas": ["Test Focus 1", "Test Focus 2"]}

    pm_instance = PromptManager(
        config=mock_config_pm_test,
        logger_func=mock_logger_pm,
        query_llm_func=mock_query_llm_pm,
        mission_manager_instance=MockMissionManagerForPMTest()
    )

    print("\n1. Testing get_full_system_prompt():")
    full_prompt = pm_instance.get_full_system_prompt()
    print(f"   Full system prompt (first 150 chars): {full_prompt[:150]}...")
    if "MOCK MISSION" not in full_prompt or "default operational prompt" not in full_prompt:
        print("   ERROR: Full prompt doesn't seem to combine mission and operational parts correctly.")

    print("\n2. Testing update_operational_prompt_text():")
    pm_instance.update_operational_prompt_text("New operational text for testing update.")
    updated_op_text = pm_instance.get_operational_prompt_text()
    if "New operational text" not in updated_op_text:
        print(f"   ERROR: Operational text not updated. Got: {updated_op_text}")
    else:
        print("   Operational text updated successfully.")

    print("\n3. Testing auto_update_prompt() (reflection):")
    # Create dummy context files for reflection
    for f_name_key in ["goal_file", "suggestion_file", "changelog_file", "tool_registry_file", "last_reflection_file", "conversation_review_log_file"]:
        f_path = os.path.join(mock_config_pm_test["meta_dir"], mock_config_pm_test[f_name_key])
        if not os.path.exists(f_path):
            with open(f_path, "w", encoding="utf-8") as f_dummy_ctx: json.dump([], f_dummy_ctx)
    
    pm_instance.auto_update_prompt()
    reflected_op_text = pm_instance.get_operational_prompt_text()
    if "new operational instruction from mock LLM" not in reflected_op_text:
        print(f"   ERROR: Reflection did not update prompt as expected. Got: {reflected_op_text}")
    else:
        print(f"   Operational prompt after reflection: {reflected_op_text[:100]}...")
    
    print("\n4. Testing get_template_content() and render_prompt_with_dynamic_content():")
    template_content = pm_instance.get_template_content("planner_decide_action", sub_component="planner")
    if template_content:
        print("   SUCCESS: Loaded template 'planner_decide_action'.")
        dynamic_data = {"user_input": "Test user query", "tool_capabilities_summary": "Tool A, Tool B"}
        rendered = pm_instance.render_prompt_with_dynamic_content("planner_decide_action", dynamic_data, sub_component="planner")
        print(f"   Rendered prompt: {rendered}")
        if "Test user query" not in rendered or "Tool A, Tool B" not in rendered:
            print("   ERROR: Rendering with dynamic content failed.")
    else:
        print("   ERROR: Failed to load template 'planner_decide_action'.")

    print("\n--- PromptManager Class Test Complete ---")
    print(f"Please review meta_pm_test/ and prompts_pm_test/ directories for test artifacts.")