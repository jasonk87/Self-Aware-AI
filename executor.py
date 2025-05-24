# Self-Aware/executor.py
import json
import os
import subprocess
import sys
import traceback
import re
import uuid
from datetime import datetime, timezone # Ensure this is imported
from typing import List, Dict, Optional, Any, Callable, Tuple

# --- Module-level Imports from your script ---
# These will be used by the Executor class methods or for module-level constants.
# The Executor class will primarily rely on instances passed to its __init__ for these.

# Original top-level imports for dependencies.
# Global fallbacks are provided for context if this module were somehow run in isolation
# or if instances aren't correctly passed to the Executor class.
from logger_utils import should_log # Added import
try:
    # These are imported for potential fallback use if instances are not provided to the class.
    # The class itself will prefer using injected instances (self.tool_builder, self.tool_runner, etc.)
    from tool_builder_module import build_tool as _executor_global_build_tool_func
except ImportError as e_tb_exec:
    if should_log("WARNING"): print(f"WARNING (executor_module_load): tool_builder_module.py not found. Executor will rely on ToolBuilder instance. Error: {e_tb_exec}")
    _executor_global_build_tool_func = None

try:
    from tool_runner import run_tool_safely as _executor_global_run_tool_safely_func
except ImportError as e_tr_exec:
    if should_log("WARNING"): print(f"WARNING (executor_module_load): tool_runner.py not found. Executor will rely on ToolRunner instance. Error: {e_tr_exec}")
    _executor_global_run_tool_safely_func = None

try:
    from notifier_module import log_update as _executor_global_log_update_func
except ImportError as e_nm_exec:
    if should_log("WARNING"): print(f"WARNING (executor_module_load): notifier_module.py not found. Executor will rely on Notifier instance. Error: {e_nm_exec}")
    _executor_global_log_update_func = None

# Centralized logger and query_llm fallbacks for module-level use IF ai_core isn't available
# The Executor class instance will use what's passed to its __init__ or its own instance fallbacks.
_EXECUTOR_MODULE_LOGGER_FALLBACK = lambda level, msg: (print(f"[{level.upper()}] (ExecutorModuleGlobalLogFallback) {msg}") if should_log(level.upper()) else None)
_EXECUTOR_MODULE_QUERY_LLM_FALLBACK = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=180: \
    f"[Error: Global Fallback LLM from Executor module. Prompt: {prompt_text[:100]}...]"

_executor_module_logger = _EXECUTOR_MODULE_LOGGER_FALLBACK
_executor_module_query_llm = _EXECUTOR_MODULE_QUERY_LLM_FALLBACK
# _executor_module_ai_core_model_name = "module_model_fallback" # Not used directly by class

try:
    from ai_core import query_llm_internal, log_background_message # MODEL_NAME is not used by executor directly
    _executor_module_logger = log_background_message
    _executor_module_query_llm = query_llm_internal
    # _executor_module_ai_core_model_name = AI_CORE_MODEL_NAME_IMPORTED # Not used by executor directly
except ImportError:
    _executor_module_logger("WARNING", "(ExecutorModuleLoad) ai_core not fully imported. Executor class relies on injected dependencies.")

_executor_module_mission_manager_global = None
try:
    import mission_manager as mm_module_exec_import # Alias for clarity
    _executor_module_mission_manager_global = mm_module_exec_import
except ImportError:
    _executor_module_logger("CRITICAL", "(ExecutorModuleLoad) mission_manager.py not found. Executor methods needing mission context will be impaired if instance not provided.")


# --- Module-level Constants (from your original script) ---
META_DIR = "meta" # This is your original global constant

# Goal Priorities (module-level, as they are definitions)
PRIORITY_URGENT = 1
PRIORITY_HIGH = 2
PRIORITY_NORMAL = 3
PRIORITY_LOW = 4

# Subtask Categorization
CAT_COMMAND_EXECUTION = "command_execution"
CAT_CODE_GENERATION_TOOL = "code_generation_tool"
CAT_CODE_GENERATION_SNIPPET = "code_generation_snippet"
CAT_INFORMATION_GATHERING = "information_gathering"
CAT_USE_EXISTING_TOOL = "use_existing_tool"
CAT_REFINEMENT = "code_refinement"
CAT_CORE_FILE_UPDATE = "core_file_update"

# Goal Statuses
STATUS_PENDING = "pending"
STATUS_DECOMPOSED = "decomposed"
STATUS_APPROVED = "approved"
STATUS_COMPLETED = "completed"
STATUS_EXECUTED_WITH_ERRORS = "executed_with_errors"
STATUS_BUILD_FAILED = "build_failed"
STATUS_AWAITING_CORRECTION = "awaiting_correction"
STATUS_FAILED_MAX_RETRIES = "failed_max_retries"
STATUS_FAILED_UNCLEAR = "failed_correction_unclear"

# Goal Sources
SOURCE_USER = "user"
SOURCE_DECOMPOSITION = "decomposition"
SOURCE_SUGGESTION = "suggestion_approved"
SOURCE_SELF_CORRECTION = "AI_self_correction"
SOURCE_AUTO_REFINEMENT = "AI_auto_refinement"


class Executor:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None,
                 mission_manager_instance: Optional[Any] = None,
                 notifier_instance: Optional[Any] = None,
                 tool_builder_instance: Optional[Any] = None,
                 tool_runner_instance: Optional[Any] = None
                 ):
        self.config = config if config else {}
        self.logger = logger_func if logger_func else _executor_module_logger
        self.query_llm = query_llm_func if query_llm_func else _executor_module_query_llm
        
        self.mission_manager = mission_manager_instance
        self.notifier = notifier_instance
        self.tool_builder = tool_builder_instance
        self.tool_runner = tool_runner_instance

        # Fallbacks for injected instances if they are None
        if not self.mission_manager:
            self.logger("WARNING", "(Executor Init) No MissionManager instance provided. Trying to use globally imported module (limited functionality).")
            if _executor_module_mission_manager_global and hasattr(_executor_module_mission_manager_global, 'load_mission'):
                self.mission_manager = _executor_module_mission_manager_global
            else:
                class _DummyMissionManager: # Minimal fallback
                    def load_mission(self): return {"current_focus_areas": ["MissionManager Unavailable"]}
                self.mission_manager = _DummyMissionManager()
                self.logger("ERROR", "(Executor Init) MissionManager instance AND global module import failed.")
        
        if not self.notifier:
            self.logger("WARNING", "(Executor Init) No Notifier instance provided. Using global log_update_func if available.")
            class _TempNotifier:
                def __init__(self, logger): self.logger = logger
                def log_update(self, summary, goals, approved_by):
                    if _executor_global_log_update_func: _executor_global_log_update_func(summary, goals, approved_by)
                    else: self.logger("ERROR", f"DummyNotifier: log_update called for: {summary}, but global func missing.")
            self.notifier = _TempNotifier(self.logger)

        if not self.tool_builder:
            self.logger("WARNING", "(Executor Init) No ToolBuilder instance provided. Using global build_tool_func if available.")
            class _TempToolBuilder:
                def __init__(self, logger): self.logger = logger
                def build_tool(self, description, tool_name_suggestion, thread_id, goals_list_ref, current_goal_id):
                    if _executor_global_build_tool_func: return _executor_global_build_tool_func(description, tool_name_suggestion, thread_id, goals_list_ref, current_goal_id)
                    self.logger("ERROR", f"DummyToolBuilder: build_tool for '{tool_name_suggestion}' called but no implementation."); return None
            self.tool_builder = _TempToolBuilder(self.logger)

        if not self.tool_runner:
            self.logger("WARNING", "(Executor Init) No ToolRunner instance provided. Using global run_tool_safely_func if available.")
            class _TempToolRunner:
                def __init__(self, logger): self.logger = logger
                def run_tool_safely(self, tool_path, tool_args=None):
                    if _executor_global_run_tool_safely_func: return _executor_global_run_tool_safely_func(tool_path, tool_args)
                    self.logger("ERROR", f"DummyToolRunner: run_tool_safely for '{tool_path}' called but no implementation."); return {"status": "error", "error": "ToolRunner unavailable"}
            self.tool_runner = _TempToolRunner(self.logger)

        # Initialize paths and config values from self.config, using original script's defaults
        self.meta_dir_path = self.config.get("meta_dir", META_DIR) # Use global META_DIR from your script as default
        self.goal_file_path = self.config.get("goals_file", os.path.join(self.meta_dir_path, "goals.json"))
        self.tool_registry_file_path = self.config.get("tool_registry_file", os.path.join(self.meta_dir_path, "tool_registry.json"))

        executor_specific_cfg = self.config.get("executor_config", {})
        self.decomposition_threshold = executor_specific_cfg.get("decomposition_threshold", 70) # From your original script
        self.max_self_correction_attempts = executor_specific_cfg.get("max_self_correction_attempts", 2) # From your original script
        
        self._ensure_meta_dir() # Call on init

    def _ensure_meta_dir(self): # Was _ensure_meta_dir_executor
        """Ensures the meta directory (self.meta_dir_path) exists."""
        try:
            os.makedirs(self.meta_dir_path, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(Executor class): Could not create meta directory {self.meta_dir_path}: {e}")

    # --- Start of File Helper Methods ---
    def load_goals(self) -> list:
        """Loads goals from the configured goals file, applying migrations/defaults."""
        self._ensure_meta_dir()
        if not os.path.exists(self.goal_file_path):
            try:
                with open(self.goal_file_path, "w", encoding="utf-8") as f: json.dump([], f)
                self.logger("INFO", f"(Executor|load_goals): Initialized empty goals file: {self.goal_file_path}")
            except IOError as e_create:
                self.logger("ERROR", f"(Executor|load_goals): Could not create file {self.goal_file_path}: {e_create}")
            return []
        try:
            with open(self.goal_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip(): return []
            loaded_goals = json.loads(content)
            # Migration logic from your original load_goals:
            for goal in loaded_goals:
                goal.setdefault("goal_id", str(uuid.uuid4()))
                goal.setdefault("thread_id", str(uuid.uuid4()))
                goal.setdefault("priority", goal.get("priority", PRIORITY_NORMAL))
                goal.setdefault("created_at", goal.get("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
                goal.setdefault("history", goal.get("history", [{"timestamp": goal.get("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")), "status": goal.get("status", STATUS_PENDING), "action": "Goal loaded/migrated"}]))
                goal.setdefault("failure_category", goal.get("failure_category", None))
                goal.setdefault("self_correction_attempts", goal.get("self_correction_attempts", 0))
                goal.setdefault("is_subtask", goal.get("is_subtask", False))
                goal.setdefault("parent", goal.get("parent", None))
                goal.setdefault("parent_goal_id", goal.get("parent_goal_id", None))
                goal.setdefault("subtasks", goal.get("subtasks", None))
                goal.setdefault("tool_file", goal.get("tool_file", None))
                goal.setdefault("execution_result", goal.get("execution_result", None))
                goal.setdefault("error", goal.get("error", None))
                goal.setdefault("source", goal.get("source", SOURCE_USER))
            return loaded_goals
        except json.JSONDecodeError:
            self.logger("WARNING", f"(Executor|load_goals): Could not decode {self.goal_file_path}. Returning empty list.")
            return []
        except Exception as e:
            self.logger("ERROR", f"(Executor|load_goals): Unexpected error loading {self.goal_file_path}: {e}\n{traceback.format_exc()}")
            return []

    def save_goals(self, goals: list):
        """Saves the provided list of goals to the configured file."""
        self._ensure_meta_dir()
        try:
            with open(self.goal_file_path, "w", encoding="utf-8") as f:
                json.dump(goals, f, indent=2)
        except Exception as e:
            self.logger("ERROR", f"(Executor|save_goals): Unexpected error saving {self.goal_file_path}: {e}\n{traceback.format_exc()}")

    def get_structured_tool_registry(self) -> list:
        """Loads the tool registry, applying migrations/defaults to entries."""
        self._ensure_meta_dir()
        if not os.path.exists(self.tool_registry_file_path):
            try:
                with open(self.tool_registry_file_path, "w", encoding="utf-8") as f: json.dump([], f)
                self.logger("INFO", f"(Executor|get_tool_registry): Initialized empty tool registry: {self.tool_registry_file_path}")
            except IOError as e_create_reg:
                 self.logger("ERROR", f"(Executor|get_tool_registry): Could not create file {self.tool_registry_file_path}: {e_create_reg}")
            return []
        try:
            with open(self.tool_registry_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip(): return []
            registry = json.loads(content)
            if not isinstance(registry, list): # From your original code
                self.logger("WARNING", f"(Executor|get_tool_registry): Content in {self.tool_registry_file_path} is not a list. Reinitializing.")
                return []
            migrated_registry = []
            for tool_entry in registry: # Migration logic from your original
                if not isinstance(tool_entry, dict):
                    self.logger("WARNING", f"(Executor|get_tool_registry): Skipping non-dictionary entry: {str(tool_entry)[:100]}")
                    continue
                tool_entry.setdefault("last_run_time", None)
                tool_entry.setdefault("success_count", 0)
                tool_entry.setdefault("failure_count", 0)
                tool_entry.setdefault("total_runs", tool_entry.get("success_count",0) + tool_entry.get("failure_count",0))
                tool_entry.setdefault("failure_details", []) 
                tool_entry.setdefault("args_info", tool_entry.get("args_info", []))
                tool_entry.setdefault("capabilities", tool_entry.get("capabilities", []))
                tool_entry.setdefault("description", tool_entry.get("description", "No description provided."))
                tool_entry.setdefault("module_path", tool_entry.get("module_path", None))
                tool_entry.setdefault("last_updated", tool_entry.get("last_updated", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
                migrated_registry.append(tool_entry)
            return migrated_registry
        except json.JSONDecodeError as e_json:
            self.logger("ERROR", f"(Executor|get_tool_registry): Loading failed (JSON decode): {e_json}. File might be corrupt.")
            return []
        except Exception as e:
            self.logger("ERROR", f"(Executor|get_tool_registry): Loading failed: {e}\n{traceback.format_exc()}")
            return []

    def save_structured_tool_registry(self, registry_data: list):
        """Saves the provided tool registry data to the configured file."""
        self._ensure_meta_dir()
        try:
            with open(self.tool_registry_file_path, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            self.logger("ERROR", f"(Executor|save_tool_registry): Saving failed: {e}\n{traceback.format_exc()}")

    # --- Goal Management Methods (converted from your global functions) ---
    def register_goal_in_memory(self, goal_obj_to_add: dict, goals_list_in_memory: list) -> bool:
        """Adds goal to list if not a functional duplicate (same core fields, pending)."""
        # This logic is exactly from your script.
        if any(existing_goal.get("goal_id") == goal_obj_to_add.get("goal_id") for existing_goal in goals_list_in_memory):
            # self.logger("DEBUG", f"(Executor|register_goal) Goal ID {goal_obj_to_add.get('goal_id')} already exists.")
            return False

        for existing_goal in goals_list_in_memory:
            if existing_goal.get("goal") == goal_obj_to_add.get("goal") and \
               existing_goal.get("parent_goal_id") == goal_obj_to_add.get("parent_goal_id") and \
               existing_goal.get("status") == STATUS_PENDING and \
               existing_goal.get("source") == goal_obj_to_add.get("source") and \
               existing_goal.get("thread_id") == goal_obj_to_add.get("thread_id"):
                # self.logger("DEBUG", f"(Executor|register_goal) Functionally duplicate goal for '{goal_obj_to_add.get('goal')}' found.")
                return False
            
        goals_list_in_memory.append(goal_obj_to_add)
        return True

    def add_goal(
        self, goals_list_ref: list, description: str, source: str = SOURCE_USER, # Using module-level SOURCE_USER
        approved_by: Optional[str] = None, is_subtask: bool = False,
        parent_goal_obj: Optional[Dict[str, Any]] = None,
        related_to_failed_goal_obj: Optional[Dict[str, Any]] = None,
        thread_id_param: Optional[str] = None,
        priority: int = PRIORITY_NORMAL, # Using module-level PRIORITY_NORMAL
        subtask_category_override: Optional[str] = None,
        target_file_for_processing: Optional[str] = None,
        code_content_for_core_update: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Creates and adds a new goal object to the provided goals list."""
        # This logic is exactly from your script.
        goal_uuid = str(uuid.uuid4())
        effective_thread_id = thread_id_param

        if not effective_thread_id:
            if parent_goal_obj and isinstance(parent_goal_obj, dict):
                effective_thread_id = parent_goal_obj.get("thread_id")
            elif related_to_failed_goal_obj and isinstance(related_to_failed_goal_obj, dict):
                effective_thread_id = related_to_failed_goal_obj.get("thread_id")
            
            if not effective_thread_id: # If still not found, generate a new one
                effective_thread_id = str(uuid.uuid4())

        timestamp_now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        new_goal = {
            "goal_id": goal_uuid,
            "thread_id": effective_thread_id,
            "goal": description,
            "source": source,
            "status": STATUS_PENDING, # Using module-level STATUS_PENDING
            "priority": priority,
            "is_subtask": is_subtask,
            "parent": parent_goal_obj.get("goal") if parent_goal_obj and isinstance(parent_goal_obj, dict) else None,
            "parent_goal_id": parent_goal_obj.get("goal_id") if parent_goal_obj and isinstance(parent_goal_obj, dict) else None,
            "self_correction_attempts": 0,
            "failure_category": None,
            "created_at": timestamp_now,
            "history": [{"timestamp": timestamp_now, "status": STATUS_PENDING, "action": f"Goal added (Source: {source}, Priority: {priority})"}]
            # Other optional fields from your original add_goal
        }

        if approved_by: new_goal["approved_by"] = approved_by
        if related_to_failed_goal_obj and isinstance(related_to_failed_goal_obj, dict):
            new_goal["related_to_failed_goal_id"] = related_to_failed_goal_obj.get("goal_id")
        if subtask_category_override: new_goal["subtask_category"] = subtask_category_override
        if target_file_for_processing: new_goal["target_file_for_processing"] = target_file_for_processing
        if code_content_for_core_update: new_goal["code_content_for_core_update"] = code_content_for_core_update
        
        if self.register_goal_in_memory(new_goal, goals_list_ref): # Call the instance method
            self.logger("INFO", f"(Executor|add_goal): Added GID ..{new_goal['goal_id'][-6:]}, Thread ..{new_goal['thread_id'][-6:]}, Desc: '{description[:50]}...'")
            return True, goal_uuid
        self.logger("WARNING", f"(Executor|add_goal): Failed to register (likely duplicate) GID ..{new_goal['goal_id'][-6:]}, Desc: '{description[:50]}...'")
        return False, None

    def _update_goal_status_and_history(self, goal_obj: Dict[str, Any], new_status: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Updates the status and history of a goal object in-place."""
        # This logic is exactly from your script.
        goal_obj["status"] = new_status
        history_entry: Dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "status": new_status}
        if message: history_entry["message"] = message
        if details: history_entry["details"] = details
        
        if not isinstance(goal_obj.get("history"), list):
            goal_obj["history"] = [] # Initialize if missing or wrong type
        goal_obj["history"].append(history_entry)

    def _categorize_subtask(self, description: str) -> str: # Was global categorize_subtask
        """Categorizes a subtask based on its description."""
        # This logic is exactly from your script, using module-level category constants.
        desc_lower = description.lower()
        if "update core file" in desc_lower or "modify ai_core.py" in desc_lower or "apply core update" in desc_lower:
            return CAT_CORE_FILE_UPDATE
        if desc_lower.startswith("execute python code:") or desc_lower.startswith("run python snippet:"):
            return CAT_CODE_GENERATION_SNIPPET
        if desc_lower.startswith("install ") or "pip install" in desc_lower or "apt-get install" in desc_lower:
            return CAT_COMMAND_EXECUTION
        if desc_lower.startswith("refine tool") or desc_lower.startswith("modify script") or desc_lower.startswith("improve code in"):
            return CAT_REFINEMENT
        if desc_lower.startswith("define ") or desc_lower.startswith("document ") or desc_lower.startswith("research ") or "api for" in desc_lower or desc_lower.startswith("what is") or desc_lower.startswith("explain"):
            return CAT_INFORMATION_GATHERING
        return CAT_CODE_GENERATION_TOOL # Default category

    def _get_context_for_llm_from_thread(self, thread_id: Optional[str], goals_list_ref: list, current_goal_id_to_exclude: Optional[str] = None, max_entries: int = 3) -> str:
        """Constructs a string of recent goal history for the given thread."""
        # This logic is exactly from your script.
        if not thread_id or not goals_list_ref: return ""
        
        related_goals_info = []
        # Filter and sort goals belonging to the thread
        thread_goals = sorted(
            [g for g in goals_list_ref if isinstance(g, dict) and g.get("thread_id") == thread_id and g.get("goal_id") != current_goal_id_to_exclude],
            key=lambda x: x.get("created_at", ""), # Sort by creation time
            reverse=True # Newest first
        )
        
        for g_ctx in thread_goals[:max_entries]:
            entry = f"- Goal: \"{g_ctx.get('goal','Unnamed Goal')[:80]}...\" (ID: ..{g_ctx.get('goal_id','N/A')[-6:]}, Status: {g_ctx.get('status','N/A')}"
            if g_ctx.get("error"):
                error_str = str(g_ctx['error'])
                first_line_error = error_str.splitlines()[0] if error_str.splitlines() else error_str
                entry += f", LastError: {first_line_error[:100]}"
            if g_ctx.get("tool_file"):
                entry += f", Tool: {os.path.basename(g_ctx['tool_file'])}"
            entry += ")"
            related_goals_info.append(entry)
            
        if not related_goals_info: return ""
        return "\n### Relevant Goal History (same thread):\n" + "\n".join(related_goals_info)

    # --- Goal Decomposition Method ---
    def decompose_goal(self, parent_goal_obj: Dict[str, Any], goals_list_ref: list) -> List[str]:
        """
        Decomposes a parent goal into subtasks using an LLM.
        Uses self.query_llm, self.logger, self.config, and self._get_context_for_llm_from_thread.
        """
        # This logic is exactly from your script.
        system_prompt_decompose = (
            "You are a meticulous planner AI. Break down a complex goal into actionable, distinct subtasks. "
            "Consider the parent goal's thread context. Focus on creating independent steps. "
            "Respond ONLY with a JSON array of strings. Example: [\"Subtask 1\", \"Subtask 2\"]"
        )
        thread_context = self._get_context_for_llm_from_thread(parent_goal_obj.get("thread_id"), goals_list_ref, parent_goal_obj.get("goal_id"))
        prompt_for_llm = (f"Parent Goal to Decompose: \"{parent_goal_obj['goal']}\" (ID: ..{parent_goal_obj.get('goal_id','N/A')[-6:]})\n{thread_context}\n\nJSON Array of Subtask Descriptions:") # Added .get for safety
        
        raw_response_text = self.query_llm(
            prompt_text=prompt_for_llm,
            system_prompt_override=system_prompt_decompose,
            timeout=self.config.get("decompose_llm_timeout", 240) # Using configured timeout
        )

        if raw_response_text.startswith("[Error:"):
            self.logger("ERROR", f"(Executor|decompose_goal): LLM failed for '{parent_goal_obj.get('goal','N/A')[:50]}...': {raw_response_text}")
            return []
        try:
            tasks = json.loads(raw_response_text)
            if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                return [t.strip() for t in tasks if t.strip()]
        except json.JSONDecodeError:
            # Attempt to extract JSON array using regex if direct parse fails (as in original)
            match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', raw_response_text, re.DOTALL)
            if match:
                try:
                    tasks = json.loads(match.group(0))
                    if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
                        return [t.strip() for t in tasks if t.strip()]
                except json.JSONDecodeError:
                    pass # Fall through to warning if regex-extracted part also fails
            self.logger("WARNING", f"(Executor|decompose_goal): Could not parse JSON array for '{parent_goal_obj.get('goal','N/A')[:50]}...'. Raw: {raw_response_text[:150]}")
        return []

    # --- Tool Related Methods ---
    def _ask_llm_for_tool_selection(self, task_description: str, available_tools: list, 
                                   thread_id: Optional[str], goals_list_ref: list, 
                                   current_goal_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Asks LLM to select an existing tool for a task.
        Uses self.query_llm, self.logger, self.config, and self._get_context_for_llm_from_thread.
        """
        # This logic is exactly from your script.
        if not available_tools:
            return None
        
        system_prompt_tool_select = (
            "You are an AI tool dispatcher. Select the most appropriate pre-existing tool for the task. "
            "Consider tool descriptions, capabilities, arguments, and the goal's thread history. "
            "Respond with the exact 'Tool Name' (from the list) or 'NONE'."
        )
        tools_summary = "\n".join([
            f"Tool: \"{t.get('name', 'N/A')}\" (Desc: {t.get('description', 'N/A')[:60]}..., Args: {len(t.get('args_info',[]))}, LastRun: {t.get('last_run_time', 'Never')}, Successes: {t.get('success_count',0)}/{t.get('total_runs',0)})"
            for t in available_tools
        ])
        thread_context = self._get_context_for_llm_from_thread(thread_id, goals_list_ref, current_goal_id)
        prompt = (
            f"Current Task: \"{task_description}\"\n\nAvailable Tools:\n{tools_summary}\n{thread_context}\n\n"
            "Which tool name from the list is best suited? (Respond with exact name or NONE):"
        )
        
        selected_tool_name_raw = self.query_llm(
            prompt,
            system_prompt_override=system_prompt_tool_select,
            raw_output=True, # As in original
            timeout=self.config.get("tool_select_llm_timeout", 180) # Using configured timeout
        ).strip()
        selected_tool_name = selected_tool_name_raw.replace("\"", "") # Remove quotes

        if selected_tool_name and selected_tool_name.upper() != "NONE":
            for tool_entry in available_tools:
                if tool_entry.get("name") == selected_tool_name:
                    self.logger("INFO", f"(Executor|_ask_llm_tool_select): LLM selected tool '{selected_tool_name}' for task '{task_description[:30]}...'.")
                    return tool_entry
            self.logger("WARNING", f"(Executor|_ask_llm_tool_select): LLM selected tool '{selected_tool_name}' but it's not in the list. Raw: '{selected_tool_name_raw}'")
        return None

    def _extract_arguments_for_tool(self, goal_description: str, tool_definition: Dict[str, Any], 
                                   thread_id: Optional[str], goals_list_ref: list, 
                                   current_goal_id: Optional[str]) -> Optional[List[str]]:
        """
        Asks LLM to extract arguments for a selected tool based on the goal.
        Uses self.query_llm, self.logger, self.config, and self._get_context_for_llm_from_thread.
        """
        # This logic is exactly from your script.
        tool_name = tool_definition.get("name", "UnknownTool")
        args_spec = tool_definition.get("args_info", [])
        if not args_spec: # If no arguments are defined for the tool
            return [] 

        system_prompt_arg_extract = (
            "You are an AI argument extractor. Based on the goal, tool info, and thread history, "
            "provide a JSON object mapping argument names (from 'args_info') to string values. "
            "For required args, provide a value or indicate error. For optional, provide if implied, else omit. "
            "Example: {\"query\": \"AI safety\", \"limit\": \"10\"} or {\"error\": \"Missing required argument: input_file\"}"
        )
        arg_details_prompt = "\n".join([
            f"- {a.get('name','N/A')} ({'req' if a.get('required') else 'opt'}, type: {a.get('type','str')}, desc: {a.get('description','')[:50]}...)" 
            for a in args_spec if isinstance(a, dict) # Ensure 'a' is a dict
        ])
        thread_context = self._get_context_for_llm_from_thread(thread_id, goals_list_ref, current_goal_id)
        prompt = (
            f"Goal: \"{goal_description}\"\nTool: \"{tool_name}\" (Desc: {tool_definition.get('description','')[:60]}...)\n"
            f"Expected Arguments:\n{arg_details_prompt}{thread_context}\n\nJSON Response (argument map or error object):"
        )
        
        llm_response_str = self.query_llm(
            prompt,
            system_prompt_override=system_prompt_arg_extract,
            raw_output=True, # As in original
            timeout=self.config.get("arg_extract_llm_timeout", 180) # Using configured timeout
        )

        try:
            extracted_values_map = json.loads(llm_response_str)
            if not isinstance(extracted_values_map, dict):
                raise json.JSONDecodeError("LLM response is not a JSON object.", llm_response_str, 0)
            if "error" in extracted_values_map:
                self.logger("ERROR", f"(Executor|_extract_args): LLM indicated error for '{tool_name}': {extracted_values_map['error']}")
                return None
        except json.JSONDecodeError as e:
            self.logger("ERROR", f"(Executor|_extract_args): Failed to parse LLM JSON for '{tool_name}'. Response: '{llm_response_str[:100]}'. Error: {e}")
            return None

        final_args_for_runner = []
        for arg_spec_item in args_spec:
            if not isinstance(arg_spec_item, dict): continue # Skip if arg_spec_item is not a dict
            arg_name = arg_spec_item.get("name")
            cli_arg_name = f"--{arg_name.replace('_', '-')}" if arg_name else None
            if not cli_arg_name: continue

            if arg_name in extracted_values_map:
                final_args_for_runner.extend([cli_arg_name, str(extracted_values_map[arg_name])])
            elif arg_spec_item.get("required", False):
                self.logger("ERROR", f"(Executor|_extract_args): Required arg '{arg_name}' for tool '{tool_name}' not extracted from '{goal_description[:30]}...'.")
                return None
        self.logger("INFO", f"(Executor|_extract_args): LLM extracted args for tool '{tool_name}': {final_args_for_runner}")
        return final_args_for_runner

    def _document_and_register_new_tool(self, tool_path: str, original_goal_description: str, 
                                       tool_code: str, existing_tool_data: Optional[Dict[str,Any]] = None, 
                                       thread_id: Optional[str] = None, goals_list_ref: Optional[list] = None, 
                                       current_goal_id: Optional[str] = None):
        """
        Asks LLM to document a new/updated tool and saves it to the tool registry.
        Uses self.query_llm, self.logger, self.config, self.get_structured_tool_registry, 
        self.save_structured_tool_registry, and self._get_context_for_llm_from_thread.
        """
        # This logic is exactly from your script.
        tool_name = os.path.splitext(os.path.basename(tool_path))[0]
        self.logger("INFO", f"(Executor|_doc_tool): Documenting/Registering {'updated ' if existing_tool_data else 'new '}tool: {tool_name}")
        
        system_prompt_doc_tool = (
            "You are an AI tool documenter. From Python code, its goal, and thread history, provide: "
            "1. 'description': Concise one-sentence summary of the tool's purpose. "
            "2. 'capabilities': List of keywords or short phrases (max 5) describing its functions. "
            "3. 'args_info': For each `argparse` argument: 'name' (as in script), 'type' (e.g. str, int), 'required' (bool), 'description', 'default' (if any). "
            "Respond ONLY with a single JSON object: {\"description\": \"...\", \"capabilities\": [\"...\"], \"args_info\": [{...}]}."
        )
        thread_context_doc = self._get_context_for_llm_from_thread(thread_id, goals_list_ref if goals_list_ref else [], current_goal_id)
        prompt = (
            f"Original Goal for this tool: \"{original_goal_description}\"\n"
            f"Tool Script Code (first 3000 characters):\n```python\n{tool_code[:3000]}...\n```\n{thread_context_doc}\n\n"
            "JSON Response (with fields: description, capabilities, args_info):"
        )
        
        llm_response_str = self.query_llm(
            prompt,
            system_prompt_override=system_prompt_doc_tool,
            raw_output=True, # As in original
            timeout=self.config.get("doc_tool_llm_timeout", 180) # Using configured timeout
        )
        tool_doc_data: Dict[str, Any] = {} # Ensure type
        try:
            tool_doc_data = json.loads(llm_response_str)
            if not isinstance(tool_doc_data, dict) or "description" not in tool_doc_data:
                raise json.JSONDecodeError("LLM response for tool doc missing 'description' or not a dict.", llm_response_str, 0)
        except json.JSONDecodeError as e:
            self.logger("ERROR", f"(Executor|_doc_tool): Failed to parse LLM JSON for tool doc of '{tool_name}'. Resp: '{llm_response_str[:100]}'. Error: {e}")
            # Fallback data structure
            tool_doc_data["description"] = (existing_tool_data.get("description") if existing_tool_data and isinstance(existing_tool_data, dict) else f"Tool to help with: {original_goal_description}")
            tool_doc_data["capabilities"] = (existing_tool_data.get("capabilities", []) if existing_tool_data and isinstance(existing_tool_data, dict) else [])
            tool_doc_data["args_info"] = (existing_tool_data.get("args_info", []) if existing_tool_data and isinstance(existing_tool_data, dict) else [])

        registry_data_list = self.get_structured_tool_registry() # Use instance method
        entry_index = -1
        for i, t_entry in enumerate(registry_data_list): # Renamed t to t_entry
            if isinstance(t_entry, dict) and t_entry.get("name") == tool_name: # Check if t_entry is dict
                entry_index = i
                break
        
        current_time_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if entry_index != -1:
            # Ensure existing entry is a dict before updating
            if isinstance(registry_data_list[entry_index], dict):
                registry_data_list[entry_index].update({
                    "description": tool_doc_data.get("description", registry_data_list[entry_index].get("description")),
                    "capabilities": tool_doc_data.get("capabilities", registry_data_list[entry_index].get("capabilities")),
                    "args_info": tool_doc_data.get("args_info", registry_data_list[entry_index].get("args_info")),
                    "last_updated": current_time_iso,
                })
            else: # Should not happen if get_structured_tool_registry filters, but as safety:
                self.logger("ERROR", f"(Executor|_doc_tool): Found non-dict entry at index {entry_index} for tool '{tool_name}'. Cannot update.")
        else:
            new_entry = {
                "name": tool_name, "module_path": tool_path,
                "description": tool_doc_data.get("description", f"Tool for: {original_goal_description}"),
                "capabilities": tool_doc_data.get("capabilities", []),
                "args_info": tool_doc_data.get("args_info", []),
                "last_updated": current_time_iso, "last_run_time": None,
                "success_count": 0, "failure_count": 0, "total_runs": 0, "failure_details": []
            }
            registry_data_list.append(new_entry)
        
        self.save_structured_tool_registry(registry_data_list) # Use instance method
        self.logger("INFO", f"(Executor|_doc_tool): Tool '{tool_name}' information saved to registry.")

    def _determine_failure_category(self, error_message: str, tool_file_path: Optional[str] = None) -> str:
        """Determines a failure category based on an error message."""
        # This logic is exactly from your script.
        err_lower = str(error_message).lower()
        if "syntaxerror" in err_lower: return "SyntaxError"
        if "modulenotfounderror" in err_lower: return "ModuleNotFoundError"
        if "filenotfounderror" in err_lower: return "FileNotFoundError"
        if "attributeerror" in err_lower: return "AttributeError"
        if "typeerror" in err_lower: return "TypeError"
        if "valueerror" in err_lower: return "ValueError"
        if "indexerror" in err_lower: return "IndexError"
        if "keyerror" in err_lower: return "KeyError"
        if "nameerror" in err_lower: return "NameError"
        if "importerror" in err_lower: return "ImportError"
        if "lookuperror" in err_lower and "nltk.download" in err_lower: return "NLTKResourceMissing"
        if "argumenterror" in err_lower or ("systemexit" in err_lower and ("arguments are required" in err_lower or "invalid choice" in err_lower)): return "ArgumentError"
        if "no main() function found" in err_lower and tool_file_path: return "NoMainFunction"
        if "timed out" in err_lower or "timeouterror" in err_lower : return "TimeoutError"
        if tool_file_path and ("tool execution failed" in err_lower or "error" in err_lower): return "ToolExecutionFailure"
        if "compilation failed" in err_lower or "build failed" in err_lower: return "BuildCompilationError"
        if "core update" in err_lower and "staging failed" in err_lower: return "CoreUpdateStageFailure"
        if "core update" in err_lower and "apply failed" in err_lower: return "CoreUpdateApplyFailure"
        return "UnknownRuntimeError"

    # --- Self-Correction & Refinement Logic Methods ---
    def _formulate_and_add_corrective_goal(self, failed_goal_obj: Dict[str, Any], goals_list_ref: list) -> bool:
        """
        Formulates and adds a corrective goal based on a failed goal.
        Uses self.logger, self.query_llm, self.add_goal, self.config, 
        self._determine_failure_category, self._get_context_for_llm_from_thread, 
        and self._categorize_subtask.
        """
        # This logic is exactly from your script.
        error_message = failed_goal_obj.get("error", "Unknown error")
        original_goal_text = failed_goal_obj.get("goal", "Unknown original goal text")
        target_tool_p = failed_goal_obj.get("tool_file") or failed_goal_obj.get("target_file_for_processing")
        
        failure_cat = failed_goal_obj.get("failure_category") 
        if not failure_cat: # If not already set on the goal, determine it
            failure_cat = self._determine_failure_category(str(error_message), target_tool_p)
        failed_goal_obj["failure_category"] = failure_cat # Ensure it's stored

        self.logger("INFO", f"(ExecutorC|_formulate_corrective): GID ..{failed_goal_obj.get('goal_id','')[-6:]} ('{original_goal_text[:30]}...'), Cat: '{failure_cat}'. Formulating fix.")
        
        system_prompt_fixer = (
            "You are an AI error resolution specialist. Based on the error, original goal, potentially affected file, and thread history, "
            "generate a concise, actionable, and specific goal description to fix the problem. "
            "The new goal should clearly state what to modify, install, or research. "
            "If modifying a script, explicitly name the script path in the goal. "
            "Reference the original goal's THREAD_ID (e.g., 'related to thread ..xxxxxx')."
        )
        
        user_prompt_parts = [
            f"Original Failed Goal: \"{original_goal_text}\" (ID: ..{failed_goal_obj.get('goal_id','N/A')[-6:]}, Thread: ..{failed_goal_obj.get('thread_id','N/A')[-6:]})",
            f"Error Snippet (first few lines): \"{(str(error_message).splitlines()[0] if error_message else 'N/A')[:200]}\"",
            f"Determined Failure Category: {failure_cat}"
        ]
        if target_tool_p: 
            user_prompt_parts.append(f"Potentially Affected File Path: `{target_tool_p}`")
        
        thread_context_fix = self._get_context_for_llm_from_thread(failed_goal_obj.get("thread_id"), goals_list_ref, failed_goal_obj.get("goal_id"))
        if thread_context_fix: user_prompt_parts.append(thread_context_fix)
        
        corrective_goal_desc_llm = None

        # Pre-defined fixes from your script
        if failure_cat == "NLTKResourceMissing" and target_tool_p:
            r_match = re.search(r"Resource\s+[`'\"]([^`'\"]+)[`'\"]\s+not found|nltk\.download\(['\"]([^\"']+)['\"]\)", str(error_message), re.IGNORECASE)
            res_name = (r_match.group(1) or r_match.group(2)) if r_match else "corpus_placeholder_error"
            corrective_goal_desc_llm = f"Modify Python script `{target_tool_p}` to ensure NLTK resource '{res_name}' is downloaded (e.g., `import nltk; nltk.download('{res_name}')`) within a try-except LookupError block before it's used. Related to thread ..{failed_goal_obj.get('thread_id','N/A')[-6:]}."
        elif failure_cat == "ModuleNotFoundError":
            m_match = re.search(r"No module named '([^']*)'", str(error_message))
            mod_name = m_match.group(1) if m_match else "missing_module_error"
            corrective_goal_desc_llm = f"Install the Python package '{mod_name}' (e.g., using a command like 'pip install {mod_name}') for the system. Related to thread ..{failed_goal_obj.get('thread_id','N/A')[-6:]}."
            if target_tool_p:
                corrective_goal_desc_llm += f" Also, review imports in `{target_tool_p}` if '{mod_name}' is a custom module."
        elif failure_cat == "NoMainFunction" and target_tool_p:
             corrective_goal_desc_llm = f"Add a `main()` function to Python script `{target_tool_p}` and ensure it's called (e.g., under `if __name__ == '__main__':`). This function should encapsulate the script's primary logic. Related to thread ..{failed_goal_obj.get('thread_id','N/A')[-6:]}."
        elif failure_cat == "ArgumentError" and target_tool_p:
            corrective_goal_desc_llm = f"Review and correct argparse setup or argument handling in script `{target_tool_p}` to resolve argument-related errors (e.g., missing arguments, invalid choices, SystemExit code 2). Original error hint: {str(error_message).splitlines()[0][:100]}. Related to thread ..{failed_goal_obj.get('thread_id','N/A')[-6:]}."
        elif failure_cat == "RefinementTargetMissing":
            corrective_goal_desc_llm = None # Force LLM
            user_prompt_parts.append("Note: This failed goal was a refinement task. The target file path was missing. The parent goal might need re-evaluation to determine the correct file for refinement, or the previous step that created this refinement goal needs fixing." if failed_goal_obj.get("parent_goal_id") else "Note: This failed goal was a refinement task, but the target file path was missing. The AI needs to determine the intended target file and formulate a corrected refinement goal.")

        if not corrective_goal_desc_llm: # If not a predefined fix, use LLM
            user_prompt_for_llm_fix = "\n".join(user_prompt_parts) + "\n\nSuggest a concise corrective goal text (e.g., 'Modify script X to fix Y due to Z error related to thread TTT' or 'Install package P for thread TTT', or 'Research how to resolve error Z for script X'):"
            llm_fix_text = self.query_llm(user_prompt_for_llm_fix, system_prompt_override=system_prompt_fixer, timeout=self.config.get("correction_llm_timeout", 240))
            if llm_fix_text and not llm_fix_text.startswith("[Error:") and len(llm_fix_text.strip()) > 10:
                corrective_goal_desc_llm = llm_fix_text.strip()
                if target_tool_p and failure_cat not in ["ModuleNotFoundError", "NLTKResourceMissing", "CommandParseFailure", "RefinementTargetMissing"] and \
                   target_tool_p.lower() not in corrective_goal_desc_llm.lower() and \
                   any(kw in corrective_goal_desc_llm.lower() for kw in ["modify", "fix", "correct", "update", "add to", "change", "debug", "refactor", "implement", "script"]):
                     corrective_goal_desc_llm = f"Modify script `{target_tool_p}`: {corrective_goal_desc_llm}"
                if f"thread ..{failed_goal_obj.get('thread_id','N/A')[-6:]}" not in corrective_goal_desc_llm.lower():
                     corrective_goal_desc_llm += f" (Context: Thread ..{failed_goal_obj.get('thread_id','N/A')[-6:]})"
            else:
                self.logger("WARNING", f"(ExecutorC|_formulate_corrective): LLM failed to generate valid corrective goal for GID ..{failed_goal_obj.get('goal_id','')[-6:]}. LLM raw: '{llm_fix_text[:100]}'")

        if corrective_goal_desc_llm:
            corrective_cat = self._categorize_subtask(corrective_goal_desc_llm)
            new_target_file = None
            if corrective_cat == CAT_REFINEMENT:
                if target_tool_p: new_target_file = target_tool_p
                else:
                    file_path_match_in_new_goal = re.search(r"`(tools/[^`]+?\.(?:py|txt|json|md))`", corrective_goal_desc_llm) # From your original
                    if file_path_match_in_new_goal:
                        new_target_file = file_path_match_in_new_goal.group(1)
                        self.logger("INFO", f"(ExecutorC|_formulate_corrective): Extracted target file '{new_target_file}' from corrective goal description for refinement.")
                    else:
                        self.logger("WARNING", f"(ExecutorC|_formulate_corrective): Corrective goal is CAT_REFINEMENT but no target file could be determined: '{corrective_goal_desc_llm}'.")

            success, new_goal_id = self.add_goal(
                goals_list_ref, 
                description=corrective_goal_desc_llm, 
                source=SOURCE_SELF_CORRECTION,
                approved_by="AI_System", 
                is_subtask=True,
                parent_goal_obj=next((g for g in goals_list_ref if g.get("goal_id") == failed_goal_obj.get("parent_goal_id")), None),
                related_to_failed_goal_obj=failed_goal_obj,
                thread_id_param=failed_goal_obj.get("thread_id"),
                priority=PRIORITY_HIGH, 
                subtask_category_override=corrective_cat,
                target_file_for_processing=new_target_file
            )
            if success:
                self._update_goal_status_and_history(failed_goal_obj, STATUS_AWAITING_CORRECTION, f"Self-correction goal GID ..{new_goal_id[-6:] if new_goal_id else 'N/A'} added: {corrective_goal_desc_llm[:70]}...")
                failed_goal_obj["self_correction_attempts"] = failed_goal_obj.get("self_correction_attempts", 0) + 1
                return True
            else:
                self.logger("WARNING", f"(ExecutorC|_formulate_corrective): Failed to add corrective goal for GID ..{failed_goal_obj.get('goal_id','')[-6:]} (likely duplicate or error).")
        else:
            self.logger("WARNING", f"(ExecutorC|_formulate_corrective): Unable to formulate self-correction for GID ..{failed_goal_obj.get('goal_id','')[-6:]}. Error: {str(error_message).splitlines()[0][:70]}")
            self._update_goal_status_and_history(failed_goal_obj, STATUS_FAILED_UNCLEAR, "Failed to formulate a self-correction goal via LLM or predefined logic.")
            failed_goal_obj["self_correction_attempts"] = failed_goal_obj.get("self_correction_attempts", 0) + 1
        return False

    def _formulate_and_add_refinement_goal(self, completed_goal_obj: Dict[str, Any], tool_run_result: Dict[str, Any], goals_list_ref: list) -> bool:
        """
        Formulates and adds a goal to refine a tool if its output suggests a need.
        Uses self.logger, self.query_llm, self.add_goal, self.config, and self._get_context_for_llm_from_thread.
        """
        # This logic is exactly from your script.
        tool_file_path = completed_goal_obj.get("tool_file")
        if not tool_file_path or not tool_run_result or not os.path.exists(tool_file_path):
            return False
        
        output_sample = str(tool_run_result.get("output", "")).lower()[:300]
        error_sample = str(tool_run_result.get("error", "")).lower()[:300]
        refinement_trigger_reason = None
        if "warning:" in output_sample or "deprecated" in output_sample:
            refinement_trigger_reason = "Tool output contains warnings or mentions deprecated features."
        elif "warning:" in error_sample or "deprecated" in error_sample:
            refinement_trigger_reason = "Tool's error stream has warnings or mentions deprecated features (check for actual errors vs. just warnings)."
        # Example from your original:
        # elif len(str(tool_run_result.get("output",""))) > 5000: refinement_trigger_reason = "Tool output is very long, consider making it more concise or providing a summary option."

        if not refinement_trigger_reason:
            return False
            
        self.logger("INFO", f"(ExecutorC|_formulate_refine): Potential refinement for '{tool_file_path}' (Thread ..{completed_goal_obj.get('thread_id','N/A')[-6:]}): {refinement_trigger_reason}")
        
        system_prompt_refiner = (
            "You are an AI code quality analyst. Based on a tool's execution details, thread history, and a reason for refinement, "
            "suggest a specific, actionable goal to improve the tool's code. Focus on clarity, fixing warnings, or minor optimizations. "
            "Example: 'Refactor tool X on thread TTT to use the updated Y library call instead of the deprecated Z function'."
        )
        thread_context_refine = self._get_context_for_llm_from_thread(completed_goal_obj.get("thread_id"), goals_list_ref, completed_goal_obj.get("goal_id"))
        user_prompt_refine_llm = (
            f"Tool Path: `{tool_file_path}` (Thread ..{completed_goal_obj.get('thread_id','N/A')[-6:]})\n"
            f"Original Goal for this tool: \"{completed_goal_obj.get('goal','N/A')[:80]}...\"\n"
            f"Reason for Refinement: \"{refinement_trigger_reason}\"\n"
            f"Tool Output Snippet (if any): ```{output_sample if output_sample else 'N/A'}```\n"
            f"Tool Error Stream Snippet (if any): ```{error_sample if error_sample else 'N/A'}```\n{thread_context_refine}\n\n"
            "Suggest a specific goal text to refine this tool's code (e.g., 'Refine tool X to address Y for thread TTT'):"
        )
        
        refinement_goal_text_llm = self.query_llm(
            user_prompt_refine_llm,
            system_prompt_override=system_prompt_refiner,
            timeout=self.config.get("refinement_llm_timeout", 180) # Using configured timeout
        )

        if refinement_goal_text_llm and not refinement_goal_text_llm.startswith("[Error:"):
            final_refinement_goal_text = f"Refine tool `{os.path.basename(tool_file_path)}`: {refinement_goal_text_llm.strip()}"
            if "thread" not in final_refinement_goal_text.lower(): # Ensure thread context mentioned
                final_refinement_goal_text += f" (Context: Thread ..{completed_goal_obj.get('thread_id','N/A')[-6:]})"

            parent_obj_for_refine = next((g for g in goals_list_ref if g.get("goal_id") == completed_goal_obj.get("parent_goal_id")), None)
            success, _ = self.add_goal(
                goals_list_ref, description=final_refinement_goal_text, source=SOURCE_AUTO_REFINEMENT,
                approved_by="AI_System", is_subtask=True, parent_goal_obj=parent_obj_for_refine,
                thread_id_param=completed_goal_obj.get("thread_id"), priority=PRIORITY_LOW,
                subtask_category_override=CAT_REFINEMENT, target_file_for_processing=tool_file_path
            )
            if success:
                self.logger("INFO", f"(ExecutorC|_formulate_refine): Added auto-refinement goal for '{tool_file_path}'.")
                return True
        else:
            self.logger("WARNING", f"(ExecutorC|_formulate_refine): LLM failed to generate refinement goal for '{tool_file_path}'. LLM Resp: {refinement_goal_text_llm[:100]}")
        return False

    def is_single_tool_creation_goal(self, goal_text: str, thread_context_str: str) -> bool:
        """
        Uses an LLM to determine if a goal is primarily about creating a single tool.
        Uses self.logger, self.query_llm, self.config.
        """
        # This logic is exactly from your script.
        system_prompt_classifier = (
            "You are an AI assistant that classifies user goals. Your task is to determine if the given goal "
            "is primarily focused on creating a single, self-contained software tool or Python script that can likely be "
            "generated in one main coding step. Consider if the goal implies a direct output of a runnable script."
            "\nRespond with only 'YES' or 'NO'."
        )
        
        max_context_len = self.config.get("tool_classification_max_context_len", 500) # From your original config defaults
        truncated_thread_context = thread_context_str[:max_context_len] + ('...' if len(thread_context_str) > max_context_len else '')

        prompt_for_llm = (
            f"Analyze the following user goal and its surrounding context:\n\n"
            f"User Goal: \"{goal_text}\"\n\n"
            f"Recent Thread Context (if any):\n{truncated_thread_context if truncated_thread_context else '(No specific thread context provided for this goal classification)'}\n\n"
            f"Based on this, is the User Goal primarily about the creation of a single, self-contained software tool or Python script, "
            f"which could reasonably be generated by an AI in one primary coding effort (even if complex)? "
            f"For example, 'create a tool that searches the web' is YES. 'Refactor the entire codebase' is NO. 'Plan a software project' is NO. "
            f"'Write a python script to parse a CSV and output JSON' is YES."
            f"\n\nRespond with only the word YES or the word NO."
        )

        try:
            response = self.query_llm(
                prompt_text=prompt_for_llm,
                system_prompt_override=system_prompt_classifier,
                raw_output=False, 
                timeout=self.config.get("tool_classification_llm_timeout", 90) # Using configured timeout
            )
            
            response_clean = response.strip().upper()
            self.logger("DEBUG", f"(ExecutorC|is_single_tool): Goal: '{goal_text[:50]}...' LLM_Response: '{response_clean}'")

            if response_clean == "YES":
                return True
            elif response_clean == "NO":
                return False
            else:
                self.logger("WARNING", f"(ExecutorC|is_single_tool): Ambiguous LLM response ('{response_clean}'). Defaulting to NO.")
                return False
                
        except Exception as e:
            self.logger("ERROR", f"(ExecutorC|is_single_tool): LLM call failed: {e}. Defaulting to NO.")
            return False

    # Continuing Self-Aware/executor.py
# Make sure this is INSIDE the Executor class definition,
# after the is_single_tool_creation_goal method.

    # --- Core Goal Processing Methods ---
    def process_single_goal(self, goal_obj_to_process: Dict[str, Any], all_goals_list_ref: list):
        """
        Processes a single, specific goal object. Contains the core execution logic.
        Updates the goal_obj_to_process in-place and saves all_goals_list_ref via self.save_goals().
        Uses self.tool_builder, self.tool_runner, self.notifier, and other internal helper methods.
        """
        # Ensure dependencies are met (ToolBuilder, ToolRunner)
        if not self.tool_builder or not hasattr(self.tool_builder, 'build_tool'):
            self.logger("ERROR", "(ExecutorC|process_single) ToolBuilder unavailable. Cannot process generation/refinement goals.")
            if goal_obj_to_process.get("subtask_category") in [CAT_CODE_GENERATION_TOOL, CAT_CODE_GENERATION_SNIPPET, CAT_REFINEMENT]:
                self._update_goal_status_and_history(goal_obj_to_process, STATUS_BUILD_FAILED, "System Error: ToolBuilder component is not available.")
                goal_obj_to_process["error"] = "ToolBuilder component not available."
                goal_obj_to_process["failure_category"] = "ComponentMissing"
                self.save_goals(all_goals_list_ref)
            return
        if not self.tool_runner or not hasattr(self.tool_runner, 'run_tool_safely'):
            self.logger("ERROR", "(ExecutorC|process_single) ToolRunner unavailable. Cannot execute tools.")
            if goal_obj_to_process.get("tool_file") or goal_obj_to_process.get("subtask_category") == CAT_USE_EXISTING_TOOL:
                self._update_goal_status_and_history(goal_obj_to_process, STATUS_EXECUTED_WITH_ERRORS, "System Error: ToolRunner component is not available.")
                goal_obj_to_process["error"] = "ToolRunner component not available."
                goal_obj_to_process["failure_category"] = "ComponentMissing"
                self.save_goals(all_goals_list_ref)
            return

        # Logic from your original process_single_goal, meticulously adapted
        if not goal_obj_to_process or goal_obj_to_process.get("status") not in [STATUS_PENDING, STATUS_AWAITING_CORRECTION]:
            self.logger("DEBUG", f"(ExecutorC|process_single): Goal GID ..{goal_obj_to_process.get('goal_id', 'N/A')[-6:]} not processable (Status: {goal_obj_to_process.get('status')}).")
            return

        current_goal_text = goal_obj_to_process["goal"]
        current_goal_id = goal_obj_to_process.get("goal_id", "unknown_gid_in_process")
        current_thread_id = goal_obj_to_process.get("thread_id", "unknown_tid_in_process")
        any_goal_status_changed_this_cycle = False

        self.logger("INFO", f"(ExecutorC|process_single): Processing P:{goal_obj_to_process.get('priority')} GID:{current_goal_id[-6:]} '{current_goal_text[:30]}...' (Thread: ..{current_thread_id[-6:]})")
        
        if goal_obj_to_process.get("status") == STATUS_PENDING:
            self._update_goal_status_and_history(goal_obj_to_process, STATUS_APPROVED, "Auto-approved for processing by executor.")
            any_goal_status_changed_this_cycle = True
            
        execution_successful = False
        run_result_for_potential_refinement = None

        subtask_cat = goal_obj_to_process.get("subtask_category") or self._categorize_subtask(current_goal_text)
        if not goal_obj_to_process.get("subtask_category"):
            goal_obj_to_process["subtask_category"] = subtask_cat
            any_goal_status_changed_this_cycle = True
        self.logger("DEBUG", f"(ExecutorC|process_single): Task '{current_goal_text[:30]}...' (Cat: {subtask_cat})")

        all_tools_registry = self.get_structured_tool_registry()
        selected_existing_tool_data = None
        
        if subtask_cat not in [CAT_CODE_GENERATION_TOOL, CAT_CODE_GENERATION_SNIPPET, CAT_COMMAND_EXECUTION, CAT_INFORMATION_GATHERING, CAT_REFINEMENT, CAT_CORE_FILE_UPDATE] and all_tools_registry:
            self.logger("DEBUG", f"(ExecutorC|process_single): Category '{subtask_cat}' allows checking for existing tools.")
            selected_existing_tool_data = self._ask_llm_for_tool_selection(current_goal_text, all_tools_registry, current_thread_id, all_goals_list_ref, current_goal_id)

        if selected_existing_tool_data:
            # ... (Logic for executing an existing tool from your script)
            tool_module_p = selected_existing_tool_data.get("module_path")
            tool_arguments_list = self._extract_arguments_for_tool(current_goal_text, selected_existing_tool_data, current_thread_id, all_goals_list_ref, current_goal_id)
            if tool_module_p and tool_arguments_list is not None:
                self.logger("INFO", f"(ExecutorC|process_single): Running existing tool '{tool_module_p}' with args: {tool_arguments_list}")
                run_tool_result = self.tool_runner.run_tool_safely(tool_module_p, tool_args=tool_arguments_list)
                run_result_for_potential_refinement = run_tool_result
                goal_obj_to_process['execution_result'] = run_tool_result
                goal_obj_to_process['used_existing_tool'] = selected_existing_tool_data.get('name', 'UnknownTool')
                any_goal_status_changed_this_cycle = True
                if run_tool_result["status"] == "success": execution_successful = True
                else:
                    goal_obj_to_process['error'] = run_tool_result.get("error", f"Existing tool '{selected_existing_tool_data.get('name')}' execution failed.")
                    goal_obj_to_process['failure_category'] = self._determine_failure_category(goal_obj_to_process['error'], tool_module_p)
            else:
                self.logger("WARNING", f"(ExecutorC|process_single): Could not get args/path for selected tool '{selected_existing_tool_data.get('name')}'. Will attempt build if applicable.")
                goal_obj_to_process['error'] = f"Failed to determine path or arguments for LLM-selected tool '{selected_existing_tool_data.get('name')}'."
                goal_obj_to_process['failure_category'] = "ToolArgumentExtractionError"
                any_goal_status_changed_this_cycle = True
                selected_existing_tool_data = None # Fall through to build
        
        if not execution_successful and not selected_existing_tool_data:
            try:
                if subtask_cat == CAT_COMMAND_EXECUTION:
                    # --- CAT_COMMAND_EXECUTION --- (Copied verbatim from your original file)
                    command_parts_list = []
                    current_goal_text_lower = current_goal_text.lower()
                    install_match = re.search(r"(?:install|ensure\s+packages?|add\s+packages?)\s+(.+)", current_goal_text_lower)
                    if install_match:
                        package_names_str = install_match.group(1).strip()
                        packages_to_install_raw = []
                        if ',' in package_names_str: packages_to_install_raw = [p.strip() for p in package_names_str.split(',')]
                        elif ' and ' in package_names_str: packages_to_install_raw = [p.strip() for p in package_names_str.split(' and ')]
                        else: packages_to_install_raw = package_names_str.split()
                        cleaned_packages = []
                        ignore_keywords = ["python", "package", "packages", "module", "modules", "library", "libraries", "etc", "other", "necessary", "related", "dependencies"]
                        for pkg_name_candidate in packages_to_install_raw:
                            pkg_name_candidate = pkg_name_candidate.strip().split('==')[0].split('>')[0].split('<')[0].strip()
                            if pkg_name_candidate and pkg_name_candidate.lower() not in ignore_keywords and len(pkg_name_candidate) > 1 :
                                cleaned_packages.append(pkg_name_candidate)
                        if cleaned_packages: command_parts_list = [sys.executable, "-m", "pip", "install"] + list(set(cleaned_packages))
                        else: self.logger("WARNING", f"(ExecutorC|process_single): Could not identify valid package names from '{package_names_str}' for goal: '{current_goal_text[:50]}...'")
                    if not command_parts_list:
                        pip_match = re.search(r"(?:pip\s+install)\s+([\w\-=<>.~\[\]]+(?:==[\w\.\*]+)?)", current_goal_text, re.IGNORECASE)
                        if pip_match:
                            package_name_str = pip_match.group(1)
                            command_parts_list = [sys.executable, "-m", "pip", "install", package_name_str]
                    if command_parts_list:
                        command_to_run_str = " ".join(command_parts_list)
                        self.logger("INFO", f"(ExecutorC|process_single): Executing command: {command_to_run_str}")
                        env = os.environ.copy() 
                        proc_result = subprocess.run(command_parts_list, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace', timeout=360, env=env)
                        goal_obj_to_process["execution_result"] = {"command": command_to_run_str, "returncode": proc_result.returncode, "stdout": proc_result.stdout[-2000:], "stderr": proc_result.stderr[-2000:]}
                        if proc_result.returncode == 0:
                            execution_successful = True
                            self.logger("INFO", f"(ExecutorC|process_single): Command '{command_to_run_str}' executed successfully.")
                        else:
                            err_output = (proc_result.stderr[-1000:] if proc_result.stderr else "") + ("\n" + proc_result.stdout[-1000:] if proc_result.stderr and proc_result.stdout else proc_result.stdout[-1000:])
                            goal_obj_to_process["error"] = err_output.strip() or f"Command failed with return code {proc_result.returncode}."
                            goal_obj_to_process["failure_category"] = "CommandExecutionFailure"
                            self.logger("ERROR", f"(ExecutorC|process_single): Command '{command_to_run_str}' failed. Stderr: {proc_result.stderr[:300]}... Stdout: {proc_result.stdout[:300]}...")
                    else:
                        goal_obj_to_process["error"] = "Could not parse a valid command (e.g., pip install) from subtask description."
                        goal_obj_to_process["failure_category"] = "CommandParseFailure"
                    any_goal_status_changed_this_cycle = True
                elif subtask_cat == CAT_INFORMATION_GATHERING:
                    # --- CAT_INFORMATION_GATHERING --- (Copied verbatim, using self.query_llm, self.logger, self._get_context_for_llm_from_thread)
                    llm_info_user_prompt = f"Provide detailed and factual information for the following query, considering its context within Thread ..{current_thread_id[-6:] if current_thread_id else 'N/A'}: '{current_goal_text}'. Output only the answer text directly."
                    llm_info_system_prompt = "You are an AI assistant specialized in providing factual information concisely, aware of surrounding goal context."
                    thread_context_info = self._get_context_for_llm_from_thread(current_thread_id, all_goals_list_ref, current_goal_id, max_entries=1)
                    self.logger("INFO", f"(ExecutorC|process_single): Gathering information for '{current_goal_text[:30]}...' using LLM.")
                    information_content_str = self.query_llm(f"{thread_context_info}\n\n{llm_info_user_prompt}", system_prompt_override=llm_info_system_prompt, timeout=self.config.get("info_llm_timeout", 240))
                    if information_content_str and not information_content_str.startswith("[Error:"):
                        goal_obj_to_process["execution_result"] = {"status": "success", "information_gathered": information_content_str}
                        execution_successful = True
                    else:
                        goal_obj_to_process["error"] = f"LLM information gathering failed or returned error: {information_content_str[:100]}"
                        goal_obj_to_process["failure_category"] = "InformationGatheringLLMFailure"
                    any_goal_status_changed_this_cycle = True
                elif subtask_cat == CAT_CODE_GENERATION_SNIPPET:
                    # --- CAT_CODE_GENERATION_SNIPPET --- (Copied verbatim, using self.tool_builder.build_tool, self.tool_runner.run_tool_safely, self._document_and_register_new_tool, self.logger)
                    code_match = re.search(r"code:\s*(.*)", current_goal_text, re.IGNORECASE | re.DOTALL)
                    code_to_execute_str = code_match.group(1).strip() if code_match else current_goal_text
                    snippet_tool_name_str = "_".join("".join(c if c.isalnum() else '_' for c in current_goal_text[:25]).split()).lower() + f"_exec_snippet_{current_thread_id[-4:] if current_thread_id else 'glob'}"
                    build_tool_description_for_snippet = f"Create a Python script (for Thread ..{current_thread_id[-6:] if current_thread_id else 'N/A'}) whose main function executes the following Python code block directly: \n```python\n{code_to_execute_str}\n```\nThe script should print any output from this block. It should not require any command-line arguments."
                    self.logger("INFO", f"(ExecutorC|process_single): Generating temporary tool for snippet: '{current_goal_text[:30]}...'")
                    tool_file_p = self.tool_builder.build_tool(build_tool_description_for_snippet, snippet_tool_name_str, current_thread_id, all_goals_list_ref, current_goal_id)
                    goal_obj_to_process['tool_file'] = tool_file_p
                    run_tool_res = self.tool_runner.run_tool_safely(tool_file_p)
                    run_result_for_potential_refinement = run_tool_res
                    goal_obj_to_process['execution_result'] = run_tool_res
                    if run_tool_res["status"] == "success":
                        execution_successful = True
                        with open(tool_file_p, 'r', encoding='utf-8') as f_code_content: tool_code_text = f_code_content.read()
                        self._document_and_register_new_tool(tool_file_p, f"Wrapper for snippet: {current_goal_text}", tool_code_text, thread_id=current_thread_id, goals_list_ref=all_goals_list_ref, current_goal_id=current_goal_id)
                    else:
                        goal_obj_to_process['error'] = run_tool_res.get("error", "Snippet tool execution failed.")
                        goal_obj_to_process['failure_category'] = self._determine_failure_category(goal_obj_to_process['error'], tool_file_p)
                    any_goal_status_changed_this_cycle = True
                elif subtask_cat == CAT_REFINEMENT:
                    # --- CAT_REFINEMENT --- (Copied verbatim)
                    target_tool_file_path_refine = goal_obj_to_process.get("target_file_for_processing")
                    if not target_tool_file_path_refine or not os.path.exists(target_tool_file_path_refine):
                        goal_obj_to_process["error"] = f"Target file for refinement '{target_tool_file_path_refine}' not found or not specified."
                        goal_obj_to_process["failure_category"] = "RefinementTargetMissing"
                    else:
                        self.logger("INFO", f"(ExecutorC|process_single): Refining tool '{target_tool_file_path_refine}' based on goal: '{current_goal_text[:30]}...' (Thread ..{current_thread_id[-6:] if current_thread_id else 'N/A'})")
                        existing_tool_code_refine = ""
                        try:
                            with open(target_tool_file_path_refine, 'r', encoding='utf-8') as f_existing_code: existing_tool_code_refine = f_existing_code.read()
                        except IOError as e_io_refine:
                            self.logger("WARNING", f"(ExecutorC|process_single): Could not read {target_tool_file_path_refine} for refinement context: {e_io_refine}")
                        tool_name_from_path_refine = os.path.splitext(os.path.basename(target_tool_file_path_refine))[0]
                        refinement_build_description = (f"Refinement Task (for Thread ..{current_thread_id[-6:] if current_thread_id else 'N/A'}): {current_goal_text}\n\nContext - Existing code of `{target_tool_file_path_refine}` (first 5000 chars):\n```python\n{existing_tool_code_refine[:5000]}\n```\n\nPlease provide the FULL MODIFIED Python script for `{target_tool_file_path_refine}` incorporating the refinement. Ensure all critical requirements (argparse, no input(), main()) are met in the modified script.")
                        modified_tool_path_refine = self.tool_builder.build_tool(refinement_build_description, tool_name_from_path_refine, current_thread_id, all_goals_list_ref, current_goal_id)
                        goal_obj_to_process['tool_file'] = modified_tool_path_refine
                        if os.path.exists(modified_tool_path_refine):
                            with open(modified_tool_path_refine, 'r', encoding='utf-8') as f_tool_code_check: modified_tool_code_text = f_tool_code_check.read()
                            if "# TOOL GENERATION FAILED" not in modified_tool_code_text and "empty code" not in modified_tool_code_text:
                                execution_successful = True
                                goal_obj_to_process['execution_result'] = {"status": "success", "output": f"Tool {modified_tool_path_refine} (re)generated based on refinement goal."}
                                existing_reg_data_refine = next((t for t in all_tools_registry if isinstance(t,dict) and t.get("name") == tool_name_from_path_refine), None) # Ensure t is dict
                                self._document_and_register_new_tool(modified_tool_path_refine, f"Refined version for goal: {current_goal_text}", modified_tool_code_text, existing_tool_data=existing_reg_data_refine, thread_id=current_thread_id, goals_list_ref=all_goals_list_ref, current_goal_id=current_goal_id)
                                self.logger("INFO", f"(ExecutorC|process_single): Test running refined tool '{modified_tool_path_refine}'")
                                test_run_refined_result = self.tool_runner.run_tool_safely(modified_tool_path_refine)
                                goal_obj_to_process.setdefault('execution_result',{})['refinement_test_run'] = test_run_refined_result
                                if test_run_refined_result["status"] != "success":
                                    self.logger("WARNING", f"Refined tool '{modified_tool_path_refine}' FAILED its post-refinement test run: {test_run_refined_result.get('error','')[:100]}")
                            else:
                                goal_obj_to_process['error'] = f"Tool refinement for {modified_tool_path_refine} produced invalid placeholder code."
                                goal_obj_to_process['failure_category'] = "RefinementGenerationFailure"
                        else:
                            goal_obj_to_process['error'] = f"Tool refinement build failed: {modified_tool_path_refine} was not created/found."
                            goal_obj_to_process['failure_category'] = "RefinementGenerationFailure"
                    any_goal_status_changed_this_cycle = True
                elif subtask_cat == CAT_CODE_GENERATION_TOOL:
                    # --- CAT_CODE_GENERATION_TOOL --- (Copied verbatim)
                    tool_name_base_str = "".join(c if c.isalnum() else '_' for c in current_goal_text)
                    tool_name_str = "_".join(tool_name_base_str.split()).lower()[:40] + f"_{current_thread_id[-6:] if current_thread_id else uuid.uuid4().hex[:4]}"
                    is_self_correction_for_this_tool = (goal_obj_to_process.get("source") == SOURCE_SELF_CORRECTION and goal_obj_to_process.get("target_file_for_processing") and os.path.basename(goal_obj_to_process.get("target_file_for_processing","")) == f"{tool_name_str}.py")
                    tool_description_for_build = f"Goal for Thread ..{current_thread_id[-6:] if current_thread_id else 'N/A'}: {current_goal_text}"
                    if is_self_correction_for_this_tool: self.logger("INFO", f"(ExecutorC|process_single): Self-correcting/regenerating tool '{tool_name_str}' for: '{tool_description_for_build[:30]}...'")
                    else: self.logger("INFO", f"(ExecutorC|process_single): Building NEW tool '{tool_name_str}' for: '{tool_description_for_build[:30]}...'")
                    tool_file_p = self.tool_builder.build_tool(tool_description_for_build, tool_name_str, current_thread_id, all_goals_list_ref, current_goal_id)
                    goal_obj_to_process['tool_file'] = tool_file_p
                    if os.path.exists(tool_file_p):
                        with open(tool_file_p, 'r', encoding='utf-8') as f_tool_code_check_gen: tool_code_text_gen = f_tool_code_check_gen.read()
                        if "# TOOL GENERATION FAILED" not in tool_code_text_gen and "empty code" not in tool_code_text_gen:
                            if is_self_correction_for_this_tool:
                                execution_successful = True 
                                goal_obj_to_process['execution_result'] = {"status": "success", "output": f"Tool {tool_file_p} (re)generated for self-correction."}
                                existing_reg_data_sc = next((t for t in all_tools_registry if isinstance(t,dict) and t.get("name") == tool_name_str), None) # Ensure t is dict
                                self._document_and_register_new_tool(tool_file_p, f"Self-corrected version for: {goal_obj_to_process.get('related_to_failed_goal_text', current_goal_text)}", tool_code_text_gen, existing_tool_data=existing_reg_data_sc, thread_id=current_thread_id, goals_list_ref=all_goals_list_ref, current_goal_id=current_goal_id)
                            else: # New tool, needs initial run
                                self.logger("INFO", f"(ExecutorC|process_single): Performing initial run of newly built tool '{tool_file_p}'")
                                run_tool_res_new = self.tool_runner.run_tool_safely(tool_file_p)
                                run_result_for_potential_refinement = run_tool_res_new
                                goal_obj_to_process['execution_result'] = run_tool_res_new
                                if run_tool_res_new["status"] == "success":
                                    execution_successful = True
                                    self._document_and_register_new_tool(tool_file_p, current_goal_text, tool_code_text_gen, thread_id=current_thread_id, goals_list_ref=all_goals_list_ref, current_goal_id=current_goal_id)
                                else:
                                    goal_obj_to_process['error'] = run_tool_res_new.get("error", f"New tool '{tool_name_str}' initial run failed.")
                                    goal_obj_to_process['failure_category'] = self._determine_failure_category(goal_obj_to_process['error'], tool_file_p)
                        else:
                            goal_obj_to_process['error'] = f"Tool (re)generation for {tool_file_p} produced invalid placeholder code."
                            goal_obj_to_process['failure_category'] = "ToolGenerationLLMError" if not is_self_correction_for_this_tool else "SelfCorrectionBuildLLMError"
                    else:
                        goal_obj_to_process['error'] = f"Tool generation failed: {tool_file_p} not found after build attempt."
                        goal_obj_to_process['failure_category'] = "ToolGenerationFileSystemError" if not is_self_correction_for_this_tool else "SelfCorrectionBuildFileSystemError"
                    any_goal_status_changed_this_cycle = True
                elif subtask_cat == CAT_CORE_FILE_UPDATE:
                    # --- CAT_CORE_FILE_UPDATE --- (Copied verbatim, importing core_update_tool locally)
                    target_core_file = goal_obj_to_process.get("target_file_for_processing")
                    new_core_code = goal_obj_to_process.get("code_content_for_core_update")
                    if not target_core_file or not new_core_code:
                        goal_obj_to_process["error"] = "Core file update goal missing target file or new code content."
                        goal_obj_to_process["failure_category"] = "CoreUpdateInputMissing"
                    else:
                        self.logger("INFO", f"(ExecutorC|process_single) Attempting core file update for: {target_core_file} (Thread ..{current_thread_id[-6:] if current_thread_id else 'N/A'})")
                        try:
                            from core_update_tool import update_core_file as stage_core_update, apply_update as apply_core_update_final
                            staged_ok, stage_msg = stage_core_update(new_core_code, target_core_file, approved_by="AI_Executor")
                            goal_obj_to_process['execution_result'] = {"stage_status": "success" if staged_ok else "failure", "message": stage_msg}
                            if staged_ok:
                                self.logger("INFO", f"(ExecutorC|process_single) Core update staged for {target_core_file}. Applying...")
                                apply_core_update_final(target_core_file, approved_by="AI_Executor_AutoApply")
                                goal_obj_to_process.setdefault('execution_result',{})["apply_status"] = "success"
                                execution_successful = True
                            else:
                                goal_obj_to_process["error"] = f"Core update staging failed for {target_core_file}: {stage_msg}"
                                goal_obj_to_process["failure_category"] = "CoreUpdateStageFailure"
                        except ImportError:
                            goal_obj_to_process["error"] = "core_update_tool.py not found or cannot be imported."
                            goal_obj_to_process["failure_category"] = "CoreUpdateToolMissing"
                        except Exception as e_apply_core:
                            goal_obj_to_process["error"] = f"Failed to apply core update to {target_core_file}: {e_apply_core}"
                            goal_obj_to_process["failure_category"] = "CoreUpdateApplyFailure"
                            goal_obj_to_process.setdefault('execution_result',{})["apply_status"] = "failure"
                            goal_obj_to_process['execution_result']["apply_error"] = str(e_apply_core)
                    any_goal_status_changed_this_cycle = True
                else: 
                    goal_obj_to_process["error"] = f"Unhandled subtask category: {subtask_cat}"
                    goal_obj_to_process["failure_category"] = "UnknownSubtaskCategory"
                    any_goal_status_changed_this_cycle = True
            except Exception as e_process_task: # Main try-except catch for task processing
                self.logger("CRITICAL", f"(ExecutorC|process_single): Exception while processing GID ..{current_goal_id[-6:]} '{current_goal_text[:30]}...': {e_process_task}\n{traceback.format_exc()}")
                goal_obj_to_process['error'] = str(e_process_task)
                goal_obj_to_process["failure_category"] = "GenericProcessingError"
                execution_successful = False
                any_goal_status_changed_this_cycle = True

        # --- Post-Execution Logic for this single goal (Copied verbatim, using self.notifier, self._formulate_and_add_refinement_goal, self._formulate_and_add_corrective_goal) ---
        if execution_successful:
            self._update_goal_status_and_history(goal_obj_to_process, STATUS_COMPLETED, "Execution successful.")
            any_goal_status_changed_this_cycle = True
            if self.notifier and hasattr(self.notifier, 'log_update'): # Check if notifier instance and method exist
                self.notifier.log_update(f"Completed GID ..{current_goal_id[-6:]}: {current_goal_text[:40]}...", [current_goal_text], approved_by="AI_System")
            else: self.logger("WARNING", "(ExecutorC|process_single) Notifier instance or log_update method not available for completion notice.")
            if run_result_for_potential_refinement and goal_obj_to_process.get("tool_file") and subtask_cat not in [CAT_CORE_FILE_UPDATE]:
                 if self._formulate_and_add_refinement_goal(goal_obj_to_process, run_result_for_potential_refinement, all_goals_list_ref):
                     self.logger("INFO", f"(ExecutorC|process_single): Refinement goal potentially added for GID ..{current_goal_id[-6:]}")
        else: # Execution was not successful
            final_failure_status = STATUS_BUILD_FAILED
            if goal_obj_to_process.get("tool_file") or goal_obj_to_process.get("used_existing_tool") or subtask_cat == CAT_COMMAND_EXECUTION :
                final_failure_status = STATUS_EXECUTED_WITH_ERRORS
            self._update_goal_status_and_history(goal_obj_to_process, final_failure_status, f"Failed. Error: {str(goal_obj_to_process.get('error', 'Unspecified error during processing'))[:150]}")
            if not goal_obj_to_process.get('failure_category'):
                goal_obj_to_process['failure_category'] = self._determine_failure_category(goal_obj_to_process.get("error",""), goal_obj_to_process.get("tool_file"))
            any_goal_status_changed_this_cycle = True
            if goal_obj_to_process.get("self_correction_attempts", 0) < self.max_self_correction_attempts:
                self.logger("INFO", f"(ExecutorC|process_single): Attempting self-correction for GID ..{current_goal_id[-6:]} (Attempt #{goal_obj_to_process.get('self_correction_attempts', 0) + 1})")
                if self._formulate_and_add_corrective_goal(goal_obj_to_process, all_goals_list_ref):
                    self.logger("INFO", f"(ExecutorC|process_single): Corrective goal added for GID ..{current_goal_id[-6:]}")
            else:
                self._update_goal_status_and_history(goal_obj_to_process, STATUS_FAILED_MAX_RETRIES, "Maximum self-correction attempts reached.")
                self.logger("WARNING", f"(ExecutorC|process_single): Max self-correction for GID ..{current_goal_id[-6:]} '{current_goal_text[:30]}...'. Final status: {STATUS_FAILED_MAX_RETRIES}")
                any_goal_status_changed_this_cycle = True

        if any_goal_status_changed_this_cycle:
            self.save_goals(all_goals_list_ref)
        else:
            self.logger("DEBUG", f"(ExecutorC|process_single): No status change or new goals from processing GID ..{current_goal_id[-6:]}. Goals not re-saved at this point.")

    def reprioritize_and_select_next_goal(self, goals_list: list, strategy: str = "evaluation_adjusted") -> Optional[Dict[str, Any]]:
        """Selects the next best goal based on a strategy, using self.mission_manager and self.logger."""
        # This logic is exactly from your script.
        if not goals_list: return None
        pending_goals = [g for g in goals_list if g.get("status") == STATUS_PENDING or g.get("status") == STATUS_AWAITING_CORRECTION]
        if not pending_goals: return None

        if strategy == "evaluation_adjusted":
            current_mission_data = {}
            if self.mission_manager and hasattr(self.mission_manager, 'load_mission'):
                try:
                    current_mission_data = self.mission_manager.load_mission()
                except Exception as e_load_mission:
                    self.logger("ERROR", f"(ExecutorC|reprioritize): Could not load mission for prioritization: {e_load_mission}")
            else:
                self.logger("WARNING", "(ExecutorC|reprioritize): MissionManager instance or load_mission method not available for prioritization.")

            mission_focus_areas = current_mission_data.get("current_focus_areas", []) if isinstance(current_mission_data, dict) else []

            def calculate_dynamic_score(goal_obj: Dict[str, Any]) -> float:
                # This inner function's logic is exactly from your script.
                base_priority_val = goal_obj.get("priority", PRIORITY_NORMAL)
                dynamic_score = (PRIORITY_LOW - base_priority_val + 1) * 15.0
                try:
                    created_at_str = goal_obj.get("created_at", "1970-01-01T00:00:00Z")
                    if created_at_str.endswith('Z'): created_at_str = created_at_str[:-1] + '+00:00'
                    created_dt = datetime.fromisoformat(created_at_str)
                    age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600.0
                    dynamic_score += min(age_hours * 0.5, 15.0)
                except ValueError as ve: self.logger("WARNING", f"(ExecutorC|reprioritize_score): Date parse error for GID ..{goal_obj.get('goal_id', 'N/A')[-6:]}: {ve}")
                except Exception as e_date: self.logger("ERROR", f"(ExecutorC|reprioritize_score): Date processing error for GID ..{goal_obj.get('goal_id', 'N/A')[-6:]}: {e_date}")
                if goal_obj.get("status") == STATUS_AWAITING_CORRECTION:
                    dynamic_score += 25.0 
                    dynamic_score -= goal_obj.get("self_correction_attempts", 0) * 5.0 
                elif goal_obj.get("source") == SOURCE_SELF_CORRECTION and goal_obj.get("status") == STATUS_PENDING:
                    dynamic_score += 20.0
                parent_id = goal_obj.get("parent_goal_id")
                if parent_id:
                    parent_goal = next((g_parent for g_parent in goals_list if g_parent.get("goal_id") == parent_id), None)
                    if parent_goal:
                        parent_eval = parent_goal.get("evaluation")
                        if parent_eval and isinstance(parent_eval.get("final_score"), (int, float)):
                            dynamic_score += parent_eval["final_score"] * 0.1
                        if parent_goal.get("status") == STATUS_FAILED_MAX_RETRIES:
                            dynamic_score -= 10
                alignment_bonus = 0.0
                goal_text_lower = goal_obj.get("goal","").lower()
                for focus_area in mission_focus_areas:
                    if focus_area.lower() in goal_text_lower: alignment_bonus += 8.0
                dynamic_score += alignment_bonus
                if goal_obj.get("status") == STATUS_PENDING :
                    dynamic_score -= goal_obj.get("self_correction_attempts", 0) * 2.0
                return dynamic_score

            sorted_pending_goals = sorted(pending_goals, key=calculate_dynamic_score, reverse=True)
            if sorted_pending_goals:
                self.logger("DEBUG", "(ExecutorC|reprioritize): Top goal candidates after reprioritization:")
                for i, g_cand in enumerate(sorted_pending_goals[:min(3, len(sorted_pending_goals))]):
                    score = calculate_dynamic_score(g_cand)
                    self.logger("DEBUG", f"  {i+1}. GID: ..{g_cand.get('goal_id','N/A')[-6:]}, DynScore: {score:.2f}, OrigPrio: {g_cand.get('priority', 'N/A')}, Status: {g_cand.get('status','N/A')}, Text: '{g_cand.get('goal','')[:40]}...'")
                return sorted_pending_goals[0]
            return None
        else: # Default strategy
            self.logger("DEBUG", "(ExecutorC|reprioritize): Using default priority/age sorting.")
            return sorted(pending_goals, key=lambda g: (g.get("priority", PRIORITY_NORMAL), g.get("created_at", "")))[0] if pending_goals else None

    def evaluate_goals(self): # This is the main orchestrator method
        """
        Main evaluation loop called by GoalWorker. Orchestrates decomposition, processing, and updates.
        Uses self.load_goals, self.save_goals, self.is_single_tool_creation_goal, self.decompose_goal, 
        self.add_goal, self.reprioritize_and_select_next_goal, self.process_single_goal, 
        self._update_goal_status_and_history, and self.notifier.
        """
        # This logic is exactly from your script, adapted to call self.methods
        goals_in_memory = self.load_goals()
        if not goals_in_memory:
            self.logger("DEBUG", "(ExecutorC|evaluate_goals): No goals in memory to evaluate.")
            return

        initial_goal_count = len(goals_in_memory)
        any_state_changed_in_evaluate_outer_loop = False

        # --- Phase 1: Pre-Decomposition (Copied from your script) ---
        parent_goals_to_consider = sorted(
            [g for g in goals_in_memory if g.get("status") == STATUS_PENDING and not g.get("is_subtask") and not g.get("subtasks") and not g.get("subtask_descriptions")],
            key=lambda g_sort_p1: (g_sort_p1.get("priority", PRIORITY_NORMAL), g_sort_p1.get("created_at", "")) # Renamed g to g_sort_p1
        )
        if parent_goals_to_consider:
            parent_goal_obj_to_process = parent_goals_to_consider[0] 
            thread_context_for_classification = self._get_context_for_llm_from_thread(parent_goal_obj_to_process.get("thread_id"), goals_in_memory, parent_goal_obj_to_process.get("goal_id"), max_entries=2)
            if self.is_single_tool_creation_goal(parent_goal_obj_to_process["goal"], thread_context_for_classification):
                self.logger("INFO", f"(ExecutorC|evaluate_goals): Goal GID ..{parent_goal_obj_to_process.get('goal_id', '')[-6:]} classified as single tool creation. Bypassing standard decomposition.")
                self._update_goal_status_and_history(parent_goal_obj_to_process, STATUS_DECOMPOSED, "Classified as single tool creation; creating one direct build subtask.")
                parent_goal_obj_to_process["subtask_descriptions"] = [parent_goal_obj_to_process["goal"]] 
                success, new_sub_id = self.add_goal(goals_in_memory, description=parent_goal_obj_to_process["goal"], source=SOURCE_DECOMPOSITION, approved_by="AI_System_Classifier", is_subtask=True, parent_goal_obj=parent_goal_obj_to_process,thread_id_param=parent_goal_obj_to_process.get("thread_id"),priority=parent_goal_obj_to_process.get("priority", PRIORITY_NORMAL),subtask_category_override=CAT_CODE_GENERATION_TOOL)
                if success and new_sub_id: parent_goal_obj_to_process["subtasks"] = [new_sub_id]
                any_state_changed_in_evaluate_outer_loop = True
            elif len(parent_goal_obj_to_process["goal"]) > self.decomposition_threshold or parent_goal_obj_to_process.get("source") == SOURCE_USER or parent_goal_obj_to_process.get("priority", PRIORITY_NORMAL) <= PRIORITY_HIGH:
                self.logger("INFO", f"(ExecutorC|evaluate_goals): Goal GID ..{parent_goal_obj_to_process.get('goal_id', '')[-6:]} not classified as single tool OR meets other criteria for standard decomposition.")
                subtask_descriptions_list = self.decompose_goal(parent_goal_obj_to_process, goals_in_memory)
                if subtask_descriptions_list:
                    self._update_goal_status_and_history(parent_goal_obj_to_process, STATUS_DECOMPOSED, f"Decomposed into {len(subtask_descriptions_list)} subtasks via standard method.")
                    parent_goal_obj_to_process["subtask_descriptions"] = subtask_descriptions_list
                    current_subtask_goal_ids = []
                    for sub_desc_text in subtask_descriptions_list:
                        s_success, s_new_sub_id = self.add_goal(goals_in_memory, description=sub_desc_text, source=SOURCE_DECOMPOSITION, approved_by="AI_System", is_subtask=True, parent_goal_obj=parent_goal_obj_to_process,thread_id_param=parent_goal_obj_to_process.get("thread_id"),priority=parent_goal_obj_to_process.get("priority", PRIORITY_NORMAL))
                        if s_success and s_new_sub_id: current_subtask_goal_ids.append(s_new_sub_id)
                    parent_goal_obj_to_process["subtasks"] = current_subtask_goal_ids
                    any_state_changed_in_evaluate_outer_loop = True
                    self.logger("INFO", f"(ExecutorC|evaluate_goals): Parent GID ..{parent_goal_obj_to_process.get('goal_id', '')[-6:]} decomposed into {len(current_subtask_goal_ids)} sub-goals via standard method.")
                else: 
                    self.logger("WARNING", f"(ExecutorC|evaluate_goals): Standard decomposition of GID ..{parent_goal_obj_to_process['goal_id'][-6:]} yielded no subtasks or failed.")
                    self._update_goal_status_and_history(parent_goal_obj_to_process, STATUS_BUILD_FAILED, "Standard decomposition returned no subtasks or LLM call failed.")
                    parent_goal_obj_to_process["error"] = "Standard decomposition returned no subtasks or LLM call failed."
                    parent_goal_obj_to_process["failure_category"] = "DecompositionFailure"
                    any_state_changed_in_evaluate_outer_loop = True
        
        if any_state_changed_in_evaluate_outer_loop or len(goals_in_memory) > initial_goal_count:
            self.save_goals(goals_in_memory)

        # --- Phase 2: Process one actionable goal (Copied from your script) ---
        goals_in_memory = self.load_goals() # Reload fresh list for prioritizer
        next_goal_to_process = self.reprioritize_and_select_next_goal(goals_in_memory)
        if next_goal_to_process:
            self.process_single_goal(next_goal_to_process, goals_in_memory) # This handles its own save
        else:
            self.logger("DEBUG", "(ExecutorC|evaluate_goals): No actionable goal selected by prioritizer in this cycle.")

        # --- Phase 3: Reactivate goals if corrections succeeded (Copied from your script) ---
        goals_in_memory = self.load_goals() 
        any_state_changed_phase3 = False # Renamed to avoid conflict
        goals_awaiting_correction_list = [g_await for g_await in goals_in_memory if g_await.get("status") == STATUS_AWAITING_CORRECTION] # Renamed g to g_await
        for goal_obj_awaiting in goals_awaiting_correction_list:
            original_failed_goal_id_awaiting = goal_obj_awaiting.get("goal_id")
            correction_succeeded_for_this_goal = False
            all_corrective_actions_concluded = True
            for corrective_goal_obj in goals_in_memory:
                if corrective_goal_obj.get("source") == SOURCE_SELF_CORRECTION and \
                   corrective_goal_obj.get("related_to_failed_goal_id") == original_failed_goal_id_awaiting and \
                   corrective_goal_obj.get("thread_id") == goal_obj_awaiting.get("thread_id"):
                    if corrective_goal_obj.get("status") == STATUS_COMPLETED: correction_succeeded_for_this_goal = True; break 
                    if corrective_goal_obj.get("status") in [STATUS_PENDING, STATUS_APPROVED, STATUS_AWAITING_CORRECTION]: all_corrective_actions_concluded = False
            if correction_succeeded_for_this_goal:
                self._update_goal_status_and_history(goal_obj_awaiting, STATUS_PENDING, "Reset to PENDING after successful self-correction of a related sub-goal.")
                goal_obj_awaiting["error"] = None; goal_obj_awaiting["failure_category"] = None; goal_obj_awaiting["execution_result"] = None
                any_state_changed_phase3 = True
                self.logger("INFO", f"(ExecutorC|evaluate_goals): Self-correction for GID:{goal_obj_awaiting.get('goal_id','')[-6:]} deemed successful. Reset to PENDING for retry.")
            elif all_corrective_actions_concluded and not correction_succeeded_for_this_goal:
                self._update_goal_status_and_history(goal_obj_awaiting, STATUS_FAILED_MAX_RETRIES, "All self-correction attempts for this goal have failed or did not resolve the issue.")
                any_state_changed_phase3 = True
                self.logger("WARNING", f"(ExecutorC|evaluate_goals): All self-correction attempts for GID:{goal_obj_awaiting.get('goal_id','')[-6:]} did not lead to success. Marked as FAILED_MAX_RETRIES.")
        if any_state_changed_phase3: self.save_goals(goals_in_memory)

        # --- Phase 4: Update parent goals (Copied from your script) ---
        goals_in_memory = self.load_goals()
        any_state_changed_phase4 = False # Renamed
        parent_goals_decomposed_list = [g_parent_dec for g_parent_dec in goals_in_memory if g_parent_dec.get("status") == STATUS_DECOMPOSED and g_parent_dec.get("subtasks")] # Renamed g
        completed_parent_goal_ids_this_cycle_phase4 = []
        for parent_goal_obj_decomposed in parent_goals_decomposed_list:
            all_subtasks_completed_for_parent = True; any_subtask_failed_definitively_for_parent = False; all_subtasks_accounted_for = True 
            if not parent_goal_obj_decomposed.get("subtasks"): continue
            for sub_task_id_str in parent_goal_obj_decomposed["subtasks"]:
                matching_subtask_goal_obj = next((g_sub for g_sub in goals_in_memory if g_sub.get("goal_id") == sub_task_id_str), None)
                if not matching_subtask_goal_obj: self.logger("WARNING", f"(ExecutorC|evaluate_goals): Subtask ID '{sub_task_id_str}' for parent GID ..{parent_goal_obj_decomposed['goal_id'][-6:]} not found. Parent completion check might be inaccurate."); all_subtasks_accounted_for = False; all_subtasks_completed_for_parent = False; break 
                sub_status_val = matching_subtask_goal_obj.get("status")
                if sub_status_val != STATUS_COMPLETED:
                    all_subtasks_completed_for_parent = False
                    if sub_status_val in [STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR, STATUS_BUILD_FAILED]:
                        is_subtask_still_being_actively_corrected = any(cg.get("related_to_failed_goal_id") == sub_task_id_str and cg.get("source") == SOURCE_SELF_CORRECTION and cg.get("thread_id") == matching_subtask_goal_obj.get("thread_id") and cg.get("status") in [STATUS_PENDING, STATUS_APPROVED, STATUS_AWAITING_CORRECTION] for cg in goals_in_memory)
                        if not is_subtask_still_being_actively_corrected: any_subtask_failed_definitively_for_parent = True; break 
            if any_subtask_failed_definitively_for_parent: break
            if not all_subtasks_accounted_for: continue
            if all_subtasks_completed_for_parent:
                self._update_goal_status_and_history(parent_goal_obj_decomposed, STATUS_COMPLETED, "All subtasks successfully completed.")
                any_state_changed_phase4 = True
                parent_goal_id = parent_goal_obj_decomposed.get('goal_id')
                if parent_goal_id: completed_parent_goal_ids_this_cycle_phase4.append(parent_goal_id)
                self.logger("INFO", f"(ExecutorC|evaluate_goals): Parent goal GID ..{parent_goal_id[-6:] if parent_goal_id else 'N/A'} completed as all subtasks done.")
            elif any_subtask_failed_definitively_for_parent:
                self._update_goal_status_and_history(parent_goal_obj_decomposed, STATUS_EXECUTED_WITH_ERRORS, "One or more critical subtasks failed definitively and could not be corrected.")
                parent_goal_obj_decomposed["error"] = "One or more subtasks failed terminally."
                parent_goal_obj_decomposed["failure_category"] = "SubtaskTerminalFailure"
                any_state_changed_phase4 = True
                self.logger("WARNING", f"(ExecutorC|evaluate_goals): Parent goal GID ..{parent_goal_obj_decomposed.get('goal_id', '')[-6:]} failed due to definitive subtask failure.")
        if any_state_changed_phase4: self.save_goals(goals_in_memory)

        if completed_parent_goal_ids_this_cycle_phase4 and self.notifier and hasattr(self.notifier, 'log_update'):
            completed_parent_texts_for_log = []
            for gid_comp_parent in completed_parent_goal_ids_this_cycle_phase4:
                comp_parent_obj = next((g_p_log for g_p_log in goals_in_memory if g_p_log.get("goal_id") == gid_comp_parent), None) # Renamed g to g_p_log
                if comp_parent_obj: completed_parent_texts_for_log.append(comp_parent_obj["goal"])
            if completed_parent_texts_for_log:
                summary_str_parents = f"Completed parent goal(s) via subtask completion: {', '.join(txt[:40]+ ('...' if len(txt)>40 else '') for txt in completed_parent_texts_for_log)}"
                self.notifier.log_update(summary_str_parents, completed_parent_texts_for_log, approved_by="AI_System_Subtask_Completion")
        self.logger("DEBUG", "(ExecutorC|evaluate_goals): Full evaluation cycle finished.")

    def init_executor(self): # Was global init_executor
        """Initializes the executor by ensuring meta dir and loading initial data."""
        self._ensure_meta_dir()
        self.load_goals() # Load and potentially migrate/initialize
        self.get_structured_tool_registry() # Load and potentially migrate/initialize
        self.logger("INFO", "(Executor class): Executor initialized.")

# --- End of Executor Class ---

if __name__ == "__main__":
    if should_log("INFO"): print("--- Testing Executor Class (Standalone - Full User Code) ---")
    
    # Mock dependencies for testing
    mock_config_exec_main = {
        "meta_dir": "meta_executor_test_full", # Use a distinct test directory
        "goals_file": os.path.join("meta_executor_test_full", "goals.json"),
        "tool_registry_file": os.path.join("meta_executor_test_full", "tool_registry.json"),
        "executor_config": { 
            "decomposition_threshold": 50, 
            "max_self_correction_attempts": 1 
        },
        "decompose_llm_timeout": 10, "tool_select_llm_timeout": 10, 
        "arg_extract_llm_timeout":10, "doc_tool_llm_timeout":10,
        "correction_llm_timeout":10, "refinement_llm_timeout":10,
        "tool_classification_llm_timeout": 10
    }

    def main_test_logger_exec_main(level, message):
        if should_log(level.upper()): print(f"[{level.upper()}] (Exec_MainTest) {message}")
    
    def main_test_query_llm_exec_main(prompt_text, system_prompt_override=None, raw_output=False, timeout=180):
        main_test_logger_exec_main("INFO", f"ExecMainTest Mock LLM. Sys: '{str(system_prompt_override)[:50]}...' Prompt: {prompt_text[:80]}...")
        if "Decompose" in prompt_text: return json.dumps(["Subtask A for main test", "Subtask B for main test"])
        if "Tool Name" in prompt_text and "Available Tools" in prompt_text: return "test_tool_main_registry"
        if "Expected Arguments" in prompt_text: return json.dumps({"main_arg1": "val_main1", "main_file": "main_test.txt"})
        if "Tool Script Code" in prompt_text: return json.dumps({"description": "Main test tool doc.", "capabilities": ["main_testing"], "args_info": [{"name":"main_file"}]})
        if "Corrective Goal" in prompt_text: return "Fix the main test error by checking main file paths."
        if "Refine tool" in prompt_text: return "Refine main test tool code for better messages."
        if "Classify user goals" in prompt_text: return "NO"
        return "[ExecMainTest Mock LLM: Default Response]"

    class MainTestMockMissionManagerExecMain:
        def load_mission(self): return {"current_focus_areas": ["Testing Main Executor"]}
    class MainTestMockNotifierExecMain:
        def log_update(self, summary, goals_completed_texts, approved_by): main_test_logger_exec_main("UPDATE_NOTIFY", f"By {approved_by}: {summary}")
    
    class MainTestMockToolBuilderExecMain:
        def __init__(self, *args, **kwargs): self.logger = kwargs.get('logger_func', print)
        def build_tool(self, description, tool_name_suggestion, thread_id, goals_list_ref, current_goal_id):
            self.logger("INFO", f"Mock ToolBuilder INSTANCE: build_tool for '{tool_name_suggestion}'. Desc: {description[:50]}...")
            # Simulate creating a dummy tool file relative to where executor.py might be tested from
            # To avoid issues with relative paths, use a known sub-directory within the test meta_dir
            test_tools_dir = os.path.join(mock_config_exec_main.get("meta_dir", "meta_executor_test_full"), "test_tools")
            os.makedirs(test_tools_dir, exist_ok=True)
            dummy_tool_path = os.path.join(test_tools_dir, f"{tool_name_suggestion}.py")
            with open(dummy_tool_path, "w", encoding="utf-8") as f:
                f.write(f"# Dummy tool: {tool_name_suggestion}\ndef main(): print('Main in {tool_name_suggestion} ran')\nif __name__=='__main__':main()")
            return dummy_tool_path
            
    class MainTestMockToolRunnerExecMain:
        def __init__(self, *args, **kwargs): self.logger = kwargs.get('logger_func', print)
        def run_tool_safely(self, tool_path, tool_args=None):
            self.logger("INFO", f"Mock ToolRunner INSTANCE: run_tool_safely for '{tool_path}' with args {tool_args}")
            if "error" in os.path.basename(tool_path).lower(): return {"status": "error", "error": "Simulated tool run error from main test"}
            return {"status": "success", "output": f"Mock success from {os.path.basename(tool_path)} in main test"}

    # Clean up and setup test directory
    if os.path.exists(mock_config_exec_main["meta_dir"]):
        import shutil
        shutil.rmtree(mock_config_exec_main["meta_dir"])
    os.makedirs(mock_config_exec_main["meta_dir"], exist_ok=True)

    dummy_tool_reg_path_main = os.path.join(mock_config_exec_main["meta_dir"], "tool_registry.json")
    # Path for dummy tool in registry should align with where MockToolBuilder creates it
    mock_tool_path_in_registry = os.path.join(mock_config_exec_main.get("meta_dir","meta_executor_test_full"), "test_tools", "test_tool_main_registry.py")
    with open(dummy_tool_reg_path_main, "w", encoding='utf-8') as f_tr_main:
        json.dump([{"name": "test_tool_main_registry", "module_path": mock_tool_path_in_registry, "args_info":[{"name":"main_arg1"}, {"name":"main_file"}]}], f_tr_main)

    # Instantiate mock dependencies
    mock_mission_mgr_main = MainTestMockMissionManagerExecMain()
    mock_notifier_main = MainTestMockNotifierExecMain()
    mock_tool_builder_main = MainTestMockToolBuilderExecMain(logger_func=main_test_logger_exec_main)
    mock_tool_runner_main = MainTestMockToolRunnerExecMain(logger_func=main_test_logger_exec_main)

    exec_instance_main_test = Executor(
        config=mock_config_exec_main,
        logger_func=main_test_logger_exec_main,
        query_llm_func=main_test_query_llm_exec_main,
        mission_manager_instance=mock_mission_mgr_main,
        notifier_instance=mock_notifier_main,
        tool_builder_instance=mock_tool_builder_main,
        tool_runner_instance=mock_tool_runner_main   
    )
    exec_instance_main_test.init_executor()

    # Test logic from your original __main__
    if should_log("INFO"): print("\n1. Main Test: Adding a test parent goal for decomposition:")
    initial_goals_main_test = exec_instance_main_test.load_goals()
    success_add_parent_main, parent_id_main_test = exec_instance_main_test.add_goal(initial_goals_main_test, "Test Parent Goal for Full Executor Main Test", priority=PRIORITY_HIGH)
    if success_add_parent_main:
        main_test_logger_exec_main("INFO", f"Added parent goal GID ..{parent_id_main_test[-6:] if parent_id_main_test else 'N/A'}")
        exec_instance_main_test.save_goals(initial_goals_main_test)
    else: main_test_logger_exec_main("ERROR", "Failed to add parent goal for main test.")
    
    if should_log("INFO"): print("\n2. Main Test: Running evaluate_goals() cycle (should decompose parent):")
    exec_instance_main_test.evaluate_goals()
    
    goals_after_decomp_main = exec_instance_main_test.load_goals()
    parent_after_decomp_main = next((g for g in goals_after_decomp_main if g.get("goal_id") == parent_id_main_test), None)
    if parent_after_decomp_main and parent_after_decomp_main.get("status") == STATUS_DECOMPOSED:
        main_test_logger_exec_main("INFO", f"Parent goal GID ..{parent_id_main_test[-6:] if parent_id_main_test else 'N/A'} decomposed. Subtasks: {parent_after_decomp_main.get('subtasks')}")
    else: main_test_logger_exec_main("WARNING", f"Parent GID ..{parent_id_main_test[-6:] if parent_id_main_test else 'N/A'} not decomposed. Status: {parent_after_decomp_main.get('status') if parent_after_decomp_main else 'Not Found'}")

    if should_log("INFO"): print("\n3. Main Test: Running evaluate_goals() cycle again (should process a subtask):")
    exec_instance_main_test.evaluate_goals()
    
    if should_log("INFO"): print("\n4. Main Test: Running evaluate_goals() cycle again (should process next subtask or complete parent):")
    exec_instance_main_test.evaluate_goals()

    final_goals_main_test_run = exec_instance_main_test.load_goals()
    if should_log("INFO"): print("\n--- Final Goals List (Executor Full Main Test) ---")
    for g_final_main_run in final_goals_main_test_run:
        if should_log("DEBUG"): print(f"  GID: ..{g_final_main_run.get('goal_id','N/A')[-6:]}, Status: {g_final_main_run.get('status','N/A')}, Goal: '{g_final_main_run.get('goal','')[:60]}...'")

    if should_log("INFO"): print("\n--- Executor Class Test (from user's full code) Complete ---")