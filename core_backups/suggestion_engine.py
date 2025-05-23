# Self-Aware/suggestion_engine.py
import json
import os
import traceback
from datetime import datetime, timezone, timedelta
import re
from difflib import SequenceMatcher
import sys
import uuid
from typing import List, Dict, Optional, Any, Literal, Union, Callable

# --- Type Hinting for Suggestion Management (defined at top level) ---
SuggestionStatus = Literal[
    "pending", "approved_by_user", "approved_by_ai",
    "rejected_by_user", "rejected_by_ai",
    "modified_by_ai", "archived"
]
ActorType = Optional[Literal["user", "AI"]]

# --- Fallback definitions for when dependencies are not passed to the class instance ---
# These are used if None is passed to __init__ for logger, query_llm, etc.
_CLASS_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (SuggestionEngineClassFallbackLog) {message}")
_CLASS_FALLBACK_QUERY_LLM = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=180: \
    f"[Error: Fallback LLM query from SuggestionEngine instance. Prompt: {prompt_text[:100]}...]"

class SuggestionEngine:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None,
                 mission_manager_instance: Optional[Any] = None,
                 prompt_manager_load_prompt_func: Optional[Callable[[], str]] = None, # For get_system_prompt
                 executor_module_for_lazy_load: Optional[Any] = None):

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_FALLBACK_LOGGER
        self.query_llm = query_llm_func if query_llm_func else _CLASS_FALLBACK_QUERY_LLM
        self.mission_manager = mission_manager_instance
        self._prompt_manager_load_prompt_func = prompt_manager_load_prompt_func
        self._executor_module_for_lazy_load = executor_module_for_lazy_load

        # --- Initialize paths and config values from self.config ---
        self.meta_dir = self.config.get("meta_dir", "meta")
        
        # File paths used by SuggestionEngine
        self.suggestions_file_path = self.config.get("suggestions_file", os.path.join(self.meta_dir, "suggestions.json"))
        self.goals_file_path = self.config.get("goals_file", os.path.join(self.meta_dir, "goals.json"))
        self.tool_registry_file_path = self.config.get("tool_registry_file", os.path.join(self.meta_dir, "tool_registry.json"))
        self.conversation_review_log_file_path = self.config.get("conversation_review_log_file", os.path.join(self.meta_dir, "conversation_review_log.json"))
        
        self.ai_core_model_name = self.config.get("llm_model_name", "fallback_model_suggestion_engine_class")

        # Tunable parameters from config, with defaults from your script
        sugg_cfg = self.config.get("suggestion_engine_config", {})
        self.max_pending_suggestions_in_system = sugg_cfg.get("max_pending_suggestions_in_system", 15)
        self.max_pending_suggestions_for_generation_context = sugg_cfg.get("max_pending_suggestions_for_generation_context", 5)
        self.max_recent_goals_context = sugg_cfg.get("max_recent_goals_context", 5)
        self.max_recent_tools_context = sugg_cfg.get("max_recent_tools_context", 3)
        self.max_recent_suggestions_for_de_dup_context = sugg_cfg.get("max_recent_suggestions_for_de_dup_context", 10)
        self.max_recently_rejected_suggestions_context = sugg_cfg.get("max_recently_rejected_suggestions_context", 3)
        self.max_conversation_snippets_for_context = sugg_cfg.get("max_conversation_snippets_for_context", 2)
        self.max_turns_per_conversation_snippet = sugg_cfg.get("max_turns_per_conversation_snippet", 4)
        self.similarity_threshold_for_de_dup = sugg_cfg.get("similarity_threshold_for_de_dup", 0.85)

        self._ensure_meta_dir()
        self.init_suggestions_file() # Ensure file exists on init

    def _ensure_meta_dir(self):
        try:
            os.makedirs(self.meta_dir, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(SuggestionEngine class): Could not create meta directory {self.meta_dir}: {e}")

    def _format_timestamp(self, ts_input: Any) -> str:
        if not ts_input: return "Unknown Time"
        try:
            dt_obj: Optional[datetime] = None
            if isinstance(ts_input, str):
                if ts_input.endswith('Z'):
                    if sys.version_info < (3, 11):
                        ts_input = ts_input[:-1] + '+00:00'
                dt_obj = datetime.fromisoformat(ts_input)
            elif isinstance(ts_input, (int, float)):
                dt_obj = datetime.fromtimestamp(ts_input, tz=timezone.utc)

            if dt_obj:
                if dt_obj.tzinfo: dt_obj = dt_obj.astimezone(None)
                return dt_obj.strftime("%Y-%m-%d %H:%M")
            return str(ts_input)
        except Exception:
            return str(ts_input)

    def init_suggestions_file(self):
        self._ensure_meta_dir()
        if not os.path.exists(self.suggestions_file_path):
            try:
                with open(self.suggestions_file_path, 'w', encoding='utf-8') as f: json.dump([], f, indent=2)
                self.logger("INFO", f"(SuggestionEngine class): Initialized suggestions file at {self.suggestions_file_path}")
            except IOError as e:
                self.logger("ERROR", f"(SuggestionEngine class): Could not init suggestions file {self.suggestions_file_path}: {e}")

    def load_suggestions(self, load_all_history: bool = False) -> List[Dict[str, Any]]:
        self.init_suggestions_file() # Ensures file exists
        try:
            with open(self.suggestions_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip(): return []
            loaded_suggs_raw = json.loads(content)
            if not isinstance(loaded_suggs_raw, list):
                self.logger("WARNING", f"(SuggestionEngine class): {self.suggestions_file_path} content not a list. Reinitializing.")
                self.save_suggestions([]) # Use instance method
                return []

            migrated_suggs: List[Dict[str, Any]] = []
            for s_item_raw in loaded_suggs_raw:
                if not isinstance(s_item_raw, dict):
                    self.logger("WARNING", f"(SuggestionEngine class): Skipping non-dict item in suggestions: {str(s_item_raw)[:100]}")
                    continue
                s_item = s_item_raw.copy()
                s_item.setdefault("id", f"sugg_{uuid.uuid4().hex[:8]}")
                s_item.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
                s_item.setdefault("suggestion", s_item_raw.get("description", "Unknown suggestion content"))
                if "description" in s_item and "suggestion" not in s_item_raw:
                    del s_item["description"]
                s_item.setdefault("status", "pending")
                s_item.setdefault("source_model", self.ai_core_model_name)
                s_item.setdefault("related_thread_id", "")
                if s_item.get("related_thread_id") is None: s_item["related_thread_id"] = ""
                s_item.setdefault("approved_by", None)
                s_item.setdefault("rejected_by", None)
                s_item.setdefault("rejection_reason", None)
                s_item.setdefault("modification_history", [])
                s_item.setdefault("priority", s_item_raw.get("priority", 5))
                s_item.setdefault("details", s_item_raw.get("details", {}))
                s_item.setdefault("origin", s_item_raw.get("origin", None))
                migrated_suggs.append(s_item)

            migrated_suggs.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
            return migrated_suggs
        except Exception as e:
            self.logger("CRITICAL", f"(SuggestionEngine class): Error loading suggestions from {self.suggestions_file_path}: {e}\n{traceback.format_exc()}. Returning empty list.")
            return []

    def save_suggestions(self, suggestions: List[Dict[str, Any]]):
        self.init_suggestions_file() # Ensures file and dir exist
        try:
            valid_suggestions = [s for s in suggestions if isinstance(s, dict)]
            if len(valid_suggestions) != len(suggestions):
                self.logger("WARNING", "(SuggestionEngine class): Attempted to save non-dict items in suggestions list. Filtering non-dicts.")

            for s_item in valid_suggestions:
                if "description" in s_item and "suggestion" not in s_item:
                    s_item["suggestion"] = s_item.pop("description")
                s_item.setdefault("id", f"sugg_{uuid.uuid4().hex[:8]}")

            valid_suggestions.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
            with open(self.suggestions_file_path, 'w', encoding='utf-8') as f:
                json.dump(valid_suggestions, f, indent=2)
        except Exception as e:
            self.logger("ERROR", f"(SuggestionEngine class): Could not save suggestions to {self.suggestions_file_path}: {e}\n{traceback.format_exc()}")

    def _load_json_for_context(self, filepath: str, default_value: Union[List[Any], Dict[str,Any]], context_name:str="data") -> Union[List[Any], Dict[str,Any]]:
        if not os.path.exists(filepath): return default_value
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip(): return default_value
            loaded_data = json.loads(content)
            if isinstance(default_value, list) and isinstance(loaded_data, list): return loaded_data
            if isinstance(default_value, dict) and isinstance(loaded_data, dict): return loaded_data
            self.logger("WARNING", f"(SuggestionEngine class): Type mismatch for {context_name} from {filepath}. Expected {type(default_value)}, got {type(loaded_data)}. Using default.")
            return default_value
        except Exception as e:
            self.logger("WARNING", f"(SuggestionEngine class): Error loading {context_name} from {filepath}: {e}. Using default.")
            return default_value

    def _save_json_data_for_suggestion_context(self, filepath: str, data: Union[Dict[str, Any], List[Any]]): # This was a global function
        self._ensure_meta_dir()
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            self.logger("ERROR", f"(SuggestionEngine class): Could not write to {filepath}: {e}")
        except Exception as e_unexp:
            self.logger("CRITICAL", f"(SuggestionEngine class): Unexpected error saving to {filepath}: {e_unexp}\n{traceback.format_exc()}.")

    def _is_suggestion_too_similar(self, new_suggestion_text: str, recent_suggestions: List[Dict[str, Any]], threshold: float) -> bool:
        if not new_suggestion_text: return False
        new_lower = new_suggestion_text.lower()
        for old_sugg_obj in recent_suggestions:
            if not isinstance(old_sugg_obj, dict): continue
            old_sugg_text = old_sugg_obj.get("suggestion")
            if not isinstance(old_sugg_text, str): continue
            similarity = SequenceMatcher(None, new_lower, old_sugg_text.lower()).ratio()
            if similarity > threshold:
                self.logger("DEBUG", f"(SuggestionEngine class): New suggestion '{new_suggestion_text[:50]}...' (sim: {similarity:.2f}) too similar to: '{old_sugg_text[:50]}...'")
                return True
        return False

    def get_suggestion_by_id_or_timestamp(self, identifier: str, all_suggestions: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        suggs_to_search = all_suggestions if all_suggestions is not None else self.load_suggestions(load_all_history=True)
        for sugg in suggs_to_search:
            if sugg.get("id") == identifier or sugg.get("timestamp") == identifier:
                return sugg
        return None

    def update_suggestion_status_details(
        self, suggestion_identifier: str,
        actor: ActorType = None,
        updates_dict: Optional[Dict[str, Any]] = None,
        reason_for_change: Optional[str] = None,
        set_approved_by: ActorType = None,
        set_rejected_by: ActorType = None,
        set_rejection_reason: Optional[str] = None,
        new_status_override: Optional[SuggestionStatus] = None
    ) -> bool:
        all_suggs = self.load_suggestions(load_all_history=True)
        suggestion_to_update: Optional[Dict[str, Any]] = None
        sugg_index = -1

        for i, sugg_item in enumerate(all_suggs):
            if sugg_item.get("id") == suggestion_identifier or sugg_item.get("timestamp") == suggestion_identifier:
                suggestion_to_update = sugg_item
                sugg_index = i
                break

        if not suggestion_to_update:
            self.logger("WARNING", f"(SuggestionEngine class) Suggestion with identifier '{suggestion_identifier}' not found for update.")
            return False

        modification_made = False
        current_changes_log: Dict[str, Any] = {}

        if updates_dict and isinstance(updates_dict, dict):
            for key, value in updates_dict.items():
                field_to_update = "suggestion" if key == "description" else key
                if field_to_update in suggestion_to_update and suggestion_to_update[field_to_update] != value:
                    current_changes_log[field_to_update] = {"old": suggestion_to_update[field_to_update], "new": value}
                    suggestion_to_update[field_to_update] = value
                    modification_made = True
                elif field_to_update not in suggestion_to_update:
                    current_changes_log[field_to_update] = {"old": None, "new": value}
                    suggestion_to_update[field_to_update] = value
                    modification_made = True
        
        determined_status: SuggestionStatus = suggestion_to_update.get("status", "pending") # type: ignore

        if set_approved_by:
            approved_status_val: SuggestionStatus = f"approved_by_{set_approved_by.lower()}" # type: ignore
            if determined_status != approved_status_val or suggestion_to_update.get("approved_by") != set_approved_by:
                determined_status = approved_status_val
                suggestion_to_update["approved_by"] = set_approved_by
                suggestion_to_update["rejected_by"] = None
                suggestion_to_update["rejection_reason"] = None
                modification_made = True

        if set_rejected_by:
            rejected_status_val: SuggestionStatus = f"rejected_by_{set_rejected_by.lower()}" # type: ignore
            if determined_status != rejected_status_val or suggestion_to_update.get("rejected_by") != set_rejected_by:
                determined_status = rejected_status_val
                suggestion_to_update["rejected_by"] = set_rejected_by
                suggestion_to_update["approved_by"] = None
                if set_rejection_reason is not None:
                    if suggestion_to_update.get("rejection_reason") != set_rejection_reason:
                        current_changes_log.setdefault("rejection_reason", {"old": suggestion_to_update.get("rejection_reason"), "new": set_rejection_reason})
                        suggestion_to_update["rejection_reason"] = set_rejection_reason
                modification_made = True

        if new_status_override and determined_status != new_status_override:
            determined_status = new_status_override
            modification_made = True

        if suggestion_to_update.get("status") != determined_status:
            current_changes_log.setdefault("status", {"old": suggestion_to_update.get("status"), "new": determined_status})
            suggestion_to_update["status"] = determined_status
            modification_made = True

        if modification_made:
            if actor and current_changes_log:
                mod_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(), "actor": actor,
                    "changes": current_changes_log
                }
                if reason_for_change: mod_entry["reason"] = reason_for_change
                suggestion_to_update.setdefault("modification_history", []).append(mod_entry)

                if actor == "AI" and \
                   not suggestion_to_update["status"].startswith("approved_") and \
                   not suggestion_to_update["status"].startswith("rejected_") and \
                   suggestion_to_update["status"] != "modified_by_ai" and \
                   not new_status_override:
                        if suggestion_to_update.get("status") != "modified_by_ai":
                            current_changes_log.setdefault("status", {"old": suggestion_to_update.get("status"), "new": "modified_by_ai"})
                        suggestion_to_update["status"] = "modified_by_ai" # type: ignore

            all_suggs[sugg_index] = suggestion_to_update
            self.save_suggestions(all_suggs)
            self.logger("INFO", f"(SuggestionEngine class) Suggestion '{suggestion_identifier}' updated by {actor or 'system'}. New status: {suggestion_to_update['status']}.")
            return True

        self.logger("DEBUG", f"(SuggestionEngine class) No effective changes made to suggestion '{suggestion_identifier}'.")
        return False

    def archive_suggestion_as_rejected_by_actor(
        self, suggestion_identifier: str, reason: str, rejected_by: ActorType = "AI"
    ) -> bool:
        return self.update_suggestion_status_details(
            suggestion_identifier, set_rejected_by=rejected_by,
            set_rejection_reason=reason, actor=rejected_by,
            reason_for_change=f"Suggestion rejected: {reason}"
        )

    def mark_suggestion_as_approved_by_actor(
        self, suggestion_identifier: str, approved_by: ActorType = "AI"
    ) -> bool:
        return self.update_suggestion_status_details(
            suggestion_identifier, set_approved_by=approved_by, actor=approved_by,
            reason_for_change="Suggestion approved for goal creation."
        )

    def add_new_suggestion_object(self, suggestion_obj: Dict[str, Any]) -> bool:
        if not isinstance(suggestion_obj, dict) or "suggestion" not in suggestion_obj:
            self.logger("ERROR", "(SuggestionEngine class) Invalid suggestion object provided to add_new_suggestion_object.")
            return False

        suggestion_obj.setdefault("id", f"sugg_{uuid.uuid4().hex[:8]}")
        suggestion_obj.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        suggestion_obj.setdefault("status", "pending") # type: ignore
        suggestion_obj.setdefault("priority", 5)
        suggestion_obj.setdefault("details", {})
        suggestion_obj.setdefault("approved_by", None)
        suggestion_obj.setdefault("rejected_by", None)
        suggestion_obj.setdefault("rejection_reason", None)
        suggestion_obj.setdefault("modification_history", [])
        suggestion_obj.setdefault("origin", suggestion_obj.get("origin", "unknown"))

        all_suggs = self.load_suggestions(load_all_history=True)
        all_suggs.append(suggestion_obj)
        self.save_suggestions(all_suggs)
        self.logger("INFO", f"(SuggestionEngine class) Added new suggestion object ID: {suggestion_obj['id']}")
        return True

    def _get_system_prompt_for_suggestion_generation(self) -> str:
        """Helper to get system prompt, using passed-in function or fallback."""
        if self._prompt_manager_load_prompt_func:
            try:
                return self._prompt_manager_load_prompt_func()
            except Exception as e:
                self.logger("WARNING", f"(SuggestionEngine class) Error calling load_prompt_func: {e}. Using fallback prompt.")
        # Fallback if function not provided or fails
        return "You are a helpful AI suggesting improvements."


    def generate_ai_suggestion(self):
        executor_module: Optional[Any] = None
        if self._executor_module_for_lazy_load:
            executor_module = self._executor_module_for_lazy_load
            self.logger("DEBUG", "(SuggestionEngine class) Using pre-passed executor module for generate_ai_suggestion.")
        else:
            try:
                import importlib # Ensure importlib is imported for this specific use
                executor_module = sys.modules.get("executor") or importlib.import_module("executor")
                self._executor_module_for_lazy_load = executor_module # Cache it for next time
                self.logger("DEBUG", "(SuggestionEngine class) Lazy loaded executor module for generate_ai_suggestion.")
            except ImportError:
                self.logger("ERROR", "(SuggestionEngine class) Failed to lazy-load executor module for generate_ai_suggestion. Active goal context will be limited.")
        
        all_suggestions_history = self.load_suggestions(load_all_history=True)
        pending_suggestions_list = [s for s in all_suggestions_history if isinstance(s, dict) and s.get("status") == "pending"]
        if len(pending_suggestions_list) >= self.max_pending_suggestions_in_system:
            self.logger("DEBUG", "(SuggestionEngine class): Max pending suggestions in system reached. Skipping new generation.")
            return

        system_prompt_text = self._get_system_prompt_for_suggestion_generation() # Use helper

        mission_focus_str = "Mission focus areas not available."
        if self.mission_manager and hasattr(self.mission_manager, 'load_mission'):
            try:
                mission_data = self.mission_manager.load_mission()
                focus_areas = mission_data.get("current_focus_areas", []) if isinstance(mission_data, dict) else []
                if focus_areas: mission_focus_str = ", ".join(focus_areas)
            except Exception as e_mission:
                self.logger("WARNING", f"(SuggestionEngine class) Failed to load mission for suggestion context: {e_mission}")

        goals_data_list = self._load_json_for_context(self.goals_file_path, [], "goals")
        
        # --- Context building (Copied verbatim from your original script, adapted for self) ---
        performance_summary_parts = []
        if isinstance(goals_data_list, list):
            sorted_goals_for_perf = sorted(
                [g for g in goals_data_list if isinstance(g, dict)],
                key=lambda x: (x.get("evaluation", {}).get("last_evaluated_at") or x.get("created_at", "")), reverse=True
            )
            low_score_thresh, high_score_thresh = self.config.get("low_eval_score_threshold",0), self.config.get("high_eval_score_threshold",20)
            problem_goals, success_goals = [], []
            for goal in sorted_goals_for_perf[:self.max_recent_goals_context * 2]:
                goal_desc_key = "description" if "description" in goal else "goal"
                eval_data = goal.get("evaluation"); score = eval_data.get("final_score") if isinstance(eval_data, dict) else None
                if isinstance(score, int):
                    preview = f"'{goal.get(goal_desc_key,'N/A')[:70]}...' (Score:{score}, FailCat:{goal.get('failure_category','N/A')})"
                    if score < low_score_thresh and len(problem_goals) < 3: problem_goals.append(f"- Low: {preview}")
                    elif score >= high_score_thresh and len(success_goals) < 2: success_goals.append(f"- High: {preview}")
            if problem_goals: performance_summary_parts.append("Recent Problematic Goal Outcomes:\n" + "\n".join(problem_goals))
            if success_goals: performance_summary_parts.append("Recent Successful Goal Outcomes:\n" + "\n".join(success_goals))
        performance_context_str = "\n".join(performance_summary_parts) if performance_summary_parts else "No specific recent performance insights from evaluations."

        recent_suggs_for_llm = all_suggestions_history[:self.max_recent_suggestions_for_de_dup_context]
        rejected_suggs_for_ctx = [s for s in recent_suggs_for_llm if s.get("status", "").startswith("rejected")][:self.max_recently_rejected_suggestions_context]

        recent_suggs_ctx_parts = []
        for s_ctx in recent_suggs_for_llm:
             s_id_ctx = s_ctx.get('id','N/A')[-6:] if s_ctx.get('id') else 'N/A'
             recent_suggs_ctx_parts.append(f"- ID ..{s_id_ctx} \"{s_ctx.get('suggestion','N/A')[:80]}...\" (St: {s_ctx.get('status')})")
        recent_suggs_ctx = "\n### Recently Generated Suggestions (Awareness for Novelty):\n" + \
                           (("\n".join(recent_suggs_ctx_parts)) or "None")

        rejected_suggs_ctx_parts = []
        for s_rej_ctx in rejected_suggs_for_ctx:
            s_id_rej_ctx = s_rej_ctx.get('id','N/A')[-6:] if s_rej_ctx.get('id') else 'N/A'
            reason_text = s_rej_ctx.get('rejection_reason')
            if reason_text is None: reason_text = 'N/A'
            rejected_suggs_ctx_parts.append(f"- ID ..{s_id_rej_ctx} \"{s_rej_ctx.get('suggestion','N/A')[:80]}...\" (Reason: {str(reason_text)[:50]})")
        rejected_suggs_ctx = "\n### Recently REJECTED Suggestions (CRITICAL: Avoid re-suggesting similar ideas unless context changed significantly):\n" + \
                                (("\n".join(rejected_suggs_ctx_parts)) or "None")

        active_goals_ctx = "No specific high-priority active goals identified."
        if executor_module and hasattr(executor_module, 'reprioritize_and_select_next_goal'):
            try:
                all_goals_for_active = self._load_json_for_context(self.goals_file_path, [], "all_goals_for_active_check")
                if isinstance(all_goals_for_active, list) and all_goals_for_active:
                    top_goal = executor_module.reprioritize_and_select_next_goal(all_goals_for_active)
                    if top_goal and isinstance(top_goal, dict):
                        goal_desc_key_active = "description" if "description" in top_goal else "goal"
                        active_goals_ctx = f"\n### Current Top Priority Active Goal (Avoid redundancy):\n- \"{top_goal.get(goal_desc_key_active, 'N/A')[:80]}...\" (St: {top_goal.get('status')})"
            except Exception as e_active: self.logger("WARNING", f"(SuggestionEngine class) Failed to get active goal for suggestion context: {e_active}")

        tool_reg_list = self._load_json_for_context(self.tool_registry_file_path, [], "tool_registry")
        tool_ctx_parts = []
        if isinstance(tool_reg_list, list):
            sorted_tools = sorted([t for t in tool_reg_list if isinstance(t,dict)], key=lambda x: x.get("last_run_time") or x.get("last_updated") or "", reverse=True)
            for tool in sorted_tools[:self.max_recent_tools_context]:
                tool_ctx_parts.append(f"- {tool.get('name','N/A')} (Runs:{tool.get('total_runs',0)}, Fails:{tool.get('failure_count',0)})")
        tool_reg_ctx = "\n### Tool Insights:\n" + ("\n".join(tool_ctx_parts) or "No tool data.")

        conversation_snippets = self._load_json_for_context(self.conversation_review_log_file_path, [], "conversation_review_log")
        conv_summary_parts = []
        if isinstance(conversation_snippets, list):
            for snippet_entry in conversation_snippets[-self.max_conversation_snippets_for_context:]:
                if isinstance(snippet_entry, dict) and "conversation_snippet" in snippet_entry:
                    tid_snip = snippet_entry.get("thread_id", "N/A")[-6:]
                    turns = [f"    {t.get('speaker','?')}: {t.get('text','')[:80]}..." for t in snippet_entry.get("conversation_snippet",[])[-self.max_turns_per_conversation_snippet:] if isinstance(t,dict)]
                    if turns: conv_summary_parts.append(f"  - Chat Snippet (Th: ..{tid_snip}, Logged: {self._format_timestamp(snippet_entry.get('timestamp'))}):\n" + "\n".join(turns))
        conv_ctx_str = "\n### Recent Conversation Highlights (Consider if these reveal needs, user confusion, or opportunities for better interaction):\n" + \
                       ("\n".join(conv_summary_parts) if conv_summary_parts else "  (No recent conversation snippets available.)")
        # --- End of context building ---

        suggestion_task_prompt_context = (
            f"### Suggestion Generation Context ###\n"
            f"**Current Mission Focus Areas:** {mission_focus_str}\n"
            f"{performance_context_str}\n{active_goals_ctx}\n{tool_reg_ctx}\n{conv_ctx_str}\n"
            f"{recent_suggs_ctx}\n{rejected_suggs_ctx}\n\n"
            f"**Your Task: Propose ONE Novel, Actionable, High-Impact Goal Suggestion**\n"
             f"Based on your core operational instructions (provided by the system and implied in the overall context) AND ALL the above specific context (mission focus, performance, active work, tool status, conversation highlights, and past suggestions), "
            f"propose **ONE NEW and UNIQUE** goal suggestion. It should be a concise, actionable task description.\n"
            f"Prioritize suggestions that:\n"
            f"  a) Directly align with a 'Current Mission Focus Area'.\n"
            f"  b) Address underlying causes of recently poor goal evaluations or common failure categories.\n"
            f"  c) Improve conversational abilities or address issues revealed in 'Recent Conversation Highlights'.\n"
            f"  d) Offer a novel approach to a persistent problem if previous attempts (reflected in rejected suggestions or problematic goals) have failed.\n"
            f"  e) Build upon recent successes or enhance well-performing tools.\n"
            f"**Critically, ensure your suggestion is substantially different from the 'Recently REJECTED Suggestions' and the broader 'Recently Generated Suggestions' list.** "
            f"If a similar topic is unavoidable due to high importance, the approach or scope must be clearly novel.\n"
            f"If suggesting a fix for a recurring problem, try to propose a more fundamental or different solution.\n"
            f"If the suggestion relates to a specific existing goal thread (e.g., from performance or conversation context), include 'for thread ..xxx' (using last 6 chars of thread_id) in the suggestion text.\n"
            f"If no truly novel, valuable, and distinct suggestion is apparent from the current context, respond ONLY with the exact word 'NONE'."
        )

        llm_reply_text = self.query_llm(
            prompt_text=suggestion_task_prompt_context,
            system_prompt_override=system_prompt_text,
            raw_output=False,
            timeout=self.config.get("suggestion_llm_timeout", 360) # Use config for timeout
        )

        suggestion_text_from_llm = llm_reply_text.strip() if isinstance(llm_reply_text, str) else ""

        if not suggestion_text_from_llm or suggestion_text_from_llm.upper() == "NONE" or "[Error:" in suggestion_text_from_llm :
            self.logger("DEBUG", f"(SuggestionEngine class): LLM suggested NONE or empty/error. LLM Raw: '{suggestion_text_from_llm[:100]}'")
            return

        if self._is_suggestion_too_similar(suggestion_text_from_llm, all_suggestions_history[:self.max_recent_suggestions_for_de_dup_context*2], self.similarity_threshold_for_de_dup):
            self.logger("DEBUG", f"(SuggestionEngine class): Generated suggestion '{suggestion_text_from_llm[:50]}...' too similar. Skipping.")
            return

        current_pending_sugg_texts_lower = {s.get('suggestion','').lower() for s in pending_suggestions_list if isinstance(s.get('suggestion'), str)}
        if suggestion_text_from_llm.lower() in current_pending_sugg_texts_lower:
            self.logger("DEBUG", f"(SuggestionEngine class): Suggestion '{suggestion_text_from_llm[:50]}...' (exact match) already pending. Skipping.")
            return

        extracted_thread_id_for_sugg = None
        thread_match = re.search(r"for\s+thread\s*\.\.([a-f0-9]{4,12})", suggestion_text_from_llm, re.IGNORECASE)
        if thread_match:
            partial_thread_id = thread_match.group(1)
            all_goals_sugg_th_check = self._load_json_for_context(self.goals_file_path, [], "goals_for_thread_id_linking")
            if isinstance(all_goals_sugg_th_check, list):
                for g_check in all_goals_sugg_th_check:
                    if isinstance(g_check, dict) and isinstance(g_check.get("thread_id"), str) and g_check.get("thread_id","").endswith(partial_thread_id):
                        extracted_thread_id_for_sugg = g_check["thread_id"]
                        self.logger("DEBUG", f"(SuggestionEngine class): Linked suggestion to existing thread ..{extracted_thread_id_for_sugg[-6:]}")
                        break

        new_suggestion_object = {
            "id": f"sugg_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suggestion": suggestion_text_from_llm,
            "status": "pending", # type: ignore
            "source_model": self.ai_core_model_name,
            "related_thread_id": extracted_thread_id_for_sugg if extracted_thread_id_for_sugg else "",
            "details": {"generated_by_llm": True, "generation_context_preview": suggestion_task_prompt_context[:200] + "..."},
            "priority": 5, "approved_by": None, "rejected_by": None, "rejection_reason": None,
            "modification_history": [], "origin": "generate_ai_suggestion"
        }
        self.add_new_suggestion_object(new_suggestion_object)
        self.logger("SUGGESTION", f"New AI Suggestion (ID: ..{new_suggestion_object['id'][-6:]}, Th: ..{str(extracted_thread_id_for_sugg)[-6:] if extracted_thread_id_for_sugg else 'New'}): {suggestion_text_from_llm}")


if __name__ == "__main__":
    print("--- Testing SuggestionEngine Class (Standalone from User's Full Code) ---")

    # Setup mock config and logger for standalone test
    mock_config_se_main = {
        "meta_dir": "meta_se_test_full",
        "suggestions_file": os.path.join("meta_se_test_full", "suggestions.json"),
        "goals_file": os.path.join("meta_se_test_full", "goals.json"),
        "tool_registry_file": os.path.join("meta_se_test_full", "tool_registry.json"),
        "conversation_review_log_file": os.path.join("meta_se_test_full", "conv_review.json"),
        "llm_model_name": "test_model_se_full",
        "suggestion_engine_config": {
            "max_pending_suggestions_in_system": 3, # Low for testing
            "max_recent_goals_context": 2,
            "similarity_threshold_for_de_dup": 0.95
        },
        "suggestion_llm_timeout": 10 # Short for testing
    }

    def main_test_logger_se(level, message):
        print(f"[{level.upper()}] (SE_MainTest) {message}")

    def main_test_query_llm_se(prompt_text, system_prompt_override=None, raw_output=False, timeout=180):
        main_test_logger_se("INFO", f"MainTest Mock LLM called. Prompt starts: {prompt_text[:150]}...")
        if "### Suggestion Generation Context ###" in prompt_text:
            if "problematic goal outcomes" in prompt_text.lower(): # Example condition
                return "Suggest detailed review of recent problematic goal outcomes for thread ..prob12"
            return "Consider refactoring the main user interface for better clarity." # Generic suggestion
        return "[MainTest Mock LLM: Default Response SE]"

    class MainTestMockMissionManager:
        def load_mission(self): return {"current_focus_areas": ["Test Main: Focus Area 1", "Improving test coverage"]}
        def get_mission_statement_for_prompt(self): return "### MainTest Mission For SE ###\nFocus on testing."

    class MainTestMockPromptManager: # Simulating that load_prompt function is available via an instance
        def load_prompt(self): return "You are a suggestion engine test AI."
    
    mock_pm_instance_main = MainTestMockPromptManager()

    # Clean up and setup test directory
    if os.path.exists(mock_config_se_main["meta_dir"]):
        import shutil
        shutil.rmtree(mock_config_se_main["meta_dir"])
    os.makedirs(mock_config_se_main["meta_dir"], exist_ok=True)

    # Create dummy context files
    for f_key in ["goals_file", "tool_registry_file", "conversation_review_log_file"]:
        f_path = mock_config_se_main[f_key]
        with open(f_path, "w", encoding="utf-8") as df: json.dump([], df)
    with open(os.path.join(mock_config_se_main["meta_dir"],"mission.json"),"w") as mf: json.dump(MainTestMockMissionManager().load_mission(), mf)


    se_instance_main = SuggestionEngine(
        config=mock_config_se_main,
        logger_func=main_test_logger_se,
        query_llm_func=main_test_query_llm_se,
        mission_manager_instance=MainTestMockMissionManager(),
        prompt_manager_load_prompt_func=mock_pm_instance_main.load_prompt 
        # executor_module_for_lazy_load could be a mock if generate_ai_suggestion's executor path is tested
    )

    main_test_logger_se("INFO", "Standalone SE Test: Initializing with empty suggestions file.")
    se_instance_main.init_suggestions_file() # Should create if not exists

    print("\n1. Testing add_new_suggestion_object directly:")
    s1 = {"id": "smain001", "suggestion": "First test suggestion for main.", "origin": "main_test"}
    se_instance_main.add_new_suggestion_object(s1)
    loaded_after_add = se_instance_main.load_suggestions()
    print(f"   Suggestions after add: {len(loaded_after_add)}")
    if not any(s['id'] == 'smain001' for s in loaded_after_add): print("   ERROR: smain001 not found after add.")

    print("\n2. Testing generate_ai_suggestion (first pass):")
    se_instance_main.generate_ai_suggestion()
    loaded_after_gen1 = se_instance_main.load_suggestions()
    print(f"   Suggestions after first generate_ai_suggestion: {len(loaded_after_gen1)}")
    for s_item_gen1 in loaded_after_gen1[-2:]: # Show last few
         print(f"     - \"{s_item_gen1.get('suggestion','')[:60]}...\" (Origin: {s_item_gen1.get('origin')})")

    print("\n3. Testing update_suggestion_status_details:")
    se_instance_main.update_suggestion_status_details("smain001", actor="AI", updates_dict={"priority": 1, "suggestion": "Updated first suggestion!"}, reason_for_change="AI Test Update")
    updated_s1 = se_instance_main.get_suggestion_by_id_or_timestamp("smain001")
    if updated_s1 and updated_s1.get("priority") == 1 and "Updated first suggestion!" in updated_s1.get("suggestion",""):
        print(f"   SUCCESS: smain001 updated. Status: {updated_s1.get('status')}")
    else:
        print(f"   ERROR: smain001 not updated correctly. Found: {updated_s1}")

    print("\n4. Testing generate_ai_suggestion again (to check de-dup and limits):")
    # Add a "problematic goal" to influence the mock LLM
    with open(mock_config_se_main["goals_file"], "w", encoding="utf-8") as gf_main:
         json.dump([{"goal_id": "prob12", "thread_id":"th_prob12", "goal":"A problematic task", "status":"failed_max_retries", "evaluation":{"final_score":-10}}], gf_main)
    se_instance_main.generate_ai_suggestion() # Should try to generate a specific suggestion
    se_instance_main.generate_ai_suggestion() # Might hit similarity or pending limit
    se_instance_main.generate_ai_suggestion() # ""
    loaded_after_gen_multi = se_instance_main.load_suggestions(load_all_history=True)
    print(f"   Suggestions after multiple generate_ai_suggestion calls: {len(loaded_after_gen_multi)}")
    for s_item_gen_multi in loaded_after_gen_multi:
         print(f"     - \"{s_item_gen_multi.get('suggestion','')[:60]}...\" (Status: {s_item_gen_multi.get('status')}, Origin: {s_item_gen_multi.get('origin')})")


    print("\n--- SuggestionEngine Class Test (from user's full code) Complete ---")