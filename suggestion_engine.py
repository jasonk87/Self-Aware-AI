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
import time

# Module-level singleton instance
_suggestion_engine_instance = None

def load_suggestions() -> List[Dict[str, Any]]:
    """Load all suggestions from the suggestions file"""
    try:
        suggestions_file = os.path.abspath(os.path.join("meta", "suggestions.json"))
        if os.path.exists(suggestions_file):
            with open(suggestions_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load suggestions: {e}")
    return []

def save_suggestions(suggestions: List[Dict[str, Any]]) -> bool:
    """Save suggestions to the suggestions file"""
    try:
        suggestions_file = os.path.abspath(os.path.join("meta", "suggestions.json"))
        with open(suggestions_file, "w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save suggestions: {e}")
        return False

def init_suggestions_file():
    """Initialize suggestions file if it doesn't exist"""
    try:
        suggestions_file = os.path.abspath(os.path.join("meta", "suggestions.json"))
        if not os.path.exists(suggestions_file):
            os.makedirs(os.path.dirname(suggestions_file), exist_ok=True)
            save_suggestions([])
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize suggestions file: {e}")
        return False

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
    """Manages system suggestions with autonomous learning capabilities."""

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None,
                 mission_manager_instance: Optional[Any] = None,
                 prompt_manager_load_prompt_func: Optional[Callable[[], str]] = None,
                 executor_module_for_lazy_load: Optional[Any] = None):
        """Initialize the SuggestionEngine with autonomous capabilities."""
        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_FALLBACK_LOGGER
        self.query_llm = query_llm_func if query_llm_func else _CLASS_FALLBACK_QUERY_LLM
        self.mission_manager = mission_manager_instance
        self._prompt_manager_load_prompt_func = prompt_manager_load_prompt_func
        self._executor_module = executor_module_for_lazy_load

        # Initialize configuration
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
        self.autonomous_approval_threshold = sugg_cfg.get("autonomous_approval_threshold", 0.9)
        self.max_auto_suggestions_per_hour = sugg_cfg.get("max_auto_suggestions_per_hour", 5)

        # Initialize paths - ensure we're not doubling up meta directory
        workspace_root = os.path.dirname(os.path.abspath(__file__))
        self.meta_dir = os.path.join(workspace_root, self.config.get("meta_dir", "meta"))
        self.suggestions_file_path = os.path.join(self.meta_dir, "suggestions.json")
        self.goals_file_path = os.path.join(self.meta_dir, "goals.json")
        self.tool_registry_file_path = os.path.join(self.meta_dir, "tool_registry.json")
        self.conversation_review_log_file_path = os.path.join(self.meta_dir, "conversation_review_log.json")

        # Ensure meta directory exists 
        os.makedirs(self.meta_dir, exist_ok=True)

        # Initialize suggestions file if it doesn't exist
        if not os.path.exists(self.suggestions_file_path):
            self.save_suggestions([])

        # Advanced features
        self.last_auto_suggestions = []
        self.suggestion_patterns = {}
        self.failed_suggestions = {}
        self.context_effectiveness = {}
        
        # Initialize systems
        self._ensure_meta_dir()
        self.init_suggestions_file()
        self.ai_core_model_name = self.config.get("llm_model_name", "fallback_model_suggestion_engine_class")

    def _lazy_load_executor(self) -> Optional[Any]:
        """Lazy load the executor module if needed."""
        if self._executor_module:
            return self._executor_module
            
        try:
            import importlib
            self._executor_module = importlib.import_module("executor")
            self.logger("DEBUG", "(SuggestionEngine) Lazy loaded executor module")
            return self._executor_module
        except ImportError:
            self.logger("ERROR", "(SuggestionEngine) Failed to lazy-load executor module")
            return None

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

    def _create_suggestion_object(self, suggestion_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a standardized suggestion object from generated text and context.
        """
        timestamp_now = datetime.now(timezone.utc).isoformat()
        suggestion_id = f"sugg_{uuid.uuid4().hex[:8]}" # Generate a unique ID

        # Basic suggestion structure
        suggestion_obj = {
            "id": suggestion_id,
            "timestamp": timestamp_now,
            "suggestion": suggestion_text,
            "status": "pending",  # Default status for new AI suggestions
            "priority": context.get("derived_priority", 5), # You might derive priority from context
            "origin": context.get("origin_details", "ai_generated_via_generate_ai_suggestion"),
            "source_model": self.ai_core_model_name, # Assuming self.ai_core_model_name is set
            "related_thread_id": context.get("related_thread_id", ""),
            "details": {
                "generation_context_summary": {k: str(v)[:200] + '...' if len(str(v)) > 200 else v 
                                               for k, v in context.items()}, # Example detail
                "generation_method": "autonomous_generation" # Or based on context
            },
            "approved_by": None,
            "rejected_by": None,
            "rejection_reason": None,
            "modification_history": []
        }
        
        self.logger("INFO", f"(SuggestionEngine class) Created new suggestion object ID: {suggestion_id} with text: \"{suggestion_text[:50]}...\"")
        return suggestion_obj

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


    def generate_ai_suggestion(self, autonomous_mode: bool = False) -> Optional[Dict[str, Any]]:
        """Generate an AI suggestion with enhanced autonomous capabilities."""
        if autonomous_mode and not self._check_autonomous_generation_limits():
            return None

        # Load executor if needed
        executor_module = self._lazy_load_executor()
        
        # Check suggestion limits
        all_suggestions = self.load_suggestions(load_all_history=True)
        pending_suggestions = [s for s in all_suggestions if isinstance(s, dict) and s.get("status") == "pending"]
        if len(pending_suggestions) >= self.max_pending_suggestions_in_system:
            self.logger("DEBUG", "(SuggestionEngine) Max pending suggestions reached")
            return None

        try:
            # Get base context
            context = self._gather_suggestion_context(executor_module)
            
            # Add learning context
            if self.suggestion_patterns:
                context["successful_patterns"] = self._get_successful_patterns()
                
            # Generate suggestion
            suggestion_text = self._generate_suggestion_text(context)
            if not suggestion_text:
                return None
                
            # Create suggestion
            suggestion = self._create_suggestion_object(suggestion_text, context)
            
            # Handle autonomous mode
            if autonomous_mode:
                quality_score = self._evaluate_suggestion_quality(suggestion, context)
                suggestion["quality_score"] = quality_score
                suggestion["autonomous"] = True
                
                if quality_score >= self.autonomous_approval_threshold:
                    suggestion["auto_approved"] = True
                    self._update_learning_patterns(suggestion, context)
            
            return suggestion
            
        except Exception as e:
            self.logger("ERROR", f"Error generating suggestion: {e}")
            return None

    def _gather_suggestion_context(self, executor_module: Optional[Any] = None) -> Dict[str, Any]:
        """Gather context for suggestion generation with enhanced pattern recognition."""
        context = {}
        
        # Get mission focus
        context["mission_focus"] = self._get_mission_focus()
        
        # Get active goals
        if executor_module and hasattr(executor_module, 'reprioritize_and_select_next_goal'):
            context["active_goals"] = self._get_active_goals(executor_module)
            
        # Get performance data
        context["performance"] = self._gather_performance_data()
        
        # Get tool insights
        context["tools"] = self._gather_tool_insights()
        
        # Get conversation patterns
        context["conversations"] = self._gather_conversation_patterns()
        
        # Get learning insights
        context["learning"] = {
            "successful_patterns": list(self.suggestion_patterns.keys()),
            "failed_approaches": list(self.failed_suggestions.keys()),
            "effective_contexts": self.context_effectiveness
        }
        
        return context

    def _generate_suggestion_text(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate suggestion text using the enhanced context."""
        try:
            # Load all suggestions for deduplication
            all_suggestions = self.load_suggestions(load_all_history=True)
            pending_suggestions = [s for s in all_suggestions if isinstance(s, dict) and s.get("status") == "pending"]

            # Get base system prompt
            system_prompt = self._get_system_prompt_for_suggestion_generation()
            
            # Build context for LLM
            prompt_parts = []
            
            # Add mission focus
            mission_focus = context.get("mission_focus", "Mission focus not available")
            prompt_parts.append(f"Current Mission Focus Areas: {mission_focus}")
            
            # Add active goals context
            if active_goals := context.get("active_goals"):
                prompt_parts.append("Current Active Goals:\n" + "\n".join(f"- {g}" for g in active_goals))
            
            # Add performance context
            if perf := context.get("performance"):
                prompt_parts.append("Recent Performance Insights:")
                for metric, value in perf.items():
                    prompt_parts.append(f"- {metric}: {value}")
            
            # Add tool insights
            if tools := context.get("tools"):
                prompt_parts.append("Tool Usage Patterns:")
                for tool in tools:
                    prompt_parts.append(f"- {tool}")
            
            # Add learning insights
            if learning := context.get("learning"):
                if patterns := learning.get("successful_patterns"):
                    prompt_parts.append("Successful Suggestion Patterns:")
                    for pattern in patterns[:3]:  # Top 3 patterns
                        prompt_parts.append(f"- {pattern}")
                        
            # Generate suggestion
            prompt = "\n\n".join(prompt_parts)
            suggestion_text = self.query_llm(prompt, system_prompt)
            
            # Validate suggestion
            if suggestion_text:
                # Check for similarity with existing suggestions
                if self._is_suggestion_too_similar(suggestion_text, all_suggestions[:self.max_recent_suggestions_for_de_dup_context*2], self.similarity_threshold_for_de_dup):
                    self.logger("DEBUG", "Generated suggestion too similar to existing ones")
                    return None
                    
                # Check for duplicates in pending suggestions
                current_pending_texts = {s.get('suggestion','').lower() for s in pending_suggestions if isinstance(s.get('suggestion'), str)}
                if suggestion_text.lower() in current_pending_texts:
                    self.logger("DEBUG", "Generated suggestion duplicates a pending suggestion")
                    return None
                    
                return suggestion_text
                
        except Exception as e:
            self.logger("ERROR", f"Error generating suggestion text: {e}")
            
        return None

    def _get_mission_focus(self) -> str:
        """Get current mission focus areas."""
        try:
            if self.mission_manager and hasattr(self.mission_manager, 'load_mission'):
                mission_data = self.mission_manager.load_mission()
                if isinstance(mission_data, dict):
                    focus_areas = mission_data.get("current_focus_areas", [])
                    if focus_areas:
                        return ", ".join(focus_areas)
        except Exception as e:
            self.logger("WARNING", f"Error getting mission focus: {e}")
            
        return "Mission focus not available"

    def _get_active_goals(self, executor_module: Any) -> List[str]:
        """Get current active goals."""
        active_goals = []
        try:
            all_goals = self._load_json_for_context(self.goals_file_path, [], "all_goals_for_active_check")
            if isinstance(all_goals, list) and all_goals:
                if hasattr(executor_module, 'reprioritize_and_select_next_goal'):
                    top_goal = executor_module.reprioritize_and_select_next_goal(all_goals)
                    if isinstance(top_goal, dict):
                        desc_key = "description" if "description" in top_goal else "goal"
                        active_goals.append(f"{top_goal.get(desc_key, 'N/A')} (Priority: {top_goal.get('priority', 'N/A')})")
        except Exception as e:
            self.logger("WARNING", f"Error getting active goals: {e}")
            
        return active_goals

    def _gather_performance_data(self) -> Dict[str, Any]:
        """Gather recent performance metrics."""
        metrics = {
            "goal_completion_rate": 0.0,
            "tool_success_rate": 0.0,
            "suggestion_approval_rate": 0.0
        }
        
        try:
            # Calculate goal completion rate
            goals = self._load_json_for_context(self.goals_file_path, [], "goals_for_metrics")
            if isinstance(goals, list) and goals:
                completed = sum(1 for g in goals if isinstance(g, dict) and g.get("status") == "completed")
                metrics["goal_completion_rate"] = completed / len(goals)
            
            # Calculate tool success rate
            tools = self._load_json_for_context(self.tool_registry_file_path, [], "tools_for_metrics")
            if isinstance(tools, list) and tools:
                success_rate = 1 - sum(t.get("failure_count", 0) for t in tools) / max(sum(t.get("total_runs", 1) for t in tools), 1)
                metrics["tool_success_rate"] = success_rate
            
            # Calculate suggestion approval rate
            all_suggs = self.load_suggestions(load_all_history=True)
            if all_suggs:
                approved = sum(1 for s in all_suggs if isinstance(s, dict) and s.get("status") in ["approved_by_user", "approved_by_ai"])
                metrics["suggestion_approval_rate"] = approved / len(all_suggs)
                
        except Exception as e:
            self.logger("ERROR", f"Error gathering performance data: {e}")
            
        return metrics

    def _gather_tool_insights(self) -> List[str]:
        """Gather insights about tool usage and performance."""
        insights = []
        try:
            tools = self._load_json_for_context(self.tool_registry_file_path, [], "tool_registry")
            if isinstance(tools, list):
                sorted_tools = sorted(
                    [t for t in tools if isinstance(t, dict)],
                    key=lambda x: x.get("last_run_time") or x.get("last_updated") or "",
                    reverse=True
                )
                
                for tool in sorted_tools[:self.max_recent_tools_context]:
                    name = tool.get("name", "N/A")
                    runs = tool.get("total_runs", 0)
                    fails = tool.get("failure_count", 0)
                    insights.append(f"{name} (Runs: {runs}, Fails: {fails})")
                    
        except Exception as e:
            self.logger("ERROR", f"Error gathering tool insights: {e}")
            
        return insights

    def _gather_conversation_patterns(self) -> List[Dict[str, Any]]:
        """Analyze conversation patterns for context."""
        patterns = []
        try:
            convs = self._load_json_for_context(self.conversation_review_log_file_path, [], "conversation_review_log")
            if isinstance(convs, list):
                for conv in convs[-self.max_conversation_snippets_for_context:]:
                    if isinstance(conv, dict) and "conversation_snippet" in conv:
                        patterns.append({
                            "timestamp": conv.get("timestamp"),
                            "thread_id": conv.get("thread_id"),
                            "summary": conv.get("summary", "No summary available")
                        })
        except Exception as e:
            self.logger("ERROR", f"Error gathering conversation patterns: {e}")
            
        return patterns

    def get_pending_suggestions_summary_for_prompt(self, max_suggestions: Optional[int] = None) -> str: # Add max_suggestions argument
        """Return a summary string of pending suggestions for prompt context."""
        try:
            suggestions = self.load_suggestions() if hasattr(self, 'load_suggestions') else []
            pending = [s for s in suggestions if s.get('status') == 'pending']
            if not pending:
                return "No pending suggestions."

            # Use max_suggestions if provided
            num_to_display = len(pending)
            if max_suggestions is not None and max_suggestions > 0:
                num_to_display = min(len(pending), max_suggestions)

            summary_lines = [f"- {s.get('suggestion', '[No text]')} (added {s.get('created_at', 'unknown')})" for s in pending[:num_to_display]]
            return f"Pending suggestions ({len(pending)} total, showing up to {num_to_display}):\n" + "\n".join(summary_lines)
        except Exception as e:
            return f"[Error generating pending suggestions summary: {e}]"

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