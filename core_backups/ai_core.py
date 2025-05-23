# Self-Aware/ai_core.py
import json
import time
import os
from datetime import datetime, timezone
import uuid
import re
import queue
import threading
import traceback
import requests # Make sure requests is imported for perform_sync_llm_query

from typing import List, Dict, Optional, Any, Callable, Tuple, Union, TYPE_CHECKING

# --- Conditional Imports for Type Checking ---
if TYPE_CHECKING:
    from prompt_manager import PromptManager
    from goal_monitor import GoalMonitor, Goal
    from planner_module import Planner
    from suggestion_engine import SuggestionEngine, ActorType, SuggestionStatus
    from evaluator import Evaluator
    from mission_manager import MissionManager
    from goal_worker import GoalWorker

# --- Runtime Imports of truly BASE utilities (that don't import AICore or other major components) ---
try:
    from notifier_module import get_current_version
except ImportError:
    print("CRITICAL ERROR (ai_core): notifier_module.py not found or get_current_version missing.")
    def get_current_version(): return "N/A (notifier_module missing)"

# --- Global Queues and Settings ---
OLLAMA_URL_AICORE = "http://localhost:11434/api/generate"
MODEL_NAME_AICORE = "gemma3:4B" # Default, overridden by config if present

response_queue_aicore = queue.Queue()
log_message_queue_aicore = queue.Queue()

AI_STREAMING_ENABLED_AICORE = True

# --- Module-level Logger ---
def log_background_message(level: str, message: str):
    """
    Logs a message to the background message queue.
    This function is imported and used by other modules.
    """
    log_message_queue_aicore.put((level.upper(), message))

# --- Module-level LLM Query Function ---
def query_llm_internal(
    prompt_text: str,
    system_prompt_override: Optional[str] = None,
    raw_output: bool = False,
    timeout: int = 300,
    model_override: Optional[str] = None,
    ollama_url_override: Optional[str] = None,
    options_override: Optional[Dict[str, Any]] = None
) -> str:
    final_prompt_for_api = ""
    if raw_output:
        final_prompt_for_api = prompt_text
    else:
        if system_prompt_override:
            final_prompt_for_api = f"{system_prompt_override}\n\nUser: {prompt_text}\nAI:"
        else:
            final_prompt_for_api = f"User: {prompt_text}\nAI:"

    current_model = model_override if model_override else MODEL_NAME_AICORE
    current_url = ollama_url_override if ollama_url_override else OLLAMA_URL_AICORE

    default_options = {
        "temperature": 0.7,
        "num_predict": 512
    }
    final_options = default_options.copy()
    if options_override:
        final_options.update(options_override)

    payload = {
        "model": current_model,
        "prompt": final_prompt_for_api,
        "stream": False,
        "options": final_options
    }

    try:
        response = requests.post(current_url, json=payload, timeout=timeout)
        response.raise_for_status()
        full_response_data = response.json()
        if "response" in full_response_data:
            return full_response_data["response"].strip()
        else:
            log_background_message("ERROR", f"(ai_core.query_llm_internal) LLM response missing 'response' key. URL: {current_url}, Data: {str(full_response_data)[:200]}")
            return "[Error: Malformed LLM response from API]"
    except requests.exceptions.Timeout:
        log_background_message("ERROR", f"(ai_core.query_llm_internal) LLM request to {current_url} timed out after {timeout}s.")
        return f"[Error: LLM request timed out after {timeout}s]"
    except requests.exceptions.RequestException as e:
        log_background_message("ERROR", f"(ai_core.query_llm_internal) LLM request to {current_url} failed: {e}")
        return f"[Error: LLM request failed: {e}]"
    except Exception as e_sync_llm:
        log_background_message("CRITICAL", f"(ai_core.query_llm_internal) Unexpected error during LLM call to {current_url}: {e_sync_llm}\n{traceback.format_exc()}")
        return f"[Error: Unexpected LLM call failure: {e_sync_llm}]"


class AICore:
    """Core AI system that manages all components and operations.
    
    This class requires all core components to be properly initialized and fails fast if any are missing.
    No fallback implementations are allowed for critical components.
    """
    
    # Class constants
    _FB_SUFFIX_AICORE = "_FB_AICORE_INIT"
    _REQUIRED_COMPONENT_METHODS = {
        'planner': ['decide_next_action'],
        'suggestion_engine': ['generate_ai_suggestion', 'init_suggestions_file'],
        'goal_monitor': ['get_active_goal'],
        'mission_manager': ['get_current_mission'],
        'prompt_manager': ['get_system_prompt', 'auto_update_prompt'],
        'goal_worker': ['start_worker', 'background_goal_loop'],
        'evaluator': ['perform_evaluation_cycle']
    }

    def __init__(self, config_file: str = "config.json"):
        """Initialize the AI Core with strict component requirements.
        
        All core components must be properly initialized without fallbacks.
        Raises ImportError if any critical component is missing or using a fallback.
        
        Args:
            config_file (str): Path to the configuration file
        """
        # Initialize system status and recovery tracking
        self.system_status = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "crashes_recovered": 0,
            "last_error": None,
            "tool_success_rate": 1.0,
            "errors_last_cycle": 0,
            "component_health": {},
            "memory_usage": {},
            "offline_mode": False,
            "autonomous_fixes_applied": 0
        }
        
        # Initialize basic services
        self.logger = log_background_message
        self.query_llm = query_llm_internal
        self._query_llm_for_planner_adapter = query_llm_internal
        self.config = self._load_config(config_file)
        
        # Initialize memory and state management
        self._initialize_memory_system()
        
        try:
            # Initialize all core components with strict requirements
            self._init_prompt_manager()
            self._init_mission_manager()
            self._init_suggestion_engine()
            self._init_goal_monitor()
            self._init_planner()
            self._init_evaluator()
            self._init_goal_worker()
            
            # Verify final state
            self._verify_components_after_init()
            
            # Initialize autonomous systems after core components are ready
            self._init_autonomous_systems()
        except Exception as e:
            self.system_status["last_error"] = str(e)
            raise

    def _initialize_memory_system(self):
        """Initialize the persistent memory system."""
        self.memory_config = self.config.get("system", {})
        self.max_memory_entries = self.memory_config.get("max_memory_entries", 1000)
        self.memory_persistence = self.memory_config.get("memory_persistence", True)
        
        self.conversation_history: Dict[str, List[Dict[str,str]]] = {}
        self.knowledge_graph = {}  # For tool/goal/concept linking
        self.learning_history = []  # Track system improvements
        
        if self.memory_persistence:
            self._load_persistent_memory()

    def _init_autonomous_systems(self):
        """Initialize systems for autonomous operation."""
        self.auto_recovery_enabled = self.memory_config.get("auto_recovery_enabled", True)
        self.offline_mode_enabled = self.memory_config.get("offline_mode_enabled", False)
        self.crash_recovery_attempts = self.memory_config.get("crash_recovery_attempts", 3)
        
        # Initialize auto-suggestion system
        suggestion_config = self.config.get("suggestion_engine_config", {})
        self.auto_suggestion_enabled = suggestion_config.get("enabled", True)
        self.autonomous_approval_threshold = suggestion_config.get("autonomous_approval_threshold", 0.9)
        
        # Initialize learning system
        reflection_config = self.config.get("reflection_config", {})
        self.learning_rate = reflection_config.get("learning_rate", 0.1)
        self.min_confidence_for_auto_fix = reflection_config.get("min_confidence_for_auto_fix", 0.8)

    def _verify_components_after_init(self):
        """Verify that all components are properly initialized without using fallbacks.
        
        This method is called at the end of initialization to ensure all components
        are properly initialized and configured.
        
        Raises:
            ImportError: If any critical component is missing or using a fallback implementation
        """
        self.conversation_history: Dict[str, List[Dict[str,str]]] = {}
        self.system_status: Dict[str, Any] = {"tool_success_rate": 1.0, "errors_last_cycle": 0}
        self.version = get_current_version()
        self.current_llm_model = self.config.get("llm_model_name", MODEL_NAME_AICORE)
        global MODEL_NAME
        MODEL_NAME = self.current_llm_model

        # Verify core components
        if not self._verify_core_components():
            raise ImportError("Critical AI Core components failed verification. System cannot operate without fallbacks.")

        self.logger("INFO", f"AI Core initialized successfully. Version: {self.version}, LLM: {self.current_llm_model}")

    def query_llm_wrapper(self, 
                         prompt_text: str,
                         system_prompt_override: Optional[str] = None,
                         raw_output: bool = False,
                         timeout: int = 300,
                         model_override: Optional[str] = None,
                         ollama_url_override: Optional[str] = None,
                         options_override: Optional[Dict[str, Any]] = None) -> str:
        """Instance method wrapper around the module-level query_llm_internal function"""
        return query_llm_internal(
            prompt_text=prompt_text,
            system_prompt_override=system_prompt_override,
            raw_output=raw_output,
            timeout=timeout,
            model_override=model_override,
            ollama_url_override=ollama_url_override,
            options_override=options_override
        )

    def _init_prompt_manager(self):
        """Initialize PromptManager without fallbacks."""
        try:
            from prompt_manager import PromptManager
            self.prompt_manager = PromptManager(
                config=self.config,
                logger_func=self.logger
            )
            self.logger("DEBUG", "(AICore Init) PromptManager initialized successfully.")
        except ImportError:
            raise ImportError("Critical component PromptManager could not be initialized")

    def _init_mission_manager(self):
        """Initialize MissionManager without fallbacks."""
        try:
            from mission_manager import MissionManager
            self.mission_manager = MissionManager(
                config=self.config,
                logger_func=self.logger,
                prompt_manager=self.prompt_manager,
                query_llm_func=self.query_llm
            )
            self.logger("DEBUG", "(AICore Init) MissionManager initialized successfully.")
        except ImportError:
            raise ImportError("Critical component MissionManager could not be initialized")
            
    def _init_suggestion_engine(self):
        """Initialize SuggestionEngine without fallbacks."""
        try:
            from suggestion_engine import SuggestionEngine, ActorType, SuggestionStatus
            self.suggestion_engine = SuggestionEngine(
                config=self.config,
                logger_func=self.logger,
                query_llm_func=self.query_llm
            )
            self.ActorType = ActorType
            self.SuggestionStatus = SuggestionStatus
            self.logger("DEBUG", "(AICore Init) SuggestionEngine initialized successfully.")
        except ImportError:
            raise ImportError("Critical component SuggestionEngine could not be initialized")
            
    def _init_goal_monitor(self):
        """Initialize GoalMonitor without fallbacks."""
        try:
            from goal_monitor import GoalMonitor, Goal
            self.goal_monitor = GoalMonitor(
                goals_file=self.config.get("goals_file", os.path.join("meta", "goals.json")),
                active_goal_file=self.config.get("active_goal_file", os.path.join("meta", "active_goal.json")),
                suggestion_engine_instance=self.suggestion_engine,
                logger=self.logger
            )
            self.Goal = Goal
            self.logger("DEBUG", "(AICore Init) GoalMonitor initialized successfully.")
        except ImportError:
            raise ImportError("Critical component GoalMonitor could not be initialized")
            
    def _init_planner(self):
        """Initialize Planner without fallbacks."""
        try:
            from planner_module import Planner
            self.planner = Planner(
                query_llm_func=self._query_llm_for_planner_adapter,
                prompt_manager=self.prompt_manager,
                goal_monitor=self.goal_monitor,
                mission_manager=self.mission_manager,
                suggestion_engine=self.suggestion_engine,
                config=self.config,
                logger=self.logger,
                tool_registry_or_path=self.config.get("tool_registry_file", os.path.join("meta","tool_registry.json"))
            )
            self.logger("DEBUG", "(AICore Init) Planner initialized successfully.")
        except ImportError:
            raise ImportError("Critical component Planner could not be initialized")
            
    def _init_evaluator(self):
        """Initialize Evaluator without fallbacks."""
        try:
            from evaluator import Evaluator
            self.evaluator = Evaluator(
                config=self.config.get("Evaluator", self.config.get("evaluator_config", {})),
                logger_func=self.logger,
                query_llm_func=self.query_llm,
                mission_manager_instance=self.mission_manager
            )
            self.logger("DEBUG", "(AICore Init) Evaluator initialized successfully.")
        except ImportError:
            raise ImportError("Critical component Evaluator could not be initialized")
            
    def _init_goal_worker(self):
        """Initialize GoalWorker without fallbacks."""
        try:
            from goal_worker import GoalWorker
            self.goal_worker = GoalWorker(
                config=self.config.get("GoalWorker", self.config.get("goal_worker_config", {})),
                logger_func=self.logger,
                executor_instance=self.executor,
                evaluator_instance=self.evaluator,
                suggestion_engine_instance=self.suggestion_engine
            )
            self.logger("DEBUG", "(AICore Init) GoalWorker initialized successfully.")
        except ImportError:
            raise ImportError("Critical component GoalWorker could not be initialized")
        

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f: return json.load(f)
            except Exception as e:
                self.logger("ERROR", f"Failed to load config {config_file}: {e}. Using defaults.")
        self.logger("INFO", f"Config file {config_file} not found. Using default settings.")
        return {
            "llm_model_name": MODEL_NAME_AICORE, "prompt_files_dir": "prompts",
            "mission_file": os.path.join("meta","mission.json"),
            "suggestions_file": os.path.join("meta","suggestions.json"),
            "goals_file": os.path.join("meta","goals.json"),
            "active_goal_file": os.path.join("meta","active_goal.json"),
            "evaluation_log_file": os.path.join("meta","evaluation_log.json"),
            "tool_registry_file": os.path.join("meta", "tool_registry.json"),
            "max_conversation_history_per_thread": 20,
            "planner_config": {
                "planner_context_conversation_history_length": 5,
                "planner_context_max_pending_suggestions": 3,
                "planner_context_recent_evaluations_count": 2,
                "planner_context_max_tools": 5,
                "planner_max_tokens": 800,
                "planner_temperature": 0.5,
                "planner_llm_timeout": 300
            },
            "goal_worker_config": {},
            "suggestion_engine_enabled": True,
            "reflection_interval_interactions": 10,
            "internal_llm_max_tokens": 300,
            "internal_llm_temperature": 0.6,
            "ollama_api_url": OLLAMA_URL_AICORE,
            "streaming_chunk_size": 30,
            "streaming_delay_ms": 30
        }

    def _query_llm_for_planner_adapter(self,
                                   prompt_text: str,
                                   system_prompt_override_ignored: Optional[str] = None,
                                   raw_output_ignored: bool = True,
                                   timeout_override_from_planner: Optional[int] = None,
                                   max_tokens_override: Optional[int] = None,
                                   temperature_override: Optional[float] = None
                                   ) -> str:
        planner_cfg = self.config.get("planner_config", {})
        final_max_tokens = max_tokens_override if max_tokens_override is not None else planner_cfg.get("planner_max_tokens", 800)
        final_temperature = temperature_override if temperature_override is not None else planner_cfg.get("planner_temperature", 0.5)
        final_timeout = timeout_override_from_planner if timeout_override_from_planner is not None else planner_cfg.get("planner_llm_timeout", 300)

        llm_options = {
            "temperature": final_temperature,
            "num_predict": final_max_tokens
        }
        return query_llm_internal(
            prompt_text=prompt_text,
            raw_output=True,
            timeout=final_timeout,
            model_override=self.current_llm_model,
            ollama_url_override=self.config.get("ollama_api_url", OLLAMA_URL_AICORE),
            options_override=llm_options
        )

    def _update_system_status(self):
        _FB_SUFFIX_AICORE = "_FB_AICORE_INIT"
        is_evaluator_fb = not self.evaluator or (hasattr(self.evaluator, '__class__') and self.evaluator.__class__.__name__.endswith(_FB_SUFFIX_AICORE))

        if not is_evaluator_fb:
            if hasattr(self.evaluator, 'get_recent_evaluations_summary_for_planner'):
                self.system_status["recent_performance_summary"] = self.evaluator.get_recent_evaluations_summary_for_planner(count=5) # type: ignore
            elif hasattr(self.evaluator, 'get_recent_evaluations'):
                 recent_evals = self.evaluator.get_recent_evaluations(count=10) # type: ignore
                 if recent_evals and isinstance(recent_evals, list):
                     valid_evals = [e for e in recent_evals if isinstance(e, dict)]
                     successes = sum(1 for e in valid_evals if e.get("success", False))
                     self.system_status["tool_success_rate"] = successes / len(valid_evals) if valid_evals else 1.0
                 else: self.system_status["tool_success_rate"] = 1.0
            else: self.system_status["tool_success_rate"] = 0.0
        else:
            self.system_status["recent_performance_summary"] = "Evaluator not available."
            self.system_status["tool_success_rate"] = 0.0

        self.system_status["current_time"] = datetime.now(timezone.utc).isoformat()

        is_gm_fb = not self.goal_monitor or (hasattr(self.goal_monitor, '__class__') and self.goal_monitor.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        if not is_gm_fb and hasattr(self.goal_monitor, 'get_active_goals'):
            self.system_status["active_goals_count"] = len(self.goal_monitor.get_active_goals()) # type: ignore
        else: self.system_status["active_goals_count"] = 0

        is_se_fb = not self.suggestion_engine or (hasattr(self.suggestion_engine, '__class__') and self.suggestion_engine.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        if not is_se_fb and hasattr(self.suggestion_engine, 'load_suggestions'):
            all_suggs = self.suggestion_engine.load_suggestions(load_all_history=True) # type: ignore
            self.system_status["pending_suggestions_count"] = len([s for s in all_suggs if isinstance(s, dict) and s.get("status")=="pending"])
        else: self.system_status["pending_suggestions_count"] = 0


    def _add_to_thread_history(self, thread_id: str, role: str, text: str):
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        self.conversation_history[thread_id].append({"role": role, "timestamp": datetime.now(timezone.utc).isoformat(), "content": text})
        max_hist = self.config.get("max_conversation_history_per_thread", 20)
        if len(self.conversation_history[thread_id]) > max_hist:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_hist:]

    def _get_thread_history_for_planner(self, thread_id: str) -> List[Dict[str,str]]:
        return self.conversation_history.get(thread_id, [])


    def handle_command(self, command_text: str, current_thread_id: str) -> str:
        parts = command_text.lower().strip().split()
        command = parts[0]
        args = parts[1:]
        response = f"Unknown command: {command}. Type /help for options."

        _FB_SUFFIX_AICORE = "_FB_AICORE_INIT"
        is_gm_fb = not self.goal_monitor or (hasattr(self.goal_monitor, '__class__') and self.goal_monitor.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        is_se_fb = not self.suggestion_engine or (hasattr(self.suggestion_engine, '__class__') and self.suggestion_engine.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        is_mm_fb = not self.mission_manager or (hasattr(self.mission_manager, '__class__') and self.mission_manager.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        is_pl_fb = not self.planner or (hasattr(self.planner, '__class__') and self.planner.__class__.__name__.endswith(_FB_SUFFIX_AICORE))

        if command in ["/goals", "/active_goal", "/create_goal"] and is_gm_fb:
            return "[Error: GoalMonitor not available/fallback. Cannot process goal commands.]"
        if command == "/approve_suggestion" and is_gm_fb :
            return "[Error: GoalMonitor (for suggestion approval) not available/fallback.]"
        if command in ["/suggestions", "/reject_suggestion"] and is_se_fb:
             return "[Error: SuggestionEngine not available/fallback. Cannot process suggestion commands.]"
        if command == "/mission" and is_mm_fb:
            return "[Error: MissionManager not available/fallback. Cannot process mission commands.]"
        if command == "/reflect" and is_pl_fb:
            return "[Error: Planner not available/fallback. Cannot process reflection command.]"

        if command == "/help":
            response = ("Available commands:\n"
                    "/goals [all|pending|active|archived <n>] - View goals.\n"
                    "/suggestions [all|pending] - View suggestions.\n"
                    "/approve_suggestion <id> - Approve a suggestion to become a goal.\n"
                    "/reject_suggestion <id> [reason] - Reject a suggestion.\n"
                    "/mission [view|set <new_mission>] - View or set AI mission.\n"
                    "/status - View system status.\n"
                    "/active_goal [set <id>|clear|view] - Manage the active goal.\n"
                    "/create_goal <description> --priority <1-10> --details <json_string> --type <type> --due <due_date_str> --thread <thread_id> - Create new goal.\n"
                    "/reflect - Force a reflection cycle (planner decides).")
        elif command == "/goals" and self.goal_monitor: # type: ignore
            filter_type = args[0] if args else "pending"
            goals_list: List[Any] = []
            header = ""
            if filter_type == "all": goals_list = self.goal_monitor.goals if hasattr(self.goal_monitor, 'goals') else []; header="All Goals:\n" # type: ignore
            elif filter_type == "pending": goals_list = self.goal_monitor.get_pending_goals(); header="Pending Goals (dependencies met):\n" # type: ignore
            elif filter_type == "active": goals_list = self.goal_monitor.get_active_goals(include_paused=True); header="Active/Paused Goals:\n" # type: ignore
            elif filter_type == "archived":
                count = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
                goals_list = self.goal_monitor.get_archived_goals(last_n=count); header = f"Last {count} Archived Goals:\n" # type: ignore
            else: return f"Unknown goal filter: {filter_type}. Use all, pending, active, or archived [n]."
            if not goals_list: response = header + "(No goals in this category)"
            else: response = header + "".join([f"  ID: {g.id[-8:]} (P{g.priority}) Th:{g.thread_id[-6:]} [{g.status}] - {g.description[:70]}\n" for g in goals_list]) # type: ignore
        elif command == "/suggestions" and self.suggestion_engine: # type: ignore
            filter_type = args[0] if args else "pending"
            suggs_list: List[Any] = []
            header_s = ""
            all_suggs_loaded = self.suggestion_engine.load_suggestions(load_all_history=(filter_type == "all")) # type: ignore
            if filter_type == "pending":
                suggs_list = [s for s in all_suggs_loaded if isinstance(s, dict) and s.get("status") == "pending"]
                header_s="Pending Suggestions:\n"
            elif filter_type == "all": suggs_list = all_suggs_loaded; header_s="All Suggestions:\n"
            else: return f"Unknown suggestion filter: {filter_type}. Use 'all' or 'pending'."
            if not suggs_list: response = header_s + "(No suggestions in this category)"
            else: response = header_s + "".join([f"  ID: {s.get('id', 'N/A')[-8:]} (P{s.get('priority')}) [{s.get('status')}] - {s.get('suggestion','')[:70]}\n" for s in suggs_list])
        elif command == "/approve_suggestion" and self.goal_monitor: # type: ignore
            if not args: return "Usage: /approve_suggestion <suggestion_id_or_timestamp>"
            s_id_or_ts = args[0]
            goal = self.goal_monitor.create_goal_from_suggestion(suggestion_identifier=s_id_or_ts, approved_by="user") # type: ignore
            response = f"Suggestion {s_id_or_ts} approved by user. New goal created: {goal.id[-8:] if goal else 'N/A'}" if goal else f"Failed to approve suggestion {s_id_or_ts}." # type: ignore
        elif command == "/reject_suggestion" and self.suggestion_engine: # type: ignore
            if not args: return "Usage: /reject_suggestion <suggestion_id_or_timestamp> [reason...]"
            s_id_or_ts = args[0]
            reason = " ".join(args[1:]) if len(args) > 1 else "User rejected."
            rejected = self.suggestion_engine.archive_suggestion_as_rejected_by_actor( # type: ignore
                suggestion_identifier=s_id_or_ts, reason=reason, rejected_by="user" # type: ignore
            )
            response = f"Suggestion {s_id_or_ts} rejected by user." if rejected else f"Failed to reject suggestion {s_id_or_ts}."
        elif command == "/mission" and self.mission_manager: # type: ignore
            mission_data = self.mission_manager.load_mission() # type: ignore
            if not args or args[0] == "view":
                response = f"Current Mission: {mission_data.get('identity_statement', 'Not set.')}\nDirective: {mission_data.get('core_directive', 'N/A')}"
            elif args[0] == "set" and len(args) > 1:
                new_mission_text = " ".join(args[1:])
                current_m = self.mission_manager.load_mission() # type: ignore
                current_m["core_directive"] = new_mission_text
                self.mission_manager.save_mission(current_m) # type: ignore
                response = f"Mission core_directive updated to: {new_mission_text}"
            else: response = "Usage: /mission [view|set <new_mission_description_for_directive>]"
        elif command == "/status":
            self._update_system_status()
            response = (f"System Status:\n"
                    f"  Version: {self.version}\n"
                    f"  LLM Model: {self.current_llm_model}\n"
                    f"  Current Time (UTC): {self.system_status.get('current_time')}\n"
                    f"  Tool Success Rate (evaluator): {self.system_status.get('tool_success_rate', 'N/A')}\n"
                    f"  Active Goals: {self.system_status.get('active_goals_count', 0)}\n"
                    f"  Pending Suggestions: {self.system_status.get('pending_suggestions_count', 0)}")
        elif command == "/active_goal" and self.goal_monitor: # type: ignore
            if not args or args[0] == "view":
                active_g = self.goal_monitor.get_active_goal() # type: ignore
                response = f"Current Active Goal: {active_g.id[-8:]} - {active_g.description}" if active_g else "No active goal." # type: ignore
            elif args[0] == "set" and len(args) > 1:
                g_id = args[1]
                new_active = self.goal_monitor.set_active_goal(g_id) # type: ignore
                response = f"Active goal set to {new_active.id[-8:]}." if new_active else f"Failed to set active goal to {g_id}." # type: ignore
            elif args[0] == "clear":
                self.goal_monitor.set_active_goal(None) # type: ignore
                response = "Active goal cleared."
            else: response = "Usage: /active_goal [view|set <goal_id>|clear]"
        elif command == "/create_goal" and self.goal_monitor: # type: ignore
            description_parts = []
            params: Dict[str, Any] = {"details": {}, "created_by": "user", "thread_id_param": current_thread_id}
            i = 0
            while i < len(args):
                if args[i].startswith("--"): break
                description_parts.append(args[i]); i += 1
            params["description"] = " ".join(description_parts)
            while i < len(args):
                arg_key = args[i]
                if arg_key == "--priority" and i + 1 < len(args):
                    try: params["priority"] = int(args[i+1]); i+=2; continue
                    except ValueError: return "Invalid priority value for --priority."
                elif arg_key == "--details" and i + 1 < len(args):
                    try: params["details"] = json.loads(args[i+1]); i+=2; continue
                    except json.JSONDecodeError: return "Invalid JSON for --details."
                elif arg_key == "--type" and i + 1 < len(args):
                    params["goal_type"] = args[i+1]; i+=2; continue
                elif arg_key == "--due" and i + 1 < len(args):
                    params["due_date_str"] = args[i+1]; i+=2; continue
                elif arg_key == "--thread" and i + 1 < len(args):
                    params["thread_id_param"] = args[i+1]; i+=2; continue
                else:
                    self.logger("WARNING", f"Unknown parameter in /create_goal: {arg_key}")
                    i += 1
            if not params.get("description"): return "Usage: /create_goal <description> [--priority P] [--details JSON] [--type T] [--due DUE_STR] [--thread TID]"
            goal = self.goal_monitor.create_new_goal(**params) # type: ignore
            response = f"New goal created: {goal.id[-8:]}" if goal else "Failed to create goal." # type: ignore
        elif command == "/reflect" and self.planner: # type: ignore
            self.logger("INFO", f"User initiated reflection cycle for thread {current_thread_id}.")
            planner_hist_reflect = self._get_thread_history_for_planner(current_thread_id)
            status_for_reflect = self.system_status.copy()
            status_for_reflect["current_action_intent"] = "user_forced_reflection"
            action_taken, details_from_planner = self.planner.decide_next_action(planner_hist_reflect, status_for_reflect) # type: ignore
            if action_taken == "respond_to_user" and details_from_planner and details_from_planner.get("response_text"):
                response = details_from_planner.get("response_text", "Reflection initiated, specific outcome pending.")
            elif details_from_planner and details_from_planner.get("internal_response"):
                response = f"Reflection cycle initiated. AI decided: {details_from_planner.get('internal_response')}"
            else:
                response = f"Reflection cycle processed. AI internal action: {action_taken}. No direct user message formulated by planner for this."
            self.logger("INFO", f"Reflection outcome: Action={action_taken}, Details={str(details_from_planner)[:200]}")
        return response

    def get_response_for_user_input_async(self, user_input: str, system_prompt_base: str,
                                           current_thread_id: str, callback_to_console: Optional[Callable[[str, str], None]] = None):
        _FB_SUFFIX_AICORE = "_FB_AICORE_INIT"
        is_planner_fb = not self.planner or (hasattr(self.planner, '__class__') and self.planner.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        if is_planner_fb:
            self.logger("CRITICAL", f"AICore cannot process input for thread '{current_thread_id}'; Planner component is a FALLBACK.")
            err_msg = "[Error: AI Core's Planner component is not available. Please check startup logs for import errors.]"
            response_queue_aicore.put((user_input, err_msg, True, str(uuid.uuid4()), current_thread_id))
            if callback_to_console: callback_to_console(user_input, err_msg)
            return

        self._add_to_thread_history(current_thread_id, "user", user_input)
        final_response_to_user = ""

        if user_input.startswith("/"):
            final_response_to_user = self.handle_command(user_input, current_thread_id)
        else:
            self._update_system_status()
            planner_thread_history = self._get_thread_history_for_planner(current_thread_id)
            action, action_details = self.planner.decide_next_action(planner_thread_history, self.system_status) # type: ignore
            self.logger(f"INFO: Planner for thread '{current_thread_id[-6:]}' decided action: {action}, Details: {str(action_details)[:200]}...")
            action_details = action_details if isinstance(action_details, dict) else {}

            if action == "respond_to_user":
                final_response_to_user = action_details.get("response_text", "I've processed that, but I'm not sure how best to reply just now.")
                if action_details.get("internal_response"):
                    self.logger(f"INFO: Internal AI action result for thread '{current_thread_id[-6:]}': {action_details['internal_response']}")
            elif action == "create_goal":
                desc_from_planner = action_details.get('description', 'Unnamed objective from AI planning.')
                final_response_to_user = f"I've identified a new objective: '{desc_from_planner[:70]}...'. It has been added to my goals."
            elif action == "manage_suggestions":
                final_response_to_user = action_details.get("response_text",
                                                        f"I've reviewed my internal suggestions. {action_details.get('internal_response', 'Outcome logged internally.')}")
                self.logger(f"INFO: Suggestion management outcome for thread '{current_thread_id[-6:]}': {action_details.get('internal_response')}")
            elif action == "execute_tool":
                tool_name_to_exec = action_details.get("tool_name", "unknown_tool")
                final_response_to_user = f"I need to use a tool ('{tool_name_to_exec}') to help with that. (Autonomous tool execution pathway is under development)."
                self.logger(f"INFO: Planner suggests direct execution of tool '{tool_name_to_exec}' for thread '{current_thread_id[-6:]}'.")
            elif action == "no_action_needed":
                reason = action_details.get("reason", "No specific action seems necessary from my side regarding that.")
                final_response_to_user = f"AI: {reason}" if not reason.startswith("AI:") else reason
            else:
                self.logger(f"INFO: Planner action '{action}' for thread '{current_thread_id[-6:]}' taken internally. Formulating general response.")
                mission_description = "Mission Description Unavailable (MissionManager Fallback)"
                is_mm_fb = not self.mission_manager or (hasattr(self.mission_manager, '__class__') and self.mission_manager.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
                if not is_mm_fb:
                    mission_data = self.mission_manager.get_mission() # type: ignore
                    mission_description = mission_data.get("description", "Mission N/A") if mission_data else "Mission data N/A"

                general_response_prompt_context = {
                    "user_input": user_input, "conversation_history": planner_thread_history,
                    "mission": mission_description,
                    "ai_action_taken_internally": action,
                    "ai_action_details_summary": str(action_details)[:200]
                }
                is_pm_fb = not self.prompt_manager or (hasattr(self.prompt_manager, '__class__') and self.prompt_manager.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
                if not is_pm_fb:
                    final_prompt_for_general_response = self.prompt_manager.get_filled_template( # type: ignore
                        "general_response_after_planner_action", general_response_prompt_context
                    )
                else:
                     final_prompt_for_general_response = f"User asked: {user_input}. My internal thought process led to the action: {action}. Formulate a suitable response to the user."
                final_response_to_user = self._query_llm_for_planner_adapter(
                    prompt_text=final_prompt_for_general_response,
                    max_tokens_override=self.config.get("internal_llm_max_tokens", 250),
                    temperature_override=self.config.get("internal_llm_temperature", 0.7)
                )

        stream_id = str(uuid.uuid4())
        if AI_STREAMING_ENABLED_AICORE and len(final_response_to_user) > 50:
            chunk_size = self.config.get("streaming_chunk_size", 30)
            for i in range(0, len(final_response_to_user), chunk_size):
                chunk = final_response_to_user[i:i+chunk_size]
                is_final_chunk = (i + chunk_size >= len(final_response_to_user))
                response_queue_aicore.put((user_input, chunk, is_final_chunk, stream_id, current_thread_id))
                if not is_final_chunk: time.sleep(self.config.get("streaming_delay_ms", 30) / 1000.0)
        else:
            response_queue_aicore.put((user_input, final_response_to_user, True, stream_id, current_thread_id))

        self._add_to_thread_history(current_thread_id, "AI", final_response_to_user)
        if callback_to_console:
            callback_to_console(user_input, final_response_to_user)


    def background_work_cycle(self):
        """Background work cycle for AICore. Manages autonomous operations that require all critical components.
        
        The cycle performs:
        1. System health verification
        2. Memory management
        3. Autonomous suggestion processing
        4. Learning and self-improvement
        5. Error recovery if needed
        
        Returns early if any critical components are missing or using fallbacks.
        """
        self.logger("INFO", "(AICore BG) Starting background work cycle...")

        try:
            # Verify components and system health
            if not self._verify_core_components():
                self.logger("CRITICAL", "(AICore BG) Cannot start background cycle - core component verification failed")
                return

            # Update system status and perform health checks
            self._update_system_status()
            self._check_memory_usage()
            
            # Handle offline mode if needed
            if self.offline_mode_enabled and not self._check_online_status():
                self._handle_offline_mode()
                return

            # Process active goals and suggestions
            active_goal = self.goal_monitor.get_active_goal()
            
            if active_goal is None:
                # System is idle - check for pending suggestions
                if self.system_status.get("pending_suggestions_count", 0) > 0:
                    self._process_pending_suggestions()
                elif self.auto_suggestion_enabled:
                    self._generate_autonomous_suggestion()

            # Perform learning and self-improvement
            self._perform_autonomous_learning()

            # Check for and apply any pending system improvements
            if self.system_status.get("pending_improvements", False):
                self._apply_system_improvements()

        except Exception as e:
            self.logger("CRITICAL", f"(AICore BG) Error during background work cycle: {e}\n{traceback.format_exc()}")
            if self.auto_recovery_enabled:
                self._attempt_error_recovery()
        finally:
            self._save_system_state()
            self.logger("INFO", "(AICore BG) Background work cycle finished.")

    def _process_pending_suggestions(self):
        """Process pending suggestions with the planner."""
        self.logger("INFO", "(AICore BG) System idle, reviewing pending suggestions.")
        idle_context = [{
            "role": "system", 
            "content": "System is idle. Review pending suggestions for autonomous management."
        }]
        status_for_mgmt = self.system_status.copy()
        status_for_mgmt["current_action_intent"] = "autonomously_manage_suggestions"
        
        action, details = self.planner.decide_next_action(idle_context, status_for_mgmt)
        
        if action == "manage_suggestions" and details:
            confidence = details.get("confidence", 0)
            if confidence >= self.autonomous_approval_threshold:
                self._auto_approve_suggestion(details)
            else:
                self.logger("INFO", f"(AICore BG) Suggestion confidence {confidence} below threshold {self.autonomous_approval_threshold}")

    def _perform_autonomous_learning(self):
        """Perform autonomous learning and self-improvement."""
        if not hasattr(self, '_last_reflection_time'):
            self._last_reflection_time = 0

        current_time = time.time()
        reflection_interval = self.config.get("reflection_config", {}).get("interval_interactions", 600)
        
        if current_time - self._last_reflection_time >= reflection_interval:
            self.logger("INFO", "(AICore BG) Initiating autonomous learning cycle")
            
            try:
                # Update prompts based on performance
                if self.prompt_manager and hasattr(self.prompt_manager, 'auto_update_prompt'):
                    self.prompt_manager.auto_update_prompt()
                
                # Review and optimize tools
                self._optimize_tools()
                
                # Update system knowledge
                self._update_knowledge_graph()
                
                self._last_reflection_time = current_time
                
            except Exception as e:
                self.logger("ERROR", f"(AICore BG) Error during learning cycle: {e}")

    def _attempt_error_recovery(self):
        """Attempt to recover from errors autonomously."""
        if self.system_status["crashes_recovered"] >= self.crash_recovery_attempts:
            self.logger("CRITICAL", "(AICore BG) Maximum recovery attempts reached")
            return

        try:
            self.system_status["crashes_recovered"] += 1
            
            # Verify component health
            unhealthy_components = []
            for component, methods in self._REQUIRED_COMPONENT_METHODS.items():
                component_instance = getattr(self, component, None)
                if not component_instance or not all(hasattr(component_instance, m) for m in methods):
                    unhealthy_components.append(component)
            
            if unhealthy_components:
                self.logger("WARNING", f"(AICore BG) Unhealthy components found: {unhealthy_components}")
                self._reinitialize_components(unhealthy_components)
            
            # Verify memory system
            if not self._verify_memory_integrity():
                self._restore_memory_from_backup()
            
        except Exception as e:
            self.logger("CRITICAL", f"(AICore BG) Recovery attempt failed: {e}")

    def _update_knowledge_graph(self):
        """Update the system's knowledge graph with new learnings."""
        try:
            # Analyze recent goals and their relationships
            goals_data = self._load_json_for_knowledge("goals.json", [])
            for goal in goals_data:
                if isinstance(goal, dict):
                    self._add_to_knowledge_graph(
                        node_type="goal",
                        node_id=goal.get("goal_id"),
                        properties=goal,
                        relationships={"thread": goal.get("thread_id")}
                    )
            
            # Analyze tool usage patterns
            tool_data = self._load_json_for_knowledge("tool_registry.json", [])
            for tool in tool_data:
                if isinstance(tool, dict):
                    self._add_to_knowledge_graph(
                        node_type="tool",
                        node_id=tool.get("name"),
                        properties=tool,
                        relationships={"used_by": tool.get("used_by", [])}
                    )
            
            self._save_knowledge_graph()
            
        except Exception as e:
            self.logger("ERROR", f"(AICore BG) Error updating knowledge graph: {e}")

    def _optimize_tools(self):
        """Analyze and optimize tool performance."""
        try:
            tool_config = self.config.get("tool_builder_config", {})
            if not tool_config.get("auto_optimization", True):
                return
                
            tool_registry = self._load_json_for_knowledge("tool_registry.json", [])
            for tool in tool_registry:
                if isinstance(tool, dict):
                    failure_rate = tool.get("failure_count", 0) / max(tool.get("total_runs", 1), 1)
                    if failure_rate > 0.2:  # 20% failure rate threshold
                        self._schedule_tool_optimization(tool)
                        
        except Exception as e:
            self.logger("ERROR", f"(AICore BG) Error during tool optimization: {e}")

# --- For console_ai.py to use (Interface Functions) ---
_ai_core_instance_singleton: Optional[AICore] = None

def initialize_ai_core_singleton(config_path="config.json", logger_override=None):
    global _ai_core_instance_singleton, MODEL_NAME
    if _ai_core_instance_singleton is None:
        logger_to_use = (
            logger_override
            if logger_override
            else lambda *parts: print(
                "[" + (parts[0] if parts else "INFO") + "]",
                *parts[1:],
            )
        )

        try:
            _ai_core_instance_singleton = AICore(config_file=config_path, logger_func=logger_to_use)
            MODEL_NAME = _ai_core_instance_singleton.current_llm_model
            logger_to_use("INFO", "(ai_core_wrapper) AICore singleton instance CREATED.")
        except Exception as e_init_aicore_s:
            logger_to_use("CRITICAL", f"(ai_core_wrapper) Failed to create AICore singleton instance: {e_init_aicore_s}\n{traceback.format_exc()}")
            _ai_core_instance_singleton = None

def get_response_async(user_input: str, system_prompt_with_history: str, callback:Optional[Callable[[str,str],None]]=None, stream_response: bool = True, associated_thread_id: Optional[str] = None):
    global AI_STREAMING_ENABLED_AICORE, _ai_core_instance_singleton
    AI_STREAMING_ENABLED_AICORE = stream_response

    if _ai_core_instance_singleton is None:
        log_background_message("ERROR", "(ai_core_wrapper) AICore singleton is None for get_response_async. Attempting lazy init.")
        initialize_ai_core_singleton()
        if _ai_core_instance_singleton is None:
            err_msg = "[FATAL ERROR: AI Core could not be initialized for get_response_async.]"
            log_background_message("CRITICAL", err_msg)
            response_queue_aicore.put((user_input, err_msg, True, str(uuid.uuid4()), associated_thread_id))
            if callback: callback(user_input, err_msg)
            return

    ai_instance_to_use = _ai_core_instance_singleton
    final_thread_id = associated_thread_id if associated_thread_id else str(uuid.uuid4())
    if not associated_thread_id:
        log_background_message("WARNING", f"(ai_core_wrapper) No thread_id from console. New thread: ..{final_thread_id[-6:]}")

    threading.Thread(
        target=ai_instance_to_use.get_response_for_user_input_async,
        args=(user_input, system_prompt_with_history, final_thread_id, callback),
        daemon=True, name=f"AICoreUserInputThread-{final_thread_id[-6:]}"
    ).start()

response_queue = response_queue_aicore
log_message_queue = log_message_queue_aicore
MODEL_NAME = MODEL_NAME_AICORE # Ensure this is accessible

if __name__ == "__main__":
    print("--- Testing AICore Standalone (basic init and LLM call) ---")
    def test_logger_main(level, message): print(f"[{level} - MAIN_TEST] {message}")

    if not os.path.exists("config.json"):
        with open("config.json", "w") as f_cfg_main:
            json.dump({"llm_model_name": "gemma3:4B", "ollama_api_url": OLLAMA_URL_AICORE, "prompt_files_dir": "prompts"}, f_cfg_main)
        test_logger_main("INFO", "Created dummy config.json for __main__ test.")

    if not os.path.exists("prompts"): os.makedirs("prompts", exist_ok=True)
    dummy_prompts_dir = "prompts"
    if not os.path.exists(os.path.join(dummy_prompts_dir, "default_system_prompt.txt")):
        with open(os.path.join(dummy_prompts_dir, "default_system_prompt.txt"), "w") as fdp: fdp.write("Fallback System Prompt.")
    if not os.path.exists(os.path.join(dummy_prompts_dir, "planner_decide_action.txt")):
        with open(os.path.join(dummy_prompts_dir, "planner_decide_action.txt"), "w") as fpa: fpa.write("Decide action.")


    initialize_ai_core_singleton(logger_override=test_logger_main)
    if _ai_core_instance_singleton:
        test_logger_main("INFO", "AICore singleton initialized successfully for __main__ test.")

        test_logger_main("INFO", "Testing query_llm_internal directly...")
        test_prompt = "What is the capital of France?"
        try:
            llm_direct_response = query_llm_internal(test_prompt, model_override="gemma3:4B")
            test_logger_main("INFO", f"Direct LLM response for '{test_prompt}': {llm_direct_response}")
        except Exception as e_main_q_llm:
            test_logger_main("ERROR", f"Error testing query_llm_internal: {e_main_q_llm}")
    else:
        test_logger_main("CRITICAL", "Failed to initialize AICore singleton for __main__ test.")

    print("--- AICore __main__ Test Complete ---")