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
from logger_utils import log # Changed import
try:
    from notifier_module import get_current_version
except ImportError:
    log("ERROR", "CRITICAL ERROR (ai_core): notifier_module.py not found or get_current_version missing.")
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
    # Required methods for each component
    _REQUIRED_COMPONENT_METHODS = {
        # ... other components from the previous comprehensive list I provided
        "notifier": ["log_update"],
        "prompt_manager": ["get_full_system_prompt", "update_operational_prompt_text", "auto_update_prompt"],
        "mission_manager": ["load_mission", "save_mission", "get_mission_statement_for_prompt"],
        "suggestion_engine": ["load_suggestions", "save_suggestions", "generate_ai_suggestion"],
        "goal_monitor": ["get_active_goals", "get_pending_goals", "set_active_goal", "create_new_goal"],
        "planner": ["decide_next_action"],
        "evaluator": ["perform_evaluation_cycle"],
        "executor": ["evaluate_goals", "process_single_goal"],
        "goal_worker": ["start_worker", "stop", "set_pause"], # Corrected and added set_pause
        "tool_builder": ["build_tool"],
        "tool_runner": ["run_tool_safely"]
    }

    def __init__(self, 
                 config_file: Optional[str] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None):
        """Initialize AICore with components and memory systems."""
        
        # Initialize basic services
        self.logger = logger_func or log_background_message
        self.query_llm = query_llm_internal
        self._query_llm_for_planner_adapter = query_llm_internal
        self.config = self._load_config(config_file)
        
        # Initialize memory and state management
        self._initialize_memory_system()
        
        try:
            # Initialize components in dependency order
            self._init_notifier()
            self._init_tool_builder()
            self._init_tool_runner()
            self._init_prompt_manager()
            self._init_mission_manager()
            self._init_suggestion_engine()
            self._init_goal_monitor()
            self._init_planner()
            self._init_evaluator()

            # Initialize executor before goal worker
            from executor import Executor
            self.executor = Executor(
                config=self.config.get("executor_config", {}),
                logger_func=self.logger,
                query_llm_func=self.query_llm,
                mission_manager_instance=self.mission_manager,
                notifier_instance=self.notifier,
                tool_builder_instance=self.tool_builder,
                tool_runner_instance=self.tool_runner
            )
            self.logger("DEBUG", "(AICore Init) Executor initialized successfully.")
            
            self._init_goal_worker()
            
            # Verify final state
            self._verify_components_after_init()
            
        except Exception as e:
            self.logger("CRITICAL", f"Failed to initialize AICore: {str(e)}")
            traceback.print_exc()
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

    def _init_notifier(self):
        """Initialize notifier without fallbacks."""
        try:
            from notifier_module import Notifier
            self.notifier = Notifier(
                config=self.config,
                logger_func=self.logger,
                query_llm_func=self.query_llm
            )
            self.logger("DEBUG", "(AICore Init) Notifier initialized successfully.")
        except ImportError:
            raise ImportError("Critical component Notifier could not be initialized")

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
            meta_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meta")
            os.makedirs(meta_dir, exist_ok=True)

            goals_file = os.path.join(meta_dir, "goals.json")
            active_goal_file = os.path.join(meta_dir, "active_goal.json")

            self.goal_monitor = GoalMonitor(
                goals_file=self.config.get("goals_file", goals_file),
                active_goal_file=self.config.get("active_goal_file", active_goal_file),
                suggestion_engine_instance=self.suggestion_engine,
                logger=self.logger  # Fixed: Changed logger_func to logger
            )
            self.Goal = Goal
            self.logger("DEBUG", "(AICore Init) GoalMonitor initialized successfully.")
        except ImportError:
            raise ImportError("Critical component GoalMonitor could not be initialized")
            
    def _init_planner(self):
        """Initialize Planner without fallbacks."""
        try:
            from planner_module import Planner

            meta_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meta")
            tool_registry_file = os.path.join(meta_dir, "tool_registry.json")
            
            self.planner = Planner(
                query_llm_func=self._query_llm_for_planner_adapter,
                prompt_manager=self.prompt_manager,
                goal_monitor=self.goal_monitor,
                mission_manager=self.mission_manager,
                suggestion_engine=self.suggestion_engine,
                config=self.config.get("planner_config", {}),
                logger=self.logger,
                tool_registry_or_path=self.config.get("tool_registry_file", tool_registry_file)
            )
            self.logger("DEBUG", "(AICore Init) Planner initialized successfully.")
        except ImportError:
            raise ImportError("Critical component Planner could not be initialized")
            
    def _init_evaluator(self):
        """Initialize Evaluator without fallbacks."""
        try:
            from evaluator import Evaluator
            self.evaluator = Evaluator(
                config=self.config.get("evaluator_config", {}),
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
                config=self.config.get("goal_worker_config", {}),
                logger_func=self.logger,
                executor_instance=self.executor,
                evaluator_instance=self.evaluator,
                suggestion_engine_instance=self.suggestion_engine
            )
            self.logger("DEBUG", "(AICore Init) GoalWorker initialized successfully.")
        except ImportError:
            raise ImportError("Critical component GoalWorker could not be initialized")
    
    def _init_tool_builder(self):
        """Initialize tool builder without fallbacks."""
        try:
            from tool_builder_module import ToolBuilder
            self.tool_builder = ToolBuilder(
                config=self.config,
                logger_func=self.logger,
                query_llm_func=self.query_llm
            )
            self.logger("DEBUG", "(AICore Init) ToolBuilder initialized successfully.")
        except ImportError:
            raise ImportError("Critical component ToolBuilder could not be initialized")

    def _init_tool_runner(self):
        """Initialize tool runner without fallbacks."""
        try:
            from tool_runner import ToolRunner
            self.tool_runner = ToolRunner(
                config=self.config,
                logger_func=self.logger,
                query_llm_func=self.query_llm
            )
            self.logger("DEBUG", "(AICore Init) ToolRunner initialized successfully.")
        except ImportError:
            raise ImportError("Critical component ToolRunner could not be initialized")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f: return json.load(f)
            except Exception as e:
                self.logger("ERROR", f"Failed to load config {config_file}: {e}. Using defaults.")
        self.logger("INFO", f"Config file {config_file} not found. Using default settings.")

        meta_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meta")
        os.makedirs(meta_dir, exist_ok=True)

        return {
            "llm_model_name": MODEL_NAME_AICORE,
            "prompt_files_dir": "prompts",
            "mission_file": os.path.join(meta_dir, "mission.json"),
            "suggestions_file": os.path.join(meta_dir, "suggestions.json"),
            "goals_file": os.path.join(meta_dir, "goals.json"),
            "active_goal_file": os.path.join(meta_dir, "active_goal.json"),
            "evaluation_log_file": os.path.join(meta_dir, "evaluation_log.json"),
            "tool_registry_file": os.path.join(meta_dir, "tool_registry.json"),
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
        """Adapter method to handle LLM queries specifically for the Planner component."""
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
        """Update the system status with current metrics and health information."""
        _FB_SUFFIX_AICORE = "_FB_AICORE_INIT"
        is_evaluator_fb = not self.evaluator or (hasattr(self.evaluator, '__class__') and self.evaluator.__class__.__name__.endswith(_FB_SUFFIX_AICORE))

        if not is_evaluator_fb:
            if hasattr(self.evaluator, 'get_recent_evaluations_summary_for_planner'):
                self.system_status["recent_performance_summary"] = self.evaluator.get_recent_evaluations_summary_for_planner(count=5)
            elif hasattr(self.evaluator, 'get_recent_evaluations'):
                 recent_evals = self.evaluator.get_recent_evaluations(count=10)
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
            self.system_status["active_goals_count"] = len(self.goal_monitor.get_active_goals())
        else: self.system_status["active_goals_count"] = 0

        is_se_fb = not self.suggestion_engine or (hasattr(self.suggestion_engine, '__class__') and self.suggestion_engine.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        if not is_se_fb and hasattr(self.suggestion_engine, 'load_suggestions'):
            all_suggs = self.suggestion_engine.load_suggestions(load_all_history=True)
            self.system_status["pending_suggestions_count"] = len([s for s in all_suggs if isinstance(s, dict) and s.get("status")=="pending"])
        else: self.system_status["pending_suggestions_count"] = 0

    def _load_persistent_memory(self):
        """Load persistent memory from storage if enabled."""
        try:
            meta_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meta")
            memory_file = os.path.join(meta_dir, "memory.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    stored_memory = json.load(f)
                    self.conversation_history = stored_memory.get("conversation_history", {})
                    self.knowledge_graph = stored_memory.get("knowledge_graph", {})
                    self.learning_history = stored_memory.get("learning_history", [])
                    self.logger("DEBUG", "(AICore) Successfully loaded persistent memory.")
        except Exception as e:
            self.logger("ERROR", f"(AICore) Error loading persistent memory: {e}. Using empty memory.")
            self.conversation_history = {}
            self.knowledge_graph = {}
            self.learning_history = []

    def _save_persistent_memory(self):
        """Save memory to persistent storage if enabled."""
        if not self.memory_persistence:
            return
        try:
            meta_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meta")
            os.makedirs(meta_dir, exist_ok=True)
            memory_file = os.path.join(meta_dir, "memory.json")
            memory_data = {
                "conversation_history": self.conversation_history,
                "knowledge_graph": self.knowledge_graph,
                "learning_history": self.learning_history
            }
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2)
            self.logger("DEBUG", "(AICore) Successfully saved persistent memory.")
        except Exception as e:
            self.logger("ERROR", f"(AICore) Error saving persistent memory: {e}")

    def _verify_core_components(self):
        """Verify that all core components are present and properly initialized."""
        try:
            for component_name, required_methods in self._REQUIRED_COMPONENT_METHODS.items():
                component = getattr(self, component_name, None)
                if not component:
                    self.logger("ERROR", f"Component {component_name} is missing")
                    return False
                
                # Check for fallback implementations
                if hasattr(component, '__class__') and component.__class__.__name__.endswith('_FB'):
                    self.logger("ERROR", f"Component {component_name} is using fallback implementation")
                    return False
                    
                # Check required methods
                for method_name in required_methods:
                    if not hasattr(component, method_name):
                        self.logger("ERROR", f"Component {component_name} missing required method: {method_name}")
                        return False
                        
                # Additional sanity check that the methods are callable
                for method_name in required_methods:
                    if not callable(getattr(component, method_name)):
                        self.logger("ERROR", f"Component {component_name}'s {method_name} is not callable")
                        return False
                        
            self.logger("INFO", "All core components verified successfully")
            return True
            
        except Exception as e:
            self.logger("ERROR", f"Component verification failed: {str(e)}")
            return False

    def get_response_for_user_input_async(self, user_input: str, system_prompt_base: str,
                                        current_thread_id: str, callback_to_console: Optional[Callable[[str, str], None]] = None):
        """Get an asynchronous response for user input.
        
        Args:
            user_input (str): The user's input text
            system_prompt_base (str): Base system prompt
            current_thread_id (str): Thread ID for the conversation
            callback_to_console (callable, optional): Callback for response handling
        """
        _FB_SUFFIX_AICORE = "_FB_AICORE_INIT"
        is_planner_fb = not self.planner or (hasattr(self.planner, '__class__') and self.planner.__class__.__name__.endswith(_FB_SUFFIX_AICORE))
        if is_planner_fb:
            self.logger("CRITICAL", f"AICore cannot process input for thread '{current_thread_id}'; Planner component is a FALLBACK.")
            err_msg = "[Error: AI Core's Planner component is not available. Please check startup logs.]"
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
            action, action_details = self.planner.decide_next_action(planner_thread_history, self.system_status)
            self.logger("INFO", f"Planner for thread '{current_thread_id[-6:]}' decided action: {action}, Details: {str(action_details)[:200]}...")
            action_details = action_details if isinstance(action_details, dict) else {}

            if action == "respond_to_user":
                final_response_to_user = action_details.get("response_text", "I've processed that, but I'm not sure how best to reply just now.")
                if action_details.get("internal_response"):
                    self.logger("INFO", f"Internal AI action result for thread '{current_thread_id[-6:]}': {action_details['internal_response']}")
            elif action == "create_goal":
                desc_from_planner = action_details.get('description', 'Unnamed objective from AI planning.')
                final_response_to_user = f"I've identified a new objective: '{desc_from_planner[:70]}...'. It has been added to my goals."
            elif action == "manage_suggestions":
                final_response_to_user = action_details.get("response_text", f"I've reviewed my internal suggestions. {action_details.get('internal_response', 'Outcome logged internally.')}")
                self.logger("INFO", f"Suggestion management outcome for thread '{current_thread_id[-6:]}': {action_details.get('internal_response')}")
            elif action == "execute_tool":
                tool_name_to_exec = action_details.get("tool_name", "unknown_tool")
                final_response_to_user = f"I need to use a tool ('{tool_name_to_exec}') to help with that. (Autonomous tool execution pathway is under development)."
                self.logger("INFO", f"Planner suggests direct execution of tool '{tool_name_to_exec}' for thread '{current_thread_id[-6:]}'.")
            else:
                final_response_to_user = action_details.get("response_text", "I understand, but I'm not sure what specific action to take right now.")

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

    def _add_to_thread_history(self, thread_id: str, role: str, text: str):
        """Add a message to the conversation history for a thread."""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        self.conversation_history[thread_id].append({
            "role": role,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": text
        })
        max_hist = self.config.get("max_conversation_history_per_thread", 20)
        if len(self.conversation_history[thread_id]) > max_hist:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_hist:]

    def _get_thread_history_for_planner(self, thread_id: str) -> List[Dict[str,str]]:
        """Get conversation history for a thread for use by the planner."""
        return self.conversation_history.get(thread_id, [])

    def handle_command(self, command_text: str, current_thread_id: str) -> str:
        """Handle a user command.
        
        Args:
            command_text (str): The command text from the user
            current_thread_id (str): The current conversation thread ID
            
        Returns:
            str: The response to the command
        """
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
        elif command == "/reflect" and self.planner:
            self.logger("INFO", f"User initiated reflection cycle for thread {current_thread_id}.")
            planner_hist_reflect = self._get_thread_history_for_planner(current_thread_id)
            status_for_reflect = self.system_status.copy()
            status_for_reflect["current_action_intent"] = "user_forced_reflection"
            action_taken, details_from_planner = self.planner.decide_next_action(planner_hist_reflect, status_for_reflect)
            if action_taken == "respond_to_user" and details_from_planner and details_from_planner.get("response_text"):
                response = details_from_planner.get("response_text", "Reflection initiated, specific outcome pending.")
            elif details_from_planner and details_from_planner.get("internal_response"):
                response = f"Reflection cycle initiated. AI decided: {details_from_planner.get('internal_response')}"
            else:
                response = f"Reflection cycle processed. AI internal action: {action_taken}. No direct user message formulated by planner for this."
            self.logger("INFO", f"Reflection outcome: Action={action_taken}, Details={str(details_from_planner)[:200]}")
        elif command == "/status":
            self._update_system_status()
            response = (f"System Status:\n"
                     f"  Version: {self.version}\n"
                     f"  LLM Model: {self.current_llm_model}\n"
                     f"  Current Time (UTC): {self.system_status.get('current_time')}\n"
                     f"  Tool Success Rate (evaluator): {self.system_status.get('tool_success_rate', 'N/A')}\n"
                     f"  Active Goals: {self.system_status.get('active_goals_count', 0)}\n"
                     f"  Pending Suggestions: {self.system_status.get('pending_suggestions_count', 0)}")

        return response

# --- Module-level Interface Functions ---
_ai_core_instance_singleton: Optional[AICore] = None # Ensure this module-level variable is defined

def initialize_ai_core_singleton(config_file="config.json", logger_override=None):
    """Initialize the AI Core singleton instance and return it.
    
    Args:
        config_file (str): Path to configuration file
        logger_override (callable, optional): Override for the default logger
    Returns:
        Optional[AICore]: The AICore instance if successful, else None.
    """
    global _ai_core_instance_singleton, MODEL_NAME # MODEL_NAME is also a global you modify
    
    # Use a reliable logger function from the start
    effective_logger = logger_override if callable(logger_override) else log_background_message
    
    if _ai_core_instance_singleton is None:
        effective_logger("INFO", "(ai_core_wrapper) AICore singleton is None. Attempting creation.")
        try:
            instance = AICore(config_file=config_file, logger_func=effective_logger)
            # If AICore.__init__ completes, instance is created.
            _ai_core_instance_singleton = instance # Assign to the global variable
            
            # This access might be too early if current_llm_model is set later in AICore init,
            # but your original code had it here. Consider moving if problematic.
            if hasattr(instance, 'current_llm_model'):
                 MODEL_NAME = instance.current_llm_model
            else:
                 effective_logger("WARNING", "(ai_core_wrapper) AICore instance created but 'current_llm_model' attribute not found immediately.")
                 # MODEL_NAME would retain its default or previously set value.
            
            effective_logger("INFO", "(ai_core_wrapper) AICore singleton instance CREATED and assigned.")
            return _ai_core_instance_singleton # Explicitly return the created instance
            
        except Exception as e_init_aicore_s:
            effective_logger("CRITICAL", f"(ai_core_wrapper) Failed to create AICore singleton instance: {e_init_aicore_s}\n{traceback.format_exc()}")
            _ai_core_instance_singleton = None # Ensure it's None on failure
            return None # Explicitly return None on failure
    else:
        effective_logger("INFO", "(ai_core_wrapper) AICore singleton instance already existed. Returning existing instance.")
        return _ai_core_instance_singleton # Return the existing instance

def get_response_async(user_input: str, system_prompt_base: str, callback:Optional[Callable[[str,str],None]]=None, stream_response: bool = True, thread_id: Optional[str] = None):
    """Get an asynchronous response from the AI Core.
    
    Args:
        user_input (str): The user's input text
        system_prompt_base (str): Base system prompt to use
        callback (callable, optional): Callback function for response handling
        stream_response (bool): Whether to stream the response
        thread_id (str, optional): Thread ID for conversation tracking
    """
    global AI_STREAMING_ENABLED_AICORE, _ai_core_instance_singleton
    AI_STREAMING_ENABLED_AICORE = stream_response

    if _ai_core_instance_singleton is None:
        log_background_message("ERROR", "(ai_core_wrapper) AICore singleton is None for get_response_async. Attempting lazy init.")
        initialize_ai_core_singleton()
        if _ai_core_instance_singleton is None:
            err_msg = "[FATAL ERROR: AI Core could not be initialized for get_response_async.]"
            log_background_message("CRITICAL", err_msg)
            response_queue_aicore.put((user_input, err_msg, True, str(uuid.uuid4()), thread_id))
            if callback: callback(user_input, err_msg)
            return

    ai_instance_to_use = _ai_core_instance_singleton
    final_thread_id = thread_id if thread_id else str(uuid.uuid4())
    if not thread_id:
        log_background_message("WARNING", f"(ai_core_wrapper) No thread_id from console. New thread: ..{final_thread_id[-6:]}")

    threading.Thread(
        target=ai_instance_to_use.get_response_for_user_input_async,
        args=(user_input, system_prompt_base, final_thread_id, callback),
        daemon=True, name=f"AICoreUserInputThread-{final_thread_id[-6:]}"
    ).start()

# Export module-level queues
response_queue = response_queue_aicore
log_message_queue = log_message_queue_aicore
MODEL_NAME = MODEL_NAME_AICORE # Ensure this is accessible

if __name__ == "__main__":
    log("INFO", "--- Testing AICore Standalone (basic init and LLM call) ---")
    def test_logger_main(level, message):
        log(level.upper(), f"[{level} - MAIN_TEST] {message}")

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

    log("INFO", "--- AICore __main__ Test Complete ---")