# Self-Aware/goal_worker.py
import json
import os
import threading
import time
import traceback 
from typing import Dict, Optional, Any, Callable, List, Union # Added List, Callable
from datetime import datetime, timezone
from logger_utils import should_log # Added import

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_GOALWORKER_FALLBACK_LOGGER = lambda level, message: (print(f"[{level.upper()}] (GoalWorkerClassFallbackLog) {message}") if should_log(level.upper()) else None)

class GoalWorker:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 executor_instance: Optional[Any] = None,
                 evaluator_instance: Optional[Any] = None,
                 suggestion_engine_instance: Optional[Any] = None):

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_GOALWORKER_FALLBACK_LOGGER
        
        self.executor = executor_instance
        self.evaluator = evaluator_instance
        self.suggestion_engine = suggestion_engine_instance

        # Instance attributes for state
        self.pause_background = False
        self.stop_worker_flag = False
        self.worker_thread: Optional[threading.Thread] = None

        # File paths from config or defaults from original script
        self.meta_dir_path = self.config.get("meta_dir", "meta")
        self.active_goal_file_path = self.config.get("active_goal_file", os.path.join(self.meta_dir_path, "active_goal.json"))
        self.goals_file_path = self.config.get("goals_file", os.path.join(self.meta_dir_path, "goals.json"))

        # Worker loop parameters from config or defaults
        worker_cfg = self.config.get("goal_worker_config", {})
        self.suggestion_generation_interval = worker_cfg.get("suggestion_generation_interval", 180)
        self.worker_loop_interval = worker_cfg.get("worker_loop_interval", 7)
        self.periodic_log_interval = worker_cfg.get("periodic_log_interval", 600) # e.g., 10 minutes

        self._ensure_meta_dir()

    def _ensure_meta_dir(self): # Was _ensure_meta_dir_worker
        try:
            os.makedirs(self.meta_dir_path, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(GoalWorker class): Could not create meta directory {self.meta_dir_path}: {e}")

    def _save_json_worker(self, filepath: str, data: Union[Dict[str, Any], List[Any]]): # Type hint added
        self._ensure_meta_dir()
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            self.logger("ERROR", f"(GoalWorker class): Could not write to {filepath}: {e}")
        except Exception as e_gen:
            self.logger("ERROR", f"(GoalWorker class): Unexpected error saving {filepath}: {e_gen}\n{traceback.format_exc()}")

    def _load_json_worker(self, filepath: str, default_value: Optional[Union[List[Any], Dict[str, Any]]] = None) -> Any: # Type hint added
        self._ensure_meta_dir()
        actual_default = [] if default_value is None else default_value # Original default was []
        if not os.path.exists(filepath):
            return actual_default
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip(): return actual_default
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e: 
            self.logger("WARNING", f"(GoalWorker class): Error loading {filepath}: {e}. Returning default.")
            return actual_default
        except Exception as e_gen:
            self.logger("ERROR", f"(GoalWorker class): Unexpected error loading {filepath}: {e_gen}\n{traceback.format_exc()}. Returning default.")
            return actual_default

    def save_active_goal_worker(self, goal_obj: Optional[Dict[str, Any]]):
        """Saves the provided goal object as the active goal, or clears the file if None."""
        # This logic is exactly from your script.
        if goal_obj and isinstance(goal_obj, dict):
            self._save_json_worker(self.active_goal_file_path, goal_obj)
        else: 
            if os.path.exists(self.active_goal_file_path):
                try:
                    self._save_json_worker(self.active_goal_file_path, {}) 
                except OSError as e:
                    self.logger("ERROR", f"(GoalWorker class): Could not clear/remove {self.active_goal_file_path}: {e}")

    def background_goal_loop(self):
        self.logger("INFO","(GoalWorker class): Background goal worker started.")
        
        initialization_successful = True
        try:
            if self.executor and hasattr(self.executor, 'init_executor'):
                self.executor.init_executor() 
            else:
                self.logger("CRITICAL", "(GoalWorker class): Executor instance or init_executor method not available. Worker cannot initialize properly.")
                initialization_successful = False
            
            if self.suggestion_engine and hasattr(self.suggestion_engine, 'init_suggestions_file'):
                self.suggestion_engine.init_suggestions_file()
            else:
                self.logger("WARNING", "(GoalWorker class): SuggestionEngine instance or init_suggestions_file method not available.")

            if self.evaluator and hasattr(self.evaluator, '_ensure_meta_dir'): # Assuming evaluator has _ensure_meta_dir
                self.evaluator._ensure_meta_dir() # Call it if it exists
            elif not self.evaluator:
                self.logger("INFO", "(GoalWorker class): Evaluator instance not available, skipping its meta dir check.")

        except Exception as e_init:
            self.logger("CRITICAL", f"(GoalWorker class): Initialization error in worker: {e_init}\n{traceback.format_exc()}. Worker may not function correctly.")
            initialization_successful = False

        if not initialization_successful:
            self.logger("CRITICAL", "(GoalWorker class): Due to initialization failures, the worker will not proceed with its main loop.")
            self.stop_worker_flag = True

        last_suggestion_gen_time = 0
        last_full_eval_log_time = time.time() 

        while not self.stop_worker_flag:
            try:
                if self.pause_background:
                    time.sleep(5) 
                    continue

                # 1. Determine and save the current top-priority goal for display (active_goal.json)
                current_goals_list_for_display = self._load_json_worker(self.goals_file_path, [])
                top_priority_goal_for_display = None
                if self.executor and hasattr(self.executor, 'reprioritize_and_select_next_goal'):
                     top_priority_goal_for_display = self.executor.reprioritize_and_select_next_goal(current_goals_list_for_display)
                self.save_active_goal_worker(top_priority_goal_for_display)

                # 2. Call the executor's main evaluation/processing function
                if self.executor and hasattr(self.executor, 'evaluate_goals'):
                    self.logger("DEBUG", "(GoalWorker class): Calling executor.evaluate_goals().")
                    self.executor.evaluate_goals()
                else:
                    self.logger("ERROR", "(GoalWorker class): Executor instance or evaluate_goals method not available. Skipping execution phase.")
                    time.sleep(self.worker_loop_interval) 
                    continue
                
                # 3. After executor runs, run the evaluator
                if self.evaluator and hasattr(self.evaluator, 'perform_evaluation_cycle'):
                    self.logger("DEBUG", "(GoalWorker class): Calling evaluator.perform_evaluation_cycle().")
                    self.evaluator.perform_evaluation_cycle()
                elif not self.evaluator:
                    pass # Logged during init if evaluator is None
                else: 
                    self.logger("WARNING", "(GoalWorker class): Evaluator instance available, but perform_evaluation_cycle missing. Skipping evaluation.")

                # 4. If idle, consider suggestions
                current_goals_list_after_processing = self._load_json_worker(self.goals_file_path, [])
                next_actionable_goal_check = None
                if self.executor and hasattr(self.executor, 'reprioritize_and_select_next_goal'):
                    next_actionable_goal_check = self.executor.reprioritize_and_select_next_goal(current_goals_list_after_processing)

                if not next_actionable_goal_check: 
                    if self.suggestion_engine and hasattr(self.suggestion_engine, 'generate_ai_suggestion'):
                        if time.time() - last_suggestion_gen_time > self.suggestion_generation_interval:
                            self.logger("INFO", "(GoalWorker class): System idle. Attempting AI suggestion generation.")
                            self.suggestion_engine.generate_ai_suggestion() 
                            last_suggestion_gen_time = time.time()
                    elif not self.suggestion_engine:
                         self.logger("DEBUG","(GoalWorker class): System idle, but SuggestionEngine instance not available for generation.")


                # 5. Periodically log pending queue summary
                if time.time() - last_full_eval_log_time > self.periodic_log_interval:
                    pending_goals_for_log = [g for g in current_goals_list_after_processing if isinstance(g, dict) and (g.get("status") == "pending" or g.get("status") == "awaiting_correction")]
                    self.logger("INFO", f"(GoalWorker class|PeriodicLog): Currently {len(pending_goals_for_log)} pending/awaiting goals.")
                    if pending_goals_for_log and self.executor and hasattr(self.executor, 'reprioritize_and_select_next_goal'):
                        try:
                            # Simply sort by original priority for this log display
                            sorted_log_display = sorted(pending_goals_for_log, key=lambda g_log: (g_log.get("priority", 99), g_log.get("created_at", "")))
                            for i_log, g_log in enumerate(sorted_log_display[:3]):
                                 self.logger("INFO", f"  Top BasicPrio Pending {i_log+1}: GID ..{g_log.get('goal_id','N/A')[-6:]}, Prio: {g_log.get('priority')}, Goal: '{g_log.get('goal','')[:40]}...'")
                        except Exception as e_log_sort:
                             self.logger("WARNING", f"(GoalWorker class|PeriodicLog): Error during display sort: {e_log_sort}")
                    last_full_eval_log_time = time.time()

                time.sleep(self.worker_loop_interval)

            except ImportError as e_imp_loop: # From your original script
                self.logger("CRITICAL", f"(GoalWorker class): Import error in main loop: {e_imp_loop}. Worker pausing for 60s.\n{traceback.format_exc()}")
                time.sleep(60)
            except Exception as e_loop: # From your original script
                self.logger("CRITICAL", f"(GoalWorker class): Unhandled error in background_goal_loop: {e_loop}\n{traceback.format_exc()}")
                self.logger("INFO", "(GoalWorker class): Background worker pausing for 30s due to error.")
                time.sleep(30) 

        self.save_active_goal_worker(None) 
        self.logger("INFO","(GoalWorker class): Background goal worker has gracefully stopped.")


    def start_worker(self) -> Optional[threading.Thread]:
        self.stop_worker_flag = False 
        
        if not self.executor: # Critical dependency check
            self.logger("CRITICAL", "(GoalWorker class): Cannot start worker thread, Executor instance is not available.")
            return None

        self.worker_thread = threading.Thread(target=self.background_goal_loop, daemon=True, name="GoalWorkerThread")
        self.worker_thread.start()
        self.logger("INFO", "(GoalWorker class): Worker thread initiated.")
        return self.worker_thread

    def set_pause(self, value: bool):
        action = "Pausing" if value else "Resuming"
        self.logger("INFO", f"(GoalWorker class): {action} background processing.")
        self.pause_background = value

    def stop(self):
        self.logger("INFO", "(GoalWorker class): Stop signal received. Setting flag for worker thread.")
        self.stop_worker_flag = True
        # Joining logic should be handled by the main thread that created the worker (e.g., in console_ai.py's finally block)

# --- End of GoalWorker Class ---

if __name__ == "__main__":
    if should_log("INFO"): print("--- Testing GoalWorker Class (Standalone) ---")

    # Mock dependencies for testing
    mock_config_gw = {
        "meta_dir": "meta_gw_test",
        "goal_worker_config": {
            "suggestion_generation_interval": 10, # Short for test
            "worker_loop_interval": 2, # Short for test
            "periodic_log_interval": 20 # Short for test
        },
        # Paths needed by worker methods or its dependencies' mocks
        "goals_file": os.path.join("meta_gw_test", "goals.json"),
        "active_goal_file": os.path.join("meta_gw_test", "active_goal.json"),
        "suggestions_file": os.path.join("meta_gw_test", "suggestions.json") # For SuggestionEngine mock
    }
    def main_test_logger_gw(level, message):
        if should_log(level.upper()): print(f"[{level.upper()}] (GW_MainTest) {message}")

    class MockExecutorGW:
        def __init__(self, logger): self.logger = logger; self.goals = []
        def init_executor(self): self.logger("INFO", "(MockExecutorGW) init_executor called.")
        def evaluate_goals(self): self.logger("INFO", "(MockExecutorGW) evaluate_goals called.")
        def reprioritize_and_select_next_goal(self, goals_list):
            self.logger("DEBUG", "(MockExecutorGW) reprioritize_and_select_next_goal called.")
            pending = [g for g in goals_list if isinstance(g, dict) and g.get("status") == "pending"]
            return sorted(pending, key=lambda x: x.get("priority", 99))[0] if pending else None
        # Add dummy load_goals and save_goals if worker uses them directly (it uses its own _load_json_worker)
        def load_goals(self): return self.goals # For internal consistency if needed by other mocks
        def save_goals(self, goals_data): self.goals = goals_data

    class MockEvaluatorGW:
        def __init__(self, logger): self.logger = logger
        def _ensure_meta_dir(self): self.logger("INFO", "(MockEvaluatorGW) _ensure_meta_dir called.") # If GoalWorker calls it
        def perform_evaluation_cycle(self): self.logger("INFO", "(MockEvaluatorGW) perform_evaluation_cycle called.")

    class MockSuggestionEngineGW:
        def __init__(self, logger): self.logger = logger
        def init_suggestions_file(self): self.logger("INFO", "(MockSuggestionEngineGW) init_suggestions_file called.")
        def generate_ai_suggestion(self): self.logger("INFO", "(MockSuggestionEngineGW) generate_ai_suggestion called.")

    # Clean up and setup test directory
    if os.path.exists(mock_config_gw["meta_dir"]):
        import shutil
        shutil.rmtree(mock_config_gw["meta_dir"])
    # _ensure_meta_dir in __init__ will create it.

    mock_exec = MockExecutorGW(main_test_logger_gw)
    mock_eval = MockEvaluatorGW(main_test_logger_gw)
    mock_sugg = MockSuggestionEngineGW(main_test_logger_gw)

    gw_instance = GoalWorker(
        config=mock_config_gw,
        logger_func=main_test_logger_gw,
        executor_instance=mock_exec,
        evaluator_instance=mock_eval,
        suggestion_engine_instance=mock_sugg
    )

    # Initialize dummy goals file for the worker's _load_json_worker to read
    # (as its reprioritize_and_select_next_goal relies on this to find top goal for display)
    # The worker's internal _load_json_worker uses self.goals_file_path
    initial_test_goals = [
        {"goal_id": "gw_test_g1", "goal": "Test goal 1 for GW", "status": "pending", "priority": 3, "created_at": datetime.now(timezone.utc).isoformat()},
        {"goal_id": "gw_test_g2", "goal": "Test goal 2 for GW", "status": "pending", "priority": 1, "created_at": datetime.now(timezone.utc).isoformat()}
    ]
    gw_instance._save_json_worker(gw_instance.goals_file_path, initial_test_goals) # Use instance method to save

    if should_log("INFO"): print("\nStarting GoalWorker thread for a short duration (e.g., 10-15 seconds)...")
    worker_thread_main = gw_instance.start_worker()

    if worker_thread_main:
        # Let it run for a bit to see a few cycles
        time.sleep(15) 
        
        if should_log("INFO"): print("\nAttempting to pause worker...")
        gw_instance.set_pause(True)
        time.sleep(7) # See if it respects pause

        if should_log("INFO"): print("\nAttempting to resume worker...")
        gw_instance.set_pause(False)
        time.sleep(7) # See if it resumes

        if should_log("INFO"): print("\nStopping GoalWorker thread...")
        gw_instance.stop()
        worker_thread_main.join(timeout=10) # Wait for thread to finish
        if worker_thread_main.is_alive():
            if should_log("ERROR"): print("ERROR: GoalWorker thread did not terminate cleanly.")
        else:
            if should_log("INFO"): print("GoalWorker thread terminated.")
    else:
        if should_log("ERROR"): print("ERROR: GoalWorker thread failed to start.")

    if should_log("INFO"): print("\n--- GoalWorker Class Test Complete ---")