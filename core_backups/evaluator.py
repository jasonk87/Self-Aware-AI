# Self-Aware/evaluator.py
import os
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List, Union # Added List, Union

# Fallback logger for standalone testing or if no logger is provided to the class
_CLASS_EVALUATOR_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (EvaluatorClassFallbackLog) {message}")
_CLASS_EVALUATOR_FALLBACK_QUERY_LLM = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=300: \
    f"[Error: Fallback LLM query from Evaluator instance. Prompt: {prompt_text[:100]}...]"

# --- Module-level imports for constants and potentially functions ---
# These are kept as module-level imports as per your original structure.
# If executor and mission_manager are refactored into classes and these
# become instance methods/attributes, this section would change, or instances
# would need to be passed to the Evaluator class for access.

_EVALUATOR_LOGGER_FUNC = _CLASS_EVALUATOR_FALLBACK_LOGGER # Default to class fallback, instance will override
if 'log_background_message' not in globals(): # If ai_core.log_background_message was not imported by a previous module
    try:
        from ai_core import log_background_message as evaluator_log_bg_msg_temp
        _EVALUATOR_LOGGER_FUNC = evaluator_log_bg_msg_temp
    except ImportError:
        def evaluator_log_bg_msg_temp(level: str, message: str): # type: ignore
            print(f"[{level.upper()}] (evaluator_module_fallback_log) {message}")
        _EVALUATOR_LOGGER_FUNC = evaluator_log_bg_msg_temp

try:
    from executor import (
        GOAL_FILE as EXECUTOR_GOAL_FILE_CONST,
        TOOL_REGISTRY_FILE as EXECUTOR_TOOL_REGISTRY_FILE_CONST,
        STATUS_COMPLETED, STATUS_EXECUTED_WITH_ERRORS,
        STATUS_BUILD_FAILED, STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR,
        PRIORITY_URGENT, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW,
        SOURCE_SELF_CORRECTION, CAT_CORE_FILE_UPDATE
    )
except ImportError:
    _EVALUATOR_LOGGER_FUNC("CRITICAL", "(evaluator module): Could not import constants from executor.py. Evaluation logic will be impaired.")
    EXECUTOR_GOAL_FILE_CONST = os.path.join("meta", "goals.json")
    EXECUTOR_TOOL_REGISTRY_FILE_CONST = os.path.join("meta", "tool_registry.json") # Fallback, though not used directly by evaluator
    STATUS_COMPLETED, STATUS_EXECUTED_WITH_ERRORS = "completed", "executed_with_errors"
    STATUS_BUILD_FAILED, STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR = "build_failed", "failed_max_retries", "failed_correction_unclear"
    PRIORITY_URGENT, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW = 1, 2, 3, 4
    SOURCE_SELF_CORRECTION, CAT_CORE_FILE_UPDATE = "AI_self_correction", "core_file_update"

_EVALUATOR_MISSION_CONTEXT_FUNC = lambda: "Mission context unavailable for evaluation (fallback)."
if 'get_mission_for_eval_context' not in globals(): # If mission_manager.get_mission_statement_for_prompt was not imported
    try:
        from mission_manager import get_mission_statement_for_prompt as get_mission_for_eval_context_temp
        _EVALUATOR_MISSION_CONTEXT_FUNC = get_mission_for_eval_context_temp
    except ImportError:
        pass # Fallback lambda is already set

class Evaluator:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None,
                 mission_manager_instance: Optional[Any] = None):

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_EVALUATOR_FALLBACK_LOGGER
        self.query_llm = query_llm_func if query_llm_func else _CLASS_EVALUATOR_FALLBACK_QUERY_LLM
        self.mission_manager = mission_manager_instance

        self.meta_dir = self.config.get("meta_dir", "meta")
        self.evaluation_log_file_path = self.config.get("evaluation_log_file", os.path.join(self.meta_dir, "evaluation_log.json"))
        
        # Path to the goals file that the executor uses (and this evaluator reads for evaluation)
        self.executor_goal_file_path = self.config.get("goals_file", EXECUTOR_GOAL_FILE_CONST) # Use constant if not in config

        self._ensure_meta_dir() # Call during initialization

    def _ensure_meta_dir(self):
        try:
            os.makedirs(self.meta_dir, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(Evaluator class): Could not create meta directory {self.meta_dir}: {e}")

    def _load_json_eval(self, filepath: str, default_value: Optional[Union[List, Dict]] = None) -> Any:
        actual_default = [] if default_value is None else default_value
        self._ensure_meta_dir() # Ensure dir exists before file operations
        if not os.path.exists(filepath):
            self.logger("DEBUG", f"(Evaluator class): File {filepath} not found, returning default for load.")
            return actual_default
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    self.logger("DEBUG", f"(Evaluator class): File {filepath} is empty, returning default for load.")
                    return actual_default
                return json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            self.logger("WARNING", f"(Evaluator class): Error loading/decoding {filepath}: {e}. Returning default.")
            return actual_default
        except Exception as e_unexp:
            self.logger("CRITICAL", f"(Evaluator class): Unexpected error loading {filepath}: {e_unexp}\n{traceback.format_exc()}. Returning default.")
            return actual_default

    def _save_json_eval(self, filepath: str, data: Union[Dict[str, Any], List[Any]]):
        self._ensure_meta_dir()
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            self.logger("ERROR", f"(Evaluator class): Could not write to {filepath}: {e}")
        except Exception as e_unexp:
            self.logger("CRITICAL", f"(Evaluator class): Unexpected error saving {filepath}: {e_unexp}\n{traceback.format_exc()}.")

    def _get_mission_context_for_llm_eval(self) -> str:
        if self.mission_manager and hasattr(self.mission_manager, 'get_mission_statement_for_prompt'):
            try:
                return self.mission_manager.get_mission_statement_for_prompt()
            except Exception as e:
                self.logger("WARNING", f"(Evaluator class): Error getting mission from MissionManager instance: {e}")
        # Fallback to module-level imported function if instance method fails or no instance
        if callable(_EVALUATOR_MISSION_CONTEXT_FUNC):
            try:
                return _EVALUATOR_MISSION_CONTEXT_FUNC()
            except Exception as e_global:
                self.logger("WARNING", f"(Evaluator class): Error calling global mission context function: {e_global}")
        return "Mission context unavailable for evaluation (complete fallback)."


    def evaluate_goal_analytically(self, goal_obj: Dict[str, Any]) -> tuple[int, str]:
        score = 0
        notes_list = []

        status = goal_obj.get("status")
        source = goal_obj.get("source")
        priority = goal_obj.get("priority", PRIORITY_NORMAL)
        attempts = goal_obj.get("self_correction_attempts", 0)
        failure_cat = goal_obj.get("failure_category")
        subtask_cat = goal_obj.get("subtask_category")

        notes_list.append(f"Status: {status}, Prio: {priority}, Src: {source}, Attempts: {attempts}.")
        if failure_cat: notes_list.append(f"FailCat: {failure_cat}.")
        if subtask_cat: notes_list.append(f"SubtaskCat: {subtask_cat}.")
        if goal_obj.get("tool_file"): notes_list.append(f"Tool: {os.path.basename(goal_obj.get('tool_file', 'N/A'))}.")

        if status == STATUS_COMPLETED:
            score += 15
            notes_list.append("Base score for completion: +15.")
            if subtask_cat == CAT_CORE_FILE_UPDATE: score += 15; notes_list.append("Core file update bonus: +15.")
            if source == SOURCE_SELF_CORRECTION: score += 8; notes_list.append("Successful self-correction bonus: +8.")
            if goal_obj.get("used_existing_tool"): score += 3; notes_list.append("Used existing tool bonus: +3.")
            if attempts == 0 : score += 5; notes_list.append("First attempt success: +5.")
            priority_bonus = (PRIORITY_LOW - priority + 1) * 2
            score += priority_bonus
            notes_list.append(f"Priority completion bonus (P{priority}): +{priority_bonus}.")
        elif status == STATUS_EXECUTED_WITH_ERRORS:
            score -= 8
            notes_list.append("Base penalty for execution errors: -8.")
            if attempts > 0:
                attempt_penalty = attempts * 3
                score -= attempt_penalty
                notes_list.append(f"Self-correction attempts penalty: -{attempt_penalty}.")
        elif status == STATUS_BUILD_FAILED:
            score -= 12
            notes_list.append("Base penalty for build failure: -12.")
        elif status in [STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR]:
            score -= 20
            notes_list.append("Base penalty for max retries/unclear failure: -20.")
            if failure_cat == "DecompositionFailure": score -= 10; notes_list.append("Decomposition failure extra penalty: -10.")
            priority_penalty = priority * 3
            score -= priority_penalty
            notes_list.append(f"Priority failure penalty (P{priority}): -{priority_penalty}.")
        
        if status not in [STATUS_COMPLETED] and status not in [STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR]:
            priority_penalty_general = int(priority * 1.5)
            score -= priority_penalty_general
            notes_list.append(f"General priority failure adjustment (P{priority}): -{priority_penalty_general}.")

        final_notes = " ".join(notes_list)
        return score, final_notes[:1000]

    def evaluate_goal_with_llm(self, goal_obj: Dict[str, Any], analytical_score: int, analytical_notes: str) -> tuple[Optional[int], Optional[str]]:
        should_llm_eval = False
        if goal_obj.get("subtask_category") == CAT_CORE_FILE_UPDATE: should_llm_eval = True
        elif goal_obj.get("status") in [STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR] and goal_obj.get("priority", PRIORITY_NORMAL) <= PRIORITY_HIGH: should_llm_eval = True
        
        if not should_llm_eval: return None, None

        self.logger("INFO", f"(Evaluator class): Performing LLM-based evaluation for goal GID: ..{goal_obj.get('goal_id', 'N/A')[-6:]}")
        mission_context = self._get_mission_context_for_llm_eval()

        llm_eval_prompt = (
            f"{mission_context}\n\n"
            f"### Goal Evaluation Task ###\n"
            f"Please evaluate the outcome of the following AI development goal. Consider its impact, alignment with the mission, and the effectiveness of the AI's actions.\n\n"
            f"**Goal Details:**\n"
            f"- Description: \"{goal_obj.get('goal', goal_obj.get('description','N/A'))}\"\n" # Check 'description' key too
            f"- Status: {goal_obj.get('status', 'N/A')}\n"
            f"- Priority: {goal_obj.get('priority', 'N/A')}\n"
            f"- Source: {goal_obj.get('source', 'N/A')}\n"
            f"- Subtask Category: {goal_obj.get('subtask_category', 'N/A')}\n"
            f"- Self-Correction Attempts: {goal_obj.get('self_correction_attempts', 0)}\n"
            f"- Analytical Score (rule-based): {analytical_score}\n"
            f"- Analytical Notes: {analytical_notes}\n"
        )
        if goal_obj.get("error"): llm_eval_prompt += f"- Error Encountered: {str(goal_obj.get('error'))[:500]}\n"
        if goal_obj.get("failure_category"): llm_eval_prompt += f"- Failure Category: {goal_obj.get('failure_category')}\n"
        if goal_obj.get("execution_result"):
            exec_res_str = str(goal_obj.get('execution_result'))
            llm_eval_prompt += f"- Execution Result Summary: {exec_res_str[:500] if len(exec_res_str) > 500 else exec_res_str}\n"
        llm_eval_prompt += (
            f"\n**Evaluation Request:**\n"
            f"1. Provide a brief qualitative summary (2-3 sentences) of this goal's outcome and its significance.\n"
            f"2. Suggest an `adjusted_score` (integer between -50 and +50) that reflects your qualitative assessment, potentially modifying the analytical_score. Explain your reasoning if you adjust it significantly.\n"
            f"3. Identify any key learnings or recommendations for future operations.\n\n"
            f"Respond ONLY with a JSON object with keys: \"qualitative_summary\", \"adjusted_score_suggestion\", \"score_reasoning\", \"key_learnings\". Example:\n"
            f"{{\"qualitative_summary\": \"The AI successfully updated a core component...\", \"adjusted_score_suggestion\": 40, \"score_reasoning\": \"High impact...\", \"key_learnings\": \"Validation process is effective.\"}}"
        )
        llm_system_prompt = "You are an AI Performance Reviewer. Analyze the provided goal outcome data and provide a structured JSON evaluation."
        
        response_str = self.query_llm(llm_eval_prompt, system_prompt_override=llm_system_prompt, timeout=self.config.get("evaluator_llm_timeout", 300), raw_output=False) # Use configured timeout

        if response_str and not response_str.startswith("[Error:"):
            try:
                eval_data = json.loads(response_str)
                summary = eval_data.get("qualitative_summary", "LLM provided no qualitative summary.")
                score_suggestion = eval_data.get("adjusted_score_suggestion")
                reasoning = eval_data.get("score_reasoning", "")
                learnings = eval_data.get("key_learnings", "LLM provided no key learnings.")
                llm_notes = f"LLM Summary: {summary} Score Reasoning: {reasoning} Learnings: {learnings}"
                final_score_from_llm = None
                if isinstance(score_suggestion, int): final_score_from_llm = max(min(score_suggestion, 50), -50)
                elif isinstance(score_suggestion, str) and score_suggestion.lstrip('-').isdigit():
                    try: final_score_from_llm = max(min(int(score_suggestion), 50), -50)
                    except ValueError: pass
                return final_score_from_llm, llm_notes[:1000]
            except json.JSONDecodeError as e:
                self.logger("WARNING", f"(Evaluator class): Failed to decode LLM JSON for GID ..{goal_obj.get('goal_id', 'N/A')[-6:]}. Resp: {response_str[:200]}. Err: {e}")
                return None, f"LLM response parsing error: {e}"
            except Exception as e_llm_parse:
                self.logger("ERROR", f"(Evaluator class): Unexpected error parsing LLM eval for GID ..{goal_obj.get('goal_id', 'N/A')[-6:]}: {e_llm_parse}\n{traceback.format_exc()}")
                return None, f"Unexpected LLM parsing error: {e_llm_parse}"
        else:
            self.logger("WARNING", f"(Evaluator class): LLM returned an error or no response for GID ..{goal_obj.get('goal_id', 'N/A')[-6:]}. LLM response: {response_str}")
            return None, f"LLM evaluation failed: {response_str}"

    def log_evaluation_entry(self, goal_id: str, analytical_score: int, analytical_notes: str,
                             llm_score: Optional[int], llm_notes: Optional[str], final_score: int):
        log_entry = {
            "goal_id": goal_id, "analytical_score": analytical_score, "analytical_notes": analytical_notes,
            "llm_adjusted_score": llm_score, "llm_qualitative_notes": llm_notes,
            "final_score_assigned": final_score, "timestamp": datetime.now(timezone.utc).isoformat()
        }
        eval_log = self._load_json_eval(self.evaluation_log_file_path, [])
        if isinstance(eval_log, list): # Ensure it's a list before append
            eval_log.append(log_entry)
        else: # Should not happen if _load_json_eval returns list for [] default
            self.logger("ERROR", f"(Evaluator class) Evaluation log at {self.evaluation_log_file_path} is not a list. Cannot append new entry.")
            eval_log = [log_entry] # Start a new list
        self._save_json_eval(self.evaluation_log_file_path, eval_log)
        self.logger("INFO", f"(Evaluator class): GID '..{goal_id[-6:]}' evaluated. Final Score: {final_score}. Notes: {analytical_notes[:50]}...")

    def perform_evaluation_cycle(self):
        self.logger("INFO", "(Evaluator class): Starting evaluation cycle...")
        goals = self._load_json_eval(self.executor_goal_file_path, []) # Use instance path
        eval_log = self._load_json_eval(self.evaluation_log_file_path, [])
        
        if not isinstance(goals, list): # Ensure goals is a list
            self.logger("ERROR", f"(Evaluator class) Goals data from {self.executor_goal_file_path} is not a list. Skipping evaluation cycle.")
            return
        if not isinstance(eval_log, list): # Ensure eval_log is a list
            self.logger("ERROR", f"(Evaluator class) Eval log data from {self.evaluation_log_file_path} is not a list. Reinitializing.")
            eval_log = []

        evaluated_goal_ids_in_log = {entry.get("goal_id") for entry in eval_log if isinstance(entry, dict)}
        updated_goals_for_saving = False

        for goal in goals:
            if not isinstance(goal, dict): continue # Skip non-dict goals
            goal_id = goal.get("goal_id")
            status = goal.get("status")
            needs_evaluation = False

            if status in [STATUS_COMPLETED, STATUS_EXECUTED_WITH_ERRORS, STATUS_BUILD_FAILED, STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR]:
                if goal.get("evaluation", {}).get("last_evaluated_at"):
                    try:
                        last_eval_dt = datetime.fromisoformat(goal["evaluation"]["last_evaluated_at"].replace("Z", "+00:00"))
                        last_goal_mod_ts_str = goal.get("history", [{}])[-1].get("timestamp")
                        if last_goal_mod_ts_str:
                            last_goal_mod_dt = datetime.fromisoformat(last_goal_mod_ts_str.replace("Z", "+00:00"))
                            if last_goal_mod_dt > last_eval_dt: needs_evaluation = True
                        else: needs_evaluation = False # No modification timestamp, assume no change since last eval
                    except Exception as e_ts_compare_eval:
                        self.logger("WARNING", f"(Evaluator class) Timestamp comparison error for GID ..{goal_id[-6:] if goal_id else 'N/A'} during re-evaluation check: {e_ts_compare_eval}. Defaulting to not re-evaluate if already evaluated.")
                        needs_evaluation = False
                else: # No 'last_evaluated_at' field in goal's evaluation dict
                    needs_evaluation = True
            
            if needs_evaluation:
                self.logger("DEBUG", f"(Evaluator class): Evaluating GID ..{goal_id[-6:] if goal_id else 'N/A'} with status {status}.")
                analytical_score, analytical_notes = self.evaluate_goal_analytically(goal)
                llm_adj_score, llm_qual_notes = self.evaluate_goal_with_llm(goal, analytical_score, analytical_notes)
                final_score = analytical_score
                if llm_adj_score is not None:
                    final_score = llm_adj_score
                    self.logger("INFO", f"(Evaluator class): LLM adjusted score for GID ..{goal_id[-6:] if goal_id else 'N/A'} from {analytical_score} to {final_score}.")
                
                goal.setdefault("evaluation", {})
                goal["evaluation"]["final_score"] = final_score
                goal["evaluation"]["analytical_score"] = analytical_score
                goal["evaluation"]["analytical_notes"] = analytical_notes
                if llm_adj_score is not None: goal["evaluation"]["llm_adjusted_score"] = llm_adj_score
                if llm_qual_notes: goal["evaluation"]["llm_qualitative_notes"] = llm_qual_notes
                goal["evaluation"]["last_evaluated_at"] = datetime.now(timezone.utc).isoformat()
                
                self.log_evaluation_entry(goal_id, analytical_score, analytical_notes, llm_adj_score, llm_qual_notes, final_score)
                updated_goals_for_saving = True
            
        if updated_goals_for_saving:
            self._save_json_eval(self.executor_goal_file_path, goals)
        self.logger("INFO", "(Evaluator class): Evaluation cycle finished.")


if __name__ == "__main__":
    print("--- Testing Evaluator Class (Standalone) ---")
    
    # Mock config for testing
    test_eval_config = {
        "meta_dir": "meta_eval_test",
        "evaluation_log_file": os.path.join("meta_eval_test", "evaluation_log_test.json"),
        "goals_file": os.path.join("meta_eval_test", "goals_executor_test.json"), # Path to where executor goals are
        "evaluator_llm_timeout": 10 # Short for test
    }

    def main_test_logger_eval(level, message): print(f"[{level.upper()}] (Eval_Test) {message}")
    
    def main_test_query_llm_eval(prompt_text, system_prompt_override=None, raw_output=False, timeout=180):
        main_test_logger_eval("INFO", f"MainTest Mock LLM (Evaluator) called. Prompt starts: {prompt_text[:100]}...")
        if "### Goal Evaluation Task ###" in prompt_text:
            # Simulate LLM evaluation response
            return json.dumps({
                "qualitative_summary": "Mock LLM: Goal seems reasonably handled despite minor issues.",
                "adjusted_score_suggestion": (10 if "completed" in prompt_text.lower() else -5), # Example dynamic score
                "score_reasoning": "Mock LLM adjusted based on simulated complexity and outcome.",
                "key_learnings": "Mock LLM suggests reviewing error patterns for future improvements."
            })
        return "[MainTest Mock LLM Eval: Default Response]"

    class MainTestMockMissionManagerEval:
        def get_mission_statement_for_prompt(self): return "### EVAL MOCK MISSION ###\nTest mission for Evaluator."

    # Clean up and setup test directory
    if os.path.exists(test_eval_config["meta_dir"]):
        import shutil
        shutil.rmtree(test_eval_config["meta_dir"])
    os.makedirs(test_eval_config["meta_dir"], exist_ok=True)

    # Create dummy goal file for testing (as if written by executor)
    dummy_goals_for_eval_main = [
        {"goal_id": "eval_g1", "goal": "Test goal 1 (completed)", "status": "completed", "priority": 2, "history": [{"timestamp":datetime.now(timezone.utc).isoformat()}]},
        {"goal_id": "eval_g2", "goal": "Test goal 2 (failed)", "status": "failed_max_retries", "priority": 1, "history": [{"timestamp":datetime.now(timezone.utc).isoformat()}], "error": "It broke.", "failure_category": "TestFailure"},
        {"goal_id": "eval_g3", "goal": "Test goal 3 (pending)", "status": "pending", "priority": 3, "history": [{"timestamp":datetime.now(timezone.utc).isoformat()}]},
        {"goal_id": "eval_g4", "goal": "Core update test", "status": "completed", "subtask_category": "core_file_update", "priority": 1, "history": [{"timestamp":datetime.now(timezone.utc).isoformat()}]}
    ]
    # Use the path from config for saving the dummy goals file
    eval_test_goals_file_path = test_eval_config["goals_file"]
    with open(eval_test_goals_file_path, "w", encoding='utf-8') as f_goals_eval_test:
        json.dump(dummy_goals_for_eval_main, f_goals_eval_test, indent=2)
    main_test_logger_eval("INFO", f"Created dummy goals file at {eval_test_goals_file_path}")

    eval_instance_main = Evaluator(
        config=test_eval_config,
        logger_func=main_test_logger_eval,
        query_llm_func=main_test_query_llm_eval,
        mission_manager_instance=MainTestMockMissionManagerEval()
    )

    print("\nRunning perform_evaluation_cycle()...")
    eval_instance_main.perform_evaluation_cycle()

    print(f"\n--- Contents of {eval_instance_main.evaluation_log_file_path} after cycle ---")
    eval_log_content_main = eval_instance_main._load_json_eval(eval_instance_main.evaluation_log_file_path, [])
    if isinstance(eval_log_content_main, list) and eval_log_content_main: # Check if list and not empty
        for entry_main in eval_log_content_main: print(json.dumps(entry_main, indent=2))
    else: print("Evaluation log is empty or not a list.")
        
    print(f"\n--- Contents of {eval_instance_main.executor_goal_file_path} after cycle (showing evaluation scores) ---")
    goals_after_eval_main = eval_instance_main._load_json_eval(eval_instance_main.executor_goal_file_path, [])
    if isinstance(goals_after_eval_main, list) and goals_after_eval_main: # Check if list and not empty
        for goal_entry_main in goals_after_eval_main:
            goal_id_short_main = goal_entry_main.get('goal_id', 'N/A')[-6:]
            if "evaluation" in goal_entry_main and isinstance(goal_entry_main["evaluation"], dict):
                print(f"  Goal GID: ..{goal_id_short_main}, Final Score: {goal_entry_main['evaluation'].get('final_score')}")
            elif goal_entry_main.get('status') not in ["pending", "active", "decomposed", "approved", "awaiting_correction"]: # Only print if it SHOULD have been evaluated
                print(f"  Goal GID: ..{goal_id_short_main}, Status: {goal_entry_main.get('status')} (No 'evaluation' field written).")
    else: print(f"{eval_instance_main.executor_goal_file_path} is empty or not a list after evaluation.")

    print("\n--- Evaluator Class Test Complete ---")