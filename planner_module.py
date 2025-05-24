# Self-Aware/planner_module.py
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, List, Any, Union, Callable, TYPE_CHECKING # Added TYPE_CHECKING

# --- Conditional Imports for Type Checking ---
if TYPE_CHECKING:
    from prompt_manager import PromptManager
    from goal_monitor import GoalMonitor
    from mission_manager import MissionManager
    from suggestion_engine import SuggestionEngine
    # Add other types used in hints if they are from external, conditionally available modules

# --- Runtime Imports with Fallbacks ---
try:
    from prompt_manager import PromptManager
except ImportError:
    print("CRITICAL (Planner): prompt_manager.py not found.")
    class PromptManager: pass # Fallback class

try:
    from goal_monitor import GoalMonitor
except ImportError:
    print("CRITICAL (Planner): goal_monitor.py not found.")
    class GoalMonitor: pass # Fallback class

try:
    from mission_manager import MissionManager
except ImportError:
    print("CRITICAL (Planner): mission_manager.py not found.")
    class MissionManager: pass # Fallback class

try:
    from suggestion_engine import SuggestionEngine
except ImportError:
    print("CRITICAL (Planner): suggestion_engine.py not found.")
    class SuggestionEngine: pass # Fallback class


TOOL_REGISTRY_FILE_PLANNER = os.path.join("meta", "tool_registry.json")

class Planner:
    # Errors for lines 38-41 (constructor parameters) should be resolved.
    def __init__(self,
             query_llm_func: Callable[..., str], # Simplified for brevity, ensure it matches adapter
             prompt_manager: Optional[PromptManager],
             goal_monitor: Optional[GoalMonitor],
             mission_manager: Optional[MissionManager],
             suggestion_engine: Optional[SuggestionEngine],
             config: Dict[str, Any],
             logger=None,
             tool_registry_or_path: Optional[Union[Any, str]] = None):

        self.planner_cfg = config.get("planner_config", {}) if config else {}

        # Runtime check for critical dependencies (using isinstance with the runtime types)
        # This ensures that even if TYPE_CHECKING provided a type, at runtime we have a usable class.
        # Note: prompt_manager, goal_monitor, etc. here are the runtime variables from parameters.
        if not all([
            callable(query_llm_func),
            isinstance(prompt_manager, (PromptManager, type(None))), # Allow None if Optional
            isinstance(goal_monitor, (GoalMonitor, type(None))),
            isinstance(mission_manager, (MissionManager, type(None))),
            isinstance(suggestion_engine, (SuggestionEngine, type(None)))
        ]):
            temp_logger = logger if logger else print
            # Be more specific about which runtime component is missing or not a class
            missing_runtime_deps = []
            if not callable(query_llm_func): missing_runtime_deps.append("query_llm_func (not callable)")
            if not isinstance(prompt_manager, (globals().get('PromptManager', object), type(None))): missing_runtime_deps.append("prompt_manager")
            # ... repeat for others, using globals().get('ClassName', object) to get the runtime class name safely
            temp_logger(f"CRITICAL (Planner Init): One or more critical runtime dependencies missing or invalid type: {missing_runtime_deps}. Planner may not function.")


        self.query_llm = query_llm_func
        # Attribute type hints also use direct names
        self.prompt_manager: Optional[PromptManager] = prompt_manager
        self.goal_monitor: Optional[GoalMonitor] = goal_monitor
        self.mission_manager: Optional[MissionManager] = mission_manager
        self.suggestion_engine: Optional[SuggestionEngine] = suggestion_engine
        self.config = config
        self.logger = logger if logger else print

        self.tool_registry_data: List[Dict[str, Any]] = []
        if isinstance(tool_registry_or_path, str):
            self._load_tools_from_json(tool_registry_or_path)
        elif tool_registry_or_path is None:
             self._load_tools_from_json(TOOL_REGISTRY_FILE_PLANNER)
        else:
            self.logger(f"WARNING (Planner): Tool registry data not loaded. Invalid type or path provided: {type(tool_registry_or_path)}")

        self.available_actions = [
            "respond_to_user", "execute_tool", "create_goal", "generate_new_tool",
            "update_goal_status", "reflect_on_performance", "manage_suggestions",
            "request_clarification", "no_action_needed", "initiate_conversation_review"
        ]
        self.evaluation_log_path = self.config.get("evaluation_log_file", os.path.join("meta", "evaluation_log.json"))

    def _load_tools_from_json(self, filepath: str):
        if not os.path.exists(filepath):
            self.logger(f"WARNING (Planner): Tool registry file not found at {filepath}.")
            self.tool_registry_data = []
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list): 
                    self.tool_registry_data = data
                else: 
                    self.logger(f"WARNING (Planner): Tool registry at {filepath} is not a list as expected. Data: {str(data)[:100]}")
                    self.tool_registry_data = [] 
        except (json.JSONDecodeError, IOError) as e:
            self.logger(f"ERROR (Planner): Failed to load/parse tool registry from {filepath}: {e}")
            self.tool_registry_data = []

    def _gather_context(self, conversation_history: List[Dict[str,str]], system_status: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure dependencies are available
        mission_desc = self.mission_manager.get_mission().get("description", "No mission defined.") if self.mission_manager else "Mission manager N/A."
        active_goals_desc = [g.description for g in self.goal_monitor.get_active_goals()] if self.goal_monitor else []
        pending_goals_count = len(self.goal_monitor.get_pending_goals()) if self.goal_monitor else 0

        context: Dict[str, Any] = { 
            "current_time": system_status.get("current_time", datetime.now(timezone.utc).isoformat()),
            "mission": mission_desc,
            "active_goals": active_goals_desc,
            "pending_goals_count": pending_goals_count,
            # Use planner_config from main config if present, else general config for these keys
            "conversation_history": conversation_history[-self.config.get("planner_config", {}).get("planner_context_conversation_history_length", 10):],
            "system_status": system_status,
            "available_actions": self.available_actions,
        }

        if self.tool_registry_data:
            tool_summaries = []
            planner_cfg_tools = self.config.get("planner_config", self.config)
            for tool_dict in self.tool_registry_data[:planner_cfg_tools.get("planner_context_max_tools", 5)]: 
                name = tool_dict.get("name", "Unknown Tool")
                desc = tool_dict.get("description", "No description.")[:80] 
                caps_list = tool_dict.get("capabilities", [])
                caps = ", ".join(caps_list) if isinstance(caps_list, list) and caps_list else 'N/A'
                tool_summaries.append(f"- {name}: {desc} (Caps: {caps})")
            context["tool_capabilities_summary"] = "\n".join(tool_summaries) if tool_summaries else "No tools currently registered or loaded."
        else:
            context["tool_capabilities_summary"] = "Tool registry not available or empty."

        if self.suggestion_engine:
            try:
                all_suggs = self.suggestion_engine.load_suggestions(load_all_history=True) 
                pending_for_context = [s for s in all_suggs if isinstance(s, dict) and s.get("status") == "pending"]
                
                if pending_for_context:
                    planner_cfg_suggs = self.config.get("planner_config", self.config)
                    context["pending_suggestions"] = [
                        {"id": s.get("id"), "description": s.get("suggestion"), "priority": s.get("priority"), "origin": s.get("origin")}
                        for s in pending_for_context
                    ][:planner_cfg_suggs.get("planner_context_max_pending_suggestions", 5)] 
                else:
                    context["pending_suggestions"] = []
            except Exception as e:
                self.logger(f"Error loading pending suggestions for planner context: {e}")
                context["pending_suggestions"] = []
        else:
            context["pending_suggestions"] = [] 
            
        try:
            if os.path.exists(self.evaluation_log_path):
                with open(self.evaluation_log_path, "r", encoding="utf-8") as f:
                    evals_raw = json.load(f)
                    evals = [e for e in evals_raw if isinstance(e, dict)] if isinstance(evals_raw, list) else []
                planner_cfg_evals = self.config.get("planner_config", self.config)
                context["recent_performance_summary"] = evals[-planner_cfg_evals.get("planner_context_recent_evaluations_count", 3):]
            else:
                context["recent_performance_summary"] = []
        except Exception as e:
            self.logger(f"Error loading recent performance for planner context: {e}")
            context["recent_performance_summary"] = []
        return context

    def _construct_prompt(self, context: Dict[str, Any]) -> str:
        """
        Constructs the full prompt for the LLM based on the provided context.
        """
        if not self.prompt_manager:
            self.logger("ERROR", "(Planner._construct_prompt) PromptManager not available.")
            return "Error: PromptManager not available."

        base_prompt_template_name = "planner_decide_action"
        
        # Prepare template_fill_data with defaults
        template_fill_data = {
            "mission_statement": "Mission not set." if not self.mission_manager else self.mission_manager.get_mission_statement_for_prompt(),
            "conversation_history": context.get("conversation_history_str", "No recent conversation."),
            "active_goals": [], # Will be populated with strings
            "pending_suggestions_summary": "No pending suggestions." if not self.suggestion_engine else self.suggestion_engine.get_pending_suggestions_summary_for_prompt(
                max_suggestions=self.planner_cfg.get("planner_context_max_pending_suggestions", 3)
            ),
            "recent_evaluations_summary": context.get("system_status", {}).get("recent_performance_summary", "No recent evaluations."),
            "available_tools_summary": "Tool information not available.", # Default, will be overwritten
            "system_status_summary": f"Current overall status: {context.get('system_status', {}).get('tool_success_rate', 'N/A')}" # Example
        }

        # Populate active goals, ensuring no None values are passed to join
        active_goals_list = []
        if self.goal_monitor:
            active_goal_objects = self.goal_monitor.get_active_goals() # This should return List[Goal]
            for goal_obj in active_goal_objects:
                if hasattr(goal_obj, 'get_short_summary_for_prompt'):
                    summary = goal_obj.get_short_summary_for_prompt()
                    if summary is not None:  # Filter out None summaries
                        active_goals_list.append(summary)
                elif hasattr(goal_obj, 'goal'): # Fallback if Goal object structure is different
                    active_goals_list.append(str(goal_obj.goal)[:100] + "...") 
                else:
                    active_goals_list.append("[Unnamed or invalid goal object]")


        # This will be passed to the template as a list, the template should handle joining or iterating
        template_fill_data["active_goals"] = active_goals_list

        # Prepare available tools summary
        tool_details_list = []
        # Check if self.tool_registry_data exists and is populated
        if hasattr(self, 'tool_registry_data') and self.tool_registry_data:
            try:
                # self.tool_registry_data is already the list of tool metadata dictionaries
                all_tools_metadata = self.tool_registry_data 

                for tool_meta in all_tools_metadata[:self.planner_cfg.get("planner_context_max_tools", 5)]:
                    # tool_meta is expected to be a dict from the JSON
                    if isinstance(tool_meta, dict):
                        tool_name = tool_meta.get('name', 'Unknown Tool')
                        tool_desc_raw = tool_meta.get('description', 'No description.')
                        # Ensure description is a string before slicing
                        tool_desc = str(tool_desc_raw) if tool_desc_raw is not None else 'No description.'
                        tool_details_list.append(f"- {tool_name}: {tool_desc[:100]}{'...' if len(tool_desc) > 100 else ''}")
                    # The 'elif hasattr(tool_meta, 'name')' part is removed 
                    # as items in tool_registry_data are expected to be dictionaries.

                if tool_details_list:
                    template_fill_data["available_tools_summary"] = "\n".join(tool_details_list)
                else:
                    # This message indicates that tool_registry_data was present but might be empty or yielded no details.
                    template_fill_data["available_tools_summary"] = "No tools currently registered or no details processed from tool_registry_data."
            except Exception as e_tool_fetch:
                self.logger("ERROR", f"(Planner._construct_prompt) Error processing tool_registry_data: {e_tool_fetch}")
                template_fill_data["available_tools_summary"] = "Error processing tool information from tool_registry_data."
        else:
            # This 'else' executes if self.tool_registry_data is not found or is empty.
            template_fill_data["available_tools_summary"] = "Tool registry data not available or empty in Planner."

        # Construct the main part of the prompt using the base template
        # This was the first place we corrected get_filled_template
        prompt = self.prompt_manager.render_prompt_with_dynamic_content(
            base_prompt_template_name, 
            template_fill_data
        )

        # Append suggestion management instructions if there are pending suggestions
        # This was the second place we corrected get_filled_template
        if template_fill_data["pending_suggestions_summary"] != "No pending suggestions.":
            suggestion_management_instructions = self.prompt_manager.render_prompt_with_dynamic_content(
                "planner_manage_suggestions_instructions", # Ensure this template key exists
                {"pending_suggestions_summary": template_fill_data["pending_suggestions_summary"],
                 "max_pending_suggestions": self.planner_cfg.get("planner_context_max_pending_suggestions", 3)}
            )
            if suggestion_management_instructions and "Template not found" not in suggestion_management_instructions: # Basic check
                prompt += f"\n\n{suggestion_management_instructions}"
            else:
                self.logger("WARNING", "(Planner._construct_prompt) Could not load 'planner_manage_suggestions_instructions' template or it was empty.")
        
        self.logger("DEBUG", f"(Planner._construct_prompt) Constructed prompt: {prompt[:300]}...") # Log a snippet
        return prompt

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_text = response_text[start_index : end_index+1]
                parsed = json.loads(json_text)
                if "next_action" not in parsed:
                    # Corrected logger call:
                    self.logger("WARNING", f"LLM response for planner missing 'next_action': {json_text}")
                    return None
                return parsed
            else:
                # Corrected logger call:
                self.logger("WARNING", f"Could not find valid JSON object in LLM response: {response_text}") 
                return None
        except json.JSONDecodeError as e:
            # Corrected logger call:
            self.logger("ERROR", f"Error decoding LLM JSON response for planner: {e}\nResponse was: {response_text}")
            return None
        except Exception as e_unexp: 
            # Corrected logger call:
            self.logger("ERROR", f"Unexpected error parsing LLM response for planner: {e_unexp}\nResponse was: {response_text}")
            return None

    def decide_next_action(self, conversation_history: List[Dict[str,str]], system_status: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not self.query_llm or not self.goal_monitor or not self.suggestion_engine: 
            self.logger("CRITICAL", "(Planner): Missing query_llm function, GoalMonitor, or SuggestionEngine. Cannot decide action.")
            return "respond_to_user", {"response_text": "[Error: Planner critical component (LLM access or state manager) missing.]"}

        context = self._gather_context(conversation_history, system_status)
        prompt = self._construct_prompt(context)
        
        planner_cfg = self.config.get("planner_config", self.config) 
        
        # Initialize response_text to prevent UnboundLocalError
        response_text: Optional[str] = None 
        
        # This logger call is for the prompt, not response_text, and should be fine if self.logger expects (level, msg)
        self.logger("DEBUG", f"\n--- Planner Prompt for LLM (first 1000 chars) ---\n{prompt[:1000]}...\n--- End Planner LLM Raw Response ---\n") # Corrected from your file to assume this line meant End Planner *Prompt*

        try:
            # Call the LLM function passed during __init__
            response_text = self.query_llm(
                prompt_text=prompt,
                # system_prompt_override_ignored is not meaningfully used by the adapter to call query_llm_internal
                # raw_output_ignored is redundant as the adapter sets raw_output=True for the planner
                # timeout_override_from_planner=planner_cfg.get("planner_llm_timeout", 300)
            )
            # Conditionally log response_text only if it's not None
            if response_text is not None: # This was your line 309
                # Corrected logger call:
                self.logger("DEBUG", f"\n--- Planner LLM Raw Response ---\n{response_text}\n--- End Planner LLM Raw Response ---\n")
            else:
                self.logger("WARNING", "(Planner): self.query_llm returned None, which was unexpected. Treating as an LLM call failure.")
                # Set to an error string so parsing below can handle it as a known error type
                response_text = "[Error: Planner LLM query unexpectedly returned None]"
        except Exception as e:
            # Corrected logger call:
            self.logger("ERROR", f"LLM call failed for planner via query_llm: {e}") # This was your line 312
            return "respond_to_user", {"response_text": f"[Error: Planner LLM call failed: {e}]"}

        # Ensure response_text is not None before parsing
        if response_text is None:
            self.logger("ERROR", "(Planner): response_text is None before parsing. This indicates a critical failure in the LLM call.")
            return "respond_to_user", {"response_text": "[Error: Planner failed to get any response from LLM.]"}
            
        parsed_response = self._parse_llm_response(response_text)

        if not parsed_response or "next_action" not in parsed_response:
            self.logger("WARNING", "Planner failed to get a valid action from LLM. Defaulting to error response for user.") # Changed from ERROR to WARNING for this specific case
            return "respond_to_user", {"response_text": "[Error: Planner had an internal issue interpreting AI strategy.]"}

        action = parsed_response.get("next_action")
        action_details = parsed_response.get("action_details", {})
        if not isinstance(action_details, dict): action_details = {} 

        if action not in self.available_actions:
            self.logger("WARNING", f"LLM proposed an invalid action: {action}. Defaulting to error response.") # Changed from ERROR
            last_turn = conversation_history[-1] if conversation_history else {}
            if last_turn.get("role") == "user" or last_turn.get("speaker") == "user":
                 return "respond_to_user", {"response_text": "I'm having a little trouble deciding on my next internal step due to an unexpected suggestion from my strategy module. Could you perhaps rephrase or try a different approach?"}
            return "no_action_needed", {"reason": f"LLM proposed invalid action: {action}"}

        # --- Suggestion Management ---
        if action == "manage_suggestions":
            sugg_decision = action_details.get("suggestion_decision")
            if not sugg_decision or not isinstance(sugg_decision, dict):
                self.logger("WARNING", "(Planner): 'manage_suggestions' action chosen, but 'suggestion_decision' details missing/invalid.") # Changed from ERROR
                return "respond_to_user", {"response_text": "I reviewed internal suggestions but had an issue processing the decision."}

            sugg_id_or_ts = sugg_decision.get("suggestion_id") 
            sugg_action_llm = sugg_decision.get("action_to_take") 
            
            if not self.suggestion_engine: # Guard against SuggestionEngine being None
                self.logger("ERROR", "(Planner): SuggestionEngine is not available to manage_suggestions.")
                return "respond_to_user", {"response_text": "Internal error: Suggestion management component not ready."}

            original_suggestion = self.suggestion_engine.get_suggestion_by_id_or_timestamp(sugg_id_or_ts)
            if not original_suggestion:
                display_sugg_id = str(sugg_id_or_ts)[-6:] if sugg_id_or_ts else "unknown"
                self.logger("WARNING", f"(Planner): Suggestion identifier '{sugg_id_or_ts}' from LLM not found.") # Changed from ERROR
                return "respond_to_user", {"response_text": f"I tried to manage suggestion (ID like ..{display_sugg_id}), but couldn't find it."}

            original_desc_for_response = original_suggestion.get('suggestion', 'an internal item')[:70]

            if sugg_action_llm == "approve":
                self.logger("INFO", f"(Planner): AI is approving suggestion '{sugg_id_or_ts}': {original_desc_for_response}...")
                if not self.goal_monitor: # Guard
                    self.logger("ERROR", "(Planner): GoalMonitor not available to approve suggestion.")
                    return "respond_to_user", {"response_text": "Internal error: Goal management component not ready for suggestion approval."}
                new_goal = self.goal_monitor.create_goal_from_suggestion(sugg_id_or_ts, approved_by="AI")
                if new_goal:
                    return "respond_to_user", {"response_text": f"I've reviewed suggestions and decided to act on: '{original_desc_for_response}...'. New goal (ID ..{new_goal.id[-6:]}) created."}
                else:
                    return "respond_to_user", {"response_text": f"I tried to approve suggestion '{original_desc_for_response}...' but failed to create a goal. Status: {original_suggestion.get('status')}."}

            elif sugg_action_llm == "modify":
                new_desc_llm = sugg_decision.get("new_description")
                mod_reason_llm = sugg_decision.get("modification_reason", "AI refined the suggestion.")
                new_priority_llm = sugg_decision.get("new_priority") 
                
                if new_desc_llm:
                    self.logger("INFO", f"(Planner): AI is modifying suggestion '{sugg_id_or_ts}'. New desc: {new_desc_llm[:50]}...")
                    update_payload: Dict[str,Any] = {"suggestion": new_desc_llm}
                    if isinstance(new_priority_llm, int) and 1 <= new_priority_llm <= 10:
                        update_payload["priority"] = new_priority_llm
                    
                    updated_sugg_flag = self.suggestion_engine.update_suggestion_status_details(
                        sugg_id_or_ts, actor="AI", updates_dict=update_payload, reason_for_change=mod_reason_llm
                    )
                    if updated_sugg_flag:
                        return "respond_to_user", {"response_text": f"I've refined an internal suggestion regarding: '{original_desc_for_response}...'. Updated idea: '{new_desc_llm[:70]}...'."}
                    else:
                         return "respond_to_user", {"response_text": f"I tried to modify suggestion ('{original_desc_for_response}...') but the update failed."}
                else:
                    self.logger("WARNING", "(Planner): 'modify' suggestion action by LLM, but 'new_description' missing.") # Changed from ERROR
                    return "respond_to_user", {"response_text": "I considered modifying a suggestion but lacked clear refinement details."}

            elif sugg_action_llm == "reject":
                rej_reason_llm = sugg_decision.get("rejection_reason", "AI determined it's not currently viable or necessary.")
                self.logger("INFO", f"(Planner): AI is rejecting suggestion '{sugg_id_or_ts}': {original_desc_for_response}... Reason: {rej_reason_llm}")
                rejected_flag = self.suggestion_engine.archive_suggestion_as_rejected_by_actor(
                    sugg_id_or_ts, reason=rej_reason_llm, rejected_by="AI" 
                )
                if rejected_flag:
                    return "respond_to_user", {"response_text": f"I've reviewed suggestion '{original_desc_for_response}...' and decided not to pursue it: {rej_reason_llm[:100]}..."}
                else:
                    return "respond_to_user", {"response_text": f"I tried to reject suggestion ('{original_desc_for_response}...') but archival failed."}
            else: 
                self.logger("WARNING", f"(Planner): Unknown suggestion management sub-action '{sugg_action_llm}' for '{sugg_id_or_ts}'.") # Changed from ERROR
                return "respond_to_user", {"response_text": f"I tried to manage a suggestion but received an unclear instruction ('{sugg_action_llm}')."}

        # --- Other Actions ---
        elif action == "create_goal":
            desc = action_details.get("description")
            if desc:
                thread_id_for_new_goal = action_details.get("thread_id", f"thr_{uuid.uuid4().hex[:8]}") 
                if not self.goal_monitor: # Guard
                    self.logger("ERROR", "(Planner): GoalMonitor not available to create_goal.")
                    return "respond_to_user", {"response_text": "Internal error: Goal management component not ready for goal creation."}
                self.goal_monitor.create_new_goal(
                    description=desc, details=action_details.get("details", {}),
                    priority=action_details.get("priority", 5), 
                    goal_type=action_details.get("goal_type", "task"), 
                    created_by="AI", thread_id_param=thread_id_for_new_goal
                )
                self.logger("INFO", f"(Planner): AI Created new goal - '{desc[:50]}...' (Thread: ..{thread_id_for_new_goal[-6:]})")
                user_response_after_goal = action_details.get("response_to_user_after_goal_creation", 
                                                            f"I've created a new internal objective: '{desc[:60]}...'")
                return "respond_to_user", {"response_text": user_response_after_goal}
            else: 
                self.logger("WARNING", "(Planner): 'create_goal' by LLM, but 'description' missing.") # Changed from ERROR
                return "respond_to_user", {"response_text": "I identified a new objective, but details were incomplete."}
        
        action_details.setdefault("internal_action_description", f"Planner decided action: {action}")
        return action, action_details


if __name__ == '__main__':
    # This __main__ block is for testing planner_module.py standalone.
    # It will print the prompt sent to the (mock) LLM.
    # Ensure "meta" and "prompts" directories exist with dummy files for a basic run.
    print("--- Testing Planner Module (Standalone - Corrected, LLM Call via Passed Function) ---")
    
    class MockLLMCallableForPlanner: 
        def __call__(self, prompt_text: str, system_prompt_override: Optional[str]=None, raw_output: bool=False, timeout: int=300) -> str:
            print("\n--- MOCK LLM FUNCTION CALLED (PROMPT RECEIVED BY PLANNER'S LLM FUNCTION) ---")
            print(prompt_text[:1000] + "..." if len(prompt_text) > 1000 else prompt_text)
            print(f"System Override: {str(system_prompt_override)[:100]}..., Raw: {raw_output}, Timeout: {timeout}")
            print("--- END MOCK LLM FUNCTION INPUT ---\n")
            # Simulate LLM deciding to manage a suggestion based on context
            # This mock should be made more dynamic if specific suggestion IDs from prompt are needed for test.
            # For now, it always tries to approve a hardcoded ID or responds generically.
            sugg_id_to_act_on = "sugg_test_planner001" # Assume this ID might be in pending_suggestions context
            if "manage_suggestions" in prompt_text.lower() and f"ID: {sugg_id_to_act_on}" in prompt_text: # Basic check
                mock_decision = {
                    "next_action": "manage_suggestions",
                    "action_details": {
                        "suggestion_decision": { "suggestion_id": sugg_id_to_act_on, "action_to_take": "approve" }
                    }
                }
            else:
                mock_decision = {"next_action": "respond_to_user", 
                                 "action_details": {"response_text": "Mock Planner (via func): LLM call simulated. Review prompt."}}
            print(f"Mock LLM Function Decision: {mock_decision}")
            return json.dumps(mock_decision)

    class MockPromptManagerForPlanner:
        def __init__(self, prompts_dir="prompts"):
            self.prompts_dir = prompts_dir; os.makedirs(self.prompts_dir, exist_ok=True)
            self._ensure_dummy_prompt_pm("planner_decide_action.txt", "Context: {{conversation_history_str}}\nSystem Status: {{system_status}}\nMission: {{mission}}\nTools: {{tool_capabilities_summary}}\nDecide next action: {{available_actions}}")
            self._ensure_dummy_prompt_pm("planner_manage_suggestions_instructions.txt", "Review these suggestions:\n{{pending_suggestions_formatted_list}}\nDecide to approve, modify, or reject one.")
        def _ensure_dummy_prompt_pm(self, filename, content): # Renamed to avoid conflict
            fpath = os.path.join(self.prompts_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w", encoding="utf-8") as f: f.write(content)
        def get_filled_template(self, template_name, context_dict):
            fpath = os.path.join(self.prompts_dir, f"{template_name}.txt")
            if not os.path.exists(fpath): return f"Error: Template {template_name}.txt not found."
            with open(fpath, "r", encoding="utf-8") as f: template_str = f.read()
            for key, value in context_dict.items():
                str_value = str(value) 
                template_str = template_str.replace(f"{{{{{key}}}}}", str_value)
            return template_str
    
    if not os.path.exists("meta"): os.makedirs("meta")
    if not os.path.exists(TOOL_REGISTRY_FILE_PLANNER):
        with open(TOOL_REGISTRY_FILE_PLANNER, "w", encoding="utf-8") as f: json.dump([{"name":"test_tool_pm", "description":"Tool for planner main."}], f)
    
    mock_config_for_pm_test = {
        "planner_config": { # Sub-dictionary for planner specific settings
            "planner_context_conversation_history_length": 2,
            "planner_context_max_pending_suggestions": 1,
            "planner_context_recent_evaluations_count": 1,
            "planner_context_max_tools": 1,
            "planner_max_tokens": 500, "planner_temperature": 0.1, "planner_llm_timeout": 180
        },
        "evaluation_log_file": os.path.join("meta", "evaluation_log.json"),
        # tool_registry_file is handled by TOOL_REGISTRY_FILE_PLANNER default or constructor arg
    }
    if not os.path.exists(mock_config_for_pm_test["evaluation_log_file"]):
        with open(mock_config_for_pm_test["evaluation_log_file"], 'w', encoding="utf-8") as f: json.dump([],f)

    s_engine_for_pm_test = SuggestionEngine(logger=print)
    # Ensure at least one suggestion exists for the mock LLM to "find"
    existing_suggs = s_engine_for_pm_test.load_suggestions()
    target_sugg_found = any(s.get("id") == "sugg_test_planner001" for s in existing_suggs if isinstance(s,dict))
    if not target_sugg_found:
         s_engine_for_pm_test.add_new_suggestion_object({
             "id": "sugg_test_planner001", 
             "suggestion": "A specific suggestion for the planner __main__ test to manage.", 
             "origin": "planner_main_test_setup", "status": "pending"})

    g_monitor_for_pm_test = GoalMonitor(suggestion_engine_instance=s_engine_for_pm_test, logger=print)
    m_manager_for_pm_test = MissionManager(logger=print) 

    planner_for_main_test = Planner(
        query_llm_func=MockLLMCallableForPlanner(), 
        prompt_manager=MockPromptManagerForPlanner(), 
        goal_monitor=g_monitor_for_pm_test, 
        mission_manager=m_manager_for_pm_test,
        suggestion_engine=s_engine_for_pm_test,
        config=mock_config_for_pm_test, # Pass the main config here
        logger=print,
        tool_registry_or_path=TOOL_REGISTRY_FILE_PLANNER 
    )
    
    test_convo_pm = [{"role": "user", "content": "What about my suggestions?"}]
    test_status_pm = {"current_time": datetime.now(timezone.utc).isoformat(), "tool_success_rate": 0.95}
    
    print("\n--- Running Planner decide_next_action with mock LLM function (for prompt review) ---")
    action_res_pm, details_res_pm = planner_for_main_test.decide_next_action(test_convo_pm, test_status_pm)
    print(f"\nPlanner __main__ Test - Decided Action: {action_res_pm}")
    print(f"Planner __main__ Test - Action Details: {details_res_pm}")
    print("\n--- Planner Module __main__ Test Complete ---")