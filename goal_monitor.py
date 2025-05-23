# Self-Aware/goal_monitor.py
import json
import uuid
import os
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union, Literal, TYPE_CHECKING # Added TYPE_CHECKING

# --- Conditional Imports for Type Checking ---
if TYPE_CHECKING:
    from suggestion_engine import SuggestionEngine, SuggestionStatus, ActorType
    # 'Goal' is defined in this file, so it's directly available for type checking.
    # If 'Goal' were imported, it would also go here if subject to similar issues.

# --- Runtime Imports with Fallbacks ---
try:
    from suggestion_engine import SuggestionEngine, SuggestionStatus, ActorType
except ImportError:
    print("CRITICAL (GoalMonitor): suggestion_engine.py not found or types not defined. Suggestion-to-goal features will be impaired.")
    # Define fallback CLASS for SuggestionEngine
    class SuggestionEngine: # Fallback class definition
        def __init__(self, logger=None):
            self.logger = logger if logger else print
            self.logger("WARNING (GoalMonitor): Using Fallback SuggestionEngine class.")
        def get_suggestion_by_id_or_timestamp(self, identifier: str) -> Optional[Dict[str, Any]]:
            self.logger(f"WARNING (GoalMonitor): Fallback SuggestionEngine.get_suggestion_by_id_or_timestamp called for {identifier}")
            return None
        def mark_suggestion_as_approved_by_actor(self, suggestion_identifier: str, approved_by: Any = "AI") -> bool: # Use Any for ActorType here
            self.logger(f"WARNING (GoalMonitor): Fallback SuggestionEngine.mark_suggestion_as_approved_by_actor called for {suggestion_identifier}")
            return False
        def add_new_suggestion_object(self, suggestion_obj: Dict[str, Any]) -> bool:
             self.logger(f"WARNING (GoalMonitor): Fallback SuggestionEngine.add_new_suggestion_object called.")
             return False
        def load_suggestions(self, load_all_history: bool = False) -> List[Dict[str, Any]]:
            self.logger(f"WARNING (GoalMonitor): Fallback SuggestionEngine.load_suggestions called.")
            return []

    # Define fallback TYPE ALIASES
    SuggestionStatus = str # Fallback type alias
    ActorType = Any        # Fallback type alias (or str, depending on usage)


GOALS_FILE = os.path.join("meta", "goals.json")
ACTIVE_GOAL_FILE = os.path.join("meta", "active_goal.json")

@dataclass
class SubGoal:
    id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    description: str = ""
    status: str = "pending"  
    result: Optional[str] = None

@dataclass
class Goal:
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    parent_id: Optional[str] = None 
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict) 
    status: str = "pending" 
    priority: int = 5  
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    due_date: Optional[str] = None
    sub_goals: List[SubGoal] = field(default_factory=list) 
    sub_goal_ids: List[str] = field(default_factory=list) 
    goal_type: str = "task" 
    created_by: str = "system" 
    related_data: Optional[Dict[str, Any]] = None 
    result: Optional[Dict[str, Any]] = None 
    progress: float = 0.0 
    dependencies: List[str] = field(default_factory=list)
    thread_id: str = field(default_factory=lambda: f"thr_{uuid.uuid4().hex[:8]}")
    failure_category: Optional[str] = None
    self_correction_attempts: int = 0
    tool_file: Optional[str] = None 
    execution_result: Optional[Dict[str, Any]] = None 
    raw_data: Optional[Dict[str, Any]] = field(default=None, repr=False) # Not saved, for loading temp

class GoalMonitor:
    def __init__(self, goals_file=GOALS_FILE, active_goal_file=ACTIVE_GOAL_FILE,
                 suggestion_engine_instance: Optional[SuggestionEngine] = None, logger=None): # No quotes
        self.goals_file = goals_file
        self.active_goal_file = active_goal_file
        self.logger = logger if logger else print
        # Ensure SuggestionEngine is always a class (real or fallback) before instantiation
        current_suggestion_engine_type = SuggestionEngine # This refers to the runtime version (real or fallback)
        self.suggestion_engine: SuggestionEngine = suggestion_engine_instance if suggestion_engine_instance else current_suggestion_engine_type(logger=self.logger) # No quotes

        self._ensure_meta_dir_goal()
        self.goals: List[Goal] = self._load_goals()
        self.active_goal_id: Optional[str] = self._load_active_goal_id()

    def _ensure_meta_dir_goal(self): # Renamed to avoid conflict
        try:
            os.makedirs(os.path.dirname(self.goals_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.active_goal_file), exist_ok=True)
        except OSError as e:
            self.logger(f"ERROR (GoalMonitor): Could not create meta directory for goal files: {e}")


    def _load_goals(self) -> List[Goal]:
        if not os.path.exists(self.goals_file):
            return []
        try:
            with open(self.goals_file, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip(): return [] # Handle empty file
                goals_data = json.loads(content)
                if not isinstance(goals_data, list): # Ensure root is a list
                    self.logger(f"WARNING (GoalMonitor): Goals file content is not a list. Reinitializing {self.goals_file}")
                    self._save_goals_internal([]) # Pass empty list to internal save
                    return []
                
                loaded_goals_list = []
                for g_data in goals_data:
                    if not isinstance(g_data, dict): continue 

                    sub_goals_list_obj = []
                    raw_sub_goals = g_data.get("sub_goals", [])
                    if isinstance(raw_sub_goals, list):
                        for sg_data in raw_sub_goals:
                            if isinstance(sg_data, dict):
                                sub_goals_list_obj.append(SubGoal(**sg_data))
                            elif isinstance(sg_data, SubGoal): 
                                sub_goals_list_obj.append(sg_data)
                    
                    # Prepare args for Goal dataclass, ensuring all fields are present or defaulted
                    goal_args = {key: g_data.get(key) for key in Goal.__annotations__} # Get all annotated fields
                    goal_args.update(g_data) # Overlay with actual data, newer values will take precedence
                    
                    # Set defaults for fields that might be missing from older data
                    goal_args.setdefault("id", f"goal_{uuid.uuid4().hex[:8]}")
                    goal_args.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                    goal_args.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
                    goal_args.setdefault("sub_goals", sub_goals_list_obj)
                    goal_args.setdefault("sub_goal_ids", g_data.get("sub_goal_ids", []))
                    goal_args.setdefault("details", g_data.get("details", {}))
                    goal_args.setdefault("dependencies", g_data.get("dependencies", []))
                    goal_args.setdefault("thread_id", g_data.get("thread_id", f"thr_{uuid.uuid4().hex[:8]}"))
                    goal_args.setdefault("priority", g_data.get("priority", 5))
                    goal_args.setdefault("status", g_data.get("status", "pending"))
                    goal_args.setdefault("progress", g_data.get("progress", 0.0))
                    goal_args.setdefault("self_correction_attempts", g_data.get("self_correction_attempts", 0))
                    goal_args["raw_data"] = g_data # Keep original raw data temporarily

                    try:
                        loaded_goals_list.append(Goal(**{k: v for k, v in goal_args.items() if k in Goal.__annotations__}))
                    except TypeError as te_goal:
                        self.logger(f"ERROR (GoalMonitor): Type error creating Goal object from data: {g_data}. Error: {te_goal}")
                return loaded_goals_list
        except (json.JSONDecodeError, TypeError) as e:
            self.logger(f"Error loading goals from {self.goals_file}: {e}. Initializing with empty list.")
            if os.path.exists(self.goals_file): 
                backup_file = f"{self.goals_file}.bak_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                try:
                    os.rename(self.goals_file, backup_file)
                    self.logger(f"Corrupt goals file backed up to {backup_file}")
                except OSError as ose:
                    self.logger(f"Could not back up corrupt goals file: {ose}")
            return []

    def _save_goals_internal(self, goals_list: List[Goal]): # Internal save method
        try:
            goals_data_to_save = [asdict(g) for g in goals_list] 
            for g_dict in goals_data_to_save:
                if 'raw_data' in g_dict: del g_dict['raw_data']
            
            with open(self.goals_file, "w", encoding="utf-8") as f:
                json.dump(goals_data_to_save, f, indent=4)
        except Exception as e:
            self.logger(f"Error saving goals: {e}")

    def _save_goals(self): # Public save method uses self.goals
        self._save_goals_internal(self.goals)


    def _load_active_goal_id(self) -> Optional[str]:
        if not os.path.exists(self.active_goal_file):
            return None
        try:
            with open(self.active_goal_file, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip(): return None
                data = json.loads(content)
                return data.get("active_goal_id")
        except (json.JSONDecodeError, TypeError, FileNotFoundError) as e: 
            self.logger(f"Error loading active goal ID (or file not found): {e}")
        return None

    def _save_active_goal_id(self):
        try:
            with open(self.active_goal_file, "w", encoding="utf-8") as f:
                json.dump({"active_goal_id": self.active_goal_id}, f, indent=4)
        except Exception as e:
            self.logger(f"Error saving active goal ID: {e}")

    def create_new_goal(self, description: str,
                          details: Optional[Dict[str, Any]] = None,
                          priority: int = 5,
                          parent_id: Optional[str] = None,
                          goal_type: str = "task",
                          created_by: str = "system", 
                          due_date_str: Optional[str] = None, 
                          related_data: Optional[Dict[str, Any]] = None,
                          dependencies: Optional[List[str]] = None,
                          initial_status: str = "pending",
                          thread_id_param: Optional[str] = None 
                          ) -> Optional[Goal]:
        if details is None: details = {}
        
        parsed_due_date = None 
        if due_date_str:
            try: 
                if "days" in due_date_str.lower() and re.search(r'\d+', due_date_str): 
                    num_days = int(re.search(r'\d+', due_date_str).group()) 
                    parsed_due_date = (datetime.now(timezone.utc) + timedelta(days=num_days)).isoformat()
                else: 
                    datetime.fromisoformat(due_date_str.replace("Z", "+00:00")) 
                    parsed_due_date = due_date_str
            except Exception as e_due:
                self.logger(f"Could not parse due_date_str '{due_date_str}': {e_due}")

        effective_thread_id = thread_id_param if thread_id_param else f"thr_{uuid.uuid4().hex[:8]}"
        if parent_id: 
            parent_goal_obj = self.get_goal_by_id(parent_id)
            if parent_goal_obj and not thread_id_param:
                effective_thread_id = parent_goal_obj.thread_id

        goal = Goal(
            description=description, details=details, priority=priority, parent_id=parent_id,
            goal_type=goal_type, created_by=created_by, due_date=parsed_due_date,
            related_data=related_data, dependencies=dependencies if dependencies else [],
            status=initial_status, thread_id=effective_thread_id
        )
        self.goals.append(goal)
        self._save_goals()
        self.logger(f"Created new goal: {goal.id} (Thread: {goal.thread_id[-6:]}) - '{goal.description[:50]}...'")
        return goal

    def get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        return next((goal for goal in self.goals if goal.id == goal_id), None)

    def update_goal(self, goal_id: str, **kwargs) -> Optional[Goal]:
        goal = self.get_goal_by_id(goal_id)
        if not goal:
            self.logger(f"Goal with ID {goal_id} not found for update.")
            return None

        updated_fields = False
        for key, value in kwargs.items():
            if hasattr(goal, key):
                if getattr(goal, key) != value:
                    setattr(goal, key, value)
                    updated_fields = True
            elif key.startswith("details.") and isinstance(goal.details, dict):
                detail_key = key.split(".", 1)[1]
                if goal.details.get(detail_key) != value : goal.details[detail_key] = value; updated_fields = True
            elif key.startswith("related_data."):
                if goal.related_data is None: goal.related_data = {} 
                if isinstance(goal.related_data, dict): # Check if it became a dict
                    r_data_key = key.split(".", 1)[1]
                    if goal.related_data.get(r_data_key) != value : goal.related_data[r_data_key] = value; updated_fields = True
        
        if updated_fields:
            goal.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_goals()
            self.logger(f"Goal {goal_id} updated. New status: {goal.status if 'status' in kwargs else '(status not changed)'}")
        return goal

    def get_active_goals(self, include_paused=False) -> List[Goal]:
        statuses = ["active", "awaiting_correction"]
        if include_paused: statuses.append("paused")
        return [g for g in self.goals if g.status in statuses]

    def get_pending_goals(self) -> List[Goal]:
        return [g for g in self.goals if g.status == "pending" and not self._are_dependencies_pending(g)]

    def _are_dependencies_pending(self, goal: Goal) -> bool:
        if not goal.dependencies: return False
        for dep_id in goal.dependencies:
            dep_goal = self.get_goal_by_id(dep_id)
            if not dep_goal or dep_goal.status not in ["completed"]: 
                return True 
        return False 

    def get_prioritized_pending_goal(self) -> Optional[Goal]:
        pending_goals = self.get_pending_goals()
        if not pending_goals: return None
        return sorted(pending_goals, key=lambda g: (-g.priority, g.created_at))[0]

    def set_active_goal(self, goal_id: Optional[str]) -> Optional[Goal]:
        current_active = self.get_active_goal() 
        if current_active and current_active.id != goal_id:
            if current_active.status == "active": 
                self.update_goal(current_active.id, status="pending")

        if goal_id is None:
            self.active_goal_id = None
            self._save_active_goal_id()
            self.logger("No active goal.")
            return None

        goal = self.get_goal_by_id(goal_id)
        if goal:
            if self._are_dependencies_pending(goal):
                self.logger(f"Cannot activate goal {goal.id} (Thread: {goal.thread_id[-6:]}): dependencies pending.")
                if goal.status == "active": self.update_goal(goal.id, status="pending") 
                return None

            self.active_goal_id = goal.id
            self.update_goal(goal.id, status="active") 
            self._save_active_goal_id()
            self.logger(f"Active goal set: {goal.id} (Thread: {goal.thread_id[-6:]}) - '{goal.description[:50]}...'")
            return goal
        else:
            self.logger(f"Goal ID {goal_id} not found to set active.")
            if self.active_goal_id == goal_id: 
                 self.active_goal_id = None; self._save_active_goal_id()
            return None

    def get_active_goal(self) -> Optional[Goal]:
        if self.active_goal_id:
            goal = self.get_goal_by_id(self.active_goal_id)
            if goal and goal.status == "active": 
                return goal
            elif goal: 
                self.active_goal_id = None 
                self._save_active_goal_id()
        return None

    def complete_goal(self, goal_id: str, result: Optional[Dict[str, Any]] = None) -> Optional[Goal]:
        goal = self.update_goal(goal_id, status="completed", result=result, progress=1.0)
        if goal and self.active_goal_id == goal_id:
            self.set_active_goal(None) 
        return goal

    def fail_goal(self, goal_id: str, failure_reason: Optional[Dict[str, Any]] = None, failure_category_val: Optional[str] = None) -> Optional[Goal]:
        updates: Dict[str, Any] = {"status": "failed", "result": failure_reason if failure_reason else {"error": "Unknown failure"}}
        if failure_category_val: updates["failure_category"] = failure_category_val
        
        goal = self.update_goal(goal_id, **updates)
        if goal and self.active_goal_id == goal_id:
            self.set_active_goal(None) 
        return goal

    # Using 'Any' for 'approved_by' as a workaround for a persistent Pylance error
    # with the 'ActorType' type alias in this specific parameter hint.
    # Runtime validation of 'approved_by' is performed within the method.
    def create_goal_from_suggestion(self, suggestion_identifier: str, approved_by: Any = "user") -> Optional[Goal]:
        if not self.suggestion_engine:
            self.logger("ERROR (GoalMonitor): SuggestionEngine not available. Cannot create goal from suggestion.")
            return None

        suggestion: Optional[Dict[str, Any]] = self.suggestion_engine.get_suggestion_by_id_or_timestamp(suggestion_identifier)
        if not suggestion:
            self.logger(f"Suggestion '{suggestion_identifier}' not found by GoalMonitor.")
            return None

        current_sugg_status: Any = suggestion.get("status", "pending")
        if not (current_sugg_status == "pending" or (isinstance(current_sugg_status, str) and current_sugg_status.startswith("modified"))):
            self.logger(f"Suggestion '{suggestion_identifier}' (status: {current_sugg_status}) not suitable for goal creation.")
            return None

        valid_approved_by_str: str = "user"
        # Runtime check for approved_by against expected values for ActorType
        # Pylance will see "ActorType" as a string and defer actual type resolution,
        # but at runtime, approved_by will still be 'user' or 'AI'.
        # The original TYPE_CHECKING block ensures Pylance knows what "ActorType" should resolve to.
        if approved_by == "AI": # Directly check the value
            valid_approved_by_str = "AI"
        elif approved_by == "user":
            valid_approved_by_str = "user"
        else:
            # This happens if approved_by is passed as something else,
            # or if the default "user" is overridden with an invalid value.
            # The type hint "ActorType" (even as string) guides the developer.
            self.logger(f"Warning (GoalMonitor): Invalid 'approved_by' value '{approved_by}'. Defaulting to 'user'.")
            # valid_approved_by_str remains "user"

        new_goal = self.create_new_goal(
            description=f"Implement suggestion: {suggestion.get('suggestion', 'N/A')}",
            details=suggestion.get("details", {}),
            priority=suggestion.get("priority", 5),
            goal_type="from_suggestion",
            created_by=valid_approved_by_str, # Use the validated string
            related_data={
                "suggestion_id": suggestion.get("id"),
                "suggestion_timestamp": suggestion.get("timestamp"),
                "original_suggestion_text": suggestion.get("suggestion")
            },
            initial_status="pending",
            thread_id_param=suggestion.get("related_thread_id")
        )

        if new_goal:
            # The approved_by parameter for mark_suggestion_as_approved_by_actor
            # also expects ActorType. Pass the validated string.
            self.suggestion_engine.mark_suggestion_as_approved_by_actor(suggestion_identifier, approved_by=valid_approved_by_str) # type: ignore
            self.logger(f"Goal {new_goal.id} created from suggestion '{suggestion_identifier}', approved by {valid_approved_by_str}.")
        return new_goal

    def get_archived_goals(self, last_n: Optional[int] = None) -> List[Goal]:
        # Consider what statuses truly mean "archived" for display.
        # 'rejected_by_user'/'rejected_by_ai' might apply to suggestions, not goals directly unless goals can be rejected.
        archived_statuses = ["completed", "failed", "archived"] # Added "archived" as a generic terminal state
        archived = sorted(
            [g for g in self.goals if g.status in archived_statuses],
            key=lambda g: g.updated_at, reverse=True
        )
        return archived[:last_n] if last_n is not None else archived


if __name__ == '__main__':
    # This __main__ block is for testing goal_monitor.py standalone.
    # It's not part of the AI's core operational logic when AICore runs it.
    print("--- Testing GoalMonitor (Standalone - Corrected Imports & Structure) ---")
    if not os.path.exists("meta"): os.makedirs("meta") # Ensure meta dir for test files
    
    # For testing, SuggestionEngine needs to be instantiated
    # It will try to load/create meta/suggestions.json
    s_engine_for_gm_test = SuggestionEngine(logger=print) 
    g_monitor_for_gm_test = GoalMonitor(suggestion_engine_instance=s_engine_for_gm_test, logger=print)

    print(f"Initial goals for GM test: {len(g_monitor_for_gm_test.goals)}")

    # Test creating a goal from a suggestion (AI approved)
    # Add a suggestion directly via engine for test
    # The add_suggestion in my provided SuggestionEngine returns the suggestion dict
    test_sugg_obj_for_gm_ai = s_engine_for_gm_test.add_new_suggestion_object( # Use the direct add method
        {"suggestion": "GM Test: AI approved suggestion.", "details": {"type": "gm_test_ai_approve"}, "priority": 7, "origin": "goal_monitor_main_ai_test"}
    )
    # We need the ID of the added suggestion
    added_sugg_for_gm_ai = None
    if test_sugg_obj_for_gm_ai: # Check if add_new_suggestion_object was successful and if we can retrieve it
        # Assuming add_new_suggestion_object adds to its internal list and saves, then load_suggestions will get it.
        # Or, if add_new_suggestion_object returns the object with an ID:
        # For this test, let's retrieve it by description if ID is tricky due to uuid.
        all_current_suggs = s_engine_for_gm_test.load_suggestions(load_all_history=True)
        added_sugg_for_gm_ai = next((s for s in all_current_suggs if s.get("suggestion") == "GM Test: AI approved suggestion."), None)

    if added_sugg_for_gm_ai and added_sugg_for_gm_ai.get("id"): 
        print(f"Created test suggestion ID for GM test (AI approve): {added_sugg_for_gm_ai['id']}")
        goal_from_sugg_gm_ai_test = g_monitor_for_gm_test.create_goal_from_suggestion(added_sugg_for_gm_ai["id"], approved_by="AI")
        if goal_from_sugg_gm_ai_test:
            print(f"  Goal from AI approved suggestion: {goal_from_sugg_gm_ai_test.id}, Status: {goal_from_sugg_gm_ai_test.status}")
            updated_s_gm_ai_test = s_engine_for_gm_test.get_suggestion_by_id_or_timestamp(added_sugg_for_gm_ai["id"])
            if updated_s_gm_ai_test: print(f"  Suggestion status after AI approval: {updated_s_gm_ai_test.get('status')}, Approved by: {updated_s_gm_ai_test.get('approved_by')}")
        else:
            print(f"  Failed to create goal from AI approved suggestion {added_sugg_for_gm_ai['id']}.")
    else:
        print("  Failed to create/retrieve initial test suggestion for AI approval in GM test.")
    
    print(f"\nTotal goals after GM tests: {len(g_monitor_for_gm_test.goals)}")
    for g_item_main_gm in g_monitor_for_gm_test.goals[-3:]: 
        print(f" - ID:{g_item_main_gm.id[:10]}.. Th:{g_item_main_gm.thread_id[-6:]} P{g_item_main_gm.priority} St:{g_item_main_gm.status:<12} Desc:'{g_item_main_gm.description[:40]}...'")
    print("\n--- GoalMonitor __main__ Test Complete ---")