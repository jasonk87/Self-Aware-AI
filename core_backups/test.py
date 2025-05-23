# Self-Aware/console_ai.py
import json
import os
import queue # For fallback queue definitions if ai_core fails
import sys
import time
import traceback
from datetime import datetime, timezone
import uuid
import re
import msvcrt # For Windows non-blocking input (standard library)
import threading # For type hinting worker_thread_instance
from speech_recognition import Recognizer

from typing import Optional

# --- Section 1: Initial Setup, AICore Initialization, and GoalWorker Acquisition ---

# --- Global Variables for Core Components & Logger ---
# These will hold functional components if AICore initializes successfully.
ai_core_singleton_instance = None
goal_worker_instance_ref: threading.Thread | None = None # Will hold the GoalWorker instance from AICore
log_to_console_logger: callable = lambda lvl, msg: print(f"[{lvl.upper()}_CONSOLE_PRE_INIT_LOG] {msg}", file=sys.stderr if lvl.upper() in ["ERROR", "CRITICAL"] else sys.stdout) # Basic pre-init logger
response_queue_console = queue.Queue() # Fallback queue
log_message_queue_console = queue.Queue() # Fallback queue, primarily for ai_core to put messages if console were to read them.
AI_CORE_MODEL_NAME_CONSOLE = "N/A (ai_core_not_loaded)"
get_response_async_console_func = None # Fallback for the main async function

# --- Attempt to Import and Initialize AI Core ---
try:
    from ai_core import (
        initialize_ai_core_singleton,
        _ai_core_instance_singleton, # Direct access to the singleton instance variable
        log_background_message as ai_core_logger_func,
        response_queue as ai_core_response_queue,
        log_message_queue as ai_core_log_message_queue,
        MODEL_NAME as AI_CORE_MODEL_NAME_FROM_MODULE,
        get_response_async as ai_core_get_response_async
    )

    # Initialize AI Core. Pass a basic console logger for AICore's own early init messages.
    # If AICore initializes its logger successfully, ai_core_logger_func will be more sophisticated.
    initialize_ai_core_singleton(logger_override=log_to_console_logger) # Use current basic logger for AICore's init

    if _ai_core_instance_singleton:
        ai_core_singleton_instance = _ai_core_instance_singleton
        log_to_console_logger = ai_core_logger_func # Switch to using AICore's logger
        response_queue_console = ai_core_response_queue
        log_message_queue_console = ai_core_log_message_queue # Console doesn't typically read this, but good to have the alias
        AI_CORE_MODEL_NAME_CONSOLE = AI_CORE_MODEL_NAME_FROM_MODULE
        get_response_async_console_func = ai_core_get_response_async # Get the functional async call

        log_to_console_logger("INFO", "(console_ai) AICore singleton initialized. Switched to AICore logger.")

        # Obtain the GoalWorker instance from the initialized AICore singleton
        if hasattr(ai_core_singleton_instance, 'goal_worker') and ai_core_singleton_instance.goal_worker is not None:
            goal_worker_instance_ref = ai_core_singleton_instance.goal_worker
            log_to_console_logger("INFO", "(console_ai) Successfully obtained GoalWorker instance from AICore.")
        else:
            log_to_console_logger("CRITICAL", "(console_ai) AICore initialized, but its GoalWorker instance is missing or None. Background processing will be disabled.")
            goal_worker_instance_ref = None # Ensure it's None
    else:
        # This means initialize_ai_core_singleton() itself failed to create the instance.
        # log_to_console_logger is still the basic print-based one here.
        log_to_console_logger("CRITICAL", "(console_ai) Failed to initialize AICore singleton instance. AI Core functionalities will be unavailable.")
        goal_worker_instance_ref = None

except ImportError as e_ai_core_import:
    log_to_console_logger("CRITICAL", f"(console_ai) FATAL: Failed to import from ai_core.py: {e_ai_core_import}. Core AI functionalities disabled. Console will be severely limited.")
    goal_worker_instance_ref = None
    # All ai_core dependent vars remain their basic fallbacks.
except Exception as e_aicore_general_init:
    log_to_console_logger("CRITICAL", f"(console_ai) FATAL: An unexpected error occurred during AICore initialization: {e_aicore_general_init}\n{traceback.format_exc()}")
    goal_worker_instance_ref = None

# --- Console-Specific Module Imports & Fallbacks ---
# These are modules console_ai.py uses for its own UI/commands,
# potentially distinct from AICore's instances.
try:
    from notifier_module import init_versioning, get_current_version, LAST_UPDATE_FLAG_FILE as NOTIFIER_LAST_UPDATE_FLAG
except ImportError:
    log_to_console_logger("CRITICAL", "(console_ai) notifier_module.py not found. Versioning/update features disabled.")
    def init_versioning(): log_to_console_logger("WARNING", "(console_ai_fallback) init_versioning called, notifier_module missing.")
    def get_current_version(): return "N/A (notifier_module missing)"
    NOTIFIER_LAST_UPDATE_FLAG = os.path.join("meta", "last_update_fallback_console.json")

try:
    # These are for console UI commands like /prompt, /update_prompt
    from prompt_manager import load_prompt, update_prompt, auto_update_prompt
    # Note: AICore also instantiates a PromptManager. For true consistency, console commands
    # might eventually route through ai_core_singleton_instance.prompt_manager.
    # This is a deeper refactoring consideration beyond the current scope.
except ImportError:
    log_to_console_logger("CRITICAL", "(console_ai) prompt_manager.py not found. Prompt commands (/prompt, /update_prompt, /reflect) disabled.")
    def load_prompt(): log_to_console_logger("ERROR", "(console_ai_fallback) load_prompt: prompt_manager missing."); return "Fallback Prompt: Prompt manager missing."
    def update_prompt(p): log_to_console_logger("ERROR", "(console_ai_fallback) update_prompt: prompt_manager missing.")
    def auto_update_prompt(): log_to_console_logger("ERROR", "(console_ai_fallback) auto_update_prompt: prompt_manager missing.")

try:
    # For console UI commands like /suggestions, /approve, /reject
    from suggestion_engine import load_suggestions as load_suggestions_console, save_suggestions as save_suggestions_console, init_suggestions_file as init_suggestions_file_console
except ImportError:
    log_to_console_logger("ERROR", "(console_ai) suggestion_engine.py not found. Suggestion commands disabled.")
    def load_suggestions_console(): log_to_console_logger("ERROR", "(console_ai_fallback) load_suggestions_console: suggestion_engine missing."); return []
    def save_suggestions_console(s): log_to_console_logger("ERROR", "(console_ai_fallback) save_suggestions_console: suggestion_engine missing.")
    def init_suggestions_file_console(): log_to_console_logger("WARNING", "(console_ai_fallback) init_suggestions_file_console: suggestion_engine missing.")

# --- UI Enhancement Imports (Optional) ---
try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    # No log message here on purpose, it's an optional feature. Banner will indicate status.
    VOICE_ENABLED = False
    sr = None # Ensure sr is None if not imported

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False
    class EmptyColorama: # Minimal fallback for Fore and Style
        def __getattr__(self, name): return ""
    Fore = EmptyColorama()
    Style = EmptyColorama()

# --- Global Constants for Console File Paths & Settings ---
META_DIR = "meta"
GOAL_FILE_CONSOLE = os.path.join(META_DIR, "goals.json") # For /goals display, /addgoal
CHANGELOG_FILE_CONSOLE = os.path.join(META_DIR, "changelog.json") # For /changelog
ACTIVE_GOAL_FILE_CONSOLE = os.path.join(META_DIR, "active_goal.json") # For /status, shows GoalWorker's current focus
TOOL_REGISTRY_CONSOLE_FILE = os.path.join(META_DIR, "tool_registry.json") # For /stats
LAST_REFLECTION_CONSOLE_FILE = os.path.join(META_DIR, "last_reflection.json") # For /status, from prompt_manager

partial_input: str = "" # Buffer for typed user input
active_ai_streams = {} # Tracks active LLM response streams for UI printing

# Goal priorities (used by /addgoal_pri and /goals display formatting)
CONSOLE_PRIORITY_URGENT = 1
CONSOLE_PRIORITY_HIGH = 2
CONSOLE_PRIORITY_NORMAL = 3
CONSOLE_PRIORITY_LOW = 4

# Conversation history (managed by console_ai for prompt construction)
conversation_histories = {} # Key: thread_id, Value: list of {"speaker": "User/AI", "text": "..."}
MAX_CONVERSATION_HISTORY_TURNS = 10 # Store last N *pairs* of User/AI turns
last_ai_interaction_thread_id: str | None = None # Tracks the last thread AI responded on

# Conversation review logging settings (for console_ai's own review log)
CONVERSATION_REVIEW_LOG_FILE = os.path.join(META_DIR, "conversation_review_log.json")
MAX_CONVERSATION_LOG_ENTRIES = 50 # Max entries in the review log file
MAX_SNIPPET_TURNS = 6 # Number of individual turns (e.g., 3 User/AI pairs) for snippet
MAX_SNIPPET_TEXT_LEN = 250 # Max characters per turn in snippet

# --- Section 2: Global Helper Functions ---

def _ensure_meta_dir_console():
    """Ensures the META_DIR directory exists."""
    try:
        os.makedirs(META_DIR, exist_ok=True)
    except OSError as e:
        # Use the console's unified logger, established in Section 1
        log_to_console_logger("ERROR", f"(console_ai_ensure_meta) Could not create meta directory '{META_DIR}': {e}")

def color_text(text: str, color_code) -> str:
    """Applies color to text if COLOR_ENABLED."""
    # COLOR_ENABLED, Fore, Style are from Section 1
    if COLOR_ENABLED:
        return f"{color_code}{text}{Style.RESET_ALL}"
    return text

try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    sr = None # sr can be None

def format_timestamp_console(ts_input: str | int | float | None) -> str:
    """Formats a timestamp (ISO string or Unix epoch) into a human-readable local time string."""
    if not ts_input:
        return "Never"
    try:
        dt_obj = None
        if isinstance(ts_input, str):
            # Handle 'Z' for UTC, common in ISO 8601, for Python versions before 3.11
            if ts_input.endswith('Z'):
                if sys.version_info < (3, 11):
                    ts_input = ts_input[:-1] + '+00:00' # Replace 'Z' with '+00:00'
            try:
                dt_obj = datetime.fromisoformat(ts_input)
            except ValueError: # Fallback for slightly different ISO formats (e.g., no milliseconds)
                try:
                    dt_obj = datetime.strptime(ts_input.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    log_to_console_logger("DEBUG", f"(format_timestamp_console) Could not parse string timestamp '{ts_input}' with known ISO formats.")
                    pass # Could not parse with common ISO variants
            if dt_obj:
                if dt_obj.tzinfo: # If the datetime object is timezone-aware
                    dt_obj = dt_obj.astimezone(None) # Convert to local timezone
                return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(ts_input, (int, float)): # Assume Unix timestamp (seconds since epoch)
            dt_obj = datetime.fromtimestamp(ts_input)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e: # Broad exception to catch any parsing/formatting errors
        log_to_console_logger("WARNING", f"(format_timestamp_console) Error formatting timestamp '{ts_input}': {e}")
        return str(ts_input) # Fallback to string representation of input
    return str(ts_input) # Default fallback if no conversion happened

def add_turn_to_history(thread_id: str, speaker: str, text: str):
    """Adds a turn to the conversation history for a given thread_id."""
    # conversation_histories, MAX_CONVERSATION_HISTORY_TURNS are globals from Section 1
    global conversation_histories, MAX_CONVERSATION_HISTORY_TURNS
    if not thread_id:
        log_to_console_logger("WARNING", "(console_ai_history) Attempted to add turn without a valid thread_id.")
        return

    if thread_id not in conversation_histories:
        conversation_histories[thread_id] = []

    conversation_histories[thread_id].append({"speaker": speaker, "text": text.strip()})

    # Keep only the last N *pairs* of turns, so MAX_CONVERSATION_HISTORY_TURNS * 2 items
    if len(conversation_histories[thread_id]) > MAX_CONVERSATION_HISTORY_TURNS * 2:
        conversation_histories[thread_id] = conversation_histories[thread_id][-(MAX_CONVERSATION_HISTORY_TURNS * 2):]

def get_formatted_history(thread_id: str) -> str:
    """Formats the conversation history for a thread_id into a string for the LLM prompt."""
    # conversation_histories is a global from Section 1
    global conversation_histories
    if not thread_id or thread_id not in conversation_histories or not conversation_histories[thread_id]:
        return "" # No history for this thread or thread_id is None

    history_str_parts = [f"### Recent Conversation History (Thread: ..{thread_id[-6:]}):"]
    turns_to_include = conversation_histories[thread_id]

    for turn in turns_to_include:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        history_str_parts.append(f"{speaker}: {text}")

    if len(history_str_parts) > 1: # Means there was actual history beyond the header
        return "\n".join(history_str_parts) + "\n### End History\n"
    return "" # No actual turns to show

def log_conversation_snippet_for_review(thread_id: str):
    """Logs a snippet of the current conversation for a given thread_id to a review file."""
    # Globals from Section 1: conversation_histories, CONVERSATION_REVIEW_LOG_FILE, etc.
    global conversation_histories, CONVERSATION_REVIEW_LOG_FILE, MAX_CONVERSATION_LOG_ENTRIES, MAX_SNIPPET_TURNS, MAX_SNIPPET_TEXT_LEN

    if not thread_id or thread_id not in conversation_histories or not conversation_histories[thread_id]:
        log_to_console_logger("DEBUG", f"(log_snippet) No history for thread_id '{thread_id}' or history empty. Skipping log.")
        return

    _ensure_meta_dir_console() # Ensure meta directory exists
    review_log_entries = []
    if os.path.exists(CONVERSATION_REVIEW_LOG_FILE):
        try:
            with open(CONVERSATION_REVIEW_LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip(): # Ensure content is not just whitespace
                    loaded_entries = json.loads(content)
                    if isinstance(loaded_entries, list): # Ensure it's a list
                        review_log_entries = loaded_entries
                    else:
                        log_to_console_logger("WARNING", f"(log_snippet) Content of '{CONVERSATION_REVIEW_LOG_FILE}' is not a list. Starting fresh.")
                        review_log_entries = [] # Reset if not a list
        except (json.JSONDecodeError, IOError) as e:
            log_to_console_logger("WARNING", f"(log_snippet) Error reading conversation review log '{CONVERSATION_REVIEW_LOG_FILE}': {e}. Starting fresh.")
            review_log_entries = [] # Reset to empty list on error

    current_thread_history = conversation_histories[thread_id]
    # MAX_SNIPPET_TURNS defines how many individual messages to take from the end of the history.
    turns_to_log_for_snippet = current_thread_history[-MAX_SNIPPET_TURNS:]

    if not turns_to_log_for_snippet: # Don't log if there's no relevant history section for the snippet
        log_to_console_logger("DEBUG", f"(log_snippet) No turns to log for snippet from thread '{thread_id}'.")
        return

    snippet_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "conversation_snippet": []
    }
    for turn in turns_to_log_for_snippet:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        snippet_entry["conversation_snippet"].append({
            "speaker": speaker,
            "text": text[:MAX_SNIPPET_TEXT_LEN] + ('...' if len(text) > MAX_SNIPPET_TEXT_LEN else '')
        })

    review_log_entries.append(snippet_entry)

    # Keep the log from growing indefinitely
    if len(review_log_entries) > MAX_CONVERSATION_LOG_ENTRIES:
        review_log_entries = review_log_entries[-MAX_CONVERSATION_LOG_ENTRIES:]

    try:
        with open(CONVERSATION_REVIEW_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(review_log_entries, f, indent=2)
    except IOError as e:
        log_to_console_logger("ERROR", f"(log_snippet) Could not write to conversation review log '{CONVERSATION_REVIEW_LOG_FILE}': {e}")

def print_and_restore_prompt(message_to_print: str, is_stream_chunk: bool = False, stream_id: str | None = None, is_final_stream_chunk: bool = False):
    """Handles printing messages to the console while preserving and restoring the user's partial input line."""
    # Globals from Section 1: partial_input, active_ai_streams
    global partial_input, active_ai_streams

    prompt_display_text = color_text("You: ", Fore.GREEN) # color_text and Fore from Section 1
    current_user_input_line_len = len(prompt_display_text) + len(partial_input)

    # Clear the current input line
    sys.stdout.write('\r' + ' ' * current_user_input_line_len + '\r')

    if is_stream_chunk and stream_id:
        stream_state = active_ai_streams.get(stream_id)
        if stream_state and not stream_state["prefix_printed"]:
            # This is the FIRST chunk of a NEW AI stream.
            original_query_display = stream_state["original_query"][:40] + ('...' if len(stream_state["original_query"]) > 40 else '')
            stream_thread_id_hint = stream_state.get("thread_id_for_stream", "N/A")
            if stream_thread_id_hint != "N/A": stream_thread_id_hint = stream_thread_id_hint[-6:] # Show last 6 chars
            
            # Prefix ensures it starts on new lines relative to cleared input line.
            prefix = f"\n{color_text('🧠 AI (re: ', Fore.YELLOW)}'{original_query_display}{color_text(f"', Th: ..{stream_thread_id_hint}):", Fore.YELLOW)}\n" # Ends with a newline
            sys.stdout.write(prefix)
            stream_state["prefix_printed"] = True
        
        # For ALL stream chunks (first or subsequent), print the chunk.
        sys.stdout.write(message_to_print)

        if is_final_stream_chunk:
            sys.stdout.write('\n') # Add a newline after the *entire* streamed response.
    else: # This is for non-streamed messages
        sys.stdout.write(message_to_print + ('' if message_to_print.endswith('\n') else '\n'))

    # Restore the "You: " prompt line with partial input.
    sys.stdout.write(prompt_display_text + partial_input)
    sys.stdout.flush()

def load_goals_for_console() -> list:
    """Loads goals from the console's JSON file, with error handling and defaults."""
    _ensure_meta_dir_console() # Ensure meta directory exists
    # GOAL_FILE_CONSOLE, CONSOLE_PRIORITY_NORMAL from Section 1
    if not os.path.exists(GOAL_FILE_CONSOLE):
        return []
    try:
        with open(GOAL_FILE_CONSOLE, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip(): # Handle empty or whitespace-only file
                return []
            loaded_goals = json.loads(content)
            if not isinstance(loaded_goals, list):
                log_to_console_logger("ERROR", f"(load_goals_console) Content of '{GOAL_FILE_CONSOLE}' is not a list. Returning empty.")
                return []

            # Ensure essential fields for display/sorting, providing defaults
            for goal in loaded_goals:
                if not isinstance(goal, dict):
                    log_to_console_logger("WARNING", f"(load_goals_console) Skipping non-dictionary item in '{GOAL_FILE_CONSOLE}': {str(goal)[:100]}")
                    continue # Skip non-dict items if file is partially corrupt
                goal.setdefault("priority", CONSOLE_PRIORITY_NORMAL)
                goal.setdefault("thread_id", goal.get("thread_id", "N/A_" + str(uuid.uuid4())[:8])) # Ensure thread_id
                goal.setdefault("goal_id", goal.get("goal_id", "N/A_" + str(uuid.uuid4())[:8])) # Ensure goal_id
                goal.setdefault("created_at", goal.get("created_at", datetime.now(timezone.utc).isoformat()))
                goal.setdefault("status", goal.get("status", "pending"))
            return loaded_goals
    except (json.JSONDecodeError, FileNotFoundError) as e: # FileNotFoundError can happen if file deleted between check and open
        log_to_console_logger("WARNING", f"(load_goals_console) Error loading goals file '{GOAL_FILE_CONSOLE}': {e}")
        print_and_restore_prompt(color_text(f"Warning: Error loading goals for display: {e}", Fore.YELLOW))
        return []
    except Exception as e_load: # Catch other unexpected errors
        log_to_console_logger("ERROR", f"(load_goals_console) Unexpected error loading goals from '{GOAL_FILE_CONSOLE}': {e_load}\n{traceback.format_exc()}")
        print_and_restore_prompt(color_text(f"Critical Warning: Unexpected error loading goals: {e_load}", Fore.RED))
        return []

def save_goals_for_console(goals: list) -> None:
    """Saves the list of goals to the console's JSON file."""
    _ensure_meta_dir_console() # GOAL_FILE_CONSOLE from Section 1
    try:
        with open(GOAL_FILE_CONSOLE, "w", encoding="utf-8") as f:
            json.dump(goals, f, indent=2)
    except Exception as e:
        log_to_console_logger("ERROR", f"(save_goals_console) Error saving goals to file '{GOAL_FILE_CONSOLE}': {e}\n{traceback.format_exc()}")
        print_and_restore_prompt(color_text(f"Error saving goals from console: {e}", Fore.RED))

def add_goal_via_console(
    goal_text: str, *,
    source: str = "user_console", # Clarified source
    approved_by: str | None = None, # Typically "user" if added via console direct command
    priority: int = CONSOLE_PRIORITY_NORMAL, # CONSOLE_PRIORITY_NORMAL from Section 1
    thread_id_to_relate: str | None = None,
    parent_goal_id_to_relate: str | None = None
    ) -> None:
    """Adds a new goal object to the console's goals list and saves it."""
    current_goals = load_goals_for_console() # Load current goals
    new_goal_id = str(uuid.uuid4())
    effective_thread_id = thread_id_to_relate

    # If parent_goal_id is provided but no thread_id, try to inherit thread_id from parent
    if not effective_thread_id and parent_goal_id_to_relate:
        parent_obj = next((g for g in current_goals if isinstance(g, dict) and g.get("goal_id") == parent_goal_id_to_relate), None)
        if parent_obj:
            effective_thread_id = parent_obj.get("thread_id")
            log_to_console_logger("DEBUG", f"(add_goal_console) Inherited thread_id '..{str(effective_thread_id)[-6:]}' from parent goal '..{parent_goal_id_to_relate[-6:]}'.")
    # If still no thread_id (e.g. new goal, or parent had no thread_id), create a new one
    if not effective_thread_id:
        effective_thread_id = str(uuid.uuid4())
        log_to_console_logger("DEBUG", f"(add_goal_console) Generated new thread_id '..{effective_thread_id[-6:]}' for new goal.")

    timestamp_now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z") # Ensure 'Z' for UTC
    
    new_goal_obj = {
        "goal_id": new_goal_id, "thread_id": effective_thread_id, "goal": goal_text.strip(),
        "source": source, "status": "pending", "priority": priority,
        "is_subtask": bool(parent_goal_id_to_relate), "parent_goal_id": parent_goal_id_to_relate,
        "parent": None, "subtasks": None, # These are often populated by planner/executor later
        "self_correction_attempts": 0, "failure_category": None,
        "created_at": timestamp_now_iso, "updated_at": timestamp_now_iso, # Add updated_at
        "tool_file": None, "execution_result": None, "error": None, "evaluation": None,
        "history": [{"timestamp": timestamp_now_iso, "status": "pending", "action": f"Goal added via console (Priority: {priority}, Thread: ..{effective_thread_id[-6:]})"}]
    }
    if approved_by:
        new_goal_obj["approved_by"] = approved_by
    
    # Check for exact duplicates (same text, pending status, same thread) before adding
    is_dup = any(
        isinstance(g, dict) and
        g.get("goal", "").lower() == goal_text.strip().lower() and
        g.get("status") == "pending" and
        g.get("thread_id") == effective_thread_id
        for g in current_goals
    )
    if is_dup:
        log_to_console_logger("INFO", f"(add_goal_console) Duplicate pending goal addition attempted for thread ..{effective_thread_id[-6:]}: '{goal_text.strip()[:60]}...'")
        print_and_restore_prompt(color_text(f"Note: A similar pending goal already exists in thread ..{effective_thread_id[-6:]}: '{goal_text.strip()[:60]}...'", Fore.YELLOW))
        # Current logic appends anyway. If this should prevent adding, add 'return' here.

    current_goals.append(new_goal_obj)
    save_goals_for_console(current_goals)
    log_to_console_logger("INFO", f"(add_goal_console) Goal added. ID: ..{new_goal_id[-6:]}, Thread: ..{effective_thread_id[-6:]}, Prio: {priority}, Goal: '{goal_text.strip()[:60]}...'")

def load_changelog_for_console() -> list:
    """Loads changelog data from its JSON file."""
    _ensure_meta_dir_console() # CHANGELOG_FILE_CONSOLE from Section 1
    if not os.path.exists(CHANGELOG_FILE_CONSOLE):
        return []
    try:
        with open(CHANGELOG_FILE_CONSOLE, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip(): return [] # Handle empty or whitespace-only file
            loaded_changelog = json.loads(content)
            if not isinstance(loaded_changelog, list):
                log_to_console_logger("ERROR", f"(load_changelog_console) Content of '{CHANGELOG_FILE_CONSOLE}' is not a list. Returning empty.")
                return []
            return loaded_changelog
    except (json.JSONDecodeError, FileNotFoundError) as e:
        log_to_console_logger("WARNING", f"(load_changelog_console) Error loading changelog file '{CHANGELOG_FILE_CONSOLE}': {e}")
        return [] # Return empty on error
    except Exception as e_load_cl:
        log_to_console_logger("ERROR", f"(load_changelog_console) Unexpected error loading changelog from '{CHANGELOG_FILE_CONSOLE}': {e_load_cl}\n{traceback.format_exc()}")
        return []

def parse_indices_from_command(arg_str: str) -> list[int]:
    """Parses a string of comma-separated numbers and ranges (e.g., "1,2,5-7") into a sorted list of 0-based indices."""
    # This function primarily reports errors via print_and_restore_prompt for immediate user feedback.
    indices: set[int] = set()
    if not arg_str.strip(): # Handle empty or whitespace-only input string
        return []
    
    for part in arg_str.split(','):
        part = part.strip()
        if not part: # Skip empty parts resulting from multiple commas, e.g., "1,,2"
            continue
        if '-' in part: # Handle ranges like "1-3"
            try:
                start_str, end_str = part.split('-', maxsplit=1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start <= end and start > 0: # Ensure valid range (1-based index from user, start <= end)
                    indices.update(range(start - 1, end)) # Convert to 0-based for list indexing (exclusive end)
                else:
                    msg_range_err = f"Warning: Invalid range '{start_str}-{end_str}'. Numbers must be positive and start <= end."
                    print_and_restore_prompt(color_text(msg_range_err, Fore.YELLOW))
                    log_to_console_logger("DEBUG", f"(parse_indices) {msg_range_err}")
            except ValueError: # If int() conversion fails
                msg_val_err = f"Warning: Invalid range format '{part}'. Use numbers like '1-3'."
                print_and_restore_prompt(color_text(msg_val_err, Fore.YELLOW))
                log_to_console_logger("DEBUG", f"(parse_indices) {msg_val_err}")
        elif part.isdigit(): # Handle single numbers
            try:
                val = int(part)
                if val > 0: # Ensure 1-based index from user is positive
                    indices.add(val - 1) # Convert to 0-based
                else:
                    msg_num_err = f"Warning: Invalid number '{part}'. Must be positive."
                    print_and_restore_prompt(color_text(msg_num_err, Fore.YELLOW))
                    log_to_console_logger("DEBUG", f"(parse_indices) {msg_num_err}")
            except ValueError: # Should not happen if part.isdigit() is true, but as a safeguard
                msg_digit_err = f"Warning: Invalid number format '{part}' despite isdigit() check."
                print_and_restore_prompt(color_text(msg_digit_err, Fore.RED)) # More severe if logic fails
                log_to_console_logger("ERROR", f"(parse_indices) {msg_digit_err}")
        else: # Not a range, not a digit
            msg_fmt_err = f"Warning: Invalid format '{part}'. Use numbers or ranges like '1,2-4'."
            print_and_restore_prompt(color_text(msg_fmt_err, Fore.YELLOW))
            log_to_console_logger("DEBUG", f"(parse_indices) {msg_fmt_err}")
            
    return sorted(list(indices))

def is_status_or_greeting_query(text: str) -> bool:
    """Determines if input text is likely a status request or a simple greeting."""
    # No direct logging needed for this simple classification helper.
    text_lower = text.lower().strip()
    if not text_lower: return False

    # Status keywords and patterns (often questions)
    status_keywords = ["status", "progress", "working on", "your goals", "what are you doing", "what's your status"]
    # More specific patterns for status queries
    is_status = any(k in text_lower for k in status_keywords) and \
                ("?" in text or text_lower.startswith("what are you") or text_lower.startswith("what is your") or "tell me about your status" in text_lower)

    # Greeting keywords and patterns (typically short)
    greeting_keywords = ["how are you", "what's up", "hello", "hi ", " hi", "hey ", " hey", "good morning", "good afternoon", "good evening"]
    is_greeting = any(text_lower.startswith(k) for k in greeting_keywords) and len(text_lower.split()) < 5

    return is_status or is_greeting

# --- Section 3: Main Console Loop Function ---
def main_console_loop() -> None:
    # Global variables used in this loop, established in Section 1 or earlier in this function
    global partial_input, active_ai_streams, VOICE_ENABLED, last_ai_interaction_thread_id
    # goal_worker_instance_ref, log_to_console_logger, response_queue_console,
    # AI_CORE_MODEL_NAME_CONSOLE, get_response_async_console_func,
    # init_versioning, get_current_version, NOTIFIER_LAST_UPDATE_FLAG,
    # load_prompt, update_prompt, auto_update_prompt,
    # load_suggestions_console, save_suggestions_console, init_suggestions_file_console
    # Fore, Style, sr (if VOICE_ENABLED) were all set up in Section 1.

    worker_thread_instance: threading.Thread | None = None # To hold the GoalWorker thread
    recognizer: Optional[Recognizer] = None
    if VOICE_ENABLED and sr: # sr is None if import failed
        try:
            recognizer = sr.Recognizer()
        except Exception as e_voice_init:
            err_msg_voice = f"ERROR initializing voice recognition: {e_voice_init}. Voice commands disabled."
            log_to_console_logger("ERROR", f"(console_ai) {err_msg_voice}")
            # Direct print as this is early in console setup
            print(color_text(err_msg_voice, Fore.RED))
            VOICE_ENABLED = False
    else: # If sr is None because import failed
        VOICE_ENABLED = False


    voice_input_active_flag = False
    partial_input = "" # Clear partial input at the start

    try:
        _ensure_meta_dir_console()
        # These init functions are from direct imports in Section 1
        init_versioning()
        init_suggestions_file_console() # For console's /suggestions, distinct from AICore's SE

        # --- Print Welcome Banner ---
        print(color_text("🤖 Reflective AI Console Interface v2.5 (UI Input Refined)", Fore.CYAN))
        print(f"AI System Version: {get_current_version()} | LLM: {AI_CORE_MODEL_NAME_CONSOLE}")
        print(color_text("Type /help for options. Threading & Priorities Active.", Fore.CYAN))
        if VOICE_ENABLED:
            print(color_text("Voice commands available via /voice.", Fore.CYAN))
        else:
            print(color_text("Voice commands disabled (module missing or initialization failed).", Fore.YELLOW))

        sys.stdout.write(color_text("You: ", Fore.GREEN)) # Initial prompt
        sys.stdout.flush()

        # --- Start Goal Worker Thread ---
        if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'start_worker'):
            try:
                log_to_console_logger("INFO", "(console_ai) Attempting to start GoalWorker thread...")
                worker_thread_instance = goal_worker_instance_ref.start_worker()
                if worker_thread_instance and worker_thread_instance.is_alive():
                    log_to_console_logger("INFO", "(console_ai) GoalWorker thread started successfully.")
                else:
                    msg = "GoalWorker start_worker() did not return an active thread. Background processing may be non-functional."
                    log_to_console_logger("CRITICAL", f"(console_ai) {msg}")
                    print_and_restore_prompt(color_text(f"CRITICAL: {msg} Check logs.", Fore.RED))
                    worker_thread_instance = None
            except Exception as e_worker_start:
                msg = f"Exception starting GoalWorker: {e_worker_start}"
                log_to_console_logger("CRITICAL", f"(console_ai) {msg}\n{traceback.format_exc()}")
                print_and_restore_prompt(color_text(f"CRITICAL ERROR starting GoalWorker: {e_worker_start}. Background processing disabled.", Fore.RED))
                worker_thread_instance = None
        else:
            msg = "GoalWorker instance not available or lacks start_worker method (AICore init failed or GoalWorker component issue). Background processing disabled."
            log_to_console_logger("CRITICAL", f"(console_ai) {msg}")
            if goal_worker_instance_ref is None: # Only print to console if it's a major setup failure
                 print_and_restore_prompt(color_text(f"CRITICAL: {msg} Console functionality severely limited.", Fore.RED))
            # worker_thread_instance remains None

        last_activity_time = time.time()

        # --- Main Input and Message Processing Loop ---
        while True:
            processed_async_message_this_iteration = False

            # --- Process Asynchronous Messages from AI Core ---
            try:
                while not response_queue_console.empty(): # response_queue_console from Section 1
                    try:
                        original_query, chunk, is_final, stream_id_val, associated_thread_id = response_queue_console.get_nowait()
                        if stream_id_val not in active_ai_streams:
                            active_ai_streams[stream_id_val] = {"prefix_printed": False, "accumulated_text": "", "original_query": original_query, "thread_id_for_stream": associated_thread_id}
                        
                        print_and_restore_prompt(chunk, is_stream_chunk=True, stream_id=stream_id_val, is_final_stream_chunk=is_final)
                        
                        current_stream_state = active_ai_streams.get(stream_id_val)
                        if current_stream_state:
                            current_stream_state["accumulated_text"] += chunk
                            if is_final:
                                full_ai_response = current_stream_state["accumulated_text"].strip()
                                current_thread_for_log = current_stream_state["thread_id_for_stream"]
                                if current_thread_for_log:
                                    add_turn_to_history(current_thread_for_log, "AI", full_ai_response)
                                    last_ai_interaction_thread_id = current_thread_for_log
                                    log_conversation_snippet_for_review(current_thread_for_log)
                                del active_ai_streams[stream_id_val]
                        
                        processed_async_message_this_iteration = True
                        response_queue_console.task_done()
                    except queue.Empty: break
                    except Exception as e_resp_q_item:
                        err_msg = f"Error processing item from response_queue: {e_resp_q_item}"
                        log_to_console_logger("ERROR", f"(console_ai) {err_msg}\n{traceback.format_exc()}")
                        print_and_restore_prompt(color_text(err_msg, Fore.RED))
                        if 'stream_id_val' in locals() and stream_id_val in active_ai_streams: del active_ai_streams[stream_id_val]
                        try: response_queue_console.task_done() # Attempt task_done even on error if item was fetched
                        except ValueError: pass # If called too many times
            except Exception as e_outer_resp_q:
                 log_to_console_logger("CRITICAL", f"(console_ai) Outer error with response_queue processing: {e_outer_resp_q}\n{traceback.format_exc()}")

            # --- Process System Update Notifications ---
            if os.path.exists(NOTIFIER_LAST_UPDATE_FLAG): # NOTIFIER_LAST_UPDATE_FLAG from Section 1
                try:
                    with open(NOTIFIER_LAST_UPDATE_FLAG, "r", encoding="utf-8") as f_upd: note_data = json.load(f_upd)
                    update_msg = f"🆕 System Update → v{note_data.get('version', 'N/A')}: {note_data.get('summary', 'N/A')}"
                    log_to_console_logger("INFO", f"(console_ai) Displaying system update: {update_msg}")
                    print_and_restore_prompt(update_msg)
                    os.remove(NOTIFIER_LAST_UPDATE_FLAG)
                    processed_async_message_this_iteration = True
                except FileNotFoundError:
                    log_to_console_logger("WARNING", f"(console_ai) Update flag {NOTIFIER_LAST_UPDATE_FLAG} gone before read.")
                except json.JSONDecodeError as e_json_flag:
                    log_to_console_logger("ERROR", f"(console_ai) Error decoding update flag {NOTIFIER_LAST_UPDATE_FLAG}: {e_json_flag}")
                    print_and_restore_prompt(color_text("Error reading update notification: Malformed data.", Fore.RED))
                    try: os.remove(NOTIFIER_LAST_UPDATE_FLAG) # Attempt to remove corrupted flag
                    except Exception as e_rem_flag: log_to_console_logger("ERROR", f"(console_ai) Could not remove corrupted update flag: {e_rem_flag}")
                except Exception as e_flag:
                    log_to_console_logger("ERROR", f"(console_ai) Error processing update flag {NOTIFIER_LAST_UPDATE_FLAG}: {e_flag}\n{traceback.format_exc()}")
                    print_and_restore_prompt(color_text(f"Error processing update flag: {e_flag}", Fore.RED))

            if processed_async_message_this_iteration: last_activity_time = time.time()

            # --- Handle User Input (Voice or Keyboard) ---
            user_input_to_process: str | None = None

            if voice_input_active_flag and VOICE_ENABLED and recognizer and sr:
                # Original code (line 526) had set_pause(False). This implies ensuring the worker is *running*
                # (or unpaused if it was paused by a command). This seems a bit counter-intuitive if voice recognition
                # is CPU intensive, as one might expect to pause other heavy work.
                # However, to match the original structure, we will call set_pause(False).
                if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'set_pause'):
                    try:
                        log_to_console_logger("DEBUG", "(console_ai) Ensuring GoalWorker is not paused for voice input listening.")
                        goal_worker_instance_ref.set_pause(False) # Ensure worker is running
                    except Exception as e_gw_unpause_voice:
                        log_to_console_logger("ERROR", f"(console_ai) Error calling GoalWorker set_pause(False) for voice: {e_gw_unpause_voice}")
                try:
                    with sr.Microphone() as source_mic:
                        try: recognizer.adjust_for_ambient_noise(source_mic, duration=0.1)
                        except Exception as e_ambient: log_to_console_logger("DEBUG", f"(console_ai) Voice ambient noise adjustment failed: {e_ambient}")
                        print_and_restore_prompt(color_text("🎤 Listening... (say 'cancel voice' to stop)", Fore.CYAN))
                        audio_data = recognizer.listen(source_mic, timeout=5, phrase_time_limit=10)
                    print_and_restore_prompt(color_text("🎤 Recognizing...", Fore.YELLOW))
                    spoken_input_text = recognizer.recognize_google(audio_data)
                    log_to_console_logger("INFO", f"(console_ai) Voice recognized: '{spoken_input_text}'")
                    print_and_restore_prompt(color_text(f"🗣️ You said: {spoken_input_text}", Fore.BLUE))
                    if spoken_input_text.lower().strip() in ["cancel voice", "stop listening", "exit voice", "nevermind voice"]:
                        user_input_to_process = None
                        print_and_restore_prompt(color_text("🎤 Voice mode deactivated by phrase.", Fore.CYAN))
                    else: user_input_to_process = spoken_input_text
                except sr.WaitTimeoutError: pass
                except sr.UnknownValueError:
                    log_to_console_logger("INFO", "(console_ai) Voice input: Could not understand audio.")
                    print_and_restore_prompt(color_text("❓ Could not understand audio.", Fore.YELLOW))
                except sr.RequestError as e_sr_req:
                    log_to_console_logger("ERROR", f"(console_ai) Voice input: Speech service error: {e_sr_req}")
                    print_and_restore_prompt(color_text(f"Speech service error; {e_sr_req}", Fore.RED))
                except Exception as e_voice_gen:
                    log_to_console_logger("ERROR", f"(console_ai) Voice input: Unexpected error: {e_voice_gen}\n{traceback.format_exc()}")
                    print_and_restore_prompt(color_text(f"Voice input error: {e_voice_gen}", Fore.RED))
                finally:
                    voice_input_active_flag = False
                    if user_input_to_process: partial_input = ""
            
            elif msvcrt.kbhit(): # Keyboard input
                char_typed_bytes = msvcrt.getch()
                last_activity_time = time.time()
                try:
                    char_typed = char_typed_bytes.decode('utf-8', errors='ignore')
                    if not char_typed and len(char_typed_bytes) == 1: char_typed = char_typed_bytes.decode('cp1252', errors='ignore')
                    if char_typed in ('\r', '\n'):
                        sys.stdout.write('\n'); sys.stdout.flush()
                        user_input_to_process = partial_input; partial_input = ""
                    elif char_typed_bytes == b'\x08': # Backspace
                        if partial_input:
                            partial_input = partial_input[:-1]
                            sys.stdout.write('\r' + color_text("You: ", Fore.GREEN) + partial_input + ' ' + '\r' + color_text("You: ", Fore.GREEN) + partial_input)
                            sys.stdout.flush()
                    elif char_typed_bytes == b'\x03': raise KeyboardInterrupt # Ctrl+C
                    elif char_typed_bytes == b'\xe0' or char_typed_bytes == b'\x00': msvcrt.getch() # Special keys (arrows etc.), consume second byte
                    elif char_typed and char_typed.isprintable():
                        partial_input += char_typed
                        sys.stdout.write(char_typed); sys.stdout.flush()
                except UnicodeDecodeError:
                    log_to_console_logger("DEBUG", f"(console_ai) Keyboard input: UnicodeDecodeError for byte sequence: {char_typed_bytes}")
                    pass # Ignore

            # --- Process Full User Input (Command or Query) ---
            if user_input_to_process is not None:
                # Pause GoalWorker during command processing (original line 576, also 719/723 contextually)
                if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'set_pause'):
                    try:
                        log_to_console_logger("DEBUG", "(console_ai) Pausing GoalWorker for command/query processing.")
                        goal_worker_instance_ref.set_pause(True)
                    except Exception as e_gw_pause_cmd:
                        log_to_console_logger("ERROR", f"(console_ai) Error calling GoalWorker set_pause(True) for command processing: {e_gw_pause_cmd}")
                
                last_activity_time = time.time()
                cmd_lower = user_input_to_process.lower().strip()

                # --- Command Handlers ---
                if cmd_lower in {"/exit", "/quit"}:
                    print_and_restore_prompt(color_text("Exiting… Goodbye!", Fore.MAGENTA))
                    log_to_console_logger("INFO", "(console_ai) /exit or /quit command received. Shutting down.")
                    break # Exit the main while True loop

                elif cmd_lower == "/help":
                    help_parts = [
                        color_text("\nConsole AI Commands:", Fore.CYAN),
                        f"  {color_text('/goals', Fore.YELLOW):<40} - List active/pending goals",
                        f"  {color_text('/addgoal <text>', Fore.YELLOW):<40} - Add goal (normal prio, new thread)",
                        f"  {color_text('/addgoal_pri <prio> <text>', Fore.YELLOW):<40} - Add goal with priority (1-Urg..4-Low)",
                        f"  {color_text('/addgoal_thread <th_id> <text>', Fore.YELLOW):<40} - Add goal to existing thread",
                        f"  {color_text('/status', Fore.YELLOW):<40} - Show current AI status & active goal",
                        f"  {color_text('/stats', Fore.YELLOW):<40} - Display system statistics (goals, tools)",
                        f"  {color_text('/prompt', Fore.YELLOW):<40} - Show current full system prompt",
                        f"  {color_text('/update_prompt', Fore.YELLOW):<40} - Manually update operational system prompt",
                        f"  {color_text('/reflect', Fore.YELLOW):<40} - Trigger AI reflection on operational prompt",
                        f"  {color_text('/suggestions', Fore.YELLOW):<40} - List pending AI suggestions",
                        f"  {color_text('/approve <nums>', Fore.YELLOW):<40} - Approve suggestions to become goals",
                        f"  {color_text('/reject <nums>', Fore.YELLOW):<40} - Reject suggestions",
                        f"  {color_text('/clearsuggestions', Fore.YELLOW):<40} - Clear all pending suggestions",
                        f"  {color_text('/removegoal <nums>', Fore.YELLOW):<40} - Remove goal(s) by number from /goals list",
                        f"  {color_text('/recompose <num>', Fore.YELLOW):<40} - Reset a parent goal for decomposition",
                        f"  {color_text('/changelog', Fore.YELLOW):<40} - Show system changelog (last 10)",
                        f"  {color_text('/version', Fore.YELLOW):<40} - Show current system version",
                        f"  {color_text('/evolve', Fore.YELLOW):<40} - Activate/reinforce system evolution directive"
                    ]
                    if VOICE_ENABLED:
                        help_parts.append(f"  {color_text('/voice', Fore.YELLOW):<40} - Activate voice input mode")
                        help_parts.append(f"  {color_text('/voice_cancel', Fore.YELLOW):<40} - (If stuck) Deactivate voice input mode")
                    help_parts.append(f"  {color_text('/exit or /quit', Fore.YELLOW):<40} - Quit the program")
                    print_and_restore_prompt("\n".join(help_parts))

                elif cmd_lower.startswith("/addgoal_pri"):
                    parts = user_input_to_process.split(maxsplit=2)
                    if len(parts) < 3 or not parts[1].isdigit(): print_and_restore_prompt(color_text("Usage: /addgoal_pri <priority (1-4)> <goal_text>", Fore.RED))
                    else:
                        try:
                            priority_val = int(parts[1])
                            goal_description = parts[2]
                            if not (CONSOLE_PRIORITY_URGENT <= priority_val <= CONSOLE_PRIORITY_LOW): print_and_restore_prompt(color_text(f"Priority must be {CONSOLE_PRIORITY_URGENT}(U)-{CONSOLE_PRIORITY_LOW}(L).", Fore.RED))
                            else:
                                add_goal_via_console(goal_description, priority=priority_val) # Helper from Sec 2
                                print_and_restore_prompt(color_text(f"✅ Goal added (P{priority_val}): {goal_description[:60]}...", Fore.GREEN))
                        except ValueError: print_and_restore_prompt(color_text("Invalid priority for /addgoal_pri.", Fore.RED))

                elif cmd_lower.startswith("/addgoal_thread"):
                    parts = user_input_to_process.split(maxsplit=2)
                    if len(parts) < 3: print_and_restore_prompt(color_text("Usage: /addgoal_thread <thread_id_prefix> <goal_text>", Fore.RED))
                    else:
                        thread_id_prefix_search = parts[1]
                        goal_description = parts[2]
                        all_console_goals = load_goals_for_console() # Helper from Sec 2
                        found_thread_id = None
                        for g_con in reversed(all_console_goals):
                            if isinstance(g_con, dict) and g_con.get("thread_id","").startswith(thread_id_prefix_search):
                                found_thread_id = g_con["thread_id"]; break
                        if not found_thread_id:
                            log_to_console_logger("INFO", f"(console_ai) /addgoal_thread: No thread found for prefix '{thread_id_prefix_search}'. Adding to new thread.")
                            print_and_restore_prompt(color_text(f"No thread_id found starting with '{thread_id_prefix_search}'. Goal added to NEW thread.", Fore.YELLOW))
                            add_goal_via_console(goal_description, priority=CONSOLE_PRIORITY_NORMAL) # Helper from Sec 2
                            # add_goal_via_console logs success itself
                        else:
                            add_goal_via_console(goal_description, priority=CONSOLE_PRIORITY_NORMAL, thread_id_to_relate=found_thread_id) # Helper from Sec 2
                            # add_goal_via_console logs success
                
                elif cmd_lower.startswith("/addgoal"):
                    goal_description_simple = user_input_to_process.split(maxsplit=1)[1] if len(user_input_to_process.split(maxsplit=1)) > 1 else ""
                    if not goal_description_simple: print_and_restore_prompt(color_text("Usage: /addgoal <goal_text>", Fore.RED))
                    else:
                        new_thread_id = str(uuid.uuid4()) # Create a new thread for simple /addgoal
                        add_goal_via_console(goal_description_simple, priority=CONSOLE_PRIORITY_NORMAL, thread_id_to_relate=new_thread_id) # Helper from Sec 2
                        # add_goal_via_console logs success and prints its own confirmation

                elif cmd_lower == "/goals":
                    all_goals_list_display = load_goals_for_console() # Helper from Sec 2
                    output_parts = [color_text("\n--- Active / Pending Goals (Sorted by Effective Priority, then Age) ---", Fore.CYAN)]
                    parent_goals_display = []
                    sub_goals_map = {}
                    for g_obj_iter in all_goals_list_display:
                        if not isinstance(g_obj_iter, dict): continue
                        if g_obj_iter.get("is_subtask") and g_obj_iter.get("parent_goal_id"):
                            parent_id = g_obj_iter["parent_goal_id"]
                            if parent_id not in sub_goals_map: sub_goals_map[parent_id] = []
                            sub_goals_map[parent_id].append(g_obj_iter)
                        elif g_obj_iter.get("status") not in ("completed", "rejected_by_user", "failed_max_retries", "failed_correction_unclear"):
                            parent_goals_display.append(g_obj_iter)
                    
                    sorted_active_parent_goals = []
                    # Check if GoalWorker and its executor are available for dynamic scoring
                    # goal_worker_instance_ref is from Section 1
                    if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'executor') and goal_worker_instance_ref.executor and \
                       hasattr(goal_worker_instance_ref.executor, 'calculate_dynamic_score') and \
                       callable(goal_worker_instance_ref.executor.calculate_dynamic_score):
                        scored_parent_goals = []
                        for pg_obj in parent_goals_display:
                            try:
                                score = goal_worker_instance_ref.executor.calculate_dynamic_score(pg_obj)
                                scored_parent_goals.append((score, pg_obj))
                            except Exception as e_score:
                                log_to_console_logger("WARNING", f"(console_ai_/goals) Scoring failed for GID ..{pg_obj.get('goal_id','N/A')[-6:]}: {e_score}")
                                print_and_restore_prompt(color_text(f"Warning: Could not score goal GID ..{pg_obj.get('goal_id','N/A')[-6:]} for display: {e_score}", Fore.YELLOW))
                                scored_parent_goals.append((-float('inf'), pg_obj)) # Sort errored items last
                        sorted_active_parent_goals = [item[1] for item in sorted(scored_parent_goals, key=lambda x: x[0], reverse=True)]
                    else:
                        log_to_console_logger("WARNING", "(console_ai_/goals) GoalWorker executor or calculate_dynamic_score not available. Using basic sorting for /goals.")
                        sorted_active_parent_goals = sorted(parent_goals_display, key=lambda g_item: (g_item.get("priority", CONSOLE_PRIORITY_NORMAL), g_item.get("created_at", "1970-01-01T00:00:00Z")))

                    if sorted_active_parent_goals:
                        for idx, parent_g_obj in enumerate(sorted_active_parent_goals, 1):
                            # ... (Display logic for parent goals and subtasks - unchanged from original, but uses color_text) ...
                            # This part is extensive and primarily for UI formatting. I will keep it as in the original.
                            # Example of a line:
                            status_str = parent_g_obj.get('status', 'N/A'); pri_str = parent_g_obj.get('priority', 'N'); thread_hint_str = parent_g_obj.get('thread_id', '----')[-6:]
                            goal_txt_display = parent_g_obj.get('goal','Unnamed Parent Goal')[:70] + ('...' if len(parent_g_obj.get('goal','')) > 70 else '')
                            col_status_map = {'pending': Fore.YELLOW, 'decomposed': Fore.MAGENTA, 'approved': Fore.BLUE, 'awaiting_correction': Fore.LIGHTYELLOW_EX, 'build_failed': Fore.LIGHTRED_EX, 'executed_with_errors': Fore.LIGHTRED_EX, 'completed': Fore.GREEN}
                            col_status = col_status_map.get(status_str, Fore.WHITE)
                            output_parts.append(f"{color_text(str(idx)+'.', Fore.MAGENTA)} [P:{pri_str}] [{color_text(status_str, col_status)}] {goal_txt_display} (Th: {color_text('..'+thread_hint_str, Fore.LIGHTBLACK_EX if thread_hint_str != '----' else Fore.WHITE)})")
                            if parent_g_obj.get("failure_category") and status_str not in ['pending', 'approved', 'decomposed', 'completed']:
                                error_preview = str(parent_g_obj.get('error','N/A'))[:60].replace('\n', ' ').replace('\r', '')
                                output_parts.append(f"     └─ {color_text('Fail:', Fore.RED)} {parent_g_obj['failure_category']} - {error_preview}...")
                            parent_goal_id_val = parent_g_obj.get("goal_id")
                            actual_subtask_ids_from_parent = parent_g_obj.get("subtasks")
                            if status_str == 'decomposed' and parent_g_obj.get('subtask_descriptions') and isinstance(parent_g_obj.get('subtask_descriptions'), list):
                                for sub_idx, sub_desc_text in enumerate(parent_g_obj.get('subtask_descriptions',[]),1):
                                    sub_goal_obj_for_desc = None
                                    if actual_subtask_ids_from_parent and sub_idx <= len(actual_subtask_ids_from_parent):
                                        target_sub_id = actual_subtask_ids_from_parent[sub_idx-1]
                                        sub_goal_obj_for_desc = next((sg for sg in sub_goals_map.get(parent_goal_id_val, []) if sg.get("goal_id") == target_sub_id), None)
                                    sub_status_display = ""
                                    if sub_goal_obj_for_desc:
                                        sub_stat = sub_goal_obj_for_desc.get('status', 'N/A'); sub_col = col_status_map.get(sub_stat, Fore.WHITE)
                                        sub_status_display = f" [{color_text(sub_stat, sub_col)}]"
                                    output_parts.append(f"       {color_text(f'{idx}.{sub_idx}', Fore.LIGHTBLACK_EX)} {sub_desc_text[:60]}...{sub_status_display}")
                            elif parent_goal_id_val in sub_goals_map:
                                for sub_idx, sub_g_obj in enumerate(sub_goals_map[parent_goal_id_val], 1):
                                    if sub_g_obj.get("status") in ("completed", "rejected_by_user", "failed_max_retries"): continue
                                    sub_status_str = sub_g_obj.get('status', 'N/A'); sub_pri_str = sub_g_obj.get('priority', 'N')
                                    sub_goal_txt = sub_g_obj.get('goal', 'Unnamed Subtask')[:60] + ('...' if len(sub_g_obj.get('goal','')) > 60 else '')
                                    sub_col_status = col_status_map.get(sub_status_str, Fore.WHITE)
                                    output_parts.append(f"       {color_text(f'{idx}.{sub_idx}', Fore.LIGHTBLACK_EX)} [P:{sub_pri_str}] [{color_text(sub_status_str, sub_col_status)}] {sub_goal_txt}")
                                    if sub_g_obj.get("failure_category") and sub_status_str not in ['pending', 'approved', 'decomposed', 'completed']:
                                        sub_error_preview = str(sub_g_obj.get('error','N/A'))[:50].replace('\n', ' ').replace('\r','')
                                        output_parts.append(f"            └─ {color_text('Fail:', Fore.RED)} {sub_g_obj['failure_category']} - {sub_error_preview}...")
                    else: output_parts.append(color_text("  (No active non-subtask goals.)", Fore.GREEN))
                    archived_display_goals = sorted([g for g in all_goals_list_display if isinstance(g,dict) and g.get("status") in ("completed", "rejected_by_user", "failed_max_retries", "failed_correction_unclear") and not g.get("is_subtask")], key=lambda g_item: (g_item.get("history", [{}])[-1].get("timestamp", g_item.get("created_at", "1970-01-01T00:00:00Z"))), reverse=True)
                    output_parts.append(color_text("\n--- Archived Parent Goals (Last 5) ---", Fore.CYAN))
                    if archived_display_goals:
                         for idx_arc, g_arc_obj in enumerate(archived_display_goals[:5], 1):
                            status_arc_str = g_arc_obj.get('status', 'N/A'); goal_arc_txt_display = g_arc_obj.get('goal','Unnamed Goal')[:70] + ('...' if len(g_arc_obj.get('goal','')) > 70 else '')
                            col_arc_status = Fore.GREEN if status_arc_str == 'completed' else Fore.LIGHTBLACK_EX
                            output_parts.append(f"{color_text(str(idx_arc)+'.', Fore.LIGHTBLACK_EX)} [{color_text(status_arc_str, col_arc_status)}] {goal_arc_txt_display}")
                    else: output_parts.append(color_text("  (No archived parent goals.)", Fore.GREEN))
                    print_and_restore_prompt("\n".join(output_parts))

                elif cmd_lower == "/prompt": # Uses load_prompt from Section 1
                    print_and_restore_prompt(color_text("\nCurrent Full System Prompt:\n", Fore.CYAN) + load_prompt())
                
                elif cmd_lower == "/update_prompt": # Uses update_prompt from Section 1
                    print_and_restore_prompt(color_text("Enter new OPERATIONAL prompt. Empty line to save, Ctrl+C to cancel.", Fore.CYAN))
                    # Unpause worker while user types multiline prompt, if worker is pausable.
                    # Original logic: if callable(goal_worker.set_pause): goal_worker.set_pause(False)
                    if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'set_pause'):
                        try:
                            log_to_console_logger("DEBUG","(console_ai_/update_prompt) Ensuring GoalWorker is running for multiline input.")
                            goal_worker_instance_ref.set_pause(False)
                        except Exception as e_gw_unpause_prompt: log_to_console_logger("ERROR", f"(console_ai_/update_prompt) Error unpausing GoalWorker: {e_gw_unpause_prompt}")
                    
                    new_prompt_lines = []
                    try:
                        while True: # Loop for multiline input
                            # Display current partial input for this multiline session differently
                            sys.stdout.write(">> " + "".join(new_prompt_lines) + partial_input); sys.stdout.flush()
                            # Note: The original's "partial_input" usage here for multiline might be complex.
                            # A simpler readline() is usually used for multiline.
                            # The original code uses sys.stdin.readline() which is fine.
                            line_input = sys.stdin.readline().rstrip('\n') # Standard blocking readline
                            # partial_input is not directly used by readline, it was for char-by-char.
                            # Resetting console's global partial_input here if it was used by msvcrt.
                            # However, this input block is modal, so partial_input from msvcrt shouldn't interfere.
                            if not line_input and not new_prompt_lines: # Allow empty prompt if that's intended for clear
                                if not new_prompt_lines : # Only break if it's truly the first empty line
                                    break
                            elif not line_input and new_prompt_lines : # An empty line after some content means finish
                                break
                            new_prompt_lines.append(line_input + "\n") # Add newline for multiline structure
                    except KeyboardInterrupt:
                        print_and_restore_prompt(color_text("\nPrompt update cancelled.", Fore.YELLOW))
                        new_prompt_lines = None # Signal cancellation
                    
                    # No matter how exited, clear the console's global partial_input if it was affected
                    # (though it shouldn't be by stdin.readline())
                    # partial_input = "" # Already cleared before loop or after processing.

                    # Re-pause worker after multiline input session (matches the pause at start of command block)
                    if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'set_pause'):
                        try:
                            log_to_console_logger("DEBUG", "(console_ai_/update_prompt) Re-pausing GoalWorker after multiline input.")
                            goal_worker_instance_ref.set_pause(True)
                        except Exception as e_gw_repause_prompt: log_to_console_logger("ERROR", f"(console_ai_/update_prompt) Error re-pausing GoalWorker: {e_gw_repause_prompt}")

                    if new_prompt_lines is not None:
                        new_operational_prompt = "".join(new_prompt_lines).strip()
                        # update_prompt comes from prompt_manager (or its fallback) via Section 1
                        update_prompt(new_operational_prompt) # This saves only the operational part
                        if new_operational_prompt:
                             print_and_restore_prompt(color_text("✅ Operational system prompt updated.", Fore.GREEN))
                        else: print_and_restore_prompt(color_text("⚠️ Prompt update cancelled or cleared (empty).", Fore.YELLOW))
                
                elif cmd_lower == "/reflect": # Uses auto_update_prompt from Section 1
                    print_and_restore_prompt(color_text("INFO: Triggering AI reflection on operational prompt...", Fore.CYAN))
                    auto_update_prompt() # prompt_manager's function, logs its own success/failure
                    # The message below might be premature as auto_update_prompt might be async or take time.
                    print_and_restore_prompt(color_text("INFO: Reflection initiated. Monitor logs for updates to operational prompt.", Fore.CYAN))

                elif cmd_lower == "/evolve":
                    try:
                        import mission_manager # Direct import here as it's specific to this command's logic
                        mission_data_evolve = mission_manager.load_mission()
                        evolve_goal_text = mission_data_evolve.get('core_directive', "Continuously improve and evolve to be the most effective AI assistant.")
                        all_goals_evolve_check = load_goals_for_console() # Helper from Sec 2
                        existing_evolve_goal = next((g for g in all_goals_evolve_check if isinstance(g,dict) and "evolve to be" in g.get("goal","").lower() and g.get("status") == "pending" and g.get("source") == "system_directive"), None)
                        if existing_evolve_goal:
                            print_and_restore_prompt(color_text(f"INFO: System evolution directive already active (Goal ..{existing_evolve_goal['goal_id'][-6:]}).", Fore.CYAN))
                        else:
                            add_goal_via_console(evolve_goal_text, source="system_directive", priority=CONSOLE_PRIORITY_URGENT, thread_id_to_relate="SYSTEM_EVOLUTION_THREAD") # Helper from Sec 2
                            # add_goal_via_console prints its own success message.
                    except ImportError:
                        log_to_console_logger("ERROR", "(console_ai_/evolve) mission_manager.py not found. Cannot activate /evolve.")
                        print_and_restore_prompt(color_text("ERROR: mission_manager.py not found. Cannot activate /evolve.", Fore.RED))
                    except Exception as e_evolve:
                        log_to_console_logger("ERROR", f"(console_ai_/evolve) Error during /evolve: {e_evolve}\n{traceback.format_exc()}")
                        print_and_restore_prompt(color_text(f"ERROR during /evolve: {e_evolve}", Fore.RED))

                elif cmd_lower == "/status":
                    # ACTIVE_GOAL_FILE_CONSOLE from Section 1
                    active_goal_content = {}
                    if os.path.exists(ACTIVE_GOAL_FILE_CONSOLE):
                        try:
                            with open(ACTIVE_GOAL_FILE_CONSOLE, "r", encoding='utf-8') as f_active: content_active = f_active.read()
                            if content_active.strip(): active_goal_content = json.loads(content_active) # Corrected: load from content
                        except json.JSONDecodeError:
                             log_to_console_logger("WARNING", f"(console_ai_/status) {ACTIVE_GOAL_FILE_CONSOLE} is malformed.")
                             print_and_restore_prompt(color_text(f"Warning: {ACTIVE_GOAL_FILE_CONSOLE} is malformed.", Fore.YELLOW))
                        except Exception as e_ag_read:
                            log_to_console_logger("ERROR", f"(console_ai_/status) Error reading {ACTIVE_GOAL_FILE_CONSOLE}: {e_ag_read}")
                            active_goal_content = {}
                    
                    status_msg_parts = [color_text("\n=== Current AI Status ===\n", Fore.CYAN)]
                    if active_goal_content and isinstance(active_goal_content, dict) and active_goal_content.get("goal"):
                        active_thread_hint = str(active_goal_content.get('thread_id','N/A'))[-6:]
                        status_msg_parts.append(f"{color_text('Worker Active Goal:', Fore.YELLOW)} {str(active_goal_content['goal'])[:70]}... (P:{active_goal_content.get('priority','N')}, St:{active_goal_content.get('status','N/A')}, Th: ..{active_thread_hint})")
                    else: status_msg_parts.append(color_text("Worker not actively processing a specific goal (or active_goal.json is empty/stale).", Fore.GREEN))
                    
                    all_goals_status_list = load_goals_for_console() # Helper from Sec 2
                    next_pending_goal_obj = None
                    # Use GoalWorker's executor to find the true next goal
                    if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'executor') and goal_worker_instance_ref.executor and \
                       hasattr(goal_worker_instance_ref.executor, 'reprioritize_and_select_next_goal') and \
                       callable(goal_worker_instance_ref.executor.reprioritize_and_select_next_goal):
                        try:
                            next_pending_goal_obj = goal_worker_instance_ref.executor.reprioritize_and_select_next_goal(all_goals_status_list)
                        except Exception as e_reprio:
                            log_to_console_logger("ERROR", f"(console_ai_/status) Error calling reprioritize_and_select_next_goal: {e_reprio}")
                            print_and_restore_prompt(color_text(f"Error determining next goal via executor: {e_reprio}", Fore.RED))
                    else:
                        log_to_console_logger("WARNING", "(console_ai_/status) GoalWorker executor or reprioritize_and_select_next_goal not available. Using basic pending sort for /status.")
                        pending_goals_basic_sort = sorted([g for g in all_goals_status_list if isinstance(g,dict) and g.get('status')=='pending'], key=lambda x: (x.get('priority',CONSOLE_PRIORITY_NORMAL), x.get('created_at','')))
                        if pending_goals_basic_sort: next_pending_goal_obj = pending_goals_basic_sort[0]

                    if next_pending_goal_obj and isinstance(next_pending_goal_obj, dict):
                        next_thread_hint = str(next_pending_goal_obj.get('thread_id','N/A'))[-6:]
                        status_msg_parts.append(f"{color_text('Next Top Priority Goal:', Fore.YELLOW)} {str(next_pending_goal_obj['goal'])[:70]}... (P:{next_pending_goal_obj['priority']}, Th: ..{next_thread_hint})")
                    else: status_msg_parts.append(color_text("No pending goals in queue.", Fore.GREEN))

                    # load_suggestions_console from Section 1
                    pending_suggs_status = [s for s in load_suggestions_console() if isinstance(s,dict) and s.get("status", "pending") == "pending"]
                    status_msg_parts.append(f"{color_text(f'Pending Suggestions ({len(pending_suggs_status)}):', Fore.YELLOW)} " + (", ".join([str(s['suggestion'])[:30]+"..." for s in pending_suggs_status[:2]]) if pending_suggs_status else "(None)"))
                    
                    # LAST_REFLECTION_CONSOLE_FILE from Section 1
                    last_reflect_ts_val = None 
                    if os.path.exists(LAST_REFLECTION_CONSOLE_FILE):
                        try:
                            with open(LAST_REFLECTION_CONSOLE_FILE, "r", encoding='utf-8') as f_reflect: content_reflect = f_reflect.read()
                            if content_reflect.strip(): last_reflect_ts_val = json.loads(content_reflect).get("timestamp")
                        except Exception as e_lrf: log_to_console_logger("WARNING", f"(console_ai_/status) Error reading {LAST_REFLECTION_CONSOLE_FILE}: {e_lrf}")
                    status_msg_parts.append(f"{color_text('Last Ops Prompt Reflection:', Fore.YELLOW)} {format_timestamp_console(last_reflect_ts_val)}") # format_timestamp_console from Sec 2
                    print_and_restore_prompt("\n".join(status_msg_parts))

                elif cmd_lower == "/stats": # Uses load_goals_for_console, TOOL_REGISTRY_CONSOLE_FILE, load_suggestions_console
                    # This command's logic for assembling stats is complex but doesn't directly use goal_worker instance methods.
                    # It relies on reading files and the output of load_goals_for_console / load_suggestions_console.
                    # Keeping original logic, ensuring file paths and load functions are correct.
                    stats_out_parts = [color_text("\n--- System Statistics ---", Fore.CYAN)]
                    all_goals_stats = load_goals_for_console()
                    stats_out_parts.append(f"Total Goals Loaded: {len(all_goals_stats)}")
                    status_counts_stats, priority_counts_pending_stats = {}, {}; active_threads_stats = set()
                    for g_stat_item in all_goals_stats:
                        if not isinstance(g_stat_item, dict): continue
                        s, p = g_stat_item.get('status', 'N/A'), g_stat_item.get('priority', CONSOLE_PRIORITY_NORMAL)
                        status_counts_stats[s] = status_counts_stats.get(s, 0) + 1
                        if s == 'pending': priority_counts_pending_stats[p] = priority_counts_pending_stats.get(p, 0) + 1
                        active_threads_stats.add(g_stat_item.get("thread_id","N/A_thread"))
                    stats_out_parts.append("Goal Status Counts:"); 
                    for status_key_stat, count_val_stat in sorted(status_counts_stats.items()): stats_out_parts.append(f"  - {status_key_stat.capitalize()}: {count_val_stat}")
                    if priority_counts_pending_stats:
                        stats_out_parts.append("Pending Goal Priorities (Original):")
                        for pri_key_stat, ct_val_pri_stat in sorted(priority_counts_pending_stats.items()): stats_out_parts.append(f"    - P{pri_key_stat}: {ct_val_pri_stat}")
                    stats_out_parts.append(f"Unique Goal Threads: {len(active_threads_stats)}")
                    try: # TOOL_REGISTRY_CONSOLE_FILE from Section 1
                        if os.path.exists(TOOL_REGISTRY_CONSOLE_FILE):
                            with open(TOOL_REGISTRY_CONSOLE_FILE,'r',encoding='utf-8') as f_tr_stat: content_tr_stat = f_tr_stat.read()
                            tool_reg_stats_data = json.loads(content_tr_stat) if content_tr_stat.strip() else []
                            stats_out_parts.append(f"Registered Tools: {len(tool_reg_stats_data if isinstance(tool_reg_stats_data, list) else [])}")
                            if isinstance(tool_reg_stats_data, list):
                                total_runs_stat = sum(t.get('total_runs',0) for t in tool_reg_stats_data if isinstance(t, dict))
                                total_successes_stat = sum(t.get('success_count',0) for t in tool_reg_stats_data if isinstance(t, dict))
                                stats_out_parts.append(f"  - Total Tool Runs: {total_runs_stat}")
                                stats_out_parts.append(f"  - Total Tool Successes: {total_successes_stat} ({(total_successes_stat/total_runs_stat*100 if total_runs_stat else 0):.1f}%)")
                                unstable_tools_list_stat = [t.get('name') for t in tool_reg_stats_data if isinstance(t,dict) and t.get('total_runs',0) > 2 and (t.get('failure_count',0) / t['total_runs']) > 0.6]
                                if unstable_tools_list_stat: stats_out_parts.append(f"  - Potentially Unstable Tools (>2 runs, >60% fail): {', '.join(unstable_tools_list_stat)}")
                        else: stats_out_parts.append(f"Tool Registry file ('{TOOL_REGISTRY_CONSOLE_FILE}') not found for stats.")
                    except Exception as e_stat_tool:
                        log_to_console_logger("ERROR", f"(console_ai_/stats) Could not load tool stats: {e_stat_tool}")
                        stats_out_parts.append(f"Could not load tool stats: {e_stat_tool}")
                    suggestions_stats_list = load_suggestions_console() # load_suggestions_console from Section 1
                    stats_out_parts.append(f"Total Suggestions Stored (Console's view): {len(suggestions_stats_list)}")
                    sugg_status_counts_stats = {}
                    for s_stat_sugg_item in suggestions_stats_list:
                        if not isinstance(s_stat_sugg_item, dict): continue
                        sugg_stat_val = s_stat_sugg_item.get('status', 'unknown')
                        sugg_status_counts_stats[sugg_stat_val] = sugg_status_counts_stats.get(sugg_stat_val, 0) + 1
                    for sugg_status_key_stat, sugg_count_val_stat in sorted(sugg_status_counts_stats.items()): stats_out_parts.append(f"  - Suggestions {sugg_status_key_stat.capitalize()}: {sugg_count_val_stat}")
                    stats_out_parts.append("-------------------------")
                    print_and_restore_prompt("\n".join(stats_out_parts))

                elif cmd_lower == "/suggestions": # Uses load_suggestions_console, format_timestamp_console
                    suggs_list_disp = load_suggestions_console()
                    pending_suggs_disp = [s for s in suggs_list_disp if isinstance(s,dict) and s.get("status", "pending") == "pending"]
                    sugg_out_str = color_text("\n--- Pending AI Suggestions ---\n", Fore.CYAN) if pending_suggs_disp else color_text("\nNo pending AI suggestions.\n", Fore.YELLOW)
                    for i_sugg_disp, s_obj_disp in enumerate(pending_suggs_disp, 1):
                        sugg_text_disp = str(s_obj_disp.get('suggestion','N/A')).split("\n\nNone")[0].split("\n\n**None**")[0].strip()
                        sugg_text_preview = sugg_text_disp[:100] + ('...' if len(sugg_text_disp) > 100 else '')
                        ts_sugg_disp = format_timestamp_console(s_obj_disp.get('timestamp', 'N/A'))
                        sugg_thread_hint = s_obj_disp.get('related_thread_id')
                        sugg_out_str += f"{color_text(f'{i_sugg_disp}.', Fore.MAGENTA)} {sugg_text_preview} ({ts_sugg_disp})"
                        if sugg_thread_hint: sugg_out_str += f" (Rel.Th: ..{str(sugg_thread_hint)[-6:]})"
                        sugg_out_str += "\n"
                    print_and_restore_prompt(sugg_out_str.strip())

                elif cmd_lower.startswith("/approve") or cmd_lower.startswith("/reject"): # Uses parse_indices_from_command, load_suggestions_console, add_goal_via_console, save_suggestions_console
                    is_approve_cmd = cmd_lower.startswith("/approve")
                    cmd_name_sugg_act = "approve" if is_approve_cmd else "reject"
                    parts_sugg_act = user_input_to_process.split(maxsplit=1)
                    sugg_act_msg_str = ""
                    if len(parts_sugg_act) < 2: sugg_act_msg_str = color_text(f"Usage: /{cmd_name_sugg_act} <nums_from_/suggestions>\n", Fore.RED)
                    else:
                        indices_sugg_0b_act = parse_indices_from_command(parts_sugg_act[1]) # Helper from Sec 2
                        all_suggs_to_act_on = load_suggestions_console() # Func from Sec 1
                        pending_suggs_to_act_on = [s for s in all_suggs_to_act_on if isinstance(s,dict) and s.get("status", "pending") == "pending"]
                        action_taken_on_sugg = False
                        if not indices_sugg_0b_act and parts_sugg_act[1].strip(): sugg_act_msg_str += color_text(f"No valid suggestion numbers provided for /{cmd_name_sugg_act}.\n", Fore.YELLOW)
                        elif not indices_sugg_0b_act: sugg_act_msg_str += color_text(f"Please provide suggestion numbers to /{cmd_name_sugg_act}.\n", Fore.YELLOW)
                        processed_sugg_timestamps_for_this_command = set()
                        for idx_0b_s_act in sorted(indices_sugg_0b_act, reverse=True):
                            if 0 <= idx_0b_s_act < len(pending_suggs_to_act_on):
                                sugg_to_proc_obj = pending_suggs_to_act_on[idx_0b_s_act]
                                sugg_timestamp_id = sugg_to_proc_obj.get('timestamp')
                                if sugg_timestamp_id in processed_sugg_timestamps_for_this_command: continue
                                original_sugg_in_main_list_found = False
                                for i_orig_sugg_main, orig_sugg_main_obj in enumerate(all_suggs_to_act_on):
                                    if isinstance(orig_sugg_main_obj, dict) and orig_sugg_main_obj.get('timestamp') == sugg_timestamp_id and orig_sugg_main_obj.get('suggestion') == sugg_to_proc_obj.get('suggestion') and orig_sugg_main_obj.get("status","pending") == "pending":
                                        sugg_text_cleaned = str(sugg_to_proc_obj.get('suggestion','Error processing suggestion')).split("\n\nNone")[0].split("\n\n**None**")[0].strip()
                                        sugg_preview_for_log = sugg_text_cleaned[:60] + ('...' if len(sugg_text_cleaned) > 60 else '')
                                        if is_approve_cmd:
                                            all_suggs_to_act_on[i_orig_sugg_main]['status'] = 'approved_and_added_as_goal'
                                            sugg_related_thread_id = sugg_to_proc_obj.get('related_thread_id')
                                            if not sugg_related_thread_id: sugg_related_thread_id = str(uuid.uuid4())
                                            add_goal_via_console(sugg_text_cleaned, source="suggestion_approved", approved_by="user_console", priority=CONSOLE_PRIORITY_NORMAL, thread_id_to_relate=sugg_related_thread_id) # Helper from Sec 2
                                            sugg_act_msg_str += color_text(f"✅ Approved & added as goal: '{sugg_preview_for_log}' (Thread: ..{str(sugg_related_thread_id)[-6:]})\n", Fore.GREEN)
                                        else: # Reject
                                            all_suggs_to_act_on[i_orig_sugg_main]['status'] = 'rejected_by_user'
                                            sugg_act_msg_str += color_text(f"🗑️ Rejected suggestion: '{sugg_preview_for_log}'\n", Fore.YELLOW)
                                        action_taken_on_sugg = True; original_sugg_in_main_list_found = True; processed_sugg_timestamps_for_this_command.add(sugg_timestamp_id); break
                                if not original_sugg_in_main_list_found: sugg_act_msg_str += color_text(f"Error: Suggestion (idx {idx_0b_s_act+1}) not found or already processed.\n", Fore.RED)
                            else: sugg_act_msg_str += color_text(f"⚠️ Invalid suggestion number: {idx_0b_s_act + 1}.\n", Fore.RED)
                        if action_taken_on_sugg: save_suggestions_console(all_suggs_to_act_on) # Func from Sec 1
                        elif not sugg_act_msg_str: sugg_act_msg_str = color_text("No action taken. Ensure numbers are from /suggestions.", Fore.YELLOW)
                    print_and_restore_prompt(sugg_act_msg_str.strip() if sugg_act_msg_str else "No action on suggestions.")

                elif cmd_lower == "/clearsuggestions": # Uses load_suggestions_console, save_suggestions_console
                    all_suggs_clear = load_suggestions_console()
                    kept_suggs = [s for s in all_suggs_clear if not (isinstance(s,dict) and s.get("status","pending") == "pending")]
                    cleared_count = len(all_suggs_clear) - len(kept_suggs)
                    if cleared_count > 0:
                        save_suggestions_console(kept_suggs)
                        print_and_restore_prompt(color_text(f"🗑️ Cleared {cleared_count} pending suggestion(s).", Fore.YELLOW))
                    else: print_and_restore_prompt(color_text("No pending suggestions to clear.", Fore.YELLOW))

                elif cmd_lower.startswith("/removegoal"): # Uses parse_indices_from_command, load_goals_for_console, goal_worker_instance_ref.executor.calculate_dynamic_score, save_goals_for_console
                    parts_rem_cmd = user_input_to_process.split(maxsplit=1)
                    remove_msg_str = ""
                    if len(parts_rem_cmd) < 2: remove_msg_str = color_text("Usage: /removegoal <nums_from_active_goals_in_/goals list>\nNumbers refer to main (non-subtask) goals.", Fore.RED)
                    else:
                        indices_rem_0b_cmd = parse_indices_from_command(parts_rem_cmd[1])
                        all_goals_state_rem_cmd = load_goals_for_console()
                        parent_goals_for_rem = [g for g in all_goals_state_rem_cmd if isinstance(g,dict) and g.get("status") not in ("completed", "rejected_by_user", "failed_max_retries", "failed_correction_unclear") and not g.get("is_subtask")]
                        active_goals_for_rem_cmd = []
                        if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'executor') and goal_worker_instance_ref.executor and hasattr(goal_worker_instance_ref.executor, 'calculate_dynamic_score') and callable(goal_worker_instance_ref.executor.calculate_dynamic_score):
                            scored_parent_goals_rem = []
                            for pg_obj_rem in parent_goals_for_rem:
                                try: score_rem = goal_worker_instance_ref.executor.calculate_dynamic_score(pg_obj_rem)
                                except Exception as e_score_rem:
                                    log_to_console_logger("WARNING", f"(console_ai_/removegoal) Scoring failed for GID ..{pg_obj_rem.get('goal_id','N/A')[-6:]}: {e_score_rem}"); score_rem = -float('inf')
                                scored_parent_goals_rem.append((score_rem, pg_obj_rem))
                            active_goals_for_rem_cmd = [item[1] for item in sorted(scored_parent_goals_rem, key=lambda x_rem: x_rem[0], reverse=True)]
                        else:
                            log_to_console_logger("WARNING", "(console_ai_/removegoal) GoalWorker executor or score func not available. Basic sort.")
                            active_goals_for_rem_cmd = sorted(parent_goals_for_rem, key=lambda g_rem_item: (g_rem_item.get("priority", CONSOLE_PRIORITY_NORMAL), g_rem_item.get("created_at", "")))
                        goal_ids_to_rem_set_cmd = set(); valid_indices_provided = True
                        if not indices_rem_0b_cmd and parts_rem_cmd[1].strip(): remove_msg_str += color_text("No valid goal numbers for removal.\n", Fore.YELLOW); valid_indices_provided = False
                        for idx_0b_r_cmd in indices_rem_0b_cmd:
                            if 0 <= idx_0b_r_cmd < len(active_goals_for_rem_cmd):
                                goal_to_remove_obj = active_goals_for_rem_cmd[idx_0b_r_cmd]
                                goal_ids_to_rem_set_cmd.add(goal_to_remove_obj["goal_id"])
                                if isinstance(goal_to_remove_obj.get("subtasks"), list):
                                    for sub_id in goal_to_remove_obj.get("subtasks", []): goal_ids_to_rem_set_cmd.add(sub_id)
                                remove_msg_str += color_text(f"Marked goal '{str(goal_to_remove_obj.get('goal',''))[:30]}...' (and subtasks) for removal.\n", Fore.CYAN)
                            else: remove_msg_str += color_text(f"⚠️ Invalid goal number '{idx_0b_r_cmd + 1}'.\n", Fore.RED); valid_indices_provided = False
                        if valid_indices_provided and goal_ids_to_rem_set_cmd:
                            final_kept_goals_cmd = [g_kept for g_kept in all_goals_state_rem_cmd if not (isinstance(g_kept,dict) and g_kept.get("goal_id") in goal_ids_to_rem_set_cmd)]
                            removed_count_cmd = len(all_goals_state_rem_cmd) - len(final_kept_goals_cmd)
                            if removed_count_cmd > 0: save_goals_for_console(final_kept_goals_cmd); remove_msg_str += color_text(f"🗑️ Removed {removed_count_cmd} goal object(s).\n", Fore.YELLOW)
                            else: remove_msg_str += color_text("No goals ultimately removed.\n", Fore.YELLOW)
                        elif valid_indices_provided and not goal_ids_to_rem_set_cmd and parts_rem_cmd[1].strip(): remove_msg_str += color_text("No matching active parent goals for numbers.\n", Fore.YELLOW)
                        elif not valid_indices_provided and not goal_ids_to_rem_set_cmd : remove_msg_str = color_text("No valid goal numbers entered for removal.\n", Fore.YELLOW)
                    print_and_restore_prompt(remove_msg_str.strip() if remove_msg_str else "No removal action.")

                elif cmd_lower.startswith("/recompose"): # Uses load_goals_for_console, goal_worker_instance_ref.executor.calculate_dynamic_score, save_goals_for_console
                    parts_recomp_cmd = user_input_to_process.split(maxsplit=1) 
                    recomp_msg_str = ""
                    if len(parts_recomp_cmd) < 2 or not parts_recomp_cmd[1].strip().isdigit(): recomp_msg_str = color_text("Usage: /recompose <num_from_active_parent_goals_in_/goals list>\nOnly parent goals can be recomposed.", Fore.RED)
                    else:
                        idx_1b_recomp_cmd = int(parts_recomp_cmd[1].strip())
                        all_goals_recomp_cmd = load_goals_for_console()
                        active_parent_candidates_recomp = [g_rec_p for g_rec_p in all_goals_recomp_cmd if isinstance(g_rec_p,dict) and g_rec_p.get("status") not in ("completed", "rejected_by_user", "failed_max_retries", "failed_correction_unclear") and not g_rec_p.get("is_subtask")]
                        sorted_active_parents_recomp = []
                        if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'executor') and goal_worker_instance_ref.executor and hasattr(goal_worker_instance_ref.executor, 'calculate_dynamic_score') and callable(goal_worker_instance_ref.executor.calculate_dynamic_score):
                            active_parents_scored_recomp = []
                            for pg_obj_recomp in active_parent_candidates_recomp:
                                try: score_recomp = goal_worker_instance_ref.executor.calculate_dynamic_score(pg_obj_recomp)
                                except Exception as e_score_recomp: log_to_console_logger("WARNING", f"(console_ai_/recompose) Scoring GID ..{pg_obj_recomp.get('goal_id','N/A')[-6:]} failed: {e_score_recomp}"); score_recomp = -float('inf')
                                active_parents_scored_recomp.append((score_recomp, pg_obj_recomp))
                            sorted_active_parents_recomp = [item[1] for item in sorted(active_parents_scored_recomp, key=lambda x_p: x_p[0], reverse=True)]
                        else:
                            log_to_console_logger("WARNING","(console_ai_/recompose) GoalWorker executor or score func not available. Basic sort.")
                            sorted_active_parents_recomp = sorted(active_parent_candidates_recomp, key=lambda g_rec_p_item: (g_rec_p_item.get("priority", CONSOLE_PRIORITY_NORMAL), g_rec_p_item.get("created_at", "")))
                        if 0 < idx_1b_recomp_cmd <= len(sorted_active_parents_recomp):
                            goal_to_recomp_obj = sorted_active_parents_recomp[idx_1b_recomp_cmd - 1]
                            goal_id_recomp_val = goal_to_recomp_obj.get("goal_id"); goal_text_recomp_val = str(goal_to_recomp_obj.get("goal",""))[:50]
                            if goal_to_recomp_obj.get("is_subtask"): recomp_msg_str = color_text(f"⚠️ Goal '{goal_text_recomp_val}...' is a subtask. Recompose its parent.", Fore.YELLOW)
                            else:
                                modified_list_recomp_cmd = []; found_modified_recomp_cmd = False; subtasks_removed_count = 0
                                for g_main_iter_cmd in all_goals_recomp_cmd:
                                    if not isinstance(g_main_iter_cmd, dict): modified_list_recomp_cmd.append(g_main_iter_cmd); continue # Keep non-dict items
                                    if g_main_iter_cmd.get("goal_id") == goal_id_recomp_val:
                                        g_main_iter_cmd["status"] = "pending"; g_main_iter_cmd.pop("subtasks", None); g_main_iter_cmd.pop("subtask_descriptions", None)
                                        for key_pop_recomp in ["review", "tool_file", "execution_result", "error", "failure_category", "evaluation", "used_existing_tool"]: g_main_iter_cmd.pop(key_pop_recomp, None)
                                        g_main_iter_cmd.setdefault("history",[]).append({"timestamp":datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "status":"pending", "action":"Recomposed by user."})
                                        g_main_iter_cmd["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                                        modified_list_recomp_cmd.append(g_main_iter_cmd); found_modified_recomp_cmd = True
                                    elif g_main_iter_cmd.get("parent_goal_id") == goal_id_recomp_val: subtasks_removed_count += 1 # Remove subtask
                                    else: modified_list_recomp_cmd.append(g_main_iter_cmd)
                                if found_modified_recomp_cmd:
                                    save_goals_for_console(modified_list_recomp_cmd)
                                    recomp_msg_str = color_text(f"🔄 Goal '{goal_text_recomp_val}...' (GID ..{str(goal_id_recomp_val)[-6:]}) reset to PENDING. ", Fore.CYAN)
                                    if subtasks_removed_count > 0: recomp_msg_str += f"{subtasks_removed_count} previous subtask(s) removed.\n"
                                    else: recomp_msg_str += "No existing subtasks found/removed for this parent.\n"
                                else: recomp_msg_str = color_text(f"⚠️ Could not find GID {str(goal_id_recomp_val)[-6:]} to modify.", Fore.RED)
                        else: recomp_msg_str = color_text(f"⚠️ Invalid goal number '{idx_1b_recomp_cmd}'. Use num from active parent goals in /goals.\n", Fore.RED)
                    print_and_restore_prompt(recomp_msg_str.strip())

                elif cmd_lower == "/changelog": # Uses load_changelog_for_console, format_timestamp_console
                    cl_data_disp = load_changelog_for_console() # Func from Sec 2
                    cl_out_str = color_text("\n--- Changelog (Newest First - Max 10) ---\n", Fore.CYAN) if cl_data_disp else color_text("\nNo changelog entries.\n", Fore.YELLOW) # Corrected color
                    for entry_cl in reversed(cl_data_disp[-10:]):
                        if not isinstance(entry_cl, dict): continue
                        ts_cl_disp = format_timestamp_console(entry_cl.get('timestamp', 'N/A')) # Func from Sec 2
                        summary_cl_disp = str(entry_cl.get('summary','No summary.'))[:100] + ('...' if len(str(entry_cl.get('summary',''))) > 100 else '')
                        cl_out_str += f"  v{entry_cl.get('version','N/A')} ({ts_cl_disp}) by {entry_cl.get('approved_by','System')}: {summary_cl_disp}\n"
                    print_and_restore_prompt(cl_out_str.strip())

                elif cmd_lower == "/version": # Uses get_current_version from Section 1
                    print_and_restore_prompt(color_text(f"\nCurrent System Version: {get_current_version()}", Fore.CYAN))
                
                elif cmd_lower == "/voice_cancel":
                    if voice_input_active_flag:
                        voice_input_active_flag = False
                        print_and_restore_prompt(color_text("🎤 Voice input manually cancelled.", Fore.YELLOW))
                    else: print_and_restore_prompt(color_text("Voice input not currently active.", Fore.YELLOW))

                elif cmd_lower == "/voice" and VOICE_ENABLED:
                    voice_input_active_flag = True # Activate flag for next loop iteration
                    print_and_restore_prompt(color_text("🎤 Voice input mode will activate on next cycle. Say your command.", Fore.CYAN))
                
                elif not cmd_lower.startswith("/"): # Default: Not a known command, treat as input for AI
                    current_interaction_thread_id = last_ai_interaction_thread_id if last_ai_interaction_thread_id and not partial_input else str(uuid.uuid4())
                    if current_interaction_thread_id == last_ai_interaction_thread_id: log_to_console_logger("DEBUG", f"(console_ai) Continuing conversation on thread ..{current_interaction_thread_id[-6:]}")
                    else: log_to_console_logger("DEBUG", f"(console_ai) Starting new conversation on thread ..{current_interaction_thread_id[-6:]}")
                    
                    add_turn_to_history(current_interaction_thread_id, "User", user_input_to_process) # Helper from Sec 2
                    history_for_prompt = get_formatted_history(current_interaction_thread_id) # Helper from Sec 2
                    base_system_prompt = load_prompt() # Func from Sec 1 (prompt_manager or fallback)
                    prompt_payload_to_ai_core = f"{base_system_prompt}\n{history_for_prompt}".strip()

                    if is_status_or_greeting_query(user_input_to_process): # Helper from Sec 2
                        active_g_content_stat_llm = {} 
                        if os.path.exists(ACTIVE_GOAL_FILE_CONSOLE): # ACTIVE_GOAL_FILE_CONSOLE from Sec 1
                            try:
                                with open(ACTIVE_GOAL_FILE_CONSOLE, "r", encoding='utf-8') as f_stat_llm: content_active_stat = f_stat_llm.read()
                                if content_active_stat.strip(): active_g_content_stat_llm = json.loads(content_active_stat)
                            except Exception as e_stat_load: log_to_console_logger("WARNING", f"(console_ai) Error loading active goal for status query: {e_stat_load}")
                        status_context_llm = f"\n### System Status Context (for your information) ###\n- Active worker goal: {str(active_g_content_stat_llm.get('goal','None'))[:70]}\n"
                        all_g_ctx_query = load_goals_for_console() # Helper from Sec 2
                        pending_g_ctx_count = len([g for g in all_g_ctx_query if isinstance(g,dict) and g.get("status") == "pending"])
                        status_context_llm += f"- Total pending goals in queue: {pending_g_ctx_count}\n### End System Status Context ###\n"
                        final_prompt_for_status_query = f"{prompt_payload_to_ai_core}{status_context_llm}"
                        if get_response_async_console_func: # get_response_async_console_func from Sec 1
                            get_response_async_console_func(user_input_to_process, final_prompt_for_status_query, associated_thread_id=current_interaction_thread_id)
                        else: log_to_console_logger("ERROR", "(console_ai) get_response_async function not available (AICore issue). Cannot process query.")
                    else: # General query or implicit goal
                        goal_keywords = ["build", "make", "create", "generate", "develop", "implement", "add", "refactor", "fix", "write", "set up", "run", "test", "deploy", "research", "find", "analyze", "update", "optimize", "search"]
                        is_likely_goal = any(cmd_lower.startswith(w) for w in goal_keywords) or ("?" not in user_input_to_process and len(user_input_to_process.split()) >= 3)
                        if is_likely_goal:
                            add_goal_via_console(user_input_to_process, thread_id_to_relate=current_interaction_thread_id) # Helper from Sec 2
                            # add_goal_via_console prints its own confirmation.
                        elif get_response_async_console_func: # General LLM query
                             get_response_async_console_func(user_input_to_process, prompt_payload_to_ai_core, associated_thread_id=current_interaction_thread_id)
                        else: log_to_console_logger("ERROR", "(console_ai) get_response_async function not available (AICore issue). Cannot process query.")
                elif cmd_lower.startswith("/"): # Fallback for unknown slash commands
                    print_and_restore_prompt(color_text(f"⚠️ Unknown command: '{user_input_to_process}'. Try /help", Fore.RED))

                # Unpause GoalWorker after command/query has been processed (original line 1186 context)
                if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'set_pause'):
                    try:
                        log_to_console_logger("DEBUG", "(console_ai) Unpausing GoalWorker after command/query processing.")
                        goal_worker_instance_ref.set_pause(False)
                    except Exception as e_gw_unpause_cmd:
                        log_to_console_logger("ERROR", f"(console_ai) Error calling GoalWorker set_pause(False) after command processing: {e_gw_unpause_cmd}")
            
            # --- Loop Timing ---
            loop_duration = 0.02 # Minimum loop time
            if processed_async_message_this_iteration: loop_duration = 0.01
            elif not msvcrt.kbhit() and (time.time() - last_activity_time > 0.2): loop_duration = 0.05
            time.sleep(loop_duration)

    except KeyboardInterrupt:
        if partial_input: sys.stdout.write('\n') # Ensure newline if interrupted mid-input
        log_to_console_logger("INFO", "(console_ai) Ctrl+C detected. Exiting AI console...")
        print(color_text("\nCtrl+C detected. Exiting AI console...", Fore.MAGENTA))
    except EOFError: 
        if partial_input: sys.stdout.write('\n')
        log_to_console_logger("INFO", "(console_ai) EOF detected. Exiting AI console...")
        print(color_text("\nEOF detected. Exiting AI console...", Fore.MAGENTA))
    except Exception as e_main_unhandled:
        if partial_input: sys.stdout.write('\n')
        final_err_msg_main = f"CRITICAL ERROR in console_ai main loop: {e_main_unhandled}"
        log_to_console_logger("CRITICAL", f"(console_ai) {final_err_msg_main}\n{traceback.format_exc()}")
        print(color_text(final_err_msg_main, Fore.RED))
        traceback.print_exc() # Print traceback to console for immediate visibility
    finally:
        log_to_console_logger("INFO", "(console_ai) Console AI shutting down. Signaling worker...")
        print("\nINFO: Console AI shutting down. Signaling worker...")
        
        # Stop the GoalWorker thread
        if goal_worker_instance_ref and hasattr(goal_worker_instance_ref, 'stop') and callable(goal_worker_instance_ref.stop):
            try:
                log_to_console_logger("INFO", "(console_ai) Calling stop() on GoalWorker instance.")
                goal_worker_instance_ref.stop()
            except Exception as e_gw_stop:
                log_to_console_logger("ERROR", f"(console_ai) Error calling GoalWorker stop(): {e_gw_stop}")
        
        if worker_thread_instance and worker_thread_instance.is_alive():
            log_to_console_logger("INFO", "(console_ai) Waiting for GoalWorker thread to join (max 5s)...")
            print("INFO: Waiting for worker thread to join (max 5s)...")
            worker_thread_instance.join(timeout=5.0)
            if worker_thread_instance.is_alive():
                log_to_console_logger("WARNING", "(console_ai) GoalWorker thread did not terminate cleanly after 5s.")
                print(color_text("Warning: Worker thread did not terminate cleanly.", Fore.YELLOW))
            else:
                log_to_console_logger("INFO", "(console_ai) GoalWorker thread joined successfully.")
                print("INFO: Worker thread joined successfully.")
        elif worker_thread_instance is None and goal_worker_instance_ref is not None:
             log_to_console_logger("WARNING", "(console_ai) GoalWorker thread instance was None at shutdown, though goal_worker_instance_ref existed. Check startup logs.")
        elif goal_worker_instance_ref is None:
            log_to_console_logger("INFO", "(console_ai) GoalWorker was not available or not started; no thread to join.")
        else: # worker_thread_instance existed but was not alive
            log_to_console_logger("INFO", "(console_ai) GoalWorker thread was not active or already joined prior to shutdown sequence.")
            print("INFO: Worker thread was not active or already joined.")
            
        log_to_console_logger("INFO", "(console_ai) Shutdown sequence complete.")
        print("INFO: Shutdown complete.")
        if COLOR_ENABLED: print(Style.RESET_ALL) # Reset colors if colorama was used

# --- Section 4: Main Execution Block ---
if __name__ == "__main__":
    # The global 'partial_input' is already defined and initialized in Section 1.
    # Re-initializing it here is redundant but was in the original structure.
    # It doesn't harm the current refactoring's goals.
    # For cleaner code, this line could be removed if partial_input is reliably empty before main_console_loop.
    # However, main_console_loop also sets partial_input = "" at its beginning.
    partial_input = ""

    # Call the main console loop, which now incorporates all refactored logic.
    # log_to_console_logger is available globally if main_console_loop needs to log
    # something directly at this very top level of its call, though typically it uses it internally.
    log_to_console_logger("INFO", "(console_ai) Starting main_console_loop from __main__.")
    main_console_loop()