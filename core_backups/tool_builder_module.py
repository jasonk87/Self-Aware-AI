# Self-Aware/tool_builder_module.py
import os
import json
import requests # Retained from your original script
import traceback
from typing import List, Dict, Optional, Any, Callable # Added Callable

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_TOOLBUILDER_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (ToolBuilderClassFallbackLog) {message}")
_CLASS_TOOLBUILDER_FALLBACK_QUERY_LLM = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=240, model_override=None, ollama_url_override=None, options_override=None: \
    f"# TOOL GENERATION FAILED (Fallback LLM Query)\n# Error: LLM query function not available to ToolBuilder instance.\n# Prompt: {prompt_text[:100]}..."


class ToolBuilder:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None): # For making requests to Ollama via AICore's method

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_TOOLBUILDER_FALLBACK_LOGGER
        
        # If query_llm_func is provided, use it. Otherwise, use a direct requests call with internal fallbacks.
        # This allows ToolBuilder to function somewhat standalone if needed, but prefer injected query_llm_func.
        self._direct_llm_call_enabled = False
        if query_llm_func:
            self.query_llm = query_llm_func
        else:
            self.logger("WARNING", "(ToolBuilder Init) No query_llm_func provided. ToolBuilder will attempt direct HTTP requests to Ollama (less robust).")
            self._direct_llm_call_enabled = True
            # Fallback query_llm for direct calls if AICore's is not passed
            self.query_llm = self._direct_ollama_generate_fallback 


        # Configuration for paths and defaults, from original script or config
        # For direct HTTP calls (if query_llm_func not provided)
        self.ollama_url = self.config.get("ollama_url_for_tool_build", self.config.get("ollama_api_url", "http://localhost:11434/api/generate"))
        self.model_name = self.config.get("llm_model_for_tool_build", self.config.get("llm_model_name", "gemma3:4B"))
        
        self.tools_dir_path = self.config.get("tools_dir", "tools")
        # TOOL_REGISTRY_FILE is not directly used by build_tool in your script, but by executor when registering.
        
        self.tool_template_string = self.config.get("tool_builder_template", '''
You are an expert Python script generator. Your task is to create a single, self-contained Python script based on the "Goal" and "Thread Context" provided.

Goal:
"{description}"

Thread Context (Recent activity in the same logical sequence, if any):
{thread_context_str}

CRITICAL REQUIREMENTS FOR THE GENERATED TOOL:
1.  **Command-Line Arguments (`argparse`):** If the tool needs parameters (e.g., query, filename, URL), it MUST use Python's `argparse` module.
2.  **No Interactive `input()`:** The script MUST NOT use `input()` for parameters. It will be run non-interactively.
3.  **`main()` Function:**
    * MUST have a `main()` function encapsulating its primary logic.
    * This `main()` function MUST be callable.
    * If `main()` is run with no arguments (e.g., initial test), it should either:
        a) Use sensible hardcoded defaults to demonstrate functionality (e.g., 'test query').
        b) Or, if arguments are strictly required, `argparse` should print help and exit (the system handles this). DO NOT use `input()`.
4.  **Standard Libraries First:** Prefer standard Python libraries. Use third-party libraries (e.g., `requests`, `beautifulsoup4`) ONLY if essential and implied by the goal. Assume they are installed.
5.  **Output via `print()`:** Primary output should be via `print()`. The `main()` function can return values if useful for programmatic callers. Avoid GUIs unless explicitly asked.
6.  **Self-Contained Single File:** Generate a single Python script.
7.  **Clarity & Comments:** Write clear, well-commented code where appropriate.
8.  **Error Handling:** Implement basic try-except blocks for operations prone to failure (e.g., file I/O, network requests).

Output ONLY the raw Python code for the script. Do not include any explanations, comments (unless part of the Python code itself), or markdown formatting like ```python ... ```.
'''
        ) # End of TOOL_TEMPLATE
        
        # Ensure TOOLS_DIR exists
        try:
            os.makedirs(self.tools_dir_path, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(ToolBuilder class): Could not create tools directory {self.tools_dir_path}: {e}")

    def _direct_ollama_generate_fallback(self, prompt_text: str, system_prompt_override: Optional[str]=None, raw_output:bool=True, timeout: int=240, model_override:Optional[str]=None, ollama_url_override:Optional[str]=None, options_override:Optional[Dict[str,Any]]=None) -> str:
        """Internal fallback to make direct HTTP request if query_llm_func isn't provided."""
        # This method is only used if self.query_llm was set to this due to no injected func.
        # If a system_prompt_override is given with raw_output=True, it means the prompt_text IS the full API prompt.
        # If system_prompt_override is given with raw_output=False, then query_llm_internal from ai_core would typically wrap it.
        # This direct fallback is simpler and assumes prompt_text is the main content.
        
        actual_prompt = prompt_text # raw_output=True means prompt_text is already the full formatted API prompt.
        
        payload = {
            "model": model_override if model_override else self.model_name,
            "prompt": actual_prompt,
            "stream": False
        }
        if options_override: # Usually AICore's query_llm_internal handles default options
            payload["options"] = options_override
            
        current_url = ollama_url_override if ollama_url_override else self.ollama_url

        try:
            response = requests.post(current_url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except requests.exceptions.Timeout:
            self.logger("ERROR", f"(ToolBuilder DirectCall) Ollama request timed out to {current_url}.")
            return f"# TOOL GENERATION FAILED (DirectCall)\n# Error: LLM request timed out."
        except requests.exceptions.RequestException as e_req:
            self.logger("ERROR", f"(ToolBuilder DirectCall) Ollama request failed to {current_url}: {e_req}")
            return f"# TOOL GENERATION FAILED (DirectCall)\n# Error: LLM request error {e_req}."
        except json.JSONDecodeError as e_json_dec:
            self.logger("ERROR", f"(ToolBuilder DirectCall) Failed to decode Ollama JSON from {current_url}: {e_json_dec}. Raw: {response.text[:200] if 'response' in locals() else 'N/A'}")
            return f"# TOOL GENERATION FAILED (DirectCall)\n# Error: LLM JSON error."
        except Exception as e_unhandled_direct:
            self.logger("CRITICAL", f"(ToolBuilder DirectCall) Unexpected error during direct LLM call to {current_url}: {e_unhandled_direct}\n{traceback.format_exc()}")
            return f"# TOOL GENERATION FAILED (DirectCall)\n# Error: Unexpected error."


    def _get_thread_context_for_builder(self, thread_id: Optional[str], goals_list_ref: Optional[list], current_goal_id_to_exclude: Optional[str]) -> str:
        """Helper method to get thread context, identical to your global function."""
        if not thread_id or not goals_list_ref:
            return "(No specific thread context available for this tool build)"
        
        related_goals_info = []
        thread_goals = sorted(
            [g for g in goals_list_ref if isinstance(g, dict) and g.get("thread_id") == thread_id and g.get("goal_id") != current_goal_id_to_exclude],
            key=lambda x: x.get("created_at", ""), reverse=True
        )
        for g_ctx in thread_goals[:2]: # Limit to 2 most recent
            entry = f"- Prior Goal: \"{g_ctx.get('goal','N/A')[:70]}...\" (Status: {g_ctx.get('status','N/A')}" # Added .get for safety
            if g_ctx.get("tool_file"): entry += f", Used/Built Tool: {os.path.basename(g_ctx['tool_file'])}"
            if g_ctx.get("error"):
                error_str_ctx = str(g_ctx['error'])
                entry += f", Encountered Error: {error_str_ctx.splitlines()[0][:80] if error_str_ctx.splitlines() else error_str_ctx[:80]}" # Get first line
            entry += ")"
            related_goals_info.append(entry)
            
        if not related_goals_info:
            return "(No prior relevant goals found in this thread for context)"
        return "\n".join(related_goals_info)

    def build_tool(self, tool_description: str, tool_name: str, 
                   thread_id: Optional[str] = None, 
                   goals_list_ref: Optional[list] = None, 
                   current_goal_id: Optional[str] = None) -> str:
        """
        Generates Python script code for a tool based on description and context.
        Uses self.logger, self.query_llm (or _direct_ollama_generate_fallback), 
        self.tools_dir_path, self.tool_template_string, self.model_name, self.ollama_url.
        """
        filename = os.path.join(self.tools_dir_path, f"{tool_name}.py")
        
        # Ensure tools directory exists (might be redundant if called in __init__, but safe)
        try:
            os.makedirs(self.tools_dir_path, exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(ToolBuilder class): Could not create tools directory {self.tools_dir_path} in build_tool: {e}")
            # Decide on error handling: raise, or return path to a placeholder error script?
            # For now, will proceed and let LLM call potentially create placeholder if it fails.

        # Specialized handling for core_update_tool from your original script
        if tool_name.startswith("core_update_") and "apply core update to" in tool_description.lower():
            self.logger("INFO", f"(ToolBuilder class) Handling specialized core_update_tool generation for '{tool_name}'. Description assumed to be code.")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(tool_description) # tool_description is the code
                return filename
            except IOError as e_io_core:
                self.logger("ERROR", f"(ToolBuilder class) Could not write core_update code to {filename}: {e_io_core}")
                # Fall through to normal generation or return error path? For now, let it fall through
                # or return a path to an error script.
                # Let's return the filename, it will be an empty or non-existent file,
                # which subsequent checks in executor might catch.
                return filename


        thread_context_str = self._get_thread_context_for_builder(thread_id, goals_list_ref, current_goal_id)
        
        self.logger("INFO", f"(ToolBuilder class) Building tool '{tool_name}' for goal: '{tool_description[:50]}...' (Thread: ..{str(thread_id)[-6:] if thread_id else 'N/A'})")
        
        # Use the instance attribute for the template
        prompt = self.tool_template_string.format(description=tool_description, thread_context_str=thread_context_str)
        
        # Note: The query_llm_func passed from AICore is expected to handle payload construction.
        # If self.query_llm is the _direct_ollama_generate_fallback, it will construct its own payload.
        # The original global build_tool made a direct requests.post.
        # If using injected query_llm_func from AICore, it should manage the raw_output flag appropriately.
        # Here, we assume if self._direct_llm_call_enabled is True, self.query_llm is _direct_ollama_generate_fallback
        # which expects the full prompt. Otherwise, query_llm_func from AICore might expect different args.
        
        code = ""
        if self._direct_llm_call_enabled: # Use internal direct call logic
             code = self.query_llm( # This calls _direct_ollama_generate_fallback
                prompt_text=prompt, # The full prompt for Ollama's /api/generate
                timeout=self.config.get("tool_build_llm_timeout", 240),
                model_override=self.model_name, # Use class's model_name for direct call
                ollama_url_override=self.ollama_url # Use class's ollama_url for direct call
            )
        else: # Use injected query_llm_func (expected from AICore)
            # AICore's query_llm_internal expects:
            # prompt_text, system_prompt_override=None, raw_output=False, timeout=300, model_override=None, ollama_url_override=None, options_override=None
            # TOOL_TEMPLATE serves as a very detailed "system + user" prompt.
            # So we pass it as raw_output=True, meaning prompt_text is the full content for the API.
             code = self.query_llm(
                prompt_text=prompt, 
                raw_output=True, # Indicates prompt is already fully formed for the API
                timeout=self.config.get("tool_build_llm_timeout", 240), # Configurable timeout
                model_override=self.model_name, # Suggest model
                # ollama_url_override and options_override can be None if AICore's func uses its defaults
            )

        # Code cleanup (identical to your original script)
        if code.startswith("```python"): code = code[len("```python"):].strip()
        elif code.startswith("```"): code = code[len("```"):].strip()
        if code.endswith("```"): code = code[:-len("```")].strip()

        if not code.strip() and not code.startswith("# TOOL GENERATION FAILED"):
            self.logger("WARNING", f"(ToolBuilder class) Generated code for tool '{tool_name}' is empty. Creating placeholder.")
            code = f"# TOOL GENERATION for {tool_name} resulted in empty code.\nimport sys\ndef main():\n    print(\"Warning: Tool code is empty. LLM might have failed to respond adequately.\", file=sys.stderr)\n    return 1\nif __name__ == '__main__': sys.exit(main())"
        elif code.startswith("[Error:") and not code.startswith("# TOOL GENERATION FAILED"): # query_llm_internal error
            self.logger("ERROR", f"(ToolBuilder class) LLM call via query_llm_func failed for '{tool_name}': {code}")
            code = f"# TOOL GENERATION FAILED for {tool_name}\n# Error: {code}\nimport sys\ndef main():\n    print(f\"Error: Tool generation failed via injected LLM func ({code}).\", file=sys.stderr)\n    return 1\nif __name__ == '__main__': sys.exit(main())"


        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
        except IOError as e_io:
            self.logger("ERROR", f"(ToolBuilder class) Could not write tool code to {filename}: {e_io}")
            # Still return filename, executor checks existence.
        
        return filename

# --- End of ToolBuilder Class ---

if __name__ == "__main__":
    print("--- Testing ToolBuilder Class (Standalone) ---")
    
    # Mock config and logger for testing
    mock_config_tb_main = {
        "tools_dir": "tools_tb_test",
        "ollama_url_for_tool_build": "http://localhost:11434/api/generate", # Example
        "llm_model_for_tool_build": "gemma3:4B", # Example
        "tool_build_llm_timeout": 60 # Shorter for test
    }
    def main_test_logger_tb(level, message): print(f"[{level.upper()}] (TB_MainTest) {message}")

    # Test with direct HTTP call fallback first (query_llm_func=None)
    print("\n1. Testing with ToolBuilder's direct HTTP call to Ollama (if configured & Ollama running):")
    tb_instance_direct = ToolBuilder(
        config=mock_config_tb_main,
        logger_func=main_test_logger_tb,
        query_llm_func=None # Force direct call path if _direct_llm_call_enabled logic works
    )
    if not os.path.exists(tb_instance_direct.tools_dir_path):
        os.makedirs(tb_instance_direct.tools_dir_path, exist_ok=True)

    desc_direct = "Create a simple Python script that prints 'Hello from direct ToolBuilder test'."
    name_direct = "hello_tool_direct_tb"
    path_direct = tb_instance_direct.build_tool(desc_direct, name_direct)
    main_test_logger_tb("INFO", f"Direct call build result path: {path_direct}")
    if os.path.exists(path_direct):
        with open(path_direct, 'r') as f: main_test_logger_tb("INFO", f"Direct call tool content (first 100 chars): {f.read(100)}...")
    else: main_test_logger_tb("ERROR", f"Direct call tool file {path_direct} not created.")


    # Test with a mocked query_llm_func (simulating AICore's function)
    print("\n2. Testing with mocked query_llm_func:")
    def mock_query_llm_for_tb(prompt_text, raw_output=False, timeout=None, model_override=None, system_prompt_override=None, ollama_url_override=None, options_override=None):
        main_test_logger_tb("INFO", f"Mocked query_llm_func called by ToolBuilder. Prompt starts: {prompt_text[:100]}...")
        # Simulate a successful code generation
        return "import sys\n\ndef main():\n    print('Hello from mocked ToolBuilder test!')\n\nif __name__ == '__main__':\n    sys.exit(main())"

    tb_instance_mocked_llm = ToolBuilder(
        config=mock_config_tb_main,
        logger_func=main_test_logger_tb,
        query_llm_func=mock_query_llm_for_tb
    )
    if not os.path.exists(tb_instance_mocked_llm.tools_dir_path): # Ensure dir for this instance too
        os.makedirs(tb_instance_mocked_llm.tools_dir_path, exist_ok=True)

    desc_mocked = "Create a Python script that prints 'Hello from mocked ToolBuilder test'."
    name_mocked = "hello_tool_mocked_tb"
    test_goals_list_tb = [{"goal_id":"g_prev_tb", "thread_id":"thread_tb_1", "goal":"Prev task", "status":"completed", "created_at":"t"}]

    path_mocked = tb_instance_mocked_llm.build_tool(
        desc_mocked, name_mocked, 
        thread_id="thread_tb_1", 
        goals_list_ref=test_goals_list_tb, 
        current_goal_id="g_curr_tb_build"
    )
    main_test_logger_tb("INFO", f"Mocked LLM build result path: {path_mocked}")
    if os.path.exists(path_mocked):
        with open(path_mocked, 'r') as f:
            content = f.read()
            main_test_logger_tb("INFO", f"Mocked LLM tool content:\n{content}")
            if "Hello from mocked ToolBuilder test!" not in content:
                 main_test_logger_tb("ERROR", "Mocked LLM tool content seems incorrect.")
    else:
        main_test_logger_tb("ERROR", f"Mocked LLM tool file {path_mocked} not created.")
    
    # Clean up test directory
    # import shutil
    # if os.path.exists(mock_config_tb_main["tools_dir"]):
    # shutil.rmtree(mock_config_tb_main["tools_dir"])
    
    print("\n--- ToolBuilder Class Test Complete ---")