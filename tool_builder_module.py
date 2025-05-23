# Self-Aware/tool_builder_module.py
import os
import json
import requests
import traceback
from typing import List, Dict, Optional, Any, Callable

def build_tool(description: str, tool_name_suggestion: Optional[str] = None, thread_id: Optional[str] = None, goals_list_ref: Optional[List[Dict[str, Any]]] = None, current_goal_id: Optional[str] = None) -> Optional[str]:
    """Module-level wrapper for ToolBuilder.build_tool."""
    try:
        builder = ToolBuilder()
        return builder.build_tool(description, tool_name_suggestion, thread_id, goals_list_ref, current_goal_id)
    except Exception as e:
        print(f"[ERROR] Module-level build_tool failed: {str(e)}")
        return None

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_TOOLBUILDER_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (ToolBuilderClassFallbackLog) {message}")
_CLASS_TOOLBUILDER_FALLBACK_QUERY_LLM = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=240, model_override=None, ollama_url_override=None, options_override=None: \
    f"# TOOL GENERATION FAILED (Fallback LLM Query)\n# Error: LLM query function not available to ToolBuilder instance.\n# Prompt: {prompt_text[:100]}..."


class ToolBuilder:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None):

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_TOOLBUILDER_FALLBACK_LOGGER
        
        # If query_llm_func is provided, use it. Otherwise, use a direct requests call with internal fallbacks.
        self._direct_llm_call_enabled = False
        if query_llm_func:
            self.query_llm = query_llm_func
        else:
            self.logger("WARNING", "(ToolBuilder Init) No query_llm_func provided. ToolBuilder will attempt direct HTTP requests to Ollama (less robust).")
            self._direct_llm_call_enabled = True
            self.query_llm = self._direct_ollama_generate_fallback

        # Configuration for paths and defaults
        self.ollama_url = self.config.get("ollama_url_for_tool_build", self.config.get("ollama_api_url", "http://localhost:11434/api/generate"))
        self.model_name = self.config.get("llm_model_for_tool_build", self.config.get("llm_model_name", "gemma:7b"))
        self.tools_dir_path = self.config.get("tools_dir", "tools")

        self.tool_template_string = '''You are an expert Python script generator. Your task is to create a single, self-contained Python script based on the "Goal" and "Thread Context" provided.

Goal:
"{description}"

Thread Context (Recent activity in the same logical sequence, if any):
{thread_context_str}

The script should:
1. Use argparse for command line arguments if needed
2. Include proper error handling and logging
3. Be a complete, runnable Python file
4. Include a main() function and __main__ check
5. Have clear docstrings and comments
6. Return results in a structured format (preferably JSON)
7. Print useful progress information

Respond ONLY with the complete Python script. No other text.
'''

    def _direct_ollama_generate_fallback(self, prompt_text, system_prompt_override=None, raw_output=False, timeout=240, model_override=None, ollama_url_override=None, options_override=None):
        """Direct Ollama HTTP request fallback if query_llm_func not provided."""
        try:
            data = {
                "model": model_override or self.model_name,
                "prompt": prompt_text,
                "system": system_prompt_override or "",
                "stream": False,
                "options": options_override or {}
            }
            response = requests.post(ollama_url_override or self.ollama_url, json=data, timeout=timeout)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"[Error: Ollama HTTP request failed with status {response.status_code}]"
        except Exception as e:
            return f"[Error: Direct Ollama request failed: {str(e)}]"

    def build_tool(self, description: str, tool_name_suggestion: Optional[str] = None, thread_id: Optional[str] = None, goals_list_ref: Optional[List[Dict[str, Any]]] = None, current_goal_id: Optional[str] = None) -> Optional[str]:
        """Build a Python tool from a description.
        
        Args:
            description: Tool purpose/functionality description
            tool_name_suggestion: Optional suggested name for the tool
            thread_id: Optional thread ID for context
            goals_list_ref: Optional list of goals for thread context
            current_goal_id: Optional current goal ID
            
        Returns:
            Path to the created tool file, or None if creation failed
        """
        try:
            # Create a safe filename for the tool
            if not tool_name_suggestion:
                tool_name_suggestion = "_".join(description[:40].split()).lower()
            safe_name = "".join(c for c in tool_name_suggestion if c.isalnum() or c in "._- ").replace(" ", "_").lower()
            safe_name = safe_name[:40] + "_" + (thread_id[-6:] if thread_id else "standalone")
            tool_path = os.path.join(self.tools_dir_path, f"{safe_name}.py")

            # Ensure tools directory exists
            os.makedirs(self.tools_dir_path, exist_ok=True)
            
            # Get thread context if available
            thread_context_str = ""
            if thread_id and goals_list_ref:
                thread_context_str = "\n".join([
                    f"Goal: {g.get('goal', '')}" 
                    for g in goals_list_ref 
                    if g.get("thread_id") == thread_id
                ][-3:])  # Last 3 goals

            # Generate the tool code
            prompt = self.tool_template_string.format(
                description=description,
                thread_context_str=thread_context_str or "No specific thread context available."
            )

            # Get the code from LLM
            tool_code = self.query_llm(prompt)

            # Write the code to file
            if tool_code and not tool_code.startswith("[Error:"):
                os.makedirs(os.path.dirname(tool_path), exist_ok=True)
                with open(tool_path, 'w', encoding='utf-8') as f:
                    f.write(tool_code)
                self.logger("INFO", f"(ToolBuilder) Successfully created tool at {tool_path}")
                return tool_path
            else:
                self.logger("ERROR", f"(ToolBuilder) Failed to generate code for tool: {description[:50]}... Error: {tool_code[:100]}")
                return None

        except Exception as e:
            self.logger("ERROR", f"(ToolBuilder) Exception building tool: {str(e)}")
            traceback.print_exc()
            return None

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