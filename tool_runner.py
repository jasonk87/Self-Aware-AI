# Self-Aware/tool_runner.py
import os
import sys
import importlib.util
import traceback
import json
import subprocess
from typing import Dict, Optional, Any, Callable, List, Union

def run_tool_safely(tool_path: str, tool_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Module-level wrapper for ToolRunner.run_tool_safely."""
    try:
        runner = ToolRunner()
        return runner.run_tool_safely(tool_path, tool_args)
    except Exception as e:
        print(f"[ERROR] Module-level run_tool_safely failed: {str(e)}")
        return {
            "status": "error",
            "error": f"Module-level execution error: {str(e)}",
            "traceback": traceback.format_exc()
        }

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_TOOLRUNNER_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (ToolRunnerClassFallbackLog) {message}")

class ToolRunner:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None,
                 query_llm_func: Optional[Callable[..., str]] = None):

        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_TOOLRUNNER_FALLBACK_LOGGER
        self.query_llm = query_llm_func if query_llm_func else lambda *args, **kwargs: "[Error: LLM query not available]"
        
        # Initialize paths and settings from config
        self.tools_dir_path = os.path.abspath(self.config.get("tools_dir", "tools"))
        self.tool_execution_timeout = self.config.get("tool_execution_timeout", 60)
        self.capture_tool_output = self.config.get("capture_tool_output", True)

        # Ensure tools directory exists
        os.makedirs(self.tools_dir_path, exist_ok=True)

    def run_tool_safely(self, tool_path: str, tool_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a Python tool with safety measures.
        
        Args:
            tool_path: Path to the Python tool to run
            tool_args: Optional list of command line arguments
            
        Returns:
            Dict with status and results/error information
        """
        try:
            if not os.path.exists(tool_path):
                return {"status": "error", "error": f"Tool not found: {tool_path}"}

            # Load module
            spec = importlib.util.spec_from_file_location("dynamic_tool", tool_path)
            if not spec or not spec.loader:
                return {"status": "error", "error": f"Failed to load tool spec: {tool_path}"}
            
            module = importlib.util.module_from_spec(spec)
            sys.modules["dynamic_tool"] = module
            spec.loader.exec_module(module)

            # Check for main() function
            if not hasattr(module, "main"):
                return {"status": "error", "error": f"Tool lacks main() function: {tool_path}"}

            # Run the tool
            try:
                # Save original args
                original_argv = sys.argv[:]
                
                # Set up new args
                if tool_args:
                    sys.argv = [tool_path] + tool_args
                else:
                    sys.argv = [tool_path]

                # Run the tool's main function
                result = module.main()

                # Restore original args
                sys.argv = original_argv

                return {
                    "status": "success",
                    "result": result
                }

            except Exception as run_e:
                self.logger("ERROR", f"(ToolRunner) Exception running tool {os.path.basename(tool_path)}: {str(run_e)}")
                return {
                    "status": "error",
                    "error": f"Runtime error: {str(run_e)}",
                    "traceback": traceback.format_exc()
                }

        except Exception as e:
            self.logger("ERROR", f"(ToolRunner) Failed to run tool {os.path.basename(tool_path)}: {str(e)}")
            return {
                "status": "error",
                "error": f"Tool execution error: {str(e)}",
                "traceback": traceback.format_exc()
            }

# --- End of ToolRunner Class ---

if __name__ == "__main__":
    # This block is an exact copy of your original __main__,
    # adapted to instantiate and use the ToolRunner class.
    print("--- Testing ToolRunner Class (Standalone from User's Full Code) ---")
    
    # Ensure meta and tools directories for testing
    test_meta_dir_tr = "meta_tr_test"
    test_tools_dir_tr = "tools_tr_test"
    os.makedirs(test_meta_dir_tr, exist_ok=True)
    os.makedirs(test_tools_dir_tr, exist_ok=True)

    # Mock config for testing
    mock_config_tr_main = {
        "meta_dir": test_meta_dir_tr,
        "tool_registry_file": os.path.join(test_meta_dir_tr, "tool_registry_test.json")
    }
    def main_test_logger_tr(level, message): print(f"[{level.upper()}] (TR_MainTest) {message}")

    tr_instance_main = ToolRunner(
        config=mock_config_tr_main,
        logger_func=main_test_logger_tr
    )

    # Create a dummy tool for testing
    dummy_tool_name_tr = "test_runner_tool_for_main_class"
    dummy_tool_path_tr = os.path.join(test_tools_dir_tr, f"{dummy_tool_name_tr}.py")
    dummy_tool_code_success_tr = """
import argparse, sys
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="Test success from dummy tool (ToolRunner class test)!")
    parser.add_argument("--exit_code", type=int, default=0)
    args = parser.parse_args()
    print(f"Dummy tool says: {args.message}")
    if args.exit_code != 0:
        print(f"Dummy tool exiting with code {args.exit_code}", file=sys.stderr)
        sys.exit(args.exit_code)
    return "Main function of dummy tool returned this (ToolRunner class test)."

if __name__ == "__main__":
    # Intentionally no sys.exit(main()) here for testing stdout/stderr capture from main itself
    main() 
""" 
    with open(dummy_tool_path_tr, "w", encoding="utf-8") as f:
        f.write(dummy_tool_code_success_tr)
    main_test_logger_tr("INFO", f"Created dummy tool for test: {dummy_tool_path_tr}")

    # Initialize dummy tool registry file
    if not os.path.exists(tr_instance_main.tool_registry_file_path):
        with open(tr_instance_main.tool_registry_file_path, "w", encoding='utf-8') as f_reg_tr:
            json.dump([], f_reg_tr)
        main_test_logger_tr("INFO", f"Initialized dummy registry: {tr_instance_main.tool_registry_file_path}")

    print("\n1. Testing successful run with args:")
    res1_main_tr = tr_instance_main.run_tool_safely(dummy_tool_path_tr, ["--message", "Custom message for ToolRunner class test"])
    print(f"   Result: {res1_main_tr['status']}, Output Snippet: '{str(res1_main_tr['output'])[:100]}', Error Snippet: '{str(res1_main_tr['error'])[:100]}'")

    # Create a tool that will cause a SystemExit via argparse
    dummy_tool_argparse_error_name = "test_argparse_error_tool_tr"
    dummy_tool_argparse_error_path = os.path.join(test_tools_dir_tr, f"{dummy_tool_argparse_error_name}.py")
    dummy_tool_argparse_error_code = """
import argparse, sys
def main():
    parser = argparse.ArgumentParser(description="Test tool that requires an argument.")
    parser.add_argument("--required_arg", required=True, help="A required argument.")
    args = parser.parse_args() # This will exit if --required_arg is not provided
    print(f"Required arg was: {args.required_arg}")
    return "Argparse tool ran with required arg."

if __name__ == "__main__":
    main() # No sys.exit here, argparse handles exit on error.
"""
    with open(dummy_tool_argparse_error_path, "w", encoding="utf-8") as f_err_tool:
        f.write(dummy_tool_argparse_error_code) # Corrected variable name here
    main_test_logger_tr("INFO", f"Created dummy argparse error tool: {dummy_tool_argparse_error_path}")
    
    print("\n2. Testing tool run that causes SystemExit due to missing required arg:")
    res2_main_tr = tr_instance_main.run_tool_safely(dummy_tool_argparse_error_path) # Run without args
    print(f"   Result: {res2_main_tr['status']}, Output Snippet: '{str(res2_main_tr['output'])[:100]}', Error Snippet: '{str(res2_main_tr['error'])[:200]}'") # Show more error
    if res2_main_tr['status'] != "executed_with_errors":
        print(f"   ERROR: Expected 'executed_with_errors', got '{res2_main_tr['status']}'")
        
    print(f"\nCheck '{tr_instance_main.tool_registry_file_path}' for updated stats.")
    print("--- ToolRunner Class Test Complete ---")

    # Clean up test files and directory
    # import shutil
    # if os.path.exists(test_tools_dir_tr): shutil.rmtree(test_tools_dir_tr)
    # if os.path.exists(test_meta_dir_tr): shutil.rmtree(test_meta_dir_tr)