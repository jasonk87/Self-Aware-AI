# Self-Aware/tool_runner.py
import importlib.util
import traceback
import os
import sys 
import time 
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Callable, List # Added List and Callable

# Fallback logger for standalone testing or if no logger is provided to the class instance
_CLASS_TOOLRUNNER_FALLBACK_LOGGER = lambda level, message: print(f"[{level.upper()}] (ToolRunnerClassFallbackLog) {message}")

class ToolRunner:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 logger_func: Optional[Callable[[str, str], None]] = None):
        self.config = config if config else {}
        self.logger = logger_func if logger_func else _CLASS_TOOLRUNNER_FALLBACK_LOGGER

        # Initialize path for the tool registry file
        # Default from your original script: os.path.join("meta", "tool_registry.json")
        # This assumes META_DIR is "meta" if not specified in config.
        meta_dir_from_config = self.config.get("meta_dir", "meta")
        self.tool_registry_file_path = self.config.get("tool_registry_file", 
                                                  os.path.join(meta_dir_from_config, "tool_registry.json"))
        
        # Ensure the directory for the tool registry exists
        try:
            os.makedirs(os.path.dirname(self.tool_registry_file_path), exist_ok=True)
        except OSError as e:
            self.logger("ERROR", f"(ToolRunner class): Could not create directory for tool registry: {os.path.dirname(self.tool_registry_file_path)}: {e}")


    def _update_tool_registry_stats(self, tool_name_no_ext: str, run_status: str, 
                                   error_message: Optional[str] = None, 
                                   args_used: Optional[List[str]] = None):
        """Updates run statistics and last_run_time for a tool in the registry."""
        # This logic is exactly from your script, using self.tool_registry_file_path and self.logger
        
        # Ensure directory exists (might be redundant if constructor also does it, but safe)
        try:
            os.makedirs(os.path.dirname(self.tool_registry_file_path), exist_ok=True)
        except OSError as e_dir:
            self.logger("ERROR", f"(ToolRunner|_update_stats): Could not ensure directory for {self.tool_registry_file_path}: {e_dir}")
            return # Cannot proceed if directory can't be made

        registry_data: List[Dict[str, Any]] = [] # Ensure type
        if os.path.exists(self.tool_registry_file_path):
            try:
                with open(self.tool_registry_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        loaded_json = json.loads(content)
                        if isinstance(loaded_json, list): # Expect a list of tool dicts
                            registry_data = loaded_json
                        else:
                            self.logger("WARNING", f"(ToolRunner|_update_stats): Tool registry at {self.tool_registry_file_path} is not a list. Reinitializing.")
                            registry_data = []
            except (json.JSONDecodeError, IOError) as e_load_reg:
                 self.logger("WARNING", f"(ToolRunner|_update_stats): Error loading tool registry {self.tool_registry_file_path}: {e_load_reg}. Reinitializing.")
                 registry_data = []
        
        found_tool_entry = None
        for entry in registry_data: # Iterate through the list of dicts
            if isinstance(entry, dict) and entry.get("name") == tool_name_no_ext:
                found_tool_entry = entry
                break
        
        current_time_iso = datetime.now(timezone.utc).isoformat()

        if found_tool_entry: # found_tool_entry is already a dict reference from registry_data
            found_tool_entry["last_run_time"] = current_time_iso
            found_tool_entry["total_runs"] = found_tool_entry.get("total_runs", 0) + 1
            if run_status == "success":
                found_tool_entry["success_count"] = found_tool_entry.get("success_count", 0) + 1
            else:
                found_tool_entry["failure_count"] = found_tool_entry.get("failure_count", 0) + 1
                if error_message:
                    if "failure_details" not in found_tool_entry or not isinstance(found_tool_entry["failure_details"], list):
                        found_tool_entry["failure_details"] = []
                    failure_detail: Dict[str, Any] = {"timestamp": current_time_iso, "error": error_message[:500]} # Type hint
                    if args_used: failure_detail["args_used"] = args_used
                    found_tool_entry["failure_details"].append(failure_detail)
                    # Keep only the last 5 failure details (as in original logic)
                    found_tool_entry["failure_details"] = found_tool_entry["failure_details"][-5:] 
        else:
            self.logger("WARNING", f"(ToolRunner|_update_stats): Tool '{tool_name_no_ext}' not in registry. Creating new entry.")
            placeholder_entry: Dict[str, Any] = { # Type hint
                "name": tool_name_no_ext,
                "module_path": f"tools/{tool_name_no_ext}.py", # Default assumption for path
                "description": "Auto-registered placeholder by tool_runner.",
                "capabilities": [], "args_info": [],
                "last_updated": current_time_iso, # Should be when tool was created/modified ideally
                "last_run_time": current_time_iso,
                "success_count": 1 if run_status == "success" else 0,
                "failure_count": 1 if run_status != "success" else 0,
                "total_runs": 1,
                "failure_details": []
            }
            if error_message and run_status != "success":
                failure_detail_new: Dict[str, Any] = {"timestamp": current_time_iso, "error": error_message[:500]}
                if args_used: failure_detail_new["args_used"] = args_used
                placeholder_entry["failure_details"].append(failure_detail_new)
            registry_data.append(placeholder_entry)

        try:
            with open(self.tool_registry_file_path, 'w', encoding='utf-8') as f_write:
                json.dump(registry_data, f_write, indent=2)
        except IOError as e_io_reg_write:
            self.logger("ERROR", f"(ToolRunner|_update_stats): Could not update tool registry {self.tool_registry_file_path}: {e_io_reg_write}")

    def run_tool_safely(self, tool_path: str, tool_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs a Python tool script safely, capturing its output and errors.
        Uses self.logger and self._update_tool_registry_stats.
        """
        # This logic is exactly from your script.
        result: Dict[str, Any] = {"tool": tool_path, "status": "unknown", "output": "", "error": ""}
        tool_args_effective = tool_args if tool_args is not None else [] # Ensure tool_args is a list

        tool_name_no_ext = os.path.splitext(os.path.basename(tool_path))[0]

        if not os.path.exists(tool_path):
            result["status"] = "not_found"
            result["error"] = f"File does not exist: {tool_path}"
            # _update_tool_registry_stats is called in finally block, so this specific call might be redundant
            # However, keeping it here for clarity if the original script had it here before a finally.
            # Original script called it in finally, which is better.
            # For now, I will follow the original placement if it was not in a finally block.
            # Re-checking your original: it calls _update_tool_registry_stats in finally.
            # So, this direct call is not needed here.
            # self._update_tool_registry_stats(tool_name_no_ext, result["status"], result["error"], tool_args_effective)
            return result # Return early as in original logic if file not found

        original_argv = list(sys.argv) 
        # Create a more unique module name to avoid collisions if tools are run rapidly or have similar names
        unique_module_name = f"generated_tool_mod_{tool_name_no_ext}_{os.getpid()}_{time.time_ns()}" 

        try:
            simulated_argv_for_tool = [tool_path] + tool_args_effective # Use effective args
            sys.argv = simulated_argv_for_tool 
            
            spec = importlib.util.spec_from_file_location(unique_module_name, tool_path)
            
            if spec is None or spec.loader is None: 
                result["status"] = "import_error"
                result["error"] = f"Could not create module spec for {tool_path}."
                # No early return here, finally block will handle stat update
            else:    
                module_obj = importlib.util.module_from_spec(spec)
                sys.modules[unique_module_name] = module_obj # Add to sys.modules before execution
                spec.loader.exec_module(module_obj) # Execute the module's code
                
                if hasattr(module_obj, "main"):
                    self.logger("INFO", f"(ToolRunner|run_tool) 🔧 Running main() in {tool_path} with args: {tool_args_effective if tool_args_effective else 'None'}...")
                    tool_run_output = module_obj.main() 
                    result["status"] = "success"
                    result["output"] = str(tool_run_output) if tool_run_output is not None else "" 
                else:
                    result["status"] = "no_main"
                    result["error"] = f"No main() function found in tool: {tool_path}"
                    self.logger("WARNING", f"(ToolRunner|run_tool): {result['error']}")

        except SystemExit as se: # From your original script
            result["status"] = "executed_with_errors" 
            error_msg_system_exit = f"Tool exited with SystemExit code {se.code}. Likely argparse error or explicit sys.exit."
            self.logger("INFO", f"(ToolRunner|run_tool): {error_msg_system_exit} for {tool_path}")
            result["error"] = f"{error_msg_system_exit}\nFull Traceback:\n{traceback.format_exc()}"
        except Exception as e:
            result["status"] = "error" 
            error_detail_traceback = traceback.format_exc()
            result["error"] = error_detail_traceback
            self.logger("ERROR", f"(ToolRunner|run_tool): Exception running tool {tool_path}: {e}\n{error_detail_traceback}")
        finally:
            sys.argv = original_argv # Restore original sys.argv
            if unique_module_name in sys.modules: 
                del sys.modules[unique_module_name] # Clean up module from sys.modules
            
            # Update stats in finally block to ensure it's always called
            self._update_tool_registry_stats(tool_name_no_ext, result["status"], result.get("error"), tool_args_effective)

        return result

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