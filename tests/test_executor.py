import unittest
import os
import sys

# Add the parent directory to sys.path to allow imports from the main project
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from executor import Executor, PRIORITY_NORMAL, PRIORITY_HIGH, PRIORITY_LOW, PRIORITY_URGENT, \
                     CAT_COMMAND_EXECUTION, CAT_CODE_GENERATION_TOOL, CAT_CODE_GENERATION_SNIPPET, \
                     CAT_INFORMATION_GATHERING, CAT_USE_EXISTING_TOOL, CAT_REFINEMENT, CAT_CORE_FILE_UPDATE, \
                     STATUS_PENDING, STATUS_DECOMPOSED, STATUS_APPROVED, STATUS_COMPLETED, \
                     STATUS_EXECUTED_WITH_ERRORS, STATUS_BUILD_FAILED, STATUS_AWAITING_CORRECTION, \
                     STATUS_FAILED_MAX_RETRIES, STATUS_FAILED_UNCLEAR, \
                     SOURCE_USER, SOURCE_DECOMPOSITION, SOURCE_SUGGESTION, SOURCE_SELF_CORRECTION, SOURCE_AUTO_REFINEMENT


class TestExecutorCoreHelpers(unittest.TestCase):
    def setUp(self):
        # Basic config for tests, can be expanded if needed by more tests
        self.mock_config = {
            "meta_dir": "meta_executor_tests", # Use a test-specific meta directory
            "goals_file": os.path.join("meta_executor_tests", "goals.json"),
            "tool_registry_file": os.path.join("meta_executor_tests", "tool_registry.json"),
            "executor_config": {
                "decomposition_threshold": 70,
                "max_self_correction_attempts": 2
            }
        }
        # Mock logger, can be replaced with a more sophisticated mock if needed
        self.mock_logger = lambda level, msg: print(f"TEST_LOG ({level}): {msg}")

        # Mock query_llm, to be customized per test or group of tests
        self.mock_query_llm = lambda prompt_text, system_prompt_override=None, raw_output=False, timeout=180: "[Mock LLM Response]"

        # Mock other dependencies as needed, for now, basic placeholders
        class MockMissionManager:
            def load_mission(self): return {"current_focus_areas": ["Test Focus"]}

        class MockNotifier:
            def log_update(self, summary, goals, approved_by): self.mock_logger("NOTIFY", f"Update by {approved_by}: {summary}")

        class MockToolBuilder:
            def build_tool(self, description, tool_name_suggestion, thread_id, goals_list_ref, current_goal_id):
                self.mock_logger("TOOL_BUILDER", f"Build tool: {tool_name_suggestion}")
                return f"tools/{tool_name_suggestion}.py" # Dummy path

        class MockToolRunner:
            def run_tool_safely(self, tool_path, tool_args=None):
                self.mock_logger("TOOL_RUNNER", f"Run tool: {tool_path} with args {tool_args}")
                return {"status": "success", "output": "Mock tool output"}

        self.executor = Executor(
            config=self.mock_config,
            logger_func=self.mock_logger,
            query_llm_func=self.mock_query_llm,
            mission_manager_instance=MockMissionManager(),
            notifier_instance=MockNotifier(),
            tool_builder_instance=MockToolBuilder(),
            tool_runner_instance=MockToolRunner()
        )

        # Ensure the test meta directory exists
        os.makedirs(self.mock_config["meta_dir"], exist_ok=True)

    def tearDown(self):
        # Clean up the test meta directory and its contents
        # This is important to ensure tests are isolated and repeatable
        test_meta_dir = self.mock_config["meta_dir"]
        if os.path.exists(test_meta_dir):
            for item in os.listdir(test_meta_dir):
                item_path = os.path.join(test_meta_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    # Handle subdirectories if necessary, for now just removing files
                    pass
            # Potentially remove the directory itself, or leave it empty
            # For now, just cleaning contents. If tools are created in subdirs, adapt this.
            # os.rmdir(test_meta_dir) # Careful with this if tests run in parallel or if it contains unexpected items

    def test_initialization(self):
        """Test that the Executor initializes without errors."""
        self.assertIsNotNone(self.executor)
        self.assertEqual(self.executor.logger, self.mock_logger)
        self.assertEqual(self.executor.config, self.mock_config)

    def test_categorize_subtask(self):
        """Test the _categorize_subtask method."""
        test_cases = [
            ("Update core file ai_core.py to fix bug", CAT_CORE_FILE_UPDATE),
            ("Execute python code: print('hello')", CAT_CODE_GENERATION_SNIPPET),
            ("Run python snippet: for i in range(5): print(i)", CAT_CODE_GENERATION_SNIPPET),
            ("Install beautifulsoup4", CAT_COMMAND_EXECUTION),
            ("pip install requests", CAT_COMMAND_EXECUTION),
            ("Refine tool script_analyzer.py", CAT_REFINEMENT),
            ("Modify script tools/parser.py", CAT_REFINEMENT),
            ("Define the API for weather service", CAT_INFORMATION_GATHERING),
            ("Research best practices for Python async", CAT_INFORMATION_GATHERING),
            ("What is the airspeed velocity of an unladen swallow?", CAT_INFORMATION_GATHERING),
            ("Create a new tool to summarize text", CAT_CODE_GENERATION_TOOL), # Default
            ("Build a script to parse logs", CAT_CODE_GENERATION_TOOL), # Default
        ]
        for description, expected_category in test_cases:
            with self.subTest(description=description):
                category = self.executor._categorize_subtask(description)
                self.assertEqual(category, expected_category)

    def test_determine_failure_category(self):
        """Test the _determine_failure_category method."""
        test_cases = [
            ("SyntaxError: invalid syntax", None, "SyntaxError"),
            ("ModuleNotFoundError: No module named 'requests'", None, "ModuleNotFoundError"),
            ("FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'", None, "FileNotFoundError"),
            ("AttributeError: 'NoneType' object has no attribute 'foo'", None, "AttributeError"),
            ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", None, "TypeError"),
            ("ValueError: invalid literal for int() with base 10: 'abc'", None, "ValueError"),
            ("IndexError: list index out of range", None, "IndexError"),
            ("KeyError: 'missing_key'", None, "KeyError"),
            ("NameError: name 'my_var' is not defined", None, "NameError"),
            ("ImportError: cannot import name 'specific_func' from 'my_module'", None, "ImportError"),
            ("LookupError: \n**********************************************************************\n  Resource punkt not found.\n  Please use the NLTK Downloader to obtain the resource:\n\n  >>> import nltk\n  >>> nltk.download('punkt')\n  \n**********************************************************************", None, "NLTKResourceMissing"),
            ("script.py: error: the following arguments are required: -i/--input", "tools/script.py", "ArgumentError"),
            ("SystemExit: 2", "tools/script.py", "UnknownRuntimeError"), # Too generic without more context in error
            ("Tool execution failed: No main() function found in tool tools/script.py", "tools/script.py", "NoMainFunction"),
            ("Process timed out after 30 seconds", None, "TimeoutError"),
            ("Tool execution failed with an error.", "tools/script.py", "ToolExecutionFailure"),
            ("Compilation failed for new_tool.py", None, "BuildCompilationError"),
            ("Core update staging failed for core_logic.py", None, "CoreUpdateStageFailure"),
            ("Core update apply failed for core_logic.py", None, "CoreUpdateApplyFailure"),
            ("Some other generic runtime error", None, "UnknownRuntimeError"),
        ]
        for error_message, tool_file_path, expected_category in test_cases:
            with self.subTest(error_message=error_message):
                category = self.executor._determine_failure_category(error_message, tool_file_path)
                self.assertEqual(category, expected_category)

    def test_update_goal_status_and_history(self):
        """Test the _update_goal_status_and_history method."""
        goal = {
            "goal_id": "test_goal_123",
            "status": STATUS_PENDING,
            "history": []
        }
        new_status = STATUS_APPROVED
        message = "Goal approved for processing"
        details = {"approver": "test_system"}

        self.executor._update_goal_status_and_history(goal, new_status, message, details)

        self.assertEqual(goal["status"], new_status)
        self.assertEqual(len(goal["history"]), 1)
        history_entry = goal["history"][0]
        self.assertEqual(history_entry["status"], new_status)
        self.assertEqual(history_entry["message"], message)
        self.assertEqual(history_entry["details"], details)
        self.assertIn("timestamp", history_entry)

        # Test adding another history entry
        self.executor._update_goal_status_and_history(goal, STATUS_COMPLETED, "Execution finished.")
        self.assertEqual(goal["status"], STATUS_COMPLETED)
        self.assertEqual(len(goal["history"]), 2)
        self.assertEqual(goal["history"][1]["status"], STATUS_COMPLETED)

    def test_register_goal_in_memory(self):
        """Test the register_goal_in_memory method."""
        goals_list = []
        goal1 = {"goal_id": "gid1", "goal": "Test Goal 1", "parent_goal_id": None, "status": STATUS_PENDING, "source": SOURCE_USER, "thread_id": "tid1"}
        goal2 = {"goal_id": "gid2", "goal": "Test Goal 2", "parent_goal_id": None, "status": STATUS_PENDING, "source": SOURCE_USER, "thread_id": "tid2"}

        # Add new goal
        self.assertTrue(self.executor.register_goal_in_memory(goal1, goals_list))
        self.assertEqual(len(goals_list), 1)
        self.assertIn(goal1, goals_list)

        # Add another new goal
        self.assertTrue(self.executor.register_goal_in_memory(goal2, goals_list))
        self.assertEqual(len(goals_list), 2)

        # Try to add goal1 again (duplicate ID)
        self.assertFalse(self.executor.register_goal_in_memory(goal1.copy(), goals_list))
        self.assertEqual(len(goals_list), 2)

        # Try to add functionally duplicate goal (same core fields, different ID)
        goal1_functional_dup = {"goal_id": "gid3", "goal": "Test Goal 1", "parent_goal_id": None, "status": STATUS_PENDING, "source": SOURCE_USER, "thread_id": "tid1"}
        self.assertFalse(self.executor.register_goal_in_memory(goal1_functional_dup, goals_list))
        self.assertEqual(len(goals_list), 2)

        # Add a goal that is similar but different status (should be allowed)
        goal1_completed = {"goal_id": "gid4", "goal": "Test Goal 1", "parent_goal_id": None, "status": STATUS_COMPLETED, "source": SOURCE_USER, "thread_id": "tid1"}
        self.assertTrue(self.executor.register_goal_in_memory(goal1_completed, goals_list))
        self.assertEqual(len(goals_list), 3)

        # Add a goal that is similar but different source (should be allowed)
        goal1_decomp = {"goal_id": "gid5", "goal": "Test Goal 1", "parent_goal_id": None, "status": STATUS_PENDING, "source": SOURCE_DECOMPOSITION, "thread_id": "tid1"}
        self.assertTrue(self.executor.register_goal_in_memory(goal1_decomp, goals_list))
        self.assertEqual(len(goals_list), 4)

    def test_add_goal_simple_parent(self):
        """Test adding a simple parent goal."""
        goals_list = []
        description = "This is a new parent goal"
        source = SOURCE_USER
        priority = PRIORITY_HIGH

        success, new_goal_id = self.executor.add_goal(
            goals_list_ref=goals_list,
            description=description,
            source=source,
            priority=priority
        )

        self.assertTrue(success)
        self.assertIsNotNone(new_goal_id)
        self.assertEqual(len(goals_list), 1)

        added_goal = goals_list[0]
        self.assertEqual(added_goal["goal_id"], new_goal_id)
        self.assertEqual(added_goal["goal"], description)
        self.assertEqual(added_goal["source"], source)
        self.assertEqual(added_goal["priority"], priority)
        self.assertFalse(added_goal["is_subtask"])
        self.assertIsNone(added_goal["parent_goal_id"])
        self.assertIsNotNone(added_goal["thread_id"]) # Should be auto-generated
        self.assertEqual(len(added_goal["history"]), 1)
        self.assertEqual(added_goal["history"][0]["status"], STATUS_PENDING)

    def test_add_goal_subtask_with_parent_object(self):
        """Test adding a subtask with a parent goal object."""
        goals_list = []
        parent_description = "Parent goal for subtask test"
        parent_thread_id = "thread_for_parent_abc"
        parent_goal = {
            "goal_id": "parent_gid_123",
            "goal": parent_description,
            "thread_id": parent_thread_id,
            "status": STATUS_PENDING
            # ... other parent fields
        }
        # Add parent to list first, as if it exists
        self.executor.register_goal_in_memory(parent_goal, goals_list)

        subtask_description = "This is a subtask"
        success, new_subtask_id = self.executor.add_goal(
            goals_list_ref=goals_list,
            description=subtask_description,
            source=SOURCE_DECOMPOSITION,
            is_subtask=True,
            parent_goal_obj=parent_goal,
            priority=PRIORITY_NORMAL
        )

        self.assertTrue(success)
        self.assertIsNotNone(new_subtask_id)
        self.assertEqual(len(goals_list), 2) # Parent + subtask

        added_subtask = next(g for g in goals_list if g["goal_id"] == new_subtask_id)
        self.assertEqual(added_subtask["goal"], subtask_description)
        self.assertTrue(added_subtask["is_subtask"])
        self.assertEqual(added_subtask["parent_goal_id"], parent_goal["goal_id"])
        self.assertEqual(added_subtask["parent"], parent_goal["goal"])
        self.assertEqual(added_subtask["thread_id"], parent_thread_id) # Inherited from parent
        self.assertEqual(added_subtask["source"], SOURCE_DECOMPOSITION)

    def test_add_goal_with_specific_thread_id(self):
        """Test adding a goal with a specified thread_id."""
        goals_list = []
        description = "Goal for specific thread"
        specific_thread_id = "my_custom_thread_123"

        success, new_goal_id = self.executor.add_goal(
            goals_list_ref=goals_list,
            description=description,
            thread_id_param=specific_thread_id
        )
        self.assertTrue(success)
        self.assertEqual(goals_list[0]["thread_id"], specific_thread_id)

    def test_add_goal_subtask_inherits_thread_from_failed_related(self):
        """Test subtask inherits thread_id from related_to_failed_goal_obj if parent_goal_obj is None."""
        goals_list = []
        failed_goal_thread_id = "thread_from_failed_xyz"
        failed_goal = {
            "goal_id": "failed_gid_789",
            "goal": "A goal that failed",
            "thread_id": failed_goal_thread_id,
            "status": STATUS_EXECUTED_WITH_ERRORS
        }
        self.executor.register_goal_in_memory(failed_goal, goals_list)

        corrective_description = "Corrective action for failed goal"
        success, new_corrective_id = self.executor.add_goal(
            goals_list_ref=goals_list,
            description=corrective_description,
            source=SOURCE_SELF_CORRECTION,
            is_subtask=True, # Typically corrective goals might be subtasks of the original's parent or a new thread
            parent_goal_obj=None, # Explicitly no direct parent object passed
            related_to_failed_goal_obj=failed_goal,
            priority=PRIORITY_HIGH
        )
        self.assertTrue(success)
        corrective_goal = next(g for g in goals_list if g["goal_id"] == new_corrective_id)
        self.assertEqual(corrective_goal["thread_id"], failed_goal_thread_id)
        self.assertEqual(corrective_goal["related_to_failed_goal_id"], failed_goal["goal_id"])


    def test_add_goal_duplicate_functional_is_rejected(self):
        """Test that adding a functionally duplicate goal is rejected by add_goal (via register_goal_in_memory)."""
        goals_list = []
        description = "Unique goal for duplication test"
        source = SOURCE_USER
        thread_id = "dup_thread_id_1"

        # Add the first goal
        self.executor.add_goal(goals_list, description, source=source, thread_id_param=thread_id)
        self.assertEqual(len(goals_list), 1)

        # Try to add a functionally identical goal (same desc, source, status=pending, thread_id)
        success_dup, new_goal_id_dup = self.executor.add_goal(
            goals_list, description, source=source, thread_id_param=thread_id
        )
        self.assertFalse(success_dup)
        self.assertIsNone(new_goal_id_dup)
        self.assertEqual(len(goals_list), 1) # Should not have been added

    def test_decompose_goal_successful(self):
        """Test decompose_goal with a successful LLM response."""
        parent_goal = {"goal_id": "pgid1", "goal": "Plan a complex project", "thread_id": "t123"}
        goals_list = [parent_goal]

        expected_subtasks = ["Define project scope", "Identify key milestones", "Allocate resources"]
        # Configure the mock_query_llm for this specific test
        self.executor.query_llm = unittest.mock.Mock(return_value=json.dumps(expected_subtasks))

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)

        self.assertEqual(subtask_descriptions, expected_subtasks)
        self.executor.query_llm.assert_called_once()
        call_args = self.executor.query_llm.call_args
        self.assertIn(parent_goal["goal"], call_args[1]["prompt_text"]) # Check if goal text was in prompt
        self.assertIn("JSON Array of Subtask Descriptions:", call_args[1]["prompt_text"])

    def test_decompose_goal_llm_error(self):
        """Test decompose_goal when the LLM returns an error string."""
        parent_goal = {"goal_id": "pgid2", "goal": "Plan another project", "thread_id": "t456"}
        goals_list = [parent_goal]

        self.executor.query_llm = unittest.mock.Mock(return_value="[Error: LLM timeout]")

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)

        self.assertEqual(subtask_descriptions, [])
        self.executor.query_llm.assert_called_once()

    def test_decompose_goal_llm_malformed_json_direct(self):
        """Test decompose_goal with LLM response that is not valid JSON."""
        parent_goal = {"goal_id": "pgid3", "goal": "Project with bad LLM response", "thread_id": "t789"}
        goals_list = [parent_goal]

        self.executor.query_llm = unittest.mock.Mock(return_value="This is not JSON, just plain text.")

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)
        self.assertEqual(subtask_descriptions, [])

    def test_decompose_goal_llm_malformed_json_with_regex_fallback(self):
        """Test decompose_goal with LLM response where JSON is embedded and regex can find it."""
        parent_goal = {"goal_id": "pgid4", "goal": "Project with messy LLM response", "thread_id": "t101"}
        goals_list = [parent_goal]

        expected_subtasks = ["Subtask Alpha", "Subtask Beta"]
        messy_response = f"Okay, here are the tasks you requested: {json.dumps(expected_subtasks)}. I hope this helps!"
        self.executor.query_llm = unittest.mock.Mock(return_value=messy_response)

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)
        self.assertEqual(subtask_descriptions, expected_subtasks)

    def test_decompose_goal_llm_malformed_json_no_regex_match(self):
        """Test decompose_goal with LLM response where JSON is malformed and regex doesn't match."""
        parent_goal = {"goal_id": "pgid5", "goal": "Project with very messy LLM response", "thread_id": "t112"}
        goals_list = [parent_goal]

        messy_response = "Here is what I think: [\"Task 1\", \"Task 2\" but this part is broken."
        self.executor.query_llm = unittest.mock.Mock(return_value=messy_response)

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)
        self.assertEqual(subtask_descriptions, [])

    def test_decompose_goal_empty_subtasks_from_llm(self):
        """Test decompose_goal when LLM returns an empty list of subtasks."""
        parent_goal = {"goal_id": "pgid6", "goal": "A simple project", "thread_id": "t113"}
        goals_list = [parent_goal]

        self.executor.query_llm = unittest.mock.Mock(return_value=json.dumps([])) # Empty list

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)
        self.assertEqual(subtask_descriptions, [])

    def test_decompose_goal_subtasks_with_empty_strings(self):
        """Test decompose_goal filters out empty or whitespace-only subtask strings."""
        parent_goal = {"goal_id": "pgid7", "goal": "Project with some empty subtasks", "thread_id": "t114"}
        goals_list = [parent_goal]

        llm_response = json.dumps(["Valid Task 1", "", "   ", "Valid Task 2", "  "])
        self.executor.query_llm = unittest.mock.Mock(return_value=llm_response)

        subtask_descriptions = self.executor.decompose_goal(parent_goal, goals_list)
        self.assertEqual(subtask_descriptions, ["Valid Task 1", "Valid Task 2"])


# Test class for File I/O methods
class TestExecutorFileIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = "meta_executor_tests_file_io"
        os.makedirs(self.test_dir, exist_ok=True)

        self.mock_config = {
            "meta_dir": self.test_dir,
            "goals_file": os.path.join(self.test_dir, "goals.json"),
            "tool_registry_file": os.path.join(self.test_dir, "tool_registry.json"),
        }
        self.mock_logger = lambda level, msg: None # Simple mock logger for these tests

        # We don't need full mocks for other components for pure file I/O testing
        self.executor = Executor(
            config=self.mock_config,
            logger_func=self.mock_logger,
            query_llm_func=lambda p, system_prompt_override=None, raw_output=False, timeout=180: "mock llm", # Not used by these methods
            mission_manager_instance=None, # Not used
            notifier_instance=None, # Not used
            tool_builder_instance=None, # Not used
            tool_runner_instance=None # Not used
        )

    def tearDown(self):
        # Clean up the test directory and its contents
        if os.path.exists(self.test_dir):
            for item in os.listdir(self.test_dir):
                item_path = os.path.join(self.test_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(self.test_dir)

    # --- Tests for load_goals ---
    def test_load_goals_file_not_found(self):
        """Test load_goals when the goals file does not exist."""
        # Ensure file doesn't exist initially by virtue of setUp/tearDown or specific delete
        if os.path.exists(self.executor.goal_file_path):
            os.remove(self.executor.goal_file_path)

        goals = self.executor.load_goals()
        self.assertEqual(goals, [])
        # Check if an empty file was created
        self.assertTrue(os.path.exists(self.executor.goal_file_path))
        with open(self.executor.goal_file_path, "r") as f:
            self.assertEqual(f.read().strip(), "[]")


    def test_load_goals_empty_file(self):
        """Test load_goals with an empty goals file."""
        with open(self.executor.goal_file_path, "w") as f:
            f.write("") # Empty file
        goals = self.executor.load_goals()
        self.assertEqual(goals, [])

    def test_load_goals_empty_json_array(self):
        """Test load_goals with an empty JSON array in the file."""
        with open(self.executor.goal_file_path, "w") as f:
            json.dump([], f)
        goals = self.executor.load_goals()
        self.assertEqual(goals, [])

    def test_load_goals_corrupted_json(self):
        """Test load_goals with a corrupted JSON file."""
        with open(self.executor.goal_file_path, "w") as f:
            f.write("[{'id': 1, 'goal': 'test'") # Corrupted JSON
        goals = self.executor.load_goals()
        self.assertEqual(goals, []) # Should return default (empty list)

    def test_load_goals_valid_data_with_migration(self):
        """Test load_goals with valid data that needs migration/defaulting."""
        raw_goal_data = [{"goal": "Test migration goal"}] # Missing many fields
        with open(self.executor.goal_file_path, "w") as f:
            json.dump(raw_goal_data, f)

        goals = self.executor.load_goals()
        self.assertEqual(len(goals), 1)
        goal = goals[0]
        self.assertEqual(goal["goal"], "Test migration goal")
        self.assertIn("goal_id", goal)
        self.assertIn("thread_id", goal)
        self.assertEqual(goal["priority"], PRIORITY_NORMAL)
        self.assertIn("created_at", goal)
        self.assertIsInstance(goal["history"], list)
        self.assertTrue(len(goal["history"]) > 0)
        self.assertIn("status", goal["history"][0])

    # --- Tests for save_goals ---
    @unittest.mock.patch("builtins.open", new_callable=unittest.mock.mock_open)
    @unittest.mock.patch("os.makedirs") # Mock makedirs as it's called by _ensure_meta_dir
    def test_save_goals(self, mock_makedirs, mock_file_open):
        """Test save_goals successfully writes data."""
        goals_data = [{"goal_id": "g1", "goal": "Save this goal"}]
        self.executor.save_goals(goals_data)

        mock_makedirs.assert_called_once_with(self.test_dir, exist_ok=True)
        mock_file_open.assert_called_once_with(self.executor.goal_file_path, "w", encoding="utf-8")

        # Get the file handle from the mock
        handle = mock_file_open()
        # Check what was written. json.dump calls handle.write multiple times for formatted JSON.
        # We can inspect the args of all calls to write.
        written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)

        self.assertEqual(json.loads(written_content), goals_data)

    @unittest.mock.patch("builtins.open", side_effect=IOError("Test save error"))
    @unittest.mock.patch("os.makedirs")
    def test_save_goals_io_error(self, mock_makedirs, mock_file_open_error):
        """Test save_goals handles IOError during write."""
        # Logger will be called with ERROR. If we had a more complex mock logger, we could assert that.
        # For now, just ensure it doesn't crash.
        goals_data = [{"goal_id": "g1", "goal": "Try to save this"}]
        with unittest.mock.patch.object(self.executor, 'logger') as mock_logger_method:
            self.executor.save_goals(goals_data)
            mock_logger_method.assert_any_call("ERROR", unittest.mock.ANY) # Check if logger was called with ERROR

    # --- Tests for get_structured_tool_registry ---
    def test_get_tool_registry_file_not_found(self):
        """Test get_structured_tool_registry when the file does not exist."""
        if os.path.exists(self.executor.tool_registry_file_path):
            os.remove(self.executor.tool_registry_file_path)
        registry = self.executor.get_structured_tool_registry()
        self.assertEqual(registry, [])
        self.assertTrue(os.path.exists(self.executor.tool_registry_file_path))
        with open(self.executor.tool_registry_file_path, "r") as f:
            self.assertEqual(f.read().strip(), "[]")

    def test_get_tool_registry_corrupted_json(self):
        """Test get_structured_tool_registry with corrupted JSON."""
        with open(self.executor.tool_registry_file_path, "w") as f:
            f.write("{'name': 'tool1'") # Invalid JSON
        registry = self.executor.get_structured_tool_registry()
        self.assertEqual(registry, [])

    def test_get_tool_registry_not_a_list(self):
        """Test get_structured_tool_registry when content is not a list."""
        with open(self.executor.tool_registry_file_path, "w") as f:
            json.dump({"not_a_list": True}, f)
        registry = self.executor.get_structured_tool_registry()
        self.assertEqual(registry, [])

    def test_get_tool_registry_valid_with_migration(self):
        """Test get_structured_tool_registry with data needing migration."""
        raw_tool_data = [{"name": "mig_tool", "module_path": "tools/mig_tool.py"}]
        with open(self.executor.tool_registry_file_path, "w") as f:
            json.dump(raw_tool_data, f)

        registry = self.executor.get_structured_tool_registry()
        self.assertEqual(len(registry), 1)
        tool = registry[0]
        self.assertEqual(tool["name"], "mig_tool")
        self.assertIsNone(tool["last_run_time"])
        self.assertEqual(tool["success_count"], 0)
        self.assertIn("last_updated", tool)

    # --- Tests for save_structured_tool_registry ---
    @unittest.mock.patch("builtins.open", new_callable=unittest.mock.mock_open)
    @unittest.mock.patch("os.makedirs")
    def test_save_tool_registry(self, mock_makedirs, mock_file_open):
        """Test save_structured_tool_registry successfully writes data."""
        registry_data = [{"name": "tool_to_save", "module_path": "path/to/tool.py"}]
        self.executor.save_structured_tool_registry(registry_data)

        mock_makedirs.assert_called_once_with(self.test_dir, exist_ok=True)
        mock_file_open.assert_called_once_with(self.executor.tool_registry_file_path, "w", encoding="utf-8")

        handle = mock_file_open()
        written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
        self.assertEqual(json.loads(written_content), registry_data)

if __name__ == '__main__':
    unittest.main()
