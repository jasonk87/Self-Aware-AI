import unittest
import os
import sys
import json
import uuid
from unittest.mock import patch, mock_open

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ai_core import AICore, MODEL_NAME_AICORE, OLLAMA_URL_AICORE # Import necessary items

class TestAICoreKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.test_meta_dir = "meta_aicore_kg_tests"
        os.makedirs(self.test_meta_dir, exist_ok=True)

        self.config_data = {
            "llm_model_name": "test_model",
            "ollama_api_url": "http://localhost:1111",
            "meta_dir": self.test_meta_dir,
            "memory_persistence": True, # Enable persistence for testing load/save
            # Add other minimal required config keys if AICore init needs them
        }
        self.config_file_path = os.path.join(self.test_meta_dir, "config.json")
        with open(self.config_file_path, "w") as f:
            json.dump(self.config_data, f)

        # Mock logger for AICore
        self.mock_logger = unittest.mock.Mock()

        # Mock components that AICore tries to initialize, to avoid full init
        # We are only interested in KG methods here.
        patcher_notifier = patch('ai_core.Notifier')
        patcher_tb = patch('ai_core.ToolBuilder')
        patcher_tr = patch('ai_core.ToolRunner')
        patcher_pm = patch('ai_core.PromptManager')
        patcher_mm = patch('ai_core.MissionManager')
        patcher_se = patch('ai_core.SuggestionEngine')
        patcher_gm = patch('ai_core.GoalMonitor')
        patcher_planner = patch('ai_core.Planner')
        patcher_evaluator = patch('ai_core.Evaluator')
        patcher_executor = patch('ai_core.Executor')
        patcher_gw = patch('ai_core.GoalWorker')

        self.addCleanup(patcher_notifier.stop)
        self.addCleanup(patcher_tb.stop)
        self.addCleanup(patcher_tr.stop)
        self.addCleanup(patcher_pm.stop)
        self.addCleanup(patcher_mm.stop)
        self.addCleanup(patcher_se.stop)
        self.addCleanup(patcher_gm.stop)
        self.addCleanup(patcher_planner.stop)
        self.addCleanup(patcher_evaluator.stop)
        self.addCleanup(patcher_executor.stop)
        self.addCleanup(patcher_gw.stop)

        patcher_notifier.start()
        patcher_tb.start()
        patcher_tr.start()
        patcher_pm.start()
        patcher_mm.start()
        patcher_se.start()
        patcher_gm.start()
        patcher_planner.start()
        patcher_evaluator.start()
        patcher_executor.start()
        patcher_gw.start()

        # Patch query_llm_internal as _extract_concepts_from_text might use it if enhanced
        self.mock_query_llm_patch = patch('ai_core.query_llm_internal', return_value="Mock LLM Response")
        self.mock_query_llm_patch.start()
        self.addCleanup(self.mock_query_llm_patch.stop)

        self.ai_core = AICore(config_file=self.config_file_path, logger_func=self.mock_logger)
        # Clear any KG data loaded from a potentially persistent memory.json from previous test runs
        self.ai_core.knowledge_graph = {
            "lessons_by_concept": {}, "lessons_by_failure_category": {},
            "lessons_by_tool_name": {}, "all_lessons": {}
        }


    def tearDown(self):
        memory_file = os.path.join(self.test_meta_dir, "memory.json")
        if os.path.exists(memory_file):
            os.remove(memory_file)
        if os.path.exists(self.config_file_path):
            os.remove(self.config_file_path)
        if os.path.exists(self.test_meta_dir):
            # Remove other files if any created by AICore's _init methods for other components
            for f_name in os.listdir(self.test_meta_dir):
                if os.path.isfile(os.path.join(self.test_meta_dir, f_name)):
                    os.remove(os.path.join(self.test_meta_dir, f_name))
            os.rmdir(self.test_meta_dir)

    def test_extract_concepts_from_text_llm_success(self):
        """Test _extract_concepts_from_text with successful LLM response."""
        test_text = "This tool uses asynchronous requests for fetching web data and parsing HTML."
        expected_concepts = ["asynchronous requests", "web data", "html parsing"]

        # Configure the AICore's query_llm mock for this specific test
        # Ensure the instance's query_llm is the one being mocked, not the global one.
        self.ai_core.query_llm = unittest.mock.Mock(return_value=json.dumps(expected_concepts))

        concepts = self.ai_core._extract_concepts_from_text(test_text, max_concepts=3)
        self.assertEqual(concepts, [c.lower() for c in expected_concepts])
        self.ai_core.query_llm.assert_called_once()
        call_args = self.ai_core.query_llm.call_args
        self.assertIn(test_text, call_args[1]["prompt_text"])
        self.assertTrue(call_args[1]["raw_output"]) # Check if raw_output was True

    def test_extract_concepts_from_text_llm_error_response(self):
        """Test _extract_concepts_from_text when LLM returns an error string."""
        self.ai_core.query_llm = unittest.mock.Mock(return_value="[Error: LLM service unavailable]")
        concepts = self.ai_core._extract_concepts_from_text("Some text here")
        self.assertEqual(concepts, [])

    def test_extract_concepts_from_text_llm_malformed_json(self):
        """Test _extract_concepts_from_text with malformed JSON from LLM."""
        self.ai_core.query_llm = unittest.mock.Mock(return_value="Not a JSON list [\"concept\"")
        concepts = self.ai_core._extract_concepts_from_text("Another piece of text")
        self.assertEqual(concepts, [])

    def test_extract_concepts_from_text_llm_empty_list(self):
        """Test _extract_concepts_from_text when LLM returns an empty JSON list."""
        self.ai_core.query_llm = unittest.mock.Mock(return_value="[]")
        concepts = self.ai_core._extract_concepts_from_text("Text that might yield no concepts")
        self.assertEqual(concepts, [])

    def test_extract_concepts_from_text_empty_input(self):
        """Test _extract_concepts_from_text with empty input string."""
        concepts = self.ai_core._extract_concepts_from_text("")
        self.assertEqual(concepts, [])
        self.ai_core.query_llm.assert_not_called() # LLM should not be called for empty text

    @patch.object(AICore, '_extract_concepts_from_text') # Mock the now LLM-based concept extraction
    def test_add_new_lesson_to_knowledge_graph_with_mocked_concepts(self, mock_extract_concepts):
        """Test add_or_update_lesson_in_knowledge_graph with mocked concept extraction."""
        mock_extract_concepts.return_value = ["mocked_concept1", "mocked_concept2"]

        lesson_data = {
            "lesson_learned": "Network tools require robust timeout handling.",
            "outcome_summary": "Tool A failed due to network timeout.",
            "root_cause_hypothesis": "Missing timeout in requests.get().",
            "source_goal_id": "goal_network_timeout_1",
            "goal_description": "Fetch data from slow API with Tool A", # Used by concepts_text
            "tool_name_context": "ToolA",
            "failure_category_context": "TimeoutError",
        }
        self.ai_core.add_or_update_lesson_in_knowledge_graph(lesson_data)

        self.assertEqual(len(self.ai_core.knowledge_graph["all_lessons"]), 1)
        lesson_id = list(self.ai_core.knowledge_graph["all_lessons"].keys())[0]
        lesson_entry = self.ai_core.knowledge_graph["all_lessons"][lesson_id]

        self.assertEqual(lesson_entry["lesson_learned"], lesson_data["lesson_learned"])
        self.assertIn("goal_network_timeout_1", lesson_entry["source_goal_ids"])
        self.assertEqual(lesson_entry["frequency"], 1)

        # Check that mocked concepts are present, plus the context ones
        self.assertIn("mocked_concept1", lesson_entry["related_concepts"])
        self.assertIn("mocked_concept2", lesson_entry["related_concepts"])
        self.assertIn("ToolA", lesson_entry["related_concepts"])
        self.assertIn("TimeoutError", lesson_entry["related_concepts"])

        mock_extract_concepts.assert_called_once()
        # Verify concepts are used in inverted indexes
        self.assertIn(lesson_id, self.ai_core.knowledge_graph["lessons_by_concept"]["mocked_concept1"])
        self.assertIn(lesson_id, self.ai_core.knowledge_graph["lessons_by_tool_name"]["ToolA"])
        self.assertIn(lesson_id, self.ai_core.knowledge_graph["lessons_by_failure_category"]["TimeoutError"])


    @patch.object(AICore, '_extract_concepts_from_text')
    def test_update_existing_lesson_in_knowledge_graph_with_mocked_concepts(self, mock_extract_concepts):
        lesson_id = str(uuid.uuid4())
        # Define side effects for multiple calls if concept extraction text changes
        mock_extract_concepts.side_effect = [
            ["initial_concept", "csv"], # First call
            ["updated_concept", "csv", "parsing"]  # Second call
        ]

        initial_lesson_data = {
            "lesson_id": lesson_id,
            "lesson_learned": "Initial lesson about parsing.",
            "source_goal_id": "goal_parse_1",
            "goal_description": "Parse complex CSV file",
            "failure_category_context": "ParsingError"
        }
        self.ai_core.add_or_update_lesson_in_knowledge_graph(initial_lesson_data)
        self.assertEqual(self.ai_core.knowledge_graph["all_lessons"][lesson_id]["frequency"], 1)
        self.assertIn("initial_concept", self.ai_core.knowledge_graph["all_lessons"][lesson_id]["related_concepts"])


        update_lesson_data = { # This data is used for context in concept extraction
            "lesson_id": lesson_id,
            "lesson_learned": "Initial lesson about parsing.", # Lesson text itself is part of concept source
            "source_goal_id": "goal_parse_2",
            "goal_description": "Parse another more complex CSV file regarding financial data", # Changed desc
            "failure_category_context": "ParsingError"
        }
        self.ai_core.add_or_update_lesson_in_knowledge_graph(update_lesson_data)

        updated_entry = self.ai_core.knowledge_graph["all_lessons"][lesson_id]
        self.assertEqual(updated_entry["frequency"], 2)
        self.assertIn("goal_parse_1", updated_entry["source_goal_ids"])
        self.assertIn("goal_parse_2", updated_entry["source_goal_ids"])
        # Check that concepts from both calls are present (assuming simple append for now)
        self.assertIn("initial_concept", updated_entry["related_concepts"])
        self.assertIn("updated_concept", updated_entry["related_concepts"])
        self.assertIn("csv", updated_entry["related_concepts"])
        self.assertIn("parsing", updated_entry["related_concepts"])

    @patch.object(AICore, '_extract_concepts_from_text')
    def test_get_relevant_lessons_with_mocked_concepts(self, mock_extract_concepts):
        # Mocking _extract_concepts_from_text for predictable concept association during lesson addition
        def concept_side_effect(*args, **kwargs):
            text_arg = args[0] # The text being passed to _extract_concepts_from_text
            if "SQL queries" in text_arg:
                return ["sql", "injection", "user_auth_module", "input", "login"]
            elif "Network requests" in text_arg:
                return ["network", "requests", "api", "retries", "backoff"]
            return ["generic_concept"]

        mock_extract_concepts.side_effect = concept_side_effect

        lesson1_data = {
            "lesson_learned": "Always validate user input for SQL queries to prevent injection.",
            "outcome_summary": "Security vulnerability found due to unvalidated input.",
            "root_cause_hypothesis": "Direct concatenation of user input into SQL string.",
            "source_goal_id": "goal_sql_1", "goal_description": "Create user login with SQL",
            "failure_category_context": "SecurityVulnerability", "tool_name_context": "user_auth_module"
        }
        self.ai_core.add_or_update_lesson_in_knowledge_graph(lesson1_data)
        lesson1_id = list(self.ai_core.knowledge_graph["all_lessons"].keys())[0]

        lesson2_data = {
            "lesson_learned": "Network requests should have exponential backoff for retries.",
            "outcome_summary": "API call failed repeatedly without backoff.",
            "root_cause_hypothesis": "Simple retry loop without increasing delay.",
            "source_goal_id": "goal_network_2", "goal_description": "Fetch data from external API",
            "failure_category_context": "ApiError", "tool_name_context": "data_fetcher_tool"
        }
        self.ai_core.add_or_update_lesson_in_knowledge_graph(lesson2_data)

        # Test with keyword matching one lesson strongly
        relevant = self.ai_core.get_relevant_lessons(["sql", "injection", "user_auth_module"], top_n=1)
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0]["lesson_id"], lesson1_id)

        # Test with keyword matching another lesson
        relevant = self.ai_core.get_relevant_lessons(["network", "retry", "ApiError"], top_n=1)
        self.assertEqual(len(relevant), 1)
        self.assertTrue(any(l["lesson_learned"] == lesson2_data["lesson_learned"] for l in relevant))

        # Test with keyword that should match both, ensure top_n works
        # Manually boost frequency of lesson1 to test sorting
        self.ai_core.knowledge_graph["all_lessons"][lesson1_id]["frequency"] = 5
        self.ai_core.knowledge_graph["all_lessons"][lesson1_id]["confidence"] = 0.9

        # "input" is a concept in lesson1. Let's make "api" a concept for lesson2
        self.ai_core.knowledge_graph["lessons_by_concept"]["api"] = [
             next(lid for lid, lsn in self.ai_core.knowledge_graph["all_lessons"].items() if lsn["lesson_learned"] == lesson2_data["lesson_learned"])
        ]

        relevant_multi = self.ai_core.get_relevant_lessons(["input", "api", "user"], top_n=2)
        self.assertEqual(len(relevant_multi), 2)
        # Lesson 1 should be higher due to boosted frequency/confidence and more keyword matches
        self.assertEqual(relevant_multi[0]["lesson_id"], lesson1_id)

        # Test with no matching keywords
        relevant_none = self.ai_core.get_relevant_lessons(["non_existent_keyword_blah"], top_n=1)
        self.assertEqual(len(relevant_none), 0)

    def test_knowledge_graph_persistence(self):
        """Test that knowledge graph is saved and loaded correctly."""
        lesson_data = {
            "lesson_learned": "Persistence test lesson.", "source_goal_id": "goal_persist_1",
            "goal_description": "Test KG save and load"
        }
        self.ai_core.add_or_update_lesson_in_knowledge_graph(lesson_data)
        lesson_id = list(self.ai_core.knowledge_graph["all_lessons"].keys())[0]

        # Create a new AICore instance, it should load from the persisted memory.json
        # We need to ensure memory.json is written by the first instance.
        # _save_persistent_memory is called by add_or_update_lesson_in_knowledge_graph

        # To properly test loading, we need to ensure the first AICore's memory is written to disk
        # and then a new AICore instance loads it.
        # The _save_persistent_memory is called within add_or_update_lesson_in_knowledge_graph

        ai_core_new = AICore(config_file=self.config_file_path, logger_func=self.mock_logger)

        self.assertIn(lesson_id, ai_core_new.knowledge_graph["all_lessons"])
        loaded_lesson = ai_core_new.knowledge_graph["all_lessons"][lesson_id]
        self.assertEqual(loaded_lesson["lesson_learned"], "Persistence test lesson.")
        self.assertIn("persistence", loaded_lesson["related_concepts"])


if __name__ == '__main__':
    unittest.main()
