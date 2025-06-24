import unittest
import os
import sys
import json
from unittest.mock import MagicMock, patch

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from planner_module import Planner
from ai_core import AICore # For mocking AICore instance

# Mock other dependencies of Planner if they are complex to set up
class MockPromptManager:
    def __init__(self, prompts_dir="prompts_planner_test"): # Use a test-specific dir
        self.prompts_dir = prompts_dir
        os.makedirs(self.prompts_dir, exist_ok=True)
        self._ensure_dummy_template("planner_decide_action.txt", "Base Action Prompt. History: {{conversation_history_str}} Tools: {{available_tools_summary}} KG: {{relevant_learnings_str}}")
        self._ensure_dummy_template("planner_manage_suggestions_instructions.txt", "Manage suggestions: {{pending_suggestions_summary}}")

    def _ensure_dummy_template(self, filename, content):
        fpath = os.path.join(self.prompts_dir, filename)
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

    def render_prompt_with_dynamic_content(self, template_name, context_dict):
        fpath = os.path.join(self.prompts_dir, f"{template_name}.txt") # Assuming .txt extension
        if not os.path.exists(fpath):
            return f"Error: Template {template_name} not found."
        with open(fpath, "r", encoding="utf-8") as f:
            template_str = f.read()

        # Basic placeholder replacement for testing
        for key, value in context_dict.items():
            if isinstance(value, list): # Handle list of goals/suggestions if template expects strings
                str_value = "\n".join([str(v) for v in value]) if value else "None"
            else:
                str_value = str(value)
            template_str = template_str.replace(f"{{{{{key}}}}}", str_value)

        # Add a placeholder for KG if not in context, so prompt construction doesn't fail
        if "{{relevant_learnings_str}}" in template_str: # Check if placeholder exists
             template_str = template_str.replace("{{relevant_learnings_str}}", context_dict.get("relevant_learnings_str", "No specific learnings provided for this context."))

        return template_str

class MockGoalMonitor:
    def get_active_goals(self): return []
    def get_pending_goals(self): return []

class MockMissionManager:
    def get_mission(self): return {"description": "Test Mission"}
    def get_mission_statement_for_prompt(self): return "Test Mission Statement"

class MockSuggestionEngine:
    def load_suggestions(self, load_all_history=False): return []
    def get_pending_suggestions_summary_for_prompt(self, max_suggestions=3): return "No pending suggestions."


class TestPlannerKnowledgeGraphIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "planner_config": {
                "planner_context_conversation_history_length": 2,
                "planner_context_max_tools": 2,
            },
            "tool_registry_file": "dummy_tools.json" # Will be mocked or non-existent
        }
        # Create a dummy tool registry file if Planner tries to load it and it's not mocked away
        self.dummy_tool_registry_path = self.mock_config["tool_registry_file"]
        with open(self.dummy_tool_registry_path, "w") as f:
            json.dump([{"name": "sample_tool", "description": "A sample tool."}], f)

        self.mock_logger = MagicMock()
        self.mock_query_llm = MagicMock(return_value=json.dumps({"next_action": "respond_to_user", "action_details": {"response_text": "OK"}}))

        self.mock_prompt_manager = MockPromptManager()
        self.mock_goal_monitor = MockGoalMonitor()
        self.mock_mission_manager = MockMissionManager()
        self.mock_suggestion_engine = MockSuggestionEngine()

        # Mock AICore instance and its relevant KG methods
        self.mock_aicore_instance = MagicMock(spec=AICore)
        self.mock_aicore_instance._extract_concepts_from_text = MagicMock(return_value=["keyword1", "keyword2"])
        self.mock_aicore_instance.get_relevant_lessons = MagicMock(return_value=[]) # Default to no lessons

        self.planner = Planner(
            query_llm_func=self.mock_query_llm,
            prompt_manager=self.mock_prompt_manager,
            goal_monitor=self.mock_goal_monitor,
            mission_manager=self.mock_mission_manager,
            suggestion_engine=self.mock_suggestion_engine,
            config=self.mock_config,
            logger=self.mock_logger,
            tool_registry_or_path=self.dummy_tool_registry_path, # Use the dummy path
            aicore_instance=self.mock_aicore_instance
        )

    def tearDown(self):
        if os.path.exists(self.dummy_tool_registry_path):
            os.remove(self.dummy_tool_registry_path)

        # Clean up dummy prompts dir created by MockPromptManager
        prompts_test_dir = getattr(self.mock_prompt_manager, 'prompts_dir', 'prompts_planner_test')
        if os.path.exists(prompts_test_dir):
            for item in os.listdir(prompts_test_dir):
                os.remove(os.path.join(prompts_test_dir, item))
            os.rmdir(prompts_test_dir)


    def test_construct_prompt_no_lessons_from_kg(self):
        """Test _construct_prompt when no relevant lessons are found in KG."""
        self.mock_aicore_instance.get_relevant_lessons.return_value = [] # Ensure no lessons

        context = self.planner._gather_context([], {"current_time": "now"})
        # Add conversation history for concept extraction
        context["conversation_history"] = [{"role": "user", "content": "Tell me about apples"}]

        prompt = self.planner._construct_prompt(context)

        self.mock_aicore_instance._extract_concepts_from_text.assert_called_once_with("Tell me about apples", max_concepts=3)
        self.mock_aicore_instance.get_relevant_lessons.assert_called_once_with(["keyword1", "keyword2"], top_n=2)
        self.assertNotIn("Relevant Learnings from Past Experience", prompt)
        # Check if the placeholder text (or lack of KG section) is handled gracefully by the template
        self.assertIn("No specific learnings provided for this context.", prompt)


    def test_construct_prompt_with_lessons_from_kg(self):
        """Test _construct_prompt when relevant lessons are found and included."""
        lesson1 = {"lesson_learned": "Lesson about apples.", "source_goal_ids": ["g1"], "frequency": 1}
        lesson2 = {"lesson_learned": "Another fruit lesson.", "source_goal_ids": ["g2"], "frequency": 2}
        self.mock_aicore_instance.get_relevant_lessons.return_value = [lesson1, lesson2]
         # Mock _extract_concepts_from_text to return specific keywords for this test
        self.mock_aicore_instance._extract_concepts_from_text.return_value = ["apples", "fruit"]

        context = self.planner._gather_context([], {"current_time": "now"})
        context["conversation_history"] = [{"role": "user", "content": "Tell me about apples and other fruit"}]

        prompt = self.planner._construct_prompt(context)

        self.mock_aicore_instance._extract_concepts_from_text.assert_called_with("Tell me about apples and other fruit", max_concepts=3)
        self.mock_aicore_instance.get_relevant_lessons.assert_called_with(["apples", "fruit"], top_n=2)

        self.assertIn("### Relevant Learnings from Past Experience (Knowledge Graph):", prompt)
        self.assertIn("- Lesson: Lesson about apples.", prompt)
        self.assertIn("(Source Goals: 1, Freq: 1)", prompt)
        self.assertIn("- Lesson: Another fruit lesson.", prompt)
        self.assertIn("(Source Goals: 1, Freq: 2)", prompt) # len(source_goal_ids) is 1 for ["g2"]

if __name__ == '__main__':
    unittest.main()
