import requests
import json
import os # Added import
from typing import Optional # Added import
import logging # New import

# --- Reviewer Prompt Template ---
REVIEW_TEMPLATE = '''
You are a critical code reviewer. Your job is to assess whether a proposed tool idea is:
- Useful
- Non-redundant
- Implementable with our current system

The tool must not be trivial, redundant, or pointless.

--- TOOL IDEA ---
{tool_idea}

--- CONTEXT ---
Existing tools: {tools_list}
Current user goal: {user_goal}

Your response should be one of the following:
1. APPROVE: If the tool is valid and useful.
2. REJECT: If the tool is redundant, trivial, or unnecessary.
3. MODIFY: If the idea is okay but needs revision.

Also explain why in 2-3 sentences.
'''

def review_tool_idea(
    tool_idea: str,
    tools_list: list,
    user_goal: str,
    ollama_url_override: Optional[str] = None,
    model_name_override: Optional[str] = None
) -> dict:
    # Determine Ollama URL and Model Name
    ollama_url = ollama_url_override
    model_name = model_name_override

    if ollama_url is None or model_name is None:
        config_path = 'config.json'
        default_ollama_url = 'http://localhost:11434/api/generate'
        default_model_name = 'gemma3:4B'

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if ollama_url is None:
                    ollama_url = config.get('ollama_api_url', default_ollama_url)
                if model_name is None:
                    model_name = config.get('llm_model_name', default_model_name)
            except (json.JSONDecodeError, IOError):
                if ollama_url is None:
                    ollama_url = default_ollama_url
                if model_name is None:
                    model_name = default_model_name
        else:
            if ollama_url is None:
                ollama_url = default_ollama_url
            if model_name is None:
                model_name = default_model_name

    prompt = REVIEW_TEMPLATE.format(
        tool_idea=tool_idea,
        tools_list=", ".join(tools_list) if tools_list else "None",
        user_goal=user_goal or "(unspecified)"
    )

    payload = {
        "model": model_name, # Use determined model name
        "prompt": prompt,
        "stream": False
    }
    res = requests.post(ollama_url, json=payload) # Use determined Ollama URL
    reply = res.json().get("response", "").strip()

    decision = "UNKNOWN"
    explanation = ""

    for line in reply.splitlines():
        line = line.strip().upper()
        if line.startswith("APPROVE"):
            decision = "APPROVE"
        elif line.startswith("REJECT"):
            decision = "REJECT"
        elif line.startswith("MODIFY"):
            decision = "MODIFY"

    explanation = reply
    return {
        "decision": decision,
        "explanation": explanation
    }

# from logger_utils import should_log # Added import - REMOVED
logger = logging.getLogger(__name__) # New logger

# Example usage:
if __name__ == "__main__":
    # Attempt to load config.json for URL and model
    config_path = 'config.json'
    ollama_url = 'http://localhost:11434/api/generate' # Default
    model_name = 'gemma3:4B' # Default

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            ollama_url = config.get('ollama_api_url', ollama_url)
            model_name = config.get('llm_model_name', model_name)
        except (json.JSONDecodeError, IOError):
            # Keep defaults if config loading fails
            pass

    result = review_tool_idea(
        tool_idea="A tool that counts words in a file.",
        tools_list=["summarize_pdf", "read_txt"],
        user_goal="Help me parse documents faster",
        ollama_url_override=ollama_url, # Pass loaded or default URL
        model_name_override=model_name # Pass loaded or default model
    )
    logger.info(f"REVIEW DECISION: {result['decision']}")
    logger.info(f"REVIEW EXPLANATION:\n {result['explanation']}")
