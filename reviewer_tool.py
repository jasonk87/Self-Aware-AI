import requests
import json

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4B"

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

def review_tool_idea(tool_idea: str, tools_list: list, user_goal: str) -> dict:
    prompt = REVIEW_TEMPLATE.format(
        tool_idea=tool_idea,
        tools_list=", ".join(tools_list) if tools_list else "None",
        user_goal=user_goal or "(unspecified)"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    res = requests.post(OLLAMA_URL, json=payload)
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

# Example usage:
if __name__ == "__main__":
    result = review_tool_idea(
        tool_idea="A tool that counts words in a file.",
        tools_list=["summarize_pdf", "read_txt"],
        user_goal="Help me parse documents faster"
    )
    print("REVIEW DECISION:", result['decision'])
    print("REVIEW EXPLANATION:\n", result['explanation'])
