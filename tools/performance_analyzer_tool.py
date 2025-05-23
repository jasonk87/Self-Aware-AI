# Self-Aware/tools/performance_analyzer_tool.py
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

# Define paths relative to the root of the Self-Aware project,
# assuming this tool is run from that root by tool_runner.py
META_DIR = "meta"
GOALS_FILE = os.path.join(META_DIR, "goals.json")
TOOL_REGISTRY_FILE = os.path.join(META_DIR, "tool_registry.json")
MISSION_FILE = os.path.join(META_DIR, "mission.json")
EVALUATION_LOG_FILE = os.path.join(META_DIR, "evaluation_log.json") # Optional, but good for cross-referencing

# Fallback logger if ai_core is not available (e.g. when tool is tested standalone)
try:
    from ai_core import log_background_message
except ImportError:
    def log_background_message(level: str, message: str):
        print(f"[{level.upper()}] (performance_analyzer_tool_fallback_log) {message}")

def load_json_data(filepath, default_value=None):
    if default_value is None:
        default_value = []
    if not os.path.exists(filepath):
        log_background_message("WARNING", f"File not found: {filepath}. Returning default.")
        return default_value
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return default_value
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        log_background_message("ERROR", f"Error loading/decoding {filepath}: {e}. Returning default.")
        return default_value
    except Exception as e_unexp:
        log_background_message("ERROR", f"Unexpected error loading {filepath}: {e_unexp}")
        return default_value

def summarize_goal_performance(goals_data: list, num_recent_goals: int = 10) -> dict:
    summary = {
        "total_goals_analyzed": 0,
        "recent_terminal_goals_summary": [],
        "common_failure_categories": {},
        "overall_success_rate_evaluated": "N/A",
        "average_score_completed": "N/A"
    }
    if not isinstance(goals_data, list):
        return summary

    terminal_goals = sorted(
        [g for g in goals_data if isinstance(g, dict) and g.get("status") in ["completed", "executed_with_errors", "build_failed", "failed_max_retries", "failed_correction_unclear"]],
        key=lambda x: x.get("history", [{}])[-1].get("timestamp", x.get("created_at", "1970-01-01T00:00:00Z")),
        reverse=True
    )
    summary["total_goals_analyzed"] = len(terminal_goals)

    for g in terminal_goals[:num_recent_goals]:
        eval_data = g.get("evaluation", {})
        score = eval_data.get("final_score", "N/E")
        fail_cat = g.get("failure_category")
        entry = {
            "goal_id_suffix": g.get('goal_id', 'N/A')[-6:],
            "status": g.get('status'),
            "final_score": score,
            "failure_category": fail_cat,
            "goal_text_preview": g.get('goal', '')[:50] + "..."
        }
        summary["recent_terminal_goals_summary"].append(entry)
        if fail_cat and g.get("status") != "completed":
            summary["common_failure_categories"][fail_cat] = summary["common_failure_categories"].get(fail_cat, 0) + 1
    
    completed_goals = [g for g in terminal_goals if g.get("status") == "completed"]
    if terminal_goals: # All terminal goals, not just completed ones for this rate
        success_count = len(completed_goals)
        summary["overall_success_rate_evaluated"] = f"{(success_count / len(terminal_goals) * 100):.1f}% ({success_count}/{len(terminal_goals)})"

    completed_scores = [g.get("evaluation", {}).get("final_score") for g in completed_goals if isinstance(g.get("evaluation", {}).get("final_score"), (int, float))]
    if completed_scores:
        summary["average_score_completed"] = f"{sum(completed_scores) / len(completed_scores):.1f}"

    # Sort common failure categories by count
    summary["common_failure_categories"] = dict(sorted(summary["common_failure_categories"].items(), key=lambda item: item[1], reverse=True))

    return summary

def summarize_tool_performance(tool_registry_data: list) -> dict:
    summary = {
        "total_tools": 0,
        "tools_with_failures": 0,
        "overall_tool_success_rate": "N/A",
        "total_tool_runs": 0,
        "common_tool_failures": [], # List of tool names that frequently fail or have high failure rates
        "most_used_tools": []
    }
    if not isinstance(tool_registry_data, list):
        return summary

    summary["total_tools"] = len([t for t in tool_registry_data if isinstance(t, dict)])
    summary["tools_with_failures"] = len([t for t in tool_registry_data if isinstance(t, dict) and t.get("failure_count", 0) > 0])
    
    total_runs = sum(t.get("total_runs", 0) for t in tool_registry_data if isinstance(t, dict))
    total_successes = sum(t.get("success_count", 0) for t in tool_registry_data if isinstance(t, dict))
    summary["total_tool_runs"] = total_runs

    if total_runs > 0:
        summary["overall_tool_success_rate"] = f"{(total_successes / total_runs * 100):.1f}% ({total_successes}/{total_runs})"

    # Identify tools with high failure rates (e.g., > 50% failure after 3+ runs)
    for tool in tool_registry_data:
        if isinstance(tool, dict):
            runs = tool.get("total_runs", 0)
            failures = tool.get("failure_count", 0)
            if runs > 2 and failures / runs > 0.5:
                summary["common_tool_failures"].append({
                    "name": tool.get("name"),
                    "failure_rate": f"{(failures / runs * 100):.1f}% ({failures}/{runs})"
                })
    
    # Most used tools
    summary["most_used_tools"] = [
        {"name": t.get("name"), "runs": t.get("total_runs")}
        for t in sorted([td for td in tool_registry_data if isinstance(td, dict)], key=lambda x: x.get("total_runs", 0), reverse=True)[:3] # Top 3
        if isinstance(t, dict) and t.get("total_runs", 0) > 0
    ]
    return summary

def get_mission_focus(mission_data: dict) -> list:
    if not isinstance(mission_data, dict):
        return ["Mission data not available or malformed."]
    return mission_data.get("current_focus_areas", ["No specific focus areas defined in mission."])

def main():
    parser = argparse.ArgumentParser(description="Analyzes AI performance data from meta files and outputs a summary.")
    parser.add_argument(
        "--recent_goals", type=int, default=10,
        help="Number of recent terminal goals to include in the summary."
    )
    parser.add_argument(
        "--output_format", choices=['json', 'text'], default='json',
        help="Format for the output summary ('json' or 'text')."
    )
    # Add more arguments if needed, e.g., time window for analysis

    args = parser.parse_args()

    goals_data = load_json_data(GOALS_FILE, [])
    tool_registry_data = load_json_data(TOOL_REGISTRY_FILE, [])
    mission_data = load_json_data(MISSION_FILE, {})

    goal_summary = summarize_goal_performance(goals_data, args.recent_goals)
    tool_summary = summarize_tool_performance(tool_registry_data)
    mission_focus = get_mission_focus(mission_data)

    full_summary = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mission_current_focus": mission_focus,
        "goal_performance_summary": goal_summary,
        "tool_performance_summary": tool_summary
    }

    if args.output_format == 'json':
        print(json.dumps(full_summary, indent=2))
    else: # Text format
        output_lines = [
            "### AI Performance Analysis Summary ###",
            f"Generated: {full_summary['analysis_timestamp_utc']}",
            "\n--- Mission Focus ---",
        ]
        # Safely extend mission focus
        mission_focus_items = mission_focus if isinstance(mission_focus, list) else [str(mission_focus)]
        output_lines.extend(["- " + focus for focus in mission_focus_items])
        
        output_lines.extend([
            "\n--- Goal Performance ---",
            f"Overall Evaluated Success Rate: {goal_summary.get('overall_success_rate_evaluated', 'N/A')}",
            f"Average Score (Completed Goals): {goal_summary.get('average_score_completed', 'N/A')}",
            "Common Failure Categories:"
        ])
        common_failures = goal_summary.get("common_failure_categories", {})
        if common_failures:
            output_lines.extend([f"  - {cat}: {count}" for cat, count in common_failures.items()])
        else:
            output_lines.append("  (None identified)")

        output_lines.append(f"Recent Terminal Goals (last {len(goal_summary.get('recent_terminal_goals_summary',[]))} of {goal_summary.get('total_goals_analyzed')} analyzed):")
        recent_goals_summary_list = goal_summary.get("recent_terminal_goals_summary", [])
        if recent_goals_summary_list:
            for g_sum in recent_goals_summary_list:
                output_lines.append(f"  - GID ..{g_sum['goal_id_suffix']} (St: {g_sum['status']}, Score: {g_sum['final_score']}, FailCat: {g_sum['failure_category']}): '{g_sum['goal_text_preview']}'")
        else:
            output_lines.append("  (No recent terminal goals to display)")
        
        # --- Corrected Tool Performance Section ---
        output_lines.append("\n--- Tool Performance ---")
        output_lines.append(f"Total Tools: {tool_summary.get('total_tools')}")
        output_lines.append(f"Overall Tool Success Rate: {tool_summary.get('overall_tool_success_rate', 'N/A')}")
        
        output_lines.append("Tools with High Failure Rates (>50% after >2 runs):")
        common_tool_failures_list = tool_summary.get("common_tool_failures", [])
        if common_tool_failures_list:
            output_lines.extend([f"  - {t_fail['name']} ({t_fail['failure_rate']})" for t_fail in common_tool_failures_list])
        else:
            output_lines.append("  (None identified)")
            
        output_lines.append("Most Used Tools (Top 3):")
        most_used_tools_list = tool_summary.get("most_used_tools", [])
        if most_used_tools_list:
            output_lines.extend([f"  - {t_used['name']} ({t_used['runs']} runs)" for t_used in most_used_tools_list])
        else:
            output_lines.append("  (No tool usage tracked or tools not used)")
        # --- End of Corrected Tool Performance Section ---
        
        print("\n".join(output_lines))

if __name__ == "__main__":
    # Ensure meta directory and dummy files exist for standalone testing
    if not os.path.exists(META_DIR):
        os.makedirs(META_DIR)
    
    dummy_files_for_test = {
        GOALS_FILE: [{"goal_id":"testgoal1","status":"completed","history":[{"timestamp":"2025-01-01T10:00:00Z"}],"evaluation":{"final_score":20},"failure_category":None, "goal":"test goal one"}],
        TOOL_REGISTRY_FILE: [{"name":"test_tool","total_runs":5,"success_count":3,"failure_count":2}],
        MISSION_FILE: {"current_focus_areas":["testing","reliability"]}
    }
    for fp, content in dummy_files_for_test.items():
        if not os.path.exists(fp):
            try:
                with open(fp, 'w', encoding='utf-8') as d_f:
                    json.dump(content, d_f, indent=2)
                log_background_message("INFO", f"Created dummy {fp} for testing.")
            except Exception as e_create:
                log_background_message("ERROR", f"Could not create dummy {fp}: {e_create}")
    main()
    # Example of how to run from command line:
    # python tools/performance_analyzer_tool.py --output_format text --recent_goals 5