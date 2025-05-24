"""
core_update_tool.py
Safely applies core-file updates:
• backs up old file (with version + approver)
• syntax-checks new code
• runs pytest on the whole project
• moves the new code in place only if all tests pass
• supports rollback by version
"""

import os, json, shutil, subprocess, sys
from datetime import datetime, timezone
from notifier_module import get_current_version # Assuming notifier_module is in PYTHONPATH or same dir
from logger_utils import should_log # Added import

# --------------------------------------------------------------------------- #
#  Paths / constants
# --------------------------------------------------------------------------- #
BACKUP_DIR   = "core_backups"
INDEX_FILE   = os.path.join(BACKUP_DIR, "backup_index.json")
TEST_DIR     = "tests"           # where your pytest tests live
NEW_SUFFIX   = "_new.py"         # e.g. console_ai_new.py

os.makedirs(BACKUP_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Backup helpers
# --------------------------------------------------------------------------- #
def _load_index():
    if not os.path.exists(INDEX_FILE):
        return []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def _save_index(idx):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)

def backup_file(path: str, version: str, approver: str):
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = os.path.basename(path)
    dst  = os.path.join(BACKUP_DIR, f"{version}_{approver}_{ts}_{name}")
    shutil.copy2(path, dst)

    idx = _load_index()
    idx.append(
        dict(version=version, timestamp=ts,
             approved_by=approver, filename=name, backup_path=dst)
    )
    _save_index(idx)
    return dst

# --------------------------------------------------------------------------- #
#  Quality gates
# --------------------------------------------------------------------------- #
def syntax_check(pyfile: str) -> tuple[bool, str]:
    try:
        # Ensure the file path is absolute or correctly relative for subprocess
        abs_pyfile = os.path.abspath(pyfile)
        if not os.path.exists(abs_pyfile):
            return False, f"File not found for syntax check: {abs_pyfile}"

        p = subprocess.run(
            [sys.executable, "-m", "py_compile", abs_pyfile], # Use absolute path
            capture_output=True, text=True, check=False, encoding='utf-8', errors='replace'
        )
        return (p.returncode == 0, p.stderr or p.stdout or "Syntax OK")
    except Exception as e:
        return False, f"Exception during syntax check: {e}"


def run_pytest() -> tuple[bool, str]:
    """
    Runs pytest in TEST_DIR.  Returns (ok, full_output).
    If there is no tests directory, it passes by default.
    """
    if not os.path.isdir(TEST_DIR):
        return True, "No tests/ directory; skipping tests."

    try:
        # Ensure TEST_DIR is correctly passed if it's relative
        abs_test_dir = os.path.abspath(TEST_DIR)
        p = subprocess.run(
            ["pytest", abs_test_dir, "-q"], capture_output=True, text=True, check=False, encoding='utf-8', errors='replace'
        )
        ok  = p.returncode == 0
        # pytest often sends useful info to stdout on success, and stderr for errors or warnings
        out = (p.stdout or "") + (p.stderr or "")
        return ok, out.strip() or "(pytest produced no output)"
    except Exception as e:
        return False, f"Exception during pytest execution: {e}"

# --------------------------------------------------------------------------- #
#  Update workflow
# --------------------------------------------------------------------------- #
def update_core_file(new_code: str, target_file: str,
                     approved_by: str = "AI") -> tuple[bool, str]:
    """
    Write <target_file>_new.py, run syntax & tests.
    Caller should invoke apply_update(...) if this returns True.
    """
    if not os.path.isabs(target_file):
        target_file_abs = os.path.abspath(target_file)
    else:
        target_file_abs = target_file

    # new_path should be in the same directory as target_file_abs
    new_path = os.path.join(os.path.dirname(target_file_abs), os.path.basename(target_file_abs).replace(".py", NEW_SUFFIX))
    
    try:
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(new_code)
    except IOError as e:
        return False, f"❌ Failed to write new code to {new_path}: {e}"

    ok, msg = syntax_check(new_path)
    if not ok:
        # Clean up the temporary _new.py file if syntax check fails
        if os.path.exists(new_path): os.remove(new_path)
        return False, f"❌ Syntax check failed:\n{msg}"

    tests_ok, test_msg = run_pytest()
    if not tests_ok:
        # Clean up if tests fail
        if os.path.exists(new_path): os.remove(new_path)
        return False, f"❌ Tests failed:\n{test_msg}"

    return True, f"✅ Syntax and all tests passed for {new_path}."

def apply_update(target_file: str, approved_by: str = "AI"):
    """
    Backup current core file, then replace it with *target_file_new.py*.
    """
    if not os.path.isabs(target_file):
        target_file_abs = os.path.abspath(target_file)
    else:
        target_file_abs = target_file
        
    new_path = os.path.join(os.path.dirname(target_file_abs), os.path.basename(target_file_abs).replace(".py", NEW_SUFFIX))

    if not os.path.exists(new_path):
        raise FileNotFoundError(f"{new_path} not found. Run update_core_file() first and ensure it passes.")
    
    if not os.path.exists(target_file_abs):
        # If the target file doesn't exist, this is an initial creation, not an update.
        # No backup needed. Just move new_path to target_file_abs.
        if should_log("INFO"): print(f"INFO: Target file {target_file_abs} does not exist. Applying as new file.")
        shutil.move(new_path, target_file_abs)
        if should_log("INFO"): print(f"\n🚀 New core file applied: {target_file_abs}")
        # Versioning might need adjustment here if it's a "first version" scenario
        ver = "0.0.1" # Or fetch from a "next_version" mechanism
        if should_log("INFO"): print(f"✅ Core file {os.path.basename(target_file_abs)} created (version {ver} - placeholder)") # Adjust versioning as needed
        return


    ver = get_current_version() # Assumes version exists for existing file
    backup_dest_path = backup_file(target_file_abs, ver, approved_by)
    if should_log("INFO"): print(f"Backed up {target_file_abs} to {backup_dest_path}")
    
    try:
        shutil.move(new_path, target_file_abs)
    except Exception as e:
        if should_log("ERROR"): print(f"❌ Error moving {new_path} to {target_file_abs}: {e}")
        # Attempt to restore from immediate backup if move fails? Or leave for manual.
        # For now, error out. User might need to resolve permissions or other issues.
        raise
        
    if should_log("INFO"): print(f"\n🚀 Core update applied: {target_file_abs}")
    if should_log("INFO"): print(f"✅ Core file {os.path.basename(target_file_abs)} updated (version {ver} backed up)")


# --------------------------------------------------------------------------- #
#  Rollback
# --------------------------------------------------------------------------- #
def rollback_version(version_to_restore: str) -> bool:
    idx = _load_index()
    # Filter for entries matching the version, sort by timestamp descending (newest first)
    # in case multiple backups exist for the same version string (e.g. if versioning isn't strictly linear)
    items_for_version = sorted(
        [e for e in idx if e.get("version") == version_to_restore],
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )
    
    if not items_for_version:
        if should_log("WARNING"): print(f"⚠️ No backups found for version '{version_to_restore}'."); return False

    restored_files_count = 0
    # Group by original filename to restore the latest backup for that specific file for that version
    files_to_restore_map = {} # {original_filename: backup_entry}
    for entry in items_for_version:
        original_filename = entry.get("filename")
        if original_filename not in files_to_restore_map: # Take the first one (newest due to sort)
            files_to_restore_map[original_filename] = entry
            
    if not files_to_restore_map:
        if should_log("WARNING"): print(f"⚠️ No valid entries with filenames found for version '{version_to_restore}'."); return False

    for original_filename, entry_to_restore in files_to_restore_map.items():
        backup_path = entry_to_restore.get("backup_path")
        target_path = os.path.abspath(original_filename) # Ensure we restore to correct location

        if not os.path.exists(backup_path):
            if should_log("ERROR"): print(f"❌ Backup file {backup_path} for {original_filename} (version {version_to_restore}) not found. Skipping.")
            continue
        
        try:
            # Ensure target directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(backup_path, target_path)
            if should_log("INFO"): print(f"🔄 Restored {target_path} from backup {os.path.basename(backup_path)}")
            restored_files_count += 1
        except Exception as e:
            if should_log("ERROR"): print(f"❌ Failed to restore {target_path} from {backup_path}: {e}")
            return False # Stop rollback on first error to prevent partial state

    if restored_files_count > 0:
        if should_log("INFO"): print(f"\n✅ Successfully rolled back {restored_files_count} file(s) to version '{version_to_restore}'.")
        # Note: This doesn't automatically update the "current_version" in version.json.
        # That should probably be a manual step or a separate command if version.json is meant to track the *active* codebase version.
        return True
    else:
        if should_log("WARNING"): print(f"⚠️ No files were actually restored for version '{version_to_restore}'. Check backup paths and file existence.")
        return False

# --------------------------------------------------------------------------- #
#  CLI utility
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2: # Adjusted for single command like "test" without file
        if should_log("INFO"): print("Usage:")
        if should_log("INFO"): print("  python core_update_tool.py test_syntax <FILE_new.py>")
        if should_log("INFO"): print("  python core_update_tool.py test_suite")
        if should_log("INFO"): print("  python core_update_tool.py stage <FILE_to_update.py> <NEW_CODE_FILE.py> [approver]")
        if should_log("INFO"): print("  python core_update_tool.py apply <FILE_to_update.py> [approver]")
        if should_log("INFO"): print("  python core_update_tool.py rollback <VERSION>")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "test_syntax":
        if len(sys.argv) < 3:
            if should_log("INFO"): print("Usage: python core_update_tool.py test_syntax <FILE_new.py>")
            sys.exit(1)
        file_to_check = sys.argv[2]
        ok, msg = syntax_check(file_to_check)
        if ok:
            if should_log("DEBUG"): print(msg)
        else:
            if should_log("ERROR"): print(msg)
        sys.exit(0 if ok else 1)
    
    elif cmd == "test_suite":
        ok, msg = run_pytest()
        if ok:
            if should_log("INFO"): print(msg)
        else:
            if should_log("ERROR"): print(msg)
        sys.exit(0 if ok else 1)

    elif cmd == "stage":
        if len(sys.argv) < 4:
            if should_log("INFO"): print("Usage: python core_update_tool.py stage <FILE_to_update.py> <NEW_CODE_FILE.py> [approver]")
            sys.exit(1)
        target_py_file = sys.argv[2]
        new_code_file_path = sys.argv[3]
        # approver = sys.argv[4] if len(sys.argv) > 4 else "user_cli" # Not used directly by update_core_file
        
        if not os.path.exists(new_code_file_path):
            if should_log("ERROR"): print(f"Error: New code file not found: {new_code_file_path}")
            sys.exit(1)
        
        with open(new_code_file_path, 'r', encoding='utf-8') as f_new_code:
            new_code_content = f_new_code.read()
            
        if should_log("INFO"): print(f"Staging update for '{target_py_file}' with code from '{new_code_file_path}'...")
        ok, msg = update_core_file(new_code_content, target_py_file) # approver is for apply_update
        if ok: # "✅"
            if should_log("INFO"): print(msg)
            if should_log("INFO"): print(f"Staging successful. To apply, run: python core_update_tool.py apply {target_py_file} [approver]")
        else: # "❌"
            if should_log("ERROR"): print(msg)
        sys.exit(0 if ok else 1)

    elif cmd == "apply":
        if len(sys.argv) < 3:
            if should_log("INFO"): print("Usage: python core_update_tool.py apply <FILE.py> [approver]")
            sys.exit(1)
        tgt = sys.argv[2]
        approver_cli = sys.argv[3] if len(sys.argv) > 3 else "user_cli"
        try:
            apply_update(tgt, approver_cli)
        except FileNotFoundError as e:
            if should_log("ERROR"): print(f"Error: {e}")
            if should_log("ERROR"): print("Ensure you have staged the update first using 'stage' command or that _new.py file exists.")
            sys.exit(1)
        except Exception as e_apply:
            if should_log("ERROR"): print(f"An error occurred during apply: {e_apply}")
            sys.exit(1)
        sys.exit(0)
            
    elif cmd == "rollback":
        if len(sys.argv) < 3:
            if should_log("INFO"): print("Usage: python core_update_tool.py rollback <VERSION>")
            sys.exit(1)
        ver = sys.argv[2]
        rollback_version(ver)
        # Exit status could depend on rollback_version's return
        sys.exit(0) # Assuming rollback_version prints errors
    else:
        if should_log("ERROR"): print(f"Unknown command: {cmd}")
        sys.exit(1)