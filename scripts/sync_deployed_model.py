"""Keep the git-committed model artifact in sync with models/current.json.

train.py updates models/current.json locally after every run, but the deploy
setup commits exactly one .pkl to git (everything else in models/ stays
DVC-only, see .gitignore). This script closes that gap: run it after any
retrain, before committing, and it makes sure git tracks the *current* model
file instead of a stale one.

What it does:
  1. Reads models/current.json to find the current model filename.
  2. Rewrites the `!models/<file>.pkl` negation line in .gitignore to match.
  3. `git rm --cached`s the previously-committed .pkl (if it's different and
     still tracked), so it goes back to being DVC-only / ignored.
  4. `git add`s the new .pkl, current.json, features.parquet, and
     monitoring/prediction_log.csv -- the files a git-only deploy needs.

It stages changes; it does not commit or push. Review with `git status` /
`git diff --cached` and commit yourself.

Usage:
    python scripts/sync_deployed_model.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRENT_JSON = ROOT / "models" / "current.json"
GITIGNORE = ROOT / ".gitignore"

# Files that need to be up to date in git for a git-only (no DVC remote) deploy.
DEPLOY_PATHS = [
    "models/current.json",
    "data/processed/features.parquet",
    "monitoring/prediction_log.csv",
]

NEGATION_RE = re.compile(r"^!models/(.+\.pkl)\s*$", re.MULTILINE)


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result


def main():
    if not CURRENT_JSON.exists():
        sys.exit(f"models/current.json not found at {CURRENT_JSON} -- run training first.")

    current = json.loads(CURRENT_JSON.read_text())
    new_model = current["current_model"]
    print(f"Current model per models/current.json: {new_model}")

    if not GITIGNORE.exists():
        sys.exit(f".gitignore not found at {GITIGNORE}")

    gitignore_text = GITIGNORE.read_text()
    match = NEGATION_RE.search(gitignore_text)
    old_model = match.group(1) if match else None

    if old_model == new_model:
        print("gitignore negation line already points at the current model -- nothing to rewrite.")
    else:
        if match:
            new_text = gitignore_text[: match.start()] + f"!models/{new_model}\n" + gitignore_text[match.end() :]
            if not new_text.endswith("\n"):
                new_text += "\n"
        else:
            # No negation line found -- append one under the existing comment block, or at the end.
            sep = "" if gitignore_text.endswith("\n") else "\n"
            new_text = gitignore_text + sep + f"!models/{new_model}\n"
        GITIGNORE.write_text(new_text)
        print(f"Rewrote .gitignore: !models/{old_model or '(none)'} -> !models/{new_model}")

        if old_model:
            old_path = f"models/{old_model}"
            check = run(["git", "ls-files", "--error-unmatch", old_path])
            if check.returncode == 0:
                run(["git", "rm", "--cached", old_path])
                print(f"  git rm --cached {old_path} (now DVC-only / ignored again)")

    new_model_path = f"models/{new_model}"
    if not (ROOT / new_model_path).exists():
        sys.exit(f"{new_model_path} does not exist on disk -- did training actually produce it?")

    to_add = [new_model_path] + DEPLOY_PATHS
    existing = [p for p in to_add if (ROOT / p).exists()]
    missing = [p for p in to_add if p not in existing]
    if missing:
        print(f"  (skipping, not found on disk: {', '.join(missing)})")

    run(["git", "add", GITIGNORE.name] + existing)
    print("\nStaged for commit:")
    run(["git", "status", "--short"] + existing + [GITIGNORE.name])

    print(
        "\nDone. Review with `git status` / `git diff --cached`, then:\n"
        '  git commit -m "Sync deployed model to <timestamp>"\n'
        "  git push"
    )


if __name__ == "__main__":
    main()
