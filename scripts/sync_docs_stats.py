#!/usr/bin/env python3
"""Sync docs/index.html stats with actual experiment/model/test counts.

Run manually or via pre-commit hook to keep the website stats current.

Usage:
    python scripts/sync_docs_stats.py
"""

import os
import re
import subprocess
import sys


def count_experiments() -> int:
    """Count experiments from the experiment log summary table."""
    log_path = os.path.join(os.path.dirname(__file__), "..", "ops", "experiment-log.md")
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as f:
        content = f.read()
    # Count all experiment rows in the summary table (| # | or | — |)
    count = 0
    in_table = False
    for line in content.split("\n"):
        if "| # |" in line:
            in_table = True
            continue
        if in_table and line.startswith("|"):
            if "---" in line:
                continue
            count += 1
        elif in_table and not line.startswith("|"):
            if line.strip() == "":
                continue
            break
    return count


def count_models() -> int:
    """Count exported model directories."""
    exports_dir = os.path.join(os.path.dirname(__file__), "..", "exports")
    if not os.path.exists(exports_dir):
        return 0
    return sum(1 for d in os.listdir(exports_dir)
               if d.startswith("per-token-ebm-") and
               os.path.isdir(os.path.join(exports_dir, d)))


def count_principles() -> int:
    """Count principles from the experiment log."""
    log_path = os.path.join(os.path.dirname(__file__), "..", "ops", "experiment-log.md")
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as f:
        content = f.read()
    # Find the highest numbered principle
    matches = re.findall(r"^(\d+)\.\s+\*\*", content, re.MULTILINE)
    return max(int(m) for m in matches) if matches else 0


def count_tests() -> int:
    """Count Python tests by running pytest --collect-only."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/python", "--collect-only", "-q", "--no-header"],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        # Last line is like "1107 tests collected"
        for line in result.stdout.strip().split("\n"):
            match = re.search(r"(\d+) tests? collected", line)
            if match:
                return int(match.group(1))
    except Exception:
        pass
    return 0


def update_index(experiments: int, models: int, principles: int, tests: int) -> bool:
    """Update docs/index.html with current stats. Returns True if changed."""
    index_path = os.path.join(os.path.dirname(__file__), "..", "docs", "index.html")
    if not os.path.exists(index_path):
        return False

    with open(index_path) as f:
        content = f.read()

    original = content

    # Replace stat values
    def replace_stat(content: str, label: str, value: str) -> str:
        pattern = rf'(<div class="stat-num">)\d[^<]*(</div><div class="stat-label">{label})'
        return re.sub(pattern, rf"\g<1>{value}\g<2>", content)

    content = replace_stat(content, "Experiments", str(experiments))
    content = replace_stat(content, "Models", str(models))
    content = replace_stat(content, "Principles", str(principles))
    if tests > 0:
        content = replace_stat(content, "Tests", f"{tests:,}")

    if content != original:
        with open(index_path, "w") as f:
            f.write(content)
        return True
    return False


def main() -> int:
    experiments = count_experiments()
    models = count_models()
    principles = count_principles()
    tests = count_tests()

    print(f"Experiments: {experiments}")
    print(f"Models: {models}")
    print(f"Principles: {principles}")
    print(f"Tests: {tests}")

    changed = update_index(experiments, models, principles, tests)
    if changed:
        print("docs/index.html updated")
    else:
        print("docs/index.html already current")

    return 0


if __name__ == "__main__":
    sys.exit(main())
