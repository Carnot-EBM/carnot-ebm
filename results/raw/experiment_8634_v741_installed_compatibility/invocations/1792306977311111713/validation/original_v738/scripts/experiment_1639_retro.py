"""Run the Exp 1639 retrospective."""

import sys
from pathlib import Path

# Ensure python/ is in sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "python"))

from carnot.reporting.milestone_retro_125 import run

if __name__ == "__main__":
    artifact = run()
    print(f"Written retrospective for Milestone 125, met {artifact['criteria_met']}/{artifact['criteria_total']} criteria.")
