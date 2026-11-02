"""Run the Exp 1676 retrospective for Milestone 128."""

import sys
import json
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = {
    "exp1666": "experiment_1666_nsvif.json",
    "exp1667": "experiment_1667_ebcn.json",
    "exp1668": "experiment_1668_cerce.json",
    "exp1669": "experiment_1669_ltlzinc.json",
    "exp1670": "experiment_1670_egd.json",
    "exp1671": "experiment_1671_rkan.json",
    "exp1672": "experiment_1672_igd.json",
    "exp1673": "experiment_1673_rng_audit.json",
    "exp1674": "experiment_1674_pipim.json",
    "exp1675": "experiment_1675_lagonn.json",
}

def load_result(results_dir: Path, filename: str) -> dict[str, Any]:
    path = results_dir / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def run(results_dir: Path | None = None, out_path: Path | None = None) -> dict[str, Any]:
    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    if out_path is None:
        out_path = results_dir / "experiment_1676_retro.json"

    started_at = datetime.now(UTC).isoformat()
    
    artifacts = {}
    for exp_id, filename in SOURCE_FILES.items():
        artifacts[exp_id] = load_result(results_dir, filename)

    criteria_results = {}
    criteria_details = {}
    criteria_met = 0
    criteria_total = len(artifacts)

    for exp_id, data in artifacts.items():
        if data and data.get("status") == "complete":
            criteria_results[exp_id] = True
            criteria_met += 1
        else:
            criteria_results[exp_id] = False
            
        criteria_details[exp_id] = {
            "experiment": exp_id,
            "verdict": data.get("honest_verdict", "MISSING"),
            "status": data.get("status", "MISSING")
        }

    honest_verdict = f"milestone_128_retrospective_filed_{criteria_met}_of_{criteria_total}_complete"
    finished_at = datetime.now(UTC).isoformat()

    artifact = {
        "experiment": 1676,
        "schema": "carnot.experiment.v1",
        "title": "Milestone 2026.05.128 Retrospective",
        "milestone": "2026.05.128",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "complete",
        "honest_verdict": honest_verdict,
        "criteria_results": criteria_results,
        "criteria_details": criteria_details,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact

if __name__ == "__main__":
    artifact = run()
    print(f"Written retrospective for Milestone 128, met {artifact['criteria_met']}/{artifact['criteria_total']} criteria.")
