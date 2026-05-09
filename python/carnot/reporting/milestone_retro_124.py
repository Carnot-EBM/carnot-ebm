"""Build the Exp 1626 milestone .124 retrospective artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILES = {
    "exp1614": "experiment_1614_archive.json",
    "exp1615": "experiment_1615_ets_decoding.json",
    "exp1616": "experiment_1616_nabla_reasoner.json",
    "exp1617": "experiment_1617_nabla_sota_validation.json",
    "exp1618": "experiment_1618_pwa_kan.json",
    "exp1619": "experiment_1619_milp_kan_compiler.json",
    "exp1620": "experiment_1620_milp_fr11.json",
    "exp1621": "experiment_1621_kanele_mapping.json",
    "exp1622": "experiment_1622_kanele_lint.json",
    "exp1623": "experiment_1623_kanele_accounting.json",
    "exp1624": "experiment_1624_adaptive_reconfig.json",
    "exp1625": "experiment_1625_task_router.json",
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
        out_path = results_dir / "experiment_1626_retro.json"

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

    honest_verdict = f"milestone_124_retrospective_filed_{criteria_met}_of_{criteria_total}_complete"
    finished_at = datetime.now(UTC).isoformat()

    artifact = {
        "experiment": 1626,
        "schema": "carnot.experiment.v1",
        "title": "Milestone 2026.05.124 Retrospective",
        "milestone": "2026.05.124",
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
