"""Build the Exp 1651 milestone .126 retrospective artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILES = {
    "exp1640": "experiment_1640_nsvif_dsl.json",
    "exp1641": "experiment_1641_nsvif_sota.json",
    "exp1642": "experiment_1642_llguidance.json",
    "exp1643": "experiment_1643_static_csr.json",
    "exp1644": "experiment_1644_cerce_ledger.json",
    "exp1645": "experiment_1645_fr11_cerce.json",
    "exp1646": "experiment_1646_ebcn.json",
    "exp1647": "experiment_1647_rkan.json",
    "exp1648": "experiment_1648_sparse_kan.json",
    "exp1649": "experiment_1649_vivado_synthesis.json",
    "exp1650": "experiment_1650_kv260_bringup.json",
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
        out_path = results_dir / "experiment_1651_retro.json"

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

    honest_verdict = f"milestone_126_retrospective_filed_{criteria_met}_of_{criteria_total}_complete"
    finished_at = datetime.now(UTC).isoformat()

    artifact = {
        "experiment": 1651,
        "schema": "carnot.experiment.v1",
        "title": "Milestone 2026.05.126 Retrospective",
        "milestone": "2026.05.126",
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
