"""Build the Exp 1695 milestone .130 retrospective artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILES = {
    "exp1682": "experiment_1682_archive.json",
    "exp1683": "experiment_1683_self_play.json",
    "exp1684": "experiment_1684_fr11_ledger_integration.json",
    "exp1685": "experiment_1685_live_sota.json",
    "exp1686": "experiment_1686_pwa_karat.json",
    "exp1687": "experiment_1687_milp_karat_verification.json",
    "exp1688": "experiment_1688_cikan.json",
    "exp1689": "experiment_1689_certified_karat.json",
    "exp1690": "experiment_1690_nabla_ets.json",
    "exp1691": "experiment_1691_nabla_ets_ablation.json",
    "exp1692": "experiment_1692_potts_export.json",
    "exp1693": "experiment_1693_potts_sim.json",
    "exp1694": "experiment_1694_full_pipeline.json",
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
        out_path = results_dir / "experiment_1695_retro.json"

    started_at = datetime.now(UTC).isoformat()
    
    artifacts = {}
    for exp_id, filename in SOURCE_FILES.items():
        artifacts[exp_id] = load_result(results_dir, filename)

    criteria_results = {}
    criteria_details = {}
    criteria_met = 0
    criteria_total = len(artifacts)

    for exp_id, data in artifacts.items():
        if data and data.get("status") in ("complete", "success"):
            criteria_results[exp_id] = True
            criteria_met += 1
        else:
            criteria_results[exp_id] = False
            
        criteria_details[exp_id] = {
            "experiment": exp_id,
            "verdict": data.get("honest_verdict", "MISSING"),
            "status": data.get("status", "MISSING")
        }

    honest_verdict = f"milestone_130_retrospective_filed_{criteria_met}_of_{criteria_total}_complete"
    finished_at = datetime.now(UTC).isoformat()

    artifact = {
        "experiment": 1695,
        "schema": "carnot.experiment.v1",
        "title": "Milestone 2026.05.130 Retrospective",
        "milestone": "2026.05.130",
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
