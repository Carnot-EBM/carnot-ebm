"""Build the Exp 1639 milestone .125 retrospective artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILES = {
    "exp1627": "experiment_1627_nabla_debug.json",
    "exp1628": "experiment_1628_ebrm_scoring.json",
    "exp1629": "experiment_1629_ebrm_sota.json",
    "exp1630": "experiment_1630_ltlzinc.json",
    "exp1631": "experiment_1631_smgi.json",
    "exp1632": "experiment_1632_fr11_learning.json",
    "exp1633": "experiment_1633_pinet.json",
    "exp1634": "experiment_1634_pinet_vs_tskm.json",
    "exp1635": "experiment_1635_consformer.json",
    "exp1636": "experiment_1636_energy_guided_decoding.json",
    "exp1637": "experiment_1637_vivado_lint.json",
    "exp1638": "experiment_1638_kanele_rtl_simulation.json",
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
        out_path = results_dir / "experiment_1639_retro.json"

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

    honest_verdict = f"milestone_125_retrospective_filed_{criteria_met}_of_{criteria_total}_complete"
    finished_at = datetime.now(UTC).isoformat()

    artifact = {
        "experiment": 1639,
        "schema": "carnot.experiment.v1",
        "title": "Milestone 2026.05.125 Retrospective",
        "milestone": "2026.05.125",
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
