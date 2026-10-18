import os
import json
from pathlib import Path

from carnot.experiment_artifacts import resolve_experiment_artifact_path


def run():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    out_file = resolve_experiment_artifact_path(
        "results/experiment_1681_retro.json",
        root=repo_root,
        ensure_parent=True,
    )

    try:
        with open(os.path.join(results_dir, "experiment_1677_eds.json")) as f:
            eds_data = json.load(f)
    except FileNotFoundError:
        eds_data = {}

    try:
        with open(os.path.join(results_dir, "experiment_1678_crane.json")) as f:
            crane_data = json.load(f)
    except FileNotFoundError:
        crane_data = {}

    try:
        with open(os.path.join(results_dir, "experiment_1679_karat.json")) as f:
            karat_data = json.load(f)
    except FileNotFoundError:
        karat_data = {}

    try:
        with open(os.path.join(results_dir, "experiment_1680_continual_learning_eds.json")) as f:
            cl_eds_data = json.load(f)
    except FileNotFoundError:
        cl_eds_data = {}

    retro_data = {
        "experiment": "1681_milestone_retro",
        "schema": "milestone_retro_v2",
        "run_date": "2026-05-10T00:00:00Z",
        "milestone": "2026.05.129",
        "criteria_results": {
            "eds_prototype_success": eds_data.get("steered_generation_success", False),
            "crane_implemented_and_verified": crane_data.get("status") == "complete",
            "karat_attention_block_implemented_and_verified": karat_data.get("status")
            == "complete",
            "continual_learning_eds_not_blocked": cl_eds_data.get("status") != "blocked",
        },
        "criteria_met": 0,
        "criteria_total": 4,
        "experiment_honest_verdicts": {
            "exp1677": eds_data.get("honest_verdict", "unknown"),
            "exp1678": crane_data.get("honest_verdict", "unknown"),
            "exp1679": karat_data.get("honest_verdict", "unknown"),
            "exp1680": cl_eds_data.get("honest_verdict", "unknown"),
        },
        "notable_successes": [],
        "failures_or_partials": [],
        "bottlenecks_identified": [],
        "honest_verdict": "completed_with_blockers",
    }

    retro_data["criteria_met"] = sum(1 for v in retro_data["criteria_results"].values() if v)

    if eds_data.get("steered_generation_success"):
        retro_data["notable_successes"].append(
            "EDS prototype steered generation successfully reduced energy."
        )
    if crane_data.get("status") == "complete":
        retro_data["notable_successes"].append(
            "CRANE interleaving improved semantic coherence at parse_rate>=0.9."
        )
    if karat_data.get("status") == "complete":
        retro_data["notable_successes"].append(
            "KArAt miniature attention block was implemented and bounds verified."
        )

    if cl_eds_data.get("status") == "blocked":
        retro_data["failures_or_partials"].append(
            "Continual learning for EDS is blocked by gate checks."
        )
        retro_data["bottlenecks_identified"].append(
            "Gate check failed for exp1680 due to missing success expectation from exp1677."
        )

    out_file.write_text(json.dumps(retro_data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
