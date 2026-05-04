"""Tests for the Exp 1281 milestone .99 retrospective.

Spec: REQ-REPORT-023, SCENARIO-REPORT-020.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_99 import (
    CRITERION_NAMES,
    REQUIRED_SOTA_GGUF_IDS,
    SOURCE_FILES,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[int, dict[str, object]]:
    return {
        1268: {
            "status": "complete",
            "retro_backfill_complete": True,
            "stale_artifacts": [{"path": "results/experiment_1255_combined_retro_95_96_97.json"}],
        },
        1269: {"status": "complete", "critical_issues_fixed": 5},
        1270: {
            "status": "complete",
            "critical_issues_fixed": 5,
            "pdf_compiled": True,
            "bundle_path": "results/carnot-arxiv-v10-20260504.tar.gz",
        },
        1271: {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
        1272: {
            "status": "complete",
            "verifier_weight_vector_written": True,
            "verifier_weight_vector": {"SemEnergyProbe": 1.0},
        },
        1273: {
            "status": "complete",
            "honest_verdict": "smoke_only_not_headline",
            "grpo_v8_delta_pp": 83.798,
            "self_learning_delta_overall": 0.83798,
            "headline_result_allowed": False,
            "MODEL_SPECS": [],
            "models_used": [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "available": False,
                    "used_for_generation": False,
                }
            ],
        },
        1274: {
            "status": "complete",
            "self_learning_delta_overall": 0.357143,
            "memory_entries": 7,
            "skill_graph_candidate_count": 1,
        },
        1275: {"status": "complete", "feasibility_delta_overall": 4.583247},
        1276: {"status": "complete", "repair_delta_over_fsnet": 0.21996},
        1278: {"status": "complete", "gaming_defense_measured": True},
        1279: {"status": "shipped", "cartridge_shipped": True},
        1280: {"status": "completed", "cartridge_shipped": True},
    }


def test_scenario_report_020_counts_milestone_99_source_criteria() -> None:
    """SCENARIO-REPORT-020: Exp1281 reports .99 12/14 from source fields."""

    artifact = build_artifact(_scenario_sources())

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_results"] == {
        "retro_backfill_95_96_97_closed": "MET",
        "paper_v6_critical_fixes_complete": "MET",
        "arxiv_bundle_v10_written_after_gate": "MET",
        "triggered_certificate_sota_gguf_measured": "BLOCKED",
        "prime_verifier_weight_vector_written": "MET",
        "grpo_v8_prime_vprm_delta_reported": "MET",
        "online_self_learning_certificate_memory_measured": "MET",
        "fsnet_feasibility_improvement_measured": "MET",
        "snarenet_repair_layer_gated_tested": "MET",
        "cactus_constrained_acceptance_gated_measured": "GATED",
        "gaming_verifier_defense_final_measured": "MET",
        "wopr_kakuro_shipped_or_blocked": "MET",
        "wopr_masyu_shipped_or_blocked": "MET",
        "retro_99_complete": "MET",
    }
    assert artifact["criteria_met"] == 12
    assert artifact["criteria_total"] == 14
    assert artifact["retro_complete"] is True
    assert artifact["honest_verdict"] == "milestone_99_12_of_14_criteria_met"
    assert artifact["self_learning_result"]["overall_direction"] == "positive"
    assert artifact["sota_model_usage_summary"]["headline_model_ids_used"] == []
    assert artifact["sota_model_usage_summary"]["headline_result_allowed"] is False
    assert any("experiment_1271" in item["path"] for item in artifact["stale_artifacts"])
    assert any("experiment_1277" in item["path"] for item in artifact["stale_artifacts"])
    assert len(artifact["top_successes"]) == 5
    assert len(artifact["top_gaps"]) == 3
    assert len(artifact["key_carry_forwards"]) == 4


def test_req_report_023_in_progress_artifact_is_durable(tmp_path: Path) -> None:
    """REQ-REPORT-023: a run can leave an auditable in-progress artifact."""

    out_path = tmp_path / "results" / "experiment_1281_milestone_retro_99.json"

    artifact = write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["run_date"] == "20260504"
    assert written["criteria_total"] == 14
    assert written["retro_complete"] is False


def test_req_report_023_run_loads_sources_and_writes_schema(tmp_path: Path) -> None:
    """REQ-REPORT-023: run loads result JSON files and writes the final artifact."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1281_milestone_retro_99.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)

    artifact = run(results_dir=results_dir, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment"] == "1281_milestone_retro_99"
    assert written["schema"] == "milestone_retro_v4"
    assert written["milestone"] == "2026.04.99"
    assert written["status"] == "complete"
    assert written["criteria_met"] == 12
    assert written["criteria_total"] == 14


def test_req_report_023_missing_non_gated_artifact_is_missing() -> None:
    """REQ-REPORT-023: absent ungated artifacts are MISSING, not GATED."""

    sources = _scenario_sources()
    sources.pop(1270)

    artifact = build_artifact(sources)

    assert artifact["criteria_results"]["arxiv_bundle_v10_written_after_gate"] == "MISSING"
    assert any("experiment_1270" in item["path"] for item in artifact["stale_artifacts"])


def test_req_report_023_classifies_stale_and_explicit_gate_paths() -> None:
    """REQ-REPORT-023: stale, gated, and SOTA-used branches are mechanical."""

    model_id = REQUIRED_SOTA_GGUF_IDS[0]
    sources = _scenario_sources()
    sources[1270] = {"status": "in_progress", "honest_verdict": "in_progress"}
    sources[1271] = {
        "status": "complete",
        "MODEL_SPECS": [{"hf_id": model_id, "used_for_generation": True}],
        "certificate_parse_rate": 0.7,
    }
    sources[1275] = {"status": "complete", "feasibility_delta_overall": 0.0}
    sources[1277] = {"status": "complete"}

    gated_artifact = build_artifact(sources)

    assert gated_artifact["criteria_results"]["arxiv_bundle_v10_written_after_gate"] == "NOT_MET"
    assert gated_artifact["criteria_results"]["snarenet_repair_layer_gated_tested"] == "GATED"
    assert gated_artifact["criteria_results"]["cactus_constrained_acceptance_gated_measured"] == "GATED"
    assert gated_artifact["sota_model_usage_summary"]["headline_model_ids_used"] == [model_id]
    assert gated_artifact["sota_model_usage_summary"]["headline_result_allowed"] is True

    sources[1271]["certificate_parse_rate"] = 0.9
    sources[1275]["feasibility_delta_overall"] = 1.0
    sources[1277]["cactus_acceptance_rate"] = 0.5

    ungated_artifact = build_artifact(sources)

    assert ungated_artifact["criteria_results"]["cactus_constrained_acceptance_gated_measured"] == "MET"
