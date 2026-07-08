"""Tests for Exp5408 resource-accounted continuous self-learning routing.

Spec refs: REQ-LEARN-5408,
SCENARIO-LEARN-5408-RESOURCE-COUNTERS,
SCENARIO-LEARN-5408-PROVENANCE, SCENARIO-LEARN-5408-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5408_resource_accounted_csl_controller_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5408_spec_declares_resource_accounted_contract() -> None:
    """REQ-LEARN-5408: OpenSpec anchors resource-accounted CSL routing."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5408") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5408",
        "SCENARIO-LEARN-5408-RESOURCE-COUNTERS",
        "SCENARIO-LEARN-5408-PROVENANCE",
        "SCENARIO-LEARN-5408-READY",
        str(exp.RESULT_RELATIVE_PATH),
        "wall-time",
        "token-or-context",
        "memory proxy",
        "unproductive-loop",
        "SHALL NOT load, fine-tune, write, or mutate model weights or adapter weights",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5408_decisions_carry_resource_counters() -> None:
    """SCENARIO-LEARN-5408-RESOURCE-COUNTERS: every decision is accounted."""

    evaluation = exp.evaluate_resource_accounted_controller(root=REPO)
    decisions = evaluation["resource_accounted_decisions"]

    assert evaluation["session_count"] >= exp.MIN_SESSIONS
    assert evaluation["decision_count"] == len(decisions)
    assert evaluation["raw_episode_count"] > 0
    assert evaluation["influence_share_sum_valid_rate"] == 1.0
    assert evaluation["quality_delta_vs_baseline"] >= 0.0
    assert evaluation["verifier_cost_delta_vs_baseline"] > 0.0
    assert evaluation["wall_time_delta_vs_baseline"] > 0.0
    assert evaluation["token_or_context_delta_vs_baseline"] > 0.0
    assert evaluation["memory_delta_vs_baseline"] > 0.0
    assert evaluation["unproductive_loop_reduction_rate"] > 0.0

    for row in decisions:
        assert row["variant_name"] == exp.RESOURCE_ACCOUNTED_VARIANT
        assert row["influence_share_sum"] == 100
        assert sum(row["influence_shares"].values()) == 100
        assert row["baseline_resources"].keys() == row["resource_accounted_resources"].keys()
        assert tuple(row["baseline_resources"]) == exp.RESOURCE_COUNTER_NAMES
        assert tuple(row["resource_accounted_resources"]) == exp.RESOURCE_COUNTER_NAMES
        assert all(value >= 0 for value in row["resource_savings"].values())
        assert row["resource_accounted_resources"]["wall_time_ms"] <= row["baseline_resources"][
            "wall_time_ms"
        ]
        assert row["resource_accounted_resources"]["token_or_context_units"] <= row[
            "baseline_resources"
        ]["token_or_context_units"]
        assert row["resource_accounted_resources"]["memory_proxy_mb"] <= row[
            "baseline_resources"
        ]["memory_proxy_mb"]
        assert row["raw_episode_provenance"]["raw_episode_id"]
        assert row["raw_episode_provenance"]["raw_payload_checksum"].startswith("sha256:")
        assert row["no_weight_mutation"] is True
        assert row["no_adapter_weight_mutation"] is True


def test_scenario_learn_5408_provenance_and_controls_deflect_bad_memory() -> None:
    """SCENARIO-LEARN-5408-PROVENANCE: controls retain raw provenance."""

    evaluation = exp.evaluate_resource_accounted_controller(root=REPO)
    raw_ids = {row["raw_episode_id"] for row in evaluation["raw_episodes"]}
    stale_rows = [row for row in evaluation["resource_accounted_decisions"] if row["stale_probe"]]
    poison_rows = [
        row for row in evaluation["resource_accounted_decisions"] if row["poison_probe"]
    ]
    rollback_rows = [
        row
        for row in evaluation["resource_accounted_decisions"]
        if row["rollback_status"] == "recovered"
    ]

    assert stale_rows
    assert poison_rows
    assert rollback_rows
    assert evaluation["stale_memory_deflection_rate"] == 1.0
    assert evaluation["poison_memory_deflection_rate"] == 1.0
    assert evaluation["rollback_success_rate"] == 1.0
    assert evaluation["provenance_link_rate"] == 1.0
    assert evaluation["poison_control_summary"]["locally_correct_nontransferable_deflected"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "controller_resource_accounting_only",
    }

    for row in evaluation["resource_accounted_decisions"]:
        assert row["raw_episode_provenance"]["raw_episode_id"] in raw_ids

    assert all(row["stale_control_deflected"] for row in stale_rows)
    assert all(row["poison_control_deflected"] for row in poison_rows)


def test_req_learn_5408_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5408-READY: run() writes the required terminal artifact."""

    tests_run = exp.default_tests_run()
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["session_count"] >= exp.MIN_SESSIONS
    assert artifact["decision_count"] == len(artifact["resource_accounted_decisions"])
    assert artifact["raw_episode_count"] == len(artifact["raw_episodes"])
    assert artifact["influence_share_sum_valid_rate"] == 1.0
    assert artifact["quality_delta_vs_baseline"] >= 0.0
    assert artifact["verifier_cost_delta_vs_baseline"] > 0.0
    assert artifact["wall_time_delta_vs_baseline"] > 0.0
    assert artifact["token_or_context_delta_vs_baseline"] > 0.0
    assert artifact["memory_delta_vs_baseline"] > 0.0
    assert artifact["unproductive_loop_reduction_rate"] > 0.0
    assert artifact["stale_memory_deflection_rate"] == 1.0
    assert artifact["poison_memory_deflection_rate"] == 1.0
    assert artifact["rollback_success_rate"] == 1.0
    assert artifact["no_weight_mutation"] is True
    assert artifact["resource_accounted_csl_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5408_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5408-6: checked-in result is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["resource_accounted_csl_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5408_blocked_artifact_reports_failed_readiness() -> None:
    """REQ-LEARN-5408-6: missing test evidence keeps the artifact blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["resource_accounted_csl_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5408_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5408-6: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=exp.default_tests_run())

    bad_missing = deepcopy(artifact)
    bad_missing.pop("decision_count")
    with pytest.raises(ValueError, match="decision_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["session_count"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_bool = deepcopy(artifact)
    bad_bool["no_weight_mutation"] = "true"
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["session_count"] = True
    with pytest.raises(ValueError, match="session_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["wall_time_delta_vs_baseline"] = {"value": 1.0}
    with pytest.raises(ValueError, match="wall_time_delta_vs_baseline"):
        exp.validate_artifact(bad_numeric)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_ready = deepcopy(artifact)
    bad_ready["resource_accounted_csl_ready"] = False
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.491"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_share = deepcopy(artifact)
    bad_share["influence_share_sum_valid_rate"] = 0.99
    with pytest.raises(ValueError, match="influence_share_sum_valid_rate"):
        exp.validate_artifact(bad_share)

    bad_quality = deepcopy(artifact)
    bad_quality["quality_delta_vs_baseline"] = -0.1
    with pytest.raises(ValueError, match="quality_delta_vs_baseline"):
        exp.validate_artifact(bad_quality)

    bad_wall = deepcopy(artifact)
    bad_wall["wall_time_delta_vs_baseline"] = 0.0
    with pytest.raises(ValueError, match="wall_time_delta_vs_baseline"):
        exp.validate_artifact(bad_wall)

    bad_stale = deepcopy(artifact)
    bad_stale["stale_memory_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="stale_memory_deflection_rate"):
        exp.validate_artifact(bad_stale)

    bad_poison = deepcopy(artifact)
    bad_poison["poison_memory_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="poison_memory_deflection_rate"):
        exp.validate_artifact(bad_poison)

    bad_rollback = deepcopy(artifact)
    bad_rollback["rollback_success_rate"] = 0.5
    with pytest.raises(ValueError, match="rollback_success_rate"):
        exp.validate_artifact(bad_rollback)

    bad_count = deepcopy(artifact)
    bad_count["decision_count"] += 1
    with pytest.raises(ValueError, match="decision_count"):
        exp.validate_artifact(bad_count)

    bad_raw_count = deepcopy(artifact)
    bad_raw_count["raw_episode_count"] += 1
    with pytest.raises(ValueError, match="raw_episode_count"):
        exp.validate_artifact(bad_raw_count)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    assert exp._rate(1, 0) == 0.0
    assert exp._json_ready(Path("results/example.json")) == "results/example.json"
