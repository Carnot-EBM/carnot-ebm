"""Tests for Exp 2982 FR-11 independent-metric utility gate.

Spec: REQ-LEARN-2982,
      SCENARIO-LEARN-2982,
      SCENARIO-LEARN-2982-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_independent_metric_utility_gate_v4 as exp


REQUIRED_FIELDS = {
    "honest_verdict",
    "continuous_self_learning_task",
    "fr11_independent_metrics_evaluated",
    "fr11_independent_self_learning_ready",
    "update_selection_metric",
    "independent_metrics",
    "frozen_baseline_metrics",
    "random_replay_metrics",
    "prior_fr11_metrics",
    "new_replay_metrics",
    "heldout_independent_delta_vs_random",
    "negative_control_delta",
    "forgetting_guard_passed",
    "leakage_audit",
    "no_identical_metric_flag",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ready_inputs(root: Path) -> None:
    _write_json(
        root,
        exp.EXP2969_REL_PATH,
        {
            "honest_verdict": "complete: non_tautological_self_learning_ready",
            "continuous_self_learning_task": True,
            "non_tautological_self_learning_ready": True,
            "forgetting_guard_passed": True,
            "update_rule": {
                "name": "non_tautological_verifier_weighted_utility_gate_v3",
                "candidate_policy": "train_replay_reward_weighted_with_guard_mass_preserved",
            },
            "frozen_replay_weights": {
                "logic_guard": 1.0,
                "threshold_policy": 1.0,
                "verified_pass": 1.0,
            },
            "random_replay_weights": {
                "extraction_repair": 1.0,
                "logic_guard": 1.0,
                "logic_repair": 1.0,
                "runtime_repair": 1.0,
                "syntax_repair": 1.0,
                "threshold_policy": 1.0,
                "verified_pass": 1.0,
            },
            "final_replay_weights": {
                "logic_guard": 0.142857142857,
                "logic_repair": 0.3250641574,
                "syntax_repair": 0.246364414029,
                "threshold_policy": 0.142857142857,
                "verified_pass": 0.142857142857,
            },
            "new_heldout_utility": 0.236013686912,
            "random_replay_heldout_utility": 0.142857142857,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )
    _write_json(
        root,
        exp.EXP2970_REL_PATH,
        {
            "honest_verdict": "complete: kan_forgetting_guard_ready",
            "kan_forgetting_guard_ready": True,
            "selected_policy": "per_knot_importance_update",
            "forgetting_delta_by_policy": {
                "adapter_style_update": 0.0,
                "eager_update": 0.75,
                "frozen": 0.0,
                "per_knot_importance_update": 0.0,
            },
            "high_dimensional_claim_allowed": False,
            "no_synthesis_claim": True,
            "no_analog_claim": True,
            "inference_substrate": "deterministic_wiring",
        },
    )
    _write_json(
        root,
        exp.EXP2973_REL_PATH,
        {
            "honest_verdict": "complete: matrix_v13_ready=true",
            "matrix_rows": [
                {
                    "row_id": "exp2969_non_tautological_fr11",
                    "headline_eligible": False,
                    "row_class": "flagged",
                }
            ],
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 14.75,
        tests_run=("focused-req-2982",),
    )


def test_req_learn_2982_spec_anchor_exists() -> None:
    """REQ-LEARN-2982: OpenSpec declares the independent-metric replay gate."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/self-learning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-LEARN-2982" in spec
    assert "SCENARIO-LEARN-2982" in spec
    assert "SCENARIO-LEARN-2982-BLOCKED" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="aggregation_and_deterministic_replay"' in spec


def test_scenario_learn_2982_writes_ready_independent_metric_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2982: independent metrics improve under guarded replay."""

    _write_ready_inputs(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == "complete: fr11_independent_self_learning_ready"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_independent_metrics_evaluated"] is True
    assert artifact["fr11_independent_self_learning_ready"] is True
    assert artifact["update_selection_metric"] == exp.UPDATE_SELECTION_METRIC
    assert artifact["inference_substrate"] == "aggregation_and_deterministic_replay"
    assert artifact["duration_s"] == pytest.approx(4.75)

    metric_names = {metric["name"] for metric in artifact["independent_metrics"]}
    assert {
        "pass_at_1",
        "solver_verified_accuracy",
        "syntax_failure_rate",
        "schema_failure_rate",
        "verifier_false_accept_rate",
    } <= metric_names
    assert exp.UPDATE_SELECTION_METRIC not in metric_names
    assert artifact["no_identical_metric_flag"] is True

    assert (
        artifact["new_replay_metrics"]["pass_at_1"] > artifact["random_replay_metrics"]["pass_at_1"]
    )
    assert (
        artifact["new_replay_metrics"]["solver_verified_accuracy"]
        > artifact["random_replay_metrics"]["solver_verified_accuracy"]
    )
    assert (
        artifact["new_replay_metrics"]["syntax_failure_rate"]
        < artifact["random_replay_metrics"]["syntax_failure_rate"]
    )
    assert (
        artifact["new_replay_metrics"]["verifier_false_accept_rate"]
        < artifact["random_replay_metrics"]["verifier_false_accept_rate"]
    )
    assert all(delta > 0.0 for delta in artifact["heldout_independent_delta_vs_random"].values())
    assert all(delta == pytest.approx(0.0) for delta in artifact["negative_control_delta"].values())
    assert artifact["forgetting_guard_passed"] is True

    audit = artifact["leakage_audit"]
    assert audit["selection_metric_reused_as_reported_metric"] is False
    assert audit["negative_control_improved"] is False
    assert audit["deterministic_reset_controls"] == {
        "frozen_baseline": True,
        "random_replay": True,
        "negative_control": True,
    }
    assert audit["kan_evidence_scope"] == "bounded_memory_forgetting_only"
    assert audit["kan_acceleration_claimed"] is False
    assert audit["prompt_requested_exp2970_short_path_present"] is False
    assert audit["source_sha256"]["exp2969"] == _sha256(tmp_path / exp.EXP2969_REL_PATH)
    assert artifact["source_artifacts"][1]["path"] == exp.EXP2970_REL_PATH.as_posix()


def test_req_learn_2982_directional_metrics_and_controls() -> None:
    """REQ-LEARN-2982-3/4: metric deltas are directional and controls stay inert."""

    random_metrics = exp.evaluate_policy_metrics(exp.random_replay_weights())
    new_metrics = exp.evaluate_policy_metrics(exp.independent_metric_replay_weights())
    negative_metrics = exp.evaluate_policy_metrics(exp.negative_control_weights())

    deltas = exp.directional_delta(new_metrics, random_metrics)
    negative = exp.directional_delta(negative_metrics, random_metrics)

    assert all(value > 0.0 for value in deltas.values())
    assert exp.metrics_improved(deltas) is True
    assert all(value == pytest.approx(0.0) for value in negative.values())
    assert exp.negative_control_improved(negative) is False
    assert exp.no_identical_metric_flag(exp.UPDATE_SELECTION_METRIC, exp.INDEPENDENT_METRICS)

    worse = dict(deltas, pass_at_1=-0.01)
    assert exp.metrics_improved(worse) is False
    assert exp.negative_control_improved(worse) is True

    with pytest.raises(ValueError, match="positive"):
        exp.normalize_weights({"syntax_repair": 0.0})


def test_scenario_learn_2982_blocked_artifacts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2982-BLOCKED: missing or unready evidence blocks readiness."""

    missing = exp.build_artifact(_config(tmp_path))
    assert missing["honest_verdict"] == "blocked_missing_exp2969_ready_artifact"
    assert missing["fr11_independent_metrics_evaluated"] is False
    assert missing["fr11_independent_self_learning_ready"] is False
    assert missing["forgetting_guard_passed"] is False
    assert REQUIRED_FIELDS <= set(missing)

    _write_ready_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp.EXP2969_REL_PATH,
        {"non_tautological_self_learning_ready": False},
    )
    not_ready = exp.build_artifact(_config(tmp_path))
    assert not_ready["honest_verdict"] == "blocked_exp2969_not_ready"

    _write_ready_inputs(tmp_path)
    exp2969_payload = json.loads((tmp_path / exp.EXP2969_REL_PATH).read_text(encoding="utf-8"))
    exp2969_payload["forgetting_guard_passed"] = False
    _write_json(tmp_path, exp.EXP2969_REL_PATH, exp2969_payload)
    exp2969_guard_blocked = exp.build_artifact(_config(tmp_path))
    assert exp2969_guard_blocked["honest_verdict"] == ("blocked_exp2969_forgetting_guard_not_ready")

    _write_ready_inputs(tmp_path)
    (tmp_path / exp.EXP2970_REL_PATH).unlink()
    exp2970_missing = exp.build_artifact(_config(tmp_path))
    assert exp2970_missing["honest_verdict"] == (
        "blocked_missing_exp2970_forgetting_guard_artifact"
    )

    _write_ready_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp.EXP2970_REL_PATH,
        {"kan_forgetting_guard_ready": False},
    )
    kan_not_ready = exp.build_artifact(_config(tmp_path))
    assert kan_not_ready["honest_verdict"] == "blocked_exp2970_forgetting_guard_not_ready"
    assert kan_not_ready["leakage_audit"]["used_exp2970_memory_audit_path"] == (
        exp.EXP2970_REL_PATH.as_posix()
    )

    _write_ready_inputs(tmp_path)
    exp2970_payload = json.loads((tmp_path / exp.EXP2970_REL_PATH).read_text(encoding="utf-8"))
    exp2970_payload["high_dimensional_claim_allowed"] = True
    _write_json(tmp_path, exp.EXP2970_REL_PATH, exp2970_payload)
    claim_blocked = exp.build_artifact(_config(tmp_path))
    assert claim_blocked["honest_verdict"] == "blocked_exp2970_claim_boundary"

    malformed = tmp_path / exp.EXP2969_REL_PATH
    malformed.write_text("{", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}


def test_req_learn_2982_validation_defends_schema_and_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2982-2/5: schema drift and KAN overclaims are rejected."""

    _write_ready_inputs(tmp_path)
    artifact = exp.build_artifact(_config(tmp_path))

    assert exp.validate_artifact(artifact) == artifact

    incomplete = dict(artifact)
    incomplete.pop("new_replay_metrics")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(incomplete)

    bad_substrate = dict(artifact, inference_substrate="live_llm_inference")
    with pytest.raises(ValueError, match="substrate"):
        exp.validate_artifact(bad_substrate)

    bad_metric = dict(artifact, no_identical_metric_flag=False)
    with pytest.raises(ValueError, match="selection metric"):
        exp.validate_artifact(bad_metric)

    bad_claim = json.loads(json.dumps(artifact))
    bad_claim["leakage_audit"]["kan_acceleration_claimed"] = True
    with pytest.raises(ValueError, match="KAN acceleration"):
        exp.validate_artifact(bad_claim)

    assert exp._weights_from_payload({}, "missing", {"syntax_repair": 1.0}) == {
        "syntax_repair": 1.0
    }
    assert exp._weights_from_payload(
        {"weights": {"syntax_repair": "bad"}}, "weights", {"logic_guard": 1.0}
    ) == {"logic_guard": 1.0}
    assert exp._matrix_v13_exp2969_row({"matrix_rows": [{"row_id": "other"}]}) == {}

    monkeypatch.setattr(exp, "write_artifact", lambda: {})
    assert exp.main() == 0
