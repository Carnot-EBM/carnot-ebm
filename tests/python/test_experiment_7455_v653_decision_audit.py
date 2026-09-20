"""Tests for REQ-REPORT-7455 and SCENARIO-REPORT-7455-*.

Spec: REQ-REPORT-7455: Audit source conditioning and continuous learning independently
Spec: SCENARIO-REPORT-7455-INDEPENDENT: One branch cannot hide its peer
Spec: SCENARIO-REPORT-7455-ONLINE-REPLAY: Delayed updates reconstruct from saved expert predictions
Spec: SCENARIO-REPORT-7455-MUTATIONS: Evidence corruption fails closed
Spec: SCENARIO-REPORT-7455-RETIREMENT: Repeated valid nulls mark mixture retirement
Spec: SCENARIO-REPORT-7455-ARTIFACT: Exact readers control atomic publication
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7455_v653_decision_audit as audit


def _make_prediction_event(
    group_id: str,
    *,
    arm: str = "learned_mixture",
    order: str = "hash_order",
    delay: int = 0,
    seed: int = 65201,
    seq: int = 0,
    probs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a deterministic prediction event fixture."""
    expert_probs = probs or {
        "adaptive_gibbs": 0.2,
        "adaptive_spline": 0.3,
        "frozen_gibbs": 0.2,
        "frozen_spline": 0.3,
    }
    mix_weights = weights or {
        "adaptive_gibbs": 0.25,
        "adaptive_spline": 0.25,
        "frozen_gibbs": 0.25,
        "frozen_spline": 0.25,
    }
    row: dict[str, Any] = {
        "schema": "carnot.exp7454.prediction_event.v1",
        "row_type": "prediction_event",
        "group_id": group_id,
        "arm": arm,
        "ordering": order,
        "delay": delay,
        "seed": seed,
        "ledger_sequence": seq,
        "request_order": seq,
        "expert_probabilities": deepcopy(expert_probs),
        "mixture_weights": deepcopy(mix_weights),
        "mixture_probability": math.fsum(
            expert_probs[k] * mix_weights[k] for k in audit.EXPERT_NAMES
        ),
        "proposed_action": "escalate",
        "deployed_action": "escalate",
        "shadow_only": True,
        "certified_safe": False,
        "revealed": True,
        "pre_feedback_state_hash": f"sha256:pre_state_{seq}",
    }
    row["event_hash"] = audit.canonical_hash({k: v for k, v in row.items() if k != "event_hash"})
    return row


def _make_feedback_event(
    pred: dict[str, Any],
    *,
    true_label: int = 1,
    seq: int = 1,
    old_log_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a deterministic feedback event fixture corresponding to pred."""
    probs = pred["expert_probabilities"]
    old_lw = old_log_weights or {name: math.log(0.25) for name in audit.EXPERT_NAMES}
    update = audit.independent_scalar_update(old_lw, probs, true_label, eta=1.0, fixed_share=0.01)
    row: dict[str, Any] = {
        "schema": "carnot.exp7454.feedback_event.v1",
        "row_type": "feedback_event",
        "group_id": pred["group_id"],
        "arm": pred["arm"],
        "ordering": pred["ordering"],
        "delay": pred["delay"],
        "seed": pred["seed"],
        "ledger_sequence": seq,
        "request_order": pred["request_order"],
        "reveal_order": pred["request_order"] + pred["delay"],
        "prediction_index": pred["request_order"],
        "prediction_event_hash": pred["event_hash"],
        "true_label": true_label,
        "feedback_label": true_label,
        "revoked": False,
        "probability_source": "immutable_prediction_event",
        "per_expert_loss": update["losses"],
        "old_log_weights": old_lw,
        "new_log_weights": update["new_log_weights"],
        "normalizer": {
            "maximum": update["maximum"],
            "sum_exp": update["sum_exp"],
        },
        "numeric_update": {
            "penalized_log_weights": update["penalized_log_weights"],
            "posterior_before_share": update["posterior_before_share"],
        },
        "parent_state_hash": f"sha256:pre_state_{pred['request_order']}",
        "child_state_hash": f"sha256:post_state_{seq}",
        "update_applied": True,
    }
    row["event_hash"] = audit.canonical_hash({k: v for k, v in row.items() if k != "event_hash"})
    return row


def test_bernoulli_log_loss_and_validation() -> None:
    """Spec: REQ-REPORT-7455: Loss computation and domain bounds."""
    loss_0 = audit.bernoulli_log_loss(0, 0.1)
    assert math.isclose(loss_0, -math.log(0.9), abs_tol=1e-7)
    loss_1 = audit.bernoulli_log_loss(1, 0.8)
    assert math.isclose(loss_1, -math.log(0.8), abs_tol=1e-7)

    with pytest.raises(ValueError, match="binary label"):
        audit.bernoulli_log_loss(2, 0.5)
    with pytest.raises(ValueError, match="binary label"):
        audit.bernoulli_log_loss(True, 0.5)  # type: ignore
    with pytest.raises(ValueError, match="finite probability"):
        audit.bernoulli_log_loss(1, -0.1)
    with pytest.raises(ValueError, match="finite probability"):
        audit.bernoulli_log_loss(1, 1.5)


def test_independent_scalar_update_fixed_share() -> None:
    """Spec: SCENARIO-REPORT-7455-ONLINE-REPLAY: Recompute fixed-share update."""
    old_log_weights = {name: math.log(0.25) for name in audit.EXPERT_NAMES}
    probs = {
        "adaptive_gibbs": 0.2,
        "adaptive_spline": 0.8,
        "frozen_gibbs": 0.2,
        "frozen_spline": 0.8,
    }
    update = audit.independent_scalar_update(
        old_log_weights, probs, label=1, eta=1.0, fixed_share=0.01
    )
    # Check that update contains all expected keys
    assert "losses" in update
    assert "penalized_log_weights" in update
    assert "maximum" in update
    assert "sum_exp" in update
    assert "posterior_before_share" in update
    assert "new_log_weights" in update

    # The probabilities under label=1 gave lower loss to spline (0.8) than gibbs (0.2).
    # So spline's new log weight should be greater than gibbs's new log weight.
    assert (
        update["new_log_weights"]["adaptive_spline"] > update["new_log_weights"]["adaptive_gibbs"]
    )

    # Weights in exp space must sum to 1.0
    new_weights = [math.exp(update["new_log_weights"][name]) for name in audit.EXPERT_NAMES]
    assert math.isclose(sum(new_weights), 1.0, abs_tol=1e-9)


def test_initial_audit_state() -> None:
    """Spec: SCENARIO-REPORT-7455-ONLINE-REPLAY: Initial state construction."""
    state_mix = audit.initial_audit_state("learned_mixture")
    assert state_mix["arm"] == "learned_mixture"
    assert state_mix["feedback_count"] == 0
    for name in audit.EXPERT_NAMES:
        assert math.isclose(state_mix["log_weights"][name], math.log(0.25), abs_tol=1e-9)

    state_fs = audit.initial_audit_state("frozen_spline")
    assert state_fs["log_weights"]["frozen_spline"] == 0.0  # log(1.0)
    assert state_fs["log_weights"]["adaptive_gibbs"] == -math.inf


def test_replay_event_stream_matches_exact_updates() -> None:
    """Spec: SCENARIO-REPORT-7455-ONLINE-REPLAY: Replay stream matches exactly."""
    pred = _make_prediction_event("group-0", seq=0)
    fb = _make_feedback_event(pred, true_label=1, seq=1)

    result = audit.replay_event_stream([pred], [fb])
    assert result["updates_replayed"] == 1
    assert result["all_updates_matched"] is True
    assert result["max_loss_gap"] < 1e-9
    assert result["max_weight_gap"] < 1e-9
    assert result["no_feedback_immutable"] is True


def test_scenario_independent_peer_not_hidden() -> None:
    """Spec: SCENARIO-REPORT-7455-INDEPENDENT: One branch cannot hide its peer."""
    # Build a simulated pre-gated static slot and valid online slot
    static_slot = {
        "task_id": "exp7453-energy-calibration",
        "declared_path": "results/experiment_7453_v653_energy_calibration.json",
        "source_path": "results/experiment_7453_energy_calibration.json",
        "source_kind": "structured_pre_gate",
        "authenticated": True,
        "available": False,
        "valid": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "blocked_upstream": "exp7452-source-embeddings",
            "blocked_path": "results/experiment_7452_v653_source_embeddings.json",
            "blocked_check": "exp7452-source-embeddings.embedding_capture_ready_score",
            "blocked_field": "embedding_capture_ready_score",
            "blocked_expected": 1,
            "blocked_observed": 0,
        },
    }
    online_slot = {
        "task_id": "exp7454-continuous-learning",
        "declared_path": "results/experiment_7454_v653_continuous_learning.json",
        "source_path": "results/experiment_7454_v653_continuous_learning.json",
        "source_kind": "completed_artifact",
        "authenticated": True,
        "available": True,
        "valid": True,
        "verdict_class": "null",
        "honest_verdict": "complete_null_insufficient_online_benefit",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "required_checks_passed": True,
        },
    }
    checks = [
        {"check": "static:pre_gate", "passed": True, "branch": "static"},
        {"check": "online:completed", "passed": True, "branch": "online"},
    ]
    static_audit = audit.audit_static_branch(Path("."), static_slot, checks)
    online_audit = audit.audit_online_branch(Path("."), online_slot, checks, skip_rows=True)

    assert static_audit["available"] is False
    assert static_audit["verdict_class"] == "blocked"
    assert online_audit["available"] is True
    assert online_audit["verdict_class"] == "null"

    terminal = audit.classify_terminal([static_audit, online_audit], validation_passed=True)
    assert terminal["verdict_class"] == "blocked"
    assert "blocked_static_upstream_science" in terminal["honest_verdict"]


def test_scenario_mutations_rejected() -> None:
    """Spec: SCENARIO-REPORT-7455-MUTATIONS: Evidence corruption fails closed."""
    pred = _make_prediction_event("group-0", seq=0)
    fb = _make_feedback_event(pred, true_label=1, seq=1)

    # 1. Mutate prediction probability
    bad_pred = deepcopy(pred)
    bad_pred["expert_probabilities"]["adaptive_gibbs"] = 0.99
    errors1 = audit.online_integrity_errors([bad_pred], [fb])
    assert "prediction_probability_mutation" in errors1 or "event_hash_mismatch" in errors1

    # 2. Shuffle event order: feedback before prediction
    errors2 = audit.online_integrity_errors([pred], [fb], event_order_override=[fb, pred])
    assert "causal_order_violation" in errors2

    # 3. Drop feedback event
    fb2 = _make_feedback_event(pred, true_label=1, seq=2)
    fb2["parent_state_hash"] = fb["child_state_hash"]
    fb2["event_hash"] = audit.canonical_hash({k: v for k, v in fb2.items() if k != "event_hash"})
    errors3 = audit.online_integrity_errors([pred], [fb2])  # dropped fb
    assert "state_chain_break" in errors3 or "missing_feedback_parent" in errors3

    # 4. Leak test label into features
    bad_pred_label = deepcopy(pred)
    bad_pred_label["label"] = 1
    errors4 = audit.online_integrity_errors([bad_pred_label], [fb])
    assert "label_isolation_leak" in errors4

    # 5. Duplicate source group
    pred_dup = _make_prediction_event("group-0", seq=2)
    errors5 = audit.online_integrity_errors([pred, pred_dup], [fb])
    assert "duplicate_source_group" in errors5

    # 6. Replace vector with length-only
    bad_pred_vec = deepcopy(pred)
    bad_pred_vec["expert_probabilities"] = 4  # length scalar instead of vector mapping
    errors6 = audit.online_integrity_errors([bad_pred_vec], [fb])
    assert "representation_vector_invalid" in errors6

    # Test run_mutation_controls produces expected mutation rows
    mutation_rows = audit.run_mutation_controls([pred], [fb])
    assert len(mutation_rows) >= 6
    assert all(r["passed"] is True for r in mutation_rows)
    assert all(r["observed"] == "rejected" for r in mutation_rows)


def test_scenario_retirement_on_repeated_nulls() -> None:
    """Spec: SCENARIO-REPORT-7455-RETIREMENT: Repeated valid nulls mark mixture retirement."""
    continuation = audit.continuation_rows(
        static_verdict="blocked",
        online_verdict="null",
        online_retired=True,
    )
    assert len(continuation) == 2
    online_row = next(r for r in continuation if r["branch"] == "online")
    assert online_row["status"] == "retired"
    assert "repeated valid null" in online_row["reason"]
    assert online_row["capstone_action"] == "retire_mechanism"


def test_scenario_artifact_cold_replay_and_validation() -> None:
    """Spec: SCENARIO-REPORT-7455-ARTIFACT: Exact readers control atomic publication."""
    artifact = audit.build_artifact_for_test()
    errors = audit.validate_artifact(artifact, verify_source_bytes=False)
    assert errors == []

    # Check top-level required fields
    for field in audit.REQUIRED_FIELDS:
        assert field in artifact, f"Missing required field: {field}"

    # Check principles match top-level fields
    for field in audit.REQUIRED_FIELDS:
        assert field in artifact["field_principles"], f"Missing field principle: {field}"

    # Corrupt checksum
    corrupted = deepcopy(artifact)
    corrupted["reproducibility_checksum"] = "bad_checksum"
    errs = audit.validate_artifact(corrupted, verify_source_bytes=False)
    assert "reproducibility_checksum_mismatch" in errs


def test_evidence_slot_loading_and_branch_audits(tmp_path: Path) -> None:
    """Spec: SCENARIO-REPORT-7455-INDEPENDENT: Evidence slot loading and branch audits."""
    root = audit.REPO_ROOT
    static_slot = audit.load_evidence_slot(
        root,
        "exp7453-energy-calibration",
        Path("results/experiment_7453_v653_energy_calibration.json"),
        fallback_path=Path("results/experiment_7453_energy_calibration.json"),
    )
    assert static_slot["authenticated"] is True
    assert static_slot["available"] is False
    assert static_slot["source_kind"] == "structured_pre_gate"

    online_slot = audit.load_evidence_slot(
        root,
        "exp7454-continuous-learning",
        Path("results/experiment_7454_v653_continuous_learning.json"),
        fallback_path=Path("results/experiment_7454_continuous_learning.json"),
    )
    assert online_slot["authenticated"] is True
    assert online_slot["available"] is True
    assert online_slot["source_kind"] == "completed_artifact"

    # Missing file
    missing_slot = audit.load_evidence_slot(root, "exp-none", Path("results/nonexistent_123.json"))
    assert missing_slot["authenticated"] is False
    assert missing_slot["available"] is False

    missing_fallback = audit.load_evidence_slot(
        root,
        "exp-none",
        Path("results/nonexistent_123.json"),
        fallback_path=Path("results/nonexistent_456.json"),
    )
    assert missing_fallback["authenticated"] is False

    # Audit branches with real repository data
    checks: list[dict[str, Any]] = []
    static_audit = audit.audit_static_branch(root, static_slot, checks)
    assert static_audit["verdict_class"] == "blocked"

    online_audit = audit.audit_online_branch(root, online_slot, checks)
    assert online_audit["verdict_class"] == "null"
    assert online_audit["raw_row_count"] > 0

    # Test audit_online_branch when unavailable
    unavail_slot = deepcopy(online_slot)
    unavail_slot["available"] = False
    unavail_audit = audit.audit_online_branch(root, unavail_slot, checks)
    assert unavail_audit["verdict_class"] == "blocked"


def test_validate_events_error_paths() -> None:
    """Spec: SCENARIO-REPORT-7455-MUTATIONS: Validation errors on corrupted events."""
    pred = _make_prediction_event("g1")
    fb = _make_feedback_event(pred)

    # Missing expert in prediction
    bad_pred1 = deepcopy(pred)
    del bad_pred1["expert_probabilities"]["adaptive_gibbs"]
    with pytest.raises(ValueError, match="prediction_experts_invalid"):
        audit.validate_prediction_event(bad_pred1)

    # Weights don't sum to 1
    bad_pred2 = deepcopy(pred)
    bad_pred2["mixture_weights"]["adaptive_gibbs"] = 0.9
    with pytest.raises(ValueError, match="prediction_weights_invalid"):
        audit.validate_prediction_event(bad_pred2)

    # Invalid weights mapping
    bad_pred3 = deepcopy(pred)
    bad_pred3["mixture_weights"] = "bad"
    with pytest.raises(ValueError, match="prediction_weights_invalid"):
        audit.validate_prediction_event(bad_pred3)

    # Hash mismatch in prediction
    bad_pred4 = deepcopy(pred)
    bad_pred4["event_hash"] = "sha256:wrong_hash"
    with pytest.raises(ValueError, match="prediction_event_hash_mismatch"):
        audit.validate_prediction_event(bad_pred4)

    # Feedback invalid true_label
    bad_fb1 = deepcopy(fb)
    bad_fb1["true_label"] = 2
    with pytest.raises(ValueError, match="feedback_event_true_label_invalid"):
        audit.validate_feedback_event(bad_fb1)

    # Feedback missing prediction hash
    bad_fb2 = deepcopy(fb)
    bad_fb2["prediction_event_hash"] = ""
    with pytest.raises(ValueError, match="feedback_missing_prediction_hash"):
        audit.validate_feedback_event(bad_fb2)

    # Feedback hash mismatch
    bad_fb3 = deepcopy(fb)
    bad_fb3["event_hash"] = "sha256:wrong_fb_hash"
    with pytest.raises(ValueError, match="feedback_event_hash_mismatch"):
        audit.validate_feedback_event(bad_fb3)


def test_validate_artifact_disqualification_and_errors() -> None:
    """Spec: SCENARIO-REPORT-7455-ARTIFACT: Strict artifact validation error paths."""
    base = audit.build_artifact_for_test()

    for field in ("schema", "experiment_id", "milestone", "run_date"):
        bad = deepcopy(base)
        bad[field] = "wrong"
        assert len(audit.validate_artifact(bad, verify_source_bytes=False)) > 0

    bad_model = deepcopy(base)
    bad_model["MODEL_SPECS"] = ["some_model"]
    assert "current_model_declaration_invalid" in audit.validate_artifact(
        bad_model, verify_source_bytes=False
    )

    bad_counts = deepcopy(base)
    bad_counts["invocation_counts"]["model_loads_attempted"] = 1
    assert "current_invocation_counts_invalid" in audit.validate_artifact(
        bad_counts, verify_source_bytes=False
    )

    bad_sub = deepcopy(base)
    bad_sub["inference_substrate_class"] = "cuda"
    assert "inference_substrate_class_invalid" in audit.validate_artifact(
        bad_sub, verify_source_bytes=False
    )

    bad_venue = deepcopy(base)
    bad_venue["execution_venue"] = "remote"
    assert "execution_venue_invalid" in audit.validate_artifact(
        bad_venue, verify_source_bytes=False
    )

    bad_promo = deepcopy(base)
    bad_promo["promotion_score"] = 1
    assert "promotion_score_invalid" in audit.validate_artifact(
        bad_promo, verify_source_bytes=False
    )

    bad_score = deepcopy(base)
    bad_score["decision_audit_complete_score"] = 0
    assert "decision_audit_complete_score_invalid" in audit.validate_artifact(
        bad_score, verify_source_bytes=False
    )

    bad_mutations = deepcopy(base)
    bad_mutations["mutation_rows"] = []
    assert "mutation_controls_invalid" in audit.validate_artifact(
        bad_mutations, verify_source_bytes=False
    )

    bad_branch = deepcopy(base)
    bad_branch["online_audit"]["valid"] = False
    assert "terminal_classification_mismatch" in audit.validate_artifact(
        bad_branch, verify_source_bytes=False
    )


def test_cold_and_independent_replay(tmp_path: Path) -> None:
    """Spec: SCENARIO-REPORT-7455-ARTIFACT: Cold and independent replay interfaces."""
    artifact = audit.build_artifact_for_test()
    art_path = tmp_path / "candidate.json"
    art_path.write_text(json.dumps(artifact), encoding="utf-8")

    errs = audit.cold_replay(art_path, verify_source_bytes=False)
    assert errs == []

    errs_ind = audit.independent_replay(art_path)
    assert errs_ind == []

    bad_file = tmp_path / "nonexistent.json"
    assert audit.cold_replay(bad_file) == ["candidate_artifact_unreadable"]


def test_cli_main_and_validation_commands(tmp_path: Path) -> None:
    """Spec: SCENARIO-REPORT-7455-ARTIFACT: Command line interface and validation plan."""
    artifact = audit.build_artifact_for_test()
    art_path = tmp_path / "candidate.json"
    art_path.write_text(json.dumps(artifact), encoding="utf-8")

    # Test build_validation_commands
    cmds = audit.build_validation_commands(audit.REPO_ROOT, tmp_path)
    assert len(cmds) > 0

    # Test main --cold-replay
    ret_cold = audit.main(["--cold-replay", str(art_path), "--skip-source-bytes"])
    assert ret_cold == 0

    # Test main --independent-reduce
    ret_ind = audit.main(["--independent-reduce", str(art_path), "--skip-source-bytes"])
    assert ret_ind == 0

    # Test main missing --date
    with pytest.raises(SystemExit):
        audit.main([])
