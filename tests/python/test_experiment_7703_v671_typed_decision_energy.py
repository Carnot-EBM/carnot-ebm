"""Checks for REQ-REPORT-7703 and REQ-ENERGY-7703."""

from __future__ import annotations

import json

import numpy as np
import pytest

from carnot.experiment_7672_v669_bound_relations import fixture_cases
from carnot.reporting import typed_decision_energy as energy


def test_7703_features_keep_typed_unknown_and_controls() -> None:
    """SCENARIO-REPORT-7703-SEAL: source controls keep the answer fixed."""
    case = fixture_cases()[2]
    row = {
        "complete_source": case["source"],
        "complete_question": "Where is this frame?",
        "complete_answer": case["answer"] + " Extra claim.",
    }
    original = energy.feature_view(row)
    erased = energy.feature_view(row, source_override="")
    assert original["counts"]["path_line_supported"] >= 1
    assert erased["counts"]["path_line_supported"] == 0
    assert original["counts"]["residual_unknown_bytes"] > 0
    assert erased["counts"]["residual_unknown_bytes"] > 0
    assert len(original["vector"]) == len(energy.FEATURE_ORDER)
    with pytest.raises(ValueError, match="label_or_field"):
        energy.feature_view({**row, "label": 1})


def test_7703_energy_normalizes_and_reloads() -> None:
    """SCENARIO-ENERGY-7703-RELOAD: exact probabilities and actions survive JSON."""
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0], [0.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    for kind in ("prior", "logistic", "mlp"):
        head = energy.fit_head(x, y, kind, seed=7703, steps=25, ridge=0.01)
        saved = json.loads(json.dumps(head))
        for row in x:
            first = energy.probabilities(row, head)
            second = energy.probabilities(row, saved)
            assert sum(first) == pytest.approx(1.0, abs=1e-12)
            assert all(0.0 < p < 1.0 for p in first)
            assert first == pytest.approx(second, abs=1e-6)
            assert energy.action(first[1], (0.3, 0.7)) == energy.action(second[1], (0.3, 0.7))
        assert head["parameter_count"] <= 4096
        assert head["steps"] <= 500


def test_7703_no_information_retains_prior() -> None:
    """SCENARIO-ENERGY-7703-NO-INFORMATION: zero-variance inputs stay valid."""
    x = np.zeros((6, 3))
    y = np.array([0, 1, 0, 1, 0, 0])
    head = energy.fit_head(x, y, "logistic", seed=3, steps=20, ridge=0.1)
    assert head["source_information_learned"] is False
    assert energy.probabilities(x[0], head)[1] == pytest.approx(2 / 6, abs=0.03)
    assert energy.feature_information(x, np.zeros(6))["checked_coverage"] == 0


def test_7703_policy_cost_and_online_limits() -> None:
    """REQ-REPORT-7703: policy labels only freeze decisions and protocol limits."""
    thresholds, options = energy.freeze_policy([0.05, 0.95, 0.48], [0, 1, 1])
    assert options
    assert energy.action(0.05, thresholds) in {"accept", "escalate", "reject"}
    assert energy.decision_cost("escalate", 0) == pytest.approx(0.2)
    assert energy.decision_cost("accept", 1) == pytest.approx(1)
    protocol = energy.freeze_online_protocol([0.1, 0.2, 0.3, 0.4])
    assert len(protocol["primitives"]) == 8
    assert len(protocol["pairs"]) == 28
    assert protocol["max_proposals_per_arm"] == 6
    assert protocol["max_gradient_steps_per_proposal"] == 50
    assert protocol["scheduler"]["percentile"] == 75


def test_7703_preflight_reports_exact_gate(tmp_path) -> None:
    """SCENARIO-REPORT-7703-CUSTODY: missing producer stops before labels."""
    from carnot.experiment_7703_v671_typed_decision_energy import preflight

    checks, hashes = preflight(tmp_path)
    assert any(not check["passed"] for check in checks)
    assert any(
        check["upstream_id"] == "exp7700-record-span-protocol"
        and check["field"] == "exists"
        and check["observed"] is False
        for check in checks
    )
    assert hashes["producers"] == {}


def test_7703_rejects_invalid_energy_and_policy_inputs() -> None:
    """REQ-ENERGY-7703: malformed training, heads and actions fail closed."""
    x = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="head_kind_invalid"):
        energy.fit_head(x, y, "invalid", seed=1, steps=1, ridge=0)
    with pytest.raises(ValueError, match="training_budget_invalid"):
        energy.fit_head(x, y, "mlp", seed=1, steps=501, ridge=0)
    with pytest.raises(ValueError, match="training_rows_invalid"):
        energy.fit_head(x, np.array([2, 0]), "mlp", seed=1, steps=1, ridge=0)
    with pytest.raises(ValueError, match="features_not_finite"):
        energy.fit_head(np.array([[np.nan], [0.0]]), y, "mlp", seed=1, steps=1, ridge=0)
    with pytest.raises(ValueError, match="parameter_cap_exceeded"):
        energy.fit_head(
            np.vstack((np.zeros(600), np.ones(600))), y, "mlp", seed=1, steps=0, ridge=0
        )
    head = energy.fit_head(x, y, "logistic", seed=1, steps=1, ridge=0)
    with pytest.raises(ValueError, match="head_schema_invalid"):
        energy.probabilities(x[0], {**head, "schema": "wrong"})
    with pytest.raises(ValueError, match="feature_shape_invalid"):
        energy.probabilities(np.zeros(2), head)
    with pytest.raises(ValueError, match="head_kind_invalid"):
        energy.probabilities(x[0], {**head, "kind": "wrong"})
    with pytest.raises(ValueError, match="energy_not_finite"):
        energy.probabilities(x[1], {**head, "weights": [float("inf")]})
    with pytest.raises(ValueError, match="thresholds_invalid"):
        energy.action(0.5, (0.9, 0.1))
    with pytest.raises(ValueError, match="decision_or_label_invalid"):
        energy.decision_cost("invalid", 0)
    with pytest.raises(ValueError, match="policy_rows_invalid"):
        energy.freeze_policy([], [])
    with pytest.raises(ValueError, match="replay_losses_invalid"):
        energy.freeze_online_protocol([])


def test_7703_real_cohort_freeze_and_cold_replay(tmp_path, monkeypatch) -> None:
    """SCENARIO-REPORT-7703-TERMINAL: replay all 400 public families."""
    import time

    from carnot import experiment_7703_v671_typed_decision_energy as experiment
    from carnot.reporting import experiment_7303_validation_scope as validation
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    root = experiment.ROOT
    checks, hashes = experiment.preflight(root)
    assert all(check["passed"] for check in checks)
    protocol = json.loads((root / experiment.COHORT / "protocol.json").read_text())
    started = time.monotonic()
    features = experiment.build_feature_rows(root, protocol, tmp_path, started)
    assert len(features) == 1200
    assert len({row["unit_id"] for row in features}) == 400
    assert all("label" not in row for row in features)
    fit = experiment._labels(root, protocol, "fit")
    tune = experiment._labels(root, protocol, "tune")
    bundle, settings = experiment.fit_families(features, fit, tune, tmp_path, started)
    policy, online, _ = experiment.freeze_decisions(
        root, protocol, features, bundle, fit, tune, tmp_path
    )
    assert len(settings) == 50
    assert online["scheduler"]["calibration_rows"] == 168
    scored = experiment.score_rows(features, bundle, tuple(policy["thresholds"]))
    trained = experiment.training_rows(
        features,
        bundle,
        {"fit": fit, "tune": tune, "policy": experiment._labels(root, protocol, "policy")},
    )
    assert len(trained) == 50 * 208
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in validation.REQUIRED_CHECK_NAMES
    ]
    monkeypatch.setattr(experiment, "RAW", tmp_path)
    artifact = experiment.build_artifact(
        "20260926",
        started,
        checks,
        hashes,
        scored,
        bundle,
        trained,
        {
            "zero_coverage_groups": sum(
                row["censored"] for row in features if row["arm"] == "original_source"
            )
        },
        receipts,
        [],
    )
    artifact["frozen_output_hashes"] = {
        str(tmp_path / name): sha256_file(tmp_path / name)
        for name in ("heads.json", "policy.json", "online_protocol.json")
    }
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, artifact)
    reduced = experiment.cold_reduce(candidate, root)
    assert reduced["passed"] and reduced["paired_rows"] == 1200
    assert artifact["decision_energy_ready_score"] == 1

    def reject_mutation(changed: dict, reason: str) -> None:
        atomic_json(candidate, changed)
        with pytest.raises(ValueError, match=reason):
            experiment.cold_reduce(candidate, root)

    changed = json.loads(json.dumps(artifact))
    first = next(iter(changed["source_artifact_hashes"]["producers"]))
    changed["source_artifact_hashes"]["producers"][first] = "sha256:wrong"
    reject_mutation(changed, "source_hash_mismatch")
    changed = json.loads(json.dumps(artifact))
    first = next(iter(changed["frozen_output_hashes"]))
    changed["frozen_output_hashes"][first] = "sha256:wrong"
    reject_mutation(changed, "frozen_output_hash_mismatch")
    changed = json.loads(json.dumps(artifact))
    changed["rows"].append(changed["rows"][0])
    reject_mutation(changed, "duplicate_unit_arm")
    changed = json.loads(json.dumps(artifact))
    changed["rows"][0]["vector"][0] += 1
    reject_mutation(changed, "feature_reduction_mismatch")
    changed = json.loads(json.dumps(artifact))
    changed["rows"][0]["typed_action"] = "wrong"
    reject_mutation(changed, "decision_reduction_mismatch")
    changed = json.loads(json.dumps(artifact))
    changed["rows"].append({**changed["rows"][0], "unit_id": "extra"})
    reject_mutation(changed, "row_count_mismatch")

    bad_protocol = json.loads(json.dumps(protocol))
    bad_protocol["roles"]["fit"]["families"] = []
    with pytest.raises(ValueError, match="role_roster_mismatch"):
        experiment.build_feature_rows(root, bad_protocol, tmp_path, started)
    bad_protocol = json.loads(json.dumps(protocol))
    bad_protocol["evaluator_stores"]["fit"]["path"] = "fit_model_inputs.jsonl"
    with pytest.raises(ValueError, match="label_roster_mismatch"):
        experiment._labels(root, bad_protocol, "fit")
    bad_labels = tmp_path / "bad-labels.jsonl"
    bad_labels.write_text('{"family_id":"one","label":1,"role":"fit"}\n')
    bad_protocol["evaluator_stores"]["fit"]["path"] = str(bad_labels)
    with pytest.raises(ValueError, match="label_roster_mismatch"):
        experiment._labels(root, bad_protocol, "fit")
    previous = experiment.COHORT
    copied = tmp_path / "missing-role-files"
    copied.mkdir()
    for name in ("protocol.json", "public_protocol.json"):
        (copied / name).write_bytes((root / previous / name).read_bytes())
    monkeypatch.setattr(experiment, "COHORT", copied)
    missing_checks, missing_hashes = experiment.preflight(root)
    assert any(c["field"] == "model_inputs_sha256" and not c["passed"] for c in missing_checks)
    assert any(c["field"] == "sha256" and not c["passed"] for c in missing_checks)
    assert missing_hashes["missing_evidence"]
