"""Tests for the V649 independent online-trial audit.

Spec refs: REQ-REPORT-7401 and SCENARIO-REPORT-7401-DIAGNOSIS through
SCENARIO-REPORT-7401-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7401_v649_online_audit as audit


def _event(event_id: str, *, label: int, energy: float = 0.0) -> dict[str, object]:
    return {
        "event_id": event_id,
        "group_id": event_id,
        "partition": "training",
        "features": [energy, 1.0],
        "raw_energy": energy,
        "label": label,
        "feedback_disposition": "committed",
        "full_cost_s": 0.01,
    }


def _unit() -> dict[str, object]:
    return {
        "arm": "adaptive_affine_gibbs",
        "seed": 7397001,
        "affine": {"a": 1.0, "b": 0.0},
        "learning_rate": 0.01,
        "gradient_norm_cap": 1.0,
    }


def _policy() -> dict[str, object]:
    return {
        "accept_threshold": 0.25,
        "reject_threshold": 0.75,
        "accept_enabled": True,
        "reject_enabled": True,
    }


def test_fresh_affine_replay_updates_only_after_prediction() -> None:
    """SCENARIO-REPORT-7401-REPLAY: fresh scalar arithmetic reproduces rows."""

    rows = audit.recompute_stream(
        [_event("g0", label=1), _event("g1", label=0)], _unit(), _policy()
    )
    assert rows[0]["probability"] == pytest.approx(0.5)
    assert rows[0]["brier_loss"] == pytest.approx(0.25)
    assert rows[0]["log_loss"] == pytest.approx(-math.log(0.5))
    assert rows[0]["prediction_before_feedback"] is True
    assert rows[0]["state_hash_before_update"] != rows[0]["state_hash_after_update"]
    assert rows[1]["probability"] > 0.5
    assert rows[0]["typed_action_changed"] is False
    assert rows[0]["original_answer_corrected"] is False

    duplicate = [_event("g0", label=1), _event("g0", label=0)]
    with pytest.raises(ValueError, match="duplicate_update"):
        audit.recompute_stream(duplicate, _unit(), _policy())
    leaked = _event("leak", label=1)
    leaked["label_visible_before_prediction"] = True
    with pytest.raises(ValueError, match="future_label_leak"):
        audit.recompute_stream([leaked], _unit(), _policy())


def test_private_mutations_all_fail_closed() -> None:
    """SCENARIO-REPORT-7401-MUTATIONS: all six private mutations are rejected."""

    evidence = audit.build_fixture_evidence()
    assert audit.evidence_errors(evidence) == []
    rows = audit.run_mutation_checks(evidence)
    assert [row["mutation"] for row in rows] == list(audit.MUTATIONS)
    assert all(row["rejected"] is True for row in rows)
    assert all(row["rejecting_check"] for row in rows)
    assert evidence == audit.build_fixture_evidence()

    constant = audit.build_fixture_evidence()
    constant["discrimination_claim"] = True
    for row in constant["rows"]:
        row["probability"] = 0.5
    assert "learned_constant_discrimination_claim" in audit.evidence_errors(constant)
    constant["scorer_class"] = "prevalence_baseline"
    assert "learned_constant_discrimination_claim" not in audit.evidence_errors(constant)


def test_missing_current_producers_are_exact_blockers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7401-DIAGNOSIS: absent unchanged inputs block by exact path."""

    spec = tmp_path / audit.SPEC_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-REPORT-7401\n", encoding="utf-8")
    for relative in audit.SUPPORTING_INPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("support\n", encoding="utf-8")
    checks, hashes, sources = audit.collect_preconditions(tmp_path)
    missing = [row for row in checks if row["passed"] is False]
    assert [row["upstream"] for row in missing] == [
        audit.ADAPTER_PATH.as_posix(),
        audit.TRIAL_PATH.as_posix(),
    ]
    assert all(row["artifact_field"] == "bytes" for row in missing)
    assert audit.SPEC_PATH.as_posix() in hashes
    assert sources[audit.ADAPTER_PATH.as_posix()]["verdict_class"] is None

    adapter = tmp_path / audit.ADAPTER_PATH
    trial = tmp_path / audit.TRIAL_PATH
    adapter.write_text(
        json.dumps(
            {
                "experiment_id": "exp7397-delayed-adapter",
                "delayed_adapter_ready_score": 1,
                "flagged_adversarial": False,
                "verdict_class": "null",
            }
        ),
        encoding="utf-8",
    )
    trial.write_text(
        json.dumps(
            {
                "experiment_id": "exp7399-online-trial",
                "online_capture_complete_score": 1,
                "flagged_adversarial": False,
                "verdict_class": "null",
            }
        ),
        encoding="utf-8",
    )
    present_checks, _, _ = audit.collect_preconditions(tmp_path)
    assert all(row["passed"] is True for row in present_checks)
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    assert audit._load_object(non_object) == {}

    artifact = audit.build_blocked_artifact(
        run_date=audit.RUN_DATE,
        started_at="2026-09-18T00:00:00+00:00",
        completed_at="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=checks,
        source_hashes=hashes,
        source_sidecar={"path": "sidecar.json", "sha256": "sha256:test"},
        validation_receipts=[],
        phase_spans=[],
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["rows"] == []
    assert artifact["online_audit_complete_score"] == 0
    assert artifact["online_value_confirmed_score"] == 0
    assert artifact["promotion_score"] == 0
    assert (
        artifact["gate_check_summary"]["first_required_failure"]["upstream"]
        == audit.ADAPTER_PATH.as_posix()
    )
    assert audit.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["online_value_confirmed_score"] = 1
    assert "blocked_scores_nonzero" in audit.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["historical-model"]
    assert "substrate_declaration_mismatch" in audit.validate_artifact(changed)
    assert audit.validate_artifact([]) == ["artifact_not_object"]
    for mutation, expected in (
        ({"schema": "wrong"}, "identity_mismatch"),
        ({"verifier_is_oracle": True}, "oracle_declaration_mismatch"),
        ({"gate_check_summary": {}}, "blocked_gate_summary_missing"),
        ({"rows": [{}]}, "blocked_artifact_has_dependent_rows"),
        ({"promotion_score": 1}, "promotion_nonzero"),
        ({"field_principles": {}}, "field_principles_incomplete"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        assert expected in audit.validate_artifact(changed)


def test_classification_keeps_completion_and_value_separate() -> None:
    """SCENARIO-REPORT-7401-VALUE: completion does not imply online value."""

    complete_null = audit.classify_terminal(
        inputs_present=True,
        inputs_valid=True,
        audit_complete=True,
        efficacy_passed=False,
        required_validation_passed=True,
    )
    assert complete_null == {
        "verdict_class": "null",
        "honest_verdict": "complete_null_online_value_not_confirmed",
        "online_audit_complete_score": 1,
        "online_value_confirmed_score": 0,
    }
    assert audit.classify_terminal(False, False, False, False, True)["verdict_class"] == "blocked"
    assert audit.classify_terminal(True, False, True, True, True)["verdict_class"] == "disqualified"
    assert audit.classify_terminal(True, True, True, True, False)["verdict_class"] == "disqualified"
    assert audit.classify_terminal(True, True, True, True, True)["verdict_class"] == "positive"
    assert audit._sigmoid(-1000.0) == pytest.approx(0.0)
    assert audit._typed_action(0.1, _policy()) == "accept"
    assert audit._typed_action(0.9, _policy()) == "reject"


def test_affected_plan_and_atomic_artifact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7401-ARTIFACT: scoped commands and atomic readers stay fixed."""

    private = tmp_path / "private"
    commands = audit.build_validation_commands(Path.cwd(), private)
    assert [command.name for command in commands] == list(
        audit.validation_scope.REQUIRED_CHECK_NAMES
    )
    assert audit.validate_command_plan(Path.cwd(), audit.V649_MANIFEST, commands) == []
    argv = [argument for command in commands for argument in command.argv]
    assert "tests/python" not in argv
    assert "full_python_suite" not in [command.name for command in commands]
    assert any(argument.startswith("--basetemp=") for argument in argv)

    target = tmp_path / "value.json"
    audit.atomic_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
    assert audit.sha256_file(target).startswith("sha256:")

    args = audit.parse_args(["--date", audit.RUN_DATE, "--output", str(target)])
    assert args.date == audit.RUN_DATE
    monkeypatch.setattr(audit, "run_experiment", lambda *args, **kwargs: {"status": "blocked"})
    assert audit.main(["--date", audit.RUN_DATE, "--output", str(target)]) == 0
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])

    blocked = audit.build_blocked_artifact(
        run_date=audit.RUN_DATE,
        started_at="2026-09-18T00:00:00+00:00",
        completed_at="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[
            audit._precondition("missing", audit.TRIAL_PATH.as_posix(), "bytes", "present", None)
        ],
        source_hashes={},
        source_sidecar={},
        validation_receipts=[],
        phase_spans=[],
    )
    audit.atomic_json(target, blocked)
    assert audit.main(["--cold-replay", str(target)]) == 0
