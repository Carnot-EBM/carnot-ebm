"""Behavior tests for REQ-CL-7310 factor-local delayed revision."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7310_v642_factor_prototype as exp


def _masks(**overrides: int) -> dict[str, int]:
    masks = dict.fromkeys(exp.FAMILIES, exp.FULL_MASK)
    masks.update(overrides)
    return masks


def _event(event_id: str, family: str, value: int, index: int = 0) -> dict[str, object]:
    return {
        "event_id": event_id,
        "family_id": family,
        "numeric_value": value,
        "chronology_index": index,
    }


def _release(
    event_id: str,
    family: str,
    value: int,
    label: str,
    source: int = 0,
    released: int = 4,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "family_id": family,
        "numeric_value": value,
        "observed_label": label,
        "source_index": source,
        "release_index": released,
    }


def _seal(
    controller: exp.FactorLocalController,
    event_id: str,
    family: str,
    value: int,
    source: int = 0,
    released: int = 4,
) -> dict[str, object]:
    return controller.seal_prediction(
        _event(event_id, family, value, source), release_index=released
    )


def test_scenario_cl_7310_revision_is_factor_local_and_delayed() -> None:
    """SCENARIO-CL-7310-REVISION: one contradiction changes one factor only."""

    controller = exp.FactorLocalController.from_masks(
        _masks(lower_bound=1 << 5, upper_bound=1 << 20)
    )
    upper_before = controller.family_bytes("upper_bound")
    _seal(controller, "counterexample", "lower_bound", 10)
    before_release = controller.state_bytes()

    with pytest.raises(exp.FactorRevisionRejected, match="release_not_due"):
        controller.apply_release(
            _release("counterexample", "lower_bound", 10, "reject"), current_index=3
        )
    assert controller.state_bytes() == before_release

    receipt = controller.apply_release(
        _release("counterexample", "lower_bound", 10, "reject"), current_index=4
    )
    assert receipt["contradiction"] is True
    assert receipt["rebuilt_family"] == "lower_bound"
    assert controller.family_bytes("upper_bound") == upper_before
    assert controller.predict(_event("later", "lower_bound", 12, 5)) == "abstain"


def test_scenario_cl_7310_revision_keeps_longest_ambiguous_suffix() -> None:
    """SCENARIO-CL-7310-REVISION: the longest consistent suffix can stay ambiguous."""

    controller = exp.FactorLocalController.from_masks(_masks(lower_bound=1 << 5))
    _seal(controller, "old", "lower_bound", 2, 0, 4)
    controller.apply_release(_release("old", "lower_bound", 2, "accept", 0, 4), current_index=4)
    _seal(controller, "new", "lower_bound", 10, 5, 9)
    receipt = controller.apply_release(
        _release("new", "lower_bound", 10, "reject", 5, 9), current_index=9
    )

    row = controller.family_state("lower_bound")
    assert receipt["longest_consistent_suffix_length"] == 1
    assert row["survivor_mask"].bit_count() == 22
    assert [witness["event_id"] for witness in row["witnesses"]] == ["new"]
    assert controller.predict(_event("ambiguous", "lower_bound", 12, 10)) == "abstain"


def test_scenario_cl_7310_bounds_rejects_poison_and_byte_exhaustion() -> None:
    """SCENARIO-CL-7310-BOUNDS: poison and over-cap writes are atomic failures."""

    controller = exp.FactorLocalController.from_masks(_masks(lower_bound=1 << 5))
    _seal(controller, "poison", "lower_bound", 10)
    before_poison = controller.state_bytes()
    with pytest.raises(exp.FactorRevisionRejected, match="invalid_release"):
        controller.apply_release(_release("poison", "lower_bound", 10, "maybe"), current_index=4)
    assert controller.state_bytes() == before_poison

    minimum = len(exp.FactorLocalController.from_masks(_masks()).state_bytes())
    bounded = exp.FactorLocalController.from_masks(_masks(), memory_cap_bytes=minimum + 8)
    before_cap = bounded.state_bytes()
    with pytest.raises(exp.FactorRevisionRejected, match="factor_memory_cap"):
        _seal(bounded, "too-large", "lower_bound", 10)
    assert bounded.state_bytes() == before_cap


def test_scenario_cl_7310_bounds_supports_rollback_and_cold_restart(tmp_path: Path) -> None:
    """SCENARIO-CL-7310-BOUNDS: rollback and restart preserve canonical bytes."""

    controller = exp.FactorLocalController.from_masks(_masks(lower_bound=1 << 5))
    _seal(controller, "rollback", "lower_bound", 10)
    parent = controller.state_bytes()
    receipt = controller.apply_release(
        _release("rollback", "lower_bound", 10, "reject"), current_index=4
    )
    assert controller.state_bytes() != parent
    controller.rollback(receipt)
    assert controller.state_bytes() == parent

    state_path = tmp_path / "controller.json"
    controller.save(state_path)
    restored = exp.FactorLocalController.load(state_path)
    assert restored.state_bytes() == controller.state_bytes()
    assert restored.memory_usage()["within_cap"] is True


def test_scenario_cl_7310_controls_prediction_is_read_only() -> None:
    """SCENARIO-CL-7310-CONTROLS: prediction neither mutates nor reads authority."""

    controller = exp.FactorLocalController.from_masks(_masks(lower_bound=1 << 5))
    before = controller.state_bytes()
    assert controller.predict(_event("public", "lower_bound", 10)) == "accept"
    assert controller.state_bytes() == before
    with pytest.raises(exp.FactorRevisionRejected, match="private_authority_in_prediction"):
        controller.predict({**_event("private", "lower_bound", 10), "exact_label": "accept"})
    assert controller.state_bytes() == before


def test_scenario_cl_7310_streams_are_frozen_and_authority_separated() -> None:
    """SCENARIO-CL-7310-STREAMS: fixed streams keep labels and transitions private."""

    development = exp.build_stream_views("development")
    evaluation = exp.build_stream_views("evaluation")
    assert exp.stream_conformance_errors(development, "development") == []
    assert exp.stream_conformance_errors(evaluation, "evaluation") == []
    assert len(development.public) == 8 * 1_024
    assert len(evaluation.public) == 24 * 1_024
    assert evaluation.manifest["strata"] == {
        "isolated_factor_changes": 8,
        "overlapping_factor_changes": 8,
        "stationary_controls": 8,
    }
    assert all(
        "exact_label" not in row and "target_parameter" not in row for row in evaluation.public
    )
    assert all(
        exp.evaluator_exact_label(row["family_id"], row["numeric_value"], row["target_parameter"])
        == row["exact_label"]
        for row in evaluation.authority[::257]
    )


def test_scenario_cl_7310_controls_cover_all_attacks_and_arms(tmp_path: Path) -> None:
    """SCENARIO-CL-7310-CONTROLS: every frozen arm and required attack executes."""

    rows = exp.run_controller_controls(tmp_path)
    assert {row["control"] for row in rows} == set(exp.CONTROL_NAMES)
    assert all(row["passed"] is True for row in rows)
    assert exp.ARMS == (
        "factor_local_retained_witnesses",
        "global_reset_on_contradiction",
        "local_reset_without_retained_witnesses",
        "frozen_warmup",
        "label_shuffled_factor_local_revision",
    )


def test_scenario_cl_7310_e2e_opt_in_revision_is_durable(tmp_path: Path) -> None:
    """SCENARIO-CL-7310-E2E: sealed feedback changes a later durable prediction."""

    disabled = exp.FactorPipelineHook(tmp_path / "disabled")
    assert disabled.predict(_event("off", "lower_bound", 10)) == "abstain"
    assert not (tmp_path / "disabled" / "factor_controller.json").exists()

    initial = exp.FactorLocalController.from_masks(_masks(lower_bound=1 << 5))
    hook = exp.FactorPipelineHook(tmp_path / "enabled", enabled=True, controller=initial)
    pre_label = hook.pre_label(_event("e2e", "lower_bound", 10), release_index=4)
    assert pre_label["prediction"] == "accept"
    update = hook.release(_release("e2e", "lower_bound", 10, "reject"), current_index=4)
    assert update["state_hash_before"] != update["state_hash_after"]
    later = hook.predict(_event("later", "lower_bound", 10, 5))
    assert later == "reject"

    restarted = exp.FactorPipelineHook(tmp_path / "enabled", enabled=True)
    assert restarted.predict(_event("after-restart", "lower_bound", 10, 6)) == later
    assert restarted.transactional_memory.state_hash().startswith("sha256:")


def test_scenario_cl_7310_preconditions_and_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-CL-7310-PRECONDITIONS/TERMINAL: blocks are exact and readiness is not efficacy."""

    paths = exp.ExperimentPaths.under(tmp_path)
    historical = {
        "schema": "carnot.exp7297.v641_mixture_audit.v1",
        "experiment_id": 7297,
        "status": "complete",
        "verdict_class": "null",
        "mixture_audit_complete_score": 1,
        "flagged_adversarial": False,
    }
    paths.historical_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.historical_artifact.write_text(json.dumps(historical), encoding="utf-8")
    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT, paths)
    assert exp.gate_summary(checks)["passed"] is True
    assert hashes[str(paths.historical_artifact)].startswith("sha256:")

    historical["flagged_adversarial"] = True
    paths.historical_artifact.write_text(json.dumps(historical), encoding="utf-8")
    blocked_checks, blocked_hashes = exp.collect_preconditions(exp.REPO_ROOT, paths)
    blocked = exp.build_blocked_artifact(blocked_checks, blocked_hashes)
    failure = blocked["gate_check_summary"]["first_failure"]
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert failure["check"] == "historical_not_quarantined"
    assert failure["field"] == "flagged_adversarial"
    assert failure["observed_value"] is True
    assert failure["expected_value"] is False


def test_scenario_cl_7310_terminal_validation_rejects_tampering(tmp_path: Path) -> None:
    """SCENARIO-CL-7310-TERMINAL: cold validation rejects success-shaped tampering."""

    paths = exp.ExperimentPaths.under(tmp_path)
    paths.historical_artifact.write_bytes((exp.REPO_ROOT / exp.HISTORICAL_ARTIFACT).read_bytes())
    artifact = exp.build_and_seal(exp.REPO_ROOT, paths, progress=False)
    assert artifact["status"] == "complete"
    assert artifact["factor_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert exp.validate_artifact(artifact, repo_root=exp.REPO_ROOT, check_files=True) == []

    changed = deepcopy(artifact)
    changed["factor_fixture_ready_score"] = 0
    assert "fixture_ready_score" in exp.validate_artifact(changed, repo_root=exp.REPO_ROOT)
