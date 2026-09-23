"""Tests for REQ-CL-7575 and SCENARIO-CL-7575-*.

The tests use small private rows for arithmetic and lifecycle checks. The
declared entrypoint performs the full cached-corpus protocol.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7575_v662_cached_learning_protocol as exp


def _native_cell(order: list[str], condition: str, unsupported_logit: float) -> dict:
    logits = {"supported": 0.0, "contains_unsupported": unsupported_logit}
    display = {" A": logits[order[0]], " B": logits[order[1]]}
    return {
        "component_hash": "sha256:component",
        "group_hash": "sha256:group",
        "role": "fit",
        "condition": condition,
        "option_order": order,
        "display_logits": display,
        "full_logits_by_option_id": logits,
        "prompt_sha256": f"sha256:prompt-{condition}-{order[0]}",
        "request_id": f"sha256:request-{condition}-{order[0]}",
        "complete_source_window": "whole source context",
        "complete_response_window": "whole answer context",
        "disposition": "complete",
        "generated_tokens": 0,
        "readout_kind": "option_logits",
    }


def _group(role: str, index: int, probability: float, label: int) -> dict:
    return {
        "source_id": f"sha256:{role}-{index:03d}",
        "group_id": f"sha256:group-{role}-{index:03d}",
        "role": role,
        "official_split": "train" if role in {"fit", "tune", "policy"} else "test",
        "probability": probability,
        "label": label,
        "context": f"whole context {role} {index}",
        "context_sha256": exp.canonical_hash(f"whole context {role} {index}"),
        "request_hashes": [f"sha256:req-{role}-{index}"],
        "response_hashes": [f"sha256:res-{role}-{index}"],
    }


def _roles(count: int = 16) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for role in exp.ROLE_COUNTS:
        size = count if role in {"fit", "online"} else max(4, count // 2)
        result[role] = [
            _group(role, index, (index + 1) / (size + 1), index % 2) for index in range(size)
        ]
    return result


def test_original_probability_uses_option_mapping_and_preserves_context() -> None:
    """SCENARIO-CL-7575-CUSTODY reconstructs semantics from raw bytes."""

    rows = []
    for condition, logit in (("original", 2.0), ("absent", -1.0), ("mismatched", 0.5)):
        rows.append(_native_cell(["supported", "contains_unsupported"], condition, logit))
        rows.append(_native_cell(["contains_unsupported", "supported"], condition, logit))
    reduced = exp.reduce_native_group(rows, {"label": 1, "role": "fit"}, "train")
    assert reduced["probability"] == pytest.approx(exp.sigmoid(2.0))
    assert reduced["context"] == "whole source context"
    assert reduced["source_removal_probability"] == pytest.approx(exp.sigmoid(-1.0))
    assert reduced["source_mismatch_probability"] == pytest.approx(exp.sigmoid(0.5))
    assert reduced["diagnostics_are_labels"] is False
    assert len(reduced["request_hashes"]) == 6

    changed = deepcopy(rows)
    changed[0]["display_logits"][" B"] = 9.0
    with pytest.raises(ValueError, match="option_mapping_invalid"):
        exp.reduce_native_group(changed, {"label": 1, "role": "fit"}, "train")


def test_role_and_exposure_manifest_fail_closed() -> None:
    """SCENARIO-CL-7575-CUSTODY and ISOLATION retain exact identities."""

    roles = _roles()
    expected = {name: len(rows) for name, rows in roles.items()}
    exposure = exp.build_exposure_manifest(roles, expected_counts=expected)
    assert exposure["fresh_confirmatory_claim_allowed"] is False
    assert exposure["claim_scope"] == "descriptive_reuse"
    assert exposure["all_candidate_groups_previously_exposed"] is True
    assert exposure["manifest_sha256"] == exp.canonical_hash(exposure["groups"])

    incomplete = deepcopy(roles)
    incomplete["online"].pop()
    with pytest.raises(ValueError, match="role_count_invalid:online"):
        exp.build_exposure_manifest(incomplete, expected_counts=expected)

    duplicate = deepcopy(roles)
    duplicate["fit"][1]["source_id"] = duplicate["fit"][0]["source_id"]
    with pytest.raises(ValueError, match="source_id_duplicate"):
        exp.build_exposure_manifest(duplicate, expected_counts=expected)


def test_static_controls_freeze_before_policy_selection() -> None:
    """SCENARIO-CL-7575-GATES fits controls without policy leakage."""

    roles = _roles(24)
    controls = exp.fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    assert controls["temperature"]["selected_on_role"] == "tune"
    assert controls["bounded"]["fit_roles"] == ["fit", "tune"]
    assert controls["unconstrained_nine_knot"]["capacity"] == 9
    assert controls["strongest_control"]["selected_on_role"] == "tune"
    assert controls["protocol_frozen_before_policy_labels"] is True
    assert set(controls["static_rows_by_role"]) == {"policy"}


def test_protocol_hash_freezes_real_ids_and_registered_settings() -> None:
    """SCENARIO-CL-7575-ISOLATION freezes label-blind orders."""

    roles = _roles(16)
    protocol = exp.freeze_protocol(roles)
    assert protocol["order_seeds"] == list(exp.ORDER_SEEDS)
    assert all(len(order) == 16 for order in protocol["orders"].values())
    assert all(
        set(order) == {row["source_id"] for row in roles["online"]}
        for order in protocol["orders"].values()
    )
    assert protocol["feedback_delay"] == 8
    assert protocol["release_block_size"] == 8
    assert protocol["minimum_non_escalation_fraction"] == 0.10
    assert protocol["bootstrap_replays_per_order"] == 1000
    assert protocol["protocol_sha256"] == exp.canonical_hash(protocol["hash_payload"])


def test_causal_replay_updates_persists_and_reloads(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-CAUSAL exercises the complete learning lifecycle."""

    roles = _roles(16)
    protocol = exp.freeze_protocol(roles)
    static = exp.fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    replay = exp.replay_order(
        roles["online"],
        protocol["orders"][str(exp.ORDER_SEEDS[0])],
        static,
        seed=exp.ORDER_SEEDS[0],
        state_path=tmp_path / "state.json",
    )
    assert replay["chronology_violations"] == 0
    assert replay["restart_mismatches"] == 0
    assert replay["release_count"] == 1
    assert len(replay["rows"]) == 16 * len(exp.ONLINE_ARMS)
    assert {row["arm"] for row in replay["rows"]} == set(exp.ONLINE_ARMS)
    assert all(row["raw_squared_error_denominator"] == 1 for row in replay["rows"])
    assert all(
        row["metric_direction"] == "lower_brier_and_cost_are_better" for row in replay["rows"]
    )
    assert all(row["provenance"] == "cached_authentic_qwen_option_logits" for row in replay["rows"])
    assert replay["shuffled_feedback_within_release_block"] is True


def test_retention_is_label_isolated(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-RETENTION keeps evaluator labels read-only."""

    roles = _roles(16)
    static = exp.fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    learner = exp.seeded_bounded_learner(static)
    learner.update_batch(
        [(row["source_id"], row["probability"], row["label"]) for row in roles["online"][:8]]
    )
    before = learner.state_hash()
    result = exp.evaluate_retention(roles["test"], learner, seed=7578001)
    assert result["state_hash_before"] == before
    assert result["state_hash_after"] == before
    assert result["retention_labels_used_for_update"] == 0
    assert len(result["rows"]) == len(roles["test"]) * 2
    exp.atomic_json(tmp_path / "retention.json", result)


def test_comparison_rows_reduce_raw_numerators() -> None:
    """SCENARIO-CL-7575-GATES independently reduces per-unit rows."""

    rows = [
        exp.comparison_row("u1", "raw", 0.8, 1, seed=1, phase="static"),
        exp.comparison_row("u1", "bounded", 0.9, 1, seed=1, phase="static"),
        exp.comparison_row("u2", "raw", 0.2, 0, seed=1, phase="static"),
        exp.comparison_row("u2", "bounded", 0.1, 0, seed=1, phase="static"),
    ]
    reduced = exp.reduce_comparison_rows(rows)
    assert reduced["raw"]["brier_numerator"] == pytest.approx(0.08)
    assert reduced["bounded"]["brier_numerator"] == pytest.approx(0.02)
    assert reduced["raw"]["denominator"] == 2
    assert exp.paired_improvements(rows, "bounded", "raw") == pytest.approx([0.03, 0.03])


def test_blocked_artifact_names_exact_failed_operand(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-CUSTODY makes missing inputs complete blocked work."""

    checks, hashes = exp.collect_preconditions(tmp_path)
    artifact = exp.build_blocked_artifact(checks, hashes, duration_s=0.01)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "no_model_load"
    failure = artifact["gate_check_summary"]["first_failure"]
    assert {"check", "upstream", "path", "field", "op", "expected", "observed"} <= set(failure)
    assert artifact["MODEL_SPECS"] == []
    assert artifact["historical_model_id"] == exp.HISTORICAL_MODEL_ID
    assert (
        exp.validate_artifact(artifact, root=tmp_path, require_validation=False)["blocked"] is True
    )


def test_artifact_contract_keeps_exploratory_scope(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-TERMINAL keeps readiness distinct from benefit."""

    roles = _roles(16)
    protocol = exp.freeze_protocol(roles)
    exposure = exp.build_exposure_manifest(
        roles, expected_counts={name: len(rows) for name, rows in roles.items()}
    )
    rows = [
        exp.comparison_row("u1", "raw", 0.7, 1, seed=1, phase="static"),
        exp.comparison_row("u1", "bounded", 0.8, 1, seed=1, phase="static"),
    ]
    evidence = {
        "rows": rows,
        "protocol": protocol,
        "exposure_manifest": exposure,
        "custody_passed": True,
        "lifecycle_passed": True,
        "retention_isolated": True,
        "uncertainty": {"replays_per_order": 1000, "orders": 5, "order_dependence_disclosed": True},
        "effect_gates": {
            "brier_vs_raw": False,
            "brier_vs_control": False,
            "decision_cost": False,
            "retention": True,
        },
        "capability_e2e": {
            "passed": True,
            "operations": ["predict", "release", "update", "persist", "reload"],
        },
        "analytical_positive_control": {"verdict_class": "null", "row_contrast_supported": False},
    }
    artifact = exp.build_artifact(
        evidence,
        preconditions=[],
        source_hashes={},
        validation_receipts=[],
        duration_s=1.0,
        require_validation=False,
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["cached_roles_ready_score"] == 1
    assert artifact["online_protocol_ready_score"] == 1
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert artifact["claim_scope"] == "descriptive_reuse"
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["MODEL_SPECS"] == []
    assert not any(
        artifact["invocation_counts"][name][state]
        for name in artifact["invocation_counts"]
        for state in artifact["invocation_counts"][name]
    )
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=tmp_path, require_validation=False)["valid"] is True


def test_parse_args_and_thin_wrapper_contract() -> None:
    """SCENARIO-CL-7575-TERMINAL keeps the public CLI narrow."""

    args = exp.parse_args(["--root", str(exp.REPO_ROOT), "--date", "20260923"])
    assert args.root == exp.REPO_ROOT
    assert args.date == "20260923"
    assert exp.parse_args(["--verify-artifact", "candidate.json"]).verify_artifact == Path(
        "candidate.json"
    )


def test_field_principles_cover_required_terminal_fields() -> None:
    """REQ-CL-7575 carries the one-line field meanings with each artifact."""

    principles = exp.field_principles()
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(principles)
    assert all(isinstance(value, str) and value for value in principles.values())


def test_real_cached_prerequisites_and_roles_authenticate() -> None:
    """SCENARIO-CL-7575-CUSTODY reads every exact producer sidecar."""

    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    roles = exp.load_cached_roles(exp.REPO_ROOT, hashes)
    assert {name: len(rows) for name, rows in roles.items()} == exp.ROLE_COUNTS
    assert len(hashes) == 25
    assert all(len(row["contexts_by_condition"]) == 6 for rows in roles.values() for row in rows)


def test_small_causal_uncertainty_retrains_every_order() -> None:
    """SCENARIO-CL-7575-GATES keeps each resample causal."""

    roles = _roles(16)
    static = exp.fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    progress: list[tuple[int, int, int]] = []
    result = exp.causal_uncertainty_replays(
        roles["online"],
        roles["test"],
        static,
        replays_per_order=1,
        progress_callback=lambda seed, completed, total: progress.append((seed, completed, total)),
    )
    assert result["full_causal_replays"] == 5
    assert result["each_replay_retrained_in_causal_order"] is True
    assert result["real_released_labels_only"] is True
    assert len(progress) == 5
    assert set(result["intervals"]) == {
        "brier_vs_raw",
        "brier_vs_control",
        "decision_cost",
        "coverage",
        "retention_degradation",
    }
    with pytest.raises(ValueError, match="replays_per_order_invalid"):
        exp.causal_uncertainty_replays(roles["online"], roles["test"], static, replays_per_order=0)


def test_static_bootstrap_holm_and_sidecar_receipt(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-GATES binds static inference and its raw store."""

    roles = _roles(16)
    static = exp.fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    result = exp.static_bootstrap(
        static["static_rows_by_role"]["policy"],
        static["strongest_control"]["arm"],
        draws=20,
    )
    assert set(result["contrasts"]) == {"vs_raw", "vs_strongest"}
    assert all("holm_adjusted_p" in row for row in result["contrasts"].values())
    receipt = exp._write_jsonl(tmp_path / "nested" / "rows.jsonl", [{"x": 1}], tmp_path)
    assert receipt["path"] == "nested/rows.jsonl"
    assert receipt["rows"] == 1
    assert exp._read_jsonl(tmp_path / "nested" / "rows.jsonl") == [{"x": 1}]

    bad = [row for row in static["static_rows_by_role"]["policy"] if row["arm"] != "bounded"]
    with pytest.raises(ValueError, match="static_contrast_roster_invalid"):
        exp.static_bootstrap(bad, "raw", draws=2)


def test_guard_branches_reject_malformed_inputs(tmp_path: Path) -> None:
    """REQ-CL-7575 closes malformed rows instead of repairing evidence."""

    assert exp.sigmoid(-2.0) == pytest.approx(1.0 - exp.sigmoid(2.0))
    with pytest.raises(ValueError, match="logit_not_finite"):
        exp.sigmoid(float("nan"))
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        exp.comparison_row("u", "raw", 2.0, 1, seed=1, phase="test")
    with pytest.raises(ValueError, match="paired_contrast_empty"):
        exp.paired_improvements(
            [exp.comparison_row("u", "raw", 0.5, 1, seed=1, phase="test")],
            "bounded",
            "raw",
        )
    invalid_denominator = exp.comparison_row("u", "raw", 0.5, 1, seed=1, phase="test")
    invalid_denominator["raw_squared_error_denominator"] = 2
    with pytest.raises(ValueError, match="row_denominator_invalid"):
        exp.reduce_comparison_rows([invalid_denominator])
    invalid_numerator = exp.comparison_row("u", "raw", 0.5, 1, seed=1, phase="test")
    invalid_numerator["raw_squared_error_numerator"] = 7.0
    with pytest.raises(ValueError, match="row_numerator_invalid"):
        exp.reduce_comparison_rows([invalid_numerator])
    assert exp._gate("x", "benefit", 0, 1, op="gt")["passed"] is True
    assert exp._gate("x", "benefit", 1, 1, op="ge")["passed"] is True
    assert exp._gate("x", "benefit", 1, 1, op="le")["passed"] is True
    with pytest.raises(ValueError, match="gate_operator_invalid"):
        exp._gate("x", "benefit", 1, 1, op="bad")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._load_json(malformed)
    malformed_lines = tmp_path / "malformed.jsonl"
    malformed_lines.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        exp._read_jsonl(malformed_lines)


def test_native_and_reader_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-CUSTODY names malformed native and receipt evidence."""

    rows = [
        _native_cell(order, condition, 1.0)
        for condition in ("original", "absent", "mismatched")
        for order in (["supported", "contains_unsupported"], ["contains_unsupported", "supported"])
    ]
    label = {"label": 1, "role": "fit", "component_hash": "sha256:component"}
    mutations = []
    mutations.append((rows[:5], label, "native_group_incomplete"))
    changed = deepcopy(rows)
    changed[0]["group_hash"] = "sha256:other"
    mutations.append((changed, label, "native_group_identity_invalid"))
    mutations.append((rows, {"label": 2, "role": "fit"}, "label_identity_invalid"))
    mutations.append((rows, {**label, "component_hash": "sha256:other"}, "label_identity_invalid"))
    changed = deepcopy(rows)
    changed[0]["disposition"] = "failed"
    mutations.append((changed, label, "native_group_incomplete"))
    changed = deepcopy(rows)
    changed[0]["condition"] = "forbidden"
    mutations.append((changed, label, "native_cell_duplicate_or_invalid"))
    changed = deepcopy(rows)
    changed[0]["complete_source_window"] = ""
    mutations.append((changed, label, "whole_context_invalid"))
    changed = deepcopy(rows)
    changed[0]["complete_response_window"] = "different"
    mutations.append((changed, label, "whole_response_invalid"))
    for native, evaluator, error in mutations:
        with pytest.raises(ValueError, match=error):
            exp.reduce_native_group(native, evaluator, "train")
    with pytest.raises(ValueError, match="option_mapping_invalid"):
        exp._semantic_logit({})
    invalid_display = deepcopy(rows[0])
    invalid_display["display_logits"] = None
    with pytest.raises(ValueError, match="option_mapping_invalid"):
        exp._semantic_logit(invalid_display)

    receipt_path = tmp_path / "rows.jsonl"
    receipt_path.write_text('{"x":1}\n', encoding="utf-8")
    receipt = {"path": "rows.jsonl", "sha256": exp.sha256_file(receipt_path), "rows": 1}
    assert exp._read_receipt(tmp_path, receipt, []) == [{"x": 1}]
    with pytest.raises(ValueError, match="sidecar_hash_invalid"):
        exp._read_receipt(tmp_path, {**receipt, "sha256": "sha256:bad"}, [])
    with pytest.raises(ValueError, match="sidecar_count_invalid"):
        exp._read_receipt(tmp_path, {**receipt, "rows": 2}, [])


def test_control_branches_and_invalid_role_roster(tmp_path: Path) -> None:
    """REQ-CL-7575 freezes every registered static prediction path."""

    roles = _roles(16)
    static = exp.fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    row = roles["policy"][0]
    for arm in ("raw", "temperature", "unconstrained_nine_knot"):
        changed = deepcopy(static)
        changed["strongest_control"]["arm"] = arm
        probability = exp._control_probability(changed, row)
        assert 0.0 <= probability <= 1.0
    with pytest.raises(ValueError, match="static_role_invalid:fit"):
        exp.fit_static_controls([], roles["tune"], roles["policy"])
    order = [row["source_id"] for row in roles["online"]]
    with pytest.raises(ValueError, match="online_order_roster_invalid"):
        exp.replay_order(roles["online"], order[:-1], static, seed=1, state_path=tmp_path / "x")
    bad_roles = deepcopy(roles)
    bad_roles["fit"][0]["role"] = "bad"
    with pytest.raises(ValueError, match="role_identity_invalid:fit"):
        exp.build_exposure_manifest(
            bad_roles, expected_counts={name: len(values) for name, values in bad_roles.items()}
        )


def test_validation_reader_reports_each_contract_failure(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-TERMINAL rejects each terminal contract mutation."""

    roles = _roles(16)
    protocol = exp.freeze_protocol(roles)
    exposure = exp.build_exposure_manifest(
        roles, expected_counts={name: len(values) for name, values in roles.items()}
    )
    rows = [
        exp.comparison_row("u", "raw", 0.7, 1, seed=1, phase="static"),
        exp.comparison_row("u", "bounded", 0.8, 1, seed=1, phase="static"),
    ]
    evidence = {
        "rows": rows,
        "protocol": protocol,
        "exposure_manifest": exposure,
        "custody_passed": True,
        "lifecycle_passed": True,
        "retention_isolated": True,
        "uncertainty": {"replays_per_order": 1000, "orders": 5, "order_dependence_disclosed": True},
        "effect_gates": {
            "brier_vs_raw": False,
            "brier_vs_control": False,
            "decision_cost": False,
            "retention": True,
        },
        "capability_e2e": {
            "passed": True,
            "operations": ["predict", "release", "update", "persist", "reload"],
        },
        "analytical_positive_control": {"verdict_class": "null", "row_contrast_supported": False},
    }
    artifact = exp.build_artifact(
        evidence,
        preconditions=[],
        source_hashes=[],
        validation_receipts=[],
        duration_s=1.0,
        require_validation=False,
    )

    def rejected(field: str, value: object, error: str) -> None:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path, require_validation=False)

    with pytest.raises(ValueError, match="artifact_object_required"):
        exp.validate_artifact([], root=tmp_path)
    rejected("schema", "bad", "identity_invalid")
    changed = deepcopy(artifact)
    changed["duration_s"] = 2.0
    with pytest.raises(ValueError, match="checksum_invalid"):
        exp.validate_artifact(changed, root=tmp_path, require_validation=False)
    rejected("MODEL_SPECS", ["bad"], "model_specs_not_empty")
    rejected("historical_model_id", "bad", "historical_model_invalid")
    rejected("model_invoked", True, "current_invocations_not_zero")
    rejected("claim_scope", "fresh", "claim_scope_invalid")
    rejected("inference_substrate_class", "bad", "substrate_invalid")
    bad_protocol = deepcopy(artifact["frozen_protocol"])
    bad_protocol["hash_payload"]["feedback_delay"] = 99
    rejected("frozen_protocol", bad_protocol, "protocol_hash_invalid")
    rejected("protocol_sha256", "bad", "protocol_identity_invalid")
    rejected("rows", [], "rows_missing")
    rejected("independent_row_reduction", {}, "row_reduction_mismatch")
    malformed_rows = deepcopy(rows)
    malformed_rows[0].pop("probability")
    rejected("rows", malformed_rows, "row_reduction_invalid")
    rejected("cached_roles_ready_score", 2, "score_invalid")
    rejected("verdict_class", "positive", "verdict_mismatch")
    rejected("field_principles", {}, "field_principles_invalid")
    with pytest.raises(ValueError, match="validation_invalid"):
        exp.validate_artifact(artifact, root=tmp_path, require_validation=True)

    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.independent_reduce_artifact(path, root=tmp_path)["passed"] is True
    assert exp._validation_passed([{"passed": True, "exit_code": 0, "timed_out": False}]) is True

    invalid_sidecar = deepcopy(artifact)
    invalid_sidecar["raw_sidecars"] = {"rows": "not-a-receipt"}
    invalid_sidecar["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_sidecar)
    with pytest.raises(ValueError, match="raw_sidecar_invalid"):
        exp.validate_artifact(invalid_sidecar, root=tmp_path, require_validation=False)
    missing_sidecar = deepcopy(artifact)
    missing_sidecar["raw_sidecars"] = {
        "rows": {"path": "missing.jsonl", "sha256": "sha256:bad", "rows": 1}
    }
    missing_sidecar["reproducibility_checksum"] = exp.reproducibility_checksum(missing_sidecar)
    with pytest.raises(ValueError, match="raw_sidecar_hash_invalid"):
        exp.validate_artifact(missing_sidecar, root=tmp_path, require_validation=False)


def test_blocked_reader_and_malformed_producers_are_strict(tmp_path: Path) -> None:
    """SCENARIO-CL-7575-CUSTODY keeps blocked and malformed states distinct."""

    for relative in (
        exp.FIT_CAPTURE_PATH,
        exp.EVAL_CAPTURE_PATH,
        exp.PROTOCOL_SOURCE_PATH,
        exp.SOURCE_EVALUATION_PATH,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
    checks, _hashes = exp.collect_preconditions(tmp_path)
    assert any(row["check"] == "artifact_json" for row in checks)

    empty_root = tmp_path / "empty"
    blocked_checks, hashes = exp.collect_preconditions(empty_root)
    blocked = exp.build_blocked_artifact(blocked_checks, hashes, duration_s=0.1)

    def blocked_rejected(field: str, value: object, error: str) -> None:
        changed = deepcopy(blocked)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=empty_root, require_validation=False)

    blocked_rejected("honest_verdict", "bad", "blocked_verdict_invalid")
    blocked_rejected("gate_check_summary", {}, "blocked_gate_summary_invalid")
    blocked_rejected("rows", [{"bad": True}], "blocked_measurement_invalid")

    outside = tmp_path / "outside" / "rows.jsonl"
    receipt = exp._write_jsonl(outside, [{"x": 1}], tmp_path / "different-root")
    assert receipt["path"] == str(outside.resolve())
