"""Tests for REQ-VERIFY-7494 and SCENARIO-VERIFY-7494-*.

Controlled logits exercise the production reducers without loading Qwen or
opening the access-separated test and online labels.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7494_v656_window_eval_capture as exp


def _load_jsonl(path: Path) -> list[exp.JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sealed_inputs() -> tuple[list[exp.JsonDict], ...]:
    raw = exp.REPO_ROOT / exp.PROTOCOL_RAW_DIR
    return tuple(
        _load_jsonl(raw / name)
        for name in ("predictors.jsonl", "groups.jsonl", "windows.jsonl", "requests.jsonl")
    )


def test_req_verify_7494_spec_and_scenarios_precede_code() -> None:
    """REQ-VERIFY-7494 fixes every capture boundary before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7494" in text
    for suffix in (
        "PREREQUISITE",
        "ROSTER",
        "CALLS",
        "LABELS",
        "ACCOUNTING",
        "RESUME",
        "READY",
        "E2E",
    ):
        assert f"SCENARIO-VERIFY-7494-{suffix}" in text


def test_scenario_verify_7494_prerequisite_authenticates_both_producers() -> None:
    """SCENARIO-VERIFY-7494-PREREQUISITE rejects changed upstream evidence."""

    protocol = json.loads(exp.PROTOCOL_PATH.read_text(encoding="utf-8"))
    pilot = json.loads(exp.PILOT_PATH.read_text(encoding="utf-8"))
    reduced = exp.reduce_upstream_gates(protocol, pilot, protocol_errors=[], pilot_errors=[])
    assert reduced["passed"] is True
    assert all(row["passed"] and "prevent" in row["principle"] for row in reduced["checks"])

    changed = deepcopy(protocol)
    changed["window_protocol_ready_score"] = 0
    assert (
        exp.reduce_upstream_gates(changed, pilot, protocol_errors=[], pilot_errors=[])["passed"]
        is False
    )
    assert (
        exp.reduce_upstream_gates(
            protocol,
            pilot,
            protocol_errors=["changed"],
            pilot_errors=[],
        )["passed"]
        is False
    )


def test_scenario_verify_7494_roster_uses_only_sealed_test_and_online_calls() -> None:
    """SCENARIO-VERIFY-7494-ROSTER retains 120 plus 160 held-out groups."""

    plan = exp.build_capture_plan(*_sealed_inputs())
    assert len(plan) == exp.FORWARD_BUDGET == 2_192
    assert len(plan) <= exp.FORWARD_CEILING == 2_240
    assert {row["role"] for row in plan} == {"test", "online"}
    assert len({row["group_id"] for row in plan if row["role"] == "test"}) == 120
    assert len({row["group_id"] for row in plan if row["role"] == "online"}) == 160
    assert sum(row["eligible"] is True for row in plan) == exp.ELIGIBLE_FORWARD_BUDGET == 2_152
    assert sum(row["disposition"] == "excluded" for row in plan) == 40
    assert all(
        row["prompt_sha256"] == exp.window_protocol.sha256_text(row["prompt"]) for row in plan
    )
    assert all(not set(row) & exp.FORBIDDEN_CAPTURE_FIELDS for row in plan)
    assert all(row["gold_label"] is None and row["archived_pilot_reuse"] is False for row in plan)

    group = [row for row in plan if row["group_id"] == plan[0]["group_id"]]
    assert {tuple(row["option_order"]) for row in group} == {
        exp.OPTION_IDS,
        tuple(reversed(exp.OPTION_IDS)),
    }
    assert sum(row["arm"] == "whole_response" for row in group) == 2


def test_scenario_verify_7494_calls_preserve_context_offsets_and_label_seal() -> None:
    """SCENARIO-VERIFY-7494-CALLS and LABELS retain context without outcomes."""

    plan = exp.build_capture_plan(*_sealed_inputs())
    focused = next(row for row in plan if row["arm"] == "focused_window" and row["eligible"])
    assert focused["byte_start"] is not None and focused["byte_end"] is not None
    assert focused["sentence_version"] == exp.window_protocol.SENTENCE_VERSION
    assert focused["window_version"] == exp.window_protocol.WINDOW_VERSION
    assert (
        exp.window_protocol.remove_focus_markers(focused["marked_response"])
        == focused["response_text"]
    )
    assert focused["source_text"] in focused["prompt"]
    for forbidden in ("gold_label", "annotation", "response_generator", focused["group_id"]):
        assert forbidden not in focused["prompt"]


def test_capture_plan_fails_closed_on_roster_prompt_role_and_eligibility_drift() -> None:
    """REQ-VERIFY-7494 rejects changed sealed identities before inference."""

    predictors, groups, windows, requests = _sealed_inputs()
    wrong_groups = deepcopy(groups)
    next(row for row in wrong_groups if row["role"] == "test")["role"] = "training"
    with pytest.raises(exp.WindowCaptureError, match="evaluation_role_counts_invalid"):
        exp.build_capture_plan(predictors, wrong_groups, windows, requests)

    missing = deepcopy(requests)
    missing.pop(next(index for index, row in enumerate(missing) if row["role"] == "online"))
    with pytest.raises(exp.WindowCaptureError, match="evaluation_request_count_invalid"):
        exp.build_capture_plan(predictors, groups, windows, missing)

    wrong_role = deepcopy(requests)
    next(row for row in wrong_role if row["role"] == "test")["role"] = "online"
    with pytest.raises(exp.WindowCaptureError, match="sealed_role_mismatch"):
        exp.build_capture_plan(predictors, groups, windows, wrong_role)

    changed = deepcopy(requests)
    next(row for row in changed if row["role"] == "test")["prompt_sha256"] = "sha256:changed"
    with pytest.raises(exp.WindowCaptureError, match="sealed_prompt_hash_mismatch"):
        exp.build_capture_plan(predictors, groups, windows, changed)

    wrong_eligibility = deepcopy(requests)
    target = next(row for row in wrong_eligibility if row["role"] == "test")
    target["eligible"] = not target["eligible"]
    with pytest.raises(exp.WindowCaptureError, match="sealed_eligibility_mismatch"):
        exp.build_capture_plan(predictors, groups, windows, wrong_eligibility)


def test_capture_plan_rejects_missing_identity_forbidden_fields_and_budget_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7494 authenticates joins and fixed eligible accounting."""

    predictors, groups, windows, requests = _sealed_inputs()
    missing_predictor = [row for row in predictors if row["role"] not in {"test", "online"}]
    with pytest.raises(exp.WindowCaptureError, match="sealed_group_missing"):
        exp.build_capture_plan(missing_predictor, groups, windows, requests)

    monkeypatch.setattr(exp, "FORBIDDEN_CAPTURE_FIELDS", {"call_id"})
    with pytest.raises(exp.WindowCaptureError, match="forbidden_capture_field"):
        exp.build_capture_plan(predictors, groups, windows, requests)
    monkeypatch.setattr(exp, "FORBIDDEN_CAPTURE_FIELDS", set())
    monkeypatch.setattr(exp, "ELIGIBLE_FORWARD_BUDGET", 0)
    with pytest.raises(exp.WindowCaptureError, match="eligible_forward_count_invalid"):
        exp.build_capture_plan(predictors, groups, windows, requests)


def test_scenario_verify_7494_accounting_separates_disposition_support_and_ready() -> None:
    """SCENARIO-VERIFY-7494-ACCOUNTING keeps failures and missing rows explicit."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    minimums = {"test": 2, "online": 1}
    reduced = exp.reduce_capture(plan, rows, minimums=minimums)
    assert reduced["capture_complete_score"] == 1
    assert reduced["role_support_score"] == 1
    assert reduced["all_eligible_calls_valid"] is True
    assert reduced["sample_size_budget"]["calls"] == {
        "planned": 12,
        "attempted": 12,
        "complete": 12,
        "failed": 0,
        "excluded": 0,
        "censored": 0,
        "unstarted": 0,
    }

    failed = deepcopy(rows)
    failed[0].update({"disposition": "failed", "error": "RuntimeError:fixture"})
    failed_reduction = exp.reduce_capture(plan, failed, minimums=minimums)
    assert failed_reduction["capture_complete_score"] == 1
    assert failed_reduction["all_eligible_calls_valid"] is False
    assert failed_reduction["sample_size_budget"]["calls"]["failed"] == 1

    missing = exp.reduce_capture(plan, rows[:-1], minimums=minimums)
    assert missing["capture_complete_score"] == 0
    assert missing["sample_size_budget"]["calls"]["unstarted"] == 1

    nonfinite = deepcopy(rows)
    nonfinite[0]["raw_logits_by_option_id"]["supported"] = float("nan")
    assert (
        exp.reduce_capture(plan, nonfinite, minimums=minimums)["all_eligible_calls_valid"] is False
    )


def test_scenario_verify_7494_resume_binds_manifest_model_tokenizer_and_roles(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7494-RESUME rejects a changed identity or changed bytes."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    binding = exp.checkpoint_binding(
        plan,
        model_sha256="sha256:model",
        tokenizer_identity={"kind": "embedded"},
        request_manifest_sha256="sha256:manifest",
    )
    root = tmp_path / "checkpoints"
    group_id = str(plan[0]["group_id"])
    exp.write_group_checkpoint(
        root,
        binding=binding,
        group_id=group_id,
        rows=[row for row in rows if row["group_id"] == group_id],
    )
    assert len(exp.load_checkpoint_rows(root, expected_binding=binding)) == 4
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_binding_mismatch"):
        exp.load_checkpoint_rows(
            root,
            expected_binding={**binding, "model_sha256": "sha256:changed"},
        )
    shard = next(root.glob("group-*.json"))
    shard.write_text("{}", encoding="utf-8")
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_group_hash_mismatch"):
        exp.load_checkpoint_rows(root, expected_binding=binding)


def test_checkpoint_rejects_malformed_index_receipt_and_payload(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7494-RESUME rejects every malformed checkpoint layer."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)[:4]
    binding = exp.checkpoint_binding(
        plan,
        model_sha256="sha256:model",
        tokenizer_identity={"kind": "embedded"},
        request_manifest_sha256="sha256:manifest",
    )
    root = tmp_path / "checkpoints"
    exp.write_group_checkpoint(root, binding=binding, group_id="test-0", rows=rows)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_binding_mismatch"):
        exp.write_group_checkpoint(
            root,
            binding={**binding, "model_sha256": "sha256:changed"},
            group_id="test-1",
            rows=rows,
        )

    index_path = root / "checkpoint-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    malformed = deepcopy(index)
    malformed["groups"] = []
    exp.atomic_json(index_path, malformed)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_groups_invalid"):
        exp.load_checkpoint_rows(root, expected_binding=binding)

    malformed = deepcopy(index)
    malformed["groups"] = {"test-0": "bad"}
    exp.atomic_json(index_path, malformed)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_group_receipt_invalid"):
        exp.load_checkpoint_rows(root, expected_binding=binding)

    exp.atomic_json(index_path, index)
    receipt = index["groups"]["test-0"]
    group_path = root / receipt["path"]
    payload = json.loads(group_path.read_text(encoding="utf-8"))
    payload["rows_sha256"] = "sha256:changed"
    exp.atomic_json(group_path, payload)
    receipt["sha256"] = exp.sha256_file(group_path)
    exp.atomic_json(index_path, index)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_group_invalid"):
        exp.load_checkpoint_rows(root, expected_binding=binding)

    assert exp._load_json_object(tmp_path / "absent.json") == {}


def test_raw_shards_rehash_and_independently_restore_calls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7494-E2E preserves independently reducible raw bytes."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    manifest = exp.write_raw_shards(tmp_path, plan=plan, rows=rows)
    assert all(row["size_bytes"] < exp.MAX_ARTIFACT_BYTES for row in manifest)
    assert exp.reload_raw_shards(tmp_path, manifest) == {"plan": plan, "rows": rows}
    path = tmp_path / manifest[0]["path"]
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(exp.WindowCaptureError, match="raw_shard_hash_mismatch"):
        exp.reload_raw_shards(tmp_path, manifest)


def test_scenario_verify_7494_ready_gates_are_typed_and_principled() -> None:
    """SCENARIO-VERIFY-7494-READY keeps validity, readiness, and benefit separate."""

    gates = exp.build_acceptance_gates(
        validity_passed=True,
        capture_complete=True,
        role_support=True,
        eligible_calls_complete=True,
        validation_passed=True,
    )
    assert all(row["passed"] and "prevent" in row["principle"] for row in gates)
    assert {row["category"] for row in gates} == {"validity", "readiness", "benefit"}
    failed = exp.build_acceptance_gates(
        validity_passed=True,
        capture_complete=True,
        role_support=True,
        eligible_calls_complete=False,
        validation_passed=True,
    )
    assert exp.gate_check_summary(failed)["first_failure"]["check"] == "eligible_calls_complete"


def test_terminal_fixture_validates_and_principles_cover_every_field(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7494-E2E cold-reduces a schema-complete terminal fixture."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == [exp.MODEL_HF_ID]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["capture_complete_score"] == 1
    assert artifact["role_support_score"] == 1
    assert artifact["window_evaluation_ready_score"] == 1
    assert artifact["fresh_evaluation_labels_opened"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert set(artifact["field_principles"]) == set(artifact)
    assert all("prevent" in text for text in artifact["field_principles"].values())

    changed = deepcopy(artifact)
    changed["window_evaluation_ready_score"] = 0
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "window_evaluation_ready_score_mismatch" in exp.validate_artifact(
        changed,
        root=tmp_path,
        require_validation=False,
    )


def test_explicit_failure_is_complete_but_retryable_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-7494 never publishes a failed eligible call as a numeric zero."""

    artifact = exp.build_artifact_for_test(tmp_path, failed_call=True)
    assert artifact["capture_complete_score"] == 1
    assert artifact["role_support_score"] == 1
    assert artifact["window_evaluation_ready_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("complete_partial")
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []


def test_validation_manifest_is_frozen_to_three_exp7494_paths() -> None:
    """SCENARIO-VERIFY-7494-E2E forbids an expanded or full-suite scope."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_validator_rejects_identity_reduction_gate_principle_and_receipt_drift(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7494 names altered terminal evidence instead of trusting summaries."""

    for expected, mutate in (
        ("identity_mismatch:schema", lambda value: value.__setitem__("schema", "wrong")),
        ("group_rows_mismatch", lambda value: value.__setitem__("rows", [])),
        ("gate_summary_mismatch", lambda value: value.__setitem__("gate_check_summary", {})),
        ("field_principles_incomplete", lambda value: value.__setitem__("field_principles", {})),
    ):
        artifact = exp.build_artifact_for_test(tmp_path)
        mutate(artifact)
        assert expected in exp.validate_artifact(artifact, root=tmp_path, require_validation=False)

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["invocation_counts"]["forward_calls"]["completed"] = 0
    assert "invocation_counts_mismatch" in exp.validate_artifact(
        artifact,
        root=tmp_path,
        require_validation=False,
    )

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["validation_receipts"] = []
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "required_validation_failed" in exp.validate_artifact(
        artifact,
        root=tmp_path,
        require_validation=True,
    )


def test_artifact_and_independent_reduction_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-7494 rejects altered reductions and classifies invalid support."""

    invalid = exp.build_artifact_for_test(tmp_path, validation_passed=False)
    assert invalid["verdict_class"] == "disqualified"
    unsupported = exp.build_artifact_for_test(
        tmp_path,
        role_minimums={"test": 3, "online": 2},
    )
    assert unsupported["verdict_class"] == "disqualified"

    for expected, field in (
        ("capture_reduction_mismatch", "capture_reduction"),
        ("sample_size_budget_mismatch", "sample_size_budget"),
        ("role_counts_mismatch", "role_counts"),
    ):
        artifact = exp.build_artifact_for_test(tmp_path)
        artifact[field] = {}
        assert expected in exp.validate_artifact(artifact, root=tmp_path, require_validation=False)

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["raw_logit_shards"] = None
    assert exp.independent_reduce(artifact, root=tmp_path, require_terminal=False) == {
        "passed": False,
        "errors": ["raw_logit_shards_invalid"],
    }
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["role_minimums"] = None
    assert "role_minimums_invalid" in exp.validate_artifact(
        artifact,
        root=tmp_path,
        require_validation=False,
    )

    for expected, field, value in (
        ("capture_complete_score_mismatch", "capture_complete_score", 0),
        ("role_support_score_mismatch", "role_support_score", 0),
        ("acceptance_gates_invalid", "acceptance_gate_results", {}),
    ):
        artifact = exp.build_artifact_for_test(tmp_path)
        artifact[field] = value
        assert expected in exp.validate_artifact(artifact, root=tmp_path, require_validation=False)

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["invocation_counts"] = None
    assert "invocation_evidence_invalid" in exp.validate_artifact(
        artifact,
        root=tmp_path,
        require_validation=False,
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    first = next(iter(artifact["field_principles"]))
    artifact["field_principles"][first] = "missing failure wording"
    assert "field_principles_missing_failure_mode" in exp.validate_artifact(
        artifact,
        root=tmp_path,
        require_validation=False,
    )


def test_independent_cli_summary_omits_raw_rows_but_keeps_headline_counts(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7494-E2E keeps saved verifier output below artifact limits."""

    artifact = exp.build_artifact_for_test(tmp_path)
    reduced = exp.independent_reduce(artifact, root=tmp_path, require_terminal=False)
    summary = exp.independent_cli_summary(reduced)
    assert summary["passed"] is True
    assert summary["errors"] == []
    assert summary["capture_complete_score"] == 1
    assert summary["role_support_score"] == 1
    assert summary["sample_size_budget"] == artifact["sample_size_budget"]
    assert "reduction" not in summary
    assert "reconciled_rows" not in json.dumps(summary, sort_keys=True)
    assert len(json.dumps(summary)) < 20_000


def test_defensive_boundaries_reject_malformed_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-7494 fails closed on malformed artifacts, shards, and calls."""

    assert exp.validate_artifact([], root=tmp_path, require_validation=False) == [
        "artifact_not_object"
    ]
    assert exp.validation_names_passed({}, ("one",)) is False
    with pytest.raises(exp.WindowCaptureError, match="raw_manifest_invalid"):
        exp.reload_raw_shards(tmp_path, [])
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_index_missing"):
        exp.load_checkpoint_rows(tmp_path / "missing", expected_binding={})

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    rows[0]["probabilities_by_option_id"] = None
    assert (
        exp.reduce_capture(plan, rows, minimums={"test": 2, "online": 1})[
            "all_eligible_calls_valid"
        ]
        is False
    )
