"""Tests for REQ-VERIFY-7493 and SCENARIO-VERIFY-7493-*.

The tests use sealed request bytes and controlled logits. They never load Qwen
or open evaluator labels.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7493_v656_window_fit_capture as exp


def _load_jsonl(path: Path) -> list[exp.JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_verify_7493_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7493 and every task scenario exist before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7493" in text
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
        assert f"SCENARIO-VERIFY-7493-{suffix}" in text


def test_scenario_verify_7493_prerequisite_requires_both_exact_producers() -> None:
    """SCENARIO-VERIFY-7493-PREREQUISITE rejects changed producer evidence."""

    protocol = json.loads(exp.PROTOCOL_PATH.read_text(encoding="utf-8"))
    pilot = json.loads(exp.PILOT_PATH.read_text(encoding="utf-8"))
    reduced = exp.reduce_upstream_gates(
        protocol,
        pilot,
        protocol_errors=[],
        pilot_errors=[],
    )
    assert reduced["passed"] is True
    assert all(row["passed"] and row["principle"] for row in reduced["checks"])

    changed = deepcopy(protocol)
    changed["window_protocol_ready_score"] = 0
    assert (
        exp.reduce_upstream_gates(
            changed,
            pilot,
            protocol_errors=[],
            pilot_errors=[],
        )["passed"]
        is False
    )
    changed_pilot = deepcopy(pilot)
    changed_pilot["model_identity"]["model_sha256"] = "sha256:changed"
    assert (
        exp.reduce_upstream_gates(
            protocol,
            changed_pilot,
            protocol_errors=[],
            pilot_errors=["model_identity_changed"],
        )["passed"]
        is False
    )


def test_scenario_verify_7493_roster_builds_only_fit_calls_from_sealed_bytes() -> None:
    """SCENARIO-VERIFY-7493-ROSTER retains 180 plus 60 groups and 1,936 calls."""

    raw = exp.REPO_ROOT / exp.PROTOCOL_RAW_DIR
    plan = exp.build_capture_plan(
        _load_jsonl(raw / "predictors.jsonl"),
        _load_jsonl(raw / "groups.jsonl"),
        _load_jsonl(raw / "windows.jsonl"),
        _load_jsonl(raw / "requests.jsonl"),
    )
    assert len(plan) == exp.FORWARD_BUDGET == 1_936
    assert {row["role"] for row in plan} == {"training", "calibration_tuning"}
    assert len({row["group_id"] for row in plan if row["role"] == "training"}) == 180
    assert len({row["group_id"] for row in plan if row["role"] == "calibration_tuning"}) == 60
    assert sum(row["eligible"] is True for row in plan) == 1_904
    assert sum(row["disposition"] == "excluded" for row in plan) == 32
    assert all(
        row["prompt_sha256"] == exp.window_protocol.sha256_text(row["prompt"]) for row in plan
    )
    assert all(row["source_text"] is not None and row["response_text"] is not None for row in plan)
    assert all(not (set(row) & exp.FORBIDDEN_CAPTURE_FIELDS) for row in plan)
    assert all(row["gold_label"] is None for row in plan)
    assert all(
        int(row["prompt_token_count"]) <= exp.TOKEN_CEILING for row in plan if row["eligible"]
    )

    one_group = [row for row in plan if row["group_id"] == plan[0]["group_id"]]
    assert {tuple(row["option_order"]) for row in one_group} == {
        exp.OPTION_IDS,
        tuple(reversed(exp.OPTION_IDS)),
    }
    assert sum(row["arm"] == "whole_response" for row in one_group) == 2


def test_scenario_verify_7493_calls_keep_offsets_versions_and_label_blind_prompts() -> None:
    """SCENARIO-VERIFY-7493-CALLS and LABELS retain context without outcomes."""

    raw = exp.REPO_ROOT / exp.PROTOCOL_RAW_DIR
    plan = exp.build_capture_plan(
        _load_jsonl(raw / "predictors.jsonl"),
        _load_jsonl(raw / "groups.jsonl"),
        _load_jsonl(raw / "windows.jsonl"),
        _load_jsonl(raw / "requests.jsonl"),
    )
    focused = next(row for row in plan if row["arm"] == "focused_window" and row["eligible"])
    assert focused["byte_start"] is not None
    assert focused["byte_end"] is not None
    assert focused["sentence_version"] == exp.window_protocol.SENTENCE_VERSION
    assert focused["window_version"] == exp.window_protocol.WINDOW_VERSION
    assert (
        exp.window_protocol.remove_focus_markers(focused["marked_response"])
        == focused["response_text"]
    )
    assert focused["source_text"] in focused["prompt"]
    for forbidden in ("gold_label", "annotation", "response_generator", focused["group_id"]):
        assert forbidden not in focused["prompt"]


def test_capture_plan_rejects_role_request_and_prompt_drift() -> None:
    """REQ-VERIFY-7493 fails closed before inference when sealed requests drift."""

    raw = exp.REPO_ROOT / exp.PROTOCOL_RAW_DIR
    predictors = _load_jsonl(raw / "predictors.jsonl")
    groups = _load_jsonl(raw / "groups.jsonl")
    windows = _load_jsonl(raw / "windows.jsonl")
    requests = _load_jsonl(raw / "requests.jsonl")

    wrong_role = deepcopy(groups)
    next(row for row in wrong_role if row["role"] == "training")["role"] = "test"
    with pytest.raises(exp.WindowCaptureError, match="fit_role_counts_invalid"):
        exp.build_capture_plan(predictors, wrong_role, windows, requests)

    missing = deepcopy(requests)
    missing.pop(next(index for index, row in enumerate(missing) if row["role"] == "training"))
    with pytest.raises(exp.WindowCaptureError, match="fit_request_count_invalid"):
        exp.build_capture_plan(predictors, groups, windows, missing)

    changed = deepcopy(requests)
    next(row for row in changed if row["role"] == "training")["prompt_sha256"] = "sha256:changed"
    with pytest.raises(exp.WindowCaptureError, match="sealed_prompt_hash_mismatch"):
        exp.build_capture_plan(predictors, groups, windows, changed)


def test_scenario_verify_7493_accounting_separates_complete_support_and_ready() -> None:
    """SCENARIO-VERIFY-7493-ACCOUNTING keeps terminal failures explicit."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    reduced = exp.reduce_capture(plan, rows, minimums={"training": 2, "calibration_tuning": 1})
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
    failed_reduction = exp.reduce_capture(
        plan,
        failed,
        minimums={"training": 2, "calibration_tuning": 1},
    )
    assert failed_reduction["capture_complete_score"] == 1
    assert failed_reduction["role_support_score"] == 1
    assert failed_reduction["all_eligible_calls_valid"] is False
    assert failed_reduction["sample_size_budget"]["calls"]["failed"] == 1

    missing = exp.reduce_capture(
        plan,
        rows[:-1],
        minimums={"training": 2, "calibration_tuning": 1},
    )
    assert missing["capture_complete_score"] == 0
    assert missing["sample_size_budget"]["calls"]["unstarted"] == 1

    nonfinite = deepcopy(rows)
    nonfinite[0]["raw_logits_by_option_id"]["supported"] = float("nan")
    assert (
        exp.reduce_capture(
            plan,
            nonfinite,
            minimums={"training": 2, "calibration_tuning": 1},
        )["all_eligible_calls_valid"]
        is False
    )


def test_scenario_verify_7493_resume_requires_exact_manifest_identity(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7493-RESUME rejects changed bindings and shard bytes."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    binding = exp.checkpoint_binding(
        plan,
        model_sha256="sha256:model",
        tokenizer_identity={"kind": "embedded"},
        request_manifest_sha256="sha256:manifest",
    )
    checkpoint_root = tmp_path / "checkpoints"
    exp.write_group_checkpoint(
        checkpoint_root,
        binding=binding,
        group_id="training-0",
        rows=[row for row in rows if row["group_id"] == "training-0"],
    )
    loaded = exp.load_checkpoint_rows(checkpoint_root, expected_binding=binding)
    assert len(loaded) == 4

    changed = {**binding, "model_sha256": "sha256:changed"}
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_binding_mismatch"):
        exp.load_checkpoint_rows(checkpoint_root, expected_binding=changed)
    shard = next(checkpoint_root.glob("group-*.json"))
    shard.write_text("{}", encoding="utf-8")
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_group_hash_mismatch"):
        exp.load_checkpoint_rows(checkpoint_root, expected_binding=binding)


def test_raw_shards_split_rehash_and_independently_reduce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7493-E2E binds each bounded raw shard by SHA-256."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    monkeypatch.setattr(exp, "RAW_SHARD_TARGET_BYTES", 2_000)
    manifest = exp.write_raw_shards(tmp_path, plan=plan, rows=rows)
    assert len(manifest) > 2
    assert all(row["size_bytes"] < exp.MAX_ARTIFACT_BYTES for row in manifest)
    reloaded = exp.reload_raw_shards(tmp_path, manifest)
    assert reloaded == {"plan": plan, "rows": rows}

    target = tmp_path / manifest[0]["path"]
    target.write_text(target.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(exp.WindowCaptureError, match="raw_shard_hash_mismatch"):
        exp.reload_raw_shards(tmp_path, manifest)


def test_scenario_verify_7493_ready_gates_are_typed_and_principled() -> None:
    """SCENARIO-VERIFY-7493-READY keeps validity, readiness, and benefit separate."""

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


def test_terminal_fixture_validates_and_every_field_has_a_failure_principle(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7493-E2E cold-reduces raw evidence and readiness."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == [exp.MODEL_HF_ID]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["capture_complete_score"] == 1
    assert artifact["role_support_score"] == 1
    assert artifact["window_fit_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert set(artifact["field_principles"]) == set(artifact)
    assert all("prevent" in principle for principle in artifact["field_principles"].values())

    changed = deepcopy(artifact)
    changed["window_fit_ready_score"] = 0
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "window_fit_ready_score_mismatch" in exp.validate_artifact(
        changed,
        root=tmp_path,
        require_validation=False,
    )


def test_explicit_call_failure_is_complete_but_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-7493 never turns an explicit failed call into a zero score."""

    artifact = exp.build_artifact_for_test(tmp_path, failed_call=True)
    assert artifact["capture_complete_score"] == 1
    assert artifact["role_support_score"] == 1
    assert artifact["window_fit_ready_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("complete_partial")
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []


def test_validation_manifest_is_scoped_and_full_suite_is_absent() -> None:
    """SCENARIO-VERIFY-7493-E2E freezes only the three affected files."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_validator_fails_closed_on_identity_rows_shards_and_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-7493 names each invalid evidence boundary."""

    cases: list[tuple[str, exp.JsonDict]] = []
    for expected, mutate in (
        ("identity_mismatch:schema", lambda value: value.__setitem__("schema", "wrong")),
        (
            "invocation_counts_mismatch",
            lambda value: value["invocation_counts"]["forward_calls"].__setitem__("completed", 0),
        ),
        ("group_rows_mismatch", lambda value: value.__setitem__("rows", [])),
        ("gate_summary_mismatch", lambda value: value.__setitem__("gate_check_summary", {})),
        (
            "field_principles_incomplete",
            lambda value: value.__setitem__("field_principles", {}),
        ),
    ):
        value = exp.build_artifact_for_test(tmp_path)
        mutate(value)
        cases.append((expected, value))
    for expected, value in cases:
        assert expected in exp.validate_artifact(value, root=tmp_path, require_validation=False)

    bad_shard = exp.build_artifact_for_test(tmp_path)
    path = tmp_path / bad_shard["raw_logit_root"] / bad_shard["raw_logit_shards"][0]["path"]
    path.write_text("changed\n", encoding="utf-8")
    assert any(
        error.startswith("raw_shard_hash_mismatch:")
        for error in exp.validate_artifact(
            bad_shard,
            root=tmp_path,
            require_validation=False,
        )
    )
    no_receipts = exp.build_artifact_for_test(tmp_path)
    no_receipts["validation_receipts"] = []
    no_receipts["reproducibility_checksum"] = exp.artifact_checksum(no_receipts)
    assert "required_validation_failed" in exp.validate_artifact(
        no_receipts,
        root=tmp_path,
        require_validation=True,
    )


def test_defensive_boundaries_reject_malformed_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-7493 rejects malformed rows, checkpoints, and artifacts."""

    assert exp.validate_artifact([], root=tmp_path, require_validation=False) == [
        "artifact_not_object"
    ]
    assert exp.validation_names_passed({}, ("one",)) is False
    assert (
        exp.validation_names_passed(
            [{"name": "one", "passed": True, "exit_code": 0, "timed_out": False}],
            ("one",),
        )
        is True
    )
    with pytest.raises(exp.WindowCaptureError, match="raw_manifest_invalid"):
        exp.reload_raw_shards(tmp_path, [])
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_index_missing"):
        exp.load_checkpoint_rows(tmp_path / "missing", expected_binding={})

    malformed = exp.controlled_call_fixture(exp.controlled_plan_fixture())
    malformed[0]["probabilities_by_option_id"] = None
    assert (
        exp.reduce_capture(
            exp.controlled_plan_fixture(),
            malformed,
            minimums={"training": 2, "calibration_tuning": 1},
        )["all_eligible_calls_valid"]
        is False
    )


def test_capture_plan_rejects_missing_identity_role_eligibility_and_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7493 authenticates every sealed request relationship."""

    raw = exp.REPO_ROOT / exp.PROTOCOL_RAW_DIR
    predictors = _load_jsonl(raw / "predictors.jsonl")
    groups = _load_jsonl(raw / "groups.jsonl")
    windows = _load_jsonl(raw / "windows.jsonl")
    requests = _load_jsonl(raw / "requests.jsonl")

    missing_predictor = predictors[1:]
    with pytest.raises(exp.WindowCaptureError, match="sealed_group_missing"):
        exp.build_capture_plan(missing_predictor, groups, windows, requests)

    wrong_role = deepcopy(requests)
    wrong_role[0]["role"] = (
        "calibration_tuning" if wrong_role[0]["role"] == "training" else "training"
    )
    with pytest.raises(exp.WindowCaptureError, match="sealed_role_mismatch"):
        exp.build_capture_plan(predictors, groups, windows, wrong_role)

    wrong_eligibility = deepcopy(requests)
    wrong_eligibility[0]["eligible"] = not wrong_eligibility[0]["eligible"]
    with pytest.raises(exp.WindowCaptureError, match="sealed_eligibility_mismatch"):
        exp.build_capture_plan(predictors, groups, windows, wrong_eligibility)

    wrong_arm = deepcopy(requests)
    wrong_arm[0]["arm"] = "unknown"
    with pytest.raises(exp.WindowCaptureError, match="window_request_invalid"):
        exp.build_capture_plan(predictors, groups, windows, wrong_arm)

    monkeypatch.setattr(exp, "FORBIDDEN_CAPTURE_FIELDS", {"call_id"})
    with pytest.raises(exp.WindowCaptureError, match="forbidden_capture_field"):
        exp.build_capture_plan(predictors, groups, windows, requests)
    monkeypatch.setattr(exp, "FORBIDDEN_CAPTURE_FIELDS", set())
    monkeypatch.setattr(exp, "ELIGIBLE_FORWARD_BUDGET", 0)
    with pytest.raises(exp.WindowCaptureError, match="eligible_forward_count_invalid"):
        exp.build_capture_plan(predictors, groups, windows, requests)


def test_reducer_covers_excluded_censored_and_non_numeric_calls() -> None:
    """SCENARIO-VERIFY-7493-ACCOUNTING preserves every non-success disposition."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)
    malformed = deepcopy(rows)
    malformed[0]["raw_logits_by_display_label"][exp.option_protocol.DISPLAY_LABELS[0]] = (
        "not-a-number"
    )
    assert exp.reduce_capture(plan, malformed, minimums={})["all_eligible_calls_valid"] is False

    censored = deepcopy(rows)
    censored[0].update({"disposition": "censored", "error": "deadline"})
    reduced = exp.reduce_capture(plan, censored, minimums={})
    assert reduced["sample_size_budget"]["groups"]["censored"] == 1

    excluded_plan = deepcopy(plan)
    excluded_rows = deepcopy(rows)
    group_id = excluded_plan[0]["group_id"]
    for planned in excluded_plan:
        if planned["group_id"] == group_id:
            planned.update({"eligible": False, "disposition": "excluded", "attempted": False})
    for observed in excluded_rows:
        if observed["group_id"] == group_id:
            observed.update({"eligible": False, "disposition": "excluded", "attempted": False})
    excluded = exp.reduce_capture(excluded_plan, excluded_rows, minimums={})
    assert excluded["sample_size_budget"]["groups"]["excluded"] == 1


def test_checkpoint_rejects_each_malformed_index_boundary(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7493-RESUME rejects malformed indices and group payloads."""

    plan = exp.controlled_plan_fixture()
    rows = exp.controlled_call_fixture(plan)[:4]
    binding = exp.checkpoint_binding(
        plan,
        model_sha256="sha256:model",
        tokenizer_identity={"kind": "embedded"},
        request_manifest_sha256="sha256:manifest",
    )
    root = tmp_path / "checkpoint"
    exp.write_group_checkpoint(root, binding=binding, group_id="training-0", rows=rows)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_binding_mismatch"):
        exp.write_group_checkpoint(
            root,
            binding={**binding, "model_sha256": "sha256:changed"},
            group_id="training-1",
            rows=rows,
        )

    index_path = root / "checkpoint-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    bad_groups = deepcopy(index)
    bad_groups["groups"] = []
    exp.atomic_json(index_path, bad_groups)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_groups_invalid"):
        exp.load_checkpoint_rows(root, expected_binding=binding)

    bad_receipt = deepcopy(index)
    bad_receipt["groups"] = {"training-0": "bad"}
    exp.atomic_json(index_path, bad_receipt)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_group_receipt_invalid"):
        exp.load_checkpoint_rows(root, expected_binding=binding)

    exp.atomic_json(index_path, index)
    receipt = index["groups"]["training-0"]
    group_path = root / receipt["path"]
    group = json.loads(group_path.read_text(encoding="utf-8"))
    group["rows_sha256"] = "sha256:changed"
    exp.atomic_json(group_path, group)
    receipt["sha256"] = exp.sha256_file(group_path)
    exp.atomic_json(index_path, index)
    with pytest.raises(exp.WindowCaptureError, match="checkpoint_group_invalid"):
        exp.load_checkpoint_rows(root, expected_binding=binding)


def test_raw_shard_size_count_and_json_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7493-E2E rejects oversized, miscounted, and malformed shards."""

    monkeypatch.setattr(exp, "MAX_ARTIFACT_BYTES", 1_000)
    monkeypatch.setattr(exp, "RAW_SHARD_TARGET_BYTES", 10_000)
    with pytest.raises(exp.WindowCaptureError, match="single_raw_row_exceeds_20_mib"):
        exp._split_rows([{"payload": "x" * 1_100}])
    with pytest.raises(exp.WindowCaptureError, match="raw_shard_exceeds_20_mib"):
        exp.write_raw_shards(
            tmp_path / "oversized",
            plan=[{"payload": "x" * 490}, {"payload": "y" * 490}],
            rows=[{"payload": "x"}],
        )

    monkeypatch.setattr(exp, "MAX_ARTIFACT_BYTES", 20 * 1024 * 1024)
    manifest = exp.write_raw_shards(
        tmp_path / "raw",
        plan=[{"payload": "x"}],
        rows=[{"payload": "y"}],
    )
    wrong_count = deepcopy(manifest)
    wrong_count[0]["rows"] = 2
    with pytest.raises(exp.WindowCaptureError, match="raw_shard_row_count_mismatch"):
        exp.reload_raw_shards(tmp_path / "raw", wrong_count)

    monkeypatch.setattr(exp, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(exp.WindowCaptureError, match="raw_shard_exceeds_20_mib"):
        exp.reload_raw_shards(tmp_path / "raw", manifest)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{\n", encoding="utf-8")
    with pytest.raises(exp.WindowCaptureError, match="jsonl_unreadable"):
        exp._load_jsonl(malformed)
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(exp.WindowCaptureError, match="jsonl_object_required"):
        exp._load_jsonl(malformed)
    assert exp._load_json_object(tmp_path / "absent.json") == {}
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert exp._load_json_object(non_object) == {}


def test_artifact_reduction_and_validator_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-7493 independently rejects each altered terminal reduction field."""

    invalid = exp.build_artifact_for_test(tmp_path, validation_passed=False)
    assert invalid["verdict_class"] == "disqualified"
    unsupported = exp.build_artifact_for_test(
        tmp_path,
        role_minimums={"training": 3, "calibration_tuning": 2},
    )
    assert unsupported["verdict_class"] == "disqualified"

    for expected, mutate in (
        (
            "capture_reduction_mismatch",
            lambda value: value.__setitem__("capture_reduction", {}),
        ),
        (
            "sample_size_budget_mismatch",
            lambda value: value.__setitem__("sample_size_budget", {}),
        ),
        ("role_counts_mismatch", lambda value: value.__setitem__("role_counts", {})),
        (
            "capture_complete_score_mismatch",
            lambda value: value.__setitem__("capture_complete_score", 0),
        ),
        (
            "role_support_score_mismatch",
            lambda value: value.__setitem__("role_support_score", 0),
        ),
        (
            "acceptance_gates_invalid",
            lambda value: value.__setitem__("acceptance_gate_results", {}),
        ),
    ):
        artifact = exp.build_artifact_for_test(tmp_path)
        mutate(artifact)
        assert expected in exp.validate_artifact(artifact, root=tmp_path, require_validation=False)

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["invocation_counts"] = None
    assert "invocation_evidence_invalid" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=False
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["role_minimums"] = None
    assert "role_minimums_invalid" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=False
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    first_key = next(iter(artifact["field_principles"]))
    artifact["field_principles"][first_key] = "missing failure wording"
    assert "field_principles_missing_failure_mode" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=False
    )
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["raw_logit_shards"] = None
    assert exp.independent_reduce(artifact, root=tmp_path, require_terminal=False) == {
        "passed": False,
        "errors": ["raw_logit_shards_invalid"],
    }
