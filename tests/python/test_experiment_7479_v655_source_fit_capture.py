"""Tests for REQ-VERIFY-7479 and SCENARIO-VERIFY-7479-*.

The tests use sealed text and controlled logits. They never load a model.
"""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from carnot import experiment_7479_v655_source_fit_capture as exp


def _groups() -> list[exp.JsonDict]:
    """Build a small role-valid panel for reducer tests."""

    rows = []
    for index, role in enumerate(("training", "training", "calibration_tuning")):
        rows.append(
            {
                "roster_position": index,
                "group_id": f"group-{index}",
                "group_hash": exp.canonical_hash(f"group-{index}"),
                "source_hash": exp.canonical_hash(f"source-{index}"),
                "response_hash": exp.canonical_hash(f"response-{index}"),
                "source_text": f"Source {index}",
                "response_text": f"Response {index}",
                "role": role,
                "annotation_disposition": "supported" if index != 1 else "contains_unsupported",
                "label": 1 if index != 1 else 0,
                "annotation_provenance": "pinned human annotations",
            }
        )
    return rows


def _completed_rows(plan: list[exp.JsonDict]) -> list[exp.JsonDict]:
    """Create complete finite native rows for every planned cell."""

    rows = []
    for index, cell in enumerate(plan):
        order = cell["option_order"]
        logits = {order[0]: 3.0 + index / 100.0, order[1]: 1.0}
        rows.append(
            {
                **deepcopy(cell),
                "disposition": "complete",
                "attempted": True,
                "prompt_token_ids": [1, 20 + index, 30 + index],
                "prompt_token_count": 3,
                "label_token_ids": [11, 17],
                "requested_score_position": 2,
                "actual_last_evaluated_position": 2,
                "raw_logits_by_option_id": logits,
                "probabilities_by_option_id": exp.softmax_by_option(logits),
                "model_receipt_hash": "sha256:model",
                "prefill_s": 0.1,
                "generated_tokens": 0,
                "error": None,
            }
        )
    return rows


def test_req_verify_7479_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7479 and every task scenario exist before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7479" in text
    for suffix in ("ROSTER", "LENGTH", "ORDERS", "SHUFFLE", "ACCOUNTING", "RESUME", "READY", "E2E"):
        assert f"SCENARIO-VERIFY-7479-{suffix}" in text


def test_scenario_verify_7479_roster_reads_only_sealed_fit_roles() -> None:
    """SCENARIO-VERIFY-7479-ROSTER keeps the original 180 plus 60 order."""

    groups = exp.load_sealed_fit_groups(exp.REPO_ROOT)
    assert len(groups) == 240
    assert [row["roster_position"] for row in groups] == list(range(240))
    assert exp.role_counts(groups) == {"training": 180, "calibration_tuning": 60}
    assert {row["role"] for row in groups} == {"training", "calibration_tuning"}
    assert all(row["source_text"] and row["response_text"] for row in groups)
    assert all(row["annotation_provenance"] for row in groups)


def test_scenario_verify_7479_shuffle_is_frozen_label_independent_derangement() -> None:
    """SCENARIO-VERIFY-7479-SHUFFLE creates 40 gold-free control cells."""

    groups = exp.load_sealed_fit_groups(exp.REPO_ROOT)
    first = exp.build_capture_plan(groups)
    second = exp.build_capture_plan(groups)
    assert first == second
    assert len(first) == exp.FORWARD_BUDGET == 520
    main = [row for row in first if row["arm"] == "full_source_response"]
    controls = [row for row in first if row["arm"] == "shuffled_source"]
    assert len(main) == 480
    assert len(controls) == 40
    assert len({row["group_id"] for row in controls}) == 20
    assert all(row["source_group_id"] != row["group_id"] for row in controls)
    assert all(row["gold_label"] is None for row in controls)
    assert all("annotation" not in row for row in first)
    manifest = json.loads((exp.REPO_ROOT / exp.V654_RAW_DIR / "manifest.json").read_text())
    assert {row["group_id"] for row in controls} <= set(manifest["control_group_ids"])


def test_scenario_verify_7479_length_excludes_whole_group_without_replacement() -> None:
    """SCENARIO-VERIFY-7479-LENGTH retains positions and full input exclusions."""

    groups = _groups()
    plan = exp.build_capture_plan(groups, control_group_count=1)

    def tokenize(text: str, *, add_bos: bool) -> list[int]:
        count = 2_049 if "Source 1" in text else 20
        return list(range(count + int(add_bos)))

    qualified, cells = exp.apply_token_ceiling(groups, plan, tokenize=tokenize, ceiling=2_048)
    excluded = next(row for row in qualified if row["group_id"] == "group-1")
    assert excluded["eligible"] is False
    assert excluded["exclusion_reason"] == "complete_prompt_over_2048_tokens"
    assert excluded["roster_position"] == 1
    excluded_cells = [row for row in cells if row["group_id"] == "group-1"]
    assert excluded_cells
    assert all(
        row["disposition"] == "excluded" and row["attempted"] is False for row in excluded_cells
    )
    assert len(qualified) == len(groups)


def test_scenario_verify_7479_orders_remap_before_average() -> None:
    """SCENARIO-VERIFY-7479-ORDERS averages probabilities by stable ID."""

    original = {
        "option_order": list(exp.OPTION_IDS),
        "raw_logits_by_option_id": {"supported": 4.0, "contains_unsupported": 1.0},
    }
    reversed_row = {
        "option_order": list(reversed(exp.OPTION_IDS)),
        "raw_logits_by_option_id": {"contains_unsupported": 2.0, "supported": 3.0},
    }
    reduced = exp.average_order_rows(original, reversed_row)
    expected_a = exp.softmax_by_option(original["raw_logits_by_option_id"])
    expected_b = exp.softmax_by_option(reversed_row["raw_logits_by_option_id"])
    assert reduced["mapped_probabilities_by_order"] == [expected_a, expected_b]
    assert reduced["mean_probabilities_by_option_id"]["supported"] == pytest.approx(
        (expected_a["supported"] + expected_b["supported"]) / 2
    )
    with pytest.raises(exp.CaptureError, match="order_pair_invalid"):
        exp.average_order_rows(original, original)


def test_scenario_verify_7479_accounting_rejects_missing_and_nonfinite_outputs() -> None:
    """SCENARIO-VERIFY-7479-ACCOUNTING requires one terminal valid cell each."""

    plan = exp.build_capture_plan(_groups(), control_group_count=1)
    rows = _completed_rows(plan)
    reduced = exp.reduce_capture(plan, rows, minimums={"training": 2, "calibration_tuning": 1})
    assert reduced["capture_complete_score"] == 1
    assert reduced["confirmatory_support_score"] == 1
    assert reduced["all_eligible_calls_valid"] is True
    assert reduced["sample_size_budget"] == {
        "planned": 8,
        "attempted": 8,
        "complete": 8,
        "failed": 0,
        "censored": 0,
        "excluded": 0,
        "unstarted": 0,
    }

    missing = rows[:-1]
    incomplete = exp.reduce_capture(
        plan, missing, minimums={"training": 2, "calibration_tuning": 1}
    )
    assert incomplete["capture_complete_score"] == 0
    assert incomplete["sample_size_budget"]["unstarted"] == 1

    invalid = deepcopy(rows)
    invalid[0]["raw_logits_by_option_id"]["supported"] = float("nan")
    broken = exp.reduce_capture(plan, invalid, minimums={"training": 2, "calibration_tuning": 1})
    assert broken["capture_complete_score"] == 1
    assert broken["all_eligible_calls_valid"] is False


def test_scenario_verify_7479_resume_requires_identical_hash_bound_requests(tmp_path) -> None:
    """SCENARIO-VERIFY-7479-RESUME rejects changed schedule or model hashes."""

    plan = exp.build_capture_plan(_groups(), control_group_count=1)
    rows = _completed_rows(plan[:2])
    checkpoint = tmp_path / "checkpoint.json"
    binding = exp.checkpoint_binding(plan, "sha256:model", "sha256:tokenizer")
    exp.write_checkpoint(checkpoint, binding=binding, rows=rows)
    assert exp.load_checkpoint(checkpoint, expected_binding=binding) == rows
    changed = {**binding, "model_sha256": "sha256:changed"}
    with pytest.raises(exp.CaptureError, match="checkpoint_binding_mismatch"):
        exp.load_checkpoint(checkpoint, expected_binding=changed)
    checkpoint.write_text("{}", encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="checkpoint_schema_invalid"):
        exp.load_checkpoint(checkpoint, expected_binding=binding)


def test_raw_shards_rehash_and_independently_reduce(tmp_path) -> None:
    """SCENARIO-VERIFY-7479-E2E binds raw rows outside the headline artifact."""

    plan = exp.build_capture_plan(_groups(), control_group_count=1)
    rows = _completed_rows(plan)
    manifest = exp.write_raw_logit_shards(tmp_path, plan=plan, rows=rows)
    reloaded = exp.reload_raw_logit_shards(tmp_path, manifest)
    assert reloaded["rows"] == rows
    assert reloaded["plan_hash"] == exp.canonical_hash(plan)
    assert all(row["size_bytes"] < exp.MAX_ARTIFACT_BYTES for row in manifest)
    path = tmp_path / manifest[0]["path"]
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="raw_shard_hash_mismatch"):
        exp.reload_raw_logit_shards(tmp_path, manifest)


def test_scenario_verify_7479_ready_keeps_three_gate_categories_separate() -> None:
    """SCENARIO-VERIFY-7479-READY gives every acceptance gate a principle."""

    gates = exp.build_acceptance_gates(
        required_validity=True,
        capture_complete=True,
        support=True,
        benefit=False,
    )
    assert {row["category"] for row in gates} == {
        "required_validity",
        "readiness",
        "scientific_benefit",
    }
    assert all(row["passed"] is True and row["principle"] for row in gates)
    failed = exp.build_acceptance_gates(
        required_validity=True,
        capture_complete=False,
        support=True,
        benefit=False,
    )
    assert exp.gate_check_summary(failed)["failed_check"] == "capture_complete"


def test_terminal_fixture_validates_and_detects_drift(tmp_path) -> None:
    """SCENARIO-VERIFY-7479-E2E cold-reduces the exact raw rows and fields."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["fit_capture_ready_score"] == 1
    assert artifact["capture_complete_score"] == 1
    assert artifact["confirmatory_support_score"] == 1
    assert artifact["MODEL_SPECS"] == [exp.MODEL_HF_ID]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert set(exp.REQUIRED_FIELDS) <= set(artifact["field_principles"])

    drifted = deepcopy(artifact)
    drifted["fit_capture_ready_score"] = 0
    drifted["reproducibility_checksum"] = exp.artifact_checksum(drifted)
    assert "fit_capture_ready_score_mismatch" in exp.validate_artifact(
        drifted, root=tmp_path, require_validation=False
    )


def test_validation_manifest_is_frozen_to_affected_files() -> None:
    """SCENARIO-VERIFY-7479-E2E excludes broad Python-suite targets."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_defensive_boundaries_fail_closed(tmp_path) -> None:
    """REQ-VERIFY-7479 rejects malformed plans, rows, receipts, and artifacts."""

    with pytest.raises(exp.CaptureError, match="fit_role_counts_invalid"):
        exp.load_sealed_fit_groups(tmp_path)
    for relative in (exp.PREDICTOR_PATH, exp.EVALUATOR_PATH, exp.GROUP_PATH):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="fit_role_counts_invalid"):
        exp.load_sealed_fit_groups(tmp_path)
    with pytest.raises(exp.CaptureError, match="control_group_count_invalid"):
        exp.build_capture_plan(_groups(), control_group_count=2)
    with pytest.raises(exp.CaptureError, match="probability_logits_invalid"):
        exp.softmax_by_option({"supported": 1.0})
    assert exp.validate_artifact([], root=tmp_path, require_validation=False) == [
        "artifact_not_object"
    ]
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["validation_receipts"] = []
    assert "required_validation_failed" in exp.validate_artifact(
        artifact, root=tmp_path, require_validation=True
    )
    assert exp.validation_names_passed({}, ("one",)) is False
    assert exp.validation_names_passed([{"name": "one", "passed": True}], ("one",)) is True
    unreadable = tmp_path / "missing.json"
    assert exp.load_json_object(unreadable) == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("[", encoding="utf-8")
    assert exp.load_json_object(malformed) == {}


def test_independent_reader_and_validator_name_each_corruption(tmp_path) -> None:
    """SCENARIO-VERIFY-7479-E2E reports the exact broken evidence boundary."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.independent_reduce(
        {**artifact, "raw_logit_shards": None}, root=tmp_path, require_terminal=False
    )["errors"] == ["raw_logit_shards_invalid"]

    broken_hash = deepcopy(artifact)
    broken_hash["raw_logit_shards"][0]["sha256"] = "sha256:wrong"
    assert (
        "raw_shard_hash_mismatch"
        in exp.independent_reduce(broken_hash, root=tmp_path, require_terminal=False)["errors"][0]
    )

    wrong_reduction = deepcopy(artifact)
    wrong_reduction["capture_reduction"] = {}
    assert (
        "capture_reduction_mismatch"
        in exp.independent_reduce(wrong_reduction, root=tmp_path, require_terminal=False)["errors"]
    )

    wrong_rows = deepcopy(artifact)
    wrong_rows["rows"] = []
    assert (
        "group_rows_mismatch"
        in exp.independent_reduce(wrong_rows, root=tmp_path, require_terminal=False)["errors"]
    )

    variants: list[tuple[str, exp.JsonDict]] = []
    missing_events = deepcopy(artifact)
    missing_events["current_invocation_events"] = None
    variants.append(("invocation_evidence_invalid", missing_events))

    wrong_counts = deepcopy(artifact)
    wrong_counts["invocation_counts"]["forward_calls"]["completed"] -= 1
    variants.append(("invocation_counts_mismatch", wrong_counts))

    unfinished = deepcopy(artifact)
    unfinished["current_invocation_events"].pop()
    unfinished["invocation_counts"] = exp.pilot.reduce_invocation_events(
        unfinished["current_invocation_events"]
    )
    variants.append(("invocation_counts_unbalanced", unfinished))

    wrong_complete = deepcopy(artifact)
    wrong_complete["capture_complete_score"] = 0
    variants.append(("capture_complete_score_mismatch", wrong_complete))

    wrong_support = deepcopy(artifact)
    wrong_support["confirmatory_support_score"] = 0
    variants.append(("confirmatory_support_score_mismatch", wrong_support))

    bad_gates = deepcopy(artifact)
    bad_gates["acceptance_gate_results"] = None
    variants.append(("acceptance_gates_invalid", bad_gates))

    bad_summary = deepcopy(artifact)
    bad_summary["gate_check_summary"] = {}
    variants.append(("gate_summary_mismatch", bad_summary))

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    variants.append(("field_principles_invalid", bad_principles))

    for expected, value in variants:
        value["reproducibility_checksum"] = exp.artifact_checksum(value)
        assert expected in exp.validate_artifact(value, root=tmp_path, require_validation=False)

    plan = exp.build_capture_plan(_groups(), control_group_count=1)
    invalid_probability = _completed_rows(plan)
    invalid_probability[0]["probabilities_by_option_id"]["supported"] = 0.0
    assert (
        exp.reduce_capture(
            plan,
            invalid_probability,
            minimums={"training": 2, "calibration_tuning": 1},
        )["all_eligible_calls_valid"]
        is False
    )


def test_thin_entrypoint_parses_fixed_date() -> None:
    """REQ-VERIFY-7479 exposes the declared capability command modes."""

    args = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"])
    assert args.date == exp.RUN_DATE
    assert args.cold_replay == exp.Path("candidate.json")
    with pytest.raises(SystemExit, match="--date must be"):
        exp.main(["--date", "20260920", "--cold-replay", "candidate.json"])


def test_reader_cli_modes_use_fresh_artifact_bytes(tmp_path) -> None:
    """SCENARIO-VERIFY-7479-E2E exercises both fresh-process reader modes."""

    artifact = exp.build_artifact_for_test(tmp_path)
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    common = ["--date", exp.RUN_DATE, "--root", str(tmp_path)]
    assert exp.main([*common, "--cold-replay", str(candidate)]) == 0
    assert exp.main([*common, "--independent-reduce", str(candidate)]) == 0
    assert exp.main([*common, "--cold-replay", str(tmp_path / "missing.json")]) == 1
    assert exp.main([*common, "--independent-reduce", str(tmp_path / "missing.json")]) == 1
