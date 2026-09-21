"""Tests for REQ-VERIFY-7480 and SCENARIO-VERIFY-7480-*.

The tests use sealed text and controlled logits. They never load a model.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from carnot import experiment_7480_v655_source_eval_capture as exp


def _groups() -> list[exp.JsonDict]:
    """Build a small three-role panel for reducer tests."""

    roles = ("internal_test", "internal_test", "online", "external")
    return [
        {
            "roster_position": index,
            "group_id": f"group-{index}",
            "group_hash": exp.canonical_hash(f"group-{index}"),
            "source_hash": exp.canonical_hash(f"source-{index}"),
            "response_hash": exp.canonical_hash(f"response-{index}"),
            "source_text": f"Source {index}",
            "response_text": f"Response {index}",
            "role": role,
            "annotation_disposition": "supported",
            "label": 1,
            "annotation_provenance": "pinned human annotations",
        }
        for index, role in enumerate(roles)
    ]


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


def test_req_verify_7480_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7480 and every task scenario exist before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7480" in text
    for suffix in ("ROSTER", "LENGTH", "ORDERS", "SHUFFLE", "ACCOUNTING", "RESUME", "READY", "E2E"):
        assert f"SCENARIO-VERIFY-7480-{suffix}" in text


def test_scenario_verify_7480_roster_reads_only_sealed_eval_roles() -> None:
    """SCENARIO-VERIFY-7480-ROSTER keeps all 294 held-out groups in order."""

    groups = exp.load_sealed_evaluation_groups(exp.REPO_ROOT)
    assert len(groups) == 294
    assert [row["roster_position"] for row in groups] == list(range(240, 534))
    assert exp.role_counts(groups) == {"internal_test": 60, "online": 160, "external": 74}
    assert {row["role"] for row in groups} == set(exp.EVALUATION_ROLES)
    assert all(row["source_text"] and row["response_text"] for row in groups)
    assert all(row["annotation_provenance"] for row in groups)


def test_scenario_verify_7480_shuffle_is_frozen_label_independent_derangement() -> None:
    """SCENARIO-VERIFY-7480-SHUFFLE creates 40 gold-free control cells."""

    groups = exp.load_sealed_evaluation_groups(exp.REPO_ROOT)
    first = exp.build_capture_plan(groups)
    assert first == exp.build_capture_plan(groups)
    assert len(first) == exp.FORWARD_BUDGET == 628
    main = [row for row in first if row["arm"] == "full_source_response"]
    controls = [row for row in first if row["arm"] == "shuffled_source"]
    assert len(main) == exp.MAIN_FORWARD_BUDGET == 588
    assert len(controls) == exp.CONTROL_FORWARD_BUDGET == 40
    assert len({row["group_id"] for row in controls}) == 20
    assert {row["role"] for row in controls} == {"internal_test"}
    assert all(row["source_group_id"] != row["group_id"] for row in controls)
    assert all(row["gold_label"] is None for row in controls)
    assert all("annotation" not in row for row in first)


def test_scenario_verify_7480_length_excludes_whole_group_without_replacement() -> None:
    """SCENARIO-VERIFY-7480-LENGTH retains positions and full-input exclusions."""

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
    assert all(row["disposition"] == "excluded" for row in excluded_cells)
    assert len(qualified) == len(groups)


def test_scenario_verify_7480_orders_remap_before_average() -> None:
    """SCENARIO-VERIFY-7480-ORDERS averages probabilities by stable ID."""

    original = {
        "option_order": list(exp.OPTION_IDS),
        "raw_logits_by_option_id": {"supported": 4.0, "contains_unsupported": 1.0},
    }
    reversed_row = {
        "option_order": list(reversed(exp.OPTION_IDS)),
        "raw_logits_by_option_id": {"contains_unsupported": 2.0, "supported": 3.0},
    }
    reduced = exp.average_order_rows(original, reversed_row)
    first = exp.softmax_by_option(original["raw_logits_by_option_id"])
    second = exp.softmax_by_option(reversed_row["raw_logits_by_option_id"])
    assert reduced["mapped_probabilities_by_order"] == [first, second]
    assert reduced["mean_probabilities_by_option_id"]["supported"] == pytest.approx(
        (first["supported"] + second["supported"]) / 2
    )


def test_scenario_verify_7480_accounting_rejects_missing_and_nonfinite_outputs() -> None:
    """SCENARIO-VERIFY-7480-ACCOUNTING requires one terminal valid cell each."""

    plan = exp.build_capture_plan(_groups(), control_group_count=1)
    rows = _completed_rows(plan)
    minimums = {"internal_test": 2, "online": 1, "external": 1}
    reduced = exp.reduce_capture(plan, rows, minimums=minimums)
    assert reduced["capture_complete_score"] == 1
    assert reduced["confirmatory_support_score"] == 1
    assert reduced["all_eligible_calls_valid"] is True
    assert reduced["sample_size_budget"] == {
        "planned": 10,
        "attempted": 10,
        "complete": 10,
        "failed": 0,
        "censored": 0,
        "excluded": 0,
        "unstarted": 0,
    }
    incomplete = exp.reduce_capture(plan, rows[:-1], minimums=minimums)
    assert incomplete["capture_complete_score"] == 0
    assert incomplete["sample_size_budget"]["unstarted"] == 1
    invalid = deepcopy(rows)
    invalid[0]["raw_logits_by_option_id"]["supported"] = float("nan")
    broken = exp.reduce_capture(plan, invalid, minimums=minimums)
    assert broken["capture_complete_score"] == 1
    assert broken["all_eligible_calls_valid"] is False


def test_scenario_verify_7480_resume_requires_identical_hash_bound_requests(tmp_path) -> None:
    """SCENARIO-VERIFY-7480-RESUME rejects changed schedule or model hashes."""

    plan = exp.build_capture_plan(_groups(), control_group_count=1)
    rows = _completed_rows(plan[:2])
    checkpoint = tmp_path / "checkpoint.json"
    binding = exp.checkpoint_binding(plan, "sha256:model", "sha256:tokenizer")
    exp.write_checkpoint(checkpoint, binding=binding, rows=rows)
    assert exp.load_checkpoint(checkpoint, expected_binding=binding) == rows
    with pytest.raises(exp.CaptureError, match="checkpoint_binding_mismatch"):
        exp.load_checkpoint(
            checkpoint,
            expected_binding={**binding, "model_sha256": "sha256:changed"},
        )
    checkpoint.write_text("{}", encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="checkpoint_schema_invalid"):
        exp.load_checkpoint(checkpoint, expected_binding=binding)


def test_raw_shards_rehash_and_independently_reduce(tmp_path) -> None:
    """SCENARIO-VERIFY-7480-E2E binds raw rows outside the headline artifact."""

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


def test_scenario_verify_7480_ready_keeps_gate_categories_separate() -> None:
    """SCENARIO-VERIFY-7480-READY gives every acceptance gate a principle."""

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
    """SCENARIO-VERIFY-7480-E2E cold-reduces the exact raw rows and fields."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["evaluation_capture_ready_score"] == 1
    assert artifact["capture_complete_score"] == 1
    assert artifact["confirmatory_support_score"] == 1
    assert artifact["MODEL_SPECS"] == [exp.MODEL_HF_ID]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert set(exp.REQUIRED_FIELDS) <= set(artifact["field_principles"])
    drifted = deepcopy(artifact)
    drifted["evaluation_capture_ready_score"] = 0
    drifted["reproducibility_checksum"] = exp.artifact_checksum(drifted)
    assert "evaluation_capture_ready_score_mismatch" in exp.validate_artifact(
        drifted, root=tmp_path, require_validation=False
    )


def test_validation_manifest_is_frozen_to_affected_files() -> None:
    """SCENARIO-VERIFY-7480-E2E excludes broad Python-suite targets."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_defensive_readers_and_cli_fail_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7480 rejects malformed evidence and exposes cold readers."""

    with pytest.raises(exp.CaptureError, match="evaluation_role_counts_invalid"):
        exp.load_sealed_evaluation_groups(tmp_path)
    monkeypatch.setattr(exp, "SEALED_ROLE_COUNTS", {"internal_test": 59})
    with pytest.raises(exp.CaptureError, match="evaluation_role_counts_invalid"):
        exp.load_sealed_evaluation_groups(exp.REPO_ROOT)
    monkeypatch.setattr(
        exp,
        "SEALED_ROLE_COUNTS",
        {"internal_test": 60, "online": 160, "external": 74},
    )
    with pytest.raises(exp.CaptureError, match="control_group_count_invalid"):
        exp.build_capture_plan(_groups(), control_group_count=3)
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
    assert exp.validate_artifact([], root=tmp_path, require_validation=False) == [
        "artifact_not_object"
    ]
    assert exp.validation_names_passed({}, ("focused_pytest",)) is False
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    common = ["--date", exp.RUN_DATE, "--root", str(tmp_path)]
    assert exp.main([*common, "--cold-replay", str(candidate)]) == 0
    assert exp.main([*common, "--independent-reduce", str(candidate)]) == 0
    assert exp.main([*common, "--cold-replay", str(tmp_path / "missing.json")]) == 1
    assert exp.main([*common, "--independent-reduce", str(tmp_path / "missing.json")]) == 1
    with pytest.raises(SystemExit, match="--date must be"):
        exp.main(["--date", "20260920", "--cold-replay", str(candidate)])
