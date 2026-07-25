"""Tests for Exp5920 prospective event-stream admission.

Spec refs: REQ-LEARN-5920, SCENARIO-LEARN-5920-SCHEMA,
SCENARIO-LEARN-5920-REPLAY, SCENARIO-LEARN-5920-TAMPER,
SCENARIO-LEARN-5920-BOUNDARY, REQ-HARNESS-5920,
SCENARIO-HARNESS-5920.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5909_sota_constraint_synthesis_ab as exp5909
from carnot import experiment_5920_prospective_event_stream_admission as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
HARNESS_SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _task_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.TASK_OWNED_COMMANDS} | {mod.GLOBAL_PYTEST_COMMAND: 2}


def test_req_learn_5920_spec_declares_schema_boundary_and_principles() -> None:
    section = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = section[section.index("## REQ-LEARN-5920") :]
    harness = HARNESS_SPEC.read_text(encoding="utf-8")
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5920",
        "SCENARIO-LEARN-5920-SCHEMA",
        "SCENARIO-LEARN-5920-REPLAY",
        "SCENARIO-LEARN-5920-TAMPER",
        "SCENARIO-LEARN-5920-BOUNDARY",
        "REQ-HARNESS-5920",
        "SCENARIO-HARNESS-5920",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`global_suite_failure_delta<=0`",
    ):
        assert marker in section or marker in harness
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5920_schema_materializes_fresh_prefix_chained_stream(
    tmp_path: Path,
) -> None:
    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    row_output = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    artifact = mod.write_admission_artifact(
        output_path=output,
        row_output_path=row_output,
        duration_s=0.0,
        test_exit_codes=_task_exit_codes(),
        global_after_node_ids=mod.BASELINE_GLOBAL_NODE_IDS,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    rows = _load_rows(row_output)

    assert loaded == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["retired_scope_not_reopened"] is True
    assert artifact["inference_substrate"] == "deterministic_artifact_replay_no_llm"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["prospective_stream_admission_ready_score"] == pytest.approx(1.0)
    assert artifact["global_suite_baseline_and_failure_delta"]["baseline_node_count"] == 116
    assert artifact["global_suite_baseline_and_failure_delta"]["failure_delta"] == 0
    assert artifact["global_suite_baseline_and_failure_delta"]["ready_allowed"] is True
    assert artifact["task_owned_test_boundary"]["all_task_owned_commands_clean"] is True

    stream = artifact["fresh_stream_path_hash_row_count_and_prefix_chain"]
    assert stream["path"].endswith(mod.ROW_FILE_RELATIVE_PATH.name)
    assert stream["sha256"] == mod.sha256_file(row_output)
    assert stream["row_count"] == len(rows) == 198
    assert stream["final_prefix_checksum"] == rows[-1]["prefix_checksum"]
    assert stream["prefix_chain_valid"] is True

    first = rows[0]
    assert first["schema"] == mod.ROW_SCHEMA_VERSION
    assert first["event_id"] == "exp5920-event-000000"
    assert first["causal_sequence_index"] == 0
    assert first["prior_prefix_checksum"] == mod.GENESIS_PREFIX_CHECKSUM
    assert first["prompt_visibility"]["future_label_visible_to_model"] is False
    assert first["prompt_visibility"]["target_exact_labels_exposed_to_prompt"] is False
    assert first["split"] in {"train", "dev", "heldout"}
    assert first["exact_label_projection"] == first["exact_diagnostic_and_label"]["exact_labels"]
    assert first["source_artifact_hashes"]["exp5909_raw_stream_sha256"] == mod.sha256_file(
        REPO / exp5909.RAW_STREAM_RELATIVE_PATH
    )
    assert first["row_hash"] == mod.event_row_hash(first)
    assert first["prefix_checksum"] == mod.prefix_checksum(
        first["prior_prefix_checksum"], first["row_hash"]
    )

    receipt = mod.replay_stream(row_output)
    assert receipt["ok"] is True
    assert receipt["row_count"] == stream["row_count"]
    assert receipt["final_prefix_checksum"] == stream["final_prefix_checksum"]


def test_scenario_learn_5920_fresh_process_replay_accepts_stream(tmp_path: Path) -> None:
    row_output = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    artifact = mod.write_admission_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_output_path=row_output,
        duration_s=0.0,
        test_exit_codes=_task_exit_codes(),
        global_after_node_ids=mod.BASELINE_GLOBAL_NODE_IDS,
    )

    fresh = mod.run_fresh_process_replay(row_output)

    assert fresh["ok"] is True
    assert fresh["returncode"] == 0
    assert (
        fresh["row_count"]
        == artifact["fresh_stream_path_hash_row_count_and_prefix_chain"]["row_count"]
    )
    assert (
        fresh["final_prefix_checksum"]
        == artifact["fresh_stream_path_hash_row_count_and_prefix_chain"]["final_prefix_checksum"]
    )
    assert fresh["stdout_sha256"].startswith("sha256:")
    assert fresh["stderr_sha256"].startswith("sha256:")


def test_scenario_learn_5920_tamper_matrix_rejects_without_partial_promotions() -> None:
    matrix = mod.run_tamper_matrix()

    assert {case["component"] for case in matrix["cases"]} == {
        "chronology_reordered_row",
        "duplicate_event_id",
        "future_label_visibility",
        "exact_label_posthoc_relabel",
        "source_hash_drift",
        "split_drift",
    }
    assert matrix["all_rejected"] is True
    assert matrix["partial_promotions"] == 0
    assert all(case["rejected"] is True for case in matrix["cases"])
    assert all(case["partial_promotions"] == 0 for case in matrix["cases"])


def test_scenario_learn_5920_validation_rejects_each_boundary_mutation() -> None:
    rows = mod.build_event_rows()
    mod.validate_event_rows(rows)

    cases = {
        "duplicate event id": lambda item: item[1].update({"event_id": item[0]["event_id"]}),
        "chronology": lambda item: item.__setitem__(0, item.pop(1)),
        "future label": lambda item: item[0]["prompt_visibility"].update(
            {"future_label_visible_to_model": True}
        ),
        "post-hoc exact label": lambda item: item[0]["exact_label_projection"].update(
            {"parse_valid": not item[0]["exact_label_projection"]["parse_valid"]}
        ),
        "source hash drift": lambda item: item[0]["source_artifact_hashes"].update(
            {"exp5909_raw_stream_sha256": "sha256:" + "0" * 64}
        ),
        "source hash": lambda item: item[0]["source_row"].update({"source_row_id": "wrong"}),
        "split drift": lambda item: item[0].update(
            {"split": "heldout" if item[0]["split"] != "heldout" else "train"}
        ),
        "row hash": lambda item: item[0].update({"row_hash": "sha256:" + "1" * 64}),
        "prefix": lambda item: item[0].update({"prior_prefix_checksum": "sha256:" + "2" * 64}),
        "prefix chain": lambda item: item[0].update({"prefix_checksum": "sha256:" + "4" * 64}),
        "model hash": lambda item: item[0]["model_identity"].update(
            {"model_file_sha256": "sha256:" + "3" * 64}
        ),
    }

    for message, mutate in cases.items():
        broken = deepcopy(rows)
        mutate(broken)
        with pytest.raises(mod.ProspectiveEventStreamError, match=message):
            mod.validate_event_rows(broken)


def test_scenario_harness_5920_global_delta_and_artifact_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = mod.global_suite_baseline()
    assert baseline["baseline_node_count"] == 116
    assert baseline["source"] == mod.EXP5912_RESULT_RELATIVE_PATH.as_posix()

    unchanged = mod.global_suite_delta(mod.BASELINE_GLOBAL_NODE_IDS)
    assert unchanged["failure_delta"] == 0
    assert unchanged["new_node_ids"] == []
    assert unchanged["ready_allowed"] is True

    amplified = mod.global_suite_delta([*mod.BASELINE_GLOBAL_NODE_IDS, "new::node"])
    assert amplified["failure_delta"] == 1
    assert amplified["new_node_ids"] == ["new::node"]
    assert amplified["ready_allowed"] is False

    blocked = mod.write_admission_artifact(
        output_path=tmp_path / "blocked.json",
        row_output_path=tmp_path / "blocked.rows.jsonl",
        duration_s=0.0,
        test_exit_codes=_task_exit_codes(),
        global_after_node_ids=[*mod.BASELINE_GLOBAL_NODE_IDS, "new::node"],
    )
    assert blocked["status"] == "blocked"
    assert blocked["prospective_stream_admission_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    missing = dict(blocked)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(blocked)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(blocked)
    bad_verifier["verifier_is_oracle"] = False
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    bad_provenance_type = deepcopy(blocked)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(blocked)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_score = deepcopy(blocked)
    bad_score["prospective_stream_admission_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_verdict = deepcopy(blocked)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_status = deepcopy(blocked)
    bad_status["status"] = "complete_ready"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_checksum = deepcopy(blocked)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    retired = deepcopy(blocked)
    retired["retired_scope_not_reopened"] = False
    assert mod.status(retired) == "retired"
    assert mod.honest_verdict(retired).startswith("retired:")

    reasons_artifact = {
        "preconditions_checked": {"preconditions_ready": True},
        "retired_scope_not_reopened": True,
        "immutable_upstream_hashes": {"unchanged": True},
        "task_owned_test_boundary": {"all_task_owned_commands_clean": True},
        "global_suite_baseline_and_failure_delta": {"ready_allowed": True},
        "protected_files_unchanged": {"unchanged": True},
    }
    assert mod._blocked_reasons(reasons_artifact) == ["ready_score"]

    many_reasons = deepcopy(blocked)
    many_reasons["preconditions_checked"]["preconditions_ready"] = False
    many_reasons["retired_scope_not_reopened"] = False
    many_reasons["immutable_upstream_hashes"]["unchanged"] = False
    many_reasons["task_owned_test_boundary"]["all_task_owned_commands_clean"] = False
    many_reasons["protected_files_unchanged"]["unchanged"] = False
    assert {
        "preconditions",
        "retired_scope_reopened",
        "immutable_upstream_hashes",
        "task_owned_test_boundary",
        "global_suite_failure_delta",
        "protected_files",
    } <= set(mod._blocked_reasons(many_reasons))

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod.read_json(scalar_json)

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text('{"ok": true}\n\n', encoding="utf-8")
    assert mod.load_jsonl(blank_jsonl) == [{"ok": True}]

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(mod.ProspectiveEventStreamError, match="JSONL row object"):
        mod.load_jsonl(bad_jsonl)

    assert mod.canonical_json({"tuple": (1, 2)}) == '{"tuple":[1,2]}'
    with pytest.raises(mod.ProspectiveEventStreamError, match="finite"):
        mod.canonical_json({"bad": float("nan")})
    with pytest.raises(mod.ProspectiveEventStreamError, match="unsupported JSON"):
        mod.canonical_json({"bad": object()})

    baseline_root = tmp_path / "baseline"
    baseline_path = baseline_root / mod.EXP5912_RESULT_RELATIVE_PATH
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps({"current_failure_node_ids_phases_and_ownership": {"failures": []}}),
        encoding="utf-8",
    )
    assert mod._baseline_node_ids(baseline_root) == []

    eligible_rows = deepcopy(mod.build_event_rows())
    eligible_rows[0]["commit_eligibility"]["eligible"] = True
    prefix = mod.GENESIS_PREFIX_CHECKSUM
    for row in eligible_rows:
        row["prior_prefix_checksum"] = prefix
        row["row_hash"] = mod.event_row_hash(row)
        prefix = mod.prefix_checksum(prefix, row["row_hash"])
        row["prefix_checksum"] = prefix
    monkeypatch.setattr(mod, "build_event_rows", lambda root=mod.REPO_ROOT: deepcopy(eligible_rows))
    assert mod.validate_event_rows(deepcopy(eligible_rows))["eligible_commit_count"] >= 1

    monkeypatch.setattr(mod, "validate_event_rows", lambda rows: {"ok": True})
    false_negative_matrix = mod.run_tamper_matrix()
    assert false_negative_matrix["all_rejected"] is False
    assert false_negative_matrix["partial_promotions"] == 6
