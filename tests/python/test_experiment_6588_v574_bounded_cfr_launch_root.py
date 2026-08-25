"""Test the V574 bounded CFR launch root without model inference.

Spec refs: REQ-REPORT-6588, REQ-REPORT-6588-PRECONDITIONS,
REQ-REPORT-6588-REPLAY, REQ-REPORT-6588-CACHE,
REQ-REPORT-6588-BUDGETS, REQ-REPORT-6588-GATES,
REQ-REPORT-6588-AUTHORITY, REQ-REPORT-6588-ATTACKS,
REQ-REPORT-6588-REDUCER, REQ-REPORT-6588-ATOMIC,
SCENARIO-REPORT-6588-REPLAY, SCENARIO-REPORT-6588-CACHE,
SCENARIO-REPORT-6588-BUDGETS, SCENARIO-REPORT-6588-GATES,
SCENARIO-REPORT-6588-ATTACKS, SCENARIO-REPORT-6588-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6588_v574_bounded_cfr_launch_root as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TESTS_RUN = [{"command": "focused Exp6588 fixture", "exit_code": 0, "duration_s": 0.01}]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Build the source-only launch report once for all focused checks."""

    return mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_report_6588_spec_declares_launch_contract() -> None:
    """REQ-REPORT-6588 exists before implementation and names every field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6588") :]
    anchors = (
        "REQ-REPORT-6588-PRECONDITIONS",
        "REQ-REPORT-6588-REPLAY",
        "REQ-REPORT-6588-CACHE",
        "REQ-REPORT-6588-BUDGETS",
        "REQ-REPORT-6588-GATES",
        "REQ-REPORT-6588-AUTHORITY",
        "REQ-REPORT-6588-ATTACKS",
        "REQ-REPORT-6588-REDUCER",
        "REQ-REPORT-6588-ATOMIC",
        "SCENARIO-REPORT-6588-REPLAY",
        "SCENARIO-REPORT-6588-CACHE",
        "SCENARIO-REPORT-6588-BUDGETS",
        "SCENARIO-REPORT-6588-GATES",
        "SCENARIO-REPORT-6588-ATTACKS",
        "SCENARIO-REPORT-6588-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    )
    for anchor in anchors:
        assert anchor in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6588_raw_v573_contracts_recompute(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6588-REPLAY ignores stored ready-score shortcuts."""

    rows = {row["experiment_id"]: row for row in report["v573_terminal_replay_rows"]}
    exp6585 = rows["Exp6585"]
    exp6586 = rows["Exp6586"]
    exp6587 = rows["Exp6587"]

    assert exp6585["stored_ready_score"] == 1.0
    assert exp6585["recomputed_ready_score"] == 1.0
    assert exp6587["stored_ready_score"] == 1.0
    assert exp6587["recomputed_ready_score"] == 1.0
    assert exp6587["source_gate_summary_checks_closed"] is False
    assert exp6587["science_result_created"] is False

    source6586 = json.loads((REPO / mod.V573_ARTIFACT_PATHS["Exp6586"]).read_text())
    assert exp6586["honest_verdict"] == source6586["honest_verdict"]
    assert exp6586["verdict_class"] == source6586["verdict_class"]
    assert exp6586["gate_check_summary"] == source6586["gate_check_summary"]
    assert exp6586["adversarial_disposition"]["flagged_adversarial"] is True
    assert exp6586["adversarial_disposition"]["claim_audit_disposition"] == (
        "SKIPPED_ALREADY_FLAGGED"
    )
    assert exp6586["science_launch_gate"] is False


def test_req_report_6588_replayers_ignore_stored_scores_and_fail_on_raw_drift() -> None:
    """REQ-REPORT-6588-REPLAY uses raw rows, not stored aggregate fields."""

    exp6585 = mod.load_json(REPO / mod.V573_ARTIFACT_PATHS["Exp6585"])
    exp6587 = mod.load_json(REPO / mod.V573_ARTIFACT_PATHS["Exp6587"])
    exp6585["v573_execution_contract_ready_score"] = 0.0
    exp6587["v573_constraint_first_method_ready_score"] = 0.0
    assert mod.recompute_exp6585_readiness(exp6585)["ready_score"] == 1.0
    assert mod.recompute_exp6587_readiness(exp6587)["ready_score"] == 1.0

    exp6585["v573_execution_budget_contract"][0]["max_model_processes"] = 2
    assert mod.recompute_exp6585_readiness(exp6585)["ready_score"] == 0.0
    exp6587["source_binding_and_exact_authority_contract"]["llm_release_authority"] = True
    assert mod.recompute_exp6587_readiness(exp6587)["ready_score"] == 0.0


def test_scenario_report_6588_cache_identity_is_content_derived(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6588-CACHE resolves both blobs without loading them."""

    rows = {row["repository_id"]: row for row in report["model_cache_identity_rows"]}
    assert set(rows) == set(mod.MANDATED_MODEL_IDS)
    for repository_id, row in rows.items():
        assert row["resolved"] is True
        assert row["admitted"] is True
        assert row["repository_id"] == repository_id
        assert row["content_metadata"]["magic"] == "GGUF"
        assert row["content_metadata"]["is_language_model"] is True
        assert row["content_metadata"]["tokenizer_metadata"]["token_count"] > 0
        assert row["content_metadata"]["bounded_read_receipt"]["tensor_payload_bytes_read"] == 0
        assert row["provenance"]["valid"] is True
        assert row["provenance"]["trusted_hash_matches_blob_key"] is True
        assert row["model_load_performed"] is False
        assert row["download_performed"] is False
        assert row["auto_tokenizer_used"] is False


def test_req_report_6588_cache_resolver_fails_closed_without_source_identity(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6588-CACHE rejects a missing trusted cache receipt."""

    source = tmp_path / mod.GGUF_IDENTITY_SOURCE_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_text('{"gguf_blob_metadata_rows": []}\n', encoding="utf-8")
    rows = mod.build_model_cache_identity_rows(tmp_path)
    assert len(rows) == 2
    assert all(row["resolved"] is False for row in rows)
    assert {row["rejection_reason"] for row in rows} == {"trusted_cache_identity_missing"}
    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod.load_json(non_object)
    assert mod._required_artifact_fields("no field marker") == set()  # noqa: SLF001


def test_scenario_report_6588_budgets_freeze_one_model_residency(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6588-BUDGETS freezes bounded owned processes."""

    rows = report["execution_budget_contract"]
    assert mod.execution_budget_contract_ready(rows)
    assert {row["task_id"] for row in rows} == {
        "exp6590-qwen36-constraint-first-stream",
        "exp6591-gemma4-31b-constraint-first-stream",
    }
    assert all(row["max_model_processes"] == 1 for row in rows)
    assert all(row["max_concurrent_model_families"] == 1 for row in rows)
    assert all(row["checkpoint_interval_units"] == 1 for row in rows)
    assert all(row["hard_timeout_s"] < row["conductor_hard_cap_s"] for row in rows)
    assert all(row["runtime_select_idle_rtx_3090"] is True for row in rows)
    assert all(row["kill_only_owned_child_process_group"] is True for row in rows)
    assert all(row["terminal_output_required_on_failure"] is True for row in rows)
    assert all(row["atomic_terminal_output"] is True for row in rows)

    mutated = deepcopy(rows)
    mutated[0]["max_concurrent_model_families"] = 2
    assert not mod.execution_budget_contract_ready(mutated)
    mutated = deepcopy(rows)
    mutated[1]["checkpoint_interval_units"] = 0
    assert not mod.execution_budget_contract_ready(mutated)


def test_scenario_report_6588_gate_fields_match_active_roadmap(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6588-GATES binds exact V574 owner fields."""

    rows = report["current_roadmap_gate_contract_rows"]
    assert mod.current_roadmap_gate_contract_ready(rows)
    assert {row["consumer_task_id"] for row in rows} == {
        "exp6590-qwen36-constraint-first-stream",
        "exp6591-gemma4-31b-constraint-first-stream",
    }
    assert {row["artifact_field"] for row in rows} == {mod.READY_FIELD}
    assert {row["owner_output_field"] for row in rows} == {
        "qwen_cfr_rows_ready_score",
        "gemma31_cfr_rows_ready_score",
    }
    assert all(row["upstream_task_id"] == mod.TASK_ID for row in rows)
    assert all(row["all_cross_references_close"] is True for row in rows)

    mutated = deepcopy(rows)
    mutated[0]["artifact_field"] = "v574_cfr_launch_ready_scor"
    assert not mod.current_roadmap_gate_contract_ready(mutated)


def test_scenario_report_6588_attacks_fail_closed(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6588-ATTACKS closes every declared shortcut."""

    attacks = report["attack_rows"]
    assert [row["attack_id"] for row in attacks] == list(mod.REQUIRED_ATTACK_IDS)
    assert all(row["passed"] is True for row in attacks)
    assert all(row["candidate_ready_score"] == 0.0 for row in attacks)
    assert mod.readiness_reducer(report)["ready_score"] == 1.0

    candidate = deepcopy(report)
    candidate["v573_terminal_replay_rows"][0]["science_result_created"] = True
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["suite_green_launch_requirement"] = True
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["model_cache_identity_rows"][0]["source_artifact_sha256"] = "missing"
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["execution_budget_contract"][0]["max_model_processes"] = 2
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["model_cache_identity_rows"][0]["repository_id"] = "Qwen/Qwen3.5-0.8B"
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["model_cache_identity_rows"][0]["auto_tokenizer_used"] = True
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["model_cache_identity_rows"][0]["download_performed"] = True
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["v573_terminal_replay_rows"] = candidate["v573_terminal_replay_rows"][:-1]
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["current_roadmap_gate_contract_rows"][0]["artifact_field"] = "drifted"
    assert mod.readiness_reducer(candidate)["ready_score"] == 0.0


def test_req_report_6588_preconditions_name_resources_ownership_and_no_llm(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6588-PRECONDITIONS records every launch input."""

    preconditions = report["preconditions_checked"]
    assert preconditions["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert preconditions["llm_calls_issued"] == 0
    assert preconditions["model_loads_issued"] == 0
    assert preconditions["downloads_issued"] == 0
    assert preconditions["roadmap"]["sha256"].startswith("sha256:")
    assert len(preconditions["v573_artifacts"]) == 3
    assert all(row["sha256"].startswith("sha256:") for row in preconditions["v573_artifacts"])
    assert preconditions["cpu"]["count"] >= 1
    assert preconditions["ram"]["total_kib"] > 0
    assert preconditions["disk"]["total_bytes"] > 0
    assert preconditions["dirty_worktree"]["status_sha256"].startswith("sha256:")
    assert isinstance(preconditions["visible_gpu_ownership"]["gpu_rows"], list)
    assert preconditions["visible_gpu_ownership"]["signals_sent"] == []
    assert len(preconditions["local_gguf_cache_paths"]) == 2


def test_scenario_report_6588_atomic_null_artifact_validates(
    tmp_path: Path, report: dict[str, Any]
) -> None:
    """SCENARIO-REPORT-6588-ATOMIC writes one durable null artifact."""

    output = tmp_path / "experiment_6588.json"
    receipt = mod.atomic_write_report(output, report)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == report
    assert receipt["atomic_replace"] is True
    assert receipt["file_fsync"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["output_sha256"] == mod.sha256_file(output)
    assert report["status"] == "complete_v574_cfr_launch_ready"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] is None
    assert report[mod.READY_FIELD] == 1.0
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["reproducibility_checksum"] == mod.artifact_checksum(report)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_report(report) == []


def test_req_report_6588_validator_rejects_tamper_and_bad_terminal_fields(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """REQ-REPORT-6588-ATOMIC rejects checksum and authority mutation."""

    cases = []
    candidate = deepcopy(report)
    del candidate["status"]
    cases.append((candidate, "missing required fields: status"))
    cases.append(
        (
            _rehash({**deepcopy(report), "inference_substrate": "wrong"}),
            "inference_substrate mismatch",
        )
    )
    cases.append(
        (
            _rehash({**deepcopy(report), "verifier_is_oracle": False}),
            "verifier_is_oracle must be true",
        )
    )
    cases.append((_rehash({**deepcopy(report), "duration_s": 0.0}), "duration_s must be positive"))
    cases.append(
        (_rehash({**deepcopy(report), mod.READY_FIELD: 0.0}), f"{mod.READY_FIELD} mismatch")
    )
    cases.append(
        (
            _rehash({**deepcopy(report), "verdict_class": "positive"}),
            "ready launch verdict_class must be null",
        )
    )
    candidate = deepcopy(report)
    candidate["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((_rehash(candidate), "protected_files_unchanged failed"))
    candidate = deepcopy(report)
    candidate["field_provenance"].pop("status")
    cases.append((_rehash(candidate), "field_provenance missing required fields"))
    cases.append(
        (
            {**deepcopy(report), "reproducibility_checksum": "sha256:stale"},
            "reproducibility_checksum mismatch",
        )
    )
    cases.append(
        (
            _rehash({**deepcopy(report), "honest_verdict": "launch ready without prefix"}),
            "terminal success prefix missing",
        )
    )
    candidate = deepcopy(report)
    candidate["model_cache_identity_rows"][0]["resolved"] = False
    candidate[mod.READY_FIELD] = 0.0
    candidate["verdict_class"] = "blocked"
    candidate["gate_check_summary"] = {}
    cases.append((_rehash(candidate), "blocked gate_check_summary missing failure"))

    for candidate, expected in cases:
        assert expected in mod.validate_report(candidate)

    bad = _rehash({**deepcopy(report), "duration_s": 0.0})
    with pytest.raises(ValueError, match="duration_s must be positive"):
        mod.atomic_write_report(tmp_path / "invalid.json", bad)


def test_req_report_6588_blocked_report_names_failed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6588-REDUCER emits a named block for a real cache miss."""

    original = mod.build_model_cache_identity_rows
    rows = original(REPO)
    rows[0]["resolved"] = False
    rows[0]["admitted"] = False
    rows[0]["rejection_reason"] = "trusted_cache_identity_missing"
    monkeypatch.setattr(mod, "build_model_cache_identity_rows", lambda _root: rows)
    blocked = mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)
    assert blocked[mod.READY_FIELD] == 0.0
    assert blocked["status"] == "blocked_v574_cfr_launch_root"
    assert blocked["honest_verdict"].startswith("blocked_v574_cfr_launch_root:")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check_count"] > 0
    assert blocked["gate_check_summary"]["first_failure"]["check"] == "model_cache_identity_rows"
