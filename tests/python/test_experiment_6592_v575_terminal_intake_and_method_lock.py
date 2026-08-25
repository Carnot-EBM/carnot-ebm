"""Test the V575 terminal intake without model inference.

Spec refs: REQ-REPORT-6592, REQ-REPORT-6592-PRECONDITIONS,
REQ-REPORT-6592-REPLAY, REQ-REPORT-6592-METHODOLOGY,
REQ-REPORT-6592-METHOD-SOURCES, REQ-REPORT-6592-CACHE,
REQ-REPORT-6592-GPU, REQ-REPORT-6592-GATES,
REQ-REPORT-6592-ATTACKS, REQ-REPORT-6592-REDUCER,
REQ-REPORT-6592-ATOMIC, SCENARIO-REPORT-6592-REPLAY,
SCENARIO-REPORT-6592-METHODOLOGY,
SCENARIO-REPORT-6592-METHOD-SOURCES, SCENARIO-REPORT-6592-GATES,
SCENARIO-REPORT-6592-GPU, SCENARIO-REPORT-6592-ATTACKS, and
SCENARIO-REPORT-6592-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6592_v575_terminal_intake_and_method_lock as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TESTS_RUN = [{"command": "focused Exp6592 fixture", "exit_code": 0, "duration_s": 0.01}]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Build the source-only intake once for focused assertions."""

    return mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def _gpu_receipt(*, busy_second: bool) -> dict[str, Any]:
    processes = (
        [{"gpu_uuid": "GPU-1", "pid": 999, "process_name": "other", "used_memory_mb": 100}]
        if busy_second
        else []
    )
    return {
        "visible": True,
        "gpu_query_exit_code": 0,
        "process_query_exit_code": 0,
        "gpu_rows": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 3090",
                "uuid": "GPU-0",
                "memory_total_mb": 24576,
                "memory_free_mb": 24000,
                "utilization_pct": 0,
            },
            {
                "index": 1,
                "name": "NVIDIA GeForce RTX 3090",
                "uuid": "GPU-1",
                "memory_total_mb": 24576,
                "memory_free_mb": 24000 if not busy_second else 12000,
                "utilization_pct": 0 if not busy_second else 50,
            },
        ],
        "compute_process_rows": processes,
        "signals_sent": [],
    }


def test_req_report_6592_spec_declares_every_anchor_and_field() -> None:
    """REQ-REPORT-6592 exists before code and names the full contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6592") :]
    anchors = (
        "REQ-REPORT-6592-PRECONDITIONS",
        "REQ-REPORT-6592-REPLAY",
        "REQ-REPORT-6592-METHODOLOGY",
        "REQ-REPORT-6592-METHOD-SOURCES",
        "REQ-REPORT-6592-CACHE",
        "REQ-REPORT-6592-GPU",
        "REQ-REPORT-6592-GATES",
        "REQ-REPORT-6592-ATTACKS",
        "REQ-REPORT-6592-REDUCER",
        "REQ-REPORT-6592-ATOMIC",
        "SCENARIO-REPORT-6592-REPLAY",
        "SCENARIO-REPORT-6592-METHODOLOGY",
        "SCENARIO-REPORT-6592-METHOD-SOURCES",
        "SCENARIO-REPORT-6592-GATES",
        "SCENARIO-REPORT-6592-GPU",
        "SCENARIO-REPORT-6592-ATTACKS",
        "SCENARIO-REPORT-6592-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    )
    for anchor in anchors:
        assert anchor in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6592_terminal_replay_preserves_block(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6592-REPLAY keeps Exp6589 blocked and replays streams."""

    rows = {row["experiment_id"]: row for row in report["v574_terminal_replay_rows"]}
    assert set(rows) == set(mod.SOURCE_ARTIFACTS)
    assert rows["Exp6588"]["stored_ready_score"] == 1.0
    assert rows["Exp6588"]["recomputed_ready_score"] == 1.0
    assert rows["Exp6589"]["honest_verdict"] == (
        "blocked_receipt_validation_block: terminal_report_validation"
    )
    assert rows["Exp6589"]["verdict_class"] == "blocked"
    assert rows["Exp6589"]["stored_ready_score"] == 0.0
    assert rows["Exp6589"]["recomputed_ready_score"] == 0.0
    assert rows["Exp6589"]["adversarial_disposition"]["flagged_adversarial"] is True
    assert rows["Exp6590"]["recomputed_ready_score"] == 1.0
    assert rows["Exp6591"]["recomputed_ready_score"] == 1.0
    assert all(row["science_result_created"] is False for row in rows.values())
    assert all(row["source_artifact_sha256"].startswith("sha256:") for row in rows.values())


def test_req_report_6592_unwraps_principles_without_erasing_block() -> None:
    """REQ-REPORT-6592-REPLAY reads wrapper values instead of wrapper truthiness."""

    assert mod.unwrap_value({"principle": "why", "value": {"value": "blocked"}}) == "blocked"
    source = mod.load_json(REPO / mod.SOURCE_ARTIFACTS["Exp6589"]["path"])
    source["verdict_class"] = {"principle": "preserve", "value": "blocked"}
    source["pytest_receipt_remediation_ready_score"] = {
        "principle": "zero stays zero",
        "value": 0.0,
    }
    row = mod.replay_one("Exp6589", source, REPO)
    assert row["verdict_class"] == "blocked"
    assert row["stored_ready_score"] == 0.0
    source["pytest_receipt_remediation_ready_score"]["value"] = 1.0
    assert mod.replay_one("Exp6589", source, REPO)["replay_valid"] is False


def test_scenario_report_6592_every_cfr_unit_binds_nested_receipts(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6592-METHODOLOGY binds all 40 units without invention."""

    rows = report["cfr_stream_methodology_binding_rows"]
    assert len(rows) == 40
    assert {row["experiment_id"] for row in rows} == {"Exp6590", "Exp6591"}
    assert all(row["unit_row_hash"].startswith("sha256:") for row in rows)
    assert all(len(row["arm_row_hashes"]) == 3 for row in rows)
    assert all(row["raw_stage_receipt_hashes"] for row in rows)
    assert all(len(row["exact_checker_receipt_hashes"]) == 3 for row in rows)
    assert all(row["checkpoint_receipt_hash"].startswith("sha256:") for row in rows)
    assert all(row["gpu_process_receipts_hash"].startswith("sha256:") for row in rows)
    assert all(row["stream_recomputation_hash"].startswith("sha256:") for row in rows)
    assert all(row["failure_table_bound"] is True for row in rows)
    assert all(row["all_nested_receipts_bound"] is True for row in rows)
    assert all(row["invented_top_level_methodology"] is False for row in rows)
    assert all("model_specs_or_target_model" in row["top_level_methodology_warnings"] for row in rows)
    assert all("random_seed_or_random_seeds_used" in row["top_level_methodology_warnings"] for row in rows)
    assert report["v575_cfr_reducer_ready_score"] == 1.0


def test_req_report_6592_methodology_loss_forces_cfr_score_zero(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6592-METHODOLOGY rejects missing and invented bindings."""

    candidate = deepcopy(report)
    candidate["cfr_stream_methodology_binding_rows"].pop()
    assert mod.readiness_reducer(candidate)["v575_cfr_reducer_ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["cfr_stream_methodology_binding_rows"][0]["unit_row_hash"] = "missing"
    assert mod.readiness_reducer(candidate)["v575_cfr_reducer_ready_score"] == 0.0
    candidate = deepcopy(report)
    candidate["cfr_stream_methodology_binding_rows"][0][
        "invented_top_level_methodology"
    ] = True
    assert mod.readiness_reducer(candidate)["v575_cfr_reducer_ready_score"] == 0.0


def test_scenario_report_6592_method_sources_lock_boundaries(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6592-METHOD-SOURCES imports methods, not paper results."""

    rows = report["method_source_lock_rows"]
    assert [row["source_id"] for row in rows] == list(mod.METHOD_SOURCE_IDS)
    assert [row["retrieved_title"] for row in rows] == [
        "Correcting a learned physical invariant improves world-model rollouts",
        "ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings",
        "Spectral partitioning for k-block averaging kernels of finite Markov chains",
    ]
    assert all(row["bounded_import"] for row in rows)
    assert all(row["controls"] for row in rows)
    assert all(row["metrics"] for row in rows)
    assert all(row["non_claims"] for row in rows)
    assert all(row["paper_result_counts_as_carnot_evidence"] is False for row in rows)
    assert mod.method_source_locks_ready(rows)


def test_req_report_6592_cache_identities_are_content_derived(report: dict[str, Any]) -> None:
    """REQ-REPORT-6592-CACHE resolves both GGUF files without a model load."""

    rows = report["model_cache_identity_rows"]
    assert [row["repository_id"] for row in rows] == list(mod.MANDATED_MODEL_IDS)
    for row in rows:
        assert row["resolved"] is True
        assert row["content_metadata"]["magic"] == "GGUF"
        assert row["content_metadata"]["bounded_read_receipt"]["tensor_payload_bytes_read"] == 0
        assert row["model_load_performed"] is False
        assert row["download_performed"] is False
        assert row["auto_tokenizer_used"] is False


def test_scenario_report_6592_busy_gpu_stays_unowned_and_nonblocking(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6592-GPU preserves busy work and keeps CFR ready."""

    busy = mod.build_gpu_ownership_rows(_gpu_receipt(busy_second=True))
    assert mod.dual_gpu_rows_ready(busy) is False
    assert busy[1]["available_for_runtime_ownership"] is False
    assert busy[1]["unowned_processes_preserved"][0]["pid"] == 999
    assert all(row["signals_sent"] == [] for row in busy)

    idle = mod.build_gpu_ownership_rows(_gpu_receipt(busy_second=False))
    assert mod.dual_gpu_rows_ready(idle) is True
    candidate = deepcopy(report)
    candidate["gpu_ownership_rows"] = busy
    reduction = mod.readiness_reducer(candidate)
    assert reduction["v575_dual_gpu_canary_ready_score"] == 0.0
    assert reduction["v575_cfr_reducer_ready_score"] == 1.0


def test_scenario_report_6592_gate_map_names_inactive_future_task(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6592-GATES closes Exp6593 and does not invent Exp6602."""

    rows = {row["consumer_task_id"]: row for row in report["current_roadmap_gate_contract_rows"]}
    assert rows["exp6593-cfr-independent-row-reducer"]["all_cross_references_close"] is True
    assert rows["exp6593-cfr-independent-row-reducer"]["artifact_field"] == (
        "v575_cfr_reducer_ready_score"
    )
    future = rows["exp6602-dual-gpu-flagship-residency-canary"]
    assert future["design_document_consumer_exists"] is True
    assert future["consumer_task_exists"] is False
    assert future["owner_output_field_declared"] is False
    assert future["all_cross_references_close"] is False
    assert future["disposition"] == "warning_inactive_future_task_not_in_active_yaml"
    assert report["v575_cfr_reducer_ready_score"] == 1.0


def test_scenario_report_6592_all_attacks_fail_closed(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6592-ATTACKS closes each declared mutation."""

    rows = report["attack_rows"]
    assert [row["attack_id"] for row in rows] == list(mod.REQUIRED_ATTACK_IDS)
    assert all(row["candidate_acceptance_score"] == 0.0 for row in rows)
    assert all(row["passed"] is True for row in rows)
    assert mod.attack_rows_ready(rows)


def test_req_report_6592_preconditions_record_sources_resources_and_no_llm(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6592-PRECONDITIONS records all bounded intake inputs."""

    checks = report["preconditions_checked"]
    assert checks["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert checks["llm_calls_issued"] == 0
    assert checks["model_loads_issued"] == 0
    assert checks["downloads_issued"] == 0
    assert checks["gpu_eviction_signals_sent"] == []
    assert len(checks["v574_artifacts"]) == 4
    assert all(row["exists"] and row["sha256"].startswith("sha256:") for row in checks["v574_artifacts"])
    assert checks["roadmap"]["sha256"].startswith("sha256:")
    assert checks["v575_reference_refresh"]["sha256"].startswith("sha256:")
    assert checks["cpu"]["count"] >= 1
    assert checks["ram"]["total_kib"] > 0
    assert checks["disk"]["total_bytes"] > 0
    assert isinstance(checks["dirty_worktree"]["entries"], list)
    assert checks["dirty_worktree"]["status_sha256"].startswith("sha256:")


def test_scenario_report_6592_atomic_null_artifact_validates(
    tmp_path: Path, report: dict[str, Any]
) -> None:
    """SCENARIO-REPORT-6592-ATOMIC writes one durable null intake artifact."""

    output = tmp_path / "experiment_6592.json"
    receipt = mod.atomic_write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert receipt["file_fsync"] is True
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["output_sha256"] == mod.sha256_file(output)
    assert report["status"] == "complete_v575_terminal_intake_and_method_lock"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] is None
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["reproducibility_checksum"] == mod.artifact_checksum(report)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_report(report) == []


def test_req_report_6592_validator_rejects_terminal_tamper(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """REQ-REPORT-6592-ATOMIC rejects schema, authority, and checksum drift."""

    cases = []
    missing = deepcopy(report)
    missing.pop("status")
    cases.append((missing, "missing required fields: status"))
    cases.append((_rehash({**deepcopy(report), "inference_substrate": "wrong"}), "inference_substrate mismatch"))
    cases.append((_rehash({**deepcopy(report), "verifier_is_oracle": False}), "verifier_is_oracle must be true"))
    cases.append((_rehash({**deepcopy(report), "duration_s": 0.0}), "duration_s must be positive"))
    cases.append((_rehash({**deepcopy(report), "verdict_class": "positive"}), "complete intake verdict_class must be null"))
    cases.append((_rehash({**deepcopy(report), "honest_verdict": "done"}), "terminal success prefix missing"))
    candidate = deepcopy(report)
    candidate["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((_rehash(candidate), "protected_files_unchanged failed"))
    candidate = deepcopy(report)
    candidate["field_provenance"].pop("status")
    cases.append((_rehash(candidate), "field_provenance missing required fields"))
    cases.append(({**deepcopy(report), "reproducibility_checksum": "sha256:stale"}, "reproducibility_checksum mismatch"))

    for candidate, expected in cases:
        assert expected in mod.validate_report(candidate)

    with pytest.raises(ValueError, match="duration_s must be positive"):
        mod.atomic_write_report(tmp_path / "invalid.json", _rehash({**deepcopy(report), "duration_s": 0.0}))


def test_req_report_6592_blocked_report_names_exact_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6592-REDUCER names a protected-history failure."""

    monkeypatch.setattr(
        mod,
        "_protected_receipt",
        lambda _before, _after, _sources_before=None, _sources_after=None: {
            "rows": [],
            "historical_artifact_rows": [],
            "changed_paths": ["research-roadmap.yaml"],
            "all_unchanged": False,
        },
    )
    blocked = mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)
    assert blocked["status"] == "blocked_v575_terminal_intake_and_method_lock"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_blocking_check_count"] > 0
    assert blocked["gate_check_summary"]["first_blocking_failure"]["check"] == (
        "protected_files_unchanged"
    )
    assert mod.validate_report(blocked) == []
