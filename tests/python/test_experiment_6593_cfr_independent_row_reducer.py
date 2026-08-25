"""Test independent CFR row reduction without model inference.

Spec refs: REQ-REPORT-6593, REQ-REPORT-6593-PRECONDITIONS,
REQ-REPORT-6593-ROWS, REQ-REPORT-6593-COMPLETENESS,
REQ-REPORT-6593-FAMILY, REQ-REPORT-6593-PAIRED,
REQ-REPORT-6593-CONSTRAINTS, REQ-REPORT-6593-SAFETY-COST,
REQ-REPORT-6593-GATE, REQ-REPORT-6593-ATTACKS,
REQ-REPORT-6593-REDUCER, REQ-REPORT-6593-ATOMIC,
SCENARIO-REPORT-6593-REPLAY, SCENARIO-REPORT-6593-ISOLATION,
SCENARIO-REPORT-6593-PAIRED, SCENARIO-REPORT-6593-AUTHORITY,
SCENARIO-REPORT-6593-FAILURES, SCENARIO-REPORT-6593-ATTACKS, and
SCENARIO-REPORT-6593-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import scripts.adversarial_verify as adversarial

from carnot import experiment_6593_cfr_independent_row_reducer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TESTS_RUN = [{"command": "focused Exp6593 fixture", "exit_code": 0, "duration_s": 0.01}]


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """REQ-REPORT-6593 builds one immutable source-only report for tests."""

    return mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_report_6593_spec_declares_anchors_and_required_fields() -> None:
    """REQ-REPORT-6593 exists before code and names the full artifact."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6593") :]
    anchors = (
        "REQ-REPORT-6593-PRECONDITIONS",
        "REQ-REPORT-6593-ROWS",
        "REQ-REPORT-6593-COMPLETENESS",
        "REQ-REPORT-6593-FAMILY",
        "REQ-REPORT-6593-PAIRED",
        "REQ-REPORT-6593-CONSTRAINTS",
        "REQ-REPORT-6593-SAFETY-COST",
        "REQ-REPORT-6593-GATE",
        "REQ-REPORT-6593-ATTACKS",
        "REQ-REPORT-6593-REDUCER",
        "REQ-REPORT-6593-ATOMIC",
        "SCENARIO-REPORT-6593-REPLAY",
        "SCENARIO-REPORT-6593-ISOLATION",
        "SCENARIO-REPORT-6593-PAIRED",
        "SCENARIO-REPORT-6593-AUTHORITY",
        "SCENARIO-REPORT-6593-FAILURES",
        "SCENARIO-REPORT-6593-ATTACKS",
        "SCENARIO-REPORT-6593-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
    )
    for anchor in anchors:
        assert anchor in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_report_6593_preconditions_bind_gate_hashes_and_cpu(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6593-PRECONDITIONS binds the frozen no-LLM inputs."""

    pre = report["preconditions_checked"]
    assert pre["planning_date"] == "20260825"
    assert pre["structured_gate"]["field"] == "v575_cfr_reducer_ready_score"
    assert pre["structured_gate"]["stored_value"] == 1.0
    assert pre["structured_gate"]["recomputed_value"] == 1.0
    assert pre["expected_counts"] == {
        "family_count": 2,
        "units_per_family": 20,
        "arms_per_unit": 3,
        "family_unit_count": 40,
        "family_unit_arm_count": 120,
    }
    assert pre["method_hashes"]["exact_registry_hash"].startswith("sha256:")
    assert pre["method_hashes"]["metric_contract_hash"].startswith("sha256:")
    assert len(pre["seeds"]) == 20
    assert pre["paired_test_plan"]["bootstrap_resamples"] == 10_000
    assert pre["cpu_only_substrate"] is True
    assert pre["llm_calls_issued"] == 0
    assert pre["model_loads_issued"] == 0
    assert pre["cpu"]["logical_count"] >= 1
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True


def test_scenario_report_6593_replays_every_family_unit_and_arm(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6593-REPLAY emits all 120 independently replayed rows."""

    rows = report["per_unit_rows"]
    assert len(rows) == 120
    assert [row["family"] for row in rows[:60]] == [mod.FAMILY_ORDER[0]] * 60
    assert [row["family"] for row in rows[60:]] == [mod.FAMILY_ORDER[1]] * 60
    assert {row["arm_name"] for row in rows} == set(mod.ARM_ORDER)
    assert all(row["exact_replay_matches_source"] is True for row in rows)
    assert all(row["arm_replay_matches_source"] is True for row in rows)
    assert all(row["unit_replay_matches_source"] is True for row in rows)
    assert all(row["raw_outcome_references"] for row in rows)
    source_hashes = {
        row["family"]: row["sha256"]
        for row in report["source_artifact_receipts"]
        if row["family"] is not None
    }
    assert all(row["raw_source_artifact_sha256"] == source_hashes[row["family"]] for row in rows)
    assert all(row["row_reproducibility_hash"].startswith("sha256:") for row in rows)
    assert sum(row["exact_success"] for row in rows) == 120
    assert sum(row["headroom"] for row in rows) == 0
    assert sum(row["unsafe_release"] for row in rows) == 0


def test_req_report_6593_recomputes_exact_constraints_cost_and_failures(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6593-ROWS derives outcomes instead of copying aggregates."""

    qwen = [row for row in report["per_unit_rows"] if row["family"] == "qwen36"]
    gemma = [row for row in report["per_unit_rows"] if row["family"] == "gemma4_31b"]
    assert sum(row["unsupported_constraint_count"] for row in qwen) == 0
    assert sum(row["unsupported_constraint_count"] for row in gemma) == 68
    assert sum(row["contradictory_constraint_count"] for row in report["per_unit_rows"]) == 0
    assert sum(row["tokens"] for row in qwen) == 21_131
    assert sum(row["tokens"] for row in gemma) == 18_301
    assert sum(row["failure_any"] for row in qwen) == 42
    assert sum(row["failure_any"] for row in gemma) == 51
    exact_rejections = [row for row in report["per_unit_rows"] if row["failure"]["exact_rejection"]]
    assert exact_rejections
    assert all(row["charged_failure"] is True for row in exact_rejections)
    assert all(
        row["charged_cost"] == pytest.approx(row["tokens"] + row["latency_s"])
        for row in exact_rejections
    )


def test_req_report_6593_completeness_rejects_duplicate_reorder_and_drift(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6593-COMPLETENESS fails closed on key or identity drift."""

    cases: list[tuple[dict[str, Any], str]] = []
    duplicate = deepcopy(report)
    duplicate["per_unit_rows"][-1] = deepcopy(duplicate["per_unit_rows"][0])
    cases.append((duplicate, "unique_row_keys"))
    reordered = deepcopy(report)
    reordered["per_unit_rows"][0], reordered["per_unit_rows"][1] = (
        reordered["per_unit_rows"][1],
        reordered["per_unit_rows"][0],
    )
    cases.append((reordered, "frozen_row_order"))
    family_swap = deepcopy(report)
    family_swap["per_unit_rows"][0]["family"] = "gemma4_31b"
    cases.append((family_swap, "family_identity"))
    seed_drift = deepcopy(report)
    seed_drift["per_unit_rows"][0]["seed"] += 1
    cases.append((seed_drift, "seed_schedule"))
    source_drift = deepcopy(report)
    source_drift["per_unit_rows"][0]["source_bytes_sha256"] = "sha256:drift"
    cases.append((source_drift, "source_binding"))
    authority = deepcopy(report)
    authority["per_unit_rows"][0]["exact_registry_sha256"] = "sha256:substitute"
    cases.append((authority, "exact_authority"))
    for candidate, failed_check in cases:
        checks = mod.reducer_checks(candidate)
        assert checks[failed_check] is False
        assert mod.readiness_reducer(candidate)["cfr_reducer_ready_score"] == 0.0


def test_req_report_6593_completeness_counts_replay_from_raw_streams(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6593-COMPLETENESS reports raw counts and zero defects."""

    rows = report["row_completeness_recomputation"]
    assert len(rows) == 2
    for row in rows:
        assert row["expected_unit_count"] == row["present_unit_count"] == 20
        assert row["expected_arm_count"] == row["present_arm_count"] == 60
        assert row["expected_raw_stage_count"] == row["present_raw_stage_count"] == 90
        assert row["present_exact_result_count"] == 60
        assert row["duplicate_unit_count"] == 0
        assert row["duplicate_arm_count"] == 0
        assert row["missing_unit_count"] == 0
        assert row["missing_arm_count"] == 0
        assert row["reordered_unit_count"] == 0
        assert row["reordered_arm_count"] == 0
        assert row["cross_family_row_count"] == 0
        assert row["all_rows_replayed"] is True


def test_req_report_6593_paired_helpers_cover_discordance_and_ties() -> None:
    """REQ-REPORT-6593-PAIRED computes exact tests and deterministic intervals."""

    mcnemar = mod.exact_mcnemar([False, False, True, True], [True, True, False, True])
    assert mcnemar["wins"] == 2
    assert mcnemar["losses"] == 1
    assert mcnemar["ties"] == 1
    assert mcnemar["p_value"] == 1.0
    assert mcnemar["valid"] is True
    tied = mod.exact_mcnemar([True, True], [True, True])
    assert tied["valid"] is False
    assert tied["reason"] == "no_discordant_units"
    sign = mod.exact_sign_test([1.0, 2.0, -1.0, 0.0])
    assert sign["positive"] == 2
    assert sign["negative"] == 1
    assert sign["zero"] == 1
    assert sign["valid"] is True
    assert mod.exact_sign_test([0.0, 0.0])["valid"] is False
    first = mod.paired_bootstrap_ci([1.0, -1.0, 0.0], resamples=100, seed=7)
    second = mod.paired_bootstrap_ci([1.0, -1.0, 0.0], resamples=100, seed=7)
    assert first == second
    assert first["resamples"] == 100
    assert mod.paired_bootstrap_ci([], resamples=100, seed=7)["unit_count"] == 0
    with pytest.raises(ValueError, match="equal length"):
        mod.exact_mcnemar([True], [])
    assert mod.unwrap_value({"value": {"value": 1.0}, "principle": "nested"}) == 1.0


def test_req_report_6593_replay_retains_unknown_rows_as_completeness_defects() -> None:
    """REQ-REPORT-6593-COMPLETENESS counts unknown raw units without emitting them."""

    method, streams, _ = mod._load_sources(REPO)
    stream = deepcopy(streams[mod.FAMILY_ORDER[0]])
    stream["per_unit_rows"].append({"unit_id": "unknown-unit", "arms": []})
    rows, completeness = mod.replay_stream_rows(mod.FAMILY_ORDER[0], stream, method)
    assert len(rows) == 60
    assert completeness["extra_unit_count"] == 1
    assert completeness["all_rows_replayed"] is False


def test_scenario_report_6593_no_headroom_stays_null(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6593-PAIRED makes every zero-headroom case explicit."""

    effects = report["family_effect_rows"]
    assert len(effects) == 4
    for row in effects:
        assert row["direct_headroom_count"] == 0
        assert row["wins"] == row["losses"] == 0
        assert row["ties"] == 20
        assert row["exact_success_delta"] == 0.0
        assert row["exact_success_ci95"] == {"lower": 0.0, "upper": 0.0}
        assert row["no_headroom"] is True
        assert row["underpowered"] is True
        assert row["exact_test"]["valid"] is False
    pooled = report["pooled_effect_summary"]
    assert pooled["shared_unit_count"] == 20
    assert pooled["family_unit_pair_count"] == 40
    assert pooled["all_shared_units_byte_identical"] is True
    assert all(row["exact_success_delta"] == 0.0 for row in pooled["effect_rows"])
    assert report["verdict_class"] is None
    assert report["cfr_reducer_ready_score"] == 1.0


def test_scenario_report_6593_family_isolation_precedes_pooling(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6593-ISOLATION fixes two families before pooling."""

    identities = report["model_identity_replay_rows"]
    assert [row["family"] for row in identities] == list(mod.FAMILY_ORDER)
    assert identities[0]["repository_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert identities[1]["repository_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert all(row["model_identity_valid"] is True for row in identities)
    assert all(row["process_identity_valid"] is True for row in identities)
    assert all(row["cross_family_residency_detected"] is False for row in identities)
    pooled = report["pooled_effect_summary"]
    assert pooled["family_results_fixed_before_pooling"] is True
    assert pooled["pooling_cluster"] == "unit_id"
    assert pooled["family_heterogeneity_retained"] is True
    assert len(pooled["shared_unit_receipts"]) == 20


def test_req_report_6593_constraint_quality_recomputes_stage1(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6593-CONSTRAINTS rejects unsupported Stage 1 proposals."""

    rows = report["constraint_quality_summary"]
    by_key = {(row["scope"], row["candidate_arm"]): row for row in rows}
    assert by_key[("qwen36", "always_on_cfr")]["stage1_proposal_count"] == 0
    assert by_key[("qwen36", "always_on_cfr")]["stage1_precision"] == 0.0
    assert by_key[("qwen36", "always_on_cfr")]["stage1_recall"] == 0.0
    assert by_key[("gemma4_31b", "always_on_cfr")]["stage1_proposal_count"] == 48
    assert by_key[("gemma4_31b", "always_on_cfr")]["unsupported_constraint_count"] == 48
    assert by_key[("gemma4_31b", "always_on_cfr")]["stage1_precision"] == 0.0
    assert by_key[("gemma4_31b", "routed_cfr")]["stage1_proposal_count"] == 20
    assert by_key[("gemma4_31b", "routed_cfr")]["unsupported_rate"] == 1.0
    assert all(row["contradictory_constraint_count"] == 0 for row in rows)
    assert all(row["stage1_answer_leakage_count"] == 0 for row in rows)


def test_scenario_report_6593_failures_and_cost_remain_charged(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6593-FAILURES retains every failed outcome and cost."""

    summary = report["safety_and_cost_summary"]
    assert len(summary) == 9
    qwen_routed = next(
        row for row in summary if row["scope"] == "qwen36" and row["arm_name"] == "routed_cfr"
    )
    gemma_always = next(
        row
        for row in summary
        if row["scope"] == "gemma4_31b" and row["arm_name"] == "always_on_cfr"
    )
    assert qwen_routed["tokens_total"] == 6_015
    assert qwen_routed["failure_arm_count"] == 14
    assert qwen_routed["charged_failure_arm_count"] == 14
    assert gemma_always["tokens_total"] == 9_441
    assert gemma_always["failure_arm_count"] == 20
    assert gemma_always["unsupported_constraint_count"] == 48
    assert all(row["unsafe_release_count"] == 0 for row in summary)


def test_req_report_6593_gate_records_each_frozen_condition(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6593-GATE records failed benefit gates without blocking replay."""

    rows = report["acceptance_gate_rows"]
    assert len(rows) == 8 * 6
    assert {row["condition"] for row in rows} == set(mod.GATE_CONDITION_ORDER)
    assert all("expected" in row and "observed" in row and "passed" in row for row in rows)
    assert all(row["candidate_gate_passed"] is False for row in rows)
    assert all(row["verdict_if_all_pass"] == "circular_positive" for row in rows)
    exact_delta_rows = [row for row in rows if row["condition"] == "positive_exact_success_delta"]
    assert all(row["observed"] == 0.0 and row["passed"] is False for row in exact_delta_rows)
    assert report["gate_check_summary"]["failed_blocking_check_count"] == 0
    assert report["gate_check_summary"]["no_headroom_case_count"] == 6


def test_scenario_report_6593_attacks_all_fail_closed(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6593-ATTACKS rejects every preregistered mutation."""

    rows = report["attack_rows"]
    assert [row["attack_id"] for row in rows] == list(mod.REQUIRED_ATTACK_IDS)
    assert all(row["passed"] is True for row in rows)
    assert all(row["candidate_ready_score"] == 0.0 for row in rows)
    assert all(row["expected_detector"] in row["failed_checks"] for row in rows)
    missing = deepcopy(report)
    missing["attack_rows"].pop()
    assert mod.readiness_reducer(missing)["cfr_reducer_ready_score"] == 0.0
    with pytest.raises(ValueError, match="unknown attack detector"):
        mod._attack_detector_passed(report, "not-a-detector")


def test_scenario_report_6593_no_llm_receipts_do_not_claim_live_inference(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-6593-ATTACKS treats preserved GGUF receipts as source evidence."""

    path = tmp_path / "experiment_6593.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    flags = {row["kind"] for row in adversarial.verify_artifact(path)["flags"]}
    assert "DURATION_TOO_SHORT" not in flags
    assert "METHODOLOGY_MISSING" not in flags


def test_req_report_6593_verdict_classification_is_closed() -> None:
    """REQ-REPORT-6593-GATE never labels an exact-authority win positive."""

    assert mod._classify_verdict(False, False)[1] == "blocked"
    assert mod._classify_verdict(True, True)[1] == "circular_positive"
    assert mod._classify_verdict(True, False)[1] is None
    assert mod._blocking_tests_passed(
        [
            {"exit_code": 0, "blocking": True},
            {"exit_code": 2, "blocking": False},
        ]
    )
    assert not mod._blocking_tests_passed([{"exit_code": 2, "blocking": True}])
    assert not mod._blocking_tests_passed([])
    receipts = mod._tests_run_receipts(None)
    assert any(row["exit_code"] == 2 and row["blocking"] is False for row in receipts)
    summary = mod._gate_summary(
        {"tests_run": receipts, "acceptance_gate_rows": []},
        {"checks": {}, "cfr_reducer_ready_score": 1.0},
    )
    assert summary["test_diagnostic_failure_count"] == 1
    assert summary["test_diagnostic_failures"][0]["blocking"] is False


def test_req_report_6593_validate_names_blocks_and_disqualifications(
    report: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6593-REDUCER validates closed verdict and checksum states."""

    assert mod.validate_report(report, REPO) == []
    monkeypatch.setattr(
        mod,
        "readiness_reducer",
        lambda payload: {"checks": {"test": True}, "cfr_reducer_ready_score": 1.0},
    )
    missing = deepcopy(report)
    missing.pop("per_unit_rows")
    assert mod.validate_report(missing, REPO)[0].startswith("missing_required_fields:")
    invalid_class = deepcopy(report)
    invalid_class["verdict_class"] = "positive"
    assert "verdict_class_invalid" in mod.validate_report(_rehash(invalid_class), REPO)
    blocked = deepcopy(report)
    blocked["status"] = "blocked_cfr_reducer"
    blocked["honest_verdict"] = "blocked_cfr_reducer: missing source row"
    blocked["verdict_class"] = "blocked"
    blocked["gate_check_summary"]["first_blocking_failure"] = None
    assert "blocked_verdict_missing_gate_detail" in mod.validate_report(_rehash(blocked), REPO)
    checksum = deepcopy(report)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_report(checksum, REPO)
    malformed = deepcopy(report)
    malformed["inference_substrate"] = "model_loaded"
    malformed["verifier_is_oracle"] = False
    malformed["field_provenance"] = {}
    errors = mod.validate_report(_rehash(malformed), REPO)
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_mismatch" in errors
    assert "field_provenance_mismatch" in errors
    source_drift = deepcopy(report)
    source_drift["source_artifact_receipts"][0]["sha256"] = "sha256:drift"
    assert any(
        error.startswith("source_artifact_hash_mismatch:")
        for error in mod.validate_report(_rehash(source_drift), REPO)
    )
    monkeypatch.setattr(
        mod,
        "readiness_reducer",
        lambda payload: {"checks": {"test": False}, "cfr_reducer_ready_score": 0.0},
    )
    incomplete = deepcopy(report)
    errors = mod.validate_report(_rehash(incomplete), REPO)
    assert "cfr_reducer_ready_score_mismatch" in errors
    assert "null_verdict_without_complete_replay" in errors
    monkeypatch.setattr(
        mod,
        "readiness_reducer",
        lambda payload: {"checks": {"test": True}, "cfr_reducer_ready_score": 1.0},
    )
    circular = deepcopy(report)
    circular["verdict_class"] = "circular_positive"
    assert "circular_positive_without_gate_win" in mod.validate_report(_rehash(circular), REPO)


def test_scenario_report_6593_atomic_round_trip_and_protected_hashes(
    report: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6593-ATOMIC writes one valid durable result."""

    monkeypatch.setattr(
        mod,
        "readiness_reducer",
        lambda payload: {
            "checks": {"test": True},
            "cfr_reducer_ready_score": payload.get("cfr_reducer_ready_score", 1.0),
        },
    )
    target = tmp_path / "nested" / "exp6593.json"
    receipt = mod.atomic_write_report(target, report, repo_root=REPO)
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["sha256"] == mod.sha256_file(target)
    assert json.loads(target.read_text(encoding="utf-8")) == report
    protected = report["protected_files_unchanged"]
    assert protected["all_unchanged"] is True
    assert [row["path"] for row in protected["rows"]] == [
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    ]
    bad = deepcopy(report)
    bad["duration_s"] = 0.0
    bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
    with pytest.raises(ValueError, match="duration_s_invalid"):
        mod.atomic_write_report(tmp_path / "bad.json", bad, repo_root=REPO)
