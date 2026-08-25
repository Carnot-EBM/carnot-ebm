"""Test the CFR authority audit without model inference.

Spec refs: REQ-REPORT-6594 and SCENARIO-REPORT-6594-CLEAN through
SCENARIO-REPORT-6594-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from carnot import experiment_6594_cfr_counterfactual_authority_audit as mod
from scripts import adversarial_verify as adversarial


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """Build the complete audit once because every test reads immutable rows."""

    return mod.build_report(
        REPO,
        date="20260825",
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
    )


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    value["reproducibility_checksum"] = mod.artifact_checksum(value)
    return value


def _copy_upstreams(root: Path) -> None:
    (root / "results").mkdir(parents=True)
    for relative in mod.UPSTREAM_RELATIVE_PATHS:
        source = REPO / relative
        if source.is_file():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def test_req_report_6594_spec_has_all_anchors_and_fields() -> None:
    """REQ-REPORT-6594 exists before implementation and names its scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6594") :]
    anchors = (
        "REQ-REPORT-6594-PRECONDITIONS",
        "REQ-REPORT-6594-CLEAN-REPLAY",
        "REQ-REPORT-6594-SOURCE-CONSTRAINT",
        "REQ-REPORT-6594-STAGE-FAMILY",
        "REQ-REPORT-6594-TAMPER-LEAKAGE",
        "REQ-REPORT-6594-AUTHORITY",
        "REQ-REPORT-6594-MISSING",
        "REQ-REPORT-6594-ROWS",
        "REQ-REPORT-6594-ATTACKS",
        "REQ-REPORT-6594-REDUCER",
        "REQ-REPORT-6594-ATOMIC",
        "SCENARIO-REPORT-6594-CLEAN",
        "SCENARIO-REPORT-6594-SOURCE-CONSTRAINT",
        "SCENARIO-REPORT-6594-STAGE-FAMILY",
        "SCENARIO-REPORT-6594-TAMPER-LEAKAGE",
        "SCENARIO-REPORT-6594-AUTHORITY",
        "SCENARIO-REPORT-6594-MISSING",
        "SCENARIO-REPORT-6594-ATTACKS",
        "SCENARIO-REPORT-6594-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
    )
    for anchor in anchors:
        assert anchor in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_report_6594_preconditions_bind_inputs_registry_and_no_llm(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6594-PRECONDITIONS records all immutable audit inputs."""

    pre = report["preconditions_checked"]
    assert pre["planning_date"] == "20260825"
    assert pre["expected_counts"] == {
        "families": 2,
        "units_per_family": 20,
        "clean_rows": 40,
        "primary_attacks": 10,
        "primary_attack_rows": 400,
        "per_unit_rows": 440,
    }
    assert len(pre["expected_row_keys"]) == 440
    assert len(set(pre["expected_row_keys"])) == 440
    assert pre["exact_registry_sha256"].startswith("sha256:")
    assert pre["attack_seeds"] == list(mod.ATTACK_SEEDS)
    assert pre["immutable_fixture_policy"]["historical_sources_read_only"] is True
    assert pre["immutable_fixture_policy"]["attack_materialization"] == "in_memory_deep_copies"
    assert pre["immutable_fixture_policy"]["safe_temporary_root"] == "/tmp"
    assert pre["llm_calls_issued"] == 0
    assert pre["model_loads_issued"] == 0
    assert pre["downloads_issued"] == 0
    assert pre["gpu_calls_issued"] == 0
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True


def test_scenario_report_6594_clean_replays_rows_and_null_headline(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6594-CLEAN replays controls before any mutation."""

    controls = report["clean_replay_receipts"]
    assert len(controls) == 40
    assert all(row["attack_id"] == "clean_control" for row in controls)
    assert all(row["attack_passed"] is True for row in controls)
    assert all(row["clean_replay_matches_exp6593"] is True for row in controls)
    assert all(row["audit_release_decision"] in {"release", "abstain"} for row in controls)
    recomputation = report["reducer_recomputation"]
    assert recomputation["clean_family_unit_arm_row_count"] == 120
    assert recomputation["family_effect_rows_match_exp6593"] is True
    assert recomputation["pooled_effect_summary_matches_exp6593"] is True
    assert recomputation["all_exact_success_deltas"] == [0.0] * 6
    assert recomputation["scientific_effect"] == "null_no_direct_headroom"
    assert recomputation["headline_recomputed_before_attacks"] is True
    assert report["verdict_class"] is None


def test_req_report_6594_emits_one_row_per_unit_and_attack(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6594-ROWS preserves every control and primary attack row."""

    rows = report["per_unit_rows"]
    assert len(rows) == 440
    assert [row["row_key"] for row in rows] == report["preconditions_checked"]["expected_row_keys"]
    assert len({row["row_key"] for row in rows}) == 440
    assert {row["family"] for row in rows} == set(mod.FAMILY_ORDER)
    assert {row["arm"] for row in rows} == {"always_on_cfr"}
    for row in rows:
        assert row["input_hash"].startswith("sha256:")
        assert row["treatment_input_hash"].startswith("sha256:")
        assert row["expected_effect"] == mod.EXPECTED_EFFECTS[row["attack_id"]]
        assert row["exact_checker"] == mod.EXACT_CHECKER_NAME
        assert "observed_effect" in row
        assert "source_release_decision" in row
        assert "audit_release_decision" in row


def test_scenario_report_6594_source_and_constraints_are_causal(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6594-SOURCE-CONSTRAINT rejects changed causal links."""

    source_rows = report["source_counterfactual_rows"]
    constraint_rows = report["constraint_counterfactual_rows"]
    assert len(source_rows) == len(constraint_rows) == 80
    assert {row["attack_id"] for row in source_rows} == {
        "source_replacement",
        "source_span_deletion",
    }
    assert {row["attack_id"] for row in constraint_rows} == {
        "supported_constraint_deletion",
        "contradiction_injection",
    }
    assert all(row["attack_passed"] is True for row in source_rows)
    assert all(row["audit_release_decision"].startswith("blocked_") for row in source_rows)
    deletions = [
        row for row in constraint_rows if row["attack_id"] == "supported_constraint_deletion"
    ]
    assert sum(row["applicable"] for row in deletions) == 32
    assert all(row["attack_passed"] is True for row in deletions)
    contradictions = [
        row for row in constraint_rows if row["attack_id"] == "contradiction_injection"
    ]
    assert all(row["observed_effect"]["contradiction_detected"] is True for row in contradictions)
    assert all(row["audit_release_decision"] == "blocked_contradiction" for row in contradictions)


def test_scenario_report_6594_stage_and_family_swaps_do_not_hide_identity(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6594-STAGE-FAMILY preserves score under label swaps."""

    rows = report["stage_and_family_attack_rows"]
    assert len(rows) == 80
    stage_rows = [row for row in rows if row["attack_id"] == "stage1_stage2_swap"]
    family_rows = [row for row in rows if row["attack_id"] == "family_label_swap"]
    assert all(row["observed_effect"]["stage_identity_valid"] is False for row in stage_rows)
    assert all(row["audit_release_decision"] == "blocked_stage_identity" for row in stage_rows)
    assert all(row["observed_effect"]["family_binding_valid"] is False for row in family_rows)
    assert all(row["observed_effect"]["exact_score_changed"] is False for row in family_rows)
    assert all(row["observed_effect"]["source_release_changed"] is False for row in family_rows)
    assert all(row["audit_release_decision"] == "blocked_family_identity" for row in family_rows)


def test_scenario_report_6594_tamper_and_answer_leakage_fail_closed(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6594-TAMPER-LEAKAGE seals bytes and Stage 1 answers."""

    rows = report["tamper_and_leakage_rows"]
    assert len(rows) == 80
    tamper = [row for row in rows if row["attack_id"] == "raw_byte_tamper"]
    leakage = [row for row in rows if row["attack_id"] == "answer_leakage"]
    assert all(row["observed_effect"]["sealed_raw_hash_matches"] is False for row in tamper)
    assert all(row["audit_release_decision"] == "blocked_raw_byte_tamper" for row in tamper)
    assert all(row["observed_effect"]["stage1_answer_leakage"] is True for row in leakage)
    assert all(row["audit_release_decision"] == "blocked_answer_leakage" for row in leakage)


def test_scenario_report_6594_exact_authority_cannot_be_removed_or_substituted(
    report: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6594-AUTHORITY makes the frozen registry authoritative."""

    rows = report["authority_substitution_rows"]
    assert len(rows) == 80
    assert {row["attack_id"] for row in rows} == {
        "exact_checker_removal",
        "exact_checker_substitution",
    }
    assert all(row["attack_passed"] is True for row in rows)
    assert all(row["observed_effect"]["exact_authority_valid"] is False for row in rows)
    assert all(row["audit_release_decision"].startswith("blocked_exact_") for row in rows)
    assert all(row["authority_after"] != "model" for row in rows)


def test_req_report_6594_no_llm_substrate_has_a_duration_floor(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """REQ-REPORT-6594-ATOMIC makes the mandated substrate verifiable."""

    floor = adversarial.duration_floor_for_artifact(report)
    assert floor is not None
    assert floor["reason"] == "no_llm_declared"
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    kinds = {row["kind"] for row in adversarial.verify_artifact(path)["flags"]}
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in kinds


def test_scenario_report_6594_meta_attacks_fail_closed(report: dict[str, Any]) -> None:
    """SCENARIO-REPORT-6594-ATTACKS detects every audit-shortcut mutation."""

    rows = report["attack_rows"]
    assert [row["attack_id"] for row in rows] == list(mod.META_ATTACK_IDS)
    assert all(row["passed"] is True for row in rows)
    assert all(row["candidate_ready_score"] == 0.0 for row in rows)
    assert all(row["expected_detector"] in row["failed_checks"] for row in rows)


def test_req_report_6594_readiness_rejects_matrix_and_protection_mutations(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6594-REDUCER requires rows, effects, tests, and protection."""

    assert mod.readiness_reducer(report)["cfr_authority_audit_ready_score"] == 1.0
    mutations: list[tuple[dict[str, Any], str]] = []
    missing = deepcopy(report)
    missing["per_unit_rows"].pop()
    mutations.append((missing, "primary_matrix"))
    rewritten = deepcopy(report)
    rewritten["per_unit_rows"][1]["expected_effect"] = "rewritten"
    mutations.append((rewritten, "expected_effects"))
    family = deepcopy(report)
    family_row = next(
        row for row in family["per_unit_rows"] if row["attack_id"] == "family_label_swap"
    )
    family_row["observed_effect"]["exact_score_changed"] = True
    mutations.append((family, "family_label_invariance"))
    protected = deepcopy(report)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    mutations.append((protected, "protected_files"))
    tests = deepcopy(report)
    tests["tests_run"][0]["exit_code"] = 1
    mutations.append((tests, "tests_recorded"))
    for candidate, failed_check in mutations:
        reduction = mod.readiness_reducer(candidate)
        assert reduction["checks"][failed_check] is False
        assert reduction["cfr_authority_audit_ready_score"] == 0.0


def test_scenario_report_6594_missing_path_and_none_field_are_named_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6594-MISSING blocks absent and null reducer evidence."""

    missing_root = tmp_path / "missing"
    _copy_upstreams(missing_root)
    (missing_root / mod.REDUCER_RELATIVE_PATH).unlink()
    missing = mod.build_report(
        missing_root,
        date="20260825",
        duration_s=0.2,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
    )
    assert missing["status"] == "blocked_missing_cfr_evidence"
    assert missing["verdict_class"] == "blocked"
    assert missing["per_unit_rows"] == []
    assert missing["cfr_authority_audit_ready_score"] == 0.0
    first = missing["gate_check_summary"]["missing_inputs"][0]
    assert first["path"] == mod.REDUCER_RELATIVE_PATH.as_posix()
    assert first["field"] == "<artifact>"
    assert first["observed_value"] == "missing"
    assert mod.validate_report(missing, missing_root) == []

    null_root = tmp_path / "null"
    _copy_upstreams(null_root)
    reducer_path = null_root / mod.REDUCER_RELATIVE_PATH
    reducer = json.loads(reducer_path.read_text(encoding="utf-8"))
    reducer["per_unit_rows"] = None
    reducer_path.write_text(json.dumps(reducer), encoding="utf-8")
    blocked = mod.build_report(
        null_root,
        duration_s=0.2,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
    )
    assert blocked["status"] == "blocked_missing_cfr_evidence"
    assert any(
        row["path"] == mod.REDUCER_RELATIVE_PATH.as_posix()
        and row["field"] == "per_unit_rows"
        and row["observed_value"] is None
        for row in blocked["gate_check_summary"]["missing_inputs"]
    )
    assert blocked["clean_replay_receipts"] == []

    unreadable_root = tmp_path / "unreadable"
    _copy_upstreams(unreadable_root)

    def unreadable_loader(_path: Path) -> dict[str, Any]:
        raise ValueError("malformed fixture")

    monkeypatch.setattr(mod, "load_json", unreadable_loader)
    _loaded, unreadable = mod._load_upstreams(unreadable_root)
    assert unreadable[0]["observed_value"].startswith("unreadable:")
    assert mod._unit_ids(None) == []


def test_req_report_6594_validation_rejects_tamper_and_bad_block(
    report: dict[str, Any],
) -> None:
    """REQ-REPORT-6594-ATOMIC validates checksum, verdict, and gate details."""

    assert mod.validate_report(report, REPO) == []
    checksum = deepcopy(report)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_report(checksum, REPO)
    invalid_class = deepcopy(report)
    invalid_class["verdict_class"] = "positive"
    assert "verdict_class_invalid" in mod.validate_report(_rehash(invalid_class), REPO)
    missing_field = deepcopy(report)
    missing_field.pop("per_unit_rows")
    assert mod.validate_report(missing_field, REPO)[0].startswith("missing_required_fields:")
    blocked = deepcopy(report)
    blocked["status"] = "blocked_missing_cfr_evidence"
    blocked["honest_verdict"] = "blocked_missing_cfr_evidence: unknown"
    blocked["verdict_class"] = "blocked"
    blocked["gate_check_summary"] = {"missing_inputs": []}
    blocked["cfr_authority_audit_ready_score"] = 0.0
    assert "blocked_verdict_missing_gate_detail" in mod.validate_report(_rehash(blocked), REPO)
    incomplete = deepcopy(report)
    incomplete["cfr_authority_audit_ready_score"] = 0.0
    assert "cfr_authority_audit_ready_score_mismatch" in mod.validate_report(
        _rehash(incomplete), REPO
    )
    truly_incomplete = deepcopy(report)
    truly_incomplete["per_unit_rows"].pop()
    truly_incomplete["cfr_authority_audit_ready_score"] = 0.0
    assert "null_verdict_without_complete_audit" in mod.validate_report(
        _rehash(truly_incomplete), REPO
    )
    malformed = deepcopy(report)
    malformed["inference_substrate"] = "model_loaded"
    malformed["verifier_is_oracle"] = False
    malformed["field_provenance"] = {}
    errors = mod.validate_report(_rehash(malformed), REPO)
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_mismatch" in errors
    assert "field_provenance_mismatch" in errors
    blocked_nonzero = deepcopy(report)
    blocked_nonzero["status"] = "blocked_missing_cfr_evidence"
    blocked_nonzero["verdict_class"] = "blocked"
    blocked_nonzero["gate_check_summary"]["missing_inputs"] = [
        {"path": "missing", "field": "field", "observed_value": None}
    ]
    assert "blocked_verdict_ready_score_nonzero" in mod.validate_report(
        _rehash(blocked_nonzero), REPO
    )
    source_drift = deepcopy(report)
    source_drift["preconditions_checked"]["upstream_artifacts"][0]["sha256"] = "sha256:drift"
    assert any(
        error.startswith("source_artifact_hash_mismatch:")
        for error in mod.validate_report(_rehash(source_drift), REPO)
    )
    with pytest.raises(ValueError, match="unknown meta detector"):
        mod._meta_detector_passed(report, "not-a-detector")


def test_scenario_report_6594_atomic_write_and_cli_use_temporary_targets(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-6594-ATOMIC writes only caller-owned temporary paths."""

    target = tmp_path / "nested" / "audit.json"
    receipt = mod.atomic_write_report(target, report, repo_root=REPO)
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["sha256"] == mod.sha256_file(target)
    assert json.loads(target.read_text(encoding="utf-8")) == report
    bad = deepcopy(report)
    bad["duration_s"] = 0.0
    with pytest.raises(ValueError, match="duration_s_invalid"):
        mod.atomic_write_report(tmp_path / "bad.json", _rehash(bad), repo_root=REPO)
    cli_target = tmp_path / "cli.json"
    assert (
        mod.main(
            [
                "--date",
                "20260825",
                "--repo-root",
                str(REPO),
                "--output",
                str(cli_target),
                "--duration-s",
                "1.0",
            ]
        )
        == 0
    )
    assert json.loads(cli_target.read_text(encoding="utf-8"))["status"] == (
        "complete_cfr_counterfactual_authority_audit"
    )
