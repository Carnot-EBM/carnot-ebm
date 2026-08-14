"""Tests for Exp6433 CSL row-recomputation safety audit.

Spec refs: REQ-LEARN-6433, SCENARIO-LEARN-6433-HASHES,
SCENARIO-LEARN-6433-ROWS, SCENARIO-LEARN-6433-DELTAS,
SCENARIO-LEARN-6433-ATTACKS, SCENARIO-LEARN-6433-ELIGIBILITY.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6433_csl_row_recomputation_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact() -> mod.JsonDict:
    return mod.build_artifact(
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        tests_run=_passing_tests(),
        run_current_audits=False,
    )


def test_req_learn_6433_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6433: OpenSpec owns the Exp6433 audit contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6433") : spec.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6433-HASHES",
        "SCENARIO-LEARN-6433-ROWS",
        "SCENARIO-LEARN-6433-DELTAS",
        "SCENARIO-LEARN-6433-ATTACKS",
        "SCENARIO-LEARN-6433-ELIGIBILITY",
        "prospective_csl_claim_eligibility",
        "verifier_is_oracle",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for phrase in (
        "`field_principles` SHALL map every required field",
        "missing-input rule",
        "recomputation family",
        "attack id",
        "eligibility decision",
        "retirement decision",
    ):
        assert " ".join(phrase.split()) in normalized


def test_scenario_learn_6433_hashes_classify_every_expected_input() -> None:
    """SCENARIO-LEARN-6433-HASHES: inputs are hashed or visibly missing."""

    artifact = _artifact()
    inputs = artifact["expected_and_available_upstream_inputs"]
    ledger = artifact[
        "upstream_artifact_row_manifest_raw_source_test_checker_receipt_head_and_determination_hashes"
    ]
    state = artifact["upstream_state_by_task"]
    preconditions = artifact["preconditions_checked"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert inputs["required_expected_count"] > 0
    assert inputs["missing_required_count"] == 0
    assert inputs["raw_output_count"] == mod.EXP6430_RAW_OUTPUT_COUNT + mod.EXP6432_RAW_OUTPUT_COUNT
    assert ledger["file_role_counts"]["raw_output"] == inputs["raw_output_count"]
    assert ledger["file_role_counts"]["source"] >= 4
    assert ledger["file_role_counts"]["test"] >= 4
    assert ledger["file_role_counts"]["checker"] >= 3
    assert ledger["memory_heads"]["exp6430_capacity_16"]["present"] is True
    assert ledger["determination_records"]["exp6432"]["flagged_adversarial"] is True
    assert state["exp6420"]["state"] == "null"
    assert state["exp6430"]["state"] == "eligible"
    assert state["exp6431"]["underpowered_cell_count"] > 0
    assert state["exp6432"]["state"] == "flagged"
    assert preconditions["repository_state"]["head"]
    assert preconditions["system"]["cpu_count"] >= 1
    assert preconditions["system"]["ram_total_bytes"] > 0
    assert preconditions["system"]["disk_free_bytes"] > 0


def test_scenario_learn_6433_rows_recompute_without_upstream_aggregate_imports() -> None:
    """SCENARIO-LEARN-6433-ROWS: immutable rows drive every audited metric."""

    source = inspect.getsource(mod)
    assert "experiment_6430_prospective_write_once_memory_capacity_frontier" not in source
    assert "experiment_6431_controlled_memory_interference_ab" not in source
    assert "experiment_6432_held_shift_process_restart_csl_replication" not in source

    artifact = _artifact()
    rows = artifact["per_unit_rows"]
    metrics = artifact[
        "independently_recomputed_development_capacity_interference_and_held_metrics"
    ]
    safety = artifact[
        "retention_forgetting_contamination_growth_restart_and_cost_rechecks"
    ]
    uncertainty = artifact["effective_sample_sizes_and_uncertainty_rechecks"]

    assert rows["source_unit_row_count"] == (
        mod.EXP6430_PER_UNIT_ROW_COUNT
        + mod.EXP6431_PER_UNIT_ROW_COUNT
        + mod.EXP6432_PER_UNIT_ROW_COUNT
    )
    assert rows["comparison_row_count"] == len(
        artifact["reported_vs_recomputed_deltas"]["comparisons"]
    )
    assert rows["row_count"] == rows["source_unit_row_count"] + rows["comparison_row_count"]
    assert all(row["included_in_denominator"] for row in rows["rows"] if row["row_type"] == "source_unit")

    cap16 = metrics["development_capacity"]["by_capacity"]["16"]
    assert cap16["future_exact_success_count"] == 30
    assert cap16["future_exact_yield"] == 0.75
    assert cap16["growth"] == 13
    assert cap16["eviction_count"] == 0
    assert cap16["cost"]["cost_units"] == 136

    interference = metrics["interference"]["by_arm"]
    assert interference["authority_aware_retrieval_and_write_controls"]["future_exact_yield"] == 0.75
    assert interference["capacity_matched_baseline_memory"]["future_exact_yield"] == 0.15
    assert interference["authority_aware_retrieval_and_write_controls"]["accepted_invalid_memory_count"] == 0

    held = metrics["held"]["by_arm"]
    assert math.isclose(
        held["selected_capacity_memory"]["future_exact_yield"],
        0.819444444,
        abs_tol=mod.FROZEN_TOLERANCE,
    )
    assert metrics["held"]["held_future_exact_yield_delta"] > 0.0
    assert safety["protected_retention_holds"] is True
    assert safety["contamination_zero"] is True
    assert safety["restart_recovery_holds"] is True
    assert uncertainty["adequate_effective_sample_size"] is True
    assert uncertainty["development_capacity"]["16"]["future_event_count"] == 40
    assert uncertainty["held"]["selected_capacity_memory"]["effective_sample_size"] == 72


def test_scenario_learn_6433_deltas_record_population_filter_and_reason() -> None:
    """SCENARIO-LEARN-6433-DELTAS: every audited value has a comparison row."""

    artifact = _artifact()
    deltas = artifact["reported_vs_recomputed_deltas"]

    assert artifact["mismatch_count"] == 0
    assert deltas["all_within_tolerance"] is True
    assert deltas["comparison_count"] > 40
    for row in deltas["comparisons"]:
        assert row["abs_delta"] <= row["tolerance"]
        assert row["row_population"] > 0
        assert row["filter"]
        assert "numerator" in row
        assert "denominator" in row
        assert row["mismatch_reason"] == ""


def test_scenario_learn_6433_attacks_and_flags_block_claim_eligibility() -> None:
    """SCENARIO-LEARN-6433-ELIGIBILITY: current and stamped flags block claims."""

    artifact = _artifact()
    attacks = artifact["attack_matrix"]
    adversarial = artifact["current_and_stamped_adversarial_findings"]

    assert attacks["all_critical_attacks_fail_closed"] is True
    assert attacks["accepted_attack_count"] == 0
    assert attacks["promoted_attack_count"] == 0
    assert "current_adversarial_flag:DURATION_TOO_SHORT:exp6432" in artifact["open_critical_attack_ids"]
    assert adversarial["current"]["exp6432"]["critical_flag_count"] == 1
    assert adversarial["stamped"]["exp6432"]["flagged_adversarial"] is True
    assert artifact["determination_preservation_findings"]["exit_code"] == 0
    assert artifact["artifact_convention_findings"]["exit_code"] == 0
    assert artifact["prospective_csl_claim_eligibility"] is False
    assert artifact["public_factor_claim_eligibility"] is False
    assert artifact["csl_row_recomputation_audit_ready_score"] == 0.0
    assert artifact["same_verdict_retirement_decision"]["retire_same_verdict"] is False
    assert artifact["status"] == "complete_null"
    assert artifact["blocked_reason"] == "current_adversarial_critical_flag:DURATION_TOO_SHORT:exp6432"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["verifier_is_oracle"] is False
    assert mod.validate_artifact(artifact) is True


def test_req_learn_6433_validation_rejects_eligibility_and_checksum_mutations() -> None:
    """REQ-LEARN-6433: schema validation fails closed on unsafe mutations."""

    artifact = _artifact()
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("prospective_csl_claim_eligibility", lambda data: data.__setitem__("prospective_csl_claim_eligibility", True)),
        ("verifier_is_oracle", lambda data: data.__setitem__("verifier_is_oracle", True)),
        ("open_critical_attack_ids", lambda data: data.__setitem__("open_critical_attack_ids", [])),
        ("reported_vs_recomputed_deltas", lambda data: data["reported_vs_recomputed_deltas"].__setitem__("all_within_tolerance", False)),
        ("mismatch_count", lambda data: data.__setitem__("mismatch_count", 1)),
        ("per_unit_rows", lambda data: data["per_unit_rows"].__setitem__("source_unit_row_count", 1)),
        ("attack_matrix", lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6433_helper_edges_external_audits_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6433: helper edge paths fail closed and writes are stable."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)
    assert mod._num(True) == 1.0
    assert mod._num(None) == 0.0
    assert mod._mean([], "x") == 0.0
    assert mod._path_from_raw("/tmp/outside.raw.json", REPO) == Path("/tmp/outside.raw.json")
    assert mod._path_from_raw("relative.raw.json", REPO) == Path("relative.raw.json")
    assert mod._ci95(0, 0) == [0.0, 0.0]
    command_result = mod._run_command(
        [mod.sys.executable, "-c", "print('ok')"],
        REPO,
    )
    assert command_result["exit_code"] == 0
    assert "ok" in command_result["stdout"]

    context = mod._load_context(REPO)
    blocked_context = deepcopy(context)
    blocked_context["exp6430"]["status"] = "blocked_precondition"
    blocked_context["exp6431"]["status"] = "skipped_upstream"
    states = mod.upstream_state_by_task(blocked_context)
    assert states["exp6430"]["state"] == "blocked"
    assert states["exp6431"]["state"] == "skipped"

    bad_raw_context = deepcopy(context)
    bad_raw_context["exp6430"][
        "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
    ]["events"][0]["raw_output_sha256"] = "sha256:bad"
    assert (
        mod.event_and_raw_output_uniqueness_rechecks(bad_raw_context, REPO)[
            "raw_file_hash_mismatch_count"
        ]
        == 1
    )

    def fake_run(command: list[str], root: Path, timeout: int = 120) -> mod.JsonDict:
        joined = " ".join(command)
        if "adversarial_verify.py" in joined:
            body = {
                "reports": [
                    {
                        "flags": [
                            {
                                "kind": "DURATION_TOO_SHORT",
                                "severity": "critical",
                                "detail": "fixture",
                            }
                        ]
                    }
                ]
            }
            return {"command": joined, "exit_code": 1, "stdout": json.dumps(body), "stderr": ""}
        return {"command": joined, "exit_code": 0, "stdout": "OK", "stderr": ""}

    monkeypatch.setattr(mod, "_run_command", fake_run)
    current = mod.current_and_stamped_adversarial_findings(REPO, context, run_current=True)
    assert current["current"]["exp6430"]["critical_flag_count"] == 1
    assert mod.determination_preservation_findings(REPO, run_current=True)["exit_code"] == 0
    assert mod.artifact_convention_findings(REPO, run_current=True)["exit_code"] == 0

    def bad_json_run(command: list[str], root: Path, timeout: int = 120) -> mod.JsonDict:
        return {"command": "bad", "exit_code": 1, "stdout": "not-json", "stderr": "parse failed"}

    monkeypatch.setattr(mod, "_run_command", bad_json_run)
    parsed = mod.current_and_stamped_adversarial_findings(REPO, context, run_current=True)
    assert parsed["current"]["exp6430"]["flags"][0]["kind"] == "adversarial_verify_parse_error"

    def raise_os(*args: object, **kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr(mod.subprocess, "run", raise_os)
    assert mod._git(["status"], REPO) == ""

    artifact = _artifact()
    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["all_required_inputs_present"] = False
    blocked["prospective_csl_claim_eligibility"] = False
    mod.refresh_terminal_fields(blocked)
    assert blocked["status"] == "complete_blocked"

    eligible = deepcopy(artifact)
    eligible["preconditions_checked"]["all_required_inputs_present"] = True
    eligible["prospective_csl_claim_eligibility"] = True
    mod.refresh_terminal_fields(eligible)
    assert eligible["status"] == "complete_ready"

    checks = deepcopy(artifact)
    base_args = {
        "inputs": checks["expected_and_available_upstream_inputs"],
        "deltas": checks["reported_vs_recomputed_deltas"],
        "uncertainty": checks["effective_sample_sizes_and_uncertainty_rechecks"],
        "safety": checks["retention_forgetting_contamination_growth_restart_and_cost_rechecks"],
        "attacks": checks["attack_matrix"],
        "adversarial": {"current": {}},
        "metrics": checks[
            "independently_recomputed_development_capacity_interference_and_held_metrics"
        ],
    }
    bad_inputs = deepcopy(base_args)
    bad_inputs["inputs"]["missing_required_count"] = 1
    assert "missing_required_input" in mod._eligibility(**bad_inputs)[1]
    bad_deltas = deepcopy(base_args)
    bad_deltas["deltas"]["all_within_tolerance"] = False
    assert "reported_values_do_not_recompute" in mod._eligibility(**bad_deltas)[1]
    bad_uncertainty = deepcopy(base_args)
    bad_uncertainty["uncertainty"]["adequate_effective_sample_size"] = False
    assert "inadequate_effective_sample_size" in mod._eligibility(**bad_uncertainty)[1]
    bad_metrics = deepcopy(base_args)
    bad_metrics["metrics"]["development_capacity"]["by_capacity"]["16"]["future_exact_yield"] = 0.0
    bad_metrics["metrics"]["held"]["held_future_exact_yield_delta"] = 0.0
    blockers = mod._eligibility(**bad_metrics)[1]
    assert "development_future_effect_not_positive" in blockers
    assert "held_future_effect_not_positive" in blockers
    bad_safety = deepcopy(base_args)
    bad_safety["safety"]["protected_retention_holds"] = False
    bad_safety["safety"]["contamination_zero"] = False
    blockers = mod._eligibility(**bad_safety)[1]
    assert "protected_retention_regression" in blockers
    assert "contamination_nonzero" in blockers
    bad_attacks = deepcopy(base_args)
    bad_attacks["attacks"]["all_critical_attacks_fail_closed"] = False
    assert "critical_attack_open" in mod._eligibility(**bad_attacks)[1]

    output = tmp_path / "artifact.json"
    written = mod.write_artifact(
        output_path=output,
        root=REPO,
        duration_s=0.25,
        tests_run=_passing_tests(),
        run_current_audits=False,
    )
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == written
