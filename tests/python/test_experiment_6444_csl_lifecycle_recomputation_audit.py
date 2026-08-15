"""Tests for Exp6444 CSL lifecycle recomputation audit.

Spec refs: REQ-LEARN-6444, SCENARIO-LEARN-6444-INVENTORY,
SCENARIO-LEARN-6444-REDUCERS, SCENARIO-LEARN-6444-CHAINS,
SCENARIO-LEARN-6444-ATTACKS, SCENARIO-LEARN-6444-DELIVERABLE.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6444_csl_lifecycle_recomputation_audit as mod


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


def test_req_learn_6444_spec_declares_schema_and_ready_conditions() -> None:
    """REQ-LEARN-6444: OpenSpec owns the Exp6444 audit contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6444") : spec.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6444-INVENTORY",
        "SCENARIO-LEARN-6444-REDUCERS",
        "SCENARIO-LEARN-6444-CHAINS",
        "SCENARIO-LEARN-6444-ATTACKS",
        "SCENARIO-LEARN-6444-DELIVERABLE",
        "prospective_csl_eligibility",
        "csl_audit_ready_score",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for phrase in (
        "`field_principles` SHALL map every required field",
        "`csl_audit_ready_score` condition",
        "without importing upstream aggregate, uncertainty, gating, or verdict functions",
        "blocked or missing upstream evidence must stay visible",
    ):
        assert " ".join(phrase.split()) in normalized


def test_scenario_learn_6444_inventory_keeps_missing_and_blocked_evidence_visible() -> None:
    """SCENARIO-LEARN-6444-INVENTORY: V554 blockers remain audit inputs."""

    artifact = _artifact()
    inventory = artifact["upstream_inventory_and_hashes"]
    state = artifact["upstream_status_verdict_readiness_and_adversarial_findings"]
    preconditions = artifact["preconditions_checked"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert inventory["required_upstream_artifact_missing_count"] == 2
    assert inventory["tasks"]["exp6441"]["artifact"]["state"] == "missing"
    assert inventory["tasks"]["exp6442"]["artifact"]["state"] == "present"
    assert inventory["tasks"]["exp6443"]["artifact"]["state"] == "missing"
    assert state["exp6442"]["state"] == "blocked"
    assert state["exp6442"]["gate_check_summary"].startswith("1 of 1 gate")
    assert state["exp6442"]["readiness_fields"]["prospective_csl_ready_score"] is None
    assert preconditions["inventory_before_experiment_module_imports"] is True
    assert preconditions["does_not_import_upstream_experiment_modules"] is True
    assert preconditions["required_v554_upstream_artifacts_present"] is False
    assert preconditions["system"]["cpu_count"] >= 1
    assert artifact["gate_check_summary"]["failed_check_count"] >= 3
    assert "missing_required_upstream_artifact:exp6441" in artifact["csl_ineligibility_reasons"]
    assert "blocked_upstream:exp6442" in artifact["csl_ineligibility_reasons"]
    assert "missing_required_upstream_artifact:exp6443" in artifact["csl_ineligibility_reasons"]
    assert artifact["status"] == "complete_blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked:")
    assert artifact["csl_audit_ready_score"] == 0.0


def test_scenario_learn_6444_reducers_recompute_rows_without_upstream_imports() -> None:
    """SCENARIO-LEARN-6444-REDUCERS: immutable rows drive all metrics."""

    source = inspect.getsource(mod)
    forbidden_imports = (
        "from carnot import experiment_6430",
        "from carnot import experiment_6431",
        "from carnot import experiment_6432",
        "from carnot import experiment_6441",
        "from carnot import experiment_6442",
        "from carnot import experiment_6443",
        "import carnot.experiment_6430",
        "import carnot.experiment_6431",
        "import carnot.experiment_6432",
        "import carnot.experiment_6441",
        "import carnot.experiment_6442",
        "import carnot.experiment_6443",
    )
    assert not any(pattern in source for pattern in forbidden_imports)

    artifact = _artifact()
    rows = artifact["per_unit_rows"]
    development = artifact["development_metric_recomputation"]
    held = artifact["held_metric_recomputation"]
    lifecycle = artifact["lifecycle_safety_metric_recomputation"]
    mismatches = artifact["upstream_vs_recomputed_mismatches"]

    assert rows["source_unit_row_count"] == (
        mod.EXP6430_PER_UNIT_ROW_COUNT
        + mod.EXP6431_PER_UNIT_ROW_COUNT
        + mod.EXP6432_PER_UNIT_ROW_COUNT
    )
    assert rows["comparison_row_count"] == mismatches["comparison_count"]
    assert rows["row_count"] == rows["source_unit_row_count"] + rows["comparison_row_count"]
    assert all(row["included_in_denominator"] for row in rows["rows"])

    cap16 = development["by_capacity"]["16"]
    assert cap16["proposal_coverage"] == 1.0
    assert cap16["admission_precision"] == 1.0
    assert cap16["future_exact_success_count"] == 30
    assert cap16["future_exact_yield"] == 0.75
    assert development["paired_deltas"]["capacity_16_vs_frozen_future_exact_yield"] == 0.75
    assert development["memory_growth"]["16"] == 13
    assert development["online_cost"]["16"]["cost_units"] == 136

    selected = held["by_arm"]["selected_capacity_memory"]
    assert selected["future_exact_success_count"] == 59
    assert math.isclose(selected["future_exact_yield"], 0.819444444, abs_tol=mod.FROZEN_TOLERANCE)
    assert held["paired_deltas"]["selected_minus_frozen_future_exact_yield"] == 0.819444444
    assert held["uncertainty"]["selected_capacity_memory"]["effective_sample_size"] == 72

    authority = lifecycle["by_arm"]["authority_aware_retrieval_and_write_controls"]
    baseline = lifecycle["by_arm"]["capacity_matched_baseline_memory"]
    assert authority["future_exact_yield"] == 0.75
    assert baseline["future_exact_yield"] == 0.15
    assert lifecycle["unsafe_authoring_count"] == 0
    assert lifecycle["unsafe_retrieval_count"] == 0
    assert lifecycle["protected_release_count"] == 0
    assert lifecycle["resurrection_count"] == 0
    assert lifecycle["rollback_success"] == 1.0
    assert lifecycle["quarantine_precision"] == 1.0
    assert lifecycle["quarantine_recall"] == 1.0
    assert artifact["mismatch_count_and_materiality"]["material_row_mismatch_count"] == 0
    assert mismatches["all_within_tolerance"] is True


def test_scenario_learn_6444_chain_and_attack_checks_block_eligibility() -> None:
    """SCENARIO-LEARN-6444-ATTACKS: blockers force CSL ineligibility."""

    artifact = _artifact()
    raw = artifact["raw_output_uniqueness_and_cross_task_intersections"]
    chronology = artifact["chronology_future_seal_and_capacity_checks"]
    memory = artifact["memory_head_transaction_and_restart_checks"]
    chain = artifact["command_path_chain_checks"]
    veto = artifact["exact_veto_checks"]
    attacks = artifact["independent_attack_replay"]
    substrate = artifact["duration_and_substrate_eligibility"]
    adversarial = artifact["current_adversarial_findings"]

    assert raw["raw_output_count"] == mod.EXP6430_RAW_OUTPUT_COUNT + mod.EXP6432_RAW_OUTPUT_COUNT
    assert raw["raw_output_reuse_count"] == 0
    assert raw["development_held_hash_disjoint"] is True
    assert chronology["proposal_before_outcome_violation_count"] == 0
    assert chronology["future_label_used_for_proposal_count"] == 0
    assert chronology["capacity_violation_count"] == 0
    assert memory["all_recovered_heads_match"] is True
    assert memory["true_process_boundaries"] is True
    assert memory["transaction_ancestry_present"] is True
    assert chain["complete_generation_to_verdict_chain"] is False
    assert "exp6443" in chain["missing_chain_tasks"]
    assert veto["exact_veto_preserved"] is True
    assert attacks["all_present_evidence_attacks_fail_closed"] is True
    assert attacks["required_evidence_attack_open"] is True
    assert substrate["current_artifact_duration_floor_met"] is True
    assert substrate["upstream_duration_or_substrate_blockers"]
    assert adversarial["current"]["exp6432"]["critical_flag_count"] == 1
    assert "current_adversarial_critical_flag:DURATION_TOO_SHORT:exp6432" in artifact["csl_ineligibility_reasons"]
    assert artifact["prospective_csl_eligibility"] is False


def test_req_learn_6444_validation_rejects_mutations_and_checksum_drift() -> None:
    """REQ-LEARN-6444: artifact validation fails closed."""

    artifact = _artifact()
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("prospective_csl_eligibility", lambda data: data.__setitem__("prospective_csl_eligibility", True)),
        ("csl_audit_ready_score", lambda data: data.__setitem__("csl_audit_ready_score", 1.0)),
        ("mismatch_count_and_materiality", lambda data: data["mismatch_count_and_materiality"].__setitem__("material_row_mismatch_count", 1)),
        ("upstream_vs_recomputed_mismatches", lambda data: data["upstream_vs_recomputed_mismatches"].__setitem__("all_within_tolerance", False)),
        ("gate_check_summary", lambda data: data["gate_check_summary"].__setitem__("failed_check_count", 0)),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "blocked_gate_check_failed")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    branch_args = {
        "inventory": {"missing_required_upstream_artifacts": []},
        "upstream_state": {"exp6441": {"state": "eligible"}, "exp6442": {"state": "eligible"}, "exp6443": {"state": "eligible"}},
        "mismatches": {"material_row_mismatch_count": 1},
        "development": deepcopy(artifact["development_metric_recomputation"]),
        "held": deepcopy(artifact["held_metric_recomputation"]),
        "lifecycle": deepcopy(artifact["lifecycle_safety_metric_recomputation"]),
        "duration_substrate": {"upstream_duration_or_substrate_blockers": []},
        "attacks": {"required_evidence_attack_open": False, "all_present_evidence_attacks_fail_closed": False},
        "adversarial": {"current": {}},
    }
    branch_args["development"]["by_capacity"]["16"]["future_exact_yield"] = 0.0
    branch_args["held"]["paired_deltas"]["selected_minus_frozen_future_exact_yield"] = 0.0
    branch_args["lifecycle"]["safety_regression_count"] = 1
    branch_args["lifecycle"]["protected_release_count"] = 1
    branch_args["development"]["bounded_growth"] = False
    branch_reasons = mod.csl_ineligibility_reasons(**branch_args)
    for reason in (
        "material_row_mismatch",
        "development_future_exact_effect_not_positive",
        "held_future_exact_effect_not_positive",
        "lifecycle_safety_regression",
        "protected_release_nonzero",
        "growth_unbounded",
        "critical_attack_open",
    ):
        assert reason in branch_reasons


def test_req_learn_6444_helper_edges_and_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-6444-DELIVERABLE: helpers handle edge paths."""

    missing = tmp_path / "missing.json"
    assert mod.sha256_file(missing) is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(malformed)
    assert mod._file_receipt(malformed, "artifact", tmp_path)["state"] == "malformed"
    zero = tmp_path / "zero.json"
    zero.write_text("", encoding="utf-8")
    assert mod._file_receipt(zero, "source", tmp_path)["state"] == "zero_byte"
    assert mod._num(True) == 1.0
    assert mod._num(None) == 0.0
    assert mod._mean([], "x") == 0.0
    assert mod._ci95(0, 0) == [0.0, 0.0]
    assert mod._path_from_raw("/tmp/outside.raw.json", REPO) == Path("/tmp/outside.raw.json")
    assert mod._path_from_raw("relative.raw.json", REPO) == Path("relative.raw.json")
    assert mod._classify_upstream("exp6441", {"status": "skipped_upstream"}, {"state": "present"}) == "skipped"

    command_result = mod._run_command([mod.sys.executable, "-c", "print('ok')"], REPO)
    assert command_result["exit_code"] == 0
    assert "ok" in command_result["stdout"]

    context = mod.load_upstream_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6430"][
        "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
    ]["events"][0]["raw_output_sha256"] = "sha256:bad"
    assert mod.raw_output_uniqueness_and_cross_task_intersections(bad_context, REPO)["raw_file_hash_mismatch_count"] == 1

    def good_json_run(command: list[str], root: Path, timeout: int = 120) -> mod.JsonDict:
        return {"command": "good", "exit_code": 0, "stdout": '{"reports":[{"flags":[]}]}', "stderr": ""}

    monkeypatch.setattr(mod, "_run_command", good_json_run)
    parsed_good = mod.current_adversarial_findings(REPO, context, run_current=True)
    assert parsed_good["current"]["exp6430"]["flag_count"] == 0

    def bad_json_run(command: list[str], root: Path, timeout: int = 120) -> mod.JsonDict:
        return {"command": "bad", "exit_code": 1, "stdout": "not-json", "stderr": "parse failed"}

    monkeypatch.setattr(mod, "_run_command", bad_json_run)
    parsed = mod.current_adversarial_findings(REPO, context, run_current=True)
    assert parsed["current"]["exp6430"]["flags"][0]["kind"] == "adversarial_verify_parse_error"

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
