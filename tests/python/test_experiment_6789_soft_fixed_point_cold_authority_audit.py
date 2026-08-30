"""Tests for the independent V592 soft fixed-point cold authority audit.

Spec refs: REQ-VERIFY-6789 and SCENARIO-VERIFY-6789-*.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6789_soft_fixed_point_cold_authority_audit as mod


@pytest.fixture(scope="module")
def sources() -> dict:
    """REQ-VERIFY-6789: Load source JSON through the audit's independent loader."""

    return mod.load_source_bundle(mod.REPO_ROOT)


@pytest.fixture(scope="module")
def artifact(sources: dict) -> dict:
    """REQ-VERIFY-6789: Build one complete fresh-process audit for all tests."""

    return mod.build_artifact(
        sources,
        run_date="20260830",
        duration_s=0.25,
        bootstrap_resamples=mod.BOOTSTRAP_RESAMPLES,
        fresh_process=True,
    )


def test_req_verify_6789_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6789: The capability spec owns every cold-audit behavior."""

    text = (mod.REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    start = text.index("### REQ-VERIFY-6789")
    section = text[start : text.index("### SCENARIO-VERIFY-6745-DUAL", start)]
    for anchor in (
        "REQ-VERIFY-6789",
        "SCENARIO-VERIFY-6789-INDEPENDENT-RECOMPUTE",
        "SCENARIO-VERIFY-6789-DESTRUCTIVE-CONTROLS",
        "SCENARIO-VERIFY-6789-ORACLE-ORDER",
        "SCENARIO-VERIFY-6789-HARD-NEGATIVE",
        "SCENARIO-VERIFY-6789-BLOCKED",
        "complete_blocked_fixed_point_cold_audit",
        "fixed_point_audit_completed",
    ):
        assert anchor in section


def test_scenario_verify_6789_module_has_no_source_experiment_imports() -> None:
    """SCENARIO-VERIFY-6789-INDEPENDENT-RECOMPUTE forbids source helper imports."""

    tree = ast.parse((mod.REPO_ROOT / mod.MODULE_PATH).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        f"experiment_{number}" in name for name in imports for number in (6786, 6787, 6788)
    )


def test_scenario_verify_6789_preconditions_require_all_authority(sources: dict) -> None:
    """SCENARIO-VERIFY-6789-BLOCKED checks hashes, keys, candidates, and receipts."""

    clean = mod.check_preconditions(sources, repo_root=mod.REPO_ROOT)
    assert clean["all_passed"] is True
    assert clean["failed_checks"] == []

    incomplete = deepcopy(sources)
    incomplete["exp6788"]["fixed_point_comparison_completed"] = False
    assert "fixed_point_comparison_completed" in {
        row["check"]
        for row in mod.check_preconditions(incomplete, repo_root=mod.REPO_ROOT)["failed_checks"]
    }

    missing = deepcopy(sources)
    missing["exp6788"]["rows"].pop()
    checked = mod.check_preconditions(missing, repo_root=mod.REPO_ROOT)
    assert "planned_row_keys" in {row["check"] for row in checked["failed_checks"]}
    assert checked["missing_planned_rows"] is True

    malformed = deepcopy(sources)
    malformed["exp6788"]["rows"][0]["candidates"] = []
    assert "raw_candidates_and_exact_receipts" in {
        row["check"]
        for row in mod.check_preconditions(malformed, repo_root=mod.REPO_ROOT)["failed_checks"]
    }


def test_req_verify_6789_independent_exact_semantics_and_hashes(sources: dict) -> None:
    """REQ-VERIFY-6789 independently enumerates validity, distance, and hashes."""

    unit = next(
        row
        for row in sources["exp6786"]["frozen_manifest"]["units"]
        if row["split"] == "development"
    )
    graph = unit["graph"]
    valid = mod.enumerate_valid_assignments(graph)
    assert valid
    receipt = mod.evaluate_assignment(graph, valid[0])
    assert receipt["exact_valid"] is True
    assert receipt["failed_dependency_ids"] == []
    assert mod.nearest_valid_distance(graph, valid[0], valid) == 0
    assert mod.candidate_digest(valid[0]).startswith("sha256:")

    invalid = deepcopy(valid[0])
    variables = graph["local_groups"][0]["variables"]
    invalid[variables[0]] = 1
    invalid[variables[1]] = 1
    receipt = mod.evaluate_assignment(graph, invalid)
    assert receipt["exact_valid"] is False
    assert receipt["failed_local_group_ids"] == [graph["local_groups"][0]["group_id"]]
    assert mod.nearest_valid_distance(graph, invalid, valid) >= 1

    changed = deepcopy(graph)
    changed["dependency_edges"][0]["relation_type"] = "unknown"
    with pytest.raises(mod.AuditInputError, match="dependency relation"):
        mod.enumerate_valid_assignments(changed)


def test_scenario_verify_6789_source_rows_recompute_every_receipt(sources: dict) -> None:
    """SCENARIO-VERIFY-6789-INDEPENDENT-RECOMPUTE derives all 640 source rows."""

    rows = mod.recompute_source_rows(sources)
    assert len(rows) == mod.PLANNED_SOURCE_ROW_COUNT
    assert len({row["row_id"] for row in rows}) == mod.PLANNED_SOURCE_ROW_COUNT
    assert all(row["candidate_hashes_match"] for row in rows)
    assert all(row["source_exact_outcomes_match"] for row in rows)
    assert all(row["exact_checker_after_candidate_freeze"] for row in rows)
    assert all(row["candidate_budget"] == 3 for row in rows)
    assert {row["arm"] for row in rows} == set(mod.ARMS)


def test_scenario_verify_6789_controls_are_complete_deterministic_and_matched(
    sources: dict,
) -> None:
    """SCENARIO-VERIFY-6789-DESTRUCTIVE-CONTROLS emits four rows per source row."""

    source_rows = mod.recompute_source_rows(sources)
    first = mod.build_control_rows(source_rows, sources)
    second = mod.build_control_rows(source_rows, sources)
    assert first == second
    assert len(first) == mod.PLANNED_CONTROL_ROW_COUNT
    assert len({row["row_id"] for row in first}) == mod.PLANNED_CONTROL_ROW_COUNT
    counts = {control: sum(row["control"] == control for row in first) for control in mod.CONTROLS}
    assert counts == {control: mod.PLANNED_SOURCE_ROW_COUNT for control in mod.CONTROLS}
    assert all(row["budget_match"] for row in first)
    assert all(row["control_seed"] == mod.CONTROL_SEEDS[row["control"]] for row in first)
    rewired = [row for row in first if row["control"] == mod.DEPENDENCY_REWIRE]
    assert all(row["control_receipt"]["degree_strata_preserved"] for row in rewired)


def test_req_verify_6789_metrics_and_headlines_are_row_derived(
    sources: dict, artifact: dict
) -> None:
    """REQ-VERIFY-6789 recomputes every source headline and confidence interval."""

    assert artifact["fixed_point_audit_completed"] is True
    assert len(artifact["rows"]) == mod.PLANNED_AUDIT_ROW_COUNT
    reduced = mod.reduce_audit_rows(
        artifact["rows"],
        bootstrap_resamples=mod.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=mod.BOOTSTRAP_SEED,
    )
    assert artifact["cold_recomputed_metrics"] == reduced["source"]
    assert artifact["label_permutation_effect"] == reduced[mod.LABEL_PERMUTATION]
    assert artifact["group_id_permutation_effect"] == reduced[mod.GROUP_ID_PERMUTATION]
    assert artifact["dependency_rewire_effect"] == reduced[mod.DEPENDENCY_REWIRE]
    assert artifact["topology_id_swap_effect"] == reduced[mod.TOPOLOGY_ID_SWAP]
    assert artifact["headline_differences"]["all_match"] is True
    assert artifact["headline_differences"]["maximum_absolute_difference"] <= 1.0e-10
    assert artifact["cold_recomputed_metrics"]["paired_exact_valid_delta_ci95"]["resamples"] == 2000


def test_scenario_verify_6789_oracle_order_and_hard_negative_boundary(artifact: dict) -> None:
    """SCENARIO-VERIFY-6789-ORACLE-ORDER proves checking follows candidate freeze."""

    order = artifact["exact_checker_call_order"]
    assert order["static_order_proven"] is True
    assert order["runtime_order_proven"] is True
    assert order["candidate_hash_mismatch_count"] == 0
    assert artifact["oracle_feature_violations"] == []
    assert artifact["verifier_is_oracle"] is False

    findings = artifact["hard_negative_shortcut_findings"]
    assert findings["all_recomputed_candidates_pass_local_checks"] is True
    assert (
        findings["effect_after_local_checks"]
        == artifact["cold_recomputed_metrics"]["paired_exact_valid_delta"]
    )
    assert findings["explained_only_by_easy_local_failures"] is False


def test_req_verify_6789_terminal_schema_and_validation(artifact: dict) -> None:
    """REQ-VERIFY-6789 emits every principle, closed verdict, and reproducibility proof."""

    assert artifact["field_principles"].keys() == artifact.keys()
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verdict_class"] in mod.VERDICT_CLASSES
    assert artifact["honest_verdict"].startswith(mod.TERMINAL_PREFIXES)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["gate_check_summary"]["all_passed"] is True
    assert mod.validate_artifact(artifact) == []

    drifted = deepcopy(artifact)
    drifted["cold_recomputed_metrics"]["paired_exact_valid_delta"] += 0.5
    assert "row_derived_metrics" in mod.validate_artifact(drifted)
    invalid = deepcopy(artifact)
    invalid["verdict_class"] = "maybe"
    assert "verdict_class" in mod.validate_artifact(invalid)


def test_scenario_verify_6789_blocked_artifact_stops_without_rows(sources: dict) -> None:
    """SCENARIO-VERIFY-6789-BLOCKED emits a complete fail-closed artifact."""

    changed = deepcopy(sources)
    changed["exp6788"]["fixed_point_comparison_completed"] = False
    blocked = mod.build_artifact(
        changed,
        run_date="20260830",
        duration_s=0.1,
        bootstrap_resamples=32,
        fresh_process=False,
    )
    assert blocked["status"] == "complete_blocked_fixed_point_cold_audit"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["fixed_point_audit_completed"] is False
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []

    missing = deepcopy(sources)
    missing["exp6788"]["rows"].pop()
    disqualified = mod.build_artifact(
        missing,
        run_date="20260830",
        duration_s=0.1,
        bootstrap_resamples=32,
        fresh_process=False,
    )
    assert disqualified["status"] == "complete_blocked_fixed_point_cold_audit"
    assert disqualified["verdict_class"] == "disqualified"


def test_req_verify_6789_writer_and_script_entrypoint_use_explicit_output(
    artifact: dict, tmp_path: Path
) -> None:
    """REQ-VERIFY-6789 writes atomically to an explicit test path."""

    output = tmp_path / "audit.json"
    mod.write_output(artifact, output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert mod.parse_run_date("20260830") == "20260830"
    with pytest.raises(ValueError, match="YYYYMMDD"):
        mod.parse_run_date("2026-08-30")
