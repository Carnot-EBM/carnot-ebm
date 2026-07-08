"""Tests for Exp5441 .495 transition receipt.

Spec refs: REQ-REPORT-5441, SCENARIO-REPORT-5441,
SCENARIO-REPORT-5441-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5441_transition_v495 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roadmap(milestone: str, task_ids: list[str] | None = None) -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": milestone,
            "deliverable": f"results/{task_id}.json",
            "title": f"fixture {task_id}",
            "agent_type": "codex",
            "model": "gpt-5.5",
            "prompt": "REQ-REPORT-5441 fixture",
        }
        for task_id in (task_ids or mod.EXPECTED_TASK_IDS)
    ]
    return yaml.safe_dump(
        {
            "milestone": milestone,
            "milestone_title": "fixture transition",
            "milestone_doc": str(mod.VNEXT_RELATIVE_PATH),
            "tasks": tasks,
        },
        sort_keys=False,
    )


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5441-5453") -> str:
    return f"""# Research Roadmap vNEXT - Milestone {milestone}

**Milestone title:** fixture
**Previous milestone:** {mod.PREVIOUS_MILESTONE}
**Task range:** {task_range}
**Pre-staged roadmap:** `research-roadmap-next.yaml`
"""


def _lane(
    lane: str,
    classification: str,
    evidence: dict[str, Any],
    *,
    blocked_reason: str = "",
    claim_boundary: str = "fixture boundary",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "lane": lane,
        "classification": classification,
        "claim_boundary": claim_boundary,
        "evidence": evidence,
        "source_artifacts": [f"results/{lane}.json"],
    }
    if blocked_reason:
        row["blocked_reason"] = blocked_reason
    return row


def _capstone_payload() -> dict[str, Any]:
    headline = [
        _lane(
            "structured_corrigendum",
            "headline_ready",
            {
                "structured_corrigendum_clean": True,
                "row_count_recomputed": 84,
                "prefix_metric_independence_check": True,
                "risk_metric_independence_check": True,
                "gpu_offload_verified": True,
            },
            claim_boundary="clean row-level corrigendum",
        ),
        _lane(
            "structured_taxonomy_replication",
            "headline_ready",
            {
                "structured_taxonomy_replication_ready": True,
                "fixture_count": 42,
                "metric_independence_checks_passed": True,
                "semantic_false_accept_rate": 0.0,
            },
            claim_boundary="local structured taxonomy replication",
        ),
        _lane(
            "ontology_softlogic_memory",
            "headline_ready",
            {
                "ontology_constraint_memory_ready": True,
                "deterministic_solver_authority": True,
                "false_triple_rejection_rate": 1.0,
                "soft_logic_overrode_solver": False,
                "triple_count": 51,
            },
            claim_boundary="deterministic ontology-memory fixture",
        ),
        _lane(
            "verified_workflow_memory_csl",
            "headline_ready",
            {
                "verified_workflow_memory_ready": True,
                "verify_before_store_pass_rate": 0.5,
                "workflow_episode_count": 8,
                "rollback_verified": True,
                "no_weight_mutation": True,
            },
            claim_boundary="workflow memory with verification-before-store",
        ),
        _lane(
            "csl_memory_transfer_stress",
            "headline_ready",
            {
                "csl_transfer_stress_ready": True,
                "in_domain_quality_delta": 0.08,
                "negative_transfer_deflection_rate": 1.0,
                "rollback_verified": True,
                "no_weight_mutation": True,
            },
            claim_boundary="workflow-memory transfer stress",
        ),
    ]
    bounded = [
        _lane(
            "active_constraint_diversity_lns",
            "bounded",
            {
                "active_constraint_diversity_ready": True,
                "fixture_count": 4,
                "work_delta": 138,
                "solver_validity_preserved": True,
                "accepted_hint_count": 4,
            },
        ),
        _lane(
            "pbit_polarfire_timing_variance",
            "bounded",
            {
                "timing_variance_receipts_ready": True,
                "cpu_repeat_count": 10,
                "board_repeat_count": 10,
                "same_workload_hash_match": True,
                "same_result_hash_match": True,
                "hardware_speedup_claim": False,
            },
        ),
        _lane(
            "kan_ontology_certificates",
            "bounded",
            {
                "kan_ontology_certificate_ready": True,
                "certificate_count": 16,
                "false_property_rejection_rate": 1.0,
                "broad_kan_verification_claim": False,
            },
        ),
    ]
    blocked = [
        _lane(
            "token_internal_feature_lane_closed",
            "blocked",
            {"backend_receipt_present": False, "future_token_signal_allowed": False},
            blocked_reason="no_authenticated_backend_receipt",
        )
    ]
    honest_null = [
        _lane(
            "arc_live_reinduction_levelup",
            "honest_null",
            {
                "status": "honest_null",
                "target_game": "cn04",
                "target_level": "L4",
                "arc_new_level_banked": False,
                "attempt_count": 51,
                "registry_total_before": 69,
                "registry_total_after": 69,
                "failure_mode": "bounded_budget_no_levelup",
            },
            blocked_reason="bounded_budget_no_levelup",
            claim_boundary="live ARC path ran; no new level was banked",
        )
    ]
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": (
            "complete: .494 capstone emitted from actual artifacts; structured "
            "corrigendum and taxonomy replication, ontology memory, verified workflow "
            "CSL, and CSL transfer stress are headline-ready; active constraints, p-bit "
            "timing, and KAN ontology certificates are bounded; ARC no-bank keeps the "
            "north-star count unchanged; no hardware speedup is claimed; token/internal "
            "lane closed."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "hardware_speedup_claim": False,
        "headline_ready_lanes": headline,
        "bounded_lanes": bounded,
        "blocked_lanes": blocked,
        "honest_null_lanes": honest_null,
    }


def _gap_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: .494 PRD gap table read actual .494 upstream artifacts.",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "prd_gap_table_ready": True,
    }


def _make_repo(
    root: Path,
    *,
    capstone: dict[str, Any] | None = None,
    gap: dict[str, Any] | None = None,
    milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    doc_task_range: str = "Exp 5441-5453",
    task_ids: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(
        _roadmap(milestone, task_ids),
        encoding="utf-8",
    )
    (root / mod.VNEXT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        _vnext_doc(doc_milestone, doc_task_range),
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/status.md").write_text("fixture status\n", encoding="utf-8")
    (root / "ops/changelog.md").write_text("fixture changelog\n", encoding="utf-8")
    (root / "ops/conductor-log.md").write_text(
        "| 2026-07-08 19:41 UTC | Exp5440 | OK | capstone complete |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    if gap is not None:
        _write_json(root / mod.GAP_RELATIVE_PATH, gap)
    return root


def test_req_report_5441_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5441: OpenSpec anchors the .495 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5441") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5441",
        "SCENARIO-REPORT-5441",
        "SCENARIO-REPORT-5441-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "exp5428-exp5440",
        "exp5441-exp5453",
        "structured corrigendum",
        "structured taxonomy replication",
        "ontology memory",
        "verified workflow CSL",
        "CSL transfer stress",
        "ARC no-bank on `cn04` L4",
        "absence of a hardware speedup claim",
        "token/internal feature lane closed",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5441_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5441: active .495 records .494 boundaries and gates."""

    root = _make_repo(tmp_path, capstone=_capstone_payload(), gap=_gap_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-08",
        tests_run=[{"command": "unit 5441", "outcome": "passed"}],
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    mod.validate_artifact(artifact)
    assert (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8") == roadmap_before
    assert (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8") == conductor_before
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["previous_milestone"] == mod.PREVIOUS_MILESTONE
    assert artifact["prior_capstone_path"] == str(mod.CAPSTONE_RELATIVE_PATH)
    assert artifact["previous_task_range"] == mod.PREVIOUS_TASK_RANGE
    assert artifact["next_task_range"] == mod.NEXT_TASK_RANGE
    assert artifact["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    closed = {row["lane"]: row for row in artifact["closed_lanes"]}
    assert list(closed) == [
        "structured_corrigendum",
        "structured_taxonomy_replication",
        "ontology_softlogic_memory",
        "verified_workflow_memory_csl",
        "csl_memory_transfer_stress",
    ]
    assert closed["structured_corrigendum"]["terminal_evidence"][
        "row_count_recomputed"
    ] == 84
    assert closed["structured_taxonomy_replication"]["terminal_evidence"][
        "fixture_count"
    ] == 42
    assert closed["ontology_softlogic_memory"]["terminal_evidence"][
        "deterministic_solver_authority"
    ] is True
    assert closed["verified_workflow_memory_csl"]["terminal_evidence"][
        "no_weight_mutation"
    ] is True
    assert closed["csl_memory_transfer_stress"]["terminal_evidence"][
        "negative_transfer_deflection_rate"
    ] == 1.0

    partial = {row["lane"]: row for row in artifact["partial_lanes"]}
    assert list(partial) == [
        "active_constraint_diversity_lns",
        "pbit_polarfire_timing_variance",
        "kan_ontology_certificates",
    ]
    assert partial["active_constraint_diversity_lns"]["terminal_evidence"][
        "work_delta"
    ] == 138
    assert partial["pbit_polarfire_timing_variance"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert partial["kan_ontology_certificates"]["terminal_evidence"][
        "broad_kan_verification_claim"
    ] is False

    blocked = {row["lane"]: row for row in artifact["blocked_lanes"]}
    assert list(blocked) == ["token_internal_feature_lane_closed"]
    assert blocked["token_internal_feature_lane_closed"]["terminal_evidence"][
        "future_token_signal_allowed"
    ] is False

    honest_null = {row["lane"]: row for row in artifact["honest_null_lanes"]}
    assert list(honest_null) == ["arc_live_reinduction_levelup", "hardware_speedup_claim"]
    assert honest_null["arc_live_reinduction_levelup"]["terminal_evidence"][
        "target_game"
    ] == "cn04"
    assert honest_null["arc_live_reinduction_levelup"]["terminal_evidence"][
        "target_level"
    ] == "L4"
    assert honest_null["hardware_speedup_claim"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5441_missing_or_dirty_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5441-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    root = _make_repo(
        tmp_path / "missing",
        capstone=None,
        gap=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5441-5452",
        task_ids=mod.EXPECTED_TASK_IDS[:-1],
    )
    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-08",
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["closed_lanes"] == []
    assert artifact["partial_lanes"] == []
    assert artifact["blocked_lanes"] == []
    assert artifact["honest_null_lanes"] == []
    assert artifact["roadmap_yaml_unchanged"] is False
    assert artifact["conductor_unchanged"] is False
    for failure in (
        "capstone_missing_or_unloadable",
        "gap_table_missing_or_unloadable",
        "roadmap_milestone_expected_2026.07.495_observed_2026.07.494",
        "roadmap_doc_missing_or_mismatch_2026.07.495",
        "roadmap_doc_task_range_expected_exp5441-exp5453_observed_exp5441-exp5452",
        "roadmap_task_ids_mismatch",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ):
        assert failure in artifact["failed_preconditions"]

    bad_capstone = _capstone_payload()
    bad_capstone["milestone"] = "2026.07.493"
    bad_capstone["status"] = "blocked"
    bad_capstone["honest_verdict"] = "done"
    bad_capstone["hardware_speedup_claim"] = True
    bad_root = _make_repo(tmp_path / "bad-capstone", capstone=bad_capstone, gap=_gap_payload())
    bad_artifact = mod.build_artifact(
        root=bad_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(bad_artifact)
    assert bad_artifact["status"] == "blocked"
    for failure in (
        "capstone_milestone_expected_2026.07.494_observed_2026.07.493",
        "capstone_status_expected_complete_observed_blocked",
        "capstone_honest_verdict_missing_terminal_prefix",
        "capstone_hardware_speedup_claim_expected_false",
    ):
        assert failure in bad_artifact["failed_preconditions"]

    incomplete_capstone = _capstone_payload()
    incomplete_capstone["headline_ready_lanes"] = incomplete_capstone[
        "headline_ready_lanes"
    ][:-1]
    incomplete_capstone["bounded_lanes"] = incomplete_capstone["bounded_lanes"][:-1]
    incomplete_capstone["blocked_lanes"] = []
    incomplete_root = _make_repo(
        tmp_path / "incomplete-lanes",
        capstone=incomplete_capstone,
        gap=_gap_payload(),
    )
    incomplete_artifact = mod.build_artifact(
        root=incomplete_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(incomplete_artifact)
    for failure in (
        "capstone_closed_lanes_incomplete",
        "capstone_partial_lanes_incomplete",
        "capstone_blocked_lanes_incomplete",
    ):
        assert failure in incomplete_artifact["failed_preconditions"]


def test_req_report_5441_run_writes_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5441: run writes a deterministic transition receipt."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload(), gap=_gap_payload())
    result_path = tmp_path / "out" / "transition.json"

    written = mod.run(
        root=root,
        result_path=result_path,
        run_date="2026-07-08",
        tests_run=[{"command": "unit 5441", "outcome": "passed"}],
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert written == result_path
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"


def test_req_report_5441_committed_result_matches_replay() -> None:
    """REQ-REPORT-5441: checked-in deliverable is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["previous_task_range"] == "exp5428-exp5440"
    assert result["next_task_range"] == "exp5441-exp5453"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True


def test_req_report_5441_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5441: validation rejects malformed transition receipts."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload(), gap=_gap_payload())
    artifact = mod.build_artifact(
        root=root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    for field in mod.REQUIRED_FIELDS:
        missing = deepcopy(artifact)
        missing.pop(field)
        with pytest.raises(ValueError, match="missing required"):
            mod.validate_artifact(missing)
        break

    mutations = [
        ("schema", "wrong", "schema"),
        ("field_principles", {}, "field_principles"),
        ("status", "done", "status"),
        ("milestone", mod.PREVIOUS_MILESTONE, "milestone"),
        ("previous_milestone", mod.MILESTONE, "previous_milestone"),
        ("prior_capstone_path", "wrong.json", "prior_capstone_path"),
        ("previous_task_range", "exp5428-exp5439", "previous_task_range"),
        ("next_task_range", "exp5441-exp5452", "next_task_range"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("roadmap_yaml_unchanged", "true", "roadmap_yaml_unchanged"),
        ("conductor_unchanged", "true", "conductor_unchanged"),
        ("honest_verdict", "done", "honest_verdict"),
        ("roadmap_task_ids", ["wrong"], "roadmap_task_ids"),
        ("closed_lanes", "bad", "closed_lanes"),
        ("partial_lanes", "bad", "partial_lanes"),
        ("blocked_lanes", "bad", "blocked_lanes"),
        ("honest_null_lanes", "bad", "honest_null_lanes"),
        ("closed_lanes", [], "closed_lanes"),
        ("partial_lanes", [], "partial_lanes"),
        ("blocked_lanes", [], "blocked_lanes"),
        ("honest_null_lanes", [], "honest_null_lanes"),
        ("roadmap_yaml_unchanged", False, "roadmap_yaml_unchanged must be true"),
        ("conductor_unchanged", False, "conductor_unchanged must be true"),
        ("failed_preconditions", "bad", "failed_preconditions"),
        ("failed_preconditions", ["bad"], "complete status"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ]
    for field, value, message in mutations:
        mutated = deepcopy(artifact)
        mutated[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(mutated)

    blocked = mod.build_artifact(root=_make_repo(tmp_path / "blocked"))
    blocked["failed_preconditions"] = []
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked status"):
        mod.validate_artifact(blocked)

    assert mod._records_by_lane("bad") == {}
