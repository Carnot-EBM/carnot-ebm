"""Tests for Exp5454 .496 transition receipt.

Spec refs: REQ-REPORT-5454, SCENARIO-REPORT-5454,
SCENARIO-REPORT-5454-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5454_transition_v496 as mod


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
            "prompt": "REQ-REPORT-5454 fixture",
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


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5454-5467") -> str:
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
        "terminal_evidence": evidence,
        "source_artifacts": [f"results/{lane}.json"],
    }
    if blocked_reason:
        row["blocked_reason"] = blocked_reason
    return row


def _capstone_payload() -> dict[str, Any]:
    headline = [
        _lane(
            "verifier_potential_generation",
            "headline_ready",
            {
                "verifier_potential_fixture_ready": True,
                "exact_final_authority": True,
                "fixture_count": 8,
                "metric_independence_checks_passed": True,
            },
            claim_boundary="deterministic verifier-potential fixtures",
        ),
        _lane(
            "ast_kb_witnesses",
            "headline_ready",
            {
                "ast_kb_witness_ready": True,
                "fixture_count": 10,
                "valid_call_accept_rate": 1.0,
                "nonexistent_call_reject_rate": 1.0,
            },
            claim_boundary="deterministic AST/KB witness rows",
        ),
        _lane(
            "governed_csl",
            "headline_ready",
            {
                "governed_csl_loop_ready": True,
                "continuous_self_learning_task": True,
                "no_weight_mutation": True,
                "negative_transfer_deflection_rate": 1.0,
            },
            claim_boundary="governed sidecar memory promotion",
        ),
        _lane(
            "memory_stress",
            "headline_ready",
            {
                "csl_memory_stress_ready": True,
                "memory_failure_case_count": 8,
                "rollback_recovery_rate": 1.0,
                "no_weight_mutation": True,
            },
            claim_boundary="governed memory-failure stress",
        ),
        _lane(
            "prd_gap_synthesis",
            "headline_ready",
            {
                "prd_gap_table_ready": True,
                "closed_count": 6,
                "partial_count": 4,
                "blocked_count": 1,
                "honest_null_count": 2,
                "missing_count": 0,
            },
            claim_boundary="PRD gap table synthesis",
        ),
    ]
    bounded = [
        _lane(
            "active_constraint_pbit_bridge",
            "bounded",
            {
                "pbit_assumption_bridge_ready": True,
                "solver_authoritative": True,
                "fallback_completeness_rate": 1.0,
                "hardware_speedup_claim": False,
            },
        ),
        _lane(
            "hardware_receipts",
            "bounded",
            {
                "hardware_receipts_ready": True,
                "hashes_match_before_timing_compare": True,
                "hardware_speedup_claim": False,
                "timing_repeat_counts": {"cpu": 10, "polarfire": 10},
            },
        ),
        _lane(
            "kan_certificates",
            "bounded",
            {
                "kan_certificate_ready": True,
                "hardware_speedup_claim_rejected": True,
                "token_internal_claim_rejected": True,
                "broad_kan_claim_made": False,
            },
        ),
    ]
    blocked = [
        _lane(
            "local_sota_decoding",
            "blocked",
            {
                "flagged_adversarial": True,
                "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
                "verifier_guided_decoding_ready": True,
            },
            blocked_reason="flagged_adversarial_and_tautology",
            claim_boundary="local GGUF decoding pilot is blocked from headline",
        ),
        _lane(
            "token_internal_access",
            "blocked",
            {
                "backend_receipt_present": False,
                "token_internal_claim_rejected": True,
                "token_internal_lane_reopened": False,
            },
            blocked_reason="no_authenticated_backend_receipt",
            claim_boundary="closed without authenticated backend receipts",
        ),
    ]
    honest_null = [
        _lane(
            "arc_live_progress",
            "honest_null",
            {
                "arc_new_level_banked": False,
                "selected_game": "ka59",
                "selected_target_level_label": "L2",
                "new_levels_banked": 0,
                "residual_wall": "bounded_budget_no_levelup",
            },
            blocked_reason="bounded_budget_no_levelup",
            claim_boundary="live ARC path ran; no new level was banked",
        ),
        _lane(
            "hardware_speedup_claim",
            "honest_null",
            {
                "hardware_speedup_claim": False,
                "hardware_speedup_claim_rejected": True,
            },
            blocked_reason="no_authenticated_hardware_speedup",
            claim_boundary="hardware receipts do not support speedup",
        ),
    ]
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "status": "complete",
        "honest_verdict": (
            "complete: .495 capstone emitted from actual artifacts; verifier-potential "
            "fixtures, AST/KB witnesses, governed CSL, memory stress, and PRD gap "
            "synthesis are headline-ready; local SOTA decoding is blocked by "
            "adversarial tautology flags; active/p-bit, hardware receipts, and KAN "
            "certificates are bounded; ARC no-bank, no hardware speedup, and "
            "token/internal lane closed."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "task_range": mod.PREVIOUS_TASK_RANGE,
        "hardware_speedup_claim": False,
        "arc_new_level_banked": False,
        "roadmap_yaml_unchanged": True,
        "conductor_unchanged": True,
        "headline_ready_lanes": headline,
        "bounded_lanes": bounded,
        "blocked_lanes": blocked,
        "honest_null_lanes": honest_null,
    }


def _gap_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: .495 PRD gap table read actual Exp5441-Exp5451 artifacts.",
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
    doc_task_range: str = "Exp 5454-5467",
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
        "| 2026-07-09 00:00 UTC | Exp5453 | OK | capstone complete |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    if gap is not None:
        _write_json(root / mod.GAP_RELATIVE_PATH, gap)
    return root


def test_req_report_5454_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5454: OpenSpec anchors the .496 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5454") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5454",
        "SCENARIO-REPORT-5454",
        "SCENARIO-REPORT-5454-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "exp5441-exp5453",
        "exp5454-exp5467",
        "verifier-potential fixtures",
        "AST/KB witnesses",
        "governed CSL",
        "memory stress",
        "PRD gap synthesis",
        "Exp5444 adversarial tautology",
        "ARC `ka59` L2 no-bank",
        "absence of a hardware speedup claim",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5454_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5454: active .496 records .495 boundaries and gates."""

    root = _make_repo(tmp_path, capstone=_capstone_payload(), gap=_gap_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-09",
        tests_run=[{"command": "unit 5454", "outcome": "passed"}],
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
        "verifier_potential_generation",
        "ast_kb_witnesses",
        "governed_csl",
        "memory_stress",
        "prd_gap_synthesis",
    ]
    assert closed["verifier_potential_generation"]["terminal_evidence"][
        "exact_final_authority"
    ] is True
    assert closed["ast_kb_witnesses"]["terminal_evidence"]["fixture_count"] == 10
    assert closed["governed_csl"]["terminal_evidence"]["no_weight_mutation"] is True
    assert closed["memory_stress"]["terminal_evidence"]["memory_failure_case_count"] == 8
    assert closed["prd_gap_synthesis"]["terminal_evidence"]["closed_count"] == 6

    partial = {row["lane"]: row for row in artifact["partial_lanes"]}
    assert list(partial) == [
        "active_constraint_pbit_bridge",
        "hardware_receipts",
        "kan_certificates",
    ]
    assert partial["active_constraint_pbit_bridge"]["terminal_evidence"][
        "solver_authoritative"
    ] is True
    assert partial["hardware_receipts"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert partial["kan_certificates"]["terminal_evidence"][
        "broad_kan_claim_made"
    ] is False

    blocked = {row["lane"]: row for row in artifact["blocked_lanes"]}
    assert list(blocked) == ["local_sota_decoding", "token_internal_access"]
    assert blocked["local_sota_decoding"]["blocked_reason"] == "flagged_adversarial_and_tautology"
    assert blocked["local_sota_decoding"]["terminal_evidence"]["flagged_adversarial"] is True
    assert blocked["token_internal_access"]["terminal_evidence"][
        "token_internal_lane_reopened"
    ] is False

    honest_null = {row["lane"]: row for row in artifact["honest_null_lanes"]}
    assert list(honest_null) == ["arc_live_progress", "hardware_speedup_claim"]
    assert honest_null["arc_live_progress"]["terminal_evidence"]["selected_game"] == "ka59"
    assert honest_null["arc_live_progress"]["terminal_evidence"][
        "selected_target_level_label"
    ] == "L2"
    assert honest_null["hardware_speedup_claim"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5454_missing_or_dirty_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5454-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    root = _make_repo(
        tmp_path / "missing",
        capstone=None,
        gap=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5454-5466",
        task_ids=mod.EXPECTED_TASK_IDS[:-1],
    )
    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-09",
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
        "roadmap_milestone_expected_2026.07.496_observed_2026.07.495",
        "roadmap_doc_missing_or_mismatch_2026.07.496",
        "roadmap_doc_task_range_expected_exp5454-exp5467_observed_exp5454-exp5466",
        "roadmap_task_ids_mismatch",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ):
        assert failure in artifact["failed_preconditions"]

    bad_capstone = _capstone_payload()
    bad_capstone["milestone"] = "2026.07.494"
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
        "capstone_milestone_expected_2026.07.495_observed_2026.07.494",
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
    incomplete_capstone["honest_null_lanes"] = incomplete_capstone[
        "honest_null_lanes"
    ][:-1]
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
        "capstone_honest_null_lanes_incomplete",
    ):
        assert failure in incomplete_artifact["failed_preconditions"]


def test_req_report_5454_run_writes_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5454: run writes a deterministic transition receipt."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload(), gap=_gap_payload())
    result_path = tmp_path / "out" / "transition.json"

    written = mod.run(
        root=root,
        result_path=result_path,
        run_date="2026-07-09",
        tests_run=[{"command": "unit 5454", "outcome": "passed"}],
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert written == result_path
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"


def test_req_report_5454_committed_result_matches_replay() -> None:
    """REQ-REPORT-5454: checked-in deliverable is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["previous_task_range"] == "exp5441-exp5453"
    assert result["next_task_range"] == "exp5454-exp5467"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True


def test_req_report_5454_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5454: validation rejects malformed transition receipts."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload(), gap=_gap_payload())
    artifact = mod.build_artifact(
        root=root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    missing = deepcopy(artifact)
    missing.pop("milestone")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    mutations = [
        ("schema", "wrong", "schema"),
        ("field_principles", {}, "field_principles"),
        ("status", "done", "status"),
        ("milestone", mod.PREVIOUS_MILESTONE, "milestone"),
        ("previous_milestone", mod.MILESTONE, "previous_milestone"),
        ("prior_capstone_path", "wrong.json", "prior_capstone_path"),
        ("previous_task_range", "exp5441-exp5452", "previous_task_range"),
        ("next_task_range", "exp5454-exp5466", "next_task_range"),
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
    assert mod._terminal_evidence({"evidence": {"fallback": True}}) == {"fallback": True}
    assert mod._terminal_evidence({}) == {}
    assert mod._source_artifacts({}) == []
    assert mod._source_artifacts({"source_artifact": "results/source.json"}) == [
        "results/source.json"
    ]
