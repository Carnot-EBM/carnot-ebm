"""Tests for Exp5428 .494 transition receipt.

Spec refs: REQ-REPORT-5428, SCENARIO-REPORT-5428,
SCENARIO-REPORT-5428-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5428_transition_v494 as mod


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
            "prompt": "REQ-REPORT-5428 fixture",
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


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5428-5440") -> str:
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
    return {
        "status": "complete",
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": (
            "complete: .493 capstone emitted from actual artifacts; CSL reliance and gated "
            "promotion are headline-ready, active constraints/p-bit timing/KAN and comparable "
            "hardware timing are bounded, structured lanes remain blocked, ARC no-bank keeps "
            "the north-star count unchanged, no hardware speedup is claimed, and token/internal "
            "lane closed."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "hardware_speedup_claim": False,
        "headline_ready_lanes": [
            _lane(
                "evidence_reliance_csl",
                "headline_ready",
                {
                    "evidence_reliance_csl_ready": True,
                    "hidden_forgetting_detected": True,
                    "reliance_drift_metric": 0.8075,
                    "rollback_verified": True,
                    "no_weight_mutation": True,
                },
                claim_boundary="controller-level CSL reliance audit; no model weight mutation",
            ),
            _lane(
                "gated_csl_promotion",
                "headline_ready",
                {
                    "csl_promotion_reliance_scale_ready": True,
                    "promoted_fragment_count": 3,
                    "rejected_fragment_count": 5,
                    "abstained_fragment_count": 3,
                    "rollback_verified": True,
                    "no_weight_mutation": True,
                },
                claim_boundary="gated promotion with rejected and abstained fragments inactive",
            ),
        ],
        "bounded_lanes": [
            _lane(
                "active_constraint_lns_scale",
                "bounded",
                {
                    "active_constraint_lns_scale_ready": True,
                    "accepted_hint_count": 5,
                    "rejected_hint_count": 10,
                    "overwritten_hint_count": 5,
                    "work_delta": 234,
                    "solver_validity_preserved": True,
                },
            ),
            _lane(
                "pbit_hardware_transfer_preflight",
                "bounded",
                {
                    "pbit_transfer_preflight_ready": True,
                    "cpu_repeat_count": 3,
                    "board_repeat_count": 3,
                    "same_workload_hash_match": True,
                    "hardware_speedup_claim": False,
                },
            ),
            _lane(
                "comparable_hardware_timing",
                "bounded",
                {
                    "comparable_timing_receipts_ready": True,
                    "same_workload_hash_match": True,
                    "same_result_hash_match": True,
                    "hardware_speedup_claim": False,
                },
            ),
            _lane(
                "kan_measurement_access_certificates",
                "bounded",
                {
                    "kan_measurement_access_certificate_ready": True,
                    "certificate_count": 14,
                    "false_property_rejection_rate": 1.0,
                    "broad_kan_verification_claim": False,
                },
            ),
        ],
        "blocked_lanes": [
            _lane(
                "risk_calibrated_structured_verification",
                "blocked",
                {
                    "flag_reasons": ["flagged_adversarial", "corrigendum_pending"],
                    "abstention_rate": 0.619048,
                    "semantic_error_rate": 0.619048,
                    "gpu_offload_verified": True,
                },
                blocked_reason="flagged_adversarial_and_corrigendum_pending",
            ),
            _lane(
                "predictive_prefix_action_safety",
                "blocked",
                {
                    "flag_reasons": ["flagged_adversarial", "corrigendum_pending"],
                    "final_only_unreachable_tool_action_rate": 0.47619,
                    "prefix_gated_unreachable_tool_action_rate": 0.0,
                    "gpu_offload_verified": True,
                },
                blocked_reason="flagged_adversarial_and_corrigendum_pending",
            ),
            _lane(
                "token_internal_feature_lane_closed",
                "blocked",
                {"backend_receipt_present": False, "future_token_signal_allowed": False},
                blocked_reason="no_authenticated_backend_receipt",
            ),
        ],
        "honest_null_lanes": [
            _lane(
                "arc_levelup",
                "honest_null",
                {
                    "status": "honest_null",
                    "arc_new_level_banked": False,
                    "attempt_count": 46,
                    "frontier_expansion_count": 22,
                    "landmark_count": 45,
                    "registry_total_before": 69,
                    "registry_total_after": 69,
                    "failure_mode": "bounded_budget_no_levelup",
                },
                blocked_reason="bounded_budget_no_levelup",
                claim_boundary="live ARC path was exercised; no reproduced new level was banked",
            )
        ],
    }


def _gap_payload() -> dict[str, Any]:
    return {
        "status": "complete",
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": "complete: .493 PRD gap table read actual upstream artifacts.",
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
    doc_task_range: str = "Exp 5428-5440",
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
        "| 2026-07-08 15:34 UTC | Exp5427 | OK | capstone complete |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    if gap is not None:
        _write_json(root / mod.GAP_RELATIVE_PATH, gap)
    return root


def test_req_report_5428_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5428: OpenSpec anchors the .494 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5428") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5428",
        "SCENARIO-REPORT-5428",
        "SCENARIO-REPORT-5428-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "exp5415-exp5427",
        "exp5428-exp5440",
        "evidence-reliance CSL",
        "gated CSL promotion",
        "ARC `lf52` L3 no-bank",
        "absence of a hardware speedup claim",
        "token/internal feature lanes still closed",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5428_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5428: active .494 records .493 boundaries and gates."""

    root = _make_repo(tmp_path, capstone=_capstone_payload(), gap=_gap_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-08",
        tests_run=[{"command": "unit 5428", "outcome": "passed"}],
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
    assert list(closed) == ["evidence_reliance_csl", "gated_csl_promotion"]
    assert closed["evidence_reliance_csl"]["terminal_evidence"][
        "hidden_forgetting_detected"
    ] is True
    assert closed["gated_csl_promotion"]["terminal_evidence"]["promoted_fragment_count"] == 3

    partial = {row["lane"]: row for row in artifact["partial_lanes"]}
    assert list(partial) == [
        "active_constraint_lns_scale",
        "pbit_hardware_transfer_preflight",
        "comparable_hardware_timing",
        "kan_measurement_access_certificates",
    ]
    assert partial["active_constraint_lns_scale"]["terminal_evidence"]["work_delta"] == 234
    assert partial["pbit_hardware_transfer_preflight"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert partial["comparable_hardware_timing"]["terminal_evidence"][
        "same_result_hash_match"
    ] is True
    assert partial["kan_measurement_access_certificates"]["terminal_evidence"][
        "broad_kan_verification_claim"
    ] is False

    blocked = {row["lane"]: row for row in artifact["blocked_lanes"]}
    assert list(blocked) == [
        "risk_calibrated_structured_verification",
        "predictive_prefix_action_safety",
        "token_internal_feature_lane_closed",
    ]
    assert blocked["risk_calibrated_structured_verification"]["terminal_evidence"][
        "semantic_error_rate"
    ] == pytest.approx(0.619048)
    assert blocked["predictive_prefix_action_safety"]["terminal_evidence"][
        "final_only_unreachable_tool_action_rate"
    ] == pytest.approx(0.47619)
    assert blocked["token_internal_feature_lane_closed"]["terminal_evidence"][
        "future_token_signal_allowed"
    ] is False

    honest_null = {row["lane"]: row for row in artifact["honest_null_lanes"]}
    assert list(honest_null) == ["arc_levelup", "hardware_speedup_claim"]
    assert honest_null["arc_levelup"]["terminal_evidence"]["attempt_count"] == 46
    assert honest_null["hardware_speedup_claim"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5428_missing_or_dirty_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5428-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    root = _make_repo(
        tmp_path / "missing",
        capstone=None,
        gap=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5428-5439",
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
        "roadmap_milestone_expected_2026.07.494_observed_2026.07.493",
        "roadmap_doc_missing_or_mismatch_2026.07.494",
        "roadmap_doc_task_range_expected_exp5428-exp5440_observed_exp5428-exp5439",
        "roadmap_task_ids_mismatch",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ):
        assert failure in artifact["failed_preconditions"]

    bad_capstone = _capstone_payload()
    bad_capstone["milestone"] = "2026.07.492"
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
        "capstone_milestone_expected_2026.07.493_observed_2026.07.492",
        "capstone_status_expected_complete_observed_blocked",
        "capstone_honest_verdict_missing_terminal_prefix",
        "capstone_hardware_speedup_claim_expected_false",
        "capstone_honest_null_lanes_incomplete",
    ):
        assert failure in bad_artifact["failed_preconditions"]


def test_req_report_5428_committed_result_matches_replay() -> None:
    """REQ-REPORT-5428: checked-in deliverable is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["previous_task_range"] == "exp5415-exp5427"
    assert result["next_task_range"] == "exp5428-exp5440"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True


def test_req_report_5428_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5428: validation rejects malformed transition receipts."""

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
        ("previous_task_range", "exp5415-exp5426", "previous_task_range"),
        ("next_task_range", "exp5428-exp5439", "next_task_range"),
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
    assert mod._records_by_lane([{"lane": "x"}, "bad"]) == {"x": {"lane": "x"}}
    assert mod._lane_record({"evidence": "bad"})["terminal_evidence"] == {}
    assert mod._lane_record({"source_artifact": "fixture.json"})["source_artifacts"] == [
        "fixture.json"
    ]
    assert mod._missing_lane_names([{"lane": "x"}], ["x", "y"]) == ["y"]
    assert mod._lane_failures(
        closed_lanes=[],
        partial_lanes=[],
        blocked_lanes=[],
        honest_null_lanes=[],
        capstone_loadable=True,
    ) == [
        "capstone_closed_lanes_incomplete",
        "capstone_partial_lanes_incomplete",
        "capstone_blocked_lanes_incomplete",
        "capstone_honest_null_lanes_incomplete",
    ]

    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / mod.ROADMAP_RELATIVE_PATH).write_text("milestone: 2026.07.494\n", encoding="utf-8")
    assert mod.git_path_modified(git_repo, mod.ROADMAP_RELATIVE_PATH) is True

    output = mod.run(
        root=root,
        result_path=tmp_path / "written.json",
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
