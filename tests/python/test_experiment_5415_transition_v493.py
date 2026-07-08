"""Tests for Exp5415 .493 transition receipt.

Spec refs: REQ-REPORT-5415, SCENARIO-REPORT-5415,
SCENARIO-REPORT-5415-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5415_transition_v493 as mod


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
            "prompt": "REQ-REPORT-5415 fixture",
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


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5415-5427") -> str:
    return f"""# Research Roadmap vNEXT - Milestone {milestone}

**Milestone title:** fixture
**Previous milestone:** {mod.PREVIOUS_MILESTONE}
**Task range:** {task_range}
**Pre-staged roadmap:** `research-roadmap-next.yaml`
"""


def _capstone_payload() -> dict[str, Any]:
    return {
        "status": "complete",
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": (
            "complete: .492 capstone emitted from actual artifacts; formal corrigendum, "
            "structured safety/action, resource-accounted CSL, and uncertainty-gated "
            "promotion are headline-ready; active-constraint, p-bit/QUBO, KAN, and "
            "local SOTA inference remain bounded; ARC no-bank, hardware has "
            "repeatability but no hardware speedup, and token/internal lane closed."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "headline_ready_lanes": [
            "formal_encoding_corrigendum",
            "structured_safety_action_panel",
            "resource_accounted_csl",
            "uncertainty_gated_promotion",
        ],
        "truth_table": [
            {
                "lane": "formal_corrigendum",
                "source_artifacts": [
                    "results/experiment_5404_formal_encoding_corrigendum_v492.json"
                ],
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "row_level_formal_encoding_safety_only",
                "evidence": {
                    "formal_encoding_corrigendum_clean": True,
                    "fixture_count": 18,
                    "gpu_offload_verified": True,
                },
            },
            {
                "lane": "structured_safety_action_scaleup",
                "source_artifacts": [
                    "results/experiment_5405_structured_safety_action_panel_v492.json"
                ],
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "structured_fixture_panel_not_general_sota_quality",
                "evidence": {
                    "structured_safety_action_panel_ready": True,
                    "fixture_count": 42,
                    "unsafe_false_accept_rate": 0.0,
                },
            },
            {
                "lane": "active_constraint_guidance",
                "source_artifacts": [
                    "results/experiment_5406_active_constraint_warmstart_guidance_v492.json"
                ],
                "classification": "bounded_ready",
                "headline_ready": False,
                "claim_boundary": "advisory_hints_solver_authority_preserved",
                "evidence": {
                    "active_constraint_warmstart_ready": True,
                    "solver_iteration_delta": 21,
                    "stale_hint_rejection_rate": 1.0,
                },
            },
            {
                "lane": "pbit_qubo_stress",
                "source_artifacts": [
                    "results/experiment_5407_pbit_qubo_active_constraint_stress_v492.json"
                ],
                "classification": "bounded_ready",
                "headline_ready": False,
                "claim_boundary": "cpu_only_no_hardware_speedup",
                "evidence": {
                    "pbit_qubo_stress_ready": True,
                    "exact_enumeration_agreement_rate": 1.0,
                    "simulation_only": True,
                    "hardware_speedup_claim": False,
                },
            },
            {
                "lane": "resource_accounted_csl",
                "source_artifacts": [
                    "results/experiment_5408_resource_accounted_csl_controller_v492.json"
                ],
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "controller_routing_no_weight_mutation",
                "evidence": {
                    "resource_accounted_csl_ready": True,
                    "no_weight_mutation": True,
                    "decision_count": 36,
                },
            },
            {
                "lane": "uncertainty_gated_promotion",
                "source_artifacts": [
                    "results/experiment_5409_uncertainty_gated_promotion_v492.json"
                ],
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "uncertainty_gate_no_ungated_memory_promotion",
                "evidence": {
                    "uncertainty_gated_promotion_ready": True,
                    "accepted_promotion_count": 3,
                    "rollback_success_rate": 1.0,
                },
            },
            {
                "lane": "arc_live_levelup",
                "source_artifacts": [
                    "results/experiment_5410_arc_live_trajectory_frontier_levelup_v492.json"
                ],
                "classification": "honest_null",
                "headline_ready": False,
                "blocked_reason": "bounded_budget_no_levelup",
                "claim_boundary": "live_agent_path_exercised_no_new_banked_level",
                "evidence": {
                    "arc_new_level_banked": False,
                    "attempt_count": 35,
                    "status": "honest_null",
                },
            },
            {
                "lane": "hardware_repeatability",
                "source_artifacts": [
                    "results/experiment_5411_hardware_repeatability_restoration_v492.json"
                ],
                "classification": "partial",
                "headline_ready": False,
                "claim_boundary": "repeatability_receipt_not_speedup_or_multi_board_ready",
                "evidence": {
                    "repeated_same_workload_ready": True,
                    "polarfire_repeat_count": 3,
                    "hardware_speedup_claim": False,
                    "kv260_ssh_reachable": False,
                    "gatemate_reachable": False,
                },
            },
            {
                "lane": "kan_active_constraint_certificate",
                "source_artifacts": [
                    "results/experiment_5412_kan_active_constraint_certificate_v492.json"
                ],
                "classification": "bounded_ready",
                "headline_ready": False,
                "claim_boundary": "bounded_certificate_no_broad_kan_verification",
                "evidence": {
                    "kan_active_constraint_certificate_ready": True,
                    "false_property_rejection_rate": 1.0,
                    "broad_kan_verification_claim": False,
                },
            },
            {
                "lane": "token_internal_lane",
                "source_artifacts": [
                    "results/experiment_5402_transition_v492.json",
                    "results/experiment_5413_evidence_table_prd_gap_analysis_v492.json",
                ],
                "classification": "blocked",
                "headline_ready": False,
                "blocked_reason": "no_logits_hidden_states_attention_or_intermediate_exit_receipt",
                "claim_boundary": "closed_without_backend_feature_receipt",
                "evidence": {
                    "backend_receipt_present": False,
                    "future_token_signal_allowed": False,
                },
            },
        ],
    }


def _make_repo(
    root: Path,
    *,
    capstone: dict[str, Any] | None = None,
    milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    doc_task_range: str = "Exp 5415-5427",
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
        "| 2026-07-08 04:40 UTC | Exp5414 | OK | capstone complete |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5415_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5415: OpenSpec anchors the .493 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5415") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5415",
        "SCENARIO-REPORT-5415",
        "SCENARIO-REPORT-5415-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "exp5402-exp5414",
        "exp5415-exp5427",
        "formal-encoding corrigendum",
        "p-bit/QUBO CPU stress",
        "Exp5410 ARC no-bank",
        "KV260/GateMate availability limits",
        "token/internal feature lanes still closed",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5415_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5415: active .493 records .492 boundaries and gates."""

    root = _make_repo(tmp_path, capstone=_capstone_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-08",
        tests_run=[{"command": "unit 5415", "outcome": "passed"}],
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
        "formal_encoding_corrigendum",
        "structured_safety_action_panel",
        "resource_accounted_csl",
        "uncertainty_gated_promotion",
    ]
    assert closed["formal_encoding_corrigendum"]["source_lane"] == "formal_corrigendum"
    assert closed["structured_safety_action_panel"]["terminal_evidence"]["fixture_count"] == 42
    assert closed["resource_accounted_csl"]["terminal_evidence"]["no_weight_mutation"] is True
    assert closed["uncertainty_gated_promotion"]["claim_boundary"] == (
        "uncertainty_gate_no_ungated_memory_promotion"
    )

    partial = {row["lane"]: row for row in artifact["partial_lanes"]}
    assert list(partial) == [
        "active_constraint_guidance",
        "pbit_qubo_cpu_stress",
        "hardware_repeatability_without_speedup",
        "bounded_kan_certificates",
    ]
    assert partial["active_constraint_guidance"]["classification"] == "bounded_ready"
    assert partial["pbit_qubo_cpu_stress"]["terminal_evidence"]["simulation_only"] is True
    assert partial["pbit_qubo_cpu_stress"]["terminal_evidence"]["hardware_speedup_claim"] is False
    assert partial["hardware_repeatability_without_speedup"]["classification"] == "partial"
    assert partial["bounded_kan_certificates"]["terminal_evidence"][
        "broad_kan_verification_claim"
    ] is False

    blocked = {row["lane"]: row for row in artifact["blocked_lanes"]}
    assert list(blocked) == [
        "exp5410_arc_no_bank",
        "kv260_gatemate_availability_limits",
        "token_internal_feature_lane_closed",
    ]
    assert blocked["exp5410_arc_no_bank"]["terminal_evidence"]["arc_new_level_banked"] is False
    assert blocked["kv260_gatemate_availability_limits"]["terminal_evidence"][
        "kv260_ssh_reachable"
    ] is False
    assert blocked["kv260_gatemate_availability_limits"]["terminal_evidence"][
        "gatemate_reachable"
    ] is False
    assert blocked["token_internal_feature_lane_closed"]["terminal_evidence"][
        "future_token_signal_allowed"
    ] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5415_missing_or_dirty_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5415-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    root = _make_repo(
        tmp_path / "missing",
        capstone=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5415-5426",
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
    assert artifact["roadmap_yaml_unchanged"] is False
    assert artifact["conductor_unchanged"] is False
    assert "capstone_missing_or_unloadable" in artifact["failed_preconditions"]
    assert "roadmap_milestone_expected_2026.07.493_observed_2026.07.492" in artifact[
        "failed_preconditions"
    ]
    assert "roadmap_doc_missing_or_mismatch_2026.07.493" in artifact["failed_preconditions"]
    assert "roadmap_doc_task_range_expected_exp5415-exp5427_observed_exp5415-exp5426" in (
        artifact["failed_preconditions"]
    )
    assert "roadmap_task_ids_mismatch" in artifact["failed_preconditions"]
    assert "research-roadmap.yaml_modified" in artifact["failed_preconditions"]
    assert "scripts/research_conductor.py_modified" in artifact["failed_preconditions"]

    bad_capstone = _capstone_payload()
    bad_capstone["milestone"] = "2026.07.491"
    bad_capstone["status"] = "blocked"
    bad_capstone["honest_verdict"] = "done"
    bad_root = _make_repo(tmp_path / "bad-capstone", capstone=bad_capstone)
    bad_artifact = mod.build_artifact(
        root=bad_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(bad_artifact)
    assert bad_artifact["status"] == "blocked"
    assert "capstone_milestone_expected_2026.07.492_observed_2026.07.491" in (
        bad_artifact["failed_preconditions"]
    )
    assert "capstone_status_expected_complete_observed_blocked" in (
        bad_artifact["failed_preconditions"]
    )
    assert "capstone_honest_verdict_missing_terminal_prefix" in (
        bad_artifact["failed_preconditions"]
    )


def test_req_report_5415_committed_result_matches_replay() -> None:
    """REQ-REPORT-5415: checked-in deliverable is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["previous_task_range"] == "exp5402-exp5414"
    assert result["next_task_range"] == "exp5415-exp5427"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True


def test_req_report_5415_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5415: validation rejects malformed transition receipts."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload())
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
        ("previous_task_range", "exp5402-exp5413", "previous_task_range"),
        ("next_task_range", "exp5415-exp5426", "next_task_range"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("roadmap_yaml_unchanged", "true", "roadmap_yaml_unchanged"),
        ("conductor_unchanged", "true", "conductor_unchanged"),
        ("honest_verdict", "done", "honest_verdict"),
        ("roadmap_task_ids", ["wrong"], "roadmap_task_ids"),
        ("closed_lanes", "bad", "closed_lanes"),
        ("partial_lanes", "bad", "partial_lanes"),
        ("blocked_lanes", "bad", "blocked_lanes"),
        ("closed_lanes", [], "closed_lanes"),
        ("partial_lanes", [], "partial_lanes"),
        ("blocked_lanes", [], "blocked_lanes"),
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

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["error"] == "malformed_json"
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(array_json)[1]["error"] == "not_json_object"
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("bad: [", encoding="utf-8")
    assert mod.read_yaml_mapping(bad_yaml)[1]["error"] == "malformed_yaml"
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert mod.read_yaml_mapping(list_yaml)[1]["error"] == "not_yaml_object"
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml")[1]["error"] == "missing"

    assert mod.normalize_task_range("Task range missing") is None
    assert mod.extract_roadmap_tasks({"tasks": "bad"}) == []
    assert mod.extract_roadmap_tasks({"tasks": [{"id": "x"}, "bad"]}) == ["x"]
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.git_path_modified(tmp_path, mod.ROADMAP_RELATIVE_PATH) is False
    assert mod._modification_status(
        tmp_path,
        mod.ROADMAP_RELATIVE_PATH,
        {str(mod.ROADMAP_RELATIVE_PATH): True},
    ) is True
    assert mod._evidence({"evidence": "bad"}) == {}
    assert mod._source_artifacts({}) == []
    assert mod._source_artifacts({"source_artifact": "fixture.json"}) == ["fixture.json"]
    assert mod._lane_failures(
        closed_lanes=[],
        partial_lanes=[],
        blocked_lanes=[],
        capstone_loadable=True,
    ) == [
        "capstone_closed_lanes_incomplete",
        "capstone_partial_lanes_incomplete",
        "capstone_blocked_lanes_incomplete",
    ]

    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / mod.ROADMAP_RELATIVE_PATH).write_text("milestone: 2026.07.493\n", encoding="utf-8")
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
