"""Tests for Exp5402 .492 transition receipt.

Spec refs: REQ-REPORT-5402, SCENARIO-REPORT-5402,
SCENARIO-REPORT-5402-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5402_transition_v492 as mod


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
            "prompt": "REQ-REPORT-5402 fixture",
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


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5402-5414") -> str:
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
            "complete: .491 capstone emitted from actual artifacts; headline-ready "
            "bounded lanes are structured scale-up, overwrite corrigendum, p-bit CPU "
            "ablation, CSL router, memory guard, and KAN certificate; Exp5392 flagged, "
            "Exp5397 no-bank, token/internal lane closed, hardware repeatability absent, "
            "and no hardware speedup."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "headline_ready_lanes": [
            "structured_constraint_tax_scaleup",
            "overwrite_guidance_corrigendum",
            "pbit_boundary_ablation",
            "continuous_self_learning_router",
            "raw_episode_memory_guard",
            "kan_dynamic_certificate",
        ],
        "flagged_artifacts": [
            {
                "path": "results/experiment_5392_formal_encoding_safety_fixture_v491.json",
                "task_id": "exp5392-v491-formal-encoding-safety-fixture",
                "reasons": [
                    "artifact flagged_adversarial=true",
                    "conductor log status FLAGGED",
                    "critical TAUTOLOGY corrigendum pending",
                ],
                "headline_eligible": False,
            }
        ],
        "conductor_flags": [
            {
                "path": "results/experiment_5392_formal_encoding_safety_fixture_v491.json",
                "task_id": "exp5392-v491-formal-encoding-safety-fixture",
                "status": "FLAGGED",
                "log_excerpt": "adversarial_verify CRITICAL: TAUTOLOGY",
            }
        ],
        "truth_table": [
            {
                "lane": "structured_constraint_tax_scaleup",
                "source_artifact": "results/experiment_5391_constraint_tax_scaleup_fixtures_v491.json",
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "bounded_deterministic_fixture_panel",
                "evidence": {
                    "fixture_count": 24,
                    "constrained_semantic_validity_rate": 1.0,
                    "unsafe_false_accept_count": 0,
                },
            },
            {
                "lane": "formal_encoding_safety_fixture",
                "source_artifact": "results/experiment_5392_formal_encoding_safety_fixture_v491.json",
                "classification": "blocked",
                "headline_ready": False,
                "blocked_reason": "flagged_adversarial_tautology",
                "claim_boundary": "safe_fixture_not_clean_headline_while_flagged",
                "evidence": {
                    "flagged_adversarial": True,
                    "corrigendum_pending_count": 2,
                },
            },
            {
                "lane": "overwrite_guidance_corrigendum",
                "source_artifact": (
                    "results/experiment_5393_overwrite_guidance_tautology_corrigendum_v491.json"
                ),
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "row_level_solver_authority_only",
                "evidence": {
                    "row_level_evidence_clean": True,
                    "tautology_checks_passed": True,
                    "unsafe_false_accept_count": 0,
                },
            },
            {
                "lane": "pbit_boundary_ablation",
                "source_artifact": "results/experiment_5394_gated_overwrite_pbit_ablation_v491.json",
                "classification": "bounded_ready",
                "headline_ready": True,
                "claim_boundary": "cpu_only_no_hardware_speedup",
                "evidence": {
                    "simulation_only": True,
                    "hardware_speedup_claim": False,
                    "unsafe_false_accepts": 0,
                },
            },
            {
                "lane": "continuous_self_learning_router",
                "source_artifact": (
                    "results/experiment_5395_influence_share_verifier_budget_router_v491.json"
                ),
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "controller_routing_no_weight_mutation",
                "evidence": {
                    "quality_delta_vs_baseline": 0.0,
                    "verifier_cost_delta_vs_baseline": 22.2,
                    "no_weight_mutation": True,
                },
            },
            {
                "lane": "raw_episode_memory_guard",
                "source_artifact": (
                    "results/experiment_5396_memory_guard_raw_episode_retention_v491.json"
                ),
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "raw_episode_retention_no_rationale_authority",
                "evidence": {
                    "raw_episode_count": 7,
                    "forged_reasoning_deflection_rate": 1.0,
                    "no_weight_mutation": True,
                },
            },
            {
                "lane": "arc_level_up",
                "source_artifact": "results/experiment_5397_arc_blob_salience_live_path_v491.json",
                "classification": "honest_null",
                "headline_ready": False,
                "blocked_reason": "bounded_budget_no_levelup",
                "claim_boundary": "live_path_reached_no_new_banked_level",
                "evidence": {
                    "new_level_banked": False,
                    "failure_mode": "bounded_budget_no_levelup",
                    "solve_provenance": "live_agent_self_discovery",
                },
            },
            {
                "lane": "hardware_repeatability",
                "source_artifact": [
                    "results/experiment_5398_hardware_evidence_graph_repeatability_v491.json",
                    "results/experiment_5398_hardware_evidence_graph_repeatability_v491.graph.json",
                ],
                "classification": "blocked",
                "headline_ready": False,
                "blocked_reason": "no_repeated_board_local_timing",
                "claim_boundary": "hash_graph_receipt_no_board_local_repeatability",
                "evidence": {
                    "repeatability_evidence_present": False,
                    "polar_fire_repeat_count": 0,
                    "hardware_speedup_claim": False,
                },
            },
            {
                "lane": "kan_dynamic_certificate",
                "source_artifact": (
                    "results/experiment_5399_kan_dynamic_counterexample_certificate_v491.json"
                ),
                "classification": "headline_ready",
                "headline_ready": True,
                "claim_boundary": "bounded_certificate_no_broad_kan_verification",
                "evidence": {
                    "false_property_rejection_rate": 1.0,
                    "true_property_preservation_rate": 1.0,
                    "broad_kan_verification_claim": False,
                },
            },
        ],
        "retired_or_blocked_lanes": [
            {
                "lane": "future_token_internal_signal",
                "source_artifact": "results/experiment_5389_transition_v491.json",
                "state": "retired_until_backend_feature_artifact",
                "next_gate": "backend artifact with logits, hidden states, attention, or intermediate exits",
            }
        ],
        "future_token_signal_allowed": False,
    }


def _make_repo(
    root: Path,
    *,
    capstone: dict[str, Any] | None = None,
    milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    doc_task_range: str = "Exp 5402-5414",
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
        "| 2026-07-08 04:40 UTC | Exp5392 | FLAGGED | TAUTOLOGY |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5402_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5402: OpenSpec anchors the .492 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5402") : spec.index("REQ-REPORT-5400")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5402",
        "SCENARIO-REPORT-5402",
        "SCENARIO-REPORT-5402-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "exp5402-exp5414",
        "Exp5392 formal-encoding `CRITICAL TAUTOLOGY`",
        "Exp5397 ARC no-bank",
        "Exp5398 hardware repeatability absent",
        "token/internal-feature lane closed",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5402_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5402: active .492 records .491 boundaries and .492 gates."""

    root = _make_repo(tmp_path, capstone=_capstone_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-08",
        tests_run=[{"command": "unit 5402", "outcome": "passed"}],
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
    assert artifact["next_task_range"] == mod.NEXT_TASK_RANGE
    assert artifact["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    closed = {row["lane"]: row for row in artifact["closed_lanes"]}
    assert list(closed) == [
        "structured_scaleup",
        "overwrite_corrigendum",
        "pbit_cpu_ablation",
        "continuous_self_learning_router",
        "memory_guard",
        "bounded_kan_certificate",
    ]
    assert closed["structured_scaleup"]["source_lane"] == "structured_constraint_tax_scaleup"
    assert closed["structured_scaleup"]["terminal_evidence"]["fixture_count"] == 24
    assert closed["pbit_cpu_ablation"]["terminal_evidence"]["hardware_speedup_claim"] is False
    assert closed["bounded_kan_certificate"]["claim_boundary"] == (
        "bounded_certificate_no_broad_kan_verification"
    )

    open_lanes = {row["lane"]: row for row in artifact["open_lanes"]}
    assert list(open_lanes) == [
        "formal_encoding_tautology_flag",
        "arc_no_bank",
        "hardware_repeatability_absent",
        "token_internal_lane_closed",
    ]
    assert open_lanes["formal_encoding_tautology_flag"]["state"] == (
        "blocked_flagged_adversarial_critical_tautology"
    )
    assert open_lanes["arc_no_bank"]["terminal_evidence"]["new_level_banked"] is False
    assert open_lanes["hardware_repeatability_absent"]["terminal_evidence"][
        "repeatability_evidence_present"
    ] is False
    assert open_lanes["token_internal_lane_closed"]["state"] == (
        "retired_until_backend_feature_artifact"
    )
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5402_missing_or_dirty_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5402-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    root = _make_repo(
        tmp_path,
        capstone=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5402-5413",
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
    assert artifact["open_lanes"] == []
    assert artifact["roadmap_yaml_unchanged"] is False
    assert artifact["conductor_unchanged"] is False
    assert "capstone_missing_or_unloadable" in artifact["failed_preconditions"]
    assert "roadmap_milestone_expected_2026.07.492_observed_2026.07.491" in artifact[
        "failed_preconditions"
    ]
    assert "roadmap_doc_task_range_expected_exp5402-exp5414_observed_exp5402-exp5413" in (
        artifact["failed_preconditions"]
    )
    assert "roadmap_task_ids_mismatch" in artifact["failed_preconditions"]
    assert "research-roadmap.yaml_modified" in artifact["failed_preconditions"]
    assert "scripts/research_conductor.py_modified" in artifact["failed_preconditions"]


def test_req_report_5402_committed_result_matches_replay() -> None:
    """REQ-REPORT-5402: checked-in deliverable is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["next_task_range"] == "exp5402-exp5414"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True


def test_req_report_5402_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5402: validation rejects malformed transition receipts."""

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
        ("next_task_range", "exp5402-exp5413", "next_task_range"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("roadmap_yaml_unchanged", "true", "roadmap_yaml_unchanged"),
        ("conductor_unchanged", "true", "conductor_unchanged"),
        ("honest_verdict", "done", "honest_verdict"),
        ("roadmap_task_ids", ["wrong"], "roadmap_task_ids"),
        ("closed_lanes", [], "closed_lanes"),
        ("open_lanes", [], "open_lanes"),
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

    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / mod.ROADMAP_RELATIVE_PATH).write_text("milestone: 2026.07.492\n", encoding="utf-8")
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
