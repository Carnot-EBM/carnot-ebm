"""Tests for the Exp5482 .498 transition receipt.

Spec refs: REQ-REPORT-5482, SCENARIO-REPORT-5482,
SCENARIO-REPORT-5482-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5482_transition_v498 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _roadmap_yaml(milestone: str, task_ids: list[str] | None = None) -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": milestone,
            "deliverable": f"results/{task_id}.json",
            "title": f"fixture {task_id}",
            "agent_type": "codex",
            "model": "gpt-5.5",
            "prompt": "REQ-REPORT-5482 fixture",
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


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5482-5495") -> str:
    return f"""# Research Roadmap vNEXT - Milestone {milestone}

**Milestone title:** fixture
**Previous milestone:** {mod.PREVIOUS_MILESTONE}
**Task range:** {task_range}
**Pre-staged roadmap:** `research-roadmap-next.yaml`
"""


def _truth_row(
    lane: str,
    classification: str,
    evidence: dict[str, Any],
    *,
    source_artifacts: list[str],
    claim_boundary: str = "fixture boundary",
) -> dict[str, Any]:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": source_artifacts,
        "evidence": evidence,
        "claim_boundary": claim_boundary,
    }


def _capstone_payload() -> dict[str, Any]:
    unreachable_boards = [
        {
            "board_identity": "kv260",
            "blocked_reason": "blocked_kv260_ssh",
            "reachable": False,
            "workload_execution_attempted": False,
        },
        {
            "board_identity": "gatemate",
            "blocked_reason": "diagnostic_only_no_exp5477_workload_receipt",
            "diagnostic_only": True,
            "reachable": False,
            "workload_execution_attempted": False,
        },
    ]
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": (
            "complete: .497 capstone read 13/13 artifacts; "
            "guided_decoding=quarantined; csl=headline_ready; "
            "arc_registry_delta=0; hardware_speedup_claim=False."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "arc_registry_delta": 0,
        "hardware_speedup_claim": False,
        "roadmap_yaml_unchanged": True,
        "conductor_unchanged": True,
        "artifact_paths": [
            "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
            "results/experiment_5471_guard_composition_scale_v497.json",
            "results/experiment_5472_sota_evidence_telemetry_v497.json",
            "results/experiment_5473_csl_kan_surrogate_assurance_v497.json",
            "results/experiment_5474_sota_csl_scale_v497.json",
            "results/experiment_5475_csl_behavioral_memory_ladder_v497.json",
            "results/experiment_5476_helper_lemma_core_witness_repair_v497.json",
            "results/experiment_5477_pdit_lns_boundary_exchange_v497.json",
            "results/experiment_5478_hardware_receipts_v497.json",
            "results/experiment_5480_arc_live_salience_levelup_v497.json",
        ],
        "truth_table": {
            "verifiable_reasoning_guards": _truth_row(
                "verifiable_reasoning_guards",
                "headline_ready",
                {
                    "rewrite_state_fixture_ready": True,
                    "exact_validator_agreement": 1.0,
                    "guard_composition_ready": True,
                    "false_accept_rate": 0.0,
                    "helper_lemma_repair_ready": True,
                    "helper_false_accept_count": 0,
                },
                source_artifacts=[
                    "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
                    "results/experiment_5471_guard_composition_scale_v497.json",
                    "results/experiment_5476_helper_lemma_core_witness_repair_v497.json",
                ],
            ),
            "local_sota_runtime": _truth_row(
                "local_sota_runtime",
                "bounded",
                {
                    "sota_evidence_telemetry_ready": True,
                    "guided_decoding_used": False,
                    "gpu_offload_receipt_count": 1,
                    "exact_validator_accuracy": 0.5,
                },
                source_artifacts=["results/experiment_5472_sota_evidence_telemetry_v497.json"],
            ),
            "csl": _truth_row(
                "csl",
                "headline_ready",
                {
                    "csl_kan_surrogate_ready": True,
                    "csl_scale_ready": True,
                    "csl_behavioral_memory_ready": True,
                    "model_weight_mutation": [False, False, False],
                    "delta_vs_no_memory": 0.75,
                    "delta_vs_naive_icl": 0.5,
                },
                source_artifacts=[
                    "results/experiment_5473_csl_kan_surrogate_assurance_v497.json",
                    "results/experiment_5474_sota_csl_scale_v497.json",
                    "results/experiment_5475_csl_behavioral_memory_ladder_v497.json",
                ],
            ),
            "pdit_lns_boundary_exchange": _truth_row(
                "pdit_lns_boundary_exchange",
                "bounded",
                {
                    "boundary_exchange_ready": True,
                    "exact_fallback_completeness_rate": 1.0,
                    "unsafe_false_accept_count": 0,
                    "hardware_speedup_claim": False,
                },
                source_artifacts=["results/experiment_5477_pdit_lns_boundary_exchange_v497.json"],
            ),
            "hardware_receipts": _truth_row(
                "hardware_receipts",
                "bounded",
                {
                    "hardware_receipts_ready": True,
                    "hardware_speedup_claim": False,
                    "result_hash_match_rate": 1.0,
                    "reachable_boards": ["polarfire"],
                    "unreachable_boards": unreachable_boards,
                },
                source_artifacts=["results/experiment_5478_hardware_receipts_v497.json"],
            ),
            "guided_decoding": _truth_row(
                "guided_decoding",
                "blocked",
                {
                    "rewrite_quarantine_lifted": False,
                    "guard_quarantine_lifted": False,
                    "sota_guided_decoding_used": False,
                },
                source_artifacts=[
                    "results/experiment_5468_transition_v497.json",
                    "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
                    "results/experiment_5471_guard_composition_scale_v497.json",
                    "results/experiment_5472_sota_evidence_telemetry_v497.json",
                ],
                claim_boundary="Guided decoding remains quarantined.",
            ),
            "arc_live_path": _truth_row(
                "arc_live_path",
                "honest_null",
                {
                    "selected_game": "sb26",
                    "selected_target_level": 3,
                    "new_level_banked": False,
                    "offline_reproduced": False,
                    "failure_mode": "bounded_budget_no_target_level_reproduction",
                    "reproduced_levels_before": 2,
                    "reproduced_levels_after": 2,
                },
                source_artifacts=[
                    "results/experiment_5479_arc_target_rotation_precheck_v497.json",
                    "results/experiment_5480_arc_live_salience_levelup_v497.json",
                ],
            ),
            "hardware_speedup_claim": _truth_row(
                "hardware_speedup_claim",
                "honest_null",
                {
                    "hardware_speedup_claim": False,
                    "reachable_boards": ["polarfire"],
                    "unreachable_boards": unreachable_boards,
                },
                source_artifacts=["results/experiment_5478_hardware_receipts_v497.json"],
            ),
        },
    }


def _exp5474_payload() -> dict[str, Any]:
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": "complete: local SOTA GGUF CSL scale-up used KAN assurance",
        "csl_scale_ready": True,
        "delta_vs_naive_icl": 0.5,
        "delta_vs_no_memory": 0.75,
        "exact_validator_pass_rate": 1.0,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "delta_vs_naive_icl=0.5 and naive_icl_score=0.5 agree.",
            }
        ],
    }


def _exp5480_payload() -> dict[str, Any]:
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": "honest_null: sb26 L3 bounded_budget_no_target_level_reproduction",
        "status": "honest_null",
        "game": "sb26",
        "target_level": 3,
        "new_level_banked": False,
        "offline_reproduced": False,
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "reproduced_levels_before": 2,
        "reproduced_levels_after": 2,
        "registry_updated": False,
        "action_count": 47,
        "explored_state_count": 9,
    }


def _make_repo(
    root: Path,
    *,
    capstone: dict[str, Any] | None = None,
    exp5474: dict[str, Any] | None = None,
    exp5480: dict[str, Any] | None = None,
    milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    doc_task_range: str = "Exp 5482-5495",
    task_ids: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(
        _roadmap_yaml(milestone, task_ids),
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
        "| 2026-07-09 12:00 UTC | Exp5474 | FLAGGED TAUTOLOGY |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture conductor\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.PRIOR_CAPSTONE_RELATIVE_PATH, capstone)
    if exp5474 is not None:
        _write_json(root / mod.EXP5474_RELATIVE_PATH, exp5474)
    if exp5480 is not None:
        _write_json(root / mod.EXP5480_RELATIVE_PATH, exp5480)
    return root


def test_req_report_5482_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5482: OpenSpec anchors the V498 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5482") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-5482",
        "SCENARIO-REPORT-5482-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.PRIOR_CAPSTONE_RELATIVE_PATH),
        "exp5468-exp5481",
        "exp5482-exp5495",
        "rewrite-state guards",
        "guard composition",
        "SOTA evidence telemetry",
        "KAN assurance",
        "behavioral memory",
        "helper repair",
        "p-bit/p-dit boundary exchange",
        "hardware receipts",
        "Exp5474",
        "TAUTOLOGY",
        "guided decoding quarantined",
        "ARC `sb26` L3",
        "KV260",
        "GateMate",
        "hardware_speedup_claim=false",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5482_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5482: complete inputs preserve .497 facts for .498."""

    root = _make_repo(
        tmp_path,
        capstone=_capstone_payload(),
        exp5474=_exp5474_payload(),
        exp5480=_exp5480_payload(),
    )
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-09",
        tests_run=[{"command": "unit 5482", "outcome": "passed"}],
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
    assert artifact["prior_capstone_path"] == str(mod.PRIOR_CAPSTONE_RELATIVE_PATH)
    assert artifact["previous_task_range"] == mod.PREVIOUS_TASK_RANGE
    assert artifact["next_task_range"] == mod.NEXT_TASK_RANGE
    assert artifact["roadmap_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["exp5474_tautology_flag_recorded"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    clean = {row["lane"]: row for row in artifact["clean_lanes"]}
    assert list(clean) == [
        "rewrite_state_guards",
        "guard_composition",
        "sota_evidence_telemetry",
        "kan_assurance",
        "behavioral_memory",
        "helper_repair",
        "pbit_pdit_boundary_exchange",
        "hardware_receipts",
    ]
    assert clean["rewrite_state_guards"]["evidence"]["rewrite_state_fixture_ready"] is True
    assert clean["guard_composition"]["evidence"]["guard_composition_ready"] is True
    assert clean["sota_evidence_telemetry"]["evidence"]["guided_decoding_used"] is False
    assert clean["kan_assurance"]["evidence"]["csl_kan_surrogate_ready"] is True
    assert clean["behavioral_memory"]["evidence"]["csl_behavioral_memory_ready"] is True
    assert clean["helper_repair"]["evidence"]["helper_false_accept_count"] == 0
    assert clean["pbit_pdit_boundary_exchange"]["evidence"]["hardware_speedup_claim"] is False
    assert (
        clean["hardware_receipts"]["evidence"]["unreachable_boards"][0]["board_identity"] == "kv260"
    )

    assert {row["lane"] for row in artifact["bounded_lanes"]} == {
        "local_sota_runtime",
        "pdit_lns_boundary_exchange",
        "hardware_receipts",
    }
    blocked = {row["lane"]: row for row in artifact["blocked_lanes"]}
    assert set(blocked) == {"guided_decoding", "kv260_board", "gatemate_board"}
    assert blocked["guided_decoding"]["evidence"]["sota_guided_decoding_used"] is False
    assert blocked["kv260_board"]["evidence"]["blocked_reason"] == "blocked_kv260_ssh"
    assert blocked["gatemate_board"]["evidence"]["reachable"] is False

    honest_null = {row["lane"]: row for row in artifact["honest_null_lanes"]}
    assert set(honest_null) == {"arc_sb26_l3_no_bank", "hardware_speedup_claim"}
    assert honest_null["arc_sb26_l3_no_bank"]["evidence"]["game"] == "sb26"
    assert honest_null["arc_sb26_l3_no_bank"]["evidence"]["target_level"] == 3
    assert honest_null["arc_sb26_l3_no_bank"]["evidence"]["new_level_banked"] is False
    assert honest_null["hardware_speedup_claim"]["evidence"]["hardware_speedup_claim"] is False

    flagged = {row["lane"]: row for row in artifact["flagged_lanes"]}
    assert list(flagged) == ["exp5474_sota_csl_scale_tautology"]
    assert (
        flagged["exp5474_sota_csl_scale_tautology"]["evidence"]["artifact_reported_csl_scale_ready"]
        is True
    )
    assert flagged["exp5474_sota_csl_scale_tautology"]["evidence"]["flag_kind"] == "TAUTOLOGY"
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5482_missing_dirty_or_unflagged_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5482-BLOCKED-INPUT: bad inputs fail closed."""

    root = _make_repo(
        tmp_path / "missing",
        capstone=None,
        exp5474=None,
        exp5480=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5482-5494",
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
    assert artifact["clean_lanes"] == []
    assert artifact["bounded_lanes"] == []
    assert artifact["blocked_lanes"] == []
    assert artifact["honest_null_lanes"] == []
    assert artifact["flagged_lanes"] == []
    assert artifact["exp5474_tautology_flag_recorded"] is False
    assert artifact["roadmap_yaml_unchanged"] is False
    assert artifact["conductor_unchanged"] is False
    for failure in (
        "capstone_missing_or_unloadable",
        "exp5474_missing_or_unloadable",
        "exp5480_missing_or_unloadable",
        "roadmap_milestone_expected_2026.07.498_observed_2026.07.497",
        "roadmap_doc_missing_or_mismatch_2026.07.498",
        "roadmap_doc_task_range_expected_exp5482-exp5495_observed_exp5482-exp5494",
        "roadmap_task_ids_mismatch",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ):
        assert failure in artifact["failed_preconditions"]

    unflagged_exp5474 = _exp5474_payload()
    unflagged_exp5474["flagged_adversarial"] = False
    unflagged_exp5474["corrigendum_pending"] = []
    unflagged_root = _make_repo(
        tmp_path / "unflagged",
        capstone=_capstone_payload(),
        exp5474=unflagged_exp5474,
        exp5480=_exp5480_payload(),
    )
    unflagged = mod.build_artifact(
        root=unflagged_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(unflagged)
    assert unflagged["status"] == "blocked"
    assert "exp5474_tautology_flag_missing" in unflagged["failed_preconditions"]
    assert unflagged["exp5474_tautology_flag_recorded"] is False

    wrong_arc = _exp5480_payload()
    wrong_arc["milestone"] = "2026.07.000"
    wrong_arc["game"] = "bp35"
    wrong_arc["target_level"] = 2
    wrong_arc["new_level_banked"] = True
    wrong_arc["offline_reproduced"] = True
    wrong_arc_root = _make_repo(
        tmp_path / "wrong-arc",
        capstone=_capstone_payload(),
        exp5474=_exp5474_payload(),
        exp5480=wrong_arc,
    )
    wrong_arc_artifact = mod.build_artifact(
        root=wrong_arc_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(wrong_arc_artifact)
    for failure in (
        "exp5480_milestone_expected_2026.07.497_observed_2026.07.000",
        "exp5480_expected_sb26_observed_bp35",
        "exp5480_target_level_expected_3_observed_2",
        "exp5480_new_level_banked_expected_false",
        "exp5480_offline_reproduced_expected_false",
    ):
        assert failure in wrong_arc_artifact["failed_preconditions"]

    incomplete_capstone = _capstone_payload()
    incomplete_capstone["truth_table"].pop("hardware_receipts")
    incomplete_root = _make_repo(
        tmp_path / "incomplete",
        capstone=incomplete_capstone,
        exp5474=_exp5474_payload(),
        exp5480=_exp5480_payload(),
    )
    incomplete = mod.build_artifact(
        root=incomplete_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(incomplete)
    for failure in (
        "clean_lanes_incomplete",
        "bounded_lanes_incomplete",
        "blocked_lanes_incomplete",
        "honest_null_lanes_incomplete",
    ):
        assert failure in incomplete["failed_preconditions"]

    bad_capstone = _capstone_payload()
    bad_capstone["milestone"] = "2026.07.000"
    bad_capstone["honest_verdict"] = "done"
    bad_capstone["hardware_speedup_claim"] = True
    bad_capstone_root = _make_repo(
        tmp_path / "bad-capstone",
        capstone=bad_capstone,
        exp5474=_exp5474_payload(),
        exp5480=_exp5480_payload(),
    )
    bad_capstone_artifact = mod.build_artifact(
        root=bad_capstone_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(bad_capstone_artifact)
    for failure in (
        "capstone_milestone_expected_2026.07.497_observed_2026.07.000",
        "capstone_honest_verdict_missing_terminal_prefix",
        "capstone_hardware_speedup_claim_not_false",
    ):
        assert failure in bad_capstone_artifact["failed_preconditions"]

    wrong_exp5474 = _exp5474_payload()
    wrong_exp5474["milestone"] = "2026.07.000"
    wrong_exp5474_root = _make_repo(
        tmp_path / "wrong-exp5474",
        capstone=_capstone_payload(),
        exp5474=wrong_exp5474,
        exp5480=_exp5480_payload(),
    )
    wrong_exp5474_artifact = mod.build_artifact(
        root=wrong_exp5474_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(wrong_exp5474_artifact)
    assert (
        "exp5474_milestone_expected_2026.07.497_observed_2026.07.000"
        in wrong_exp5474_artifact["failed_preconditions"]
    )


def test_req_report_5482_run_writes_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5482: run writes a deterministic transition receipt."""

    root = _make_repo(
        tmp_path / "repo",
        capstone=_capstone_payload(),
        exp5474=_exp5474_payload(),
        exp5480=_exp5480_payload(),
    )
    result_path = tmp_path / "out" / "transition.json"

    written = mod.run(
        root=root,
        result_path=result_path,
        run_date="2026-07-09",
        tests_run=[{"command": "unit 5482", "outcome": "passed"}],
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert written == result_path
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"


def test_req_report_5482_committed_result_matches_replay() -> None:
    """REQ-REPORT-5482: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["previous_task_range"] == "exp5468-exp5481"
    assert result["next_task_range"] == "exp5482-exp5495"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True
    assert result["exp5474_tautology_flag_recorded"] is True


def test_req_report_5482_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5482: validation rejects malformed transition receipts."""

    root = _make_repo(
        tmp_path / "repo",
        capstone=_capstone_payload(),
        exp5474=_exp5474_payload(),
        exp5480=_exp5480_payload(),
    )
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
        ("status", "done", "status"),
        ("field_principles", {}, "field_principles"),
        ("milestone", mod.PREVIOUS_MILESTONE, "milestone"),
        ("previous_milestone", mod.MILESTONE, "previous_milestone"),
        ("prior_capstone_path", "wrong.json", "prior_capstone_path"),
        ("previous_task_range", "exp5468-exp5480", "previous_task_range"),
        ("next_task_range", "exp5482-exp5494", "next_task_range"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("roadmap_yaml_unchanged", "true", "roadmap_yaml_unchanged"),
        ("conductor_unchanged", "true", "conductor_unchanged"),
        ("exp5474_tautology_flag_recorded", "true", "exp5474_tautology"),
        ("honest_verdict", "done", "honest_verdict"),
        ("roadmap_task_ids", ["wrong"], "roadmap_task_ids"),
        ("clean_lanes", "bad", "clean_lanes"),
        ("bounded_lanes", "bad", "bounded_lanes"),
        ("blocked_lanes", "bad", "blocked_lanes"),
        ("honest_null_lanes", "bad", "honest_null_lanes"),
        ("flagged_lanes", "bad", "flagged_lanes"),
        ("clean_lanes", [], "clean_lanes"),
        ("bounded_lanes", [], "bounded_lanes"),
        ("blocked_lanes", [], "blocked_lanes"),
        ("honest_null_lanes", [], "honest_null_lanes"),
        ("flagged_lanes", [], "flagged_lanes"),
        ("roadmap_yaml_unchanged", False, "roadmap_yaml_unchanged must be true"),
        ("conductor_unchanged", False, "conductor_unchanged must be true"),
        ("exp5474_tautology_flag_recorded", False, "complete status"),
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

    assert mod._truth_rows({"truth_table": "bad"}) == {}
    assert list(mod._truth_rows({"truth_table": [{"lane": "listed", "value": 1}, "bad", {}]})) == [
        "listed"
    ]
    assert mod._evidence({"terminal_evidence": {"ok": True}}) == {"ok": True}
    assert mod._source_artifacts({"source_artifact": "one.json"}, ["fallback.json"]) == ["one.json"]
    assert mod._source_artifacts({"source_artifacts": {"bad": True}}, ["fallback.json"]) == [
        "fallback.json"
    ]
    assert mod._lane_names("bad") == []
    assert mod._exp5474_tautology_record({"corrigendum_pending": "bad"})[0] is False
    assert mod._exp5474_tautology_record({"corrigendum_pending": [None]})[0] is False
