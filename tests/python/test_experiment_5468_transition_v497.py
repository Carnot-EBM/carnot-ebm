"""Tests for Exp5468 .497 transition receipt.

Spec refs: REQ-REPORT-5468, SCENARIO-REPORT-5468,
SCENARIO-REPORT-5468-BLOCKED-INPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5468_transition_v497 as mod


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
            "prompt": "REQ-REPORT-5468 fixture",
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


def _vnext_doc(milestone: str = mod.MILESTONE, task_range: str = "Exp 5468-5481") -> str:
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
    source_artifacts: list[str] | None = None,
    authority_gate: str = "fixture authority",
    claim_boundary: str = "fixture boundary",
    headline_blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "lane": lane,
        "classification": classification,
        "authority_gate": authority_gate,
        "source_artifacts": source_artifacts or [f"results/{lane}.json"],
        "evidence": evidence,
        "headline_blockers": headline_blockers or [],
        "claim_boundary": claim_boundary,
    }


def _capstone_payload() -> dict[str, Any]:
    hardware_reachability = {
        "kv260": {
            "blocked_reason": "blocked_kv260_ssh",
            "check_method": "ssh_only",
            "identity": "kria",
            "reachable": False,
            "workload_execution_attempted": False,
        },
        "polarfire": {
            "blocked_reason": None,
            "check_method": "ssh",
            "identity": "polarfire",
            "reachable": True,
            "workload_execution_attempted": True,
        },
    }
    truth_table = {
        "minimal_core_repair": _truth_row(
            "minimal_core_repair",
            "headline_ready",
            {
                "exp5458": {
                    "minimal_core_repair_ready": True,
                    "exact_final_authority": True,
                    "repaired_accept_rate_after_exact_recheck": 1.0,
                    "unrepaired_reject_rate": 1.0,
                    "honest_verdict": (
                        "complete: deterministic minimal-core repairs accepted "
                        "after exact recheck"
                    ),
                }
            },
            claim_boundary="deterministic repair acceptance after exact recheck",
        ),
        "distortion_guards": _truth_row(
            "distortion_guards",
            "headline_ready",
            {
                "exp5459": {
                    "distortion_guard_ready": True,
                    "exact_final_authority": True,
                    "fixture_count": 10,
                    "unsupported_fabrication_rate": 0.1,
                    "honest_verdict": "complete: deterministic constraint-distortion guard ready",
                }
            },
            claim_boundary="deterministic distortion-guard readiness",
        ),
        "csl_policy": _truth_row(
            "csl_policy",
            "headline_ready",
            {
                "exp5460": {
                    "continuous_self_learning_task": True,
                    "csl_policy_ready": True,
                    "no_weight_mutation": True,
                    "negative_transfer_deflection_rate": 1.0,
                    "policy_update_count": 9,
                    "honest_verdict": "complete: frozen-model governed CSL policy ready",
                }
            },
            claim_boundary="governed sidecar routing only",
        ),
        "sota_csl": _truth_row(
            "sota_csl",
            "headline_ready",
            {
                "exp5461": {
                    "csl_sota_memory_routing_ready": True,
                    "gpu_offload_verified": True,
                    "no_weight_mutation": True,
                    "negative_transfer_deflection_rate": 1.0,
                    "runtime_backend": "llama_cpp_python_cuda_gguf",
                    "honest_verdict": (
                        "complete: live SOTA GGUF governed CSL memory routing "
                        "preserved quality and deflected negative transfer with "
                        "frozen weights"
                    ),
                }
            },
            claim_boundary="live GGUF memory routing under frozen weights",
        ),
        "pbit_pdit_bridge": _truth_row(
            "pbit_pdit_bridge",
            "bounded",
            {
                "exp5462": {
                    "minimal_core_pbit_bridge_ready": True,
                    "solver_authoritative": True,
                    "fallback_completeness_rate": 1.0,
                    "hardware_speedup_claim": False,
                    "honest_verdict": "complete: active p-bit and p-dit assumptions stayed advisory",
                }
            },
            claim_boundary="bounded bridge only",
        ),
        "hardware_receipts": _truth_row(
            "hardware_receipts",
            "bounded",
            {
                "exp5463": {
                    "hardware_receipts_ready": True,
                    "hashes_match_before_timing_compare": True,
                    "hardware_speedup_claim": False,
                    "board_reachability": hardware_reachability,
                    "timing_repeat_counts": {"cpu": 10, "kv260": 0, "polarfire": 10},
                    "honest_verdict": (
                        "complete: CPU and reachable-board timing receipts are "
                        "hash-matched; hardware_speedup_claim=false"
                    ),
                }
            },
            claim_boundary="KV260 unreachable and no speedup claim",
        ),
        "synthesis": _truth_row(
            "synthesis",
            "bounded",
            {
                "exp5466": {
                    "status": "complete",
                    "missing_artifacts": [],
                    "skipped_gated_tasks": [],
                    "honest_verdict": (
                        "complete: .496 PRD gap table read actual Exp5454-Exp5465 "
                        "artifacts"
                    ),
                }
            },
            claim_boundary="bounded capstone/PRD synthesis only",
        ),
        "guided_decoding": _truth_row(
            "guided_decoding",
            "blocked",
            {
                "exp5457": {
                    "flagged_adversarial": True,
                    "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical"}],
                    "verifier_guided_decoding_ready": False,
                    "lcd_bias_check_passed": False,
                    "honest_verdict": "complete: live panel ran; readiness false (lcd_bias_check_failed)",
                }
            },
            claim_boundary="guided decoding remains quarantined after Exp5457",
            headline_blockers=["flagged_adversarial", "tautology", "lcd_bias_failed"],
        ),
        "arc": _truth_row(
            "arc",
            "honest_null",
            {
                "exp5465": {
                    "new_level_banked": False,
                    "offline_reproduced": False,
                    "failure_mode": "bounded_budget_no_levelup",
                    "live_attempt_count": 1,
                    "solve_provenance": "live_agent_self_discovery",
                    "source_reading_used": False,
                    "target_game": "bp35",
                    "honest_verdict": "honest_null: bp35 L3 bounded_budget_no_levelup",
                }
            },
            claim_boundary="bp35 L3 no-bank",
        ),
    }
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "honest_verdict": (
            "complete: .496 capstone truth table from actual artifacts; "
            "headline_ready=4, bounded=3, blocked=1, honest_null=2; guided "
            "decoding quarantined, ARC no-bank, hardware_speedup_claim=false."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "roadmap_yaml_unchanged": True,
        "conductor_unchanged": True,
        "headline_ready_lanes": [
            "distortion_guards",
            "minimal_core_repair",
            "csl_policy",
            "sota_csl",
        ],
        "bounded_lanes": ["pbit_pdit_bridge", "hardware_receipts", "synthesis"],
        "blocked_lanes": ["guided_decoding"],
        "honest_null_lanes": ["arc", "hardware_speedup_claim"],
        "truth_table": truth_table,
        "no_claim_boundaries": [
            {
                "claim_id": "hardware_speedup",
                "boundary": "No hardware speedup is claimed.",
                "evidence": {
                    "hardware_speedup_claim": False,
                    "board_reachability": hardware_reachability,
                    "timing_comparison": {
                        "hardware_speedup_claim": False,
                        "hashes_match_before_timing_compare": True,
                    },
                },
            }
        ],
    }


def _gap_payload() -> dict[str, Any]:
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "status": "complete",
        "honest_verdict": (
            "complete: .496 PRD gap table read actual Exp5454-Exp5465 artifacts; "
            "closed=8, partial=2, blocked=2, honest_null=2, missing=0, skipped=0."
        ),
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "missing_artifacts": [],
        "skipped_gated_tasks": [],
        "agent_failure_taxonomy": {
            "missing_hardware": {
                "observed": True,
                "blocked_boards": ["kv260"],
            },
            "no_bank_arc": {"observed": True},
            "tautology": {"observed": True},
        },
    }


def _make_repo(
    root: Path,
    *,
    capstone: dict[str, Any] | None = None,
    gap: dict[str, Any] | None = None,
    milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    doc_task_range: str = "Exp 5468-5481",
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
        "| 2026-07-09 05:43 UTC | Milestone .496 capstone | OK |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    if gap is not None:
        _write_json(root / mod.GAP_RELATIVE_PATH, gap)
    return root


def test_req_report_5468_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5468: OpenSpec anchors the .497 transition receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5468") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5468",
        "SCENARIO-REPORT-5468",
        "SCENARIO-REPORT-5468-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "exp5454-exp5467",
        "exp5468-exp5481",
        "deterministic minimal-core repair",
        "constraint-distortion guards",
        "governed CSL policy",
        "SOTA CSL memory routing",
        "p-bit/p-dit bridge",
        "Exp5457",
        "ARC `bp35` L3 no-bank",
        "KV260 unreachable",
        "no hardware speedup claim",
    ):
        assert marker in section or marker in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5468_builds_complete_transition_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5468: active .497 records .496 boundaries and gates."""

    root = _make_repo(tmp_path, capstone=_capstone_payload(), gap=_gap_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="2026-07-09",
        tests_run=[{"command": "unit 5468", "outcome": "passed"}],
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
        "minimal_core_repair",
        "distortion_guards",
        "csl_policy",
        "sota_csl",
    ]
    assert closed["minimal_core_repair"]["terminal_evidence"]["exp5458"][
        "exact_final_authority"
    ] is True
    assert closed["distortion_guards"]["terminal_evidence"]["exp5459"][
        "distortion_guard_ready"
    ] is True
    assert closed["csl_policy"]["terminal_evidence"]["exp5460"]["no_weight_mutation"] is True
    assert closed["sota_csl"]["terminal_evidence"]["exp5461"]["gpu_offload_verified"] is True

    bounded = {row["lane"]: row for row in artifact["bounded_lanes"]}
    assert list(bounded) == ["pbit_pdit_bridge", "hardware_receipts", "synthesis"]
    assert bounded["pbit_pdit_bridge"]["terminal_evidence"]["exp5462"][
        "solver_authoritative"
    ] is True
    assert bounded["hardware_receipts"]["terminal_evidence"]["exp5463"][
        "hardware_speedup_claim"
    ] is False
    assert bounded["hardware_receipts"]["terminal_evidence"]["exp5463"][
        "board_reachability"
    ]["kv260"]["reachable"] is False
    assert bounded["synthesis"]["terminal_evidence"]["exp5466"]["missing_artifacts"] == []

    blocked = {row["lane"]: row for row in artifact["blocked_lanes"]}
    assert list(blocked) == ["guided_decoding"]
    assert blocked["guided_decoding"]["terminal_evidence"]["exp5457"][
        "flagged_adversarial"
    ] is True
    assert blocked["guided_decoding"]["terminal_evidence"]["exp5457"][
        "lcd_bias_check_passed"
    ] is False

    honest_null = {row["lane"]: row for row in artifact["honest_null_lanes"]}
    assert list(honest_null) == ["arc", "hardware_speedup_claim"]
    assert honest_null["arc"]["terminal_evidence"]["exp5465"]["target_game"] == "bp35"
    assert honest_null["arc"]["terminal_evidence"]["exp5465"]["new_level_banked"] is False
    assert honest_null["hardware_speedup_claim"]["terminal_evidence"][
        "hardware_speedup_claim"
    ] is False
    assert honest_null["hardware_speedup_claim"]["terminal_evidence"][
        "board_reachability"
    ]["kv260"]["reachable"] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5468_missing_or_dirty_inputs_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5468-BLOCKED-INPUT: missing or dirty inputs fail closed."""

    root = _make_repo(
        tmp_path / "missing",
        capstone=None,
        gap=None,
        milestone=mod.PREVIOUS_MILESTONE,
        doc_milestone=mod.PREVIOUS_MILESTONE,
        doc_task_range="Exp 5468-5480",
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
    assert artifact["bounded_lanes"] == []
    assert artifact["blocked_lanes"] == []
    assert artifact["honest_null_lanes"] == []
    assert artifact["roadmap_yaml_unchanged"] is False
    assert artifact["conductor_unchanged"] is False
    for failure in (
        "capstone_missing_or_unloadable",
        "gap_table_missing_or_unloadable",
        "roadmap_milestone_expected_2026.07.497_observed_2026.07.496",
        "roadmap_doc_missing_or_mismatch_2026.07.497",
        "roadmap_doc_task_range_expected_exp5468-exp5481_observed_exp5468-exp5480",
        "roadmap_task_ids_mismatch",
        "research-roadmap.yaml_modified",
        "scripts/research_conductor.py_modified",
    ):
        assert failure in artifact["failed_preconditions"]

    bad_capstone = _capstone_payload()
    bad_capstone["milestone"] = "2026.07.495"
    bad_capstone["status"] = "blocked"
    bad_capstone["honest_verdict"] = "done"
    bad_capstone["no_claim_boundaries"] = []
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
        "capstone_milestone_expected_2026.07.496_observed_2026.07.495",
        "capstone_status_expected_complete_or_absent_observed_blocked",
        "capstone_honest_verdict_missing_terminal_prefix",
        "capstone_hardware_speedup_claim_boundary_missing_or_true",
        "capstone_honest_null_lanes_incomplete",
    ):
        assert failure in bad_artifact["failed_preconditions"]

    incomplete_capstone = _capstone_payload()
    incomplete_capstone["truth_table"].pop("minimal_core_repair")
    incomplete_capstone["truth_table"].pop("hardware_receipts")
    incomplete_capstone["truth_table"].pop("guided_decoding")
    incomplete_capstone["no_claim_boundaries"] = []
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
        "capstone_bounded_lanes_incomplete",
        "capstone_blocked_lanes_incomplete",
        "capstone_honest_null_lanes_incomplete",
    ):
        assert failure in incomplete_artifact["failed_preconditions"]

    bad_gap = _gap_payload()
    bad_gap["milestone"] = "2026.07.495"
    bad_gap["status"] = "blocked"
    bad_gap["honest_verdict"] = "done"
    bad_gap_root = _make_repo(
        tmp_path / "bad-gap",
        capstone=_capstone_payload(),
        gap=bad_gap,
    )
    bad_gap_artifact = mod.build_artifact(
        root=bad_gap_root,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    mod.validate_artifact(bad_gap_artifact)
    for failure in (
        "gap_milestone_expected_2026.07.496_observed_2026.07.495",
        "gap_status_expected_complete_or_absent_observed_blocked",
        "gap_honest_verdict_missing_terminal_prefix",
    ):
        assert failure in bad_gap_artifact["failed_preconditions"]


def test_req_report_5468_run_writes_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5468: run writes a deterministic transition receipt."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload(), gap=_gap_payload())
    result_path = tmp_path / "out" / "transition.json"

    written = mod.run(
        root=root,
        result_path=result_path,
        run_date="2026-07-09",
        tests_run=[{"command": "unit 5468", "outcome": "passed"}],
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert written == result_path
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"


def test_req_report_5468_committed_result_matches_replay() -> None:
    """REQ-REPORT-5468: checked-in deliverable is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    mod.validate_artifact(result)
    assert result == replay
    assert result["status"] == "complete"
    assert result["previous_task_range"] == "exp5454-exp5467"
    assert result["next_task_range"] == "exp5468-exp5481"
    assert result["roadmap_yaml_unchanged"] is True
    assert result["conductor_unchanged"] is True


def test_req_report_5468_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-REPORT-5468: validation rejects malformed transition receipts."""

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
        ("previous_task_range", "exp5454-exp5466", "previous_task_range"),
        ("next_task_range", "exp5468-exp5480", "next_task_range"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("roadmap_yaml_unchanged", "true", "roadmap_yaml_unchanged"),
        ("conductor_unchanged", "true", "conductor_unchanged"),
        ("honest_verdict", "done", "honest_verdict"),
        ("roadmap_task_ids", ["wrong"], "roadmap_task_ids"),
        ("closed_lanes", "bad", "closed_lanes"),
        ("bounded_lanes", "bad", "bounded_lanes"),
        ("blocked_lanes", "bad", "blocked_lanes"),
        ("honest_null_lanes", "bad", "honest_null_lanes"),
        ("closed_lanes", [], "closed_lanes"),
        ("bounded_lanes", [], "bounded_lanes"),
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

    assert mod._truth_rows("bad") == {}
    assert list(mod._truth_rows([{"lane": "listed", "value": 1}, "bad", {}])) == [
        "listed"
    ]
    assert mod._truth_record({"lane": "fallback", "terminal_evidence": {"ok": True}})[
        "terminal_evidence"
    ] == {"ok": True}
    assert mod._truth_record({"lane": "blocked", "blocked_reason": "fixture"})[
        "blocked_reason"
    ] == "fixture"
    assert mod._truth_record({"lane": "fallback"})["source_artifacts"] == []
    assert mod._source_artifacts({"source_artifact": "results/source.json"}) == [
        "results/source.json"
    ]
    assert (
        mod._hardware_speedup_claim_is_false(
            {"no_claim_boundaries": [{"claim_id": "hardware_speedup", "evidence": "bad"}]}
        )
        is False
    )
