"""Tests for Exp5755 V514 transition reconciliation.

Spec refs: REQ-REPORT-5755, SCENARIO-REPORT-5755,
SCENARIO-REPORT-5755-BLOCKED-VERSUS-NULL,
SCENARIO-REPORT-5755-COLLISION-BLOCK,
SCENARIO-REPORT-5755-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5755_transition_v514 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {"id": task_id, "deliverable": f"results/{task_id}.json"}
            for task_id in mod.NEXT_TASK_IDS
        ],
    }


def _artifact_payloads() -> dict[Path, JsonDict]:
    return {
        mod.EXP5743_PATH: {
            "schema": "carnot.experiment_5743.transition_v513.v1",
            "status": "complete",
            "honest_verdict": "complete: archived terminal .512 evidence into .513",
            "proposal_channel_ready": True,
            "sota_proposal_stream_ready": True,
            "rust_batched_10x_ready": False,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
        },
        mod.EXP5744_PATH: {
            "schema": "carnot.experiment_5744.v513_source_delta_ingestion.v1",
            "status": "complete",
            "honest_verdict": "complete: no new non-duplicate actionable V513 source deltas",
            "accepted_findings": [],
        },
        mod.EXP5745_PATH: {
            "honest_verdict": "complete: exp5740_lossless_scalar_gate_corrigendum_positive_count_7_admitted_leaks_0_registry_delta_0",
            "counterfactual_receipt_coverage_score": 1.0,
            "admitted_source_leak_count": 0,
            "admitted_game_identity_leak_count": 0,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
        },
        mod.EXP5746_PATH: {
            "schema": "carnot.experiment_5746.exact_proposal_utility_benchmark.v1",
            "honest_verdict": "complete: exact_proposal_utility_benchmark_ready",
            "benchmark_ready_score": 1.0,
        },
        mod.EXP5747_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "3 of 3 gate(s) failed; first failure: exp5746-exact-proposal-utility-benchmark.benchmark_ready_score",
        },
        mod.EXP5749_PATH: {
            "schema": "carnot.experiment_5749.csl_render_matched_mechanism_audit.v1",
            "honest_verdict": "complete: kan_mechanism_residual_negative_fr11_safety_retained",
            "continuous_self_learning_credited": True,
            "kan_mechanism_residual": -0.084269,
        },
        mod.EXP5750_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 3 gate(s) failed; first failure: exp5749-csl-render-matched-mechanism-audit.kan_mechanism_residual",
        },
        mod.EXP5751_PATH: {
            "honest_verdict": "complete: restart parity repaired; no timing or hardware claim",
            "restart_parity_ready_score": 1.0,
            "distributional_parity": {"passed": True},
            "production_backend_reachable": {"passed": True},
        },
        mod.EXP5752_PATH: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "3 of 4 gate(s) failed; first failure: exp5751-rust-restart-parity-repair.distributional_parity",
        },
        mod.EXP5753_PATH: {
            "honest_verdict": "complete: generic_primitive_live_registry_ab_delta_0_registry_credit_0",
            "live_level_reproduction_delta": 0,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
            "solve_provenance": "development_proxy",
        },
        mod.EXP5754_PATH: {
            "schema": "carnot.experiment_5754.v513_capstone_reconciliation.v1",
            "status": "complete",
            "honest_verdict": "complete: v513 reconciled",
            "proposal_benchmark_ready": True,
            "proposal_utility_ready": False,
            "kan_mechanism_residual": -0.084269,
            "rust_restart_parity_ready": True,
            "rust_batched_10x_ready": False,
            "arc_live_level_reproduction_delta": 0,
            "arc_registry_delta": 0,
            "arc_solve_credited": False,
        },
    }


def _conductor_log() -> str:
    return "\n".join(
        [
            "| t | Transition terminal .512 evidence, preserve mixed | OK | done |",
            "| t | Ingest post-V513 source deltas with honest bibliog | OK | done |",
            "| t | Normalize the Exp5740 ARC causal gate without chan | OK | done |",
            "| t | Build a disjoint dual-receipt exact benchmark for | OK | done |",
            "| t | Gated on Exp5746 readiness: measure SOTA proposal | GATE_BLOCK | 3 failed; first failure: exp5746-exact-proposal-utility-benchmark.benchmark_ready_score |",
            "| t | Gated on Exp5747 utility>0: allocate exact feedbac | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| t | Audit render- and parameter-matched continuous sel | OK | negative residual |",
            "| t | Gated on Exp5749 KAN residual>0: scale continuous | GATE_BLOCK | 1 failed; first failure: exp5749-csl-render-matched-mechanism-audit.kan_mechanism_residual |",
            "| t | Localize and repair one-axis Rust batch restart mi | OK | parity repaired |",
            "| t | Gated on Exp5751 parity: run allocation-free Rust/ | GATE_BLOCK | object-valued fields; first failure: exp5751-rust-restart-parity-repair.distributional_parity |",
            "| t | Gated on Exp5745 clean scalar gate: run generic pr | OK | delta zero |",
            "| t | Reconcile .513 proposal, CSL, Rust, ARC, specifica | OK | capstone |",
        ]
    ) + "\n"


def _make_root(root: Path, *, duplicate_complete_block: bool = False) -> None:
    for rel_path, payload in _artifact_payloads().items():
        _write_json(root, rel_path, payload)

    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(
        root,
        mod.VNEXT_RELATIVE_PATH,
        "# Research Roadmap vNEXT\n\n**Milestone:** `2026.07.514`\n",
    )
    complete_block = "- id: 2026.07.513\n  tasks: []\n"
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        complete_block + (complete_block if duplicate_complete_block else ""),
    )
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")


def test_spec_contains_req_report_5755_contract() -> None:
    """REQ-REPORT-5755: OpenSpec names the transition artifact and scenarios."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5755") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "SCENARIO-REPORT-5755-BLOCKED-VERSUS-NULL" in section
    assert "cached_artifact_reconciliation_no_llm" in section


def test_scenario_report_5755_archives_terminal_v513_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5755: terminal evidence is carried without inflation."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_from"] == "2026.07.513"
    assert report["milestone_to"] == "2026.07.514"
    assert report["archived_task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert report["artifact_hashes"]["exp5748-selective-exact-feedback-search"][
        "status"
    ] == "missing"
    assert report["conductor_outcomes"]["exp5747-sota-exact-proposal-utility-panel"][
        "outcome"
    ] == "GATE_BLOCK"
    assert report["conductor_outcomes"]["exp5746-exact-proposal-utility-benchmark"][
        "outcome"
    ] == "OK"
    assert report["conductor_outcomes"]["exp5749-csl-render-matched-mechanism-audit"][
        "outcome"
    ] == "OK"
    assert report["conductor_outcomes"]["exp5751-rust-restart-parity-repair"][
        "outcome"
    ] == "OK"

    assert report["proposal_benchmark_ready"] is True
    assert report["proposal_utility_measured"] is False
    assert report["kan_mechanism_residual"] == -0.084269
    assert report["rust_restart_parity_ready"] is True
    assert report["rust_10x_measured"] is False
    assert report["arc_live_delta_measured"] is True
    assert report["arc_live_delta"] == 0
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == "cached_artifact_reconciliation_no_llm"
    assert report["docs_reconciled"]["mode"] == "already_archived_once_no_rewrite"
    assert report["collision_scan"]["preexisting_collision_count"] == 0
    assert report["next_task_range"] == "exp5755-exp5768"


def test_scenario_report_5755_blocked_tasks_do_not_become_nulls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5755-BLOCKED-VERSUS-NULL: gate blocks stay separate."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["blocked_task_ids"] == [
        "exp5747-sota-exact-proposal-utility-panel",
        "exp5748-selective-exact-feedback-search",
        "exp5750-dependent-task-continuous-self-learning",
        "exp5752-one-axis-allocation-free-10x-crossover",
    ]
    assert report["scientific_null_task_ids"] == [
        "exp5753-arc-generic-primitive-live-registry-ab"
    ]
    assert set(report["blocked_task_ids"]).isdisjoint(report["scientific_null_task_ids"])
    assert report["negative_result_task_ids"] == [
        "exp5749-csl-render-matched-mechanism-audit"
    ]
    assert set(report["positive_result_task_ids"]) == {
        "exp5746-exact-proposal-utility-benchmark",
        "exp5751-rust-restart-parity-repair",
    }


def test_scenario_report_5755_collision_blocks_completion(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5755-COLLISION-BLOCK: next-range collisions fail closed."""

    _make_root(tmp_path)
    (tmp_path / "results/experiment_5757_directory").mkdir(parents=True)
    cache_path = tmp_path / "tests/python/__pycache__/experiment_5758_cache.pyc"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"cache")
    _write_json(
        tmp_path,
        "results/experiment_5756_v514_source_delta_ingestion.json",
        {"status": "preexisting"},
    )
    _write_text(
        tmp_path,
        "openspec/change-proposals/research-roadmap-v514-copy.md",
        "duplicate future reference exp5768-v514-capstone-reconciliation\n",
    )
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["collision_scan"]["preexisting_collision_count"] == 2
    assert report["collision_scan"]["preexisting_collisions"][0]["path"] == (
        "results/experiment_5756_v514_source_delta_ingestion.json"
    )
    assert report["collision_scan"]["preexisting_collisions"][1]["path"] == (
        "openspec/change-proposals/research-roadmap-v514-copy.md"
    )
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_report_5755_precondition_failures_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5755-COLLISION-BLOCK: ambiguous inputs block honestly."""

    _make_root(tmp_path)
    _write_text(tmp_path, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump({"milestone": "2026.07.513"}))
    (tmp_path / mod.VNEXT_RELATIVE_PATH).unlink()
    (tmp_path / mod.EXP5746_PATH).unlink()
    _write_text(tmp_path, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no exp5748 preemptive evidence\n")

    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )

    failures = report["preconditions_checked"]["failed_preconditions"]
    assert report["status"] == "blocked"
    assert "active_roadmap_milestone='2026.07.513'" in failures
    assert "vnext_milestone=None" in failures
    assert "research_roadmap_modified" in failures
    assert "research_conductor_modified" in failures
    assert "unexpected_missing_or_malformed=['exp5746-exact-proposal-utility-benchmark']" in failures
    assert "exp5748_preemptive_skip_not_verified" in failures


def test_scenario_report_5755_emit_report_and_field_principles(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5755-FIELD-PRINCIPLES: emitted artifact is stable."""

    _make_root(tmp_path)
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["reproducibility_checksum"] == report["reproducibility_checksum"]
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(report).issubset(report["field_principles"])
    assert all(report["field_principles"][field] for field in report)
    assert mod._vnext_milestone(tmp_path / "missing") is None
    assert mod._research_complete_block_count(tmp_path / "missing") == 0
    assert mod._planned_task_ids(tmp_path / "missing") == []
    assert mod._load_tests_run(None)[0]["status"] == "not_run"

    duplicate_root = tmp_path / "duplicate"
    _make_root(duplicate_root, duplicate_complete_block=True)
    duplicate_report = mod.build_report(
        duplicate_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert duplicate_report["status"] == "blocked"

    original = mod.FIELD_PRINCIPLES.pop("schema")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(
                tmp_path,
                modification_overrides={
                    mod.ROADMAP_RELATIVE_PATH: False,
                    mod.CONDUCTOR_RELATIVE_PATH: False,
                },
            )
    finally:
        mod.FIELD_PRINCIPLES["schema"] = original
