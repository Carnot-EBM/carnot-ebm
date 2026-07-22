"""Tests for the Exp5769 V515 transition receipt.

Spec refs: REQ-REPORT-5769, SCENARIO-REPORT-5769,
SCENARIO-REPORT-5769-COLLISION-BLOCK,
SCENARIO-REPORT-5769-DECLARED-DELIVERABLE-BLOCK,
SCENARIO-REPORT-5769-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5769_transition_v515 as mod


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


def _canonical_payload(task_id: str) -> JsonDict:
    verdicts = {
        "exp5755-transition-v514": "blocked: exp5755 transition preconditions failed: next_range_collision_count=3",
        "exp5756-v514-source-delta-ingestion": "complete: no new non-duplicate actionable V514 source deltas; references left unchanged",
        "exp5757-proposal-benchmark-scalar-bridge": "complete: exact proposal benchmark scalars bridged for downstream gates",
        "exp5758-rust-parity-scalar-bridge": "complete: Exp5751 Rust parity receipts bridged to bare scalar gates",
        "exp5759-sota-exact-proposal-utility-panel": "complete: sota_exact_proposal_utility_measured_gate_not_ready",
        "exp5760-selective-exact-feedback-search": "blocked_gate_check_failed",
        "exp5761-exact-constraint-acquisition-benchmark": "complete: exact_constraint_acquisition_benchmark_ready",
        "exp5762-query-driven-constraint-lifecycle": "complete: query_driven_constraint_lifecycle_credited",
        "exp5763-dependent-task-constraint-acquisition": "complete: dependent_task_constraint_acquisition_credited",
        "exp5764-one-axis-profiled-allocation-free-hot-path": "complete: profiled allocation-free one-axis hot path; no timing claim",
        "exp5765-one-axis-final-10x-crossover": "complete: allocation-free one-axis PyO3 technique retired",
        "exp5766-arc-loo-component-interaction-audit": "complete: loo_component_interaction_audit_no_heldout_gain_no_causal_interactions",
        "exp5767-arc-game-blind-composition-hardening": "blocked_gate_check_failed",
        "exp5768-v514-capstone-reconciliation": "complete: V514 reconciled",
    }
    blocked = task_id in {
        "exp5755-transition-v514",
        "exp5760-selective-exact-feedback-search",
        "exp5767-arc-game-blind-composition-hardening",
    }
    payload: JsonDict = {
        "experiment_id": task_id,
        "status": "blocked" if blocked else "complete",
        "honest_verdict": verdicts[task_id],
    }
    if task_id in {
        "exp5760-selective-exact-feedback-search",
        "exp5767-arc-game-blind-composition-hardening",
    }:
        payload.update(
            {"schema": "blocked_gate_check_v1", "blocked_at_layer": "conductor_pre_gate"}
        )
    if task_id == "exp5759-sota-exact-proposal-utility-panel":
        payload.update(
            {
                "proposal_utility_delta_overall": 0.003416373531,
                "proposal_utility_lcb": -0.045291796847,
                "proposal_utility_ready_score": 0.0,
                "flagship_nonregression_count": 0,
            }
        )
    if task_id == "exp5762-query-driven-constraint-lifecycle":
        payload.update(
            {
                "constraint_recovery_gain_lcb": 0.143812,
                "behavioral_exact_accuracy": 1.0,
                "unsafe_update_count": 0,
                "rollback_hash_mismatch_count": 0,
            }
        )
    if task_id == "exp5763-dependent-task-constraint-acquisition":
        payload.update(
            {
                "dependent_task_ca_ready_score": 1.0,
                "old_task_retention_delta": 0.12,
                "unsafe_update_count": 0,
                "rollback_hash_mismatch_count": 0,
            }
        )
    if task_id == "exp5765-one-axis-final-10x-crossover":
        payload.update(
            {
                "rust_10x_claimed": False,
                "rust_10x_retired": True,
                "nfr01_status": "retired_allocation_free_one_axis_pyo3_technique",
            }
        )
    if task_id == "exp5766-arc-loo-component-interaction-audit":
        payload.update(
            {
                "loo_generalization_delta": 0.0,
                "loo_generalization_delta_lcb": 0.0,
                "causal_interaction_count": 0,
                "arc_registry_delta": 0,
                "arc_solve_credited": False,
            }
        )
    return payload


def _outcome_matrix() -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        gate_skipped = task_id in {
            "exp5760-selective-exact-feedback-search",
            "exp5767-arc-game-blind-composition-hardening",
        }
        rows[task_id] = {
            "artifact_path": rel_path.as_posix(),
            "artifact_present": True,
            "artifact_status": "blocked-gate" if gate_skipped else "complete",
            "blocked_gate": gate_skipped,
            "blocked_precondition": task_id == "exp5755-transition-v514",
            "complete": not gate_skipped and task_id != "exp5755-transition-v514",
            "conductor_outcome": "GATE_BLOCK" if gate_skipped else "OK",
            "evidence_line": f"| t | {task_id} | {'GATE_BLOCK' if gate_skipped else 'OK'} | fixture |",
            "gate_block_reason": "fixture gate failed" if gate_skipped else None,
            "honest_verdict": _canonical_payload(task_id)["honest_verdict"],
            "negative": task_id == "exp5759-sota-exact-proposal-utility-panel",
            "null": task_id
            in {
                "exp5765-one-axis-final-10x-crossover",
                "exp5766-arc-loo-component-interaction-audit",
            },
            "planned": True,
            "promoted": task_id
            in {
                "exp5757-proposal-benchmark-scalar-bridge",
                "exp5758-rust-parity-scalar-bridge",
                "exp5761-exact-constraint-acquisition-benchmark",
                "exp5762-query-driven-constraint-lifecycle",
                "exp5763-dependent-task-constraint-acquisition",
                "exp5764-one-axis-profiled-allocation-free-hot-path",
            },
        }
    rows["exp5755-transition-v514"]["artifact_status"] = "blocked-precondition"
    rows["exp5755-transition-v514"]["conductor_outcome"] = "FAIL"
    rows["exp5768-v514-capstone-reconciliation"]["complete"] = True
    return rows


def _capstone_payload() -> JsonDict:
    return {
        "schema": "carnot.experiment_5768.v514_capstone_reconciliation.v1",
        "status": "complete",
        "honest_verdict": "complete: V514 reconciled",
        "task_outcome_matrix": _outcome_matrix(),
        "blocked_task_ids": [
            "exp5755-transition-v514",
            "exp5759-sota-exact-proposal-utility-panel",
            "exp5760-selective-exact-feedback-search",
            "exp5767-arc-game-blind-composition-hardening",
        ],
        "scientific_null_task_ids": [
            "exp5765-one-axis-final-10x-crossover",
            "exp5766-arc-loo-component-interaction-audit",
        ],
        "negative_result_task_ids": ["exp5759-sota-exact-proposal-utility-panel"],
        "promoted_task_ids": [
            "exp5757-proposal-benchmark-scalar-bridge",
            "exp5758-rust-parity-scalar-bridge",
            "exp5761-exact-constraint-acquisition-benchmark",
            "exp5762-query-driven-constraint-lifecycle",
            "exp5763-dependent-task-constraint-acquisition",
            "exp5764-one-axis-profiled-allocation-free-hot-path",
        ],
        "proposal_utility_ready": False,
        "constraint_acquisition_ready": True,
        "dependent_task_ca_ready": True,
        "rust_10x_retired": True,
        "arc_loo_generalization_positive": False,
        "arc_composition_executed": False,
    }


def _research_complete_payload(*, duplicate_blocks: int = 1) -> JsonDict:
    block = {
        "id": mod.MILESTONE_FROM,
        "title": "Lossless Evidence Gates, Solver-Certified Constraint Acquisition, Final Rust Crossover, and ARC Generalization",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-21",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items()
        ],
    }
    return {"milestones": [block for _ in range(duplicate_blocks)]}


def _roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {"id": task_id, "deliverable": f"results/{task_id}.json"}
            for task_id in mod.NEXT_TASK_IDS
        ],
    }


def _make_root(root: Path, *, duplicate_blocks: int = 2) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        payload = (
            _capstone_payload()
            if task_id == "exp5768-v514-capstone-reconciliation"
            else _canonical_payload(task_id)
        )
        _write_json(root, rel_path, payload)

    _write_json(
        root,
        "results/experiment_5760_cegis_refinement_induction_ab.json",
        {
            "schema": "carnot.exp5760.cegis_refinement_induction_ab.v1",
            "honest_verdict": "complete_cegis_refinement_partial_pooled_delta_-0.0128",
            "solve_provenance": "development_proxy",
        },
    )
    _write_json(
        root,
        "results/experiment_5764_gemma31b_singleshot_induction_ab.json",
        {
            "schema": "carnot.exp5764.gemma31b_singleshot_induction_ab.v1",
            "honest_verdict": "complete_gemma31b_singleshot_induction_pooled_delta_0.190883",
            "solve_provenance": "development_proxy",
        },
    )
    _write_json(
        root,
        "results/experiment_5766_gemma31b_cegis_refinement_ab.json",
        {
            "schema": "carnot.exp5766.gemma31b_cegis_refinement_ab.v1",
            "honest_verdict": "complete_gemma31b_cegis_refinement_partial_pooled_delta_-0.0598",
            "solve_provenance": "development_proxy",
        },
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(
        root,
        mod.VNEXT_RELATIVE_PATH,
        "# Research Roadmap vNEXT\n\n**Milestone:** `2026.07.515`\n\nTask range: Exp5769-Exp5781\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_research_complete_payload(duplicate_blocks=duplicate_blocks)),
    )
    _write_text(
        root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "| t | Milestone 2026.07.515 activated | OK |\n"
    )
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")


def test_spec_contains_req_report_5769_contract() -> None:
    """REQ-REPORT-5769: OpenSpec names exact deliverables and alias disclosure."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5769") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "exact_declared_deliverable" in section
    assert "SCENARIO-REPORT-5769-COLLISION-BLOCK" in section
    assert "experiment_5760_cegis_refinement_induction_ab.json" in section


def test_scenario_report_5769_archives_terminal_v514_by_declared_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5769: canonical evidence follows declared paths only."""

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
    assert report["milestone_from"] == "2026.07.514"
    assert report["milestone_to"] == "2026.07.515"
    assert report["artifact_selection_policy"] == "exact_declared_deliverable"
    assert report["archived_task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert len(report["declared_deliverable_matrix"]) == 14
    assert report["research_complete_append_count"] == 0
    assert report["docs_reconciled"]["mode"] == (
        "already_archived_preserving_duplicate_history_no_rewrite"
    )
    assert (
        report["canonical_artifact_hashes"]["exp5760-selective-exact-feedback-search"]["path"]
        == "results/experiment_5760_selective_exact_feedback_search.json"
    )
    assert report["same_number_alias_groups"]["5760"]["canonical"]["path"] == (
        "results/experiment_5760_selective_exact_feedback_search.json"
    )
    assert report["same_number_alias_groups"]["5760"]["outer_loop"]["path"] == (
        "results/experiment_5760_cegis_refinement_induction_ab.json"
    )
    assert report["next_range_collision_count"] == 0
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_report_5769_classifies_gate_negative_null_retired_and_positive(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5769: terminal V514 categories are not conflated."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["blocked_task_ids"] == [
        "exp5755-transition-v514",
        "exp5760-selective-exact-feedback-search",
        "exp5767-arc-game-blind-composition-hardening",
    ]
    assert report["gate_skipped_task_ids"] == [
        "exp5760-selective-exact-feedback-search",
        "exp5767-arc-game-blind-composition-hardening",
    ]
    assert report["negative_result_task_ids"] == ["exp5759-sota-exact-proposal-utility-panel"]
    assert report["scientific_null_task_ids"] == ["exp5766-arc-loo-component-interaction-audit"]
    assert report["retired_technique_ids"] == ["exp5765-one-axis-final-10x-crossover"]
    assert "exp5762-query-driven-constraint-lifecycle" in report["positive_result_task_ids"]
    assert "exp5763-dependent-task-constraint-acquisition" in report["positive_result_task_ids"]
    assert "exp5759-sota-exact-proposal-utility-panel" not in report["blocked_task_ids"]
    assert set(report["gate_skipped_task_ids"]).isdisjoint(report["scientific_null_task_ids"])


def test_scenario_report_5769_alias_groups_hash_outer_loop_without_conflation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5769: same-number outer-loop files stay alias evidence."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    for number in ("5760", "5764", "5766"):
        group = report["same_number_alias_groups"][number]
        outer_path = group["outer_loop"]["path"]
        canonical_path = group["canonical"]["path"]
        assert canonical_path != outer_path
        assert group["canonical"]["evidence_role"] == "v514_declared_conductor_task"
        assert group["outer_loop"]["evidence_role"] == "outer_loop_development_proxy_alias"
        assert group["canonical"]["sha256"]
        assert group["outer_loop"]["sha256"]
        assert (
            report["outer_loop_evidence_hashes"][outer_path]["sha256"]
            == group["outer_loop"]["sha256"]
        )


def test_scenario_report_5769_collision_blocks_completion(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5769-COLLISION-BLOCK: occupied next ids fail closed."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5774_existing_collision.json", {"status": "old"})
    _write_text(tmp_path, "docs/occupied.md", "stale reference to exp5781-v515-capstone\n")

    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 2
    assert [row["path"] for row in report["collision_scan"]["preexisting_collisions"]] == [
        "docs/occupied.md",
        "results/experiment_5774_existing_collision.json",
    ]


def test_scenario_report_5769_declared_deliverable_failures_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5769-DECLARED-DELIVERABLE-BLOCK: mapping defects block."""

    _make_root(tmp_path)
    (tmp_path / mod.TASK_ARTIFACT_PATHS["exp5762-query-driven-constraint-lifecycle"]).unlink()
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "blocked"
    assert (
        "missing_or_malformed_declared_deliverables=['exp5762-query-driven-constraint-lifecycle']"
        in report["preconditions_checked"]["failed_preconditions"]
    )

    mismatch_root = tmp_path / "mismatch"
    _make_root(mismatch_root)
    payload = _research_complete_payload()
    payload["milestones"][0]["tasks"][7]["deliverable"] = "results/wrong.json"
    _write_text(mismatch_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))
    mismatch_report = mod.build_report(
        mismatch_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert mismatch_report["status"] == "blocked"
    assert any(
        item.startswith("declared_deliverable_mismatch=")
        for item in mismatch_report["preconditions_checked"]["failed_preconditions"]
    )

    empty_root = tmp_path / "empty"
    _make_root(empty_root)
    _write_text(empty_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump({"milestones": []}))
    empty_report = mod.build_report(
        empty_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert empty_report["status"] == "blocked"
    assert (
        "research_complete_514_block_count=0"
        in empty_report["preconditions_checked"]["failed_preconditions"]
    )


def test_scenario_report_5769_emit_report_and_field_principles(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5769-FIELD-PRINCIPLES: emitted artifact is stable."""

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
    assert mod._research_complete_blocks(tmp_path / "missing") == []
    assert mod._load_tests_run(None)[0]["status"] == "not_run"

    original = mod.FIELD_PRINCIPLES.pop("status")
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
        mod.FIELD_PRINCIPLES["status"] = original


def test_scenario_report_5769_defensive_precondition_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-5769-DECLARED-DELIVERABLE-BLOCK: defensive checks are explicit."""

    assert mod._task_signature({"tasks": "not-list"}) == ()
    assert mod._payload_status({}, {"exists": True, "loadable": False}) == "malformed"

    ambiguous_root = tmp_path / "ambiguous"
    _make_root(ambiguous_root, duplicate_blocks=2)
    payload = _research_complete_payload()
    altered_block = json.loads(json.dumps(payload["milestones"][0]))
    altered_block["tasks"][0]["deliverable"] = "results/other.json"
    payload["milestones"].append(altered_block)
    _write_text(ambiguous_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))
    ambiguous_report = mod.build_report(
        ambiguous_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "ambiguous_research_complete_declared_task_blocks"
        in ambiguous_report["preconditions_checked"]["failed_preconditions"]
    )

    wrong_ids_root = tmp_path / "wrong-ids"
    _make_root(wrong_ids_root)
    wrong_ids_payload = _research_complete_payload()
    wrong_ids_payload["milestones"][0]["tasks"].pop()
    _write_text(
        wrong_ids_root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(wrong_ids_payload),
    )
    wrong_ids_report = mod.build_report(
        wrong_ids_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert any(
        item.startswith("declared_task_ids_mismatch=")
        for item in wrong_ids_report["preconditions_checked"]["failed_preconditions"]
    )

    fallback_root = tmp_path / "fallback"
    _make_root(fallback_root)
    _write_json(
        fallback_root,
        mod.TASK_ARTIFACT_PATHS["exp5768-v514-capstone-reconciliation"],
        _canonical_payload("exp5768-v514-capstone-reconciliation"),
    )
    fallback_report = mod.build_report(
        fallback_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )
    failures = fallback_report["preconditions_checked"]["failed_preconditions"]
    assert any(item.startswith("missing_conductor_outcomes=") for item in failures)
    assert "research_roadmap_modified" in failures
    assert "research_conductor_modified" in failures
    assert (
        fallback_report["conductor_outcomes"]["exp5760-selective-exact-feedback-search"]["outcome"]
        == "GATE_BLOCK"
    )
    assert fallback_report["conductor_outcomes"]["exp5755-transition-v514"]["outcome"] == "FAIL"
    assert fallback_report["positive_result_task_ids"] == list(mod.DEFAULT_POSITIVE_RESULT_TASK_IDS)

    alias_root = tmp_path / "alias"
    _make_root(alias_root)
    (alias_root / "results/experiment_5766_gemma31b_cegis_refinement_ab.json").unlink()
    alias_report = mod.build_report(
        alias_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "missing_or_malformed_alias_groups=['5766']"
        in alias_report["preconditions_checked"]["failed_preconditions"]
    )

    mismatch_root = tmp_path / "roadmap-mismatch"
    _make_root(mismatch_root)
    _write_text(
        mismatch_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump({"milestone": "2026.07.514", "tasks": []}),
    )
    _write_text(mismatch_root, mod.VNEXT_RELATIVE_PATH, "# missing milestone marker\n")
    mismatch_report = mod.build_report(
        mismatch_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "active_roadmap_milestone='2026.07.514'"
        in mismatch_report["preconditions_checked"]["failed_preconditions"]
    )
    assert (
        "vnext_milestone=None" in mismatch_report["preconditions_checked"]["failed_preconditions"]
    )

    scan_root = tmp_path / "scan"
    _make_root(scan_root)
    large_path = scan_root / "notes/large.txt"
    large_path.parent.mkdir(parents=True)
    large_path.write_text("x" * 1_000_001, encoding="utf-8")
    bad_path = scan_root / "notes/bad.txt"
    bad_path.write_text("ordinary text", encoding="utf-8")
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == bad_path:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    scan = mod._collision_scan(scan_root)
    assert scan["preexisting_collision_count"] == 0
