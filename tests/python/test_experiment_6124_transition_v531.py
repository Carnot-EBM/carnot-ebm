"""Tests for the Exp6124 V531 transition receipt.

Spec refs: REQ-REPORT-6124,
SCENARIO-REPORT-6124-ACTIVATED-MATRIX,
SCENARIO-REPORT-6124-TERMINAL-CLASSES,
SCENARIO-REPORT-6124-RETIREMENT-DUPLICATE-DEBT,
SCENARIO-REPORT-6124-RANGE-COLLISION,
SCENARIO-REPORT-6124-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6124_transition_v531 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(task_id: str) -> JsonDict:
    blocked_6116 = {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: "
            "exp6115-phase-d-calibration-pool.phase_d_calibration_ready_score"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp6115-phase-d-calibration-pool",
                "artifact_field": "phase_d_calibration_ready_score",
                "op": "==",
                "expected": 1.0,
                "actual": 0.0,
                "passed": False,
                "reason": "actual=0.0 == expected=1.0",
            }
        ],
    }
    blocked_6118 = {
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: "
            "exp6117-phase-d-headroom-audit.phase_d_headroom_ready_score"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp6117-phase-d-headroom-audit",
                "artifact_field": "phase_d_headroom_ready_score",
                "op": "==",
                "expected": 1.0,
                "actual": None,
                "passed": False,
                "reason": "upstream artifact not found",
            }
        ],
    }
    fixtures: dict[str, JsonDict] = {
        "exp6112-transition-v530": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived exactly four terminal .529 identities",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp6113-v530-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V530 source deltas",
            "inference_substrate": "aggregation_from_external_primary_sources",
        },
        "exp6114-phase-d-gpu-ladder-canary": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: live canary ready",
            "inference_substrate": "live_local_sota_gguf_cuda_generation",
        },
        "exp6115-phase-d-calibration-pool": {
            "status": "complete_null",
            "honest_verdict": "complete_null: no_calibration_stratum_decode_policy_met_gate",
            "inference_substrate": "live_local_sota_gguf_cuda_generation_plus_exact_validation",
            "phase_d_calibration_ready_score": 0.0,
        },
        "exp6116-phase-d-held-candidate-pool": blocked_6116,
        "exp6118-phase-d-per-layer-surface": blocked_6118,
        "exp6120-outcome-committed-reduced-order-csl": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: equal utility lower state pareto",
            "inference_substrate": "deterministic_exact_verifier_and_versioned_external_state_no_llm",
            "outcome_committed_csl_ready_score": 1.0,
            "retirement_triggered": False,
        },
        "exp6121-gatemate-changed-state-gate-v530": {
            "status": "blocked_physical_action",
            "honest_verdict": "blocked_physical_action: unchanged physical state",
            "inference_substrate": "hardware_state_gate_with_optional_non_destructive_detect",
            "physical_state_changed": False,
            "retirement_triggered": True,
        },
        "exp6122-arc-primitive-reachability-loo": {
            "status": "complete_null",
            "honest_verdict": "complete_null: no supported primitive no solve claim",
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "target_level_solve_claim_count": 0,
            "duplicate_level_and_unreachable_solver_credit_counts": {
                "duplicate_level_credit_count": 0,
                "unreachable_solver_credit_count": 0,
            },
        },
        "exp6123-v530-capstone-reconciliation": {
            "status": "complete_with_blocks",
            "honest_verdict": "complete_with_blocks: .530 exact classes preserved",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "prior_failure_same_verdict_retirement_receipts": {
                "triggered_count": 1,
                "receipts": [
                    {
                        "experiment_id": "exp5973-v528-capstone-reconciliation",
                        "retire_if_same_verdict": True,
                        "same_terminal_family": True,
                        "retirement_triggered": True,
                    }
                ],
            },
        },
    }
    return fixtures[task_id]


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "title": title,
                "deliverable": rel_path.as_posix(),
            }
            for task_id, title, rel_path in mod.ACTIVE_V531_TASKS
        ],
    }


def _completion_payload(include_530_blocks: int = 1) -> JsonDict:
    canonical = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-05",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }
    old_duplicate = {
        "id": "2026.07.527",
        "tasks": [{"id": "exp5932-transition-v527", "deliverable": "results/x.json"}],
    }
    return {
        "milestones": [
            deepcopy(old_duplicate),
            deepcopy(old_duplicate),
            *[deepcopy(canonical) for _ in range(include_530_blocks)],
        ]
    }


def _make_root(root: Path, *, include_530_blocks: int = 1) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id in mod.GATE_SKIPPED_MISSING_TASK_IDS:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6115_phase_d_calibration_pool.rows.jsonl",
        {"ignored": "same number but not the declared deliverable"},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "\n".join(
            [
                "# Research Roadmap vNEXT - Milestone 2026.08.531",
                "",
                "**Experiment range:** Exp6124-Exp6137",
                "Exp6124 exact transition",
                "Exp6137 branch-independent capstone and reconciliation",
            ]
        )
        + "\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_530_blocks=include_530_blocks)),
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-04 18:32 UTC | Plan milestone 2026.08.530 | OK | 12 tasks proposed |",
                "| 2026-08-04 18:35 UTC | Milestone 2026.08.530 activated | OK | 12 tasks queued |",
                "| 2026-08-04 21:18 UTC | Gated held authentic same-model Phase-D candidate | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-08-04 21:24 UTC | Gated question-clustered Phase-D authenticity and | GATE_BLOCK | Pre-emptive skip: upstream retired (exp6116-phase-d-held-candidate-pool) |",
                "| 2026-08-04 21:24 UTC | Gated matching-base per-layer hidden-state surface | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-08-04 21:30 UTC | Gated internal-state Phase-D selector against tune | GATE_BLOCK | Pre-emptive skip: upstream retired (exp6118-phase-d-per-layer-surface, exp6117-phase-d-headroom-audit) |",
                "| 2026-08-05 00:11 UTC | Branch-independent .530 capstone and architecture | OK | 87 passed |",
                "| 2026-08-05 01:10 UTC | Plan milestone 2026.08.531 | OK | 9 tasks proposed |",
                "| 2026-08-05 01:13 UTC | Milestone 2026.08.531 activated | OK | 9 tasks queued |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.EVIDENCE_INDEX_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
        mod.E2E_PLAN_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6124\n")


def _receipt(task_id: str) -> JsonDict:
    artifact_path = mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()
    flag_count = 1 if task_id == "exp6115-phase-d-calibration-pool" else 0
    flags = [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}] if flag_count else []
    stdout_json = {
        "reports": [
            {
                "artifact": artifact_path,
                "loaded": True,
                "flag_count": flag_count,
                "max_severity": 1 if flag_count else -1,
                "flags": flags,
            }
        ],
        "flagged_count": flag_count,
    }
    return {
        "task_id": task_id,
        "artifact_path": artifact_path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {artifact_path}",
        "exit_code": 1 if flag_count else 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id)
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id not in mod.GATE_SKIPPED_MISSING_TASK_IDS
    }


def _test_receipts() -> list[JsonDict]:
    rows = [
        {
            "command": f"task-owned {kind}",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": kind,
        }
        for kind in mod.REQUIRED_TASK_OWNED_GATE_KINDS
    ]
    rows.extend(
        [
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "ownership_class": "global_suite",
                "phase": "before",
                "failure_node_ids": ["tests/python/inherited/test_old.py::test_old"],
            },
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "ownership_class": "global_suite",
                "phase": "after",
                "failure_node_ids": ["tests/python/inherited/test_old.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "before",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "after",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": "find . -maxdepth 1 -type f -name '*.py' -print | sort",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "before",
                "root_clutter_paths": ["old_probe.py"],
            },
            {
                "command": "find . -maxdepth 1 -type f -name '*.py' -print | sort",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "after",
                "root_clutter_paths": ["old_probe.py"],
            },
        ]
    )
    return rows


def _build(root: Path, rows: list[JsonDict] | None = None) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts() if rows is None else rows,
        duration_s=1.5,
    )


def test_req_report_6124_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6124: OpenSpec names exact identity, gate-skip, and collision rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6124") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "Exp6117 and Exp6119" in section
    assert "Exp6124 through Exp6137" in section
    for scenario in (
        "SCENARIO-REPORT-6124-ACTIVATED-MATRIX",
        "SCENARIO-REPORT-6124-TERMINAL-CLASSES",
        "SCENARIO-REPORT-6124-RETIREMENT-DUPLICATE-DEBT",
        "SCENARIO-REPORT-6124-RANGE-COLLISION",
        "SCENARIO-REPORT-6124-SCHEMA",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6124_exact_matrix_gate_skips_and_retirement(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6124-ACTIVATED-MATRIX: twelve .530 identities classify once."""

    _make_root(tmp_path, include_530_blocks=2)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.08.530",
        "destination_milestone": "2026.08.531",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 12
    assert matrix["exp6115-phase-d-calibration-pool"]["same_number_aliases_ignored"] == [
        "results/experiment_6115_phase_d_calibration_pool.rows.jsonl"
    ]
    assert matrix["exp6117-phase-d-headroom-audit"]["present"] is False
    assert matrix["exp6117-phase-d-headroom-audit"]["terminal_class"] == (
        "conductor-gate-skipped-missing"
    )
    assert matrix["exp6119-phase-d-hidden-state-selector"]["terminal_class"] == (
        "conductor-gate-skipped-missing"
    )

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["all_activated_terminal"] is True
    assert classes["task_ids_by_terminal_class"]["complete-null"] == [
        "exp6113-v530-source-delta-ingestion",
        "exp6115-phase-d-calibration-pool",
        "exp6122-arc-primitive-reachability-loo",
    ]

    skips = report["structured_gate_skip_receipts"]
    assert skips["structured_gate_skip_task_ids"] == [
        "exp6116-phase-d-held-candidate-pool",
        "exp6117-phase-d-headroom-audit",
        "exp6118-phase-d-per-layer-surface",
        "exp6119-phase-d-hidden-state-selector",
    ]
    assert skips["by_task"]["exp6117-phase-d-headroom-audit"]["reported_as_run"] is False
    assert (
        skips["by_task"]["exp6119-phase-d-hidden-state-selector"]["declared_artifact_present"]
        is False
    )

    retirement = report["retirement_signals_preserved"]
    assert retirement["exp6121_physical_action_retirement"]["retirement_triggered"] is True
    assert retirement["exp6121_physical_action_retirement"]["physical_state_changed"] is False
    assert retirement["same_verdict_retirement_receipts"]["triggered_count"] == 1
    assert retirement["all_retirement_signals_preserved"] is True

    verifier = report["adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 10
    assert verifier["missing_declared_deliverables_not_verified"] == [
        "results/experiment_6117_phase_d_headroom_audit.json",
        "results/experiment_6119_phase_d_hidden_state_selector.json",
    ]
    assert verifier["warning_receipt_task_ids"] == ["exp6115-phase-d-calibration-pool"]
    mod.validate_artifact(report)


def test_scenario_report_6124_append_once_debt_delta_and_collision_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6124-RETIREMENT-DUPLICATE-DEBT: inherited debt is delta-gated."""

    _make_root(tmp_path, include_530_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert report["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"
    debt = report["inherited_debt_baselines_and_deltas"]
    assert debt["global_suite_failure_delta"] == 0
    assert debt["global_spec_gap_delta"] == 0
    assert debt["root_clutter_delta"] == 0
    assert debt["non_amplification_gate_passed"] is True
    assert report["next_task_range"]["start"] == "exp6124"
    assert report["next_task_range"]["end"] == "exp6137"
    assert report["next_range_collision_count"] == 0
    mod.validate_artifact(report)

    absent = tmp_path / "absent"
    _make_root(absent, include_530_blocks=0)
    first = _build(absent)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    second = _build(absent)
    assert second["research_complete_append_count"] == 0

    amplified_rows = deepcopy(_test_receipts())
    for row in amplified_rows:
        if row.get("ownership_class") == "global_suite" and row.get("phase") == "after":
            row["failure_node_ids"].append("tests/python/new/test_owned.py::test_new")
    amplified = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=amplified_rows,
        duration_s=1.5,
    )
    assert amplified["status"] == "blocked"
    assert (
        "global_suite_debt_amplified" in amplified["preconditions_checked"]["failed_preconditions"]
    )

    _write_json(tmp_path, "results/experiment_6137_stale_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_6137_stale_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [6137],
        }
    ]
    mod.validate_artifact(collision)


def test_scenario_report_6124_schema_validation_and_blocked_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6124-SCHEMA: required fields, protection, and checksum hold."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_6124_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "principle": mod.FIELD_PRINCIPLES["docs_reconciled"],
    }
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (
            lambda artifact: artifact.update(inference_substrate="live_llm_inference"),
            "inference_substrate",
        ),
        (lambda artifact: artifact.update(honest_verdict="complete_null: bad"), "honest_verdict"),
        (
            lambda artifact: artifact.update(next_range_collision_count="0"),
            "next_range_collision_count",
        ),
        (
            lambda artifact: artifact.update(research_complete_append_count=2),
            "research_complete_append_count",
        ),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact.update(next_range_collision_count=1, status="complete"),
            "next_range_collision_count must be zero",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp6123-v530-capstone-reconciliation"
            ),
            "exactly twelve",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp6123-v530-capstone-reconciliation": []}
            ),
            "exactly twelve",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp6121-gatemate-changed-state-gate-v530"
            ].update(
                identity=[
                    "2026.08.530",
                    "exp6121-gatemate-changed-state-gate-v530",
                    "wrong.json",
                ]
            ),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp6121-gatemate-changed-state-gate-v530": "complete"}),
            "terminal classes",
        ),
        (
            lambda artifact: artifact["structured_gate_skip_receipts"]["by_task"][
                "exp6117-phase-d-headroom-audit"
            ].update(reported_as_run=True),
            "gate skip",
        ),
        (
            lambda artifact: artifact.update(structured_gate_skip_receipts=[]),
            "gate skip",
        ),
        (
            lambda artifact: artifact["structured_gate_skip_receipts"].update(
                structured_gate_skip_task_ids=[]
            ),
            "gate skip",
        ),
        (
            lambda artifact: artifact["retirement_signals_preserved"][
                "exp6121_physical_action_retirement"
            ].update(retirement_triggered=False),
            "retirement",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=9
            ),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].update(
                command="python other.py"
            ),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].pop(
                "receipt_hash"
            ),
            "adversarial verifier receipt",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["inherited_debt_baselines_and_deltas"].update(
                non_amplification_gate_passed=False
            ),
            "debt non-amplification",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: (
                artifact["protected_files_unchanged"].update(all_unchanged=True),
                next(iter(artifact["protected_files_unchanged"]["files"].values())).update(
                    unchanged=False
                ),
            ),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["duration_s"] = 9.5
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)

    bad = tmp_path / "bad"
    _make_root(bad)
    _write_text(bad, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no activation\n")
    _write_text(bad, mod.SPEC_RELATIVE_PATH, "missing req\n")
    (bad / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    rows = [row for row in _test_receipts() if row.get("suite_kind") != "coverage"]
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda _root, _before: {
            "files": {"research-roadmap.yaml": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_atomic_output_receipt",
        lambda _path: {"declared_path": "x", "atomic_probe_write_ok": False, "ok": False},
    )
    blocked = mod.build_report(
        bad, adversarial_receipts=_receipts(), tests_run=rows, duration_s=1.5
    )
    failed = set(blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "v530_activation_line_missing_or_not_twelve",
        "v531_activation_line_missing_or_not_nine",
        "live_verifier_missing",
        "task_owned_gate_missing",
        "openspec_req_6124_missing",
        "protected_file_modified",
        "atomic_output_unavailable",
    } <= failed

    many = tmp_path / "many"
    _make_root(many)
    bad_roadmap = _active_roadmap_payload()
    bad_roadmap["milestone"] = "2026.08.999"
    bad_roadmap["tasks"] = bad_roadmap["tasks"][:-1]
    bad_roadmap["tasks"][0]["deliverable"] = "wrong.json"
    _write_text(many, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    _write_text(many, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    (many / mod.ROADMAP_DOC_RELATIVE_PATH).unlink()
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp6121-gatemate-changed-state-gate-v530"],
        {
            "status": "blocked_physical_action",
            "honest_verdict": "blocked_physical_action: unchanged physical state",
            "physical_state_changed": False,
            "retirement_triggered": False,
        },
    )
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp6113-v530-source-delta-ingestion")
    sparse_receipts["exp6120-outcome-committed-reduced-order-csl"] = {
        **sparse_receipts["exp6120-outcome-committed-reduced-order-csl"],
        "exit_code": 1,
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 3,
                    "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                }
            ]
        },
    }
    many_rows = _test_receipts()
    many_rows[0]["exit_code"] = 1
    for row in many_rows:
        if row.get("ownership_class") == "spec_coverage" and row.get("phase") == "after":
            row["missing_node_ids"].append("tests/python/new/test_spec.py::test_new")
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_probe.py")
    monkeypatch.setattr(
        mod,
        "_append_completion_if_absent",
        lambda _root, _terminal: {
            "append_count": 0,
            "duplicate_history_amplification_count": 1,
        },
    )
    many_blocked = mod.build_report(
        many,
        adversarial_receipts=sparse_receipts,
        tests_run=many_rows,
        duration_s=1.5,
    )
    many_failed = set(many_blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_milestone_mismatch",
        "active_roadmap_task_ids_mismatch",
        "active_roadmap_deliverables_mismatch",
        "roadmap_next_unloadable",
        "vnext_proposal_missing",
        "terminal_outcomes_not_preserved",
        "retirement_signal_not_preserved",
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
        "task_owned_gate_failed",
        "global_spec_debt_amplified",
        "root_clutter_debt_amplified",
        "duplicate_history_amplified",
    } <= many_failed

    with monkeypatch.context() as context:
        context.setattr(
            mod,
            "_structured_gate_skip_receipts",
            lambda _payloads, _matrix: {
                "structured_gate_skip_task_ids": list(mod.GATE_SKIP_TASK_IDS),
                "by_task": {},
                "missing_gate_skips_without_result_files": [],
                "skipped_branch_reported_as_run_count": 1,
                "principle": mod.FIELD_PRINCIPLES["structured_gate_skip_receipts"],
            },
        )
        gate_bad = mod.build_report(
            tmp_path,
            adversarial_receipts=_receipts(),
            tests_run=_test_receipts(),
            duration_s=1.5,
        )
        assert (
            "gate_skip_reported_as_run" in gate_bad["preconditions_checked"]["failed_preconditions"]
        )


def test_req_report_6124_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6124: helper failures produce explicit blocked receipts."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}
    assert mod.path_sha256(output).startswith("sha256:")
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.sha256_json({"a": 1}) == mod.sha256_bytes(b'{"a":1}')

    _, missing_json_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_json_meta["error"] == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_json_meta = mod._read_json_mapping(bad_json)
    assert bad_json_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_json_meta = mod._read_json_mapping(list_json)
    assert list_json_meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, bad_yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert bad_yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, list_yaml_meta = mod._read_yaml_mapping(list_yaml)
    assert list_yaml_meta["error"] == "yaml_not_mapping"
    assert mod._history_blocks(tmp_path / "missing-root") == []
    assert mod._task_signature({"tasks": "bad"}) == ()
    list_history = tmp_path / "history-list.yaml"
    mod._write_history_blocks(list_history, [], [mod._completion_block_data()])
    assert isinstance(yaml.safe_load(list_history.read_text(encoding="utf-8")), list)
    nonterminal_append = mod._append_completion_if_absent(tmp_path, terminal=False)
    assert nonterminal_append["reason"] == "nonterminal_identity_present"
    malformed = tmp_path / "malformed"
    _make_root(malformed, include_530_blocks=0)
    _write_text(malformed, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    malformed_append = mod._append_completion_if_absent(malformed, terminal=True)
    assert malformed_append["append_count"] == 1

    assert (
        mod._classify_task(
            "exp6117-phase-d-headroom-audit",
            {},
            {"present": False},
            {"latest_status": "GATE_BLOCK"},
        )
        == "conductor-gate-skipped-missing"
    )
    assert mod._classify_task("x", {"status": "complete"}, {"present": True}, {}) == "complete"
    assert mod._classify_task("x", {"status": "blocked"}, {"present": True}, {}) == "blocked"
    assert mod._classify_task("x", {"status": "unknown"}, {"present": True}, {}) == "missing"
    assert mod._same_number_aliases(tmp_path / "no-results", "exp6124-x", Path("x")) == []
    assert mod._range_number_mentions("Exp6124 and experiment_6137") == {6124, 6137}
    assert mod._range_number_mentions("value 0.6127 is not an experiment id") == set()
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {6124})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_DOC_RELATIVE_PATH, {6124, 6137})
        == "allowed_staged_plan_reference"
    )
    assert mod._root_clutter_inventory(tmp_path / "does-not-exist") == []
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flag_count({"flag_count": 4}) == 4
    assert mod._receipt_max_severity({"stdout_json": {"reports": [{"max_severity": None}]}}) == -1
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    grouped = mod._adversarial_receipts_group(
        {
            "exp6112-transition-v530": {
                **_receipt("exp6112-transition-v530"),
                "exit_code": 1,
                "stdout_json": {
                    "reports": [
                        {
                            "artifact": "x",
                            "loaded": True,
                            "flag_count": 1,
                            "max_severity": 3,
                            "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                        }
                    ]
                },
            }
        },
        {
            "exp6112-transition-v530": {
                "present": True,
                "declared_deliverable": "results/experiment_6112_transition_v530.json",
            },
            "exp6113-v530-source-delta-ingestion": {
                "present": True,
                "declared_deliverable": "results/experiment_6113_v530_source_delta_ingestion.json",
            },
            **{
                task_id: {"present": False, "declared_deliverable": path.as_posix()}
                for task_id, path in list(mod.ACTIVATED_TASK_ARTIFACT_PATHS.items())[2:]
            },
        },
    )
    assert grouped["failed_receipt_task_ids"] == ["exp6112-transition-v530"]
    assert grouped["verified_present_declared_deliverable_count"] == 1
    fallback_retirement = mod._retirement_signals(
        {"exp6121-gatemate-changed-state-gate-v530": {}},
        {"exp6121-gatemate-changed-state-gate-v530": {"terminal_class": "missing"}},
    )
    assert fallback_retirement["same_verdict_retirement_receipts"]["triggered_count"] == 0
    assert (
        mod._debt_baselines_and_deltas(
            tmp_path,
            [{"ownership_class": "global_suite", "phase": "after", "failure_node_ids": ["after"]}],
        )["global_suite_failure_delta"]
        == 0
    )
    assert mod._debt_baselines_and_deltas(
        tmp_path,
        [{"ownership_class": "global_suite", "phase": "before", "failure_node_ids": ["before"]}],
    )["global_suite_failure_after_node_ids"] == ["before"]
    assert (
        mod._debt_baselines_and_deltas(
            tmp_path,
            [{"ownership_class": "spec_coverage", "phase": "after", "missing_node_ids": ["after"]}],
        )["global_spec_gap_delta"]
        == 0
    )
    assert mod._debt_baselines_and_deltas(
        tmp_path,
        [{"ownership_class": "spec_coverage", "phase": "before", "missing_node_ids": ["before"]}],
    )["global_spec_gap_after_node_ids"] == ["before"]
    assert (
        mod._debt_baselines_and_deltas(
            tmp_path,
            [
                {
                    "ownership_class": "root_clutter",
                    "phase": "after",
                    "root_clutter_paths": ["after.py"],
                }
            ],
        )["root_clutter_delta"]
        == 0
    )
    assert mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {
                "ownership_class": "root_clutter",
                "phase": "before",
                "root_clutter_paths": ["before.py"],
            }
        ],
    )["root_clutter_after_paths"] == ["before.py"]
    assert mod._debt_baselines_and_deltas(tmp_path, [])["root_clutter_delta"] == 0
    dirty = mod._dirty_worktree_receipt(tmp_path / "not-git")
    assert dirty["git_present"] is False
    repo_dirty = mod._dirty_worktree_receipt(REPO)
    assert repo_dirty["git_present"] is True
    assert repo_dirty["command_exit_code"] == 0
