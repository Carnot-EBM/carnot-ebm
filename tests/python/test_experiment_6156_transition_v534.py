"""Tests for the Exp6156 V534 transition receipt.

Spec refs: REQ-REPORT-6156,
SCENARIO-REPORT-6156-ACTIVATED-MATRIX,
SCENARIO-REPORT-6156-TERMINAL-QUARANTINE-AND-SKIP,
SCENARIO-REPORT-6156-DUPLICATE-ACTIVATION,
SCENARIO-REPORT-6156-RANGE-COLLISION,
SCENARIO-REPORT-6156-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6156_transition_v534 as mod


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
    payloads: dict[str, JsonDict] = {
        "exp6142-transition-v533": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived .532 into .533",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp6143-test-artifact-isolation": {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: focused isolation passed; repo-wide blocked",
            "inference_substrate": "deterministic_infrastructure_test_isolation",
        },
        "exp6144-v533-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V533 source deltas",
            "inference_substrate": "literature_ingestion",
            "accepted_rejected_duplicate_retired_and_abstained_findings": {"accepted_count": 0},
        },
        "exp6145-constraint-shift-stream": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: exact stream",
            "constraint_shift_stream_ready_score": 1.0,
        },
        "exp6146-sota-constraint-event-corpus": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: live corpus",
            "sota_constraint_event_corpus_ready_score": 1.0,
        },
        "exp6147-task-aware-energy-calibration": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: task-aware calibration positive",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
            ],
            "task_aware_energy_calibration_ready_score": 1.0,
        },
        "exp6148-shifted-family-admission-held": {
            "status": "complete_null",
            "honest_verdict": "complete_null: shifted primary metric lower CI not positive",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
            ],
            "shifted_family_admission_ready_score": 0.0,
            "paired_task_aware_minus_global_intervals": {
                "pooled_summary_after_per_model": {
                    "sealed_shifted_family": {
                        "auroc_delta": {"ci95": [0.0, 0.0], "positive_lower_95": False}
                    }
                }
            },
        },
        "exp6149-certified-strategy-schema-fixture": {
            "status": "complete_partial",
            "honest_verdict": "complete_partial: test_commands_clean",
            "certified_strategy_fixture_ready_score": 0.0,
        },
        "exp6150-frozen-qwen-continuous-self-learning-ab": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "2 of 2 gate(s) failed; first failure: "
                "exp6148-shifted-family-admission-held.shifted_family_admission_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp6148-shifted-family-admission-held",
                    "artifact_field": "shifted_family_admission_ready_score",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                }
            ],
        },
        "exp6152-typed-stochastic-constraint-ir": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: exact typed IR",
            "typed_stochastic_ir_ready_score": 1.0,
        },
        "exp6153-thermalized-program-error-audit": {
            "status": "blocked",
            "honest_verdict": "blocked: program-level error composition held but tests failed",
            "thermalized_program_ready_score": 0.0,
            "bound_slack_and_violation_counts": {
                "violation_count": 0,
                "arms": {
                    "isolated": {"measured_joint_tv": 0.0, "precommitted_tv_bound": 0.0},
                    "context_matched": {
                        "measured_joint_tv": 0.0,
                        "precommitted_tv_bound": 0.0,
                    },
                },
            },
            "test_exit_codes": {"JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q": 2},
        },
        "exp6154-arc-task-aware-energy-generalization": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: improved two held games; no solve claim",
            "arc_task_aware_generalization_ready_score": 1.0,
            "solve_claimed": False,
            "offline_reproduced": False,
            "level_credit_delta": 0,
        },
        "exp6155-v533-capstone-reconciliation": {
            "status": "complete_with_blocks_and_quarantine",
            "honest_verdict": (
                "complete: .533 reconciled with blocks, structured skip, nulls, "
                "partials, and adversarial quarantine preserved"
            ),
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "research_complete_append_count": 0,
            "duplicate_history_amplification_count": 0,
        },
    }
    return payloads[task_id]


def _v533_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_FROM,
        "milestone_title": mod.MILESTONE_FROM_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_FROM,
                "title": title,
                "deliverable": rel_path.as_posix(),
            }
            for task_id, title, rel_path in mod.ACTIVATED_TASKS
        ],
    }


def _v534_payload() -> JsonDict:
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
            for task_id, title, rel_path in mod.NEXT_TASKS
        ],
    }


def _completion_payload(include_533_blocks: int = 1) -> JsonDict:
    canonical = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-06",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": title,
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, title, rel_path in mod.ACTIVATED_TASKS
        ],
    }
    duplicate = {"id": "2026.08.500", "tasks": [{"id": "exp5000", "deliverable": "x"}]}
    return {
        "milestones": [
            deepcopy(duplicate),
            deepcopy(duplicate),
            *[deepcopy(canonical) for _ in range(include_533_blocks)],
        ]
    }


def _make_root(
    root: Path,
    *,
    include_533_blocks: int = 1,
    include_next: bool = False,
) -> None:
    for task_id, _title, rel_path in mod.ACTIVATED_TASKS:
        if task_id == mod.STRUCTURED_SKIP_TASK_ID:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6151_same_number_alias.json",
        {"status": "complete_positive", "honest_verdict": "complete_positive: alias"},
    )
    active = _v533_payload() if include_next else _v534_payload()
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(active))
    if include_next:
        _write_text(root, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(_v534_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_533_blocks=include_533_blocks)),
    )
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "\n".join(
            [
                "# Research Roadmap vNEXT - Milestone 2026.08.534",
                "",
                "**Experiment range:** Exp6156-Exp6168",
                "Exp6156 transition",
                "Exp6168 capstone",
            ]
        )
        + "\n",
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-05 13:50 UTC | Milestone 2026.08.533 activated | OK | 14 tasks queued |",
                "| 2026-08-05 17:53 UTC | Exact terminal-boundary handoff from .532 into .53 | OK | 120 passed |",
                "| 2026-08-05 18:21 UTC | Tracked-result test artifact isolation and quarant | OK | 97 passed |",
                "| 2026-08-05 18:38 UTC | Reliable dated evidence refresh after the V533 pla | OK | 86 passed |",
                "| 2026-08-05 20:09 UTC | Exact chronological constraint-event stream with h | OK | 87 passed |",
                "| 2026-08-05 20:48 UTC | Gated on Exp6145 readiness: flagship-GGUF chronolo | OK | 100 passed |",
                "| 2026-08-05 21:10 UTC | Gated on Exp6146 corpus readiness: TOOD-style task | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                "| 2026-08-05 21:33 UTC | Gated on Exp6147 calibration readiness: one-shot s | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
                "| 2026-08-05 22:35 UTC | Gated on Exp6145 stream readiness: certified strat | OK | 89 passed |",
                "| 2026-08-05 22:11 UTC | Gated on Exp6148 and Exp6149 readiness: frozen-Qwe | GATE_BLOCK | 2 of 2 gate(s) failed |",
                "| 2026-08-05 23:23 UTC | Gated on Exp6150 positive utility: default-off tra | GATE_BLOCK | Pre-emptive skip: upstream retired |",
                "| 2026-08-05 22:35 UTC | Gated on Exp6145 stream readiness: typed Torx-comp | OK | 89 passed |",
                "| 2026-08-06 01:18 UTC | Gated on Exp6152 IR readiness: software thermaliza | FAIL | artifact_not_updated_past_bootstrap |",
                "| 2026-08-06 02:51 UTC | ARC live-path adapter-disabled task-aware energy g | OK | 108 passed |",
                "| 2026-08-06 05:46 UTC | Branch-independent .533 capstone, adversarial veri | OK | 87 passed |",
                "| 2026-08-06 06:48 UTC | Plan milestone 2026.08.534 | OK | 13 tasks proposed |",
                "| 2026-08-06 06:50 UTC | Milestone 2026.08.534 activated | OK | 13 tasks queued |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\n")
    for rel_path in set(mod.PROTECTED_FILE_PATHS + mod.PRECONDITION_CONTEXT_PATHS):
        if rel_path == mod.ACTIVATED_TASK_PATHS[mod.STRUCTURED_SKIP_TASK_ID]:
            continue
        if rel_path == mod.ROADMAP_NEXT_RELATIVE_PATH and not include_next:
            continue
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6156\n")


def _receipt(task_id: str, rel_path: Path) -> JsonDict:
    flagged = task_id in {
        "exp6147-task-aware-energy-calibration",
        "exp6148-shifted-family-admission-held",
    }
    report = {
        "artifact": rel_path.as_posix(),
        "loaded": True,
        "flag_count": 1 if flagged else 0,
        "max_severity": 3 if flagged else -1,
        "flags": (
            [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}]
            if flagged
            else []
        ),
    }
    stdout_json = {"reports": [report], "flagged_count": int(flagged)}
    return {
        "task_id": task_id,
        "artifact_path": rel_path.as_posix(),
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {rel_path.as_posix()}",
        "exit_code": int(flagged),
        "stdout_json": stdout_json,
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id, rel_path)
        for task_id, _title, rel_path in mod.ACTIVATED_TASKS
        if task_id != mod.STRUCTURED_SKIP_TASK_ID
    }


def _test_receipts() -> list[JsonDict]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_6156_transition_v534.py -q --no-cov -n 0",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kinds": [
                "unit",
                "yaml_parse",
                "exact_path",
                "terminal_quarantine",
                "duplicate_history",
                "activation",
                "exclusion_manifest",
                "range_collision",
                "adversarial_verifier",
                "protected_file",
                "applicable_e2e",
                "no_new_root_clutter",
            ],
        },
        {
            "command": ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6156_transition_v534.py -m pytest tests/python/test_experiment_6156_transition_v534.py -q --no-cov -n 0",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "coverage",
        },
        {
            "command": ".venv/bin/python scripts/check_spec_coverage.py",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": "spec_coverage",
        },
        {
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": 0,
            "ownership_class": "global_suite",
        },
        {
            "command": "find . -maxdepth 1 -type f -name '*.py' -print | sort",
            "exit_code": 0,
            "ownership_class": "root_clutter",
            "phase": "before",
            "root_clutter_paths": [],
        },
        {
            "command": "find . -maxdepth 1 -type f -name '*.py' -print | sort",
            "exit_code": 0,
            "ownership_class": "root_clutter",
            "phase": "after",
            "root_clutter_paths": [],
        },
    ]


def _build(root: Path, rows: list[JsonDict] | None = None) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts() if rows is None else rows,
        duration_s=1.25,
    )


def test_req_report_6156_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6156: OpenSpec names exact identity and transition rules."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-6156") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "Exp6142 through Exp6155" in section
    assert "Exp6156 through Exp6168" in section
    for scenario in (
        "SCENARIO-REPORT-6156-ACTIVATED-MATRIX",
        "SCENARIO-REPORT-6156-TERMINAL-QUARANTINE-AND-SKIP",
        "SCENARIO-REPORT-6156-DUPLICATE-ACTIVATION",
        "SCENARIO-REPORT-6156-RANGE-COLLISION",
        "SCENARIO-REPORT-6156-SCHEMA",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6156_matrix_terminal_quarantine_and_skip(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6156-ACTIVATED-MATRIX: fourteen declared identities classify once."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.08.533",
        "destination_milestone": "2026.08.534",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == [task_id for task_id, _title, _path in mod.ACTIVATED_TASKS]
    assert len(matrix) == 14
    assert matrix["exp6155-v533-capstone-reconciliation"]["present"] is True
    assert matrix[mod.STRUCTURED_SKIP_TASK_ID]["present"] is False
    assert matrix[mod.STRUCTURED_SKIP_TASK_ID]["terminal_class"] == "skip"
    assert matrix[mod.STRUCTURED_SKIP_TASK_ID]["same_number_aliases_ignored"] == [
        "results/experiment_6151_same_number_alias.json"
    ]
    assert matrix[mod.STRUCTURED_SKIP_TASK_ID]["same_number_alias_used"] is False

    classes = report["exact_terminal_classification"]
    assert classes["all_activated_terminal"] is True
    assert (
        classes["underlying_terminal_class_by_task_id"]["exp6148-shifted-family-admission-held"]
        == "null"
    )
    assert (
        classes["underlying_terminal_class_by_task_id"][
            "exp6150-frozen-qwen-continuous-self-learning-ab"
        ]
        == "block"
    )
    assert (
        classes["underlying_terminal_class_by_task_id"]["exp6153-thermalized-program-error-audit"]
        == "block"
    )
    assert classes["terminal_class_by_task_id"]["exp6147-task-aware-energy-calibration"] == (
        "flagged"
    )
    assert classes["terminal_class_by_task_id"]["exp6148-shifted-family-admission-held"] == (
        "flagged"
    )
    assert classes["task_ids_by_terminal_class"]["skip"] == [mod.STRUCTURED_SKIP_TASK_ID]

    verifier = report["adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 13
    assert verifier["missing_declared_deliverables_not_verified"] == [
        "results/experiment_6151_strategy_memory_shadow_adapter.json"
    ]
    assert verifier["flagged_task_ids"] == [
        "exp6147-task-aware-energy-calibration",
        "exp6148-shifted-family-admission-held",
    ]

    quarantine = report["quarantine_and_null_preservation_receipts"]
    assert quarantine["quarantined_task_ids"] == [
        "exp6147-task-aware-energy-calibration",
        "exp6148-shifted-family-admission-held",
    ]
    assert quarantine["exp6147_flag_preserved"]["underlying_terminal_class"] == "positive"
    assert quarantine["exp6148_null_preserved"]["underlying_terminal_class"] == "null"
    assert quarantine["exp6148_null_preserved"]["archive_terminal_class"] == "flagged"
    assert quarantine["exp6148_null_preserved"]["diagnostic_fields_promoted"] is False
    assert quarantine["exp6150_block_preserved"]["terminal_class"] == "block"
    assert quarantine["exp6153_zero_error_block_preserved"]["terminal_class"] == "block"
    assert quarantine["exp6153_zero_error_block_preserved"]["violation_count"] == 0
    assert quarantine["exp6154_arc_no_solve_preserved"]["solve_claimed"] is False

    skip = report["structured_gate_skip_receipts"]
    assert skip["task_id"] == mod.STRUCTURED_SKIP_TASK_ID
    assert skip["declared_artifact_present"] is False
    assert skip["reported_as_run"] is False
    assert skip["same_number_alias_used"] is False
    mod.validate_artifact(report)


def test_scenario_report_6156_append_once_activation_and_collision_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6156-DUPLICATE-ACTIVATION: history and activation are idempotent."""

    _make_root(tmp_path, include_533_blocks=0)
    first = _build(tmp_path)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    assert first["staged_roadmap_activation_receipt"]["mode"] == "already_active"
    assert first["staged_roadmap_activation_receipt"]["active_roadmap_task_count"] == 13
    assert first["next_task_range"] == {
        "start": "exp6156",
        "end": "exp6168",
        "reserved_count": 13,
        "principle": mod.FIELD_PRINCIPLES["next_task_range"],
    }
    assert first["next_range_collision_count"] == 0

    second = _build(tmp_path)
    assert second["research_complete_append_count"] == 0
    assert second["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"

    staged = tmp_path / "staged"
    _make_root(staged, include_next=True)
    staged_report = _build(staged)
    assert staged_report["staged_roadmap_activation_receipt"]["mode"] == "copied_staged_roadmap"
    assert staged_report["staged_roadmap_activation_receipt"]["copied_exactly"] is True
    assert yaml.safe_load((staged / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")) == (
        yaml.safe_load((staged / mod.ROADMAP_NEXT_RELATIVE_PATH).read_text(encoding="utf-8"))
    )

    _write_json(tmp_path, "results/experiment_6168_unowned_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_6168_unowned_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [6168],
        }
    ]
    mod.validate_artifact(collision)


def test_scenario_report_6156_schema_validation_and_blocked_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6156-SCHEMA: required fields, protection, and checksum hold."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_6156_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "principle": mod.FIELD_PRINCIPLES["docs_reconciled"],
    }
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["preexisting_worktree_changes_preserved"]["preserved"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)

    written = mod.run(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=report["test_exit_codes"],
        duration_s=1.5,
    )
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (
            lambda artifact: artifact.update(inference_substrate="live_llm_inference"),
            "inference_substrate",
        ),
        (lambda artifact: artifact.update(honest_verdict="retired: bad"), "honest_verdict"),
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
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp6155-v533-capstone-reconciliation"
            ),
            "exactly fourteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp6155-v533-capstone-reconciliation": []}
            ),
            "exactly fourteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp6142-transition-v533"
            ].update(identity=["2026.08.533", "exp6142-transition-v533", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "underlying_terminal_class_by_task_id"
            ].update({"exp6148-shifted-family-admission-held": "positive"}),
            "Exp6148 null",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp6148-shifted-family-admission-held": "positive"}),
            "Exp6148 quarantine",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp6147-task-aware-energy-calibration": "positive"}),
            "quarantine",
        ),
        (
            lambda artifact: artifact.update(exact_terminal_classification=[]),
            "terminal classification",
        ),
        (
            lambda artifact: artifact.update(
                exact_terminal_classification={"terminal_class_by_task_id": {}}
            ),
            "terminal classification mappings",
        ),
        (
            lambda artifact: artifact.update(quarantine_and_null_preservation_receipts=[]),
            "quarantine receipts",
        ),
        (
            lambda artifact: artifact["quarantine_and_null_preservation_receipts"].pop(
                "exp6148_null_preserved"
            ),
            "Exp6148 null receipt",
        ),
        (
            lambda artifact: artifact["quarantine_and_null_preservation_receipts"][
                "exp6148_null_preserved"
            ].update(diagnostic_fields_promoted=True),
            "diagnostic",
        ),
        (
            lambda artifact: artifact["structured_gate_skip_receipts"].update(reported_as_run=True),
            "structured gate skip",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=12
            ),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["staged_roadmap_activation_receipt"].update(activated=False),
            "activation",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                task_owned_failures=[{"command": "bad", "exit_code": 1}]
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: artifact.update(field_provenance=[]),
            "field provenance",
        ),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing",
        ),
        (
            lambda artifact: artifact.update(next_range_collision_count=1, status="complete"),
            "next_range_collision_count must be zero",
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
            "files": {"scripts/research_conductor.py": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    blocked = mod.build_report(
        bad, adversarial_receipts=_receipts(), tests_run=rows, duration_s=1.25
    )
    failed = set(blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "v533_activation_line_missing_or_not_fourteen",
        "v534_activation_line_missing_or_not_thirteen",
        "live_verifier_missing",
        "task_owned_gate_missing",
        "openspec_req_6156_missing",
        "protected_file_modified",
    } <= failed

    precondition_branches = deepcopy(report)
    precondition_branches["quarantine_and_null_preservation_receipts"]["exp6147_flag_preserved"][
        "archive_terminal_class"
    ] = "positive"
    precondition_branches["quarantine_and_null_preservation_receipts"]["exp6148_null_preserved"][
        "underlying_terminal_class"
    ] = "positive"
    precondition_branches["quarantine_and_null_preservation_receipts"]["exp6148_null_preserved"][
        "archive_terminal_class"
    ] = "null"
    precondition_branches["quarantine_and_null_preservation_receipts"]["exp6148_null_preserved"][
        "diagnostic_fields_promoted"
    ] = True
    precondition_branches["adversarial_verifier_receipts"][
        "verified_present_declared_deliverable_count"
    ] = 12
    precondition_branches["research_complete_append_receipt"][
        "duplicate_history_amplification_count"
    ] = 1
    precondition_branches["task_owned_gate_receipts"]["task_owned_failures"] = [
        {"command": "bad", "exit_code": 1}
    ]
    precondition_branches["root_clutter_delta_count"] = 1
    assert {
        "exp6147_quarantine_not_preserved",
        "exp6148_null_not_preserved",
        "exp6148_quarantine_not_preserved",
        "diagnostic_fields_promoted",
        "missing_adversarial_receipts",
        "duplicate_history_amplified",
        "task_owned_gate_failed",
        "root_clutter_debt_amplified",
    } <= set(mod._failed_preconditions(precondition_branches))


def test_req_report_6156_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6156: helper failures produce explicit blocked receipts."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}
    assert mod.path_sha256(output).startswith("sha256:")
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.sha256_json({"a": 1}) == mod.sha256_bytes(b'{"a":1}')

    _, missing_json_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_json_meta["error"] == "missing"
    _write_text(tmp_path, "bad.json", "{")
    _, bad_json_meta = mod._read_json_mapping(tmp_path / "bad.json")
    assert bad_json_meta["error"].startswith("json_error:")
    _write_text(tmp_path, "array.json", "[]")
    _, array_meta = mod._read_json_mapping(tmp_path / "array.json")
    assert array_meta["error"] == "json_not_mapping"

    _write_text(tmp_path, "bad.yaml", "a: [\n")
    _, bad_yaml_meta = mod._read_yaml_mapping(tmp_path / "bad.yaml")
    assert bad_yaml_meta["error"].startswith("yaml_error:")
    _write_text(tmp_path, "array.yaml", "- a\n")
    _, array_yaml_meta = mod._read_yaml_mapping(tmp_path / "array.yaml")
    assert array_yaml_meta["error"] == "yaml_not_mapping"
    assert mod._history_blocks(tmp_path / "missing-root") == []
    assert mod._task_signature({"tasks": "bad"}) == ()
    history_list = tmp_path / "history-list.yaml"
    mod._write_history_blocks(history_list, [], [mod._completion_block_data()])
    assert isinstance(yaml.safe_load(history_list.read_text(encoding="utf-8")), list)
    nonterminal_append = mod._append_completion_if_absent(tmp_path, terminal=False)
    assert nonterminal_append["reason"] == "nonterminal_identity_present"
    malformed = tmp_path / "malformed"
    _make_root(malformed, include_533_blocks=0)
    _write_text(malformed, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    malformed_append = mod._append_completion_if_absent(malformed, terminal=True)
    assert malformed_append["append_count"] == 1
    assert mod._terminal_marker("complete_ready: good") == "positive"
    assert mod._terminal_marker("complete_null: no") == "null"
    assert mod._terminal_marker("blocked: gate") == "block"
    assert mod._terminal_marker("unknown") is None
    assert mod._classify_underlying({}, False, {"latest_status": "GATE_BLOCK"}) == "skip"
    assert mod._classify_underlying({}, False, {}) == "missing"
    assert mod._classify_underlying({"status": "retired"}, True, {}) == "retired"
    assert mod._classify_underlying({"retirement_triggered": True}, True, {}) == "retired"
    assert mod._task_number("exp6156-transition-v534") == 6156
    assert mod._same_number_aliases(tmp_path / "none", "exp6151-x", Path("x")) == []
    assert mod._range_number_mentions("Exp6156 and experiment_6168") == {6156, 6168}
    assert mod._range_number_mentions("value 0.6156 is not an experiment id") == set()
    assert mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {6156}) == (
        "transition_owned_reference"
    )
    assert mod._allowed_range_reference_kind(mod.ROADMAP_RELATIVE_PATH, {6168}) == (
        "canonical_v534_plan_reference"
    )
    assert mod._allowed_range_reference_kind(mod.ROADMAP_DOC_RELATIVE_PATH, {6156, 6168}) == (
        "vnext_v534_proposal_reference"
    )
    assert mod._root_clutter_inventory(tmp_path / "does-not-exist") == []
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    assert mod._dirty_worktree_receipt(tmp_path / "not-git")["git_present"] is False
    (tmp_path / ".git").mkdir()
    assert mod._dirty_worktree_receipt(tmp_path)["git_present"] is True
    verifier = mod._adversarial_receipts_group(
        {},
        {
            task_id: {
                "present": task_id == "exp6142-transition-v533",
                "declared_deliverable": rel_path.as_posix(),
                "artifact_quarantine_fields_present": False,
            }
            for task_id, _title, rel_path in mod.ACTIVATED_TASKS
        },
    )
    assert verifier["verified_present_declared_deliverable_count"] == 0
    staged_unloadable = tmp_path / "staged-unloadable"
    _make_root(staged_unloadable)
    _write_text(staged_unloadable, mod.ROADMAP_NEXT_RELATIVE_PATH, "- not mapping\n")
    assert mod._activate_staged_roadmap(staged_unloadable)["mode"] == "staged_unloadable"
    staged_mismatch = tmp_path / "staged-mismatch"
    _make_root(staged_mismatch, include_next=True)
    bad_next = _v534_payload()
    bad_next["milestone"] = "2026.08.999"
    _write_text(staged_mismatch, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(bad_next))
    assert mod._activate_staged_roadmap(staged_mismatch)["mode"] == "staged_milestone_mismatch"
    assert (
        mod._root_clutter_delta(
            [{"ownership_class": "root_clutter", "phase": "after", "root_clutter_paths": ["x.py"]}]
        )
        == 0
    )
    assert (
        mod._root_clutter_delta(
            [{"ownership_class": "root_clutter", "phase": "before", "root_clutter_paths": ["x.py"]}]
        )
        == 0
    )
    assert (
        mod._root_clutter_delta(
            [
                {"ownership_class": "root_clutter", "phase": "before", "root_clutter_paths": []},
                {
                    "ownership_class": "root_clutter",
                    "phase": "after",
                    "root_clutter_paths": ["x.py"],
                },
            ]
        )
        == 1
    )
