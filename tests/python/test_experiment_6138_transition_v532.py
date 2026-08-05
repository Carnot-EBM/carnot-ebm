"""Tests for the Exp6138 V532 transition receipt.

Spec refs: REQ-REPORT-6138,
SCENARIO-REPORT-6138-ACTIVATED-MATRIX,
SCENARIO-REPORT-6138-TERMINAL-CLASSES,
SCENARIO-REPORT-6138-DUPLICATE-ACTIVATION,
SCENARIO-REPORT-6138-RANGE-COLLISION,
SCENARIO-REPORT-6138-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6138_transition_v532 as mod


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
    fixtures: dict[str, JsonDict] = {
        "exp6124-transition-v531": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived exactly twelve terminal .530 identities",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp6126-phase-d-exp6115-transport-forensics": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: transport failure supports native chat",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "retirement_triggered": False,
        },
        "exp6127-phase-d-native-chat-transport-canary": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: native chat transport canary passed",
            "inference_substrate": "live_local_sota_gguf_cuda_native_chat_transport_canary",
            "model_native_transport_ready_score": 1.0,
            "retirement_triggered": False,
        },
        "exp6128-phase-d-calibration-pool-v2": {
            "status": "complete_null",
            "honest_verdict": "complete_null: no_calibration_policy_met_conjunctive_gates",
            "inference_substrate": "live_local_sota_gguf_cuda_native_chat_calibration_pool_v2",
            "phase_d_calibration_ready_score": 0.0,
            "retirement_triggered": False,
            "effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics": {
                "overall": {
                    "all_wrong_rate": 0.1,
                    "mean_effective_k": 7.988889,
                    "oracle_at_k": 0.9,
                    "oracle_minus_tuned_sc": 0.211111,
                    "question_count": 90,
                    "tuned_sc_accuracy": 0.688889,
                }
            },
            "family_stratum_shortcut_relabel_and_answer_cluster_metrics": {
                "by_family": {
                    "finite_domain_scheduling": {
                        "all_wrong_rate": 0.0,
                        "oracle_at_k": 1.0,
                        "question_count": 30,
                        "tuned_sc_accuracy": 1.0,
                    },
                    "logic_grid": {
                        "all_wrong_rate": 0.0,
                        "oracle_at_k": 1.0,
                        "question_count": 30,
                        "tuned_sc_accuracy": 1.0,
                    },
                    "typed_finite_choice": {
                        "all_wrong_rate": 0.3,
                        "oracle_at_k": 0.7,
                        "question_count": 30,
                        "tuned_sc_accuracy": 0.066667,
                    },
                }
            },
            "per_candidate_accuracy_clustered_intervals_parseability_method_validity": {
                "by_family": {
                    "finite_domain_scheduling": {"accuracy": 1.0},
                    "logic_grid": {"accuracy": 1.0},
                    "typed_finite_choice": {"accuracy": 0.170833},
                },
                "overall": {"accuracy": 0.723611},
            },
        },
        "exp6129-phase-d-held-pool-v2": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp6128-phase-d-calibration-pool-v2.phase_d_calibration_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp6128-phase-d-calibration-pool-v2",
                    "artifact_field": "phase_d_calibration_ready_score",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                }
            ],
        },
        "exp6131-phase-d-mid-layer-selector-calibration": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp6130-phase-d-per-layer-surface-v2.per_layer_surface_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp6130-phase-d-per-layer-surface-v2",
                    "artifact_field": "per_layer_surface_ready_score",
                    "expected": 1.0,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
    }
    return fixtures[task_id]


def _v532_task(task_id: str, deliverable: str) -> JsonDict:
    return {
        "id": task_id,
        "milestone": mod.MILESTONE_TO,
        "title": task_id,
        "deliverable": deliverable,
    }


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            _v532_task("exp6138-transition-v532", mod.RESULT_RELATIVE_PATH.as_posix()),
            _v532_task(
                "exp6139-v532-source-delta-ingestion",
                "results/experiment_6139_v532_source_delta_ingestion.json",
            ),
            _v532_task(
                "exp6140-phase-d-exp6128-option-psychometrics",
                "results/experiment_6140_phase_d_exp6128_option_psychometrics.json",
            ),
            _v532_task(
                "exp6141-phase-d-empirical-item-bank",
                "results/experiment_6141_phase_d_empirical_item_bank.json",
            ),
        ],
    }


def _staged_roadmap_payload() -> JsonDict:
    payload = deepcopy(_active_roadmap_payload())
    payload["tasks"].extend(
        [
            _v532_task("exp6142-phase-d-two-model-irt-pilot", "results/experiment_6142.json"),
            _v532_task("exp6143-phase-d-held-pool-v3", "results/experiment_6143.json"),
            _v532_task("exp6144-phase-d-spectral-surface", "results/experiment_6144.json"),
            _v532_task("exp6145-phase-d-selector-calibration", "results/experiment_6145.json"),
            _v532_task("exp6146-phase-d-held-selector-eval", "results/experiment_6146.json"),
            _v532_task("exp6147-csl-certified-strategy-fixture", "results/experiment_6147.json"),
            _v532_task("exp6148-csl-frozen-gguf-ab", "results/experiment_6148.json"),
            _v532_task("exp6149-csl-shadow-adapter", "results/experiment_6149.json"),
            _v532_task("exp6150-arc-change-verifier-generalization", "results/experiment_6150.json"),
            _v532_task("exp6151-v532-capstone-reconciliation", "results/experiment_6151.json"),
        ]
    )
    return payload


def _completion_payload(include_531_blocks: int = 1) -> JsonDict:
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
        "id": "2026.07.510",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    return {
        "milestones": [
            deepcopy(old_duplicate),
            deepcopy(old_duplicate),
            *[deepcopy(canonical) for _ in range(include_531_blocks)],
        ]
    }


def _make_root(root: Path, *, include_531_blocks: int = 1, include_next: bool = False) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id in {
            "exp6125-v531-source-delta-ingestion",
            "exp6130-phase-d-per-layer-surface-v2",
            "exp6132-phase-d-hidden-state-held-eval",
        }:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, "results/experiment_6128_phase_d_calibration_pool_v2.rows.jsonl", "{}\n")
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    if include_next:
        _write_text(root, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(_staged_roadmap_payload()))
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "\n".join(
            [
                "# Research Roadmap vNEXT - Milestone 2026.08.532",
                "",
                "**Experiment range:** Exp6138-Exp6151",
                "proposal-only Exp6133-Exp6137 are not activated .531 experiments.",
                "Exp6140-Exp6142 use question-clustered uncertainty.",
                "Exp6151 capstone.",
            ]
        )
        + "\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_531_blocks=include_531_blocks)),
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-05 01:10 UTC | Plan milestone 2026.08.531 | OK | 9 tasks proposed |",
                "| 2026-08-05 01:13 UTC | Milestone 2026.08.531 activated | OK | 9 tasks queued |",
                "| 2026-08-05 02:59 UTC | Exact terminal-boundary handoff from .530 into .53 | OK | 86 passed |",
                "| 2026-08-05 03:02 UTC | Dated evidence refresh after the V531 planner mark | FAIL | Codex CLI error: Model metadata for `gemini-3.1-pro-preview` not found. |",
                "| 2026-08-05 03:04 UTC | Dated evidence refresh after the V531 planner mark | FAIL | Codex CLI error: Model metadata for `gemini-3.1-pro-preview` not found. |",
                "| 2026-08-05 03:06 UTC | Dated evidence refresh after the V531 planner mark | FAIL | Codex CLI error: Model metadata for `gemini-3.1-pro-preview` not found. |",
                "| 2026-08-05 03:25 UTC | Exp6115 model-transport forensics and frozen v2 sp | OK | 87 passed |",
                "| 2026-08-05 04:06 UTC | Gated on Exp6126 justification: model-native Phase | OK | 87 passed |",
                "| 2026-08-05 05:21 UTC | Gated on Exp6127 readiness: model-native Phase-D c | OK | 88 passed |",
                "| 2026-08-05 05:27 UTC | Gated on Exp6128 readiness: sealed held Phase-D po | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-08-05 05:33 UTC | Gated on Exp6129 headroom: authenticated matching- | GATE_BLOCK | Pre-emptive skip: upstream retired (exp6129-phase-d-held-pool-v2) |",
                "| 2026-08-05 05:33 UTC | Gated on Exp6130 surface: calibration-only mid-lay | GATE_BLOCK | 1 of 1 gate(s) failed |",
                "| 2026-08-05 06:48 UTC | Gated on Exp6131 readiness: frozen held mid-layer | GATE_BLOCK | Pre-emptive skip: upstream retired (exp6131-phase-d-mid-layer-selector-calibration) |",
                "| 2026-08-05 06:46 UTC | Plan milestone 2026.08.532 | OK | 4 tasks proposed |",
                "| 2026-08-05 06:48 UTC | Milestone 2026.08.532 activated | OK | 4 tasks queued |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\nretired_experiments: []\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.EVIDENCE_INDEX_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
        mod.E2E_PLAN_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6138\n")


def _receipt(task_id: str) -> JsonDict:
    artifact_path = mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()
    stdout_json = {
        "reports": [
            {
                "artifact": artifact_path,
                "loaded": True,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            }
        ],
        "flagged_count": 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": artifact_path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {artifact_path}",
        "exit_code": 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id)
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id
        not in {
            "exp6125-v531-source-delta-ingestion",
            "exp6130-phase-d-per-layer-surface-v2",
            "exp6132-phase-d-hidden-state-held-eval",
        }
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
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "before",
                "failure_node_ids": ["tests/python/inherited/test_old.py::test_old"],
            },
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
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


def test_req_report_6138_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6138: OpenSpec names identity, no-artifact, activation, and collision rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6138") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "Exp6125" in section
    assert "no-artifact-backend-failure" in section
    assert "Exp6133 through Exp6137" in section
    assert "Exp6138 through Exp6151" in section
    for scenario in (
        "SCENARIO-REPORT-6138-ACTIVATED-MATRIX",
        "SCENARIO-REPORT-6138-TERMINAL-CLASSES",
        "SCENARIO-REPORT-6138-DUPLICATE-ACTIVATION",
        "SCENARIO-REPORT-6138-RANGE-COLLISION",
        "SCENARIO-REPORT-6138-SCHEMA",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6138_matrix_no_artifact_gate_skips_and_bimodality(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6138-ACTIVATED-MATRIX: nine .531 identities classify once."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.08.531",
        "destination_milestone": "2026.08.532",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 9
    assert matrix["exp6125-v531-source-delta-ingestion"]["present"] is False
    assert matrix["exp6125-v531-source-delta-ingestion"]["terminal_class"] == (
        "no-artifact-backend-failure"
    )
    assert matrix["exp6128-phase-d-calibration-pool-v2"]["same_number_aliases_ignored"] == [
        "results/experiment_6128_phase_d_calibration_pool_v2.rows.jsonl"
    ]
    assert matrix["exp6130-phase-d-per-layer-surface-v2"]["terminal_class"] == (
        "conductor-gate-skipped-missing"
    )
    assert matrix["exp6132-phase-d-hidden-state-held-eval"]["terminal_class"] == (
        "conductor-gate-skipped-missing"
    )

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["all_activated_terminal"] is True
    assert classes["task_ids_by_terminal_class"]["complete-ready"] == [
        "exp6126-phase-d-exp6115-transport-forensics",
        "exp6127-phase-d-native-chat-transport-canary",
    ]
    assert classes["task_ids_by_terminal_class"]["complete-null"] == [
        "exp6128-phase-d-calibration-pool-v2"
    ]

    backend = report["no_artifact_backend_failure_receipt"]
    assert backend["task_id"] == "exp6125-v531-source-delta-ingestion"
    assert backend["attempt_count"] == 3
    assert backend["declared_artifact_present"] is False
    assert backend["no_artifact_invented"] is True
    assert backend["backend_failure_authenticated"] is True

    skips = report["structured_gate_skip_receipts"]
    assert skips["structured_gate_skip_task_ids"] == [
        "exp6129-phase-d-held-pool-v2",
        "exp6130-phase-d-per-layer-surface-v2",
        "exp6131-phase-d-mid-layer-selector-calibration",
        "exp6132-phase-d-hidden-state-held-eval",
    ]
    assert skips["by_task"]["exp6129-phase-d-held-pool-v2"]["reported_as_run"] is False
    assert skips["by_task"]["exp6130-phase-d-per-layer-surface-v2"]["declared_artifact_present"] is False
    assert skips["missing_gate_skips_without_result_files"] == [
        "exp6130-phase-d-per-layer-surface-v2",
        "exp6132-phase-d-hidden-state-held-eval",
    ]

    excluded = report["proposal_only_identities_excluded"]
    assert excluded["proposal_only_task_ids"] == [
        "exp6133",
        "exp6134",
        "exp6135",
        "exp6136",
        "exp6137",
    ]
    assert excluded["all_excluded_from_activated_matrix"] is True

    bimodality = report["exp6128_family_bimodality_preserved"]
    assert bimodality["preserved"] is True
    assert bimodality["phase_d_calibration_ready_score"] == 0.0
    assert bimodality["overall_accuracy"] == 0.723611
    assert bimodality["family_accuracy"]["typed_finite_choice"] == 0.170833
    assert bimodality["saturated_family_task_ids"] == [
        "finite_domain_scheduling",
        "logic_grid",
    ]

    verifier = report["adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 6
    assert verifier["missing_declared_deliverables_not_verified"] == [
        "results/experiment_6125_v531_source_delta_ingestion.json",
        "results/experiment_6130_phase_d_per_layer_surface_v2.json",
        "results/experiment_6132_phase_d_hidden_state_held_eval.json",
    ]
    mod.validate_artifact(report)


def test_scenario_report_6138_append_once_activation_and_collision_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6138-DUPLICATE-ACTIVATION: history and activation are idempotent."""

    _make_root(tmp_path, include_531_blocks=0)
    first = _build(tmp_path)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    assert first["staged_roadmap_activation_receipt"]["mode"] == "already_active"
    assert first["staged_roadmap_activation_receipt"]["active_roadmap_task_count"] == 4
    assert first["next_task_range"]["start"] == "exp6138"
    assert first["next_task_range"]["end"] == "exp6151"
    assert first["next_task_range"]["reserved_count"] == 14
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

    _write_json(tmp_path, "results/experiment_6151_unowned_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_6151_unowned_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [6151],
        }
    ]
    mod.validate_artifact(collision)


def test_scenario_report_6138_schema_validation_and_blocked_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6138-SCHEMA: required fields, protection, and checksum hold."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_6138_present": True,
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
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp6132-phase-d-hidden-state-held-eval"
            ),
            "exactly nine",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp6132-phase-d-hidden-state-held-eval": []}
            ),
            "exactly nine",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp6132-phase-d-hidden-state-held-eval"
            ].update(
                identity=[
                    "2026.08.531",
                    "exp6132-phase-d-hidden-state-held-eval",
                    "wrong.json",
                ]
            ),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp6125-v531-source-delta-ingestion": "missing"}),
            "terminal classes",
        ),
        (
            lambda artifact: artifact["no_artifact_backend_failure_receipt"].update(
                no_artifact_invented=False
            ),
            "no-artifact backend",
        ),
        (
            lambda artifact: artifact["structured_gate_skip_receipts"]["by_task"][
                "exp6130-phase-d-per-layer-surface-v2"
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
            lambda artifact: artifact["proposal_only_identities_excluded"].update(
                all_excluded_from_activated_matrix=False
            ),
            "proposal-only",
        ),
        (
            lambda artifact: artifact["exp6128_family_bimodality_preserved"].update(
                preserved=False
            ),
            "bimodality",
        ),
        (
            lambda artifact: artifact["retirement_signals_preserved"].update(
                all_retirement_signals_preserved=False
            ),
            "retirement",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=5
            ),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"].append([]),
            "adversarial verifier receipt",
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
            lambda artifact: artifact["staged_roadmap_activation_receipt"].update(
                activated=False
            ),
            "activation",
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
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
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
        "v531_activation_line_missing_or_not_nine",
        "v532_activation_line_missing",
        "live_verifier_missing",
        "task_owned_gate_missing",
        "openspec_req_6138_missing",
        "protected_file_modified",
        "atomic_output_unavailable",
    } <= failed

    many = tmp_path / "many"
    _make_root(many)
    bad_roadmap = _active_roadmap_payload()
    bad_roadmap["milestone"] = "2026.08.999"
    _write_text(many, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    _write_text(many, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    (many / mod.ROADMAP_DOC_RELATIVE_PATH).unlink()
    (many / mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp6129-phase-d-held-pool-v2"]).unlink()
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp6125-v531-source-delta-ingestion"],
        {"status": "complete", "honest_verdict": "complete: invented Exp6125 artifact"},
    )
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp6128-phase-d-calibration-pool-v2"],
        {
            **_artifact("exp6128-phase-d-calibration-pool-v2"),
            "phase_d_calibration_ready_score": 1.0,
        },
    )
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp6124-transition-v531")
    sparse_receipts["exp6126-phase-d-exp6115-transport-forensics"] = {
        **sparse_receipts["exp6126-phase-d-exp6115-transport-forensics"],
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
        if row.get("ownership_class") == "global_suite" and row.get("phase") == "after":
            row["failure_node_ids"].append("tests/python/new/test_owned.py::test_new")
        if row.get("ownership_class") == "spec_coverage" and row.get("phase") == "after":
            row["missing_node_ids"].append("tests/python/new/test_spec.py::test_new")
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_probe.py")
    monkeypatch.setattr(
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
    monkeypatch.setattr(
        mod,
        "_proposal_only_identities_excluded",
        lambda _matrix: {
            "proposal_only_task_ids": list(mod.PROPOSAL_ONLY_TASK_IDS),
            "all_excluded_from_activated_matrix": False,
            "principle": mod.FIELD_PRINCIPLES["proposal_only_identities_excluded"],
        },
    )
    monkeypatch.setattr(
        mod,
        "_retirement_signals",
        lambda _roadmap, _matrix: {
            "all_retirement_signals_preserved": False,
            "principle": mod.FIELD_PRINCIPLES["retirement_signals_preserved"],
        },
    )
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
        "roadmap_next_unloadable",
        "vnext_proposal_missing",
        "terminal_outcomes_not_preserved",
        "no_artifact_backend_failure_not_preserved",
        "gate_skip_reported_as_run",
        "proposal_only_identity_included",
        "exp6128_bimodality_not_preserved",
        "retirement_signal_not_preserved",
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
        "task_owned_gate_failed",
        "global_suite_debt_amplified",
        "global_spec_debt_amplified",
        "root_clutter_debt_amplified",
        "duplicate_history_amplified",
    } <= many_failed


def test_req_report_6138_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6138: helper failures produce explicit blocked receipts."""

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
    _make_root(malformed, include_531_blocks=0)
    _write_text(malformed, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    malformed_append = mod._append_completion_if_absent(malformed, terminal=True)
    assert malformed_append["append_count"] == 1

    assert (
        mod._classify_task(
            "exp6125-v531-source-delta-ingestion",
            {},
            {"present": False},
            {"latest_status": "FAIL", "attempt_count": 3},
        )
        == "no-artifact-backend-failure"
    )
    assert (
        mod._classify_task(
            "exp6130-phase-d-per-layer-surface-v2",
            {},
            {"present": False},
            {"latest_status": "GATE_BLOCK"},
        )
        == "conductor-gate-skipped-missing"
    )
    assert mod._classify_task("x", {"status": "complete"}, {"present": True}, {}) == "complete"
    assert mod._classify_task("x", {"status": "blocked"}, {"present": True}, {}) == "blocked"
    assert mod._classify_task("x", {"status": "unknown"}, {"present": True}, {}) == "missing"
    assert mod._same_number_aliases(tmp_path / "no-results", "exp6128-x", Path("x")) == []
    assert mod._range_number_mentions("Exp6138 and experiment_6151") == {6138, 6151}
    assert mod._range_number_mentions("value 0.6140 is not an experiment id") == set()
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {6138})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_DOC_RELATIVE_PATH, {6138, 6151})
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
    assert mod._dirty_worktree_receipt(tmp_path / "not-git")["git_present"] is False
    assert mod._nested_mapping({}, "missing") == {}
    assert mod._dirty_worktree_receipt(REPO)["git_present"] is True
    retirement = mod._retirement_signals(
        {
            "tasks": [
                "bad",
                {
                    "id": "exp6138-transition-v532",
                    "prior_failures": [
                        {
                            "experiment_id": "exp6124-transition-v531",
                            "retire_if_same_verdict": True,
                        }
                    ],
                },
            ]
        },
        {
            "exp6130-phase-d-per-layer-surface-v2": {
                "present": False,
                "conductor": {"latest_line": "1 of 1 gate(s) failed"},
            },
            "exp6132-phase-d-hidden-state-held-eval": {
                "present": False,
                "terminal_class": "conductor-gate-skipped-missing",
                "conductor": {"latest_line": "Pre-emptive skip: upstream retired"},
            },
        },
    )
    assert retirement["prior_failure_retire_if_same_verdict_receipts"] == [
        {
            "task_id": "exp6138-transition-v532",
            "experiment_id": "exp6124-transition-v531",
            "retire_if_same_verdict": True,
        }
    ]
    assert retirement["upstream_retired_gate_skip_mentions"][
        "exp6130-phase-d-per-layer-surface-v2"
    ] is False
    assert retirement["retired_gate_skip_task_ids"] == [
        "exp6132-phase-d-hidden-state-held-eval"
    ]
    assert retirement["all_retirement_signals_preserved"] is True
    reopened = mod._retirement_signals(
        {"tasks": []},
        {
            "exp6132-phase-d-hidden-state-held-eval": {
                "present": True,
                "conductor": {"latest_line": "Pre-emptive skip: upstream retired"},
            }
        },
    )
    assert reopened["all_retirement_signals_preserved"] is False
    verifier = mod._adversarial_receipts_group(
        {
            "exp6124-transition-v531": {
                **_receipt("exp6124-transition-v531"),
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
            },
            "exp6126-phase-d-exp6115-transport-forensics": {
                **_receipt("exp6126-phase-d-exp6115-transport-forensics"),
                "stdout_json": {
                    "reports": [
                        {
                            "artifact": "x",
                            "loaded": True,
                            "flag_count": 1,
                            "max_severity": 1,
                            "flags": [{"kind": "WARN", "severity": "warn"}],
                        }
                    ]
                },
            },
        },
        {
            "exp6124-transition-v531": {
                "present": True,
                "declared_deliverable": "results/experiment_6124_transition_v531.json",
            },
            "exp6126-phase-d-exp6115-transport-forensics": {
                "present": True,
                "declared_deliverable": (
                    "results/experiment_6126_phase_d_exp6115_transport_forensics.json"
                ),
            },
            "exp6127-phase-d-native-chat-transport-canary": {
                "present": True,
                "declared_deliverable": (
                    "results/experiment_6127_phase_d_native_chat_transport_canary.json"
                ),
            },
            **{
                task_id: {"present": False, "declared_deliverable": path.as_posix()}
                for task_id, path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
                if task_id
                not in {
                    "exp6124-transition-v531",
                    "exp6126-phase-d-exp6115-transport-forensics",
                    "exp6127-phase-d-native-chat-transport-canary",
                }
            },
        },
    )
    assert verifier["failed_receipt_task_ids"] == ["exp6124-transition-v531"]
    assert verifier["warning_receipt_task_ids"] == [
        "exp6126-phase-d-exp6115-transport-forensics"
    ]
    assert verifier["verified_present_declared_deliverable_count"] == 2
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
    staged_bad = tmp_path / "staged-bad"
    _make_root(staged_bad)
    _write_text(staged_bad, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    assert mod._activate_staged_roadmap(staged_bad)["mode"] == "staged_unloadable"
    staged_mismatch = tmp_path / "staged-mismatch"
    _make_root(staged_mismatch)
    bad_next = _active_roadmap_payload()
    bad_next["milestone"] = "2026.08.999"
    _write_text(staged_mismatch, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(bad_next))
    assert mod._activate_staged_roadmap(staged_mismatch)["mode"] == "staged_milestone_mismatch"
