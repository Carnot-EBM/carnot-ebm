"""Tests for the Exp5905 V525 transition receipt.

Spec refs: REQ-REPORT-5905, SCENARIO-REPORT-5905-EXACT-ARCHIVE,
SCENARIO-REPORT-5905-EXP5895-MIXED-RECEIPT,
SCENARIO-REPORT-5905-RESERVATION-AND-RANGE,
SCENARIO-REPORT-5905-APPEND-ONCE, SCENARIO-REPORT-5905-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5905_transition_v525 as mod


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


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5890-transition-v524": {
            "status": "complete",
            "honest_verdict": "complete: archived terminal .523 identities into .524",
            "next_range_collision_count": 0,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp5891-v524-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V524 source deltas",
            "accepted_finding_count": 0,
            "inference_substrate": "aggregation_from_external_primary_sources",
        },
        "exp5892-headroom-evidence-escrow": {
            "status": "complete_ready",
            "honest_verdict": "complete_ready: headroom_evidence_escrow_admitted",
            "headroom_admission_ready_score": 1.0,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp5893-grounding-shortcut-fixture": {
            "status": "ready",
            "honest_verdict": "ready: grounding_shortcut_exact_fixture_ready",
            "grounding_shortcut_fixture_ready_score": 1.0,
            "inference_substrate": "deterministic_exact_solver_labeled_dataset_no_llm",
        },
        "exp5894-one-to-one-grounding-ab": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: one_to_one_grounding_positive",
            "one_to_one_grounding_ready_score": 1.0,
            "inference_substrate": "online_exact_membership_query_sidecar_no_llm",
        },
        "exp5895-shortcut-safe-continuous-self-learning": {
            "status": "complete_null",
            "honest_verdict": "complete_null: shortcut_safe_csl_not_promotion_eligible",
            "continuous_self_learning_task": True,
            "shortcut_resistant_csl_ready_score": 0.0,
            "prospective_semantic_and_constraint_metrics": {
                "primary_minus_best_shortcut_control": {
                    "mean_delta": 0.25,
                    "ci95": [0.111111, 0.416667],
                    "n": 36,
                }
            },
            "forward_transfer_recurrence_retention_and_regret": {
                "retention": {
                    "protected_prefix_retention": 1.0,
                    "retention_regression_count": 0,
                }
            },
            "shortcut_false_accept_metrics": {
                "primary_zero_false_accepts": True,
                "unsafe_accept_count": 0,
            },
            "rollback_restart_and_state_hashes": {
                "restart_equivalence": 1.0,
                "rollback_hash_mismatch_count": 0,
            },
            "no_model_weight_mutation": {"all_unchanged": True},
            "retirement_decision": {
                "decision": "do_not_promote",
                "retired_dependency_chain_used": False,
            },
            "test_exit_codes": {".venv/bin/pytest tests/python -q": 2},
            "inference_substrate": (
                "deterministic_exact_verifier_and_versioned_external_state_no_llm"
            ),
        },
        "exp5896-typed-constraint-ir-fixture": {
            "status": "ready",
            "honest_verdict": "ready: typed ConstraintIR fixture replays",
            "typed_constraint_ir_fixture_ready_score": 1.0,
            "inference_substrate": "deterministic_exact_solver_labeled_dataset_no_llm",
        },
        "exp5897-sota-constraint-ir-repair-ab": {
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition: exp5896_gate_replayed",
            "preconditions_checked": {
                "block_reason": "exp5896_gate_replayed",
                "blocked_before_model_load": True,
                "headline_checks": {"exp5896_gate_replayed": False},
            },
            "trace_repair_mechanism_ready_score": 0.0,
            "inference_substrate": "live_llm_inference",
        },
        "exp5898-recursive-constraint-improvement": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5897-sota-constraint-ir-repair-ab.trace_repair_mechanism_ready_score"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp5897-sota-constraint-ir-repair-ab",
                    "artifact_field": "trace_repair_mechanism_ready_score",
                    "actual": 0.0,
                    "expected": 1.0,
                    "passed": False,
                }
            ],
        },
        "exp5900-arc-structured-evidence-memory-contract": {
            "status": "ready",
            "honest_verdict": "ready: structured_evidence_memory_contract_live_reachable",
            "structured_evidence_memory_contract_ready_score": 1.0,
            "public_level_solve_claimed": False,
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "exp5901-arc-structured-memory-causal-audit": {
            "status": "complete_positive",
            "honest_verdict": "complete_positive: structured_memory_causal_audit",
            "structured_memory_causal_ready_score": 1.0,
            "public_level_solve_claimed": False,
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "exp5902-arc-structured-memory-live-ab": {
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition: live_runner_permission",
            "preconditions_checked": {
                "live_runner_permission": {
                    "allow_live_env": False,
                    "ok": False,
                    "reason": "Exp5902 live runner is not enabled",
                }
            },
            "structured_memory_live_ready_score": 0.0,
            "public_level_solve_claimed": False,
            "incidental_solve_receipts": {
                "new_solve_headline_allowed": False,
                "registry_updated": False,
            },
            "inference_substrate": "live_llm_inference",
        },
        "exp5903-v524-capstone-reconciliation": {
            "status": "complete_with_nulls",
            "honest_verdict": "complete_with_nulls: all .524 identities terminal",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "exact_terminal_classification": {
                "terminal_class_by_task_id": {
                    task: terminal
                    for task, terminal in mod.EXPECTED_TERMINAL_CLASSES.items()
                }
            },
        },
    }
    return payloads[task_id]


def _completion_payload(*, include_524: bool = True, duplicate_old_blocks: int = 1) -> JsonDict:
    duplicate_block = {
        "id": "2026.07.510",
        "title": "Historical duplicate",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    milestones = [deepcopy(duplicate_block) for _ in range(duplicate_old_blocks)]
    if include_524:
        milestones.append(
            {
                "id": mod.MILESTONE_FROM,
                "title": mod.MILESTONE_FROM_TITLE,
                "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "completed": "2026-07-25",
                "finding": "Terminal outcomes preserved by Exp5903; see capstone artifact.",
                "tasks": [
                    {
                        "id": task_id,
                        "title": mod.ACTIVATED_TASK_TITLES[task_id],
                        "deliverable": rel_path.as_posix(),
                        "result": "terminal outcome preserved by Exp5903 capstone",
                    }
                    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
                ],
            }
        )
    return {"milestones": milestones}


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
            }
            for task_id, rel_path in mod.NEXT_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _vnext_doc() -> str:
    lines = [
        "# Research Roadmap vNEXT",
        "",
        "**Milestone:** 2026.07.525",
        "**Task range:** Exp5905-Exp5917",
        "Exp5904 is reserved by concurrent click-target work.",
    ]
    lines.extend(path.as_posix() for path in mod.NEXT_TASK_ARTIFACT_PATHS.values())
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-24 13:47 UTC | Exact terminal-boundary handoff from .523 into .52 | OK | 88 passed |",
            "| 2026-07-24 16:02 UTC | Dated evidence refresh after the V524 planner mark | OK | 89 passed |",
            "| 2026-07-24 17:07 UTC | Immutable hardness-headroom evidence escrow and cl | OK | 86 passed |",
            "| 2026-07-24 17:27 UTC | Gated on Exp5892 admission: exact grounding-shortc | OK | 87 passed |",
            "| 2026-07-24 17:51 UTC | Gated on Exp5893 fixture: one-to-one atom-groundin | OK | 87 passed |",
            "| 2026-07-24 19:32 UTC | Gated on Exp5894 mechanism: prospective shortcut-s | FLAGGED | adversarial_verify CRITICAL: DURATION_TOO_SHORT |",
            "| 2026-07-24 19:53 UTC | Engine-neutral typed ConstraintIR fixture with exa | OK | 89 passed |",
            "| 2026-07-24 20:16 UTC | Gated on Exp5896 fixture: three-family translate-r | OK | 87 passed |",
            "| 2026-07-24 20:23 UTC | Gated on Exp5897 trace lift: constraint-wise recur | GATE_BLOCK | 1 of 1 gate(s) failed |",
            "| 2026-07-24 21:21 UTC | Gated on Exp5898 recursion: portability, leakage, | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5898-recursive-constraint-improvement) |",
            "| 2026-07-24 20:46 UTC | Agent-owned ARC event tape and structured evidence | OK | 107 passed |",
            "| 2026-07-24 21:19 UTC | Gated on Exp5900 contract: ARC retrieval fidelity  | OK | 87 passed |",
            "| 2026-07-24 21:43 UTC | Gated on Exp5901 causality: adapter-disabled live  | OK | 91 passed |",
            "| 2026-07-24 23:16 UTC | Branch-independent terminal reconciliation for mil | OK | 169 passed |",
        ]
    )


def _make_root(
    root: Path,
    *,
    include_524_complete: bool = True,
    duplicate_old_blocks: int = 1,
) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5899-constraint-repair-portability-audit":
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(
            _completion_payload(
                include_524=include_524_complete,
                duplicate_old_blocks=duplicate_old_blocks,
            )
        ),
    )
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, _vnext_doc())
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.EVIDENCE_INDEX_RELATIVE_PATH, "# evidence index fixture\n")
    _write_text(root, mod.DOC_RECONCILE_RELATIVE_PATH, "# reconcile fixture\n")
    _write_text(root, mod.ADVERSARIAL_VERIFY_RELATIVE_PATH, "# verifier fixture\n")
    _write_text(root, "python/carnot/experiment_5904_click_target_discrimination.py", "# exp5904\n")
    _write_json(root, "results/experiment_5904_click_target_discrimination.json", {"status": "draft"})
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.E2E_TEST_PLAN_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.NORTH_STAR_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
    ):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5905\nfixture\n")


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
        if task_id != "exp5899-constraint-repair-portability-audit"
    }


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5905_transition_v525.py -q --no-cov -n 0",
                "exit_code": 0,
            },
            {
                "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5905_transition_v525.py --fail-under=100",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        duration_s=1.25,
    )


def test_req_report_5905_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5905: OpenSpec names exact archive and reservation gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5905") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5905-EXACT-ARCHIVE" in section
    assert "SCENARIO-REPORT-5905-EXP5895-MIXED-RECEIPT" in section
    assert "SCENARIO-REPORT-5905-RESERVATION-AND-RANGE" in section
    assert "Exp5905 through Exp5917" in section
    assert "exp5904_reserved=true" in section
    assert "next_range_collision_count=0" in section
    assert "aggregation_from_upstream_artifacts" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5905_exact_archive_and_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5905-EXACT-ARCHIVE: every activated .524 task classifies once."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_nulls"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.524",
        "destination_milestone": "2026.07.525",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 14
    assert "exp5904-click-target-discrimination" not in matrix
    assert matrix["exp5899-constraint-repair-portability-audit"]["present"] is False
    assert matrix["exp5899-constraint-repair-portability-audit"]["declared_deliverable"] == (
        "results/experiment_5899_constraint_repair_portability_audit.json"
    )

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["disjoint_terminal_class_count"] == 14
    assert classes["all_activated_terminal"] is True

    blocked = report["blocked_and_gate_blocked_receipts"]
    assert blocked["blocked_precondition_task_ids"] == [
        "exp5897-sota-constraint-ir-repair-ab",
        "exp5902-arc-structured-memory-live-ab",
    ]
    assert blocked["gate_blocked_task_ids"] == [
        "exp5898-recursive-constraint-improvement",
        "exp5899-constraint-repair-portability-audit",
    ]
    assert blocked["declared_deliverable_missing_but_gate_blocked_task_ids"] == [
        "exp5899-constraint-repair-portability-audit"
    ]
    assert all(row["treated_as_success"] is False for row in blocked["receipts"])
    assert len(report["adversarial_verifier_receipts"]) == 13
    mod.validate_artifact(report)


def test_scenario_report_5905_preserves_exp5895_science_and_operational_null(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5905-EXP5895-MIXED-RECEIPT: science and suite failure coexist."""

    _make_root(tmp_path)
    report = _build(tmp_path)
    receipt = report["exp5895_science_and_operational_receipt"]

    assert receipt["task_id"] == "exp5895-shortcut-safe-continuous-self-learning"
    assert receipt["terminal_class"] == "null"
    assert receipt["positive_scientific_submetrics"] == {
        "prospective_semantic_lift_mean_delta": 0.25,
        "prospective_semantic_lift_ci95": [0.111111, 0.416667],
        "protected_prefix_retention": 1.0,
        "unsafe_accept_count": 0,
        "restart_equivalence": 1.0,
        "rollback_hash_mismatch_count": 0,
        "no_model_weight_mutation_all_unchanged": True,
    }
    assert receipt["operational_null_receipt"] == {
        "shortcut_resistant_csl_ready_score": 0.0,
        "required_full_suite_command": ".venv/bin/pytest tests/python -q",
        "required_full_suite_exit_code": 2,
        "promoted_as_ready": False,
    }
    assert receipt["science_preserved"] is True
    assert receipt["operational_null_preserved"] is True
    assert receipt["laundering_detected"] is False
    mod.validate_artifact(report)


def test_scenario_report_5905_append_once_reservation_and_range(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5905-RESERVATION-AND-RANGE: Exp5904 stays reserved."""

    _make_root(tmp_path, include_524_complete=False, duplicate_old_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 1
    assert report["duplicate_history_amplification_count"] == 0
    complete = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    blocks = [block for block in complete["milestones"] if block["id"] == mod.MILESTONE_FROM]
    assert len(blocks) == 1
    assert [row["id"] for row in blocks[0]["tasks"]] == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)

    second = _build(tmp_path)
    assert second["research_complete_append_count"] == 0
    assert second["duplicate_history_amplification_count"] == 0

    reservation = report["exp5904_reservation_receipt"]
    assert reservation["exp5904_reserved"] is True
    assert reservation["edited_by_exp5905"] is False
    assert reservation["classified_by_exp5905"] is False
    assert {
        "python/carnot/experiment_5904_click_target_discrimination.py",
        "results/experiment_5904_click_target_discrimination.json",
    } <= set(reservation["existing_paths"])

    assert report["next_task_range"] == {
        "start": "exp5905",
        "end": "exp5917",
        "count": 13,
        "task_ids": list(mod.NEXT_TASK_ARTIFACT_PATHS),
        "reserved_predecessor": "exp5904",
    }
    assert report["next_range_collision_count"] == 0
    assert report["preconditions_checked"]["range_collision_scan"]["collision_count"] == 0
    mod.validate_artifact(report)


def test_scenario_report_5905_unexpected_next_range_reference_blocks(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5905-RESERVATION-AND-RANGE: stale Exp5905-Exp5917 files block."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5910_stale_collision.json", {"status": "stale"})
    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 1
    assert "next_range_collision" in report["preconditions_checked"]["failed_preconditions"]
    assert report["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {"path": "results/experiment_5910_stale_collision.json", "kind": "unexpected_next_range_reference"}
    ]
    mod.validate_artifact(report)


def test_scenario_report_5905_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5905-SCHEMA: required fields and protections are enforced."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_updated": True,
        "ops_status_deferred_to_conductor": True,
        "ops_changelog_deferred_to_conductor": True,
        "traceability_deferred_to_conductor": True,
        "ops_conductor_log_deferred_to_conductor": True,
    }
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report["field_provenance"]
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="complete_with_nulls: bad"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (lambda artifact: artifact.update(next_range_collision_count=1), "next_range_collision_count must be zero"),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact["exp5904_reservation_receipt"].update(
                exp5904_reserved=False
            ),
            "Exp5904 reservation",
        ),
        (
            lambda artifact: artifact["exp5904_reservation_receipt"].update(
                classified_by_exp5905=True
            ),
            "Exp5904 reservation",
        ),
        (
            lambda artifact: artifact["exp5895_science_and_operational_receipt"][
                "operational_null_receipt"
            ].update(promoted_as_ready=True),
            "Exp5895 laundering",
        ),
        (
            lambda artifact: artifact["exp5895_science_and_operational_receipt"][
                "operational_null_receipt"
            ].update(required_full_suite_exit_code=0),
            "Exp5895 full-suite exit",
        ),
        (
            lambda artifact: artifact["blocked_and_gate_blocked_receipts"][
                "receipts"
            ][0].update(treated_as_success=True),
            "blocked/gate receipt",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp5903-v524-capstone-reconciliation"
            ),
            "exactly fourteen",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5890-transition-v524"
            ].update(identity=["2026.07.524", "exp5890-transition-v524", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5890-transition-v524"
            ].pop("receipt_hash"),
            "missing adversarial verifier receipt fields",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5890-transition-v524"
            ].update(command="python other.py"),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact.update(adversarial_verifier_receipts=[]),
            "adversarial verifier receipts",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"]["files"][
                mod.CONDUCTOR_RELATIVE_PATH.as_posix()
            ].update(unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: artifact.update(protected_files_unchanged=[]),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(
                principle="wrong"
            ),
            "field provenance missing",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5890-transition-v524": []}
            ),
            "malformed matrix row",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"][
                "terminal_class_by_task_id"
            ].update({"exp5890-transition-v524": "null"}),
            "terminal classes",
        ),
        (
            lambda artifact: artifact.update(exp5895_science_and_operational_receipt=[]),
            "Exp5895 laundering",
        ),
        (
            lambda artifact: artifact.update(blocked_and_gate_blocked_receipts=[]),
            "blocked/gate receipt missing",
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


def test_req_report_5905_defensive_helpers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5905: defensive helper paths stay deterministic."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}

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

    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._task_signature({"tasks": "bad"}) == ()
    assert mod._status_from_log(None) == "MISSING"
    assert mod._status_from_log("| now | Mystery task | UNKNOWN | details |") == "LOGGED"
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._test_exit_code({"test_exit_codes": [{"command": "cmd", "exit_code": 7}]}, "cmd") == 7
    assert (
        mod._test_exit_code({"test_exit_codes": [{"command": "cmd", "exit_code": "bad"}]}, "cmd")
        is None
    )
    assert mod._test_exit_code({"test_exit_codes": "bad"}, "cmd") is None
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {5905})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.CONDUCTOR_LOG_RELATIVE_PATH, {5905})
        == "transition_owned_conductor_attempt_reference"
    )
    assert mod._status_and_verdict(
        [], {"terminal_class_by_task_id": {"x": "unsafe/disqualified"}}
    ) == ("blocked", "blocked: unsafe/disqualified .524 identity present")
    assert mod._status_and_verdict(
        [], {"terminal_class_by_task_id": {"x": "ready/positive"}}
    ) == (
        "complete",
        "complete: archived terminal .524 identities into .525 with collision-free allocation",
    )

    assert mod._terminal_class(
        "exp5890-transition-v524",
        {"status": "retired", "honest_verdict": "retired: closed"},
        {"present": True},
        {},
        {},
    ) == "retired"
    assert mod._terminal_class(
        "exp5890-transition-v524",
        {"status": "blocked", "honest_verdict": "blocked: failed"},
        {"present": True},
        {},
        {},
    ) == "blocked"
    assert mod._terminal_class(
        "exp5890-transition-v524",
        {},
        {"present": False},
        {"latest_status": "GATE_BLOCK"},
        {},
    ) == "gate-blocked"
    assert mod._terminal_class(
        "exp5890-transition-v524",
        {},
        {"present": True},
        {},
        {"flag_count": 1, "max_severity": 2},
    ) == "unsafe/disqualified"
    assert mod._terminal_class("exp5890-transition-v524", {}, {"present": True}, {}, {}) == "blocked"

    assert mod._append_completion_if_absent(tmp_path / "nonterminal", terminal=False) == {
        "append_count": 0,
        "appended": False,
        "reason": "nonterminal_identity_present",
        "before_sha256": None,
        "after_sha256": None,
        "before_duplicate_block_count": 0,
        "after_duplicate_block_count": 0,
        "duplicate_history_amplification_count": 0,
    }
    absent_root = tmp_path / "absent-history"
    append = mod._append_completion_if_absent(absent_root, terminal=True)
    assert append["append_count"] == 1
    assert (absent_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).exists()

    _make_root(tmp_path / "missing-receipts")
    missing_receipt_report = mod.build_report(
        tmp_path / "missing-receipts",
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=1.0,
    )
    assert "missing_adversarial_receipts" in missing_receipt_report["preconditions_checked"][
        "failed_preconditions"
    ]
    with pytest.raises(ValueError, match="missing adversarial verifier receipt"):
        mod.validate_artifact(missing_receipt_report)

    bad_root = tmp_path / "bad-preconditions"
    _make_root(bad_root)
    _write_text(bad_root, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    (bad_root / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    bad_report = mod.build_report(
        bad_root,
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 1}],
        duration_s=1.0,
    )
    assert {
        "active_roadmap_unloadable",
        "research_complete_unparseable",
        "live_verifier_missing",
        "required_tests_failed",
        "missing_adversarial_receipts",
    } <= set(bad_report["preconditions_checked"]["failed_preconditions"])

    branch_root = tmp_path / "branch-preconditions"
    _make_root(branch_root)
    branch_classes = {
        "all_activated_terminal": True,
        "terminal_class_by_task_id": dict(mod.EXPECTED_TERMINAL_CLASSES),
    }
    branch_exp5895 = {
        "science_preserved": True,
        "operational_null_preserved": True,
        "laundering_detected": False,
    }
    monkeypatch.setattr(
        mod,
        "_append_completion_if_absent",
        lambda root, terminal: {
            "append_count": 0,
            "appended": False,
            "reason": "fixture",
            "before_sha256": None,
            "after_sha256": None,
            "before_duplicate_block_count": 0,
            "after_duplicate_block_count": 1,
            "duplicate_history_amplification_count": 1,
        },
    )
    monkeypatch.setattr(
        mod,
        "_exp5904_reservation_receipt",
        lambda root: {
            "exp5904_reserved": False,
            "edited_by_exp5905": False,
            "classified_by_exp5905": False,
        },
    )
    monkeypatch.setattr(mod, "_exp5895_receipt", lambda payloads, classes: branch_exp5895)
    monkeypatch.setattr(
        mod,
        "_exact_terminal_classification",
        lambda payloads, metadata, conductor, receipts: {
            **branch_classes,
            "terminal_class_by_task_id": {
                **branch_classes["terminal_class_by_task_id"],
                "exp5890-transition-v524": "null",
            },
        },
    )
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda root, before: {"all_unchanged": False, "files": {}},
    )
    monkeypatch.setattr(
        mod,
        "_resource_receipts",
        lambda root: {
            "disk": {"ok": False},
            "memory": {"ok": True},
        },
    )
    monkeypatch.setattr(
        mod,
        "_atomic_output_receipt",
        lambda path: {"ok": False},
    )
    branch_report = mod.build_report(
        branch_root,
        adversarial_receipts=_receipts(),
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=1.0,
    )
    assert {
        "duplicate_history_amplified",
        "exp5904_not_reserved",
        "terminal_outcomes_not_preserved",
        "protected_file_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= set(branch_report["preconditions_checked"]["failed_preconditions"])

    monkeypatch.setattr(
        mod,
        "_exp5895_receipt",
        lambda payloads, classes: {
            "science_preserved": False,
            "operational_null_preserved": True,
            "laundering_detected": True,
        },
    )
    exp5895_bad_report = mod.build_report(
        branch_root,
        adversarial_receipts=_receipts(),
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=1.0,
    )
    assert "exp5895_science_or_null_not_preserved" in exp5895_bad_report[
        "preconditions_checked"
    ]["failed_preconditions"]
