"""Tests for the Exp5903 V524 capstone reconciliation.

Spec refs: REQ-REPORT-5903, SCENARIO-REPORT-5903-EXACT-IDENTITIES,
SCENARIO-REPORT-5903-BRANCH-INDEPENDENT, SCENARIO-REPORT-5903-APPEND-ONCE,
SCENARIO-REPORT-5903-PROTECTION, SCENARIO-REPORT-5903-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5903_v524_capstone_reconciliation as mod


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
            "honest_verdict": "complete: archived terminal .523 identities",
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
            "honest_verdict": "complete_positive: one_to_one_grounding_beats_controls",
            "one_to_one_grounding_ready_score": 1.0,
            "inference_substrate": "online_exact_membership_query_sidecar_no_llm",
        },
        "exp5895-shortcut-safe-continuous-self-learning": {
            "status": "complete_null",
            "honest_verdict": "complete_null: shortcut_safe_csl_not_promotion_eligible",
            "continuous_self_learning_task": True,
            "shortcut_resistant_csl_ready_score": 0.0,
            "retirement_decision": {
                "decision": "do_not_promote",
                "retired_dependency_chain_used": False,
            },
            "no_model_weight_mutation": {"all_unchanged": True},
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
            "trace_repair_mechanism_ready_score": 0.0,
            "upstream_gate_and_fixture_hashes": {"replay_ok": False},
            "model_specs": [
                {"name": "Qwen3.6-35B-A3B", "gpu": 0, "required": True},
                {"name": "Gemma4-31B-it", "gpu": 1, "required": True},
            ],
            "model_file_hashes": {"all_hashed": True},
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
            "structured_memory_live_ready_score": 0.0,
            "public_level_solve_claimed": False,
            "incidental_solve_receipts": {
                "new_solve_headline_allowed": False,
                "registry_updated": False,
            },
            "model_specs": {"never_replaces_required_pair": True},
            "model_file_hashes": {"hash_algorithm": "sha256", "models": []},
            "inference_substrate": "live_llm_inference",
        },
    }
    return payloads[task_id]


def _roadmap_payload() -> JsonDict:
    tasks: list[JsonDict] = []
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row: JsonDict = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": mod.ACTIVATED_TASK_TITLES[task_id],
            "deliverable": rel_path.as_posix(),
        }
        if task_id in mod.GATED_ON:
            row["gated_on"] = deepcopy(mod.GATED_ON[task_id])
        if task_id in mod.PRIOR_FAILURES:
            row["prior_failures"] = deepcopy(mod.PRIOR_FAILURES[task_id])
        tasks.append(row)
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": mod.MILESTONE_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": tasks,
    }


def _complete_payload(*, include_524: bool = False) -> JsonDict:
    milestones: list[JsonDict] = [
        {
            "id": "2026.07.523",
            "title": "Previous milestone",
            "tasks": [
                {
                    "id": "exp5877-transition-v523",
                    "deliverable": "results/experiment_5877_transition_v523.json",
                }
            ],
        }
    ]
    if include_524:
        milestones.append(
            {
                "id": mod.MILESTONE,
                "title": mod.MILESTONE_TITLE,
                "tasks": [
                    {
                        "id": task_id,
                        "deliverable": rel_path.as_posix(),
                    }
                    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
                ],
            }
        )
    return {"milestones": milestones}


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
        ]
    )


def _make_root(root: Path, *, include_524_complete: bool = False) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id in {"exp5899-constraint-repair-portability-audit", mod.EXPERIMENT_ID}:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_complete_payload(include_524=include_524_complete)),
    )
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "# Research Roadmap vNEXT\n\n**Task range:** Exp5890-Exp5903\n",
    )
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.E2E_TEST_PLAN_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.CAPSTONE_HELPER_RELATIVE_PATH,
        mod.DOC_RECONCILE_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.NORTH_STAR_RELATIVE_PATH,
        mod.DOCS_INDEX_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
    ):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5903\nfixture\n")


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
        if task_id not in {"exp5899-constraint-repair-portability-audit", mod.EXPERIMENT_ID}
    }


def _build(root: Path, *, include_524_complete: bool = False) -> JsonDict:
    _make_root(root, include_524_complete=include_524_complete)
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5903_v524_capstone_reconciliation.py -q",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.5,
    )


def test_req_report_5903_spec_declares_capstone_contract() -> None:
    """REQ-REPORT-5903: OpenSpec names exact identity and branch safeguards."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5903") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5903-EXACT-IDENTITIES" in section
    assert "SCENARIO-REPORT-5903-BRANCH-INDEPENDENT" in section
    assert "SCENARIO-REPORT-5903-APPEND-ONCE" in section
    assert "SCENARIO-REPORT-5903-PROTECTION" in section
    assert "Exp5890 through Exp5903" in section
    assert "aggregation_from_upstream_artifacts" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5903_exact_identities_and_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5903-EXACT-IDENTITIES: all activated tasks classify once."""

    report = _build(tmp_path)

    assert report["status"] == "complete_with_nulls"
    assert report["honest_verdict"].startswith("complete_with_nulls:")
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["milestone_and_task_range"] == {
        "milestone": "2026.07.524",
        "task_range": {"start": "exp5890", "end": "exp5903", "count": 14},
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }

    matrix = report["activated_task_and_declared_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 14
    assert matrix["exp5899-constraint-repair-portability-audit"]["present"] is False
    assert matrix["exp5899-constraint-repair-portability-audit"]["declared_deliverable"] == (
        "results/experiment_5899_constraint_repair_portability_audit.json"
    )
    assert matrix[mod.EXPERIMENT_ID]["terminal_evidence_source"] == "current_capstone_runtime"

    classes = report["exact_terminal_classification"]
    assert set(classes["terminal_class_by_task_id"]) == set(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert classes["terminal_class_by_task_id"] == {
        "exp5890-transition-v524": "ready/positive",
        "exp5891-v524-source-delta-ingestion": "null",
        "exp5892-headroom-evidence-escrow": "ready/positive",
        "exp5893-grounding-shortcut-fixture": "ready/positive",
        "exp5894-one-to-one-grounding-ab": "ready/positive",
        "exp5895-shortcut-safe-continuous-self-learning": "null",
        "exp5896-typed-constraint-ir-fixture": "ready/positive",
        "exp5897-sota-constraint-ir-repair-ab": "blocked-precondition",
        "exp5898-recursive-constraint-improvement": "gate-blocked",
        "exp5899-constraint-repair-portability-audit": "gate-blocked",
        "exp5900-arc-structured-evidence-memory-contract": "ready/positive",
        "exp5901-arc-structured-memory-causal-audit": "ready/positive",
        "exp5902-arc-structured-memory-live-ab": "blocked-precondition",
        "exp5903-v524-capstone-reconciliation": "ready/positive",
    }
    assert classes["disjoint_terminal_class_count"] == 14
    assert classes["all_activated_terminal"] is True
    assert classes["nonterminal_task_ids"] == []

    receipts = report["missing_gate_blocked_and_unactivated_receipts"]
    assert receipts["missing_task_ids"] == []
    assert receipts["gate_blocked_task_ids"] == [
        "exp5898-recursive-constraint-improvement",
        "exp5899-constraint-repair-portability-audit",
    ]
    assert receipts["declared_deliverable_missing_but_gate_blocked_task_ids"] == [
        "exp5899-constraint-repair-portability-audit"
    ]
    assert receipts["unactivated_task_ids"] == []
    assert all(row["treated_as_success"] is False for row in receipts["receipts"])


def test_scenario_report_5903_branch_independence_and_required_slots(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5903-BRANCH-INDEPENDENT: branch summaries do not cascade."""

    report = _build(tmp_path)
    branch = report["branch_independent_science_summary"]

    grounding = branch["grounding_and_continuous_self_learning"]
    assert grounding["positive_task_ids"] == [
        "exp5892-headroom-evidence-escrow",
        "exp5893-grounding-shortcut-fixture",
        "exp5894-one-to-one-grounding-ab",
    ]
    assert grounding["null_task_ids"] == ["exp5895-shortcut-safe-continuous-self-learning"]
    assert grounding["branch_terminal_class"] == "mixed_positive_and_null"

    constraint = branch["constraint_ir"]
    assert constraint["positive_task_ids"] == ["exp5896-typed-constraint-ir-fixture"]
    assert constraint["blocked_precondition_task_ids"] == ["exp5897-sota-constraint-ir-repair-ab"]
    assert constraint["gate_blocked_task_ids"] == [
        "exp5898-recursive-constraint-improvement",
        "exp5899-constraint-repair-portability-audit",
    ]

    arc = branch["arc_memory"]
    assert arc["positive_task_ids"] == [
        "exp5900-arc-structured-evidence-memory-contract",
        "exp5901-arc-structured-memory-causal-audit",
    ]
    assert arc["blocked_precondition_task_ids"] == ["exp5902-arc-structured-memory-live-ab"]
    assert branch["branch_overwrite_detected"] is False

    csl = report["continuous_self_learning_slot_receipt"]
    assert csl["task_id"] == "exp5895-shortcut-safe-continuous-self-learning"
    assert csl["activated"] is True
    assert csl["scientific_verdict_class"] == "null"
    assert csl["continuous_self_learning_task"] is True
    assert csl["promoted_as_ready"] is False
    assert csl["retired_dependency_chain_used"] is False

    arc_slot = report["arc_generalization_slot_receipt"]
    assert arc_slot["task_id"] == "exp5902-arc-structured-memory-live-ab"
    assert arc_slot["activated"] is True
    assert arc_slot["terminal_class"] == "blocked-precondition"
    assert arc_slot["public_arc_re_solve_claimed"] is False
    assert arc_slot["registry_updated"] is False


def test_scenario_report_5903_append_once_and_protected_files(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5903-APPEND-ONCE: history changes zero or one time."""

    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 1
    assert report["duplicate_history_amplification_count"] == 0
    complete = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    blocks = [block for block in complete["milestones"] if block["id"] == mod.MILESTONE]
    assert len(blocks) == 1
    assert [row["id"] for row in blocks[0]["tasks"]] == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)

    second = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=report["test_exit_codes"],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.6,
    )
    assert second["research_complete_append_count"] == 0
    assert second["duplicate_history_amplification_count"] == 0

    protected = report["protected_files_unchanged"]
    assert protected["all_unchanged"] is True
    for rel_path in mod.PROTECTED_FILE_PATHS:
        assert protected["files"][rel_path.as_posix()]["unchanged"] is True


def test_scenario_report_5903_policy_exclusions_schema_and_recommendations(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5903-SCHEMA: policy receipts and schema fields are stable."""

    report = _build(tmp_path)
    errors = mod.validate_report(report)
    assert errors == []

    policy = report["model_policy_and_gpu_receipts"]
    assert policy["model_policy_substitution_detected"] is False
    assert policy["required_pair_preserved"] is True
    assert policy["gpu_receipt_task_ids"] == [
        "exp5897-sota-constraint-ir-repair-ab",
        "exp5902-arc-structured-memory-live-ab",
    ]

    decisions = report["exclusion_and_retirement_decisions"]
    assert decisions["public_arc_re_solve_promoted"] is False
    assert decisions["retired_dependency_promoted"] is False
    assert decisions["protected_file_mutation_promoted"] is False
    assert decisions["model_policy_substitution_promoted"] is False

    assert report["docs_reconciled"] == {
        "openspec_research_reporting_updated": True,
        "ops_status_deferred_to_conductor": True,
        "ops_changelog_deferred_to_conductor": True,
        "traceability_deferred_to_conductor": True,
        "docs_index_modified": False,
    }
    assert len(report["adversarial_verifier_receipts"]) == 12
    assert all(row["flag_count"] == 0 for row in report["adversarial_verifier_receipts"])
    assert report["preconditions_checked"]["roadmap_next_present"] is False
    assert report["preconditions_checked"]["atomic_output"]["ok"] is True

    recommendations = report["next_three_falsifiable_recommendations"]
    assert len(recommendations) == 3
    for row in recommendations:
        assert row["future_id_allocated"] is False
        assert row["retired_scope_reopened"] is False
        assert row["evidence_field"] in {
            "exact_terminal_classification",
            "continuous_self_learning_slot_receipt",
            "arc_generalization_slot_receipt",
            "branch_independent_science_summary",
        }


def test_req_report_5903_defensive_helpers_keep_failure_states_terminal(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5903: defensive branches fail closed instead of promoting evidence."""

    assert mod._read_yaml_mapping(tmp_path / "missing.yaml")[1]["error"] == "missing"
    assert mod._task_signature({}) == ()
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: []\n")
    nonterminal_append = mod._append_completion_if_terminal(tmp_path, terminal=False)
    assert nonterminal_append["reason"] == "nonterminal_identity_present"

    assert mod._status_from_log(None) == "MISSING"
    assert mod._status_from_log("| no terminal token |") == "LOGGED"
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 2}}) == 2
    assert mod._receipt_max_severity({}) == -1
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert len(mod._test_rows(None)) == len(mod.DEFAULT_TEST_COMMANDS)

    present = {"present": True}
    assert (
        mod._terminal_class(
            "exp5890-transition-v524",
            {},
            present,
            {},
            {"flag_count": 1, "max_severity": 2},
        )
        == "unsafe/disqualified"
    )
    assert (
        mod._terminal_class(
            "exp5890-transition-v524",
            {"status": "retired", "honest_verdict": "retired: closed"},
            present,
            {},
            {},
        )
        == "retired"
    )
    assert (
        mod._terminal_class(
            "exp5890-transition-v524",
            {"status": "blocked", "honest_verdict": "blocked: failed"},
            present,
            {},
            {},
        )
        == "blocked"
    )
    assert mod._terminal_class("exp5890-transition-v524", {}, present, {}, {}) == "blocked"

    assert mod._branch_terminal_class([], [], ["exp"]) == "blocked"
    assert mod._branch_terminal_class(["exp"], [], []) == "ready/positive"
    assert mod._branch_terminal_class([], ["exp"], []) == "null"
    assert mod._branch_terminal_class([], [], []) == "mixed_terminal"

    assert (
        mod._status({"all_activated_terminal": False, "terminal_class_by_task_id": {}})[0]
        == "blocked"
    )
    assert (
        mod._status(
            {
                "all_activated_terminal": True,
                "terminal_class_by_task_id": {"exp": "unsafe/disqualified"},
            }
        )[0]
        == "blocked"
    )
    assert (
        mod._status(
            {
                "all_activated_terminal": True,
                "terminal_class_by_task_id": {"exp": "ready/positive"},
            }
        )[0]
        == "complete"
    )
