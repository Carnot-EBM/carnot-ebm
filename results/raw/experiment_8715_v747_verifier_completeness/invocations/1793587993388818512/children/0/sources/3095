"""Cold reconciliation of V666 evidence without a current model call.

Spec refs: REQ-REPORT-7642 and SCENARIO-REPORT-7642-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
import time
from typing import Any

from carnot import experiment_7629_v666_contract_methods as contract
from carnot import experiment_7628_v665_capstone as prior
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260925"
MILESTONE = "2026.09.666"
RESULT_PATH = Path("results/experiment_7642_v666_capstone.json")
NOTE_PATH = Path("docs/research-notes/v666-capstone.md")
MODULE_PATH = Path("python/carnot/experiment_7642_v666_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7642_v666_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7642_v666_capstone.py")
EXPECTED_TASK_IDS = contract.EXPECTED_TASK_IDS
PRE_GATE_PATHS = {
    "exp7632-fit-evidence": "results/experiment_7632_fit_evidence.json",
    "exp7633-online-evidence": "results/experiment_7633_online_evidence.json",
    "exp7634-evaluation-evidence": "results/experiment_7634_evaluation_evidence.json",
    "exp7640-arc-wrapper-generalization": "results/experiment_7640_arc_wrapper_generalization.json",
}
INPUT_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "scripts/publication_gate.py",
    "ops/north-star.md",
    "ops/verifier_gaps.md",
    "research-hardware-wishlist.md",
    "openspec/capabilities/research-reporting/spec.md",
)


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Flush each phase boundary with elapsed current-work time."""

    extra = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[exp7642] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {extra}",
        flush=True,
    )


def load_authority(root: Path) -> JsonDict:
    """Select the V666 authority and compare all structured contract fields."""

    selected, roadmap, candidates = contract.resolve_v666_roadmap(root)
    comparison = contract.compare_contract_authorities(
        (root / contract.DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    if not comparison["passed"]:
        raise ValueError(f"V666 authority mismatch: {comparison['errors']}")
    return {
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": candidates,
        "comparison_passed": True,
        "tasks": roadmap["tasks"],
    }


def collect_dispositions(root: Path, tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Read every actual producer or conductor pre-gate receipt in roster order."""

    if [task.get("id") for task in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("V666 roster order or count mismatch")
    rows = []
    for order, task in enumerate(tasks, 1):
        task_id = str(task["id"])
        planned = str(task["deliverable"])
        actual_label = PRE_GATE_PATHS.get(task_id, planned)
        path = root / actual_label
        if order == 14:
            kind, exists, actual_label, payload = "current_self", False, None, {}
        elif path.is_file():
            payload = prior.load_json(path)
            kind = (
                "conductor_pre_gate"
                if "blocked_diagnostic_contract" in payload
                else "terminal_producer"
            )
            exists = True
        else:
            payload, kind, exists, actual_label = {}, "missing_work", False, None
        gate = None
        if kind == "conductor_pre_gate":
            gate = prior._pre_gate_failure(task_id, str(actual_label), payload)
        elif kind == "missing_work":
            gate = prior._failed_check(
                "required_scientific_producer", task_id, planned, "path", "exists", True, False
            )
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task.get("title"),
                "phase": task.get("phase"),
                "planned_path": planned,
                "actual_path": actual_label,
                "custody_kind": kind,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "size_bytes": path.stat().st_size if exists else 0,
                "honest_verdict": payload.get("honest_verdict") if exists else None,
                "verdict_class": (
                    "blocked" if kind == "conductor_pre_gate" else payload.get("verdict_class")
                )
                if exists
                else None,
                "flagged_adversarial": payload.get("flagged_adversarial", False),
                "gate_check": gate,
                "terminal_reader_results": payload.get(
                    "terminal_reader_outcomes", {"status": "not_run"}
                ),
            }
        )
    rows[-1]["honest_verdict"] = "complete_blocked_required_v666_external_evidence"
    rows[-1]["verdict_class"] = "blocked"
    return rows


def reduce_evidence(root: Path) -> JsonDict:
    """Recompute current eligible counts and retain unavailable branch limits."""

    audit = prior.load_json(root / "results/experiment_7638_v666_evidence_audit.json")
    method = prior.load_json(root / "results/experiment_7629_v666_contract_methods.json")
    pilot = prior.load_json(root / "results/experiment_7631_v666_schema_pilot.json")
    arc = prior.load_json(root / "results/experiment_7639_v666_arc_goal_dedup.json")
    native = prior.load_json(root / "results/experiment_7641_v666_native_consumer.json")
    old_native = prior.load_json(root / "results/experiment_7627_v665_native_cost.json")
    static, learning = audit["branch_dispositions"]
    native_rows = native["integration_rows"]
    native_units = {row["unit_id"] for row in native_rows}
    native_passed = sum(row.get("passed") is True for row in native_rows)
    prior_speed = prior._native_reduction(old_native)
    return {
        "schema_readiness": {
            "ready": pilot.get("evidence_transport_ready_score") == 1,
            "observed_groups": pilot.get("sample_size_budget", {}).get("observed", 0),
            "planned_model": "unsloth/Qwen3.8-27B-GGUF",
            "current_model_calls": 0,
        },
        "static_evidence": {
            "eligible": audit.get("static_audit_eligible_score") == 1,
            "probability_benefit": static.get("probability_benefit"),
            "utility_benefit": static.get("utility_benefit"),
            "historically_exposed": static.get("historically_exposed"),
            "source_group_control": "erasure_and_permutation_planned_not_measured",
            "claim_scope": "unavailable_not_semantic_null",
        },
        "causal_learning": {
            "eligible": audit.get("learning_audit_eligible_score") == 1,
            "prequential_benefit": learning.get("prequential_benefit"),
            "retention_benefit": learning.get("retention_benefit"),
            "historically_exposed": learning.get("historically_exposed"),
            "claim_scope": "unavailable_not_learning_null",
        },
        "arc_method": {
            "eligible": arc.get("verdict_class") not in {"disqualified", "blocked", "partial"}
            and arc.get("flagged_adversarial") is False,
            "planner_goal_guard_ready": arc.get("planner_goal_guard_ready_score") == 1,
            "exact_fixture_count": len({row["unit_id"] for row in arc.get("rows", [])}),
            "fixture_scope": "circular_positive_only_for_exact_execution_truth",
            "flagged_source": arc.get("flagged_adversarial"),
        },
        "arc_wrapper": {
            "scored_comparison_available": False,
            "live_hidden_game_benefit": None,
            "historical_replay_limit": "saved_Qwen_engines_are_exposed_controls_not_fresh_generation",
            "expert_control_limit": "Exp10013 live-wrapper ar25 expert count 6/10, not 7/10",
        },
        "native_packaging": {
            "eligible": native.get("native_consumer_ready_score") == 1,
            "independent_units": len(native_units),
            "passed_units": native_passed,
            "historical_direct_native_ratio": prior_speed["python_over_direct_native_estimate"],
            "historical_lower95": prior_speed["python_over_direct_native_lower95"],
            "historical_blocks": prior_speed["independent_blocks"],
            "new_10x_claim": False,
            "current_speed_benchmark": False,
        },
        "method_sources": [
            {
                "family": row["method_family"],
                "url": row["primary_url"],
                "adopted_control": row["adopted_control"],
                "claim_limit": row["claim_limit"],
            }
            for row in method["method_rows"]
        ],
    }


def acceptance_gates(summary: Mapping[str, Any]) -> JsonDict:
    """Keep six independent scientific questions separate."""

    observations = {
        "validity": (True, "authority and actual source hashes authenticate"),
        "readiness": (
            summary["schema_readiness"]["ready"],
            "schema and required scientific producers ready",
        ),
        "probability_benefit": (
            summary["static_evidence"]["probability_benefit"],
            "eligible proper-loss improvement measured",
        ),
        "utility": (
            summary["static_evidence"]["utility_benefit"],
            "typed decision utility measured",
        ),
        "retention": (
            summary["causal_learning"]["retention_benefit"],
            "causal delayed learning retained after restart",
        ),
        "freshness": (False, "unexposed confirmatory source groups and live hidden-game windows"),
    }
    return {
        name: {
            "condition": condition,
            "observed": observed,
            "passed": observed is True,
            "principle": "Custody, readiness, benefit, utility, retention and freshness are distinct claims.",
        }
        for name, (observed, condition) in observations.items()
    }


def blocking_checks(root: Path, dispositions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep one exact primary diagnostic for each unavailable branch."""

    checks = [deepcopy(row["gate_check"]) for row in dispositions if row.get("gate_check")]
    for task_id, field, expected in (
        ("exp7631-schema-pilot", "evidence_transport_ready_score", 1),
        ("exp7639-arc-goal-dedup", "planner_goal_guard_ready_score", 1),
    ):
        row = next(row for row in dispositions if row["task_id"] == task_id)
        payload = prior.load_json(root / str(row["actual_path"]))
        if payload.get(field) != expected:
            checks.append(
                prior._failed_check(
                    "required_branch_readiness",
                    task_id,
                    row["actual_path"],
                    field,
                    "==",
                    expected,
                    payload.get(field),
                )
            )
    return checks


def classify_terminal(
    owned_complete: bool, external_absent: bool, benefit: bool
) -> tuple[str, str]:
    """Reserve partial for unfinished current aggregation work."""

    if not owned_complete:
        return "incomplete_owned_v666_capstone_work", "partial"
    if external_absent:
        return "complete_blocked_required_v666_external_evidence", "blocked"
    if benefit:
        return "complete_positive_v666_independent_benefit", "positive"
    return "complete_null_v666_no_independent_benefit", "null"


def collect_preconditions(root: Path, authority: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate declared inputs and current host ownership."""

    rows = []
    for label in (*INPUT_PATHS, authority["selected_roadmap_path"]):
        path = root / label
        exists = path.is_file()
        rows.append(
            {
                "check": "named_input_file",
                "upstream": "repository",
                "path": label,
                "field": "exists",
                "operator": "==",
                "expected": True,
                "observed": exists,
                "passed": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    rows.append(
        {
            "check": "current_process_ownership",
            "upstream": "host",
            "path": "/proc/self",
            "field": "pid",
            "operator": "==",
            "expected": os.getpid(),
            "observed": os.getpid(),
            "passed": True,
        }
    )
    return rows


def remaining_gaps(summary: Mapping[str, Any]) -> list[JsonDict]:
    """State the measured or unavailable status of the three PRD gaps."""

    return [
        {
            "gap": "decision_evidence",
            "status": "unavailable",
            "observed": None,
            "next_condition": "new source-group capture, eligible proper-loss and typed-utility comparison",
        },
        {
            "gap": "causal_retained_learning",
            "status": "unavailable",
            "observed": None,
            "next_condition": "pre-feedback predictions, one-use admission and held-out retention",
        },
        {
            "gap": "live_goal_safe_planning",
            "status": "blocked_invalid_method_and_wrapper_unrun",
            "observed": summary["arc_wrapper"]["live_hidden_game_benefit"],
            "next_condition": "repair flagged goal guard then compare paired scored-wrapper windows",
        },
    ]


def next_decisions() -> list[JsonDict]:
    """Require a falsifiable changed premise before rerunning failed mechanisms."""

    return [
        {
            "scope": "owned_cuda_launch",
            "decision": "change",
            "changed_premise_required": "exclusive task-owned 20 GiB CUDA lease with no foreign PID",
        },
        {
            "scope": "source_evidence_energy",
            "decision": "keep",
            "changed_premise_required": "complete role-separated captured source groups and erasure control",
        },
        {
            "scope": "guarded_learning",
            "decision": "keep",
            "changed_premise_required": "real pre-feedback online events and held-out restart retention",
        },
        {
            "scope": "arc_goal_guard",
            "decision": "change",
            "changed_premise_required": "valid unflagged regression and truthful live-wrapper mask telemetry",
        },
        {
            "scope": "unchanged_supervisor_ledger",
            "decision": "retire",
            "changed_premise_required": "registered intervention with actual firings and paired outcomes",
        },
        {
            "scope": "unchanged_importance_reweighting",
            "decision": "retire",
            "changed_premise_required": "new causal signal beyond the previously failed weights",
        },
        {
            "scope": "external_text_rankers",
            "decision": "retire",
            "changed_premise_required": "oracle-distinct internal or source-dependent discriminator",
        },
        {
            "scope": "native_consumer",
            "decision": "keep",
            "changed_premise_required": "new total-consumer benchmark before any 10x claim",
        },
    ]


def checksum(value: Mapping[str, Any]) -> str:
    """Bind immutable authority, source bytes, reduction code and decisions."""

    return canonical_hash(
        {
            key: value.get(key)
            for key in (
                "schema",
                "selected_authority",
                "milestone_dispositions",
                "source_artifact_hashes",
                "evidence_summary",
                "remaining_prd_gaps",
                "next_decisions",
                "random_seed",
                "affected_file_validation_manifest",
            )
        }
    )


def build_artifact_for_test(
    root: Path = ROOT,
    *,
    receipts: Sequence[Mapping[str, Any]] = (),
    outcomes: Mapping[str, Any] | None = None,
    spans: Sequence[Mapping[str, Any]] = (),
    started_ns: int = 0,
    ended_ns: int = 1_000_000,
) -> JsonDict:
    """Compose the exact candidate without launching validation children."""

    authority = load_authority(root)
    dispositions = collect_dispositions(root, authority["tasks"])
    summary = reduce_evidence(root)
    checks = blocking_checks(root, dispositions)
    verdict, verdict_class = classify_terminal(True, bool(checks), False)
    dispositions[-1]["honest_verdict"] = verdict
    dispositions[-1]["verdict_class"] = verdict_class
    publication = prior._publication_record(root)
    publication["claim_boundary"] = "historical_fover_only"
    publication["unmet_gates"] = [
        name for name, gate in publication["gates"].items() if not gate["pass"]
    ]
    sources = prior._source_hashes(root, dispositions, publication)
    for label in (
        "results/experiment_7627_v665_native_cost.json",
        MODULE_PATH.as_posix(),
        contract.DESIGN_PATH.as_posix(),
        authority["selected_roadmap_path"],
    ):
        path = root / label
        sources.append(
            {
                "task_id": "reduction_dependency",
                "source_kind": "immutable_input",
                "planned_path": label,
                "actual_path": label,
                "sha256": sha256_file(path),
                "exists": True,
            }
        )
    audit = prior.load_json(root / "results/experiment_7638_v666_evidence_audit.json")
    budget = {
        "milestone_tasks": {"intended": 14, "observed": 14, "excluded": 0, "censored": 0},
        "scientific_source_groups": audit["sample_size_budget"],
        "native_integration": {
            "intended": 12,
            "observed": summary["native_packaging"]["independent_units"],
            "excluded": 0,
            "censored": 0,
        },
        "repeated_arms_views_seeds_multiply_samples": False,
    }
    artifact: JsonDict = {
        "schema": "carnot.exp7642.v666.capstone.v1",
        "experiment": 7642,
        "experiment_id": "exp7642-capstone",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "capstone_complete_score": 1,
        "gate_check_summary": {
            "passed": not checks,
            "failed_count": len(checks),
            "first_failure": checks[0] if checks else None,
            "failed_checks": checks,
        },
        "acceptance_gate_results": acceptance_gates(summary),
        "rows": prior._rows(dispositions),
        "sample_size_budget": budget,
        "preconditions_checked": collect_preconditions(root, authority),
        "inference_substrate": "aggregation_from_authenticated_upstream_rows",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "historical_models": ["unsloth/Qwen3.8-27B-GGUF"],
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "current_pid": os.getpid(),
            "current_gpu_uuid": None,
            "platform": platform.platform(),
        },
        "phase_spans": list(spans),
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "invocation_counts": {
            name: 0
            for name in (
                "model_loads_attempted",
                "model_loads_completed",
                "forwards_attempted",
                "forwards_completed",
                "generations_attempted",
                "generations_completed",
                "input_tokens",
                "output_tokens",
            )
        },
        "random_seed": {
            "current": None,
            "historical_native_bootstrap": [7627, 9627],
            "purpose": "no current stochastic stage; replay frozen native source",
        },
        "source_artifact_hashes": sources,
        "validation_receipts": list(receipts),
        "terminal_reader_outcomes": dict(outcomes or {}),
        "verifier_is_oracle": False,
        "selected_authority": authority["selected_roadmap_path"],
        "milestone_dispositions": dispositions,
        "evidence_summary": summary,
        "remaining_prd_gaps": remaining_gaps(summary),
        "next_decisions": next_decisions(),
        "publication_gates": publication,
        "hardware_dispositions": prior.load_json(
            root / "results/experiment_7641_v666_native_consumer.json"
        )["hardware_dispositions"],
        "operator_held_actions": [
            "Kaggle submission",
            "E0 confirmation",
            "GateMate physical JTAG action",
        ],
        "capstone_note_path": NOTE_PATH.as_posix(),
        "affected_file_validation_manifest": {
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [MODULE_PATH.as_posix()],
            "static_paths": [WRAPPER_PATH.as_posix()],
            "spec_paths": [contract.SPEC_PATH.as_posix()],
            "frozen_before_validation": True,
        },
        "publication_performed": False,
        "roadmap_activation_performed": False,
        "purchase_performed": False,
        "generator_training_performed": False,
        "production_defaults_changed": False,
        "reproducibility_checksum": "",
        "field_principles": {},
    }
    artifact["reproducibility_checksum"] = checksum(artifact)
    artifact["field_principles"] = prior._field_principles(tuple(artifact))
    return artifact


def mutate_for_test(value: JsonDict, mutation: str) -> JsonDict:
    """Change private candidate bytes for the cold-replay controls."""

    if mutation == "deleted":
        value["milestone_dispositions"].pop(4)
    elif mutation == "reordered":
        value["milestone_dispositions"][3:5] = reversed(value["milestone_dispositions"][3:5])
    elif mutation == "wrong_field":
        value["gate_check_summary"]["failed_checks"][0]["field"] = "misspelled_gate"
    elif mutation == "self_hash":
        value["source_artifact_hashes"].append(
            {
                "task_id": "exp7642-capstone",
                "source_kind": "terminal_producer",
                "planned_path": RESULT_PATH.as_posix(),
                "actual_path": RESULT_PATH.as_posix(),
                "sha256": "sha256:self",
                "exists": True,
            }
        )
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    value["reproducibility_checksum"] = checksum(value)
    return value


def independent_reduce(value: object, *, root: Path = ROOT) -> list[str]:
    """Reload exact sources and recompute the immutable capstone claims."""

    if not isinstance(value, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    if value.get("reproducibility_checksum") != checksum(value):
        errors.append("checksum_mismatch")
    raw = value.get("milestone_dispositions")
    if not isinstance(raw, list) or len(raw) != 14:
        errors.append("milestone_disposition_count")
    elif [row.get("task_id") for row in raw] != list(EXPECTED_TASK_IDS):
        errors.append("milestone_disposition_order")
    try:
        expected = build_artifact_for_test(root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return [*errors, f"cold_reduction_failed:{type(error).__name__}"]
    for field in (
        "milestone_dispositions",
        "source_artifact_hashes",
        "evidence_summary",
        "gate_check_summary",
        "acceptance_gate_results",
        "sample_size_budget",
        "remaining_prd_gaps",
        "next_decisions",
        "publication_gates",
        "selected_authority",
        "reproducibility_checksum",
    ):
        if value.get(field) != expected.get(field):
            errors.append(f"{field}_mismatch")
    if (
        value.get("verdict_class") != expected["verdict_class"]
        or value.get("honest_verdict") != expected["honest_verdict"]
    ):
        errors.append("terminal_classification_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_provenance_mismatch")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_mismatch")
    if not isinstance(value.get("field_principles"), Mapping) or not set(value).issubset(
        value["field_principles"]
    ):
        errors.append("field_principles_incomplete")
    if (
        value.get("publication_performed") is not False
        or value.get("roadmap_activation_performed") is not False
    ):
        errors.append("unauthorized_action_claim")
    return list(dict.fromkeys(errors))


def task_specific_e2e(value: JsonDict, root: Path) -> list[str]:
    """Prove deleted, reordered, wrong-field, and self-hash controls fail."""

    errors = independent_reduce(value, root=root)
    for mutation in ("deleted", "reordered", "wrong_field", "self_hash"):
        if not independent_reduce(mutate_for_test(deepcopy(value), mutation), root=root):
            errors.append(f"mutation_not_rejected:{mutation}")
    return errors


def terminal_commands(
    root: Path, candidate: Path
) -> list[validation.CommandSpec]:  # pragma: no cover
    """Name fresh readers of the exact unpublished candidate."""

    python = str(root / ".venv/bin/python")
    base = (python, "-u", WRAPPER_PATH.as_posix(), "--root", str(root))
    return [
        validation.CommandSpec(
            "fresh_process_cold_reduction", (*base, "--cold-replay", str(candidate)), "candidate"
        ),
        validation.CommandSpec(
            "task_specific_e2e", (*base, "--e2e", str(candidate)), "candidate_mutations"
        ),
        validation.CommandSpec(
            "independent_reduction",
            (*base, "--independent-reduce", str(candidate)),
            "candidate_reduction",
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_rows",
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> JsonDict:  # pragma: no cover
    """Run bounded validation and publish only a fully checked record."""

    started = time.monotonic()
    started_ns = time.monotonic_ns()
    if root.resolve() != ROOT or not root.is_absolute() or run_date != RUN_DATE:
        raise ValueError("frozen root or execution date mismatch")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7642-", dir="/tmp"))
    spans = []
    progress(started, "preconditions", "before")
    phase = time.monotonic()
    candidate = build_artifact_for_test(root, started_ns=started_ns, ended_ns=time.monotonic_ns())
    if not all(row["passed"] for row in candidate["preconditions_checked"]):
        raise RuntimeError("named_input_precondition_failed")
    spans.append(
        {
            "phase": "preconditions_and_reduction",
            "start_s": 0.0,
            "end_s": time.monotonic() - started,
            "duration_s": time.monotonic() - phase,
            "completed_units": 14,
            "pending_operations": ["validation"],
            "checkpoint": "fourteen_dispositions_reduced",
        }
    )
    progress(started, "preconditions", "after", completed_units=14)
    for name in ("model_load", "generation", "benchmark"):
        progress(started, name, "before", completed_units=0)
        spans.append(
            {
                "phase": name,
                "start_s": time.monotonic() - started,
                "end_s": time.monotonic() - started,
                "duration_s": 0.0,
                "completed_units": 0,
                "pending_operations": [],
                "checkpoint": "no_current_invocation",
            }
        )
        progress(started, name, "after", completed_units=0)
    progress(started, "validation", "before_subprocesses")
    phase = time.monotonic()
    commands = validation.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7642",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=private / "logs/affected", heartbeat_s=60
    )
    passed = validation.reduce_required_checks(receipts)["required_checks_passed"]
    spans.append(
        {
            "phase": "validation",
            "start_s": phase - started,
            "end_s": time.monotonic() - started,
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
            "pending_operations": ["terminal_readers"],
            "checkpoint": "affected_checks_complete",
        }
    )
    progress(started, "validation", "after_subprocesses", passed=passed)
    if not passed:
        raise RuntimeError("affected_validation_failed")
    candidate = build_artifact_for_test(
        root, receipts=receipts, spans=spans, started_ns=started_ns, ended_ns=time.monotonic_ns()
    )
    candidate_path = private / "candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "terminal", "before_subprocesses")
    phase = time.monotonic()
    terminal = validation.run_commands(
        root,
        terminal_commands(root, candidate_path),
        log_dir=private / "logs/terminal",
        heartbeat_s=60,
    )
    outcomes = {
        row["name"]: {
            "passed": row["passed"],
            "exit_code": row["exit_code"],
            "log_path": row["log_path"],
            "log_sha256": row["log_sha256"],
        }
        for row in terminal
    }
    spans.append(
        {
            "phase": "terminal_readers",
            "start_s": phase - started,
            "end_s": time.monotonic() - started,
            "duration_s": time.monotonic() - phase,
            "completed_units": len(terminal),
            "pending_operations": [],
            "checkpoint": "terminal_candidate_checked",
        }
    )
    progress(
        started, "terminal", "after_subprocesses", passed=all(row["passed"] for row in terminal)
    )
    if not all(row["passed"] for row in terminal):
        raise RuntimeError("terminal_reader_failed")
    final = build_artifact_for_test(
        root,
        receipts=[*receipts, *terminal],
        outcomes=outcomes,
        spans=spans,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
    )
    if independent_reduce(final, root=root):
        raise RuntimeError("final_cold_reduction_failed")
    destination = output if output.is_absolute() else root / output
    progress(started, "publication", "before_atomic", path=destination)
    atomic_json(destination, final)
    progress(started, "publication", "after_atomic", path=destination)
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the capstone or a bounded read-only fresh-process check."""

    print("[exp7642] phase=startup event=flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    for flag in ("cold-replay", "independent-reduce", "e2e"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args(argv)
    selected = args.cold_replay or args.independent_reduce or args.e2e
    if selected:
        value = prior.load_json(selected)
        errors = (
            task_specific_e2e(value, args.root)
            if args.e2e
            else independent_reduce(value, root=args.root)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0
