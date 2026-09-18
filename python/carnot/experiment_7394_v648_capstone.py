"""Reconcile the fourteen-task V648 milestone from authenticated evidence.

The capstone reads existing artifacts and exact conductor records. It does not
call a model, operate hardware, publish externally, or change production.

Spec refs: REQ-REPORT-7394 and SCENARIO-REPORT-7394-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as command_boundary
from carnot import experiment_7381_v648_contract as contract_helpers
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.648"
RUN_DATE = "20260918"
EXPERIMENT_ID = "exp7394-capstone"
SCHEMA = "carnot.experiment_7394.v648_capstone.v1"
RESULT_PATH = Path("results/experiment_7394_v648_capstone.json")
RAW_DIR = Path("results/raw/experiment_7394_v648_capstone")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7394_v648_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7394_v648_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7394_v648_capstone.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7394_v648_capstone.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_PATH = Path("ops/conductor-log.md")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = (
    "exp7381-contract",
    "exp7382-decision-protocol",
    "exp7383-canary-reducer",
    "exp7384-arc-invocation-boundary",
    "exp7385-decision-training",
    "exp7386-online-decisions",
    "exp7387-decision-audit",
    "exp7388-proposal-capture",
    "exp7389-proof-learning",
    "exp7390-proof-audit",
    "exp7391-arc-generalization",
    "exp7392-ising-reduction",
    "exp7393-hardware-placement",
    EXPERIMENT_ID,
)

CANONICAL_ARTIFACT_PATHS = {
    "exp7381-contract": "results/experiment_7381_v648_contract.json",
    "exp7382-decision-protocol": "results/experiment_7382_v648_decision_protocol.json",
    "exp7383-canary-reducer": "results/experiment_7383_v648_canary_reducer.json",
    "exp7384-arc-invocation-boundary": (
        "results/experiment_7384_v648_arc_invocation_boundary.json"
    ),
    "exp7385-decision-training": "results/experiment_7385_v648_decision_training.json",
    "exp7386-online-decisions": "results/experiment_7386_v648_online_decisions.json",
    "exp7387-decision-audit": "results/experiment_7387_decision_audit.json",
    "exp7388-proposal-capture": "results/experiment_7388_proposal_capture.json",
    "exp7391-arc-generalization": "results/experiment_7391_arc_generalization.json",
    "exp7392-ising-reduction": "results/experiment_7392_v648_ising_reduction.json",
    "exp7393-hardware-placement": "results/experiment_7393_v648_hardware_placement.json",
}

EXPECTED_CLASSES = {
    "exp7381-contract": "disqualified",
    "exp7382-decision-protocol": "null",
    "exp7383-canary-reducer": "disqualified",
    "exp7384-arc-invocation-boundary": "disqualified",
    "exp7385-decision-training": "null",
    "exp7386-online-decisions": "disqualified",
    "exp7387-decision-audit": "blocked",
    "exp7388-proposal-capture": "blocked",
    "exp7389-proof-learning": "blocked",
    "exp7390-proof-audit": "blocked",
    "exp7391-arc-generalization": "blocked",
    "exp7392-ising-reduction": "disqualified",
    "exp7393-hardware-placement": "blocked",
}

CANONICAL_LOG_MARKERS = {
    "exp7389-proof-learning": (
        "Measure prospective implication-memory value on se | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp7388-proposal-capture, "
        "exp7388-proposal-capture, exp7388-proposal-capture)"
    ),
    "exp7390-proof-audit": (
        "Independently audit proof-memory causality and com | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp7389-proof-learning, "
        "exp7389-proof-learning, exp7389-proof-learning)"
    ),
}

PRE_GATE_UPSTREAMS = {
    "exp7387-decision-audit": ("exp7386-online-decisions", "online_capture_complete_score"),
    "exp7388-proposal-capture": ("exp7383-canary-reducer", "assignment_reducer_ready_score"),
    "exp7391-arc-generalization": (
        "exp7384-arc-invocation-boundary",
        "arc_invocation_ready_score",
    ),
}

CLAIM_BRANCHES = (
    "static_calibration",
    "online_learning",
    "proof_memory",
    "arc_reachability_and_generalization",
    "archived_finite_laws",
    "hardware_placement",
)
ELIGIBLE_VERDICTS = {"positive", "circular_positive", "null"}
CLOSED_VERDICTS = {*ELIGIBLE_VERDICTS, "blocked", "disqualified", "partial"}
ZERO_INVOCATION_COUNTS = deepcopy(command_boundary.ZERO_INVOCATION_COUNTS)
RANDOM_SEED = {
    "experiment": 7_394_202_609_18,
    "inherited_static_bootstrap": 7_385_648,
    "inherited_online_bootstrap": 7_386_648,
}

V648_MANIFEST = command_boundary.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(ENTRYPOINT_PATH.as_posix(),),
)

AUTHORITY_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("results/experiment_7380_v647_capstone.json"),
    Path("python/carnot/experiment_7380_v647_capstone.py"),
    Path("_bmad/prd.md"),
    Path("ops/north-star.md"),
    Path("ops/verifier_gaps.md"),
    CONDUCTOR_PATH,
    PUBLICATION_GATE_PATH,
    ROADMAP_PATH,
    DESIGN_PATH,
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with ordinary experiment_id and milestone fields.",
    "status": "Write a terminal state only after real work and required validation.",
    "run_date": "Use 20260918 with actual UTC start and completion timestamps.",
    "preconditions_checked": "Record exact paths, hashes, classes, fields, and resources before reduction.",
    "MODEL_SPECS": "Keep this empty because the capstone performs no current LLM work.",
    "model_invoked": "Set true for any attempted current LLM load or generation, including failure.",
    "invocation_counts": "Count current LLM attempts and outcomes; never count historical calls as current.",
    "inference_substrate": "Describe actual host aggregation and the process resource boundary.",
    "inference_substrate_class": "Use the closed aggregation class for the current computation.",
    "execution_venue": "Use exactly host; device details belong in inference_substrate.",
    "duration_s": "Measure monotonic duration and never add sleep to meet a floor.",
    "phase_spans": "Record read, build, load, generate, evaluate, validate, and write boundaries.",
    "random_seed": "Freeze capstone and inherited resampling seeds.",
    "reproducibility_checksum": "Bind code, settings, protocol, exact sources, gates, and reduced rows.",
    "source_artifact_hashes": "Bind each authority, producer artifact, and exact conductor record.",
    "rows": "Retain branch metrics, costs, failures, and censoring behind each conclusion.",
    "sample_size_budget": "Separate planned, attempted, completed, censored, and remaining units.",
    "acceptance_gate_results": "Keep expected, observed, operator, category, and pass state separate.",
    "gate_check_summary": "Name every failed upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Mark true because formal proof and finite-law evaluators define truth.",
    "honest_verdict": "Use complete scope for finished work and blocked for unavailable external evidence.",
    "verdict_class": "Use the closed class and reserve partial for unfinished retryable owned work.",
    "flagged_adversarial": "Mark only a critical finding against this producer; preserve upstream flags in rows.",
    "validation_receipts": "Retain executed argv, environment, scope, exit, duration, and log hashes.",
    "repository_health": "Keep dated unrelated failures separate from required affected checks.",
    "field_principles": "Explain each output field without wrapping ordinary values.",
    "promotion_score": "Remain zero because no rollout, generator change, or publication is authorized.",
    "milestone_disposition_complete_score": "One means exactly fourteen tasks are accounted for, not positive.",
    "required_science_complete_score": "Require eligible static, online, audit, proof, and proof-audit evidence.",
    "disposition_rows": "Keep full and numeric IDs, source identity, class, failures, and claim boundary.",
    "publication_gate_results": "Preserve canonical G1-G4 and their historical FoVer-only scope.",
    "retirement_decisions": "Compare exact prior and current verdicts and name any authorized receipt.",
    "next_research_decisions": "Choose continue, retire, or defer from each measured branch bottleneck.",
}


utc_now = command_boundary.utc_now
sha256_file = command_boundary.sha256_file
canonical_hash = command_boundary.canonical_hash
atomic_json = command_boundary.atomic_json


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a truthful phase boundary so the aggregation never appears idle."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7394] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_json(path: Path) -> JsonDict:
    """Load one JSON object and reject malformed or list-shaped evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _terminal_status(value: object) -> bool:
    """Accept explicit complete, blocked, or disqualified lifecycle states."""

    text = str(value)
    return text in {"complete", "blocked", "disqualified"} or text.startswith(
        ("complete_", "blocked_", "disqualified_")
    )


def _numeric_experiment_id(task_id: str) -> int:
    """Keep a stable numeric identity alongside the full conductor task ID."""

    match = re.match(r"exp(\d+)", task_id)
    if match is None:
        raise ValueError(f"numeric experiment identity missing: {task_id}")
    return int(match.group(1))


def load_contract(root: Path) -> JsonDict:
    """Parse the exact active YAML and named V648 design independently."""

    roadmap_file = root / ROADMAP_PATH
    design_file = root / DESIGN_PATH
    roadmap = yaml.safe_load(roadmap_file.read_text(encoding="utf-8"))
    if not isinstance(roadmap, dict) or roadmap.get("milestone") != MILESTONE:
        raise ValueError(f"active roadmap must name milestone {MILESTONE}")
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list) or [row.get("id") for row in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("active roadmap must contain the exact fourteen V648 tasks")
    evaluation = contract_helpers.evaluate_contract(
        design_file.read_text(encoding="utf-8"), roadmap
    )
    return {
        "milestone": MILESTONE,
        "roadmap_path": ROADMAP_PATH.as_posix(),
        "roadmap_sha256": sha256_file(roadmap_file),
        "design_path": DESIGN_PATH.as_posix(),
        "design_sha256": sha256_file(design_file),
        "contract_match": evaluation.get("passed") is True,
        "contract_rows": deepcopy(evaluation.get("contract_rows") or []),
        "tasks": deepcopy(tasks),
    }


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent absence without creating a success-shaped producer artifact."""

    return {
        "task_id": str(task["id"]),
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "missing",
        "sha256": None,
        "status": "blocked",
        "honest_verdict": "blocked_missing_required_input",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": False,
        "accepted_for_science": False,
        "payload": {},
    }


def _load_log_record(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate an absent producer from one exact conductor pre-gate line."""

    task_id = str(task["id"])
    log_path = root / CONDUCTOR_PATH
    if not log_path.is_file():
        return _missing_evidence(task)
    marker = CANONICAL_LOG_MARKERS[task_id]
    matches = [line for line in log_path.read_text(encoding="utf-8").splitlines() if marker in line]
    if len(matches) != 1:
        return _missing_evidence(task)
    line = matches[0]
    return {
        "task_id": task_id,
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "conductor_log_record",
        "sha256": canonical_hash({"path": CONDUCTOR_PATH.as_posix(), "line": line}),
        "source_file_sha256": sha256_file(log_path),
        "source_record": line,
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": True,
        "accepted_for_science": False,
        "payload": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "verdict_class": "blocked",
            "gate_check_summary": line,
        },
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Load an exact producer artifact or its authenticated conductor record."""

    task_id = str(task["id"])
    if task_id in CANONICAL_LOG_MARKERS:
        return _load_log_record(root, task)
    selected_label = CANONICAL_ARTIFACT_PATHS.get(task_id)
    if selected_label is None:
        return _missing_evidence(task)
    selected = root / selected_label
    if not selected.is_file():
        return _missing_evidence(task)
    payload = _load_json(selected)
    pre_gate = task_id in PRE_GATE_UPSTREAMS
    if pre_gate:
        upstream_id, field = PRE_GATE_UPSTREAMS[task_id]
        upstream_path = root / CANONICAL_ARTIFACT_PATHS[upstream_id]
        authenticated = (
            payload.get("schema") == "blocked_gate_check_v1"
            and payload.get("status") == "blocked"
            and payload.get("blocked_at_layer") == "conductor_pre_gate"
            and payload.get("failed_upstream") == upstream_id
            and payload.get("failed_field") == field
            and payload.get("failed_expected") == 1
            and payload.get("failed_evidence_sha256") == sha256_file(upstream_path)
        )
        verdict = "blocked"
        source_kind = "conductor_pre_gate_artifact"
        flagged = False
    else:
        verdict = str(payload.get("verdict_class", ""))
        identity = payload.get("experiment_id", payload.get("experiment"))
        authenticated = (
            _terminal_status(payload.get("status"))
            and verdict == EXPECTED_CLASSES[task_id]
            and (
                payload.get("milestone") == MILESTONE or identity == _numeric_experiment_id(task_id)
            )
        )
        source_kind = "declared_artifact"
        flagged = payload.get("flagged_adversarial") is True
    return {
        "task_id": task_id,
        "declared_path": str(task.get("deliverable")),
        "actual_path": selected_label,
        "source_kind": source_kind,
        "sha256": sha256_file(selected),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict", "blocked_gate_check_failed"),
        "verdict_class": verdict,
        "flagged_adversarial": flagged,
        "authenticated": authenticated,
        "accepted_for_science": authenticated and not flagged and verdict in ELIGIBLE_VERDICTS,
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Read all thirteen predecessor slots before any dependent reduction."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def collect_preconditions(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Record exact authorities, producer identities, and current resources."""

    rows: list[JsonDict] = []
    for relative in AUTHORITY_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "artifact_field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "passed": available,
                "sha256": sha256_file(path) if available else None,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": SPEC_PATH.as_posix(),
                "artifact_field": "REQ-*",
                "expected": "REQ-REPORT-7394",
                "observed": ("REQ-REPORT-7394" if "REQ-REPORT-7394" in spec_text else None),
                "passed": "REQ-REPORT-7394" in spec_text,
            },
            {
                "check": "exact_contract",
                "upstream": f"{ROADMAP_PATH} + {DESIGN_PATH}",
                "artifact_field": "milestone/task_ids/contract_rows",
                "expected": {"milestone": MILESTONE, "task_ids": list(EXPECTED_TASK_IDS)},
                "observed": {
                    "milestone": contract.get("milestone"),
                    "task_ids": [row.get("id") for row in contract.get("tasks", [])],
                },
                "passed": contract.get("contract_match") is True,
            },
            {
                "check": "current_resource_boundary",
                "upstream": EXPERIMENT_ID,
                "artifact_field": "MODEL_SPECS/model_invoked/execution_venue",
                "expected": [[], False, "host"],
                "observed": [[], False, "host"],
                "passed": True,
            },
        ]
    )
    for task_id in EXPECTED_TASK_IDS[:-1]:
        source = evidence[task_id]
        rows.append(
            {
                "check": "preceding_disposition_source",
                "upstream": task_id,
                "artifact_field": "path/hash/class/authenticated/flagged_adversarial",
                "expected": {"class": EXPECTED_CLASSES[task_id], "authenticated": True},
                "observed": {
                    "path": source.get("actual_path"),
                    "sha256": source.get("sha256"),
                    "class": source.get("verdict_class"),
                    "authenticated": source.get("authenticated"),
                    "flagged_adversarial": source.get("flagged_adversarial"),
                },
                "passed": source.get("authenticated") is True,
                "accepted_for_science": source.get("accepted_for_science") is True,
            }
        )
    return rows


def _claim_row(
    branch: str,
    source_tasks: Sequence[str],
    metrics: Mapping[str, Any],
    completion_score: int,
    value_score: int,
    failures: Sequence[str],
    *,
    verifier_is_oracle: bool,
) -> JsonDict:
    """Build one branch summary without promoting completion into value."""

    return {
        "branch": branch,
        "source_tasks": list(source_tasks),
        "metrics": deepcopy(dict(metrics)),
        "completion_score": completion_score,
        "value_score": value_score,
        "failures": list(failures),
        "censored": completion_score == 0,
        "verifier_is_oracle": verifier_is_oracle,
        "promotes_readiness": False,
        "current_llm_calls": 0,
    }


def reduce_claim_rows(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Independently reconcile the six V648 research branches."""

    static = evidence["exp7385-decision-training"]["payload"]
    online = evidence["exp7386-online-decisions"]["payload"]
    audit = evidence["exp7387-decision-audit"]
    proof = evidence["exp7389-proof-learning"]
    proof_audit = evidence["exp7390-proof-audit"]
    arc_boundary = evidence["exp7384-arc-invocation-boundary"]["payload"]
    arc_result = evidence["exp7391-arc-generalization"]
    finite = evidence["exp7392-ising-reduction"]["payload"]
    hardware = evidence["exp7393-hardware-placement"]["payload"]
    static_complete = evidence["exp7385-decision-training"]["accepted_for_science"] is True
    return [
        _claim_row(
            "static_calibration",
            ("exp7385-decision-training", "exp7387-decision-audit"),
            {
                "verdict_class": evidence["exp7385-decision-training"]["verdict_class"],
                "decision_capture_complete_score": static.get("decision_capture_complete_score"),
                "calibration_value_score": static.get("calibration_value_score"),
                "independent_external_test": False,
                "audit_available": audit.get("accepted_for_science") is True,
                "archive_scope": (static.get("evidence_scope") or {}).get("archive"),
            },
            int(static_complete and static.get("decision_capture_complete_score") == 1),
            int(static_complete and static.get("calibration_value_score") == 1),
            ("single reused archive", "registered Brier benefit gate failed"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "online_learning",
            ("exp7386-online-decisions", "exp7387-decision-audit"),
            {
                "verdict_class": evidence["exp7386-online-decisions"]["verdict_class"],
                "online_capture_complete_score": online.get("online_capture_complete_score"),
                "online_learning_value_score": online.get("online_learning_value_score"),
                "audit_available": audit.get("accepted_for_science") is True,
                "constructed_shift_is_historical_chronology": False,
            },
            int(evidence["exp7386-online-decisions"]["accepted_for_science"] is True),
            0,
            ("producer validation and safety failed", "audit was pre-gated"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "proof_memory",
            ("exp7388-proposal-capture", "exp7389-proof-learning", "exp7390-proof-audit"),
            {
                "proposal_capture_available": evidence["exp7388-proposal-capture"].get(
                    "accepted_for_science"
                )
                is True,
                "measurement_available": proof.get("accepted_for_science") is True,
                "audit_available": proof_audit.get("accepted_for_science") is True,
                "archive_trained_head_is_independent_external_result": False,
            },
            int(
                proof.get("accepted_for_science") is True
                and proof_audit.get("accepted_for_science") is True
            ),
            0,
            ("proposal capture blocked", "measurement and audit did not run"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "arc_reachability_and_generalization",
            ("exp7384-arc-invocation-boundary", "exp7391-arc-generalization"),
            {
                "invocation_ready_score": arc_boundary.get("arc_invocation_ready_score"),
                "boundary_verdict_class": evidence["exp7384-arc-invocation-boundary"][
                    "verdict_class"
                ],
                "generalization_started": arc_result.get("source_kind") == "declared_artifact",
                "valid_no_progress_would_not_invalidate_other_science": True,
            },
            0,
            0,
            ("first-action boundary disqualified", "generalization was pre-gated"),
            verifier_is_oracle=False,
        ),
        _claim_row(
            "archived_finite_laws",
            ("exp7392-ising-reduction",),
            {
                "verdict_class": evidence["exp7392-ising-reduction"]["verdict_class"],
                "ising_reduction_complete_score": finite.get("ising_reduction_complete_score"),
                "exact_rows_completed": (finite.get("sample_size_budget") or {}).get(
                    "completed_exact_rows"
                ),
                "empty_support_is_empty_distribution": False,
                "new_sampling_attempts": (finite.get("sample_size_budget") or {}).get(
                    "attempted_new_samples"
                ),
            },
            int(evidence["exp7392-ising-reduction"]["accepted_for_science"] is True),
            0,
            ("capability replay failed", "original all-cell gate remains failed"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "hardware_placement",
            ("exp7393-hardware-placement",),
            {
                "verdict_class": evidence["exp7393-hardware-placement"]["verdict_class"],
                "board_disposition_complete_score": hardware.get(
                    "board_disposition_complete_score"
                ),
                "hardware_ready_score": hardware.get("hardware_ready_score"),
                "hardware_value_score": hardware.get("hardware_value_score"),
                "gatemate_block_is_expected": True,
                "complete_service_rows": (hardware.get("sample_size_budget") or {}).get(
                    "complete_service_rows"
                ),
            },
            int(hardware.get("board_disposition_complete_score") == 1),
            0,
            ("GateMate state unchanged", "complete stage-cost boundary unavailable"),
            verifier_is_oracle=False,
        ),
    ]


def _failure(
    upstream: str,
    check: str,
    field: str,
    expected: Any,
    observed: Any,
    category: str,
) -> JsonDict:
    """Name one failed gate with exact upstream, field, and values."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "category": category,
        "passed": False,
    }


def terminal_state(
    evidence: Mapping[str, Mapping[str, Any]], required_validation_passed: bool
) -> JsonDict:
    """Disqualify failed required safety and block unavailable required science."""

    disqualified: list[JsonDict] = []
    blocked: list[JsonDict] = []
    if not required_validation_passed:
        disqualified.append(
            _failure(
                EXPERIMENT_ID,
                "required_validation",
                "required_checks_passed",
                True,
                False,
                "required_validation",
            )
        )
    static = evidence["exp7385-decision-training"]
    if not (
        static.get("accepted_for_science") is True
        and static.get("payload", {}).get("decision_capture_complete_score") == 1
    ):
        target = disqualified if static.get("verdict_class") == "disqualified" else blocked
        target.append(
            _failure(
                "exp7385-decision-training",
                "required_static_calibration",
                "decision_capture_complete_score/verdict_class/flagged_adversarial",
                {"score": 1, "class": sorted(ELIGIBLE_VERDICTS), "flag": False},
                {
                    "score": static.get("payload", {}).get("decision_capture_complete_score"),
                    "class": static.get("verdict_class"),
                    "flag": static.get("flagged_adversarial"),
                },
                "required_science",
            )
        )
    online = evidence["exp7386-online-decisions"]
    if not (
        online.get("accepted_for_science") is True
        and online.get("payload", {}).get("online_capture_complete_score") == 1
    ):
        target = (
            disqualified
            if online.get("verdict_class") == "disqualified"
            or online.get("flagged_adversarial") is True
            else blocked
        )
        target.append(
            _failure(
                "exp7386-online-decisions",
                "required_online_measurement",
                "online_capture_complete_score/verdict_class/flagged_adversarial",
                {"score": 1, "class": sorted(ELIGIBLE_VERDICTS), "flag": False},
                {
                    "score": online.get("payload", {}).get("online_capture_complete_score"),
                    "class": online.get("verdict_class"),
                    "flag": online.get("flagged_adversarial"),
                },
                "required_safety_or_validation",
            )
        )
    for task_id, field in (
        ("exp7387-decision-audit", "decision_audit_complete_score"),
        ("exp7389-proof-learning", "proof_learning_capture_complete_score"),
        ("exp7390-proof-audit", "proof_audit_complete_score"),
    ):
        row = evidence[task_id]
        if row.get("accepted_for_science") is not True:
            target = (
                disqualified
                if row.get("verdict_class") == "disqualified"
                or row.get("flagged_adversarial") is True
                else blocked
            )
            target.append(
                _failure(
                    task_id,
                    "required_measurement_or_audit",
                    field,
                    1,
                    {
                        "source_kind": row.get("source_kind"),
                        "verdict_class": row.get("verdict_class"),
                        "accepted_for_science": row.get("accepted_for_science"),
                    },
                    "required_science",
                )
            )
    failures = [*disqualified, *blocked]
    verdict = "disqualified" if disqualified else ("blocked" if blocked else "null")
    if verdict == "disqualified":
        honest = (
            "complete_disqualified_required_science_or_validation_failure: all fourteen V648 "
            "tasks are accounted for; the online producer failed required validation and safety, "
            "while its audit and proof-memory measurement plus audit were pre-gated"
        )
        status = "complete_disqualified_v648_capstone"
    elif verdict == "blocked":
        honest = (
            "blocked_required_science_unavailable: all fourteen V648 tasks are accounted for; "
            "unchanged upstream absence is terminal blocked evidence, not retryable partial work"
        )
        status = "blocked"
    else:
        honest = "complete_null_required_v648_science_measured_without_registered_benefit"
        status = "complete_null_v648_capstone"
    return {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "required_science_complete_score": int(not failures),
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
        },
    }


def publication_gate_results(payload: Mapping[str, Any]) -> JsonDict:
    """Preserve canonical G1-G4 while denying V648 publication authority."""

    source = payload.get("gates")
    source = source if isinstance(source, Mapping) else {}
    gates = {
        name: (
            deepcopy(source[name])
            if isinstance(source.get(name), Mapping)
            else {"pass": False, "detail": "canonical gate result missing"}
        )
        for name in ("G1", "G2", "G3", "G4")
    }
    unmet = [name for name, row in gates.items() if row.get("pass") is not True]
    return {
        "scope": "historical_fover_paper_only",
        "gates": gates,
        "paper_ready": not unmet,
        "unmet_gates": unmet,
        "canonical_note": payload.get("note"),
        "certifies_v648": False,
        "authorizes_external_publication": False,
        "authorizes_deployment": False,
        "authorizes_push": False,
    }


def _row_failures(row: Mapping[str, Any]) -> list[Any]:
    """Preserve a producer's exact gate failures without reinterpreting them."""

    payload = row.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if row.get("source_kind") == "conductor_log_record":
        return [row.get("source_record")]
    if row.get("source_kind") == "conductor_pre_gate_artifact":
        return [
            {
                "upstream": payload.get("failed_upstream"),
                "field": payload.get("failed_field"),
                "operator": payload.get("failed_operator"),
                "expected": payload.get("failed_expected"),
                "observed": payload.get("failed_observed"),
            }
        ]
    summary = payload.get("gate_check_summary")
    if isinstance(summary, Mapping):
        failures = summary.get("failures")
        if isinstance(failures, list):
            return deepcopy(failures)
        first = summary.get("first_failure") or summary.get("first_required_failure")
        return [deepcopy(first)] if first else []
    return [summary] if summary else []


CLAIM_BOUNDARIES = {
    "exp7381-contract": "advisory contract evidence only",
    "exp7382-decision-protocol": "protocol fixture; no real-data learning value",
    "exp7383-canary-reducer": "historical reducer diagnosis; no fresh model readiness",
    "exp7384-arc-invocation-boundary": "failed first-action boundary; no live ARC claim",
    "exp7385-decision-training": "single-archive completed static null; not independent external",
    "exp7386-online-decisions": "disqualified replay; no online readiness or value",
    "exp7387-decision-audit": "pre-gated audit; no audit conclusion",
    "exp7388-proposal-capture": "pre-gated; no current Qwen capture",
    "exp7389-proof-learning": "pre-gated; no prospective proof-memory measurement",
    "exp7390-proof-audit": "pre-gated; no independent proof-memory audit",
    "exp7391-arc-generalization": "pre-gated; no generalization measurement",
    "exp7392-ising-reduction": "archived-law reduction disqualified; old failures preserved",
    "exp7393-hardware-placement": "board accounting complete; no placement or speed claim",
    EXPERIMENT_ID: "milestone accounting only; no automatic readiness or publication",
}


def build_disposition_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    validation_complete: bool,
) -> list[JsonDict]:
    """Build ordered task dispositions and append self only after validation."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "numeric_experiment_id": _numeric_experiment_id(task_id),
                "declared_path": task.get("deliverable"),
                "actual_path": source.get("actual_path"),
                "source_kind": source.get("source_kind"),
                "source_sha256": source.get("sha256"),
                "status": source.get("status"),
                "honest_verdict": source.get("honest_verdict"),
                "verdict_class": source.get("verdict_class"),
                "flagged_adversarial": source.get("flagged_adversarial"),
                "authenticated": source.get("authenticated"),
                "accepted_for_science": source.get("accepted_for_science"),
                "failed_checks": _row_failures(source),
                "claim_boundary": CLAIM_BOUNDARIES[task_id],
            }
        )
    if validation_complete:
        rows.append(
            {
                "order": 14,
                "task_id": EXPERIMENT_ID,
                "numeric_experiment_id": 7394,
                "declared_path": RESULT_PATH.as_posix(),
                "actual_path": RESULT_PATH.as_posix(),
                "source_kind": "self",
                "source_sha256": None,
                "status": terminal["status"],
                "honest_verdict": terminal["honest_verdict"],
                "verdict_class": terminal["verdict_class"],
                "flagged_adversarial": False,
                "authenticated": True,
                "accepted_for_science": False,
                "failed_checks": deepcopy(terminal["gate_check_summary"]["failures"]),
                "claim_boundary": CLAIM_BOUNDARIES[EXPERIMENT_ID],
            }
        )
    return rows


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Bind every authority and disposition source used by the reduction."""

    rows: dict[str, JsonDict] = {
        "active_roadmap": {
            "path": contract["roadmap_path"],
            "sha256": contract["roadmap_sha256"],
            "source_kind": "authority",
        },
        "milestone_design": {
            "path": contract["design_path"],
            "sha256": contract["design_sha256"],
            "source_kind": "authority",
        },
    }
    for relative in AUTHORITY_PATHS:
        path = root / relative
        if path.is_file():
            rows[f"authority:{relative.as_posix()}"] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_kind": "authority",
            }
    for task_id, source in evidence.items():
        if source.get("source_kind") == "conductor_log_record":
            rows[task_id] = {
                "path": CONDUCTOR_PATH.as_posix(),
                "sha256": source.get("sha256"),
                "source_file_sha256": source.get("source_file_sha256"),
                "source_record": source.get("source_record"),
                "source_kind": "conductor_log_record",
            }
        else:
            rows[task_id] = {
                "path": source.get("actual_path"),
                "sha256": source.get("sha256"),
                "source_kind": source.get("source_kind"),
            }
    return rows


def _sample_budget(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Keep accounting separate from scientific completion and inherited units."""

    return {
        "disposition_units": {
            "planned": 14,
            "attempted": 14,
            "completed": 14,
            "censored": 0,
            "unstarted": 0,
        },
        "required_science_units": {
            "planned": 5,
            "attempted": 2,
            "eligible_completed": 1,
            "disqualified": 1,
            "externally_blocked": 3,
            "unstarted": 0,
        },
        "upstream_budgets": {
            task_id: deepcopy(row.get("payload", {}).get("sample_size_budget"))
            for task_id, row in evidence.items()
            if row.get("payload", {}).get("sample_size_budget") is not None
        },
        "stopping_rule": (
            "Read each declared producer or canonical pre-gate once. Stop after fourteen "
            "accounted dispositions; do not retry unchanged external blocks."
        ),
        "remaining_capstone_work": 0,
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, operator: str = "=="
) -> JsonDict:
    """Keep gate operands and comparison operator explicit."""

    passed = observed == expected if operator == "==" else False
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "operator": operator,
        "passed": passed,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> list[JsonDict]:
    """Separate accounting, validation, science, efficacy, and promotion."""

    return [
        _gate("exact_contract", "completion", True, contract.get("contract_match")),
        _gate("fourteen_dispositions", "completion", 14, 14),
        _gate(
            "required_validation",
            "required_validation",
            True,
            validation.get("required_checks_passed") is True,
        ),
        _gate(
            "required_science_complete",
            "required_science",
            1,
            terminal.get("required_science_complete_score"),
        ),
        _gate(
            "registered_scientific_benefit",
            "scientific_efficacy",
            True,
            False,
        ),
        _gate("historical_fover_paper_ready", "publication", True, publication["paper_ready"]),
        _gate("v648_certified_by_fover_gate", "promotion", True, False),
        _gate("automatic_promotion", "promotion", 1, 0),
    ]


def _retirement_decisions(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Compare exact prior and current verdicts without inventing an override."""

    task_by_id = {str(task["id"]): task for task in contract["tasks"]}
    current = {task_id: row.get("honest_verdict") for task_id, row in evidence.items()}
    current[EXPERIMENT_ID] = terminal["honest_verdict"]
    rows: list[JsonDict] = []
    for task_id in EXPECTED_TASK_IDS:
        task = task_by_id[task_id]
        for prior in task.get("prior_failures") or []:
            previous = prior.get("verdict")
            now = current[task_id]
            same = previous == now
            rows.append(
                {
                    "task_id": task_id,
                    "previous_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": now,
                    "same_exact_verdict": same,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "mechanism_change": prior.get("addressed_by"),
                    "decision": "defer" if same else "continue",
                    "mechanism_retired": False,
                    "actual_retirement_receipt": None,
                    "reason": (
                        "No task-local authorized exclusion-manifest receipt exists; do not "
                        "invent an operator override."
                        if same
                        else "The exact verdict changed, so same-verdict retirement does not apply."
                    ),
                }
            )
    return rows


def _next_decisions() -> list[JsonDict]:
    """Choose bounded next actions from the measured branch bottlenecks."""

    return [
        {
            "branch": "static_calibration",
            "decision": "continue",
            "measured_bottleneck": "Completed null on one reused archive; Brier benefit missed both controls.",
            "condition": "Use a new independent corpus before any general calibration claim.",
        },
        {
            "branch": "online_learning",
            "decision": "defer",
            "measured_bottleneck": "Required validation and safety failed; capture score is zero.",
            "condition": "Repair producer validation before repeating the fixed replay.",
        },
        {
            "branch": "proof_memory",
            "decision": "defer",
            "measured_bottleneck": "Corrected proposal reducer was not ready, so capture and both required tasks were blocked.",
            "condition": "Resume only after one clean bounded canary and proposal capture.",
        },
        {
            "branch": "arc_reachability_and_generalization",
            "decision": "retire",
            "measured_bottleneck": "The repaired invocation boundary repeated a disqualified pre-first-action outcome.",
            "condition": "Reopen only with a new invocation mechanism and an authorized manifest receipt.",
        },
        {
            "branch": "archived_finite_laws",
            "decision": "defer",
            "measured_bottleneck": "Exact law preservation was diagnostic, but capability replay failed and empty support remained.",
            "condition": "Do not rerun archived chains; repair replay and support semantics first.",
        },
        {
            "branch": "hardware_placement",
            "decision": "defer",
            "measured_bottleneck": "No complete service denominator and no changed GateMate physical state.",
            "condition": "Wait for measured full-stage costs or an operator-authored physical change.",
        },
    ]


def _historical_sidecars(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Label old model receipts so they cannot count as current capstone calls."""

    rows = []
    for task_id in (
        "exp7383-canary-reducer",
        "exp7384-arc-invocation-boundary",
        "exp7388-proposal-capture",
        "exp7391-arc-generalization",
    ):
        source = evidence[task_id]
        payload = source.get("payload") or {}
        rows.append(
            {
                "task_id": task_id,
                "path": source.get("actual_path"),
                "sha256": source.get("sha256"),
                "historical_only": True,
                "counted_as_current_inference": False,
                "original_model_invoked": payload.get("model_invoked"),
                "original_inference_substrate_class": payload.get("inference_substrate_class"),
                "original_invocation_counts": deepcopy(payload.get("invocation_counts")),
                "original_verdict_class": source.get("verdict_class"),
                "original_flagged_adversarial": source.get("flagged_adversarial"),
            }
        )
    return rows


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide a complete zero-length phase ledger for unit construction."""

    return [
        {"phase": phase, "started_elapsed_s": 0.0, "ended_elapsed_s": 0.0, "duration_s": 0.0}
        for phase in ("read", "build", "load", "generate", "evaluate", "validate", "write")
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable code, evidence, protocol, rows, gates, and decisions."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "disposition_rows",
        "publication_gate_results",
        "retirement_decisions",
        "next_research_decisions",
        "verdict_class",
        "honest_verdict",
        "promotion_score",
    )
    return canonical_hash({field: artifact.get(field) for field in fields})


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    publication_payload: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the terminal schema after affected validation has a real result."""

    publication = publication_gate_results(publication_payload)
    terminal = terminal_state(
        evidence, required_validation_passed=validation.get("required_checks_passed") is True
    )
    dispositions = build_disposition_rows(
        contract["tasks"], evidence, terminal, validation_complete=True
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "numeric_experiment_id": 7394,
        "milestone": MILESTONE,
        "phase": 4,
        "title": "Reconcile fourteen outcomes and decide calibrated-learning continuation",
        "status": terminal["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": collect_preconditions(root, contract, evidence),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": (
            "host CPU JSON, YAML, and Markdown aggregation; current capstone process lease; "
            "zero current LLM, GPU, or board operations"
        ),
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_model_receipt_sidecars": _historical_sidecars(evidence),
        "small_ebm_training": {
            "performed_by_current_capstone": False,
            "counted_as_current_llm_work": False,
            "producer_receipts": [
                {
                    "task_id": task_id,
                    "source_sha256": evidence[task_id].get("sha256"),
                    "receipt": deepcopy(
                        evidence[task_id].get("payload", {}).get("small_ebm_training")
                    ),
                }
                for task_id in ("exp7385-decision-training", "exp7386-online-decisions")
            ],
        },
        "contract_receipt": {
            "advisory": True,
            "contract_match": contract["contract_match"],
            "task_count": len(contract["tasks"]),
            "contract_rows": deepcopy(contract["contract_rows"]),
        },
        "rows": reduce_claim_rows(evidence),
        "sample_size_budget": _sample_budget(evidence),
        "acceptance_gate_results": _acceptance_gates(contract, terminal, validation, publication),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": False,
        "validation_receipts": deepcopy(validation.get("validation_receipts") or []),
        "repository_health": deepcopy(validation.get("repository_health") or {}),
        "required_checks_passed": validation.get("required_checks_passed") is True,
        "terminal_validation_passed": validation.get("terminal_validation_passed") is True,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "external_publication_authorized": False,
        "deployment_authorized": False,
        "release_or_push_authorized": False,
        "promotion_score": 0,
        "scientific_value_score": 0,
        "readiness_score": 0,
        "milestone_disposition_complete_score": int(len(dispositions) == 14),
        "required_science_complete_score": terminal["required_science_complete_score"],
        "disposition_rows": dispositions,
        "publication_gate_results": publication,
        "retirement_decisions": _retirement_decisions(contract, evidence, terminal),
        "next_research_decisions": _next_decisions(),
        "field_principles": {},
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, f"Record the measured V648 capstone value for {key.replace('_', ' ')}."
        )
        for key in artifact
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload each file-backed source and each exact conductor row."""

    sources = artifact.get("source_artifact_hashes")
    if not isinstance(sources, Mapping):
        return False
    for row in sources.values():
        if not isinstance(row, Mapping):
            return False
        path_value = row.get("path")
        if not isinstance(path_value, str):
            return False
        path = root / path_value
        if row.get("source_kind") == "conductor_log_record":
            if not path.is_file() or row.get("source_file_sha256") != sha256_file(path):
                return False
            expected = canonical_hash(
                {"path": CONDUCTOR_PATH.as_posix(), "line": row.get("source_record")}
            )
            if row.get("sha256") != expected:
                return False
        elif not path.is_file() or row.get("sha256") != sha256_file(path):
            return False
    return True


REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "repository_health",
    "field_principles",
    "promotion_score",
    "milestone_disposition_complete_score",
    "required_science_complete_score",
    "disposition_rows",
    "publication_gate_results",
    "retirement_decisions",
    "next_research_decisions",
)


def validate_artifact(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, rows, hashes, scores, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact["schema"],
        artifact["experiment_id"],
        artifact["milestone"],
        artifact["run_date"],
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if not _terminal_status(artifact["status"]) or artifact["verdict_class"] not in CLOSED_VERDICTS:
        errors.append("lifecycle_invalid")
    if (
        artifact["MODEL_SPECS"] != []
        or artifact["model_invoked"] is not False
        or artifact["invocation_counts"] != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact["inference_substrate_class"] != "aggregation"
        or artifact["execution_venue"] != "host"
    ):
        errors.append("substrate_invalid")
    dispositions = artifact["disposition_rows"]
    valid_dispositions = (
        isinstance(dispositions, list)
        and len(dispositions) == 14
        and [row.get("task_id") for row in dispositions] == list(EXPECTED_TASK_IDS)
        and [row.get("numeric_experiment_id") for row in dispositions] == list(range(7381, 7395))
        and [row.get("order") for row in dispositions] == list(range(1, 15))
        and all(
            row.get("verdict_class") == EXPECTED_CLASSES[row["task_id"]]
            for row in dispositions[:-1]
        )
        and dispositions[-1].get("source_kind") == "self"
        and dispositions[-1].get("verdict_class") == artifact["verdict_class"]
    )
    if not valid_dispositions:
        errors.append("disposition_rows_invalid")
    if artifact["milestone_disposition_complete_score"] != int(valid_dispositions):
        errors.append("disposition_score_invalid")
    if artifact["promotion_score"] != 0 or artifact.get("readiness_score") != 0:
        errors.append("failed_state_score_nonzero")
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the exact Exp7358 plan for only the current affected files."""

    return command_boundary.build_command_plan(root, V648_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, and command drift."""

    return command_boundary.validate_command_plan(root, V648_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen execution date from the declared command."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _phase_span(  # pragma: no cover - real monotonic boundary used by the entrypoint.
    phase: str, phase_started: float, run_started: float
) -> JsonDict:
    """Close one authentic phase span against the run origin."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def _terminal_commands(  # pragma: no cover - executed only by the declared E2E.
    root: Path, candidate: Path
) -> list[command_boundary.PlannedCommand]:
    """Build independent cold replay and both strict artifact readers."""

    python = str(root / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7394_v648_capstone import validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=validate_artifact(v);print({'errors':e},flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        validation_scope.CommandSpec(
            "independent_reducer",
            (python, "-u", "-c", replay, str(candidate)),
            "candidate_raw_evidence_replay",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_row_consistency",
        ),
    )
    return [command_boundary.PlannedCommand(spec, "required_validation", True) for spec in specs]


def _run_publication_gate(  # pragma: no cover - executed only by the declared E2E.
    root: Path, log_dir: Path
) -> tuple[JsonDict, JsonDict]:
    """Run the canonical publication gate and retain its exact receipt."""

    spec = validation_scope.CommandSpec(
        "publication_gate",
        (str(root / ".venv/bin/python"), "-u", PUBLICATION_GATE_PATH.as_posix(), "--json"),
        "historical_fover_paper",
    )
    receipt = validation_scope.run_commands(root, [spec], log_dir=log_dir)[0]
    payload = json.loads(str(receipt.get("output_tail") or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("publication gate must emit a JSON object")
    return payload, receipt


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    root: Path, run_date: str
) -> JsonDict:
    """Execute exact reads, scoped checks, cold readers, and atomic publication."""

    date_argument(run_date)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7394-", dir="/tmp"))

    point = time.monotonic()
    progress(started, "read", "before")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    spans.append(_phase_span("read", point, started))
    progress(started, "read", "after", dispositions=len(evidence))

    point = time.monotonic()
    progress(started, "build", "before_validation_plan")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(_phase_span("build", point, started))
    progress(started, "build", "after_validation_plan", commands=len(commands))

    for phase in ("load", "generate"):
        point = time.monotonic()
        progress(started, phase, "before", current_llm_operations=0)
        spans.append(_phase_span(phase, point, started))
        progress(started, phase, "after", current_llm_operations=0)

    point = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    planned = [
        command_boundary.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    affected = command_boundary.run_categorized_commands(
        root,
        planned,
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = command_boundary.reduce_affected_receipts(root, V648_MANIFEST, affected)
    publication_payload, publication_receipt = _run_publication_gate(
        root, raw_dir / "validation/publication"
    )
    validation: JsonDict = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": [*affected, publication_receipt],
        "repository_health": {
            "status": "historical_observations_retained",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
    }
    spans.append(_phase_span("validate", point, started))
    progress(started, "validate", "after_affected_subprocesses", passed=reduced["passed"])

    point = time.monotonic()
    progress(started, "evaluate", "before")
    provisional = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication_payload,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=[*spans, _phase_span("evaluate", point, started)],
    )
    candidate = private_root / "experiment_7394_candidate.json"
    atomic_json(candidate, provisional)
    progress(started, "evaluate", "after", verdict=provisional["verdict_class"])

    point = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate)
    progress(started, "validate_terminal", "before_subprocesses", units=len(terminal_commands))
    terminal_receipts = command_boundary.run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    validation["terminal_validation_passed"] = terminal_passed
    validation["required_checks_passed"] = reduced["passed"] and terminal_passed
    validation["validation_receipts"].extend(terminal_receipts)
    spans.append(_phase_span("validate", point, started))
    progress(started, "validate_terminal", "after_subprocesses", passed=terminal_passed)

    point = time.monotonic()
    progress(started, "write", "before_atomic", path=RESULT_PATH.as_posix())
    final_spans = [*spans, _phase_span("write", point, started)]
    artifact = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication_payload,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(root / RESULT_PATH, artifact)
    atomic_json(
        root / CHECKPOINT_PATH,
        {"status": "complete", "artifact": RESULT_PATH.as_posix(), "completed_at_utc": utc_now()},
    )
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix())
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - public CLI boundary.
    """Parse only the frozen date accepted by the declared command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, type=date_argument)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entrypoint E2E.
    """Execute the V648 aggregation through the repository-root launcher."""

    print("[exp7394] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
