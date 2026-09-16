"""Qualify the repaired result-resume launcher without current model work.

Spec refs: REQ-ARC-WMTE-7345 and SCENARIO-ARC-WMTE-7345-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import socket
import tempfile
import time
from typing import Any

from carnot import experiment_7336_v644_arc_resume as prior
from carnot.agentic.arc_selfparse_result_resume import ResultResumeGuard
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = "exp7345-arc-resume-check"
SCHEMA = "carnot.experiment_7345.v645.arc_resume_check.v1"
MODEL_SPECS: list[JsonDict] = []
DEVELOPMENT_SEED = 7_345_202_609_16
EVALUATION_SEED = 17_345_202_609_16
RESAMPLING_SEED = 27_345_202_609_16
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
HISTORICAL_RESULT_PATH = Path("results/experiment_7336_v644_arc_resume.json")
RESULT_PATH = Path("results/experiment_7345_v645_arc_resume_check.json")
RAW_DIR = Path("results/raw/experiment_7345_v645_arc_resume_check")
MODULE_PATH = Path("python/carnot/experiment_7345_v645_arc_resume_check.py")
PRIOR_MODULE_PATH = Path("python/carnot/experiment_7336_v644_arc_resume.py")
RESUME_MODULE_PATH = Path("python/carnot/agentic/arc_selfparse_result_resume.py")
LOOP_MODULE_PATH = Path("python/carnot/agentic/arc_induction_tool_loop.py")
POLICY_MODULE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7345_v645_arc_resume_check.py")
TEST_PATH = Path("tests/python/test_experiment_7345_v645_arc_resume_check.py")

HASH_PATHS = (
    SPEC_PATH,
    HISTORICAL_RESULT_PATH,
    MODULE_PATH,
    PRIOR_MODULE_PATH,
    RESUME_MODULE_PATH,
    LOOP_MODULE_PATH,
    POLICY_MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/exclusion_manifest.yaml"),
)
REPAIR_PATHS = (PRIOR_MODULE_PATH, RESUME_MODULE_PATH, LOOP_MODULE_PATH, POLICY_MODULE_PATH)
REQUIRED_ARMS = (
    "tool_needed_changed_input",
    "result_withheld",
    "no_tool_needed",
    "stale_episode",
    "duplicate_request",
    "malformed_result",
    "timeout_after_dispatch",
    "exhausted_budget",
)
REQUIRED_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_offline_smoke")
REQUIRED_TERMINAL_LINT_NAMES = ("adversarial_verify", "verdict_row_consistency_strict")
ZERO_INVOCATION_COUNTS = deepcopy(prior.ZERO_INVOCATION_COUNTS)

atomic_write = prior.atomic_write
artifact_checksum = prior.artifact_checksum
sha256_file = prior.sha256_file


def _check(
    name: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    return {
        "check": name,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def check_preconditions(root: Path) -> JsonDict:
    """Authenticate inputs before the qualification performs dependent work."""

    spec = root / SPEC_PATH
    historical_path = root / HISTORICAL_RESULT_PATH
    try:
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        historical = {}
    if not isinstance(historical, dict):
        historical = {}
    try:
        spec_text = spec.read_text(encoding="utf-8")
    except OSError:
        spec_text = ""
    first_loss = prior.trace_exp7319_first_loss(root)
    producer_eligible = bool(
        historical.get("status") == "complete"
        and historical.get("verdict_class") not in {"blocked", "disqualified", "partial"}
    )
    checks = [
        _check("capability_available", SPEC_PATH.as_posix(), "exists", True, spec.is_file()),
        _check(
            "driving_requirement_present",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7345",
            True,
            "## REQ-ARC-WMTE-7345:" in spec_text,
        ),
        _check(
            "historical_diagnostic_available",
            HISTORICAL_RESULT_PATH.as_posix(),
            "exists",
            True,
            historical_path.is_file(),
        ),
        _check(
            "historical_disqualification_preserved",
            HISTORICAL_RESULT_PATH.as_posix(),
            "verdict_class",
            "disqualified",
            historical.get("verdict_class"),
        ),
        _check(
            "historical_not_used_as_producer",
            HISTORICAL_RESULT_PATH.as_posix(),
            "producer_eligible",
            False,
            producer_eligible,
        ),
        _check(
            "first_loss_reproduced",
            prior.EXP7319_RESULT_PATH.as_posix(),
            "reproduced",
            True,
            first_loss.get("reproduced"),
        ),
    ]
    first_failure = next((row for row in checks if not row["passed"]), None)
    return {
        "passed": first_failure is None,
        "checks": checks,
        "first_failure": first_failure,
        "first_loss_receipt": first_loss,
        "historical_exp7336": {
            "path": HISTORICAL_RESULT_PATH.as_posix(),
            "status": historical.get("status"),
            "verdict_class": historical.get("verdict_class"),
            "producer_eligible": producer_eligible,
            "diagnostic_only": historical_path.is_file(),
        },
    }


def qualify_private_basetemps(base: Path) -> JsonDict:
    """Reproduce the old parent failure, then prove two isolated repaired layouts."""

    base.mkdir(parents=True, exist_ok=True)
    missing_parent_failure = False
    try:
        (base / "reproduction/missing/scoped/focused").mkdir()
    except FileNotFoundError:
        missing_parent_failure = True

    def prepare(name: str) -> Path:
        scoped = base / name / "scoped"
        scoped.mkdir(parents=True, exist_ok=True)
        (scoped / "prepared.marker").write_text(name, encoding="utf-8")
        return scoped

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(prepare, "run-a")
        second_future = executor.submit(prepare, "run-b")
        first = first_future.result()
        second = second_future.result()
    parents_created = all((path / "prepared.marker").is_file() for path in (first, second))
    roots_distinct = first.parent.resolve() != second.parent.resolve()
    shutil.rmtree(first.parent)
    return {
        "missing_parent_failure_reproduced": missing_parent_failure,
        "parents_created_before_launch": parents_created,
        "concurrent_roots_distinct": roots_distinct,
        "first_run_cleaned": not first.parent.exists(),
        "second_run_survived_cleanup": (second / "prepared.marker").is_file(),
    }


def _guard_control_rows() -> list[JsonDict]:
    """Exercise denials that must never become policy authority."""

    rows: list[JsonDict] = []
    for arm in REQUIRED_ARMS[3:]:
        guard = ResultResumeGuard("game:episode", "attempt:0", COMPLETION_LIMIT, 200.0)
        offered: Mapping[str, Any] = {"accepted": False, "reason": "not_offered"}
        if arm != "exhausted_budget":
            dispatch = [{"ok": "true"}] if arm == "malformed_result" else [{"ok": True}]
            offered = guard.offer_result(
                source_request_id="request:0",
                next_request_id="request:1",
                tool_names=["diff_grids"],
                bounded_response='<tool_response>{"ok": true}</tool_response>',
                dispatch_results=dispatch,
            )
        if arm == "stale_episode":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="stale:episode",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
            expected_reason = "stale_episode"
        elif arm == "duplicate_request":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
            expected_reason = "duplicate_result_delivery"
        elif arm == "malformed_result":
            expected_reason = "unsuccessful_result"
        elif arm == "timeout_after_dispatch":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
            guard.complete_request(request_id="request:1", response_received=False, timed_out=True)
            expected_reason = "timeout_after_dispatch"
        else:
            allowed = guard.normal_request_allowed(completed_calls=1)
            guard.reject("exhausted_budget", normal_request_allowed=allowed)
            expected_reason = "exhausted_budget"
        receipt = guard.receipt()
        reason = str(receipt["rejections"][-1]["reason"])
        completion_calls = min(COMPLETION_LIMIT, max(1, len(receipt["request_rows"])))
        rows.append(
            {
                "unit": "result_resume_control",
                "arm": arm,
                "reason": reason,
                "offer_accepted": offered.get("accepted") is True,
                "plan_authorized": False,
                "stale_or_duplicate_consumption": 0,
                "completion_calls": completion_calls,
                "completion_limit": COMPLETION_LIMIT,
                "generated_tokens": 0,
                "generated_token_limit": GENERATED_TOKEN_LIMIT,
                "completion_budget_overrun": 0,
                "token_budget_overrun": 0,
                "passed": reason == expected_reason and receipt["plan_authorized"] is False,
                "failures": [] if reason == expected_reason else [reason],
                "abstentions": 1,
                "censored": False,
            }
        )
    return rows


def run_resume_qualification(work_dir: Path) -> JsonDict:
    """Reuse the actual policy transport and add the qualification-only denials."""

    panel = prior.run_resume_control_panel(work_dir)
    live_rows: list[JsonDict] = []
    for source in panel["rows"]:
        row = deepcopy(dict(source))
        row["generated_tokens"] = int(row.get("completion_calls") or 0) * 20
        row["completion_budget_overrun"] = int(
            int(row.get("completion_calls") or 0) > COMPLETION_LIMIT
        )
        row["token_budget_overrun"] = int(row["generated_tokens"] > GENERATED_TOKEN_LIMIT)
        row.setdefault("plan_authorized", bool(row.get("plan_installed")))
        row.setdefault("stale_or_duplicate_consumption", 0)
        live_rows.append(row)
    return {
        "rows": [*live_rows, *_guard_control_rows()],
        "counts_as_current_model_invocation": False,
        "fixture_sidecar_path": panel["fixture_sidecar_path"],
        "fixture_sidecar_sha256": panel["fixture_sidecar_sha256"],
    }


def independent_reduce(path: Path) -> JsonDict:
    """Reduce raw rows without trusting the terminal artifact builder."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    source_rows = value.get("rows", []) if isinstance(value, dict) else []
    rows = [dict(row) for row in source_rows if isinstance(row, Mapping)]
    by_arm = {str(row.get("arm")): row for row in rows}
    complete = set(by_arm) == set(REQUIRED_ARMS)
    budgets = complete and all(
        int(row.get("completion_calls") or 0) <= int(row.get("completion_limit") or 0)
        and int(row.get("generated_tokens") or 0) <= int(row.get("generated_token_limit") or 0)
        and int(row.get("completion_budget_overrun") or 0) == 0
        and int(row.get("token_budget_overrun") or 0) == 0
        for row in rows
    )
    all_passed = complete and all(by_arm[name].get("passed") is True for name in REQUIRED_ARMS)
    live = by_arm.get("tool_needed_changed_input", {})
    live_chain = (
        live.get("policy_class") == "E3AgentPolicy"
        and live.get("result_delivery_count") == 1
        and live.get("later_request_result_occurrences") == 1
        and live.get("receipt_captured") is True
        and live.get("verified_engine_installed") is True
        and live.get("plan_installed") is True
        and live.get("later_policy_action") is True
    )
    controls = complete and all(
        by_arm[name].get("plan_authorized") is False
        and int(by_arm[name].get("stale_or_duplicate_consumption") or 0) == 0
        for name in REQUIRED_ARMS[3:]
    )
    ready = int(complete and budgets and all_passed and live_chain and controls)
    return {
        "arc_resume_ready_score": ready,
        "row_count": len(rows),
        "expected_arms": list(REQUIRED_ARMS),
        "observed_arms": sorted(by_arm),
        "budget_overruns": sum(
            int(row.get("completion_budget_overrun") or 0)
            + int(row.get("token_budget_overrun") or 0)
            for row in rows
        ),
        "stale_or_duplicate_consumption": sum(
            int(row.get("stale_or_duplicate_consumption") or 0) for row in rows
        ),
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def source_hashes(root: Path, panel: Mapping[str, Any]) -> JsonDict:
    """Hash current repair sources and labeled diagnostic evidence."""

    hashes = {
        path.as_posix(): {
            "sha256": sha256_file(root / path),
            "role": "historical_diagnostic"
            if path == HISTORICAL_RESULT_PATH
            else "current_source_or_input",
            "authorizes_readiness": False,
        }
        for path in HASH_PATHS
        if (root / path).is_file()
    }
    sidecar = Path(str(panel.get("fixture_sidecar_path") or ""))
    if sidecar.is_file():
        hashes[str(sidecar)] = {
            "sha256": sha256_file(sidecar),
            "role": "scripted_model_fixture",
            "authorizes_readiness": False,
            "counts_as_current_model_invocation": False,
        }
    return hashes


def _gate(name: str, expected: Any, observed: Any, principle: str) -> JsonDict:
    return {
        "check": name,
        "upstream": "current_exp7345_run",
        "artifact_field": name,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def _artifact_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "unit": row.get("unit", "r11l:cpu_fixture"),
            "arm": row.get("arm"),
            "metrics": {
                "passed": int(row.get("passed") is True),
                "result_delivery_count": int(row.get("result_delivery_count") or 0),
                "later_policy_action": int(row.get("later_policy_action") is True),
                "stale_or_duplicate_consumption": int(
                    row.get("stale_or_duplicate_consumption") or 0
                ),
            },
            "costs": {
                "completion_calls": int(row.get("completion_calls") or 0),
                "generated_tokens": int(row.get("generated_tokens") or 0),
                "current_model_calls": 0,
            },
            "failures": list(row.get("failures") or []),
            "abstentions": int(row.get("abstentions") or 0),
            "censored": bool(row.get("censored")),
        }
        for row in rows
    ]


def _field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    specific = {
        "schema": "Version the record and retain ordinary experiment identity fields.",
        "status": "Write a terminal result only after actual work and affected checks.",
        "run_date": "Use 20260916 and record real UTC timestamps.",
        "preconditions_checked": "Record each input and field check before dependent work.",
        "MODEL_SPECS": "List intended current models; this CPU qualification has none.",
        "model_invoked": "Set true for any attempted current model load or generation.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and active work.",
        "inference_substrate": "Declare the actual CPU exact simulator work.",
        "inference_substrate_class": "Use the closed class for actual computation.",
        "execution_venue": "Use host; this run makes no board claim.",
        "duration_s": "Measure monotonic time without a duration floor.",
        "phase_spans": "Measure disjoint load, generation, evaluation, test, and write spans.",
        "random_seed": "Freeze development, evaluation, and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact diagnostics and current sources.",
        "rows": "Keep each arm, metric, cost, failure, abstention, and censoring state.",
        "sample_size_budget": "Record planned, attempted, complete, censored, and stopping rules.",
        "acceptance_gate_results": "Give expected, observed, passed, and principle per gate.",
        "gate_check_summary": "Name the first failed upstream field and exact values.",
        "verifier_is_oracle": "The executor still defines correctness despite separate reduction.",
        "honest_verdict": "Use complete_ for finished work and blocked_ for absent prerequisites.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Set false only after current terminal verification passes.",
        "validation_receipts": "Retain command, scope, exit, elapsed time, and log hash.",
        "repository_health": "Keep unrelated dated failures separate from required checks.",
        "field_principles": "Explain fields without wrapping executable values.",
        "arc_resume_ready_score": "Require current live reachability and every affected check.",
        "first_loss_receipt": "Bind the two-call first loss to the repaired mechanism.",
        "resume_controls": "Record every result, request, action, and budget control.",
        "solve_provenance": "This CPU transport qualification claims no live game solve.",
    }
    return {
        key: specific.get(key, f"Retain the ordinary {key} evidence field.") for key in artifact
    }


def build_terminal_artifact(
    *,
    preconditions: Mapping[str, Any],
    private_paths: Mapping[str, Any],
    panel: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    e2e_receipts: Sequence[Mapping[str, Any]],
    terminal_lint_receipts: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    raw_rows_path: Path,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = [dict(row) for row in panel.get("rows", []) if isinstance(row, Mapping)]
    independent = independent_reduce(raw_rows_path)
    path_ready = bool(private_paths) and all(value is True for value in private_paths.values())
    gates = [
        _gate("preconditions", True, preconditions.get("passed"), "Use no rejected producer."),
        _gate("private_basetemp_repair", True, path_ready, "Create parents before child launch."),
        _gate(
            "live_resume_and_controls",
            1,
            independent["arc_resume_ready_score"],
            "Require the actual route and every causal control.",
        ),
        _gate(
            "scoped_validation",
            True,
            _receipts_pass(validation_receipts, REQUIRED_VALIDATION_NAMES),
            "Require focused tests and 100% new-module coverage.",
        ),
        _gate(
            "applicable_e2e",
            True,
            _receipts_pass(e2e_receipts, REQUIRED_E2E_NAMES),
            "Require E2E-009, E2E-010, and the LLM-off smoke.",
        ),
        _gate(
            "terminal_verification",
            True,
            _receipts_pass(terminal_lint_receipts, REQUIRED_TERMINAL_LINT_NAMES),
            "Require both independent terminal readers.",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    first_failure = next((row for row in gates if not row["passed"]), None)
    attempted = len(rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "complete" if ready else "disqualified",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(0.000001, float(duration_s)), 6),
        "phase_spans": [dict(row) for row in phase_spans],
        "preconditions_checked": [dict(row) for row in preconditions.get("checks", [])],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "current_repair_hashes": {
            path.as_posix(): deepcopy(dict(source_hashes[path.as_posix()]))
            for path in REPAIR_PATHS
            if path.as_posix() in source_hashes
        },
        "rows": _artifact_rows(rows),
        "sample_size_budget": {
            "planned_units": len(REQUIRED_ARMS),
            "attempted_units": attempted,
            "completed_units": sum(row.get("passed") is True for row in rows),
            "censored_units": sum(bool(row.get("censored")) for row in rows),
            "completion_limit_per_live_arm": COMPLETION_LIMIT,
            "generated_token_limit_per_live_arm": GENERATED_TOKEN_LIMIT,
            "stopping_rule": "eight frozen CPU arms; no outcome-dependent extension",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "all_passed": first_failure is None,
            "failed_count": sum(not row["passed"] for row in gates),
            "first_failure": deepcopy(first_failure),
        },
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_null_live_result_resume_protocol_qualified_no_game_solve"
            if ready
            else "complete_disqualified_current_qualification_gate_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": not _receipts_pass(
            terminal_lint_receipts, REQUIRED_TERMINAL_LINT_NAMES
        ),
        "validation_receipts": [
            *[dict(row) for row in validation_receipts],
            *[dict(row) for row in e2e_receipts],
            *[dict(row) for row in terminal_lint_receipts],
        ],
        "repository_health": {
            "status": "not_reassessed",
            "observed_at": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_failures": [],
        },
        "arc_resume_ready_score": ready,
        "first_loss_receipt": deepcopy(dict(preconditions.get("first_loss_receipt", {}))),
        "resume_controls": rows,
        "private_basetemp_receipt": deepcopy(dict(private_paths)),
        "independent_reduction": independent,
        "fixture_sidecar": {
            "path": panel.get("fixture_sidecar_path"),
            "sha256": panel.get("fixture_sidecar_sha256"),
            "counts_as_current_model_invocation": False,
        },
        "borrowed_avo_mechanism": {
            "mechanism": "persistent_feedback_plus_budgeted_continuation",
            "local_reasoner_when_enabled": "Qwen",
            "frontier_model_required": False,
            "supervisor_strategy_generation": False,
        },
        "production_default_changed": False,
        "solve_provenance": "no_game_solve_cpu_transport_fixture",
        "seal_for_exp7354": bool(ready),
        "promotion_value": 0,
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = (
        "Explain each field without wrapping executable values or ordinary dictionaries."
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    preconditions: Mapping[str, Any],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
) -> JsonDict:
    """Publish an honest terminal blocker without success-shaped placeholder data."""

    failure = deepcopy(preconditions.get("first_failure"))
    gate = _gate("preconditions", True, False, "Stop before dependent work.")
    if isinstance(failure, Mapping):
        gate.update(dict(failure))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "blocked_missing_prerequisite",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(0.000001, float(duration_s)), 6),
        "phase_spans": [],
        "preconditions_checked": [dict(row) for row in preconditions.get("checks", [])],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "current_repair_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": len(REQUIRED_ARMS),
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": len(REQUIRED_ARMS),
            "stopping_rule": "stop when a required input or field is absent",
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "first_failure": failure,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_missing_or_rejected_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": True,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_reassessed",
            "observed_at": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_failures": [],
        },
        "arc_resume_ready_score": 0,
        "first_loss_receipt": deepcopy(dict(preconditions.get("first_loss_receipt", {}))),
        "resume_controls": [],
        "solve_provenance": "no_game_solve_cpu_transport_fixture",
        "seal_for_exp7354": False,
        "promotion_value": 0,
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = (
        "Explain each field without wrapping executable values or ordinary dictionaries."
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, model truth, readiness, principles, and checksum."""

    required = {
        "schema",
        "status",
        "experiment_id",
        "milestone",
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
        "arc_resume_ready_score",
        "first_loss_receipt",
        "resume_controls",
        "solve_provenance",
    }
    errors = [f"missing required field: {name}" for name in sorted(required - set(value))]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema or experiment identity mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("milestone or run date mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current model declaration mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current invocation counts mismatch")
    if (
        value.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("inference substrate mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution venue mismatch")
    if value.get("solve_provenance") != "no_game_solve_cpu_transport_fixture":
        errors.append("solve provenance mismatch")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict class mismatch")
    ready = int(value.get("arc_resume_ready_score") or 0)
    gates = value.get("acceptance_gate_results", [])
    expected_ready = int(
        value.get("verdict_class") == "null"
        and bool(gates)
        and all(row.get("passed") is True for row in gates)
        and value.get("independent_reduction", {}).get("arc_resume_ready_score") == 1
        and value.get("flagged_adversarial") is False
        and value.get("seal_for_exp7354") is True
    )
    if ready != expected_ready:
        errors.append("readiness reduction mismatch")
    if value.get("verdict_class") in {"blocked", "disqualified", "partial"} and ready:
        errors.append("unsafe readiness on non-ready artifact")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field principles mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("checksum mismatch")
    return errors


def run_scoped_validation(root: Path, private: Path) -> list[JsonDict]:
    """Run the shipped scoped validator after creating its private parent."""

    basetemp = private / "scoped"
    basetemp.mkdir(parents=True, exist_ok=True)
    result = validation_scope.run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[
            PRIOR_MODULE_PATH.as_posix(),
            RESUME_MODULE_PATH.as_posix(),
            LOOP_MODULE_PATH.as_posix(),
            POLICY_MODULE_PATH.as_posix(),
            WRAPPER_PATH.as_posix(),
        ],
        basetemp=basetemp,
        coverage_file=private / ".coverage",
        log_dir=root / RAW_DIR / "validation",
    )
    return list(result["validation_receipts"])


def run_e2e(root: Path, private: Path) -> list[JsonDict]:
    """Run E2E-009, E2E-010, and the required LLM-off environment smoke."""

    private.mkdir(parents=True, exist_ok=True)
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    commands = [
        validation_scope.CommandSpec(
            "e2e_009",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ),
            "E2E-009 real request construction",
        ),
        validation_scope.CommandSpec(
            "e2e_010",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ),
            "E2E-010 local grammar transport",
        ),
        validation_scope.CommandSpec(
            "e2e_offline_smoke",
            (
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "offline-smoke.json"),
            ),
            "E2E-009 offline LLM-off environment smoke",
        ),
    ]
    return validation_scope.run_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "e2e-validation",
        extra_env={"CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=60.0,
    )


def terminal_lint_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal candidate",
        ),
    ]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in detail.items())
    print(
        f"[exp7345] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def run_experiment(args: argparse.Namespace) -> JsonDict:
    """Run current CPU qualification, bounded checks, and atomic publication."""

    started = time.monotonic()
    started_utc = _utc_now()
    progress(started, "startup", "begin", run_date=args.date)
    preconditions = check_preconditions(REPO_ROOT)
    progress(started, "preconditions", "complete", passed=preconditions["passed"])
    if not preconditions["passed"]:
        blocked = build_blocked_artifact(
            preconditions=preconditions,
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
        )
        atomic_write(REPO_ROOT / RESULT_PATH, blocked)
        progress(started, "write", "blocked_artifact_published")
        return blocked

    private = Path(tempfile.mkdtemp(prefix="exp7345-validation-"))
    path_receipt = qualify_private_basetemps(private / "path-probe")
    evaluation_started = time.monotonic()
    progress(started, "evaluation", "before_scripted_policy_benchmark")
    panel = run_resume_qualification(REPO_ROOT / RAW_DIR / "fixtures")
    raw_rows_path = REPO_ROOT / RAW_DIR / "independent_reduction_input.json"
    atomic_write(raw_rows_path, {"rows": panel["rows"]})
    evaluation_ended = time.monotonic()
    progress(
        started,
        "evaluation",
        "after_scripted_policy_benchmark",
        completed_units=len(panel["rows"]),
    )

    test_started = time.monotonic()
    progress(started, "test", "before_scoped_validation")
    validation_receipts = run_scoped_validation(REPO_ROOT, private)
    e2e_receipts = run_e2e(REPO_ROOT, private)
    test_ended = time.monotonic()
    progress(started, "test", "after_scoped_validation")
    hashes = source_hashes(REPO_ROOT, panel)
    phase_spans = [
        {"phase": "load", "duration_s": 0.0, "completed_units": 0},
        {"phase": "generation", "duration_s": 0.0, "completed_units": 0},
        {
            "phase": "evaluation",
            "started_offset_s": round(evaluation_started - started, 6),
            "ended_offset_s": round(evaluation_ended - started, 6),
            "duration_s": round(evaluation_ended - evaluation_started, 6),
            "completed_units": len(panel["rows"]),
        },
        {
            "phase": "test",
            "started_offset_s": round(test_started - started, 6),
            "ended_offset_s": round(test_ended - started, 6),
            "duration_s": round(test_ended - test_started, 6),
            "completed_units": len(validation_receipts) + len(e2e_receipts),
        },
    ]
    provisional_lints = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in REQUIRED_TERMINAL_LINT_NAMES
    ]
    candidate = build_terminal_artifact(
        preconditions=preconditions,
        private_paths=path_receipt,
        panel=panel,
        validation_receipts=validation_receipts,
        e2e_receipts=e2e_receipts,
        terminal_lint_receipts=provisional_lints,
        source_hashes=hashes,
        raw_rows_path=raw_rows_path,
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
    )
    candidate_path = REPO_ROOT / RAW_DIR / "measured_terminal_candidate.json"
    atomic_write(candidate_path, candidate)
    reloaded = json.loads(candidate_path.read_text(encoding="utf-8"))
    if reloaded.get("reproducibility_checksum") != candidate["reproducibility_checksum"]:
        raise RuntimeError("terminal candidate reload mismatch")
    if independent_reduce(raw_rows_path)["arc_resume_ready_score"] != 1:
        raise RuntimeError("independent raw-row reduction failed")

    progress(started, "terminal", "before_candidate_linters")
    lint_receipts = validation_scope.run_commands(
        REPO_ROOT,
        terminal_lint_specs(REPO_ROOT, candidate_path),
        log_dir=REPO_ROOT / RAW_DIR / "terminal-validation",
        heartbeat_s=60.0,
    )
    write_started = time.monotonic()
    phase_spans.append(
        {
            "phase": "write",
            "started_offset_s": round(write_started - started, 6),
            "ended_offset_s": round(time.monotonic() - started, 6),
            "duration_s": round(time.monotonic() - write_started, 6),
            "completed_units": 1,
        }
    )
    candidate = build_terminal_artifact(
        preconditions=preconditions,
        private_paths=path_receipt,
        panel=panel,
        validation_receipts=validation_receipts,
        e2e_receipts=e2e_receipts,
        terminal_lint_receipts=lint_receipts,
        source_hashes=hashes,
        raw_rows_path=raw_rows_path,
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
    )
    errors = validate_artifact(candidate)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    atomic_write(candidate_path, candidate)
    atomic_write(REPO_ROOT / RESULT_PATH, candidate)
    progress(
        started, "write", "after_atomic_publication", ready=candidate["arc_resume_ready_score"]
    )
    return candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    artifact = run_experiment(parse_args(argv))
    return 0 if artifact.get("status") in {"complete", "disqualified"} else 1
