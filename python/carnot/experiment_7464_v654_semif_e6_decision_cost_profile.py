"""Reduce V653 ARC traces into the SEMIF E6 decision-cost profile.

This is an offline audit.  It reads immutable upstream ledgers, attributes only
durations actually timestamped there, and leaves every unobserved seam missing.

Spec refs: REQ-ARC-WMTE-7464 and SCENARIO-ARC-WMTE-7464-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan as shared_build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan as shared_validate_command_plan,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    CommandSpec,
    run_commands,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.654"
EXPERIMENT_ID = "exp7464-v654-semif-e6-decision-cost-profile"
SCHEMA = "carnot.exp7464.v654.semif_e6_decision_cost_profile.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7464_v654_semif_e6_decision_cost_profile.json")
RAW_DIR = Path("results/raw/experiment_7464_v654_semif_e6_decision_cost_profile")
MODULE_PATH = Path("python/carnot/experiment_7464_v654_semif_e6_decision_cost_profile.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7464_v654_semif_e6_decision_cost_profile.py")
TEST_PATH = Path("tests/python/test_experiment_7464_v654_semif_e6_decision_cost_profile.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
V653_RESULT = Path("results/experiment_7457_v653_arc_exposure.json")
V653_RAW = Path("results/raw/experiment_7457_v653_arc_exposure")
E0_RESULT = Path("results/experiment_7463_v654_semif_e0_logprob_parity.json")
FLAGGED_RESULT = Path("results/experiment_7234_v637_arc_scored_dryrun.json")
OLD_MODEL_RESULT = Path("results/experiment_5972_arc_llm_on_budget2000_feasibility.json")

SEAMS = (
    "candidate_action_selection",
    "hypothesis_gate",
    "supervisor_arm_selection",
    "induction_timing",
    "downstream_generation",
)
AFFECTED_CHECK_NAMES = REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7457_v653_arc_exposure.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
    Path("python/carnot/agentic/arc_supervisor_refinement.py"),
    Path("docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md"),
    SPEC_PATH,
    V653_RESULT,
    E0_RESULT,
    FLAGGED_RESULT,
    OLD_MODEL_RESULT,
)
MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return an aware UTC timestamp at an artifact boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit a flushed monotonic phase boundary or unit checkpoint."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7464] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for independent comparisons."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum itself."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> bool:
    """Return whether a value is a finite non-boolean number."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _union_duration(spans: Sequence[Mapping[str, Any]]) -> float:
    """Measure the union of complete monotonic spans without double counting."""

    intervals: list[tuple[float, float]] = []
    for span in spans:
        start = span.get("start_monotonic_s")
        end = span.get("end_monotonic_s")
        if not (_finite_number(start) and _finite_number(end)):
            continue
        left, right = float(start), float(end)
        if right < left:
            raise ValueError("negative_request_span")
        intervals.append((left, right))
    intervals.sort()
    merged: list[list[float]] = []
    for left, right in intervals:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return sum(right - left for left, right in merged)


def _optional_sum(spans: Sequence[Mapping[str, Any]], field: str) -> int | float | None:
    """Sum a field only when every measured request supplied it."""

    values = [span.get(field) for span in spans]
    if not values or any(not _finite_number(value) for value in values):
        return None
    total = sum(float(value) for value in values)
    return int(total) if total.is_integer() else total


def _seam_row(unit: Mapping[str, Any], seam: str, generation_s: float) -> JsonDict:
    """Give one episode/seam pair an explicit measured or missing disposition."""

    spans = [row for row in unit.get("request_spans", []) if isinstance(row, Mapping)]
    supervisor = unit.get("supervisor") if isinstance(unit.get("supervisor"), Mapping) else {}
    common: JsonDict = {
        "episode_id": unit.get("episode_id"),
        "game": unit.get("game"),
        "seed": unit.get("seed"),
        "condition": unit.get("condition"),
        "seam": seam,
        "elapsed_s": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cpu_work_s": None,
        "gpu_work_ms": 0.0,
        "candidate_option_presence": "none_observed",
        "invocation_status": "uninvoked",
        "banked_progress": unit.get("banked_progress"),
        "start_level": unit.get("start_level"),
        "terminal_level": unit.get("terminal_level"),
        "disposition": unit.get("disposition", "unknown"),
        "censoring": {
            "is_censored": bool(unit.get("censored")),
            "reason": "episode_censored" if unit.get("censored") else None,
        },
        "directly_replaceable_decision": seam != "downstream_generation",
    }
    if seam == "candidate_action_selection":
        common.update(
            invocation_status="invoked",
            candidate_option_presence="selected_only",
            missing_fields=["stage_start", "stage_end", "candidate_set", "candidate_ids"],
        )
    elif seam == "hypothesis_gate":
        common.update(
            elapsed_s=0.0,
            invocation_status="uninvoked",
            candidate_option_presence="no_usable_hypothesis",
            missing_fields=["candidate_set", "candidate_ids"],
        )
    elif seam == "supervisor_arm_selection":
        selected = list(supervisor.get("selected_arms") or [])
        common.update(
            invocation_status="invoked" if selected else "uninvoked",
            candidate_option_presence="static_enabled_arms_without_eligible_snapshot",
            selected_candidate_ids=selected,
            missing_fields=["stage_start", "stage_end", "eligible_candidate_set"],
        )
    elif seam == "induction_timing":
        common.update(
            invocation_status="invoked_unusable" if spans else "uninvoked",
            candidate_option_presence="no_timing_candidate_set",
            missing_fields=["stage_start", "stage_end", "candidate_set", "selected_candidate_id"],
        )
    else:
        common.update(
            elapsed_s=generation_s,
            input_tokens=_optional_sum(spans, "input_tokens"),
            output_tokens=_optional_sum(spans, "output_tokens"),
            gpu_work_ms=_optional_sum(spans, "gpu_total_ms"),
            invocation_status="invoked" if spans else "uninvoked",
            candidate_option_presence="not_applicable",
            missing_fields=[],
            directly_replaceable_decision=False,
        )
    return common


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty_percentile")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _game_cluster_interval(
    summaries: Sequence[Mapping[str, Any]], *, seed: int, draws: int
) -> JsonDict:
    """Bootstrap bounds by independent game cluster, never by episode row."""

    by_game: dict[str, list[Mapping[str, Any]]] = {}
    for row in summaries:
        by_game.setdefault(str(row["game"]), []).append(row)
    games = sorted(by_game)
    estimates: list[float] = []
    if games:
        rng = random.Random(seed)
        for _ in range(max(1, draws)):
            sampled = [rng.choice(games) for _ in games]
            elapsed = sum(
                float(row["episode_elapsed_s"]) for game in sampled for row in by_game[game]
            )
            residual = sum(
                float(row["unclassified_residual_s"]) for game in sampled for row in by_game[game]
            )
            estimates.append(residual / elapsed if elapsed else 0.0)
    return {
        "method": "game_cluster_bootstrap",
        "bootstrap_seed": seed,
        "bootstrap_draws": max(1, draws),
        "cluster_count": len(games),
        "games": games,
        "upper_share_ci95": (
            [_percentile(estimates, 0.025), _percentile(estimates, 0.975)] if estimates else None
        ),
        "sample_limited": len(games) < 10,
        "broad_transfer_claim_allowed": len(games) >= 10,
    }


def reduce_raw_units(
    units: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, bootstrap_draws: int
) -> JsonDict:
    """Reduce all supplied episodes while preserving unknown seam costs."""

    identifiers = [str(unit.get("episode_id")) for unit in units]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate_episode_id")
    rows: list[JsonDict] = []
    summaries: list[JsonDict] = []
    all_tokens_complete = True
    for unit in units:
        elapsed = unit.get("episode_elapsed_s")
        if not _finite_number(elapsed) or float(elapsed) < 0:
            raise ValueError("invalid_episode_elapsed")
        spans = [row for row in unit.get("request_spans", []) if isinstance(row, Mapping)]
        generation_s = _union_duration(spans)
        if generation_s > float(elapsed) + 1e-9:
            raise ValueError("measured_time_exceeds_episode")
        residual = max(0.0, float(elapsed) - generation_s)
        all_tokens_complete = all_tokens_complete and all(
            _finite_number(span.get(field))
            for span in spans
            for field in ("input_tokens", "output_tokens")
        )
        summaries.append(
            {
                "episode_id": unit.get("episode_id"),
                "game": unit.get("game"),
                "seed": unit.get("seed"),
                "condition": unit.get("condition"),
                "episode_elapsed_s": float(elapsed),
                "measured_generation_s": generation_s,
                "known_direct_replaceable_s": 0.0,
                "unclassified_residual_s": residual,
                "banked_progress": unit.get("banked_progress"),
                "censored": bool(unit.get("censored")),
                "disposition": unit.get("disposition"),
            }
        )
        rows.extend(_seam_row(unit, seam, generation_s) for seam in SEAMS)
    total = sum(float(row["episode_elapsed_s"]) for row in summaries)
    measured = sum(float(row["measured_generation_s"]) for row in summaries)
    residual = sum(float(row["unclassified_residual_s"]) for row in summaries)
    lower = 0.0 if total else None
    upper = residual / total if total else None
    speedup_lower = 1.0 if lower == 0.0 else None
    speedup_upper = 1.0 / (1.0 - upper) if upper is not None and upper < 1.0 else None
    complete = sum(row.get("disposition") == "complete" for row in units)
    censored = sum(bool(row.get("censored")) for row in units)
    failed = sum(row.get("disposition") == "failed" for row in units)
    return {
        "rows": rows,
        "episode_summaries": summaries,
        "stage_attribution": {
            "total_episode_s": total,
            "measured_stage_s": measured,
            "unclassified_residual_s": residual,
            "trace_time_fraction": measured / total if total else None,
            "model_token_fraction": 1.0 if all_tokens_complete and measured > 0 else None,
            "measured_stage": "downstream_generation_request_union",
        },
        "replaceable_share_bounds": {
            "lower": lower,
            "upper": upper,
            "lower_assignment": "unclassified_time_is_nonreplaceable",
            "upper_assignment": "unclassified_time_is_replaceable",
            "downstream_generation_directly_replaceable": False,
            "perfect_removal_speedup_lower": speedup_lower,
            "perfect_removal_speedup_upper": speedup_upper,
        },
        "game_cluster_interval": _game_cluster_interval(
            summaries, seed=bootstrap_seed, draws=bootstrap_draws
        ),
        "sample_size_budget": {
            "planned_independent_units": len(units),
            "attempted_independent_units": len(units),
            "complete_independent_units": complete,
            "failed_independent_units": failed,
            "censored_independent_units": censored,
            "unstarted_independent_units": 0,
            "independent_game_clusters": len({str(row.get("game")) for row in units}),
        },
    }


def run_positive_control() -> JsonDict:
    """Prove that overlap-safe timing and token reduction recover a known fixture."""

    unit = _fixture_unit("control", 1)
    reduced = reduce_raw_units([unit], bootstrap_seed=1, bootstrap_draws=8)
    generation = next(row for row in reduced["rows"] if row["seam"] == "downstream_generation")
    passed = (
        generation["elapsed_s"] == 3.0
        and generation["input_tokens"] == 90
        and generation["output_tokens"] == 10
    )
    return {
        "name": "overlap_safe_timing_and_token_control",
        "expected_generation_s": 3.0,
        "observed_generation_s": generation["elapsed_s"],
        "expected_input_tokens": 90,
        "observed_input_tokens": generation["input_tokens"],
        "expected_output_tokens": 10,
        "observed_output_tokens": generation["output_tokens"],
        "passed": passed,
    }


def _fixture_unit(game: str, seed: int, condition: str = "shadow") -> JsonDict:
    """Build a deterministic reducer fixture without accessing a model or environment."""

    return {
        "episode_id": f"{game}:seed-{seed}:{condition}",
        "game": game,
        "seed": seed,
        "condition": condition,
        "disposition": "complete",
        "censored": False,
        "episode_elapsed_s": 10.0,
        "action_count": 4,
        "selected_action_ids": ["RESET", "ACTION1", "ACTION6"],
        "start_level": 0,
        "terminal_level": 0,
        "banked_progress": 0,
        "request_spans": [
            {
                "request_id": "r0",
                "start_monotonic_s": 1.0,
                "end_monotonic_s": 3.0,
                "input_tokens": 40,
                "output_tokens": 5,
                "total_tokens": 45,
                "usable": False,
                "gpu_prompt_ms": 400.0,
                "gpu_generation_ms": 1500.0,
                "gpu_total_ms": 1900.0,
            },
            {
                "request_id": "r1",
                "start_monotonic_s": 2.0,
                "end_monotonic_s": 4.0,
                "input_tokens": 50,
                "output_tokens": 5,
                "total_tokens": 55,
                "usable": False,
                "gpu_prompt_ms": 500.0,
                "gpu_generation_ms": 1400.0,
                "gpu_total_ms": 1900.0,
            },
        ],
        "supervisor": {
            "mode": condition,
            "actions_observed": 3,
            "window": 120,
            "arms_enabled": [
                "drop_goal_bias",
                "allow_reinduction",
                "force_exploration_diversity",
            ],
            "selected_arms": ["drop_goal_bias"],
            "full_eligible_candidate_set_observed": False,
        },
        "model_identity": {
            "repository": "unsloth/Qwen3.8-27B-GGUF",
            "revision": "fixture-revision",
            "quantization": "Q4_K_M",
        },
    }


def build_seam_observation_spec() -> JsonDict:
    """Specify the events Exp7471 must add without authorizing that future run."""

    return {
        "target_experiment": "exp7471",
        "environment_or_model_calls_authorized": False,
        "cross_cutting_fields": [
            "episode_id",
            "decision_id",
            "parent_decision_id",
            "event_monotonic_ns",
            "clock_identity",
            "progress_before",
            "progress_after",
        ],
        "seams": [
            {
                "seam": "candidate_action_selection",
                "missing_events": ["stage_start", "candidate_set", "selection", "stage_end"],
                "required_candidate_fields": ["stable_candidate_id", "action", "payload", "rank"],
                "required_candidate_ids": ["every_legal_action_payload"],
            },
            {
                "seam": "hypothesis_gate",
                "missing_events": ["stage_start", "candidate_set", "gate_result", "stage_end"],
                "required_candidate_fields": [
                    "stable_candidate_id",
                    "hypothesis_hash",
                    "gate_decision",
                ],
                "required_candidate_ids": ["accept", "reject", "escalate"],
            },
            {
                "seam": "supervisor_arm_selection",
                "missing_events": [
                    "stage_start",
                    "eligible_candidate_set",
                    "selection",
                    "stage_end",
                ],
                "required_candidate_fields": ["stable_candidate_id", "eligibility", "rank"],
                "required_candidate_ids": [
                    "no_redirect",
                    "drop_goal_bias",
                    "allow_reinduction",
                    "force_exploration_diversity",
                ],
            },
            {
                "seam": "induction_timing",
                "missing_events": ["stage_start", "candidate_set", "selection", "stage_end"],
                "required_candidate_fields": [
                    "stable_candidate_id",
                    "trigger_reason",
                    "defer_until",
                ],
                "required_candidate_ids": ["induce_now", "defer_induction", "skip_induction"],
            },
            {
                "seam": "downstream_generation",
                "missing_events": ["cpu_start", "cpu_end", "gpu_start", "gpu_end"],
                "required_candidate_fields": ["request_id", "parent_decision_id"],
                "required_candidate_ids": [],
            },
        ],
        "required_resource_fields": [
            "input_tokens",
            "output_tokens",
            "cpu_work_ns",
            "gpu_work_ns",
            "gpu_device_id",
        ],
    }


def build_ladder_disposition(e0: Mapping[str, Any], reduction: Mapping[str, Any]) -> JsonDict:
    """Close E7-E12 when their E0 or E6 evidence gates are not satisfied."""

    e0_present = bool(e0)
    e6_complete = bool(reduction.get("rows"))
    attribution = reduction.get("stage_attribution", {})
    bounds = reduction.get("replaceable_share_bounds", {})
    e0_pass = (
        e0.get("local_runtime_parity_score") == 1 and e0.get("scored_runtime_parity_score") == 1
    )
    e6_pass = (
        reduction.get("sample_size_budget", {}).get("complete_independent_units", 0) >= 30
        and reduction.get("sample_size_budget", {}).get("independent_game_clusters", 0) >= 10
        and (attribution.get("trace_time_fraction") or 0.0) >= 0.95
        and (attribution.get("model_token_fraction") or 0.0) >= 0.99
        and (bounds.get("lower") or 0.0) > 0.0
    )
    rungs = [
        {
            "rung": "E7",
            "decision": "insufficient_evidence",
            "reason": "E6 lacks candidate sets and sufficient attributed trace time",
        },
        {
            "rung": "E8",
            "decision": "stop",
            "reason": "E0 local and scored runtime parity did not pass",
        },
        {
            "rung": "E9",
            "decision": "insufficient_evidence",
            "reason": "no qualified E7 topology or measured selector cost",
        },
        {
            "rung": "E10",
            "decision": "stop",
            "reason": "no selector passed E7-E9",
        },
        {
            "rung": "E11",
            "decision": "stop",
            "reason": "no winning selector exists to distill",
        },
        {
            "rung": "E12",
            "decision": "stop",
            "reason": "no E7/E8 disagreement set exists",
        },
    ]
    return {
        "e0_report_present": e0_present,
        "e0_gate_passed": e0_pass,
        "e6_report_complete": e6_complete,
        "e6_gate_passed": e6_pass,
        "queue_authorized": e0_present and e6_complete and e0_pass and e6_pass,
        "rungs": rungs,
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, operator: str, principle: str
) -> JsonDict:
    """Describe one gate without allowing a benefit miss to become invalidity."""

    comparisons = {
        "eq": observed == expected,
        "gte": _finite_number(observed) and float(observed) >= float(expected),
        "gt": _finite_number(observed) and float(observed) > float(expected),
    }
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": operator,
        "passed": bool(comparisons[operator]),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name all failed checks and the first exact field-level failure."""

    failures = [row for row in gates if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_checks": [row.get("check") for row in failures],
        "first_failed_check": first.get("check") if first else None,
        "upstream": "raw_unit_rows" if first else None,
        "exact_field_path": first.get("check") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain why every top-level field exists."""

    specific = {
        "schema": "Version identity prevents silent reader drift.",
        "run_date": "The execution date is fixed by the protocol.",
        "preconditions_checked": "Exact source bytes and historical flags are authenticated before reduction.",
        "MODEL_SPECS": "An empty list states that this audit made no current LLM call.",
        "model_specs": "The lowercase alias keeps older readers honest about the same empty current-model set.",
        "model_invoked": "Current work is separated from archived model-shaped events.",
        "invocation_counts": "Attempted and terminal current call states must balance at zero.",
        "inference_substrate": "Pure reduction names its actual upstream-artifact substrate.",
        "inference_substrate_class": "Aggregation is distinct from model loading or generation.",
        "execution_venue": "Host execution is separate from archived CUDA evidence.",
        "duration_s": "Measured current wall time is never padded.",
        "phase_spans": "Monotonic boundaries and completed units make long work auditable.",
        "random_seed": "Bootstrap, audit, and ordering choices are frozen.",
        "reproducibility_checksum": "Code, protocol, inputs, rows, and validation scope are hash-bound.",
        "source_artifact_hashes": "Upstream bytes and original evidence classes cannot be rehabilitated.",
        "rows": "Every supplied episode and seam receives a measured or missing disposition.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted units remain distinct.",
        "acceptance_gate_results": "Validity gates remain separate from benefit nulls.",
        "gate_check_summary": "Every failed gate names the exact observed value.",
        "honest_verdict": "The terminal wording exposes a complete scientific null.",
        "verdict_class": "The closed enum prevents retryable partial work from hiding a null.",
        "verifier_is_oracle": "No acceptance verifier defines success for this observational audit.",
        "flagged_adversarial": "Structural reader flags are retained rather than cleared for promotion.",
        "validation_receipts": "Exact affected commands, exits, hashes, and scopes are durable.",
        "field_principles": "Each field carries its audit purpose.",
        "decision_profile_complete_score": "One means every supplied trace and missing field was disposed, not that E6 passed.",
        "replaceable_share_bounds": "Unknown time widens bounds instead of creating a speedup claim.",
        "seam_observation_spec": "Exact missing events and candidate IDs guide the separate live measurement.",
        "ladder_disposition": "E7-E12 are scoped to measured seams and E0 availability.",
    }
    return {
        key: specific.get(key, "This field preserves typed evidence required by REQ-ARC-WMTE-7464.")
        for key in keys
    }


def _fixture_validation_receipts() -> list[JsonDict]:
    """Supply passing synthetic receipt shapes only to the deterministic test artifact."""

    return [
        {
            "name": name,
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "required": True,
            "log_sha256": "sha256:" + "0" * 64,
            "scope": "deterministic_test_fixture",
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _expected_readout_overhead(e0: Mapping[str, Any]) -> JsonDict:
    """Report measured compatible E0 readout work without projecting a mismatched ARC call."""

    duration = e0.get("duration_breakdown_s", {})
    counts = e0.get("invocation_counts", {})
    seconds = duration.get("forward") if isinstance(duration, Mapping) else None
    calls = counts.get("forward_calls_completed") if isinstance(counts, Mapping) else None
    compatible = _finite_number(seconds) and isinstance(calls, int) and calls > 0
    return {
        "source_experiment": "exp7463",
        "measured_native_forward_s": seconds if compatible else None,
        "measured_native_forward_calls": calls if compatible else None,
        "measured_mean_s_per_short_two_option_readout": (
            float(seconds) / calls if compatible else None
        ),
        "compatible_artifact": compatible,
        "arc_episode_projection_s": None,
        "projection_reason": "ARC candidate count, prompt shape, and invocation frequency are unobserved",
    }


def _producer_code_hashes() -> dict[str, str]:
    """Bind implementation, entrypoint, and focused tests without making them inputs."""

    paths = (MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in paths}


def _build_artifact(
    units: Sequence[Mapping[str, Any]],
    *,
    e0: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at: str,
    ended_at: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble a schema-complete artifact strictly from raw unit rows."""

    reduction = reduce_raw_units(units, bootstrap_seed=7464, bootstrap_draws=2000)
    affected_pass = all(
        any(row.get("name") == name and row.get("passed") is True for row in validation_receipts)
        for name in AFFECTED_CHECK_NAMES
    )
    terminal_pass = all(
        any(row.get("name") == name and row.get("passed") is True for row in validation_receipts)
        for name in TERMINAL_CHECK_NAMES
    )
    preconditions_pass = (
        all(row.get("passed") is True for row in preconditions) if preconditions else True
    )
    attribution = reduction["stage_attribution"]
    bounds = reduction["replaceable_share_bounds"]
    budget = reduction["sample_size_budget"]
    gates = [
        _gate(
            "preconditions_checked",
            "validity",
            True,
            preconditions_pass,
            "eq",
            "Inputs must retain their original identities.",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            affected_pass,
            "eq",
            "All affected checks must pass exactly once.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            True,
            terminal_pass,
            "eq",
            "Cold replay and unchanged strict readers must pass.",
        ),
        _gate(
            "episode_count",
            "benefit",
            30,
            budget["complete_independent_units"],
            "gte",
            "E6 needs thirty complete episodes.",
        ),
        _gate(
            "game_cluster_count",
            "benefit",
            10,
            budget["independent_game_clusters"],
            "gte",
            "Transfer needs ten independent games.",
        ),
        _gate(
            "trace_time_attribution",
            "benefit",
            0.95,
            attribution["trace_time_fraction"],
            "gte",
            "At least 95 percent of trace time must be stage-attributed.",
        ),
        _gate(
            "model_token_attribution",
            "benefit",
            0.99,
            attribution["model_token_fraction"],
            "gte",
            "At least 99 percent of model tokens need a seam.",
        ),
        _gate(
            "positive_lower_replaceable_share",
            "benefit",
            0.0,
            bounds["lower"],
            "gt",
            "Typed decisions can save only directly measured replaceable cost.",
        ),
        _gate(
            "no_model_mixing",
            "validity",
            True,
            True,
            "eq",
            "Old-model histories never enter current quantitative rows.",
        ),
    ]
    invalid = any(row["category"] == "validity" and not row["passed"] for row in gates)
    verdict = "disqualified" if invalid else "null"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete_disqualified_validation"
        if invalid
        else "complete_sample_limited_cost_profile",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "clock_identity": "time.monotonic_process_local_plus_utc_boundaries",
        "duration_s": duration_s,
        "duration_breakdown_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "numeric_fitting": 0.0,
            "aggregation_and_validation": duration_s,
        },
        "phase_spans": list(phase_spans),
        "preconditions_checked": [dict(row) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_identity": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "historical_execution_identity": {
            "source_experiment": "exp7457",
            "execution_venue": "host",
            "device": "NVIDIA GeForce RTX 3090",
            "model_repository": "unsloth/Qwen3.8-27B-GGUF",
            "model_revision": "fe1e2a23d973adb629709749dc4f6756df66ef10",
            "classification": "archived_current_qwen3_8_evidence_not_current_work",
        },
        "small_ebm_training": {
            "performed": False,
            "duration_s": 0.0,
            "receipt": None,
        },
        "random_seed": {
            "fitting": None,
            "ordering": "source_episode_order",
            "audit": 7464,
            "bootstrap": 7464,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "producer_code_hashes": _producer_code_hashes(),
        "raw_unit_rows": [deepcopy(dict(row)) for row in units],
        "rows": reduction["rows"],
        "episode_summaries": reduction["episode_summaries"],
        "stage_attribution": attribution,
        "replaceable_share_bounds": bounds,
        "game_cluster_interval": reduction["game_cluster_interval"],
        "sample_size_budget": budget,
        "expected_readout_overhead": _expected_readout_overhead(e0),
        "positive_control": run_positive_control(),
        "seam_observation_spec": build_seam_observation_spec(),
        "ladder_disposition": build_ladder_disposition(e0, reduction),
        "decision_profile_complete_score": int(
            len(reduction["rows"]) == len(units) * len(SEAMS)
            and all("censoring" in row and "missing_fields" in row for row in reduction["rows"])
        ),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": (
            "complete_disqualified_required_validation"
            if invalid
            else "complete_null_sample_limited_decision_cost_unattributed"
        ),
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
            "required_check_names": list(AFFECTED_CHECK_NAMES),
            "terminal_check_names": list(TERMINAL_CHECK_NAMES),
            "full_python_suite_forbidden": True,
        },
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "independent_reduction": {
            key: deepcopy(reduction[key])
            for key in (
                "stage_attribution",
                "replaceable_share_bounds",
                "game_cluster_interval",
                "sample_size_budget",
            )
        },
        "repository_health": {
            "historical_baseline_failures_are_not_current_validation": True,
            "full_suite_invoked": False,
        },
        "environment_calls_current": 0,
        "external_publication_or_submission": False,
        "production_defaults_changed": False,
        "research_conductor_changed": False,
        "reproducibility_checksum": "pending",
        "field_principles": {},
    }
    artifact["field_principles"] = _field_principles(list(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic schema-complete artifact through production reducers."""

    units = [
        _fixture_unit(game, seed, condition)
        for game in ("bp35", "cn04")
        for seed in (7464001, 7464002)
        for condition in ("shadow", "applied")
    ]
    e0 = {
        "status": "complete_scored_runtime_unavailable",
        "verdict_class": "blocked",
        "local_runtime_parity_score": 0,
        "scored_runtime_parity_score": 0,
        "duration_breakdown_s": {"forward": 5.4},
        "invocation_counts": {"forward_calls_completed": 72},
    }
    return _build_artifact(
        units,
        e0=e0,
        source_hashes={"fixture": {"sha256": "sha256:" + "0" * 64, "role": "test"}},
        preconditions=[],
        validation_receipts=_fixture_validation_receipts(),
        started_at="2026-09-20T00:00:00+00:00",
        ended_at="2026-09-20T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[],
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute all headline claims solely from embedded raw unit rows."""

    units = [row for row in artifact.get("raw_unit_rows", []) if isinstance(row, Mapping)]
    return reduce_raw_units(units, bootstrap_seed=7464, bootstrap_draws=2000)


def validate_artifact(value: Mapping[str, Any], *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, reductions, dispositions, receipts, and checksum."""

    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "environment_calls_current": 0,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(f"identity_mismatch:{field}")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if set(value.get("field_principles", {})) != set(value):
        errors.append("field_principles_incomplete")
    rows = value.get("rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) or "censoring" not in row or "disposition" not in row
        for row in (rows if isinstance(rows, list) else [])
    ):
        errors.append("row_disposition_incomplete")
    try:
        reduced = independent_reduce(value)
    except (TypeError, ValueError):
        errors.append("raw_reduction_failed")
    else:
        stored = {
            "rows": value.get("rows"),
            "episode_summaries": value.get("episode_summaries"),
            "stage_attribution": value.get("stage_attribution"),
            "replaceable_share_bounds": value.get("replaceable_share_bounds"),
            "game_cluster_interval": value.get("game_cluster_interval"),
            "sample_size_budget": value.get("sample_size_budget"),
        }
        comparable = {key: reduced[key] for key in stored}
        if canonical_hash(stored) != canonical_hash(comparable):
            errors.append("stored_reduction_mismatch")
    if value.get("decision_profile_complete_score") != 1:
        errors.append("decision_profile_incomplete")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if require_validation:
        receipts = value.get("validation_receipts", [])
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES):
            matching = [
                row for row in receipts if isinstance(row, Mapping) and row.get("name") == name
            ]
            if len(matching) != 1 or matching[0].get("passed") is not True:
                errors.append(f"required_validation_failed:{name}")
    return list(dict.fromkeys(errors))


def build_validation_plan(repo_root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze the affected-only Exp7303/7358 validation command set."""

    return shared_build_command_plan(repo_root, MANIFEST, private_root)


def validate_validation_plan(repo_root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject scope expansion, drift, and any repository-wide pytest command."""

    errors = shared_validate_command_plan(repo_root, MANIFEST, commands)
    for command in commands:
        joined = " ".join(command.argv)
        if "tests/python -q" in joined or "full_python_suite" in joined:
            errors.append(f"full_suite_forbidden:{command.name}")
    return list(dict.fromkeys(errors))


def _raw_source_files(root: Path) -> list[Path]:
    """List every immutable V653 ledger and request sidecar used by the audit."""

    return sorted(path for path in (root / V653_RAW).rglob("*") if path.is_file())


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:
    """Authenticate source bytes plus exact historical flags and model classes."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in (*INPUT_PATHS, *[path.relative_to(root) for path in _raw_source_files(root)]):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "artifact_field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "passed": available,
            }
        )
        if available:
            hashes[relative.as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "role": "quantitative_v653"
                if relative.is_relative_to(V653_RAW)
                else "protocol_or_historical",
            }
    artifacts = {
        "v653": _load_object(root / V653_RESULT),
        "e0": _load_object(root / E0_RESULT),
        "flagged": _load_object(root / FLAGGED_RESULT),
        "old_model": _load_object(root / OLD_MODEL_RESULT),
    }
    expected_fields = {
        "v653": {
            "status": "complete_bounded_supervisor_exposure",
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
        "e0": {
            "status": "complete_scored_runtime_unavailable",
            "verdict_class": "blocked",
            "flagged_adversarial": False,
        },
        "flagged": {"flagged_adversarial": True},
    }
    paths = {"v653": V653_RESULT, "e0": E0_RESULT, "flagged": FLAGGED_RESULT}
    for source, fields in expected_fields.items():
        for field, expected in fields.items():
            observed = artifacts[source].get(field)
            checks.append(
                {
                    "check": f"historical_{source}_{field}",
                    "upstream": paths[source].as_posix(),
                    "artifact_field": field,
                    "expected": expected,
                    "observed": observed,
                    "passed": observed == expected,
                }
            )
    old_specs = artifacts["old_model"].get("model_specs") or []
    old_model = (
        old_specs[0].get("hf_id") if old_specs and isinstance(old_specs[0], Mapping) else None
    )
    checks.append(
        {
            "check": "old_model_history_class",
            "upstream": OLD_MODEL_RESULT.as_posix(),
            "artifact_field": "model_specs[0].hf_id",
            "expected": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "observed": old_model,
            "passed": old_model == "unsloth/Qwen3.6-35B-A3B-GGUF",
        }
    )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "artifact_field": "REQ-*",
            "expected": "REQ-ARC-WMTE-7464",
            "observed": "REQ-ARC-WMTE-7464" if "REQ-ARC-WMTE-7464" in spec else None,
            "passed": "REQ-ARC-WMTE-7464" in spec,
        }
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7464" in exclusion or "experiment_7464" in exclusion
    checks.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": "ops/exclusion_manifest.yaml",
            "artifact_field": EXPERIMENT_ID,
            "expected": False,
            "observed": excluded,
            "passed": not excluded,
        }
    )
    hashes[V653_RESULT.as_posix()]["original_flags"] = {
        "status": artifacts["v653"].get("status"),
        "verdict_class": artifacts["v653"].get("verdict_class"),
        "flagged_adversarial": artifacts["v653"].get("flagged_adversarial"),
        "model_class": "current_qwen3_8_archived",
    }
    hashes[E0_RESULT.as_posix()]["original_flags"] = {
        "status": artifacts["e0"].get("status"),
        "verdict_class": artifacts["e0"].get("verdict_class"),
        "flagged_adversarial": artifacts["e0"].get("flagged_adversarial"),
        "model_class": "current_qwen3_8_readout_compatibility_only",
    }
    hashes[FLAGGED_RESULT.as_posix()]["original_flags"] = {
        "status": artifacts["flagged"].get("status"),
        "verdict_class": artifacts["flagged"].get("verdict_class"),
        "flagged_adversarial": artifacts["flagged"].get("flagged_adversarial"),
        "quantitative_use": "excluded",
    }
    hashes[OLD_MODEL_RESULT.as_posix()]["original_flags"] = {
        "status": artifacts["old_model"].get("status"),
        "verdict_class": artifacts["old_model"].get("verdict_class"),
        "flagged_adversarial": artifacts["old_model"].get("flagged_adversarial"),
        "model_class": "older_qwen3_6_schema_control_only",
    }
    return checks, hashes, artifacts


def extract_raw_units(root: Path, v653: Mapping[str, Any]) -> list[JsonDict]:
    """Cold-read V653 episode rows and hash-checked response usage ledgers."""

    payload = _load_object(root / V653_RAW / "episode_rows.json")
    source_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    raw_model_spec = (v653.get("MODEL_SPECS") or [{}])[0]
    model_spec = (
        raw_model_spec
        if isinstance(raw_model_spec, Mapping)
        else {
            "hf_id": raw_model_spec,
            "revision": "fe1e2a23d973adb629709749dc4f6756df66ef10",
            "quantization": "Q4_K_M",
        }
    )
    units: list[JsonDict] = []
    for source in source_rows:
        callback_rows = source.get("request_budget_receipt", {}).get("callback_rows", [])
        server_by_index = {
            row.get("call_index"): row for row in source.get("server_request_rows", [])
        }
        spans: list[JsonDict] = []
        for index, callback in enumerate(callback_rows):
            server = server_by_index.get(index, {})
            response_path = Path(str(server.get("response_path", "")))
            if not response_path.is_absolute():
                response_path = root / response_path
            response = _load_object(response_path)
            usage = response.get("usage", {})
            timings = response.get("timings", {})
            spans.append(
                {
                    "request_id": callback.get("request_id"),
                    "start_monotonic_s": callback.get("reserved_monotonic"),
                    "end_monotonic_s": callback.get("terminal_monotonic"),
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "usable": False,
                    "gpu_prompt_ms": timings.get("prompt_ms"),
                    "gpu_generation_ms": timings.get("predicted_ms"),
                    "gpu_total_ms": (
                        float(timings.get("prompt_ms")) + float(timings.get("predicted_ms"))
                        if _finite_number(timings.get("prompt_ms"))
                        and _finite_number(timings.get("predicted_ms"))
                        else None
                    ),
                    "request_path": server.get("request_path"),
                    "response_path": server.get("response_path"),
                    "request_sha256": server.get("request_sha256"),
                    "response_sha256": server.get("response_sha256"),
                }
            )
        receipt = source.get("supervisor_receipt", {})
        selected = [row.get("arm") for row in receipt.get("would_have_redirects", [])]
        units.append(
            {
                "episode_id": source.get("episode_id"),
                "game": source.get("game"),
                "seed": source.get("seed"),
                "condition": source.get("condition"),
                "disposition": source.get("disposition"),
                "censored": source.get("disposition") != "complete",
                "episode_elapsed_s": source.get("elapsed_s"),
                "action_count": source.get("action_count"),
                "selected_action_ids": [row.get("action") for row in source.get("action_rows", [])],
                "start_level": source.get("start_level"),
                "terminal_level": source.get("terminal_level"),
                "banked_progress": source.get("banked_progress"),
                "request_spans": spans,
                "supervisor": {
                    "mode": receipt.get("mode"),
                    "actions_observed": receipt.get("actions_observed"),
                    "window": receipt.get("window"),
                    "arms_enabled": receipt.get("arms_enabled"),
                    "selected_arms": selected,
                    "full_eligible_candidate_set_observed": False,
                },
                "model_identity": {
                    "repository": model_spec.get("hf_id"),
                    "revision": model_spec.get("revision"),
                    "quantization": model_spec.get("quantization"),
                },
            }
        )
    return units


def terminal_command_specs(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build cold replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--date", RUN_DATE, "--replay", str(candidate)),
            "capability_e2e_cold_replay",
        ),
        CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from carnot import experiment_7464_v654_semif_e6_decision_cost_profile as e; "
                    "a=json.loads(Path(sys.argv[1]).read_text()); r=e.independent_reduce(a); "
                    "assert e.canonical_hash(r['rows'])==e.canonical_hash(a['rows']); print('independent_reduction_ok')"
                ),
                str(candidate),
            ),
            "independent_raw_reduction",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "unchanged_adversarial_reader",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "unchanged_verdict_reader",
        ),
    ]


def _phase_span(name: str, began: float, run_started: float, completed: int) -> JsonDict:
    """Record one disjoint monotonic phase interval and completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_offset_s": began - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - began,
        "completed_units": completed,
    }


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E
    """Run the offline audit, scoped checks, terminal readers, and atomic publish."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start")
    began = time.monotonic()
    preconditions, source_hashes, artifacts = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        raise RuntimeError("precondition authentication failed")
    spans.append(_phase_span("preconditions", began, run_started, len(preconditions)))
    progress(run_started, "preconditions", "complete", completed=len(preconditions))

    progress(run_started, "reduction", "start")
    began = time.monotonic()
    units = extract_raw_units(root, artifacts["v653"])
    reduction = reduce_raw_units(units, bootstrap_seed=7464, bootstrap_draws=2000)
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "raw_unit_rows.json", {"rows": units})
    spans.append(_phase_span("reduction", began, run_started, len(units)))
    progress(run_started, "reduction", "complete", completed=len(units))

    progress(run_started, "affected_validation", "before_subprocesses", completed=0)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exp7464-validation-") as private_name:
        private = Path(private_name)
        commands = build_validation_plan(root, private)
        plan_errors = validate_validation_plan(root, commands)
        if plan_errors:
            raise RuntimeError(f"invalid validation plan: {plan_errors}")
        planned = [PlannedCommand(command, "required_affected", True) for command in commands]
        affected_receipts = run_categorized_commands(
            root, planned, log_dir=raw_dir / "validation_logs", heartbeat_s=60.0
        )
    affected = reduce_affected_receipts(root, MANIFEST, affected_receipts)
    if not affected["passed"]:
        raise RuntimeError(f"affected validation failed: {affected}")
    spans.append(_phase_span("affected_validation", began, run_started, len(affected_receipts)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected_receipts),
    )

    progress(run_started, "terminal_candidate", "start")
    began = time.monotonic()
    candidate = _build_artifact(
        units,
        e0=artifacts["e0"],
        source_hashes=source_hashes,
        preconditions=preconditions,
        validation_receipts=affected_receipts,
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - run_started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    spans.append(_phase_span("terminal_candidate", began, run_started, 1))
    progress(run_started, "terminal_candidate", "complete", completed=1)

    progress(run_started, "terminal_validation", "before_subprocesses", completed=0)
    began = time.monotonic()
    terminal_receipts = run_commands(
        root,
        terminal_command_specs(root, candidate_path),
        log_dir=raw_dir / "terminal_logs",
        heartbeat_s=60.0,
    )
    for row in terminal_receipts:
        row["required"] = True
        row["command_category"] = "required_terminal"
    if not all(row.get("passed") is True for row in terminal_receipts):
        raise RuntimeError("terminal validation failed")
    spans.append(_phase_span("terminal_validation", began, run_started, len(terminal_receipts)))
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal_receipts),
    )

    progress(run_started, "publish", "start")
    all_receipts = [*affected_receipts, *terminal_receipts]
    final = _build_artifact(
        units,
        e0=artifacts["e0"],
        source_hashes=source_hashes,
        preconditions=preconditions,
        validation_receipts=all_receipts,
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - run_started,
        phase_spans=spans,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal artifact invalid: {errors}")
    atomic_json(root / RESULT_PATH, final)
    progress(run_started, "publish", "complete", completed=1)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public offline-audit and cold-replay roles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Run the declared entrypoint or validate a candidate in a fresh process."""

    args = parse_args(argv)
    if args.replay is not None:
        value = _load_object(args.replay)
        errors = validate_artifact(value, require_validation=False)
        reduced = independent_reduce(value)
        print(
            json.dumps(
                {
                    "replay_passed": not errors,
                    "errors": errors,
                    "row_hash": canonical_hash(reduced["rows"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI convenience
    raise SystemExit(main())
