"""Analyze corrected ARC induction evidence without new model or game work."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7557-arc-generalization"
SCHEMA = "carnot.exp7557.v660.arc_generalization.v1"
RESULT_PATH = Path("results/experiment_7557_v660_arc_generalization.json")
UPSTREAM_PATH = Path("results/experiment_7556_v660_arc_corrected_custody.json")
TELEMETRY_PATH = Path(
    "results/raw/experiment_10008_b2_induction_gate_measurement_v2/induction_gate_telemetry.jsonl"
)
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7557_v660_arc_generalization.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7557_v660_arc_generalization.py")
TEST_PATH = Path("tests/python/test_experiment_7557_v660_arc_generalization.py")
MODEL_SPECS: list[JsonDict] = []
SUPERVISOR_ARMS = (
    "drop_goal_bias",
    "allow_reinduction",
    "tool_loop_reinduction",
    "force_exploration_diversity",
)
ZERO_INVOCATION_COUNTS = {
    f"{kind}_{state}": 0
    for kind in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_RECEIPT_NAMES = (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush phase boundaries so bounded child work never appears stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7557] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so qualified input drift fails closed."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so row removal changes the evidence identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON only after temporary bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> JsonDict:
    """Load one required JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load every non-empty JSONL object in source order."""

    rows: list[JsonDict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_object_required:{path}:{number}")
        rows.append(value)
    return rows


def precondition_row(
    check: str, upstream: str, artifact_field: str, expected: Any, observed: Any
) -> JsonDict:
    """Record the exact comparison behind a dependency decision."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
        "principle": "A dependent analysis needs the named upstream value.",
    }


def collect_preconditions(root: Path, *, require_worktree_files: bool = True) -> list[JsonDict]:
    """Check the exact upstream gate and current worktree files before reduction."""

    path = root / UPSTREAM_PATH
    readable = path.is_file()
    checks = [
        precondition_row(
            "corrected_arc_upstream_available",
            UPSTREAM_PATH.as_posix(),
            "path",
            "readable_file",
            "readable_file" if readable else None,
        )
    ]
    upstream: JsonDict = {}
    if readable:
        try:
            upstream = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            checks[0].update(observed="malformed_json", passed=False)
    if upstream:
        checks += [
            precondition_row(
                "corrected_arc_ready",
                UPSTREAM_PATH.as_posix(),
                "corrected_arc_ready_score",
                1,
                upstream.get("corrected_arc_ready_score"),
            ),
            {
                **precondition_row(
                    "qualified_terminal_verdict",
                    UPSTREAM_PATH.as_posix(),
                    "verdict_class",
                    ["positive", "null", "circular_positive"],
                    upstream.get("verdict_class"),
                ),
                "op": "in",
                "passed": upstream.get("verdict_class")
                in {"positive", "null", "circular_positive"},
            },
            precondition_row(
                "upstream_not_flagged",
                UPSTREAM_PATH.as_posix(),
                "flagged_adversarial",
                False,
                upstream.get("flagged_adversarial"),
            ),
        ]
    if require_worktree_files:
        required = (
            Path("AGENTS.md"),
            Path("CODEX.md"),
            Path("CLAUDE.md"),
            Path("research-program.md"),
            Path("ops/exclusion_manifest.yaml"),
            Path("ops/e2e-test-plan.md"),
            SPEC_PATH,
            MODULE_PATH,
            WRAPPER_PATH,
            TEST_PATH,
            TELEMETRY_PATH,
        )
        for relative in required:
            present = (root / relative).is_file()
            checks.append(
                precondition_row(
                    f"required_input:{relative.as_posix()}",
                    relative.as_posix(),
                    "path",
                    "readable_file",
                    "readable_file" if present else None,
                )
            )
        try:
            spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
        except OSError:
            spec_text = ""
        checks.append(
            precondition_row(
                "requirement_declared",
                SPEC_PATH.as_posix(),
                "REQ-ARC-WMTE-7557",
                True,
                "REQ-ARC-WMTE-7557" in spec_text,
            )
        )
    return checks


def select_failed_precondition(checks: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Return the first failed prerequisite without hiding later failures."""

    return next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)


def _authenticated_telemetry(root: Path, upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Read telemetry only after its hash matches Exp7556 custody."""

    expected = (upstream.get("source_artifact_hashes") or {}).get(TELEMETRY_PATH.as_posix())
    path = root / TELEMETRY_PATH
    if not path.is_file():
        raise ValueError(f"raw_path_missing:{TELEMETRY_PATH.as_posix()}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"raw_hash_mismatch:{TELEMETRY_PATH.as_posix()}:expected={expected}:observed={observed}"
        )
    return load_jsonl(path)


def _support_gate(check: str, expected: Any, observed: Any, op: str, passed: bool) -> JsonDict:
    """Keep each sample floor explicit rather than collapsing it."""

    return {
        "check": check,
        "category": "support",
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": "Insufficient evidence cannot become a numerical gate claim.",
    }


def _endpoint_summary(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate historical movement, causal efficacy, and actual progress."""

    frame_count = sum(row.get("frame_change_progress") is True for row in attempts)
    level_count = sum(row.get("level_up_progress") is True for row in attempts)
    has_model = any(int(row.get("content_bytes") or 0) > 0 for row in attempts)
    has_plan = any(row.get("planned") is True for row in attempts)
    missing = []
    if not has_model:
        missing.append("accepted_world_model")
    if not has_plan:
        missing += ["executed_plan", "plan_to_action", "attributable_progress"]
    return {
        "bounded_frame_change": {
            "available": True,
            "positive_count": frame_count,
            "attempt_count": len(attempts),
            "window_actions": 32,
            "causal_efficacy": False,
            "disposition": "comparability_only_incidental_movement",
        },
        "plan_linked_execution": {
            "available": not missing,
            "observed": 0 if not missing else None,
            "missing_joins": missing,
            "disposition": "available" if not missing else "unavailable_missing_causal_joins",
        },
        "actual_level_progress": {
            "available": True,
            "positive_count": level_count,
            "attempt_count": len(attempts),
            "disposition": "observed_live_level_endpoint",
        },
    }


def _supervisor_rows(telemetry: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], int]:
    """Count each arm and credit help only after a bounded level-gain join."""

    endings = {
        str(row.get("episode_id")): row
        for row in telemetry
        if row.get("record_type") == "episode_end"
    }
    selections = [row for row in telemetry if row.get("seam") == "supervisor_arm_selection"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[str(row.get("chosen_arm"))].append(row)
    result = []
    for arm in SUPERVISOR_ARMS:
        events = grouped.get(arm, [])
        helped = 0
        for event in events:
            ending = endings.get(str(event.get("episode_id"))) or {}
            before, after = event.get("level_before"), ending.get("level_end")
            helped += int(isinstance(before, int) and isinstance(after, int) and after > before)
        result.append(
            {
                "row_type": "supervisor_arm",
                "arm": arm,
                "fired": len(events),
                "helped": helped,
                "games_fired": len({str(row.get("game_id")) for row in events}),
                "help_definition": "later_episode_level_gain",
                "disposition": "observed" if events else "no_firing_observed",
            }
        )
    return result, len(grouped.get("no_redirect", []))


def _per_game_rows(
    attempts: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Retain absolute game counts, tokens, observed cost, and uncertainty."""

    costs = {
        str(row.get("attempt_id")): float(row.get("seconds") or row.get("wall_time_s") or 0.0)
        for row in telemetry
        if row.get("record_type") == "induction_attempt"
    }
    result = []
    for game in sorted({str(row.get("game")) for row in attempts}):
        game_attempts = [row for row in attempts if row.get("game") == game]
        game_units = [row for row in schedule if row.get("game") == game]
        result.append(
            {
                "row_type": "game",
                "game": game,
                "attempts": len(game_attempts),
                "completed_responses": sum(
                    row.get("response_disposition") == "completed" for row in game_attempts
                ),
                "completion_tokens": sum(
                    int(row.get("completion_tokens") or 0) for row in game_attempts
                ),
                "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in game_attempts),
                "generation_wall_time_s": sum(
                    costs.get(str(row.get("attempt_id")), 0.0) for row in game_attempts
                ),
                "frame_change_attempts": sum(
                    row.get("frame_change_progress") is True for row in game_attempts
                ),
                "plan_linked_useful_attempts": None,
                "level_gains": sum(int(row.get("new_level_credit") or 0) for row in game_units),
                "uncertainty_available": False,
                "solve_provenance": "live_agent_self_discovery",
                "disposition": "complete_observed_feasibility",
            }
        )
    return result


def reduce_evidence(root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    """Reduce qualified rows without converting missing causal labels to false."""

    telemetry = _authenticated_telemetry(root, upstream)
    attempts = [deepcopy(dict(row)) for row in upstream.get("induction_attempt_rows") or []]
    schedule = [deepcopy(dict(row)) for row in upstream.get("rows") or []]
    opportunities = sum(
        row.get("seam") == "induction_timing" and row.get("record_type") == "decision"
        for row in telemetry
    )
    endpoints = _endpoint_summary(attempts)
    identifiable = endpoints["plan_linked_execution"]["available"] is True
    useful = [row for row in attempts if identifiable and row.get("level_up_progress") is True]
    useless = [row for row in attempts if identifiable and row.get("level_up_progress") is False]
    support = {
        "opportunities": opportunities,
        "attempts": len(attempts),
        "useful_attempts": len(useful),
        "useless_attempts": len(useless),
        "useful_games": len({str(row.get("game")) for row in useful}),
        "useless_games": len({str(row.get("game")) for row in useless}),
    }
    gates = [
        _support_gate("opportunity_floor", 1000, opportunities, ">=", opportunities >= 1000),
        _support_gate("attempt_floor", 100, len(attempts), ">=", len(attempts) >= 100),
        _support_gate(
            "both_outcome_classes",
            "useful>0 and useless>0",
            {"useful": len(useful), "useless": len(useless)},
            "both",
            bool(useful and useless),
        ),
        _support_gate(
            "useful_game_floor", 4, support["useful_games"], ">=", support["useful_games"] >= 4
        ),
        _support_gate(
            "useless_game_floor", 4, support["useless_games"], ">=", support["useless_games"] >= 4
        ),
    ]
    arms, unredirected = _supervisor_rows(telemetry)
    per_game = _per_game_rows(attempts, schedule, telemetry)
    oracle = {
        "analysis_only": True,
        "endpoint_identifiable": identifiable,
        "observed_useful_attempts_retained": len(useful),
        "observed_useless_attempts_suppressed": len(useless),
        "optimistic_tokens_avoidable": (
            sum(int(row.get("completion_tokens") or 0) for row in useless) if identifiable else None
        ),
        "counterfactual_policy_value": None,
        "disposition": (
            "optimistic_logged_data_headroom_only"
            if identifiable
            else "unavailable_missing_causal_outcomes"
        ),
    }
    source_hashes = deepcopy(dict(upstream.get("source_artifact_hashes") or {}))
    source_hashes[UPSTREAM_PATH.as_posix()] = sha256_file(root / UPSTREAM_PATH)
    return {
        "source_artifact_hashes": source_hashes,
        "historical_model_calls": deepcopy(upstream.get("historical_model_calls") or {}),
        "live_policy_path": deepcopy(upstream.get("live_policy_path") or {}),
        "attempt_rows": attempts,
        "schedule_rows": schedule,
        "sample_size_budget": deepcopy(upstream.get("sample_size_budget") or {}),
        "endpoint_identifiability": endpoints,
        "outcome_counts": {
            "useful_identifiable": len(useful),
            "useless_identifiable": len(useless),
            "unidentifiable": 0 if identifiable else len(attempts),
        },
        "support_counts": support,
        "support_gate_results": gates,
        "numeric_gate_quality_claim": all(row["passed"] for row in gates),
        "gate_fit": None,
        "gate_ready_to_ship": False,
        "analysis_only_oracle": oracle,
        "supervisor_arm_rows": arms,
        "stagnations_unredirected": unredirected,
        "per_game_results": per_game,
        "rows": [*per_game, *arms],
        "successor_hypothesis": None,
        "telemetry_receipt": {
            "path": TELEMETRY_PATH.as_posix(),
            "sha256": sha256_file(root / TELEMETRY_PATH),
            "row_count": len(telemetry),
            "gate_opportunity_count": opportunities,
        },
        "solve_provenance": {
            "required": "live_agent_self_discovery",
            "new_level_solve_claimed": False,
            "credited_level_count": sum(int(row.get("new_level_credit") or 0) for row in schedule),
        },
    }


def validate_reduction(value: Mapping[str, Any]) -> list[str]:
    """Reject endpoint, support, supervisor, or solve mutations."""

    errors = []
    support = value.get("support_counts") or {}
    if support.get("opportunities") != 11813:
        errors.append("opportunity_count_mismatch")
    if support.get("attempts") != 35 or len(value.get("attempt_rows") or []) != 35:
        errors.append("attempt_count_mismatch")
    plan = (value.get("endpoint_identifiability") or {}).get("plan_linked_execution") or {}
    if plan.get("available") is not False or plan.get("observed") is not None:
        errors.append("causal_endpoint_misclassified")
    if value.get("numeric_gate_quality_claim") is not False or value.get("gate_fit") is not None:
        errors.append("unsupported_numeric_fit")
    arms = value.get("supervisor_arm_rows") or []
    if {row.get("arm") for row in arms} != set(SUPERVISOR_ARMS):
        errors.append("supervisor_arm_coverage_mismatch")
    if any(int(row.get("helped") or 0) > int(row.get("fired") or 0) for row in arms):
        errors.append("supervisor_help_exceeds_firings")
    if value.get("gate_ready_to_ship") is not False:
        errors.append("retrospective_ship_claim")
    if (value.get("solve_provenance") or {}).get("new_level_solve_claimed") is not False:
        errors.append("new_solve_claimed")
    return errors


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    op: str = "==",
) -> JsonDict:
    """Keep validity, readiness, support, and benefit distinct."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require exactly one clean receipt for each required command."""

    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        by_name[str(row.get("name"))].append(row)
    return all(
        len(by_name[name]) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is False
        for name in REQUIRED_RECEIPT_NAMES
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all gate failures and identify the first."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _code_hashes(root: Path) -> JsonDict:
    """Bind implementation, entrypoint, tests, and specification bytes."""

    return {
        path.as_posix(): sha256_file(root / path) if (root / path).is_file() else None
        for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH)
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable code, settings, source identities, and reduced rows."""

    keys = (
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "code_hashes",
        "source_artifact_hashes",
        "live_policy_path",
        "endpoint_identifiability",
        "support_counts",
        "supervisor_arm_rows",
        "per_game_results",
        "rows",
        "sample_size_budget",
        "analysis_sample_size_budget",
        "acceptance_gate_results",
        "honest_verdict",
        "verdict_class",
        "arc_analysis_complete_score",
    )
    return canonical_hash({key: deepcopy(artifact.get(key)) for key in keys})


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain the audit failure prevented by every top-level field."""

    specific = {
        "experiment_id": "Bind the exact task ID rather than a nearby ARC report.",
        "preconditions_checked": "Show required input values before analysis.",
        "MODEL_SPECS": "Prevent historical identity from implying a current load.",
        "model_specs": "Keep both canonical no-load spellings consistent.",
        "model_invoked": "Separate current work from historical model calls.",
        "invocation_counts": "Expose current load, forward, and generation dispositions.",
        "inference_substrate_class": "Apply aggregation without padded duration.",
        "inference_substrate": "Name upstream-artifact aggregation exactly.",
        "execution_venue": "Use the legal host enum instead of host_cpu.",
        "duration_s": "Measure current work rather than historical runtime.",
        "random_seed": "Freeze fitting, controls, and bootstrap before outcomes.",
        "reproducibility_checksum": "Detect changed code, sources, settings, or rows.",
        "rows": "Keep every game and arm disposition; missing is not zero.",
        "sample_size_budget": "Separate completed, censored, and unstarted units.",
        "acceptance_gate_results": "Keep evidence validity independent of benefit.",
        "gate_check_summary": "Name exact expected and observed gate failures.",
        "honest_verdict": "Use a terminal prefix without promoting feasibility.",
        "verdict_class": "Use the closed scientific terminal vocabulary.",
        "verifier_is_oracle": "Prevent optimistic headroom from becoming proof.",
        "flagged_adversarial": "Preserve upstream determinations.",
        "validation_receipts": "Store scoped, replay, reduction, and strict-reader results.",
        "arc_analysis_complete_score": "Record analysis completeness, not benefit.",
        "numeric_gate_quality_claim": "Suppress fitting below support floors.",
        "gate_ready_to_ship": "Prevent retrospective selection from enabling policy.",
        "per_game_results": "Retain absolute counts, tokens, costs, and uncertainty.",
        "solve_provenance": "Deny a new solve without live self-discovery evidence.",
        "endpoint_identifiability": "Prevent movement from repairing missing joins.",
        "supervisor_arm_rows": "Keep zero-firing arms and observed help separate.",
    }
    return {
        key: specific.get(key, f"Retain {key} so readers detect omission or drift.") for key in keys
    }


def build_artifact(
    root: Path,
    run_date: str,
    reduced: Mapping[str, Any],
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a valid null whose readiness is independent of benefit."""

    reduction_valid = not validate_reduction(reduced)
    validation_passed = _receipts_pass(validation_receipts)
    ready = int(reduction_valid and validation_passed)
    endpoints = deepcopy(dict(reduced.get("endpoint_identifiability") or {}))
    plan_available = (endpoints.get("plan_linked_execution") or {}).get("available") is True
    gates = [
        _gate(
            "qualified_corrected_custody",
            "validity",
            True,
            reduction_valid,
            reduction_valid,
            "Invalid upstream custody cannot support science.",
        ),
        _gate(
            "required_scoped_and_terminal_validation",
            "validity",
            True,
            validation_passed,
            validation_passed,
            "Scoped checks and cold replay must pass before readiness.",
        ),
        _gate(
            "arc_analysis_complete",
            "readiness",
            1,
            ready,
            ready == 1,
            "A valid null stays auditable because readiness is not benefit.",
        ),
        *[deepcopy(dict(row)) for row in reduced.get("support_gate_results") or []],
        _gate(
            "plan_linked_efficacy_identifiable",
            "benefit",
            True,
            plan_available,
            plan_available,
            "Missing causal joins make efficacy unavailable, not false.",
        ),
        _gate(
            "live_intervention_before_shipping",
            "readiness",
            False,
            False,
            True,
            "Retrospective selection cannot establish safe deployment.",
        ),
    ]
    support = reduced.get("support_counts") or {}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_null_feasibility_only_causal_endpoint_unavailable",
        "honest_verdict": "complete_null_feasibility_only_causal_endpoint_unavailable",
        "verdict_class": "null",
        "positive_claim": False,
        "no_headroom": False,
        "no_headroom_annotation": "Missing causal outcomes prevent a no-headroom claim.",
        "flagged_adversarial": False,
        "verifier_is_oracle": True,
        "oracle_scope": "optimistic_logged_data_headroom_not_counterfactual_policy_value",
        "arc_analysis_complete_score": ready,
        "numeric_gate_quality_claim": reduced.get("numeric_gate_quality_claim", False),
        "gate_ready_to_ship": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "new_model_calls": 0,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": deepcopy(reduced.get("historical_model_calls") or {}),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "device_compute": {
            "current_device": "host CPU",
            "current_work": "authenticated JSON aggregation",
            "cuda_used_by_current_work": False,
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "worktree_root": str(root.resolve())},
        "duration_accounting": {
            "current_work_s": float(duration_s),
            "authoring_included": False,
            "validation_included": True,
            "historical_gpu_time_included": False,
        },
        "random_seed": {
            "fitting_seed": 7557001,
            "matched_rate_control_seed": 7557002,
            "cluster_bootstrap_seed": 7557003,
            "outcome_observed_before_seed_change": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "source_artifact_hashes": deepcopy(reduced.get("source_artifact_hashes") or {}),
        "code_hashes": _code_hashes(root),
        "live_policy_path": deepcopy(reduced.get("live_policy_path") or {}),
        "telemetry_receipt": deepcopy(reduced.get("telemetry_receipt") or {}),
        "endpoint_identifiability": endpoints,
        "support_counts": deepcopy(support),
        "gate_fit": None,
        "analysis_only_oracle": deepcopy(reduced.get("analysis_only_oracle") or {}),
        "supervisor_arm_rows": deepcopy(reduced.get("supervisor_arm_rows") or []),
        "stagnations_unredirected": reduced.get("stagnations_unredirected", 0),
        "successor_hypothesis": None,
        "per_game_results": deepcopy(reduced.get("per_game_results") or []),
        "rows": deepcopy(reduced.get("rows") or []),
        "sample_size_budget": deepcopy(reduced.get("sample_size_budget") or {}),
        "analysis_sample_size_budget": {
            "planned": 1000,
            "attempted": support.get("opportunities", 0),
            "completed": support.get("opportunities", 0),
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "actual_attempts": support.get("attempts", 0),
        },
        "solve_provenance": deepcopy(reduced.get("solve_provenance") or {}),
        "production_defaults_changed": False,
        "game_source_read": False,
        "kernel_submitted": False,
        "publication_mode": "none",
        "applicable_numbered_e2e": [],
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def build_blocked_artifact(
    run_date: str, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Publish external absence as blocked while retaining the planned class."""

    failed = select_failed_precondition(checks) or precondition_row(
        "corrected_arc_upstream_available",
        UPSTREAM_PATH.as_posix(),
        "path",
        "readable_file",
        None,
    )
    failed["category"] = "validity"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_blocked_corrected_arc_not_ready",
        "honest_verdict": "complete_blocked_corrected_arc_not_ready",
        "verdict_class": "blocked",
        "positive_claim": False,
        "no_headroom": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "arc_analysis_complete_score": 0,
        "numeric_gate_quality_claim": False,
        "gate_ready_to_ship": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "new_model_calls": 0,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {},
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid()},
        "random_seed": {
            "fitting_seed": 7557001,
            "matched_rate_control_seed": 7557002,
            "cluster_bootstrap_seed": 7557003,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": {},
        "code_hashes": {},
        "live_policy_path": {},
        "endpoint_identifiability": {},
        "support_counts": {
            "opportunities": 0,
            "attempts": 0,
            "useful_attempts": 0,
            "useless_attempts": 0,
            "useful_games": 0,
            "useless_games": 0,
        },
        "gate_fit": None,
        "analysis_only_oracle": {},
        "supervisor_arm_rows": [],
        "stagnations_unredirected": None,
        "successor_hypothesis": None,
        "per_game_results": [],
        "rows": [],
        "sample_size_budget": {
            "planned": 144,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 144,
        },
        "analysis_sample_size_budget": {
            "planned": 1000,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 1000,
            "actual_attempts": 0,
        },
        "solve_provenance": {
            "required": "live_agent_self_discovery",
            "new_level_solve_claimed": False,
            "credited_level_count": 0,
        },
        "production_defaults_changed": False,
        "game_source_read": False,
        "kernel_submitted": False,
        "publication_mode": "none",
        "applicable_numbered_e2e": [],
        "acceptance_gate_results": [failed],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "first_failure": deepcopy(failed),
            "failures": [deepcopy(failed)],
        },
        "validation_receipts": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute completeness and the null result from published rows."""

    support = artifact.get("support_counts") or {}
    receipts_passed = _receipts_pass(artifact.get("validation_receipts") or [])
    row_count = len(artifact.get("rows") or [])
    valid_rows = row_count == 16 and len(artifact.get("per_game_results") or []) == 12
    endpoint = artifact.get("endpoint_identifiability") or {}
    causal_available = (endpoint.get("plan_linked_execution") or {}).get("available") is True
    complete = int(
        receipts_passed
        and valid_rows
        and support.get("opportunities") == 11813
        and support.get("attempts") == 35
    )
    return {
        "arc_analysis_complete_score": complete,
        "verdict_class": "null" if complete and not causal_available else None,
        "opportunities": support.get("opportunities"),
        "attempts": support.get("attempts"),
        "numeric_gate_quality_claim": bool(artifact.get("numeric_gate_quality_claim")),
        "gate_ready_to_ship": bool(artifact.get("gate_ready_to_ship")),
        "row_count": row_count,
        "required_validation_passed": receipts_passed,
        "causal_endpoint_available": causal_available,
    }


def validate_artifact(artifact: Mapping[str, Any], *, require_terminal: bool = False) -> list[str]:
    """Cold-check identity, no-model accounting, reduction, and readiness."""

    errors = []
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("experiment_identity_mismatch")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    score = artifact.get("arc_analysis_complete_score")
    if type(score) is not int or score not in {0, 1}:
        errors.append("analysis_complete_score_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("current_model_specs_not_empty")
    if artifact.get("model_invoked") is not False or artifact.get("new_model_calls") != 0:
        errors.append("current_model_invoked")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("planned_inference_substrate_class") != "aggregation":
        errors.append("planned_substrate_class_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("positive_claim") is not False:
        errors.append("positive_claim_invalid")
    if artifact.get("gate_ready_to_ship") is not False:
        errors.append("retrospective_ship_claim")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles") or {}
    if set(artifact) - {"field_principles"} > set(principles):
        errors.append("field_principles_incomplete")
    if any(not row.get("principle") for row in artifact.get("acceptance_gate_results") or []):
        errors.append("gate_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if score != 0 or artifact.get("honest_verdict") != (
            "complete_blocked_corrected_arc_not_ready"
        ):
            errors.append("blocked_classification_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_mismatch")
        if not isinstance((artifact.get("gate_check_summary") or {}).get("first_failure"), Mapping):
            errors.append("blocked_gate_summary_missing")
        return list(dict.fromkeys(errors))
    if (
        artifact.get("honest_verdict")
        != ("complete_null_feasibility_only_causal_endpoint_unavailable")
        or artifact.get("verdict_class") != "null"
    ):
        errors.append("scientific_verdict_mismatch")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_mismatch")
    replay = independent_reduce(artifact)
    if replay["row_count"] != 16:
        errors.append("row_reduction_mismatch")
    if replay["opportunities"] != 11813 or replay["attempts"] != 35:
        errors.append("support_reduction_mismatch")
    if artifact.get("numeric_gate_quality_claim") is not False:
        errors.append("unsupported_numeric_claim")
    plan = (artifact.get("endpoint_identifiability") or {}).get("plan_linked_execution") or {}
    if plan.get("available") is not False:
        errors.append("causal_endpoint_misclassified")
    if score != replay["arc_analysis_complete_score"]:
        errors.append("analysis_complete_score_mismatch")
    if require_terminal and replay["required_validation_passed"] is not True:
        errors.append("required_validation_failed")
    if require_terminal and score != 1:
        errors.append("terminal_readiness_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> JsonDict:
    """Reload a candidate and recompute its reduction in a fresh process."""

    artifact = load_json(path)
    errors = validate_artifact(artifact, require_terminal=False)
    if errors:
        raise ValueError("cold_replay_invalid:" + ",".join(errors))
    return independent_reduce(artifact)


def _terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Declare cold replay, independent reduction, and strict readers."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--date", RUN_DATE, "--cold-replay", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--date", RUN_DATE, "--validate", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
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
            "terminal_candidate",
        ),
    ]


def _phase_span(phase: str, phase_started: float, run_started: float) -> JsonDict:
    """Record one monotonic phase interval."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def run_experiment(  # pragma: no cover - exercised by the declared capability entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Authenticate, validate, cold-replay, and atomically publish Exp7557."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    spans = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    checks = collect_preconditions(root)
    spans.append(_phase_span("preconditions", phase_started, started))
    failure = select_failed_precondition(checks)
    progress(started, "preconditions", "end", passed=failure is None)
    if failure is not None:
        blocked = build_blocked_artifact(run_date, checks, duration_s=time.monotonic() - started)
        progress(started, "publication", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "publication", "after_atomic_blocked", path=output_path)
        return blocked

    progress(started, "reduction", "start")
    phase_started = time.monotonic()
    reduced = reduce_evidence(root, load_json(root / UPSTREAM_PATH))
    reduction_errors = validate_reduction(reduced)
    spans.append(_phase_span("reduction", phase_started, started))
    progress(started, "reduction", "end", errors=len(reduction_errors))
    if reduction_errors:
        raise RuntimeError("arc_generalization_reduction_invalid:" + ",".join(reduction_errors))

    private_root = Path(tempfile.mkdtemp(prefix="exp7557-validation-", dir="/tmp"))
    raw_dir = root / "results/raw/experiment_7557_v660_arc_generalization"
    coverage_file = private_root / ".coverage.exp7557"
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=private_root / "pytest",
        coverage_file=coverage_file,
    )
    progress(started, "validation", "before_scoped_subprocesses", units=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation/affected",
        extra_env={"COVERAGE_FILE": str(coverage_file)},
    )
    for row in affected:
        row["command_environment"] = {"COVERAGE_FILE": str(coverage_file)}
    spans.append(_phase_span("validation", phase_started, started))
    progress(
        started,
        "validation",
        "after_scoped_subprocesses",
        passed=all(row["passed"] for row in affected),
    )
    if not all(row["passed"] for row in affected):
        raise RuntimeError("scoped_validation_failed")

    candidate = build_artifact(
        root,
        run_date,
        reduced,
        preconditions_checked=checks,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "terminal_validation", "before_subprocesses", units=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_phase_span("terminal_validation", phase_started, started))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row["passed"] for row in terminal),
    )
    if not all(row["passed"] for row in terminal):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        root,
        run_date,
        reduced,
        preconditions_checked=checks,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, require_terminal=True)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publication", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    progress(started, "publication", "after_atomic_terminal", path=output_path)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the entrypoint and its read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Dispatch one analysis run or fresh-process reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    if args.validate is not None:
        value = load_json(args.validate)
        errors = validate_artifact(value, require_terminal=False)
        if errors:
            raise ValueError("artifact_invalid:" + ",".join(errors))
        print(json.dumps(independent_reduce(value), sort_keys=True), flush=True)
        return 0
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution convenience.
    raise SystemExit(main())
