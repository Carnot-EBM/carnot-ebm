"""Independently audit the two V662 ARC live panels.

The module treats missing producers as evidence boundaries. It never converts
an absent episode into live efficacy evidence.

Spec refs: REQ-ARC-WMTE-7584 and SCENARIO-ARC-WMTE-7584-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7584-arc-independent-audit"
MILESTONE = "2026.09.662"
SCHEMA = "carnot.exp7584.v662.arc_independent_audit.v1"
HISTORICAL_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
RESULT_PATH = Path("results/experiment_7584_v662_arc_independent_audit.json")
RAW_DIR = Path("results/raw/experiment_7584_v662_arc_independent_audit")
MODULE_PATH = Path("python/carnot/experiment_7584_v662_arc_independent_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7584_v662_arc_independent_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7584_v662_arc_independent_audit.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
PROTOCOL_PRODUCER_PATH = Path("results/experiment_7580_v662_arc_verifier_support.json")
CANARY_PATH = Path("results/experiment_7581_v662_arc_bounded_canary.json")
PRE_GATE_PANEL_A_PATH = Path("results/experiment_7582_arc_panel_a.json")
PRODUCERS = {
    "panel_a": Path("results/experiment_7582_v662_arc_panel_a.json"),
    "panel_b": Path("results/experiment_7583_v662_arc_panel_b.json"),
}
PANEL_GAMES = {
    "A": ("su15", "sp80", "ft09"),
    "B": ("sb26", "g50t", "dc22"),
}
SEEDS = (7582001, 7582002)
ARMS = ("current_verifier", "integrity_guard")
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
CAPABILITY_CHECK_NAMES = (
    "e2e_009",
    "e2e_010",
    "e2e_011",
    "e2e_012",
    "e2e_013",
    "llm_off_environment_smoke",
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "source_artifact_hashes",
    "validation_receipts",
    "field_principles",
    "verifier_is_oracle",
    "arc_claims_qualified_score",
    "panel_dispositions",
    "support_comparison_rows",
    "supervisor_refinement_supported",
    "solve_provenance",
    "b2_gate_fit_allowed",
)
ZERO_INVOCATION_COUNTS = {
    operation: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for operation in ("model_loads", "forward_calls", "generation_calls", "tokens")
}


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so any removed row changes the audit identity."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> Json:
    """Read one JSON object. Invalid external bytes remain an empty object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_receipt(path: Path, root: Path, upstream: str, evidence_stage: str) -> Json:
    """Bind present evidence to bytes and name its producer stage."""

    return {
        "upstream": upstream,
        "path": _path_label(path, root),
        "present": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "evidence_stage": evidence_stage,
    }


def absent_source_receipt(path: Path, upstream: str) -> Json:
    """Distinguish an absent producer from a present pre-gate diagnostic."""

    return {
        "upstream": upstream,
        "path": path.as_posix(),
        "present": False,
        "sha256": None,
        "bytes": 0,
        "evidence_stage": "absent_producer",
    }


def authenticate_source_receipt(receipt: Mapping[str, Any], root: Path) -> Path:
    """Reject any byte or size change in a declared present source."""

    if receipt.get("present") is not True:
        raise ValueError("source_not_present")
    path = Path(str(receipt.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != receipt.get("sha256"):
        raise ValueError(f"source_hash_mismatch:{path}")
    if resolved.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"source_size_mismatch:{path}")
    return resolved


def check_row(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    op: str = "eq",
) -> Json:
    """Keep every operand needed to reproduce a failed prerequisite."""

    if op == "eq":
        passed = observed == expected
    elif op == "in":
        passed = observed in expected
    elif op == "starts_with":
        passed = str(observed).startswith(str(expected))
    else:
        raise ValueError(f"unknown_check_op:{op}")
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "required": True,
    }


def collect_preconditions(root: Path) -> tuple[list[Json], dict[str, Json]]:
    """Inspect both producer paths even when the first one is blocked."""

    checks: list[Json] = []
    producers: dict[str, Json] = {}
    required = ("AGENTS.md", "CODEX.md", "CLAUDE.md", SPEC_PATH.as_posix())
    for label in required:
        checks.append(
            check_row("required_path", "worktree", label, "exists", True, (root / label).is_file())
        )
    spec = root / SPEC_PATH
    checks.append(
        check_row(
            "matching_requirement",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7584",
            True,
            spec.is_file() and "REQ-ARC-WMTE-7584" in spec.read_text(encoding="utf-8"),
        )
    )
    for panel, relative in PRODUCERS.items():
        path = root / relative
        exists = path.is_file()
        checks.append(
            check_row(
                "producer_exists",
                f"exp758{2 if panel == 'panel_a' else 3}",
                relative.as_posix(),
                "exists",
                True,
                exists,
            )
        )
        producer = load_json(path) if exists else {}
        producers[panel] = producer
        if exists:
            checks.extend(
                [
                    check_row(
                        "producer_terminal",
                        panel,
                        relative.as_posix(),
                        "honest_verdict",
                        "complete_",
                        producer.get("honest_verdict"),
                        "starts_with",
                    ),
                    check_row(
                        "producer_unflagged",
                        panel,
                        relative.as_posix(),
                        "flagged_adversarial",
                        False,
                        producer.get("flagged_adversarial"),
                    ),
                    check_row(
                        "producer_class",
                        panel,
                        relative.as_posix(),
                        "verdict_class",
                        ("null", "positive", "circular_positive"),
                        producer.get("verdict_class"),
                        "in",
                    ),
                ]
            )
    return checks, producers


def fixture_protocol() -> Json:
    """Return the registered panel roster used only by isolated unit tests."""

    return {
        "schema": "carnot.exp7580.live_panel_protocol.v1",
        "panels": {key: list(value) for key, value in PANEL_GAMES.items()},
        "seeds": list(SEEDS),
        "arms": list(ARMS),
    }


def protocol_rows(protocol: Mapping[str, Any]) -> list[Json]:
    """Expand the frozen roster without inventing any live outcome."""

    output: list[Json] = []
    for panel, games in (protocol.get("panels") or {}).items():
        for game in games:
            for seed in protocol.get("seeds") or []:
                for arm in protocol.get("arms") or []:
                    output.append({"panel": panel, "game": game, "seed": seed, "arm": arm})
    return output


def _required_hash(row: Mapping[str, Any], name: str) -> None:
    value = row.get(name)
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{name}_invalid")


def _validate_episode_row(row: Mapping[str, Any]) -> None:
    if row.get("model_id") != HISTORICAL_MODEL_ID:
        raise ValueError("wrong_model_id")
    for name in (
        "request_sha256",
        "transition_sha256",
        "verifier_sha256",
        "plan_sha256",
        "action_sha256",
    ):
        _required_hash(row, name)
    prompt = [str(value) for value in row.get("prompt_transition_ids") or []]
    heldout = [str(value) for value in row.get("heldout_transition_ids") or []]
    if set(prompt) & set(heldout):
        raise ValueError("transition_overlap")
    if len(heldout) != len(set(heldout)):
        raise ValueError("duplicate_heldout")
    denominator = int(row.get("heldout_denominator", -1))
    if denominator != len(heldout) or denominator < 2:
        raise ValueError("heldout_denominator_invalid")
    if int(row.get("heldout_attempted", -1)) != denominator:
        raise ValueError("raised_transition_missing")
    if row.get("engine_order_independent", True) is not True:
        raise ValueError("order_dependent_engine")
    if bool(row.get("response_cap_hit")) and bool(row.get("accepted")):
        raise ValueError("capped_response_accepted")
    progress_count = int(row.get("executed_plan_progress_count", 0))
    action_count = int(row.get("executed_plan_action_count", 0))
    if progress_count > action_count:
        raise ValueError("unexecuted_plan_progress")
    if row.get("provenance") != "live_agent_self_discovery":
        raise ValueError("solve_provenance_invalid")


def _contrast(values: Sequence[float]) -> Json:
    """Return a broad descriptive range without an independence claim."""

    if not values:
        return {
            "raw_numerator": 0.0,
            "raw_denominator": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "raw_numerator": float(sum(values)),
        "raw_denominator": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def reduce_episode_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce paired episodes while six games remain the only clusters."""

    by_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    games: set[str] = set()
    for row in rows:
        _validate_episode_row(row)
        unit_id = str(row.get("unit_id") or "")
        arm = str(row.get("arm") or "")
        if not unit_id or arm not in ARMS or arm in by_unit[unit_id]:
            raise ValueError("episode_arm_roster_invalid")
        by_unit[unit_id][arm] = row
        games.add(str(row.get("game") or ""))
    if not by_unit or any(set(pair) != set(ARMS) for pair in by_unit.values()):
        raise ValueError("episode_pair_incomplete")
    acceptance: list[float] = []
    cost: list[float] = []
    progress: list[float] = []
    for pair in by_unit.values():
        current = pair["current_verifier"]
        guarded = pair["integrity_guard"]
        acceptance_delta = float(bool(guarded["accepted"])) - float(bool(current["accepted"]))
        if acceptance_delta and guarded.get("guard_registered") is not True:
            raise ValueError("acceptance_change_unattributed")
        acceptance.append(acceptance_delta)
        cost.append(float(current.get("cost_usd", 0.0)) - float(guarded.get("cost_usd", 0.0)))
        progress.append(
            float(guarded.get("executed_plan_progress_count", 0))
            - float(current.get("executed_plan_progress_count", 0))
        )
    accepted_change = _contrast(acceptance)
    accepted_change["direction"] = "guard_minus_current_positive_is_more_acceptance"
    accepted_change["attributed_to"] = "registered_research_guard"
    return {
        "row_count": len(rows),
        "episode_count": len(by_unit),
        "game_cluster_count": len(games),
        "cluster_unit": "game",
        "pooled_independence_claim": False,
        "attempted_calls": sum(int(row.get("generation_attempted", 0)) for row in rows),
        "completed_calls": sum(int(row.get("generation_completed", 0)) for row in rows),
        "cap_hit_count": sum(bool(row.get("response_cap_hit")) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in rows),
        "raised_heldout_count": sum(len(row.get("raised_transition_ids") or []) for row in rows),
        "frame_motion_total": sum(int(row.get("frame_motion_count", 0)) for row in rows),
        "executed_plan_progress_total": sum(
            int(row.get("executed_plan_progress_count", 0)) for row in rows
        ),
        "paired_contrasts": {
            "acceptance": accepted_change,
            "cost": {
                **_contrast(cost),
                "direction": "current_minus_guard_positive_is_lower_guard_cost",
            },
            "executed_progress": {
                **_contrast(progress),
                "direction": "guard_minus_current_positive_is_more_executed_progress",
            },
        },
    }


def mutate_rows(rows: list[Json], mutation: str) -> None:
    """Apply one private corruption used to prove the reducer fails closed."""

    if mutation == "dropped_exception":
        rows[0]["heldout_attempted"] = 1
    elif mutation == "duplicated_heldout":
        rows[0]["heldout_transition_ids"] = ["h0", "h0"]
    elif mutation == "wrong_model_id":
        rows[0]["model_id"] = "wrong/model"
    elif mutation == "unexecuted_plan":
        rows[0]["executed_plan_progress_count"] = 1
        rows[0]["executed_plan_action_count"] = 0
    elif mutation == "capped_response":
        rows[1]["response_cap_hit"] = True
        rows[1]["accepted"] = True
    else:
        raise ValueError(f"unknown_mutation:{mutation}")


def _mutation_fixture() -> list[Json]:
    rows: list[Json] = []
    for arm, accepted, executed in (
        ("current_verifier", False, 0),
        ("integrity_guard", True, 1),
    ):
        rows.append(
            {
                "unit_id": "fixture:7582001",
                "panel": "A",
                "game": "su15",
                "seed": 7582001,
                "arm": arm,
                "model_id": HISTORICAL_MODEL_ID,
                "request_sha256": "sha256:" + "1" * 64,
                "transition_sha256": "sha256:" + "2" * 64,
                "verifier_sha256": "sha256:" + "3" * 64,
                "plan_sha256": "sha256:" + "4" * 64,
                "action_sha256": "sha256:" + "5" * 64,
                "prompt_transition_ids": ["p0", "p1"],
                "heldout_transition_ids": ["h0", "h1"],
                "raised_transition_ids": ["h1"],
                "heldout_denominator": 2,
                "heldout_attempted": 2,
                "accepted": accepted,
                "guard_registered": arm == "integrity_guard",
                "response_cap_hit": False,
                "generation_attempted": 1,
                "generation_completed": 1,
                "completion_tokens": 20,
                "cost_usd": 0.1,
                "frame_motion_count": 2,
                "executed_plan_progress_count": executed,
                "executed_plan_action_count": executed,
                "supervisor_redirects": [],
                "censored": False,
                "provenance": "live_agent_self_discovery",
            }
        )
    return rows


def run_private_mutations() -> list[Json]:
    """Confirm every required corruption is rejected by the same reducer."""

    output: list[Json] = []
    for mutation in (
        "dropped_exception",
        "duplicated_heldout",
        "wrong_model_id",
        "unexecuted_plan",
        "capped_response",
    ):
        rows = _mutation_fixture()
        mutate_rows(rows, mutation)
        error: str | None = None
        try:
            reduce_episode_rows(rows)
        except ValueError as exc:
            error = str(exc)
        output.append(
            {
                "mutation": mutation,
                "passed": error is not None,
                "observed_error": error,
                "mutated_rows_published": False,
            }
        )
    return output


def reduce_supervisor(outcomes: Sequence[Mapping[str, Any]]) -> Json:
    """Support refinement only when eligible redirects separate outcomes."""

    eligible = [row for row in outcomes if row.get("eligible") is True]
    useful = [row for row in eligible if row.get("useful") is True]
    useless = [row for row in eligible if row.get("useful") is False]
    supported = bool(useful and useless)
    return {
        "eligible_firings": len(eligible),
        "useful_outcomes": len(useful),
        "useless_outcomes": len(useless),
        "games": sorted({str(row.get("game")) for row in eligible}),
        "supervisor_refinement_supported": supported,
        "arm_change": None,
        "reason": "nontrivial_useful_useless_contrast" if supported else "no_supported_contrast",
    }


def b2_gate_fit_decision(
    gate_opportunities: int,
    real_attempts: int,
    cross_game_support: Mapping[str, set[str]],
) -> bool:
    """Apply both sample floors and require useful and useless game support."""

    outcomes = {value for values in cross_game_support.values() for value in values}
    return bool(
        gate_opportunities >= 1000
        and real_attempts >= 100
        and len(cross_game_support) >= 2
        and {"useful", "useless"} <= outcomes
    )


def field_principles() -> dict[str, str]:
    """Explain the failure that each required terminal field prevents."""

    return {
        "honest_verdict": "Use a complete terminal prefix; completion does not establish benefit.",
        "verdict_class": "Exactly one closed class distinguishes blocked work from a valid null.",
        "flagged_adversarial": "Persist the exact terminal verifier outcome; flagged evidence cannot open readiness.",
        "gate_check_summary": "Every block names the failed check and both operands.",
        "acceptance_gate_results": "Validity, readiness, and benefit stay separate so a valid null remains usable.",
        "rows": "Every comparison unit and arm retains raw operands, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Actual aggregation stays distinct from planned live generation.",
        "MODEL_SPECS": "No current LLM task means the current model roster is empty.",
        "invocation_counts": "Current loads, forwards, generations, and tokens remain independently zero.",
        "duration_s": "Monotonic current work excludes inherited time and artificial sleeps.",
        "source_artifact_hashes": "Conclusions bind to exact bytes and distinguish absent producers from pre-gate records.",
        "validation_receipts": "Each check binds command, worktree, exit code, and log hash.",
        "field_principles": "Each required field states why it exists.",
        "verifier_is_oracle": "Oracle or label access cannot support an oracle-distinct positive claim.",
        "arc_claims_qualified_score": "Only complete independently reduced live evidence qualifies ARC claims.",
        "panel_dispositions": "One blocked panel cannot erase or decide the other panel.",
        "support_comparison_rows": "Paired source and denominator evidence remains available for replay.",
        "supervisor_refinement_supported": "A supervisor change needs eligible outcomes with a real contrast.",
        "solve_provenance": "Only authenticated runtime attempts can receive live self-discovery credit.",
        "b2_gate_fit_allowed": "B2 fitting needs both floors and cross-game useful and useless support.",
    }


def _gate(check: str, category: str, expected: Any, observed: Any) -> Json:
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "eq",
        "passed": observed == expected,
        "principle": "A failed branch cannot be hidden by another branch.",
    }


def _blocked_summary(checks: Sequence[Mapping[str, Any]]) -> Json:
    failed = [row for row in checks if row.get("passed") is not True]
    if not failed:
        raise ValueError("blocked_artifact_requires_failed_check")
    fields = ("check", "upstream", "path", "field", "op", "expected", "observed")
    return {
        "passed": False,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": {field: deepcopy(failed[0].get(field)) for field in fields},
    }


def _base_artifact(duration_s: float) -> Json:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "historical_model_identity": HISTORICAL_MODEL_ID,
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "historical_planned_panel_substrate_class": "model_full_generation",
        "duration_s": float(duration_s),
        "field_principles": field_principles(),
        "verifier_is_oracle": False,
        "official_score_claimed": False,
        "submitted_externally": False,
    }


def _missing_rows(protocol: Mapping[str, Any], blocked_panels: set[str]) -> list[Json]:
    output: list[Json] = []
    for registered in protocol_rows(protocol):
        panel_key = f"panel_{str(registered['panel']).lower()}"
        if panel_key not in blocked_panels:
            continue
        output.append(
            {
                "row_kind": "registered_missing_episode_arm",
                "unit_id": f"{registered['game']}:{registered['seed']}",
                **registered,
                "raw_numerator": None,
                "raw_denominator": 1,
                "metric_direction": "higher_completed_live_episode_is_better",
                "censored": True,
                "censoring_reason": "producer_absent_or_invalid",
                "attempted_calls": 0,
                "completed_calls": 0,
                "cap_hit_count": 0,
                "provenance": "frozen_exp7580_protocol_missing_producer",
                "solve_credit_allowed": False,
            }
        )
    return output


def _panel_dispositions(checks: Sequence[Mapping[str, Any]]) -> Json:
    output: Json = {}
    for panel, producer in PRODUCERS.items():
        label = f"exp758{2 if panel == 'panel_a' else 3}"
        failures = [
            dict(row)
            for row in checks
            if row.get("upstream") in {label, panel} and row.get("passed") is not True
        ]
        output[panel] = {
            "status": "blocked" if failures else "available",
            "producer_path": producer.as_posix(),
            "failure_count": len(failures),
            "first_failure": failures[0] if failures else None,
            "attempted_calls": 0,
            "completed_calls": 0,
            "cap_hit_count": 0,
            "benefit": "not_measured" if failures else "pending_reduction",
        }
    return output


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable audit content while excluding clocks and self-reference."""

    excluded = {"reproducibility_checksum", "duration_s", "phase_spans"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def build_blocked_artifact(
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    producers: Mapping[str, Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    source_hashes: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    run_date: str,
) -> Json:
    """Publish complete blocked panels while retaining the frozen roster."""

    del producers
    dispositions = _panel_dispositions(checks)
    blocked = {panel for panel, row in dispositions.items() if row["status"] == "blocked"}
    rows = _missing_rows(protocol, blocked)
    gates = [
        _gate("external_producers_valid", "validity", True, False),
        _gate("live_panel_ready", "readiness", True, False),
        _gate("live_induction_benefit", "benefit", True, "not_measured"),
    ]
    artifact = {
        **_base_artifact(duration_s),
        "worktree_root": str(root.resolve()),
        "run_date": run_date,
        "honest_verdict": "complete_blocked_live_panel_producers_missing_or_invalid",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "preconditions_checked": [dict(row) for row in checks],
        "gate_check_summary": _blocked_summary(checks),
        "acceptance_gate_results": gates,
        "rows": rows,
        "support_comparison_rows": deepcopy(rows),
        "panel_dispositions": dispositions,
        "source_artifact_hashes": [dict(row) for row in source_hashes],
        "validation_receipts": [dict(row) for row in receipts],
        "mutation_rows": run_private_mutations(),
        "arc_claims_qualified_score": 0,
        "supervisor_refinement_supported": False,
        "supervisor_reduction": reduce_supervisor([]),
        "b2_gate_fit_allowed": False,
        "b2_gate_evidence": {"gate_opportunities": 0, "real_attempts": 0},
        "solve_provenance": "no_new_solve_credit",
        "known_public_reproduction_credit": 0,
        "analytical_positive_control_disposition": "not_measured_not_circular_positive",
        "pooled_independence_claim": False,
        "game_cluster_count": 6,
        "prior_verdict_disposition": "narrow_pre_gate_transport_block_preserved_not_scientific_hypothesis_retirement",
        "phase_spans": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _required_receipt_names() -> set[str]:
    return {*AFFECTED_CHECK_NAMES, *CAPABILITY_CHECK_NAMES, *TERMINAL_CHECK_NAMES}


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        grouped[str(receipt.get("name"))].append(receipt)
    return all(
        len(grouped[name]) == 1
        and grouped[name][0].get("exit_code") == 0
        and grouped[name][0].get("timed_out") is not True
        and grouped[name][0].get("passed") is True
        for name in _required_receipt_names()
    )


def fixture_validation_receipts(root: Path) -> list[Json]:
    """Create deterministic receipt shapes for schema-only unit tests."""

    return [
        {
            "name": name,
            "command": f"fixture {name}",
            "command_argv": ["fixture", name],
            "cwd": str(root.resolve()),
            "worktree": str(root.resolve()),
            "exit_code": 0,
            "timed_out": False,
            "passed": True,
            "log_sha256": canonical_hash(name),
        }
        for name in sorted(_required_receipt_names())
    ]


def build_fixture_artifact(root: Path) -> Json:
    checks = [
        check_row(
            "producer_exists",
            "exp7582",
            PRODUCERS["panel_a"].as_posix(),
            "exists",
            True,
            False,
        ),
        check_row(
            "producer_exists",
            "exp7583",
            PRODUCERS["panel_b"].as_posix(),
            "exists",
            True,
            False,
        ),
    ]
    return build_blocked_artifact(
        root,
        checks,
        {},
        protocol=fixture_protocol(),
        source_hashes=[],
        receipts=fixture_validation_receipts(root),
        duration_s=0.1,
        run_date="20260924",
    )


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> Json:
    """Reject schema, custody, claim, invocation, row, or receipt drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE:
        errors.append("milestone_mismatch")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_calls_nonzero")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_mismatch")
    if value.get("official_score_claimed") is not False:
        errors.append("official_score_claim_forbidden")
    if value.get("solve_provenance") != "no_new_solve_credit":
        errors.append("solve_credit_invalid")
    if value.get("arc_claims_qualified_score") not in (0, 1):
        errors.append("arc_qualification_invalid")
    if set(REQUIRED_PRINCIPLE_FIELDS) - set(value.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if set(value.get("panel_dispositions") or {}) != {"panel_a", "panel_b"}:
        errors.append("panel_dispositions_incomplete")
    first = (value.get("gate_check_summary") or {}).get("first_failure")
    blocked_fields = {"check", "upstream", "path", "field", "op", "expected", "observed"}
    if value.get("verdict_class") == "blocked" and (
        not isinstance(first, Mapping) or set(first) != blocked_fields
    ):
        errors.append("blocked_gate_summary_invalid")
    for row in value.get("rows") or []:
        if any(
            field not in row
            for field in (
                "unit_id",
                "panel",
                "game",
                "seed",
                "arm",
                "raw_numerator",
                "raw_denominator",
                "metric_direction",
                "censored",
                "provenance",
            )
        ):
            errors.append("row_schema_invalid")
            break
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts_failed")
    mutations = value.get("mutation_rows") or []
    expected_mutations = {
        "dropped_exception",
        "duplicated_heldout",
        "wrong_model_id",
        "unexecuted_plan",
        "capped_response",
    }
    if {row.get("mutation") for row in mutations} != expected_mutations or not all(
        row.get("passed") is True for row in mutations
    ):
        errors.append("mutation_panel_failed")
    for receipt in value.get("source_artifact_hashes") or []:
        if receipt.get("present") is True:
            try:
                authenticate_source_receipt(receipt, root)
            except ValueError as exc:
                errors.append(str(exc))
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("checksum_mismatch")
    if errors:
        raise ValueError(";".join(dict.fromkeys(errors)))
    return {"valid": True, "errors": []}


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> Json:
    value = load_json(path)
    if not value:
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root)


def build_validation_commands(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Freeze focused tests, changed coverage, style, type, and spec checks."""

    private.mkdir(parents=True, exist_ok=True)
    (private / "focused").mkdir(parents=True, exist_ok=True)
    (private / "coverage").mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(),),
        (MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=private,
        coverage_file=private / ".coverage.exp7584",
    )
    return [
        validation_scope.CommandSpec(
            command.name,
            command.argv,
            command.scope,
            min(command.timeout_s, 600.0),
        )
        for command in commands
    ]


def capability_commands(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Declare E2E-009 through E2E-013 and the real LLM-off smoke."""

    private.mkdir(parents=True, exist_ok=True)
    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_010": ("tests/python/test_arc_tool_grammar_transport.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_012": (
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
            "tests/python/test_arc_decision_telemetry.py",
        ),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands: list[validation_scope.CommandSpec] = []
    for name, paths in targets.items():
        basetemp = private / name
        basetemp.parent.mkdir(parents=True, exist_ok=True)
        commands.append(
            validation_scope.CommandSpec(
                name,
                (pytest, *common, f"--basetemp={basetemp}", *paths, "-q"),
                f"{name.replace('_', '-').upper()} ARC CPU contract",
                600.0,
            )
        )
    smoke_dir = private / "foreign-cwd"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    commands.append(
        validation_scope.CommandSpec(
            "llm_off_environment_smoke",
            (
                "/usr/bin/env",
                "-C",
                str(smoke_dir),
                f"PYTHONPATH={root / 'python'}:{root}",
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                str(root / "scripts/arc_loop_solve.py"),
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(smoke_dir / "r11l-smoke.json"),
            ),
            "private LLM-off real E3 episode from a foreign cwd",
            300.0,
        )
    )
    return commands


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    """Declare the entrypoint, cold replay, reducer, and two strict readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    common = ("--root", str(root.resolve()), "--date", "20260924")
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "declared read-only entrypoint",
            300.0,
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "fresh-process cold replay",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent per-unit reduction",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            300.0,
        ),
    ]


def independent_reduction(path: Path, *, root: Path = REPO_ROOT) -> Json:
    """Recount every registered row without trusting panel headlines."""

    value = load_json(path)
    validate_artifact(value, root=root)
    rows = value.get("rows") or []
    keys = {(row.get("panel"), row.get("game"), row.get("seed"), row.get("arm")) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("support_row_duplicate")
    expected = {
        (panel, game, seed, arm)
        for panel, games in PANEL_GAMES.items()
        for game in games
        for seed in SEEDS
        for arm in ARMS
    }
    if value.get("verdict_class") == "blocked" and keys != expected:
        raise ValueError("blocked_roster_mismatch")
    if any(row.get("raw_denominator") != 1 for row in rows):
        raise ValueError("support_denominator_mismatch")
    return {
        "valid": True,
        "row_count": len(rows),
        "panel_counts": {
            panel: sum(row.get("panel") == panel for row in rows) for panel in PANEL_GAMES
        },
        "censored_count": sum(row.get("censored") is True for row in rows),
    }


def _load_protocol(root: Path, checks: list[Json], source_hashes: list[Json]) -> Json:
    producer_path = root / PROTOCOL_PRODUCER_PATH
    exists = producer_path.is_file()
    checks.append(
        check_row(
            "protocol_producer_exists",
            "exp7580",
            PROTOCOL_PRODUCER_PATH.as_posix(),
            "exists",
            True,
            exists,
        )
    )
    if not exists:
        return {}
    producer = load_json(producer_path)
    source_hashes.append(source_receipt(producer_path, root, "exp7580", "producer"))
    relative = Path(str(producer.get("live_panel_protocol_path") or ""))
    protocol_path = root / relative
    protocol_exists = protocol_path.is_file()
    checks.append(
        check_row(
            "protocol_sidecar_exists",
            "exp7580",
            relative.as_posix(),
            "exists",
            True,
            protocol_exists,
        )
    )
    if not protocol_exists:
        return {}
    observed_hash = sha256_file(protocol_path)
    expected_hash = (producer.get("source_artifact_hashes") or {}).get(relative.as_posix())
    checks.append(
        check_row(
            "protocol_sidecar_hash",
            "exp7580",
            relative.as_posix(),
            "sha256",
            expected_hash,
            observed_hash,
        )
    )
    source_hashes.append(source_receipt(protocol_path, root, "exp7580.protocol", "raw_sidecar"))
    protocol = load_json(protocol_path)
    checks.append(
        check_row(
            "protocol_roster",
            "exp7580",
            relative.as_posix(),
            "registered_rows",
            24,
            len(protocol_rows(protocol)),
        )
    )
    return protocol


def _collect_source_hashes(root: Path, checks: list[Json]) -> tuple[list[Json], Json]:
    hashes: list[Json] = []
    protocol = _load_protocol(root, checks, hashes)
    for panel, relative in PRODUCERS.items():
        path = root / relative
        if path.is_file():
            hashes.append(source_receipt(path, root, panel, "producer"))
        else:
            hashes.append(absent_source_receipt(relative, panel))
    for relative, upstream, stage in (
        (CANARY_PATH, "exp7581", "pre_gate_upstream"),
        (PRE_GATE_PANEL_A_PATH, "exp7582", "pre_gate_diagnostic_not_producer"),
        (MODULE_PATH, "exp7584", "implementation"),
        (WRAPPER_PATH, "exp7584", "entrypoint"),
        (TEST_PATH, "exp7584", "tests"),
        (SPEC_PATH, "exp7584", "spec"),
    ):
        path = root / relative
        if path.is_file():
            hashes.append(source_receipt(path, root, upstream, stage))
        else:
            hashes.append(absent_source_receipt(relative, upstream))
    return hashes, protocol


def _manifest(root: Path) -> tuple[Path, Json]:
    path = root / RAW_DIR / "affected_validation_manifest.json"
    value = {
        "experiment_id": EXPERIMENT_ID,
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
    }
    atomic_json(path, value)
    return path, value


def _pending_terminal_receipts(root: Path) -> list[Json]:
    return [
        {
            "name": name,
            "command": f"pending {name}",
            "command_argv": ["pending", name],
            "cwd": str(root.resolve()),
            "worktree": str(root.resolve()),
            "exit_code": 0,
            "timed_out": False,
            "passed": True,
            "log_sha256": "sha256:" + "0" * 64,
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def _enrich_receipts(receipts: Sequence[Mapping[str, Any]], root: Path) -> list[Json]:
    output: list[Json] = []
    for receipt in receipts:
        row = dict(receipt)
        row["cwd"] = str(root.resolve())
        row["worktree"] = str(root.resolve())
        output.append(row)
    return output


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush every phase and long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7584] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def run_experiment(root: Path, run_date: str) -> Json:  # pragma: no cover - capability E2E
    """Authenticate, validate, independently replay, and atomically publish."""

    root = root.resolve()
    if run_date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    started = time.monotonic()
    raw_root = root / RAW_DIR
    destination = root / RESULT_PATH
    candidate = raw_root / "terminal_candidate.json"

    progress(started, "preconditions", "start", root=root)
    checks, producers = collect_preconditions(root)
    source_hashes, protocol = _collect_source_hashes(root, checks)
    manifest_path, _ = _manifest(root)
    source_hashes.append(source_receipt(manifest_path, root, "exp7584", "validation_manifest"))
    progress(
        started,
        "preconditions",
        "complete",
        completed_units=len(checks),
        failed=sum(row.get("passed") is not True for row in checks),
    )

    private = Path(tempfile.mkdtemp(prefix="carnot-exp7584-", dir="/tmp"))
    scoped = build_validation_commands(root, private / "scoped")
    progress(started, "scoped_validation", "before_subprocesses", planned_units=len(scoped))
    scoped_receipts = validation_scope.run_commands(
        root,
        scoped,
        log_dir=raw_root / "validation" / "scoped",
        heartbeat_s=60.0,
    )
    scoped_receipts = _enrich_receipts(scoped_receipts, root)
    scoped_pass = validation_scope.reduce_required_checks(scoped_receipts)["required_checks_passed"]
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed_units=len(scoped_receipts),
        passed=scoped_pass,
    )
    if not scoped_pass:
        raise RuntimeError("required_scoped_validation_failed")

    capability = capability_commands(root, private / "capability")
    progress(started, "capability_e2e", "before_subprocesses", planned_units=len(capability))
    capability_receipts = validation_scope.run_commands(
        root,
        capability,
        log_dir=raw_root / "validation" / "capability",
        heartbeat_s=60.0,
    )
    capability_receipts = _enrich_receipts(capability_receipts, root)
    capability_pass = all(row.get("passed") is True for row in capability_receipts)
    progress(
        started,
        "capability_e2e",
        "after_subprocesses",
        completed_units=len(capability_receipts),
        passed=capability_pass,
    )
    if not capability_pass:
        raise RuntimeError("required_capability_e2e_failed")

    provisional = build_blocked_artifact(
        root,
        checks,
        producers,
        protocol=protocol,
        source_hashes=source_hashes,
        receipts=[*scoped_receipts, *capability_receipts, *_pending_terminal_receipts(root)],
        duration_s=time.monotonic() - started,
        run_date=run_date,
    )
    atomic_json(candidate, provisional)
    first_terminal = terminal_commands(candidate, root)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned_units=len(first_terminal),
    )
    terminal_receipts = validation_scope.run_commands(
        root,
        first_terminal,
        log_dir=raw_root / "validation" / "terminal_provisional",
        heartbeat_s=60.0,
    )
    terminal_receipts = _enrich_receipts(terminal_receipts, root)
    terminal_pass = all(row.get("passed") is True for row in terminal_receipts)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=terminal_pass,
    )
    if not terminal_pass:
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_blocked_artifact(
        root,
        checks,
        producers,
        protocol=protocol,
        source_hashes=source_hashes,
        receipts=[*scoped_receipts, *capability_receipts, *terminal_receipts],
        duration_s=time.monotonic() - started,
        run_date=run_date,
    )
    validate_artifact(final, root=root)
    atomic_json(candidate, final)
    exact = terminal_commands(candidate, root)
    progress(started, "exact_candidate", "before_subprocesses", planned_units=len(exact))
    exact_receipts = validation_scope.run_commands(
        root,
        exact,
        log_dir=raw_root / "validation" / "terminal_exact",
        heartbeat_s=60.0,
    )
    exact_pass = all(row.get("passed") is True for row in exact_receipts)
    progress(
        started,
        "exact_candidate",
        "after_subprocesses",
        completed_units=len(exact_receipts),
        passed=exact_pass,
    )
    if not exact_pass:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(started, "publish", "before_atomic_write")
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(candidate):
        raise RuntimeError("published_bytes_differ_from_validated_candidate")
    progress(
        started,
        "publish",
        "after_atomic_write",
        bytes=destination.stat().st_size,
        verdict=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260924")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    arguments = parser.parse_args(argv)
    arguments.root = arguments.root.resolve()
    if arguments.date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    arguments = parse_args(argv)
    if arguments.validate is not None:
        result = validate_artifact(load_json(arguments.validate), root=arguments.root)
        print(json.dumps({"mode": "validate", **result}, sort_keys=True), flush=True)
        return 0
    if arguments.cold_replay is not None:
        result = cold_replay(arguments.cold_replay, root=arguments.root)
        print(json.dumps({"mode": "cold_replay", **result}, sort_keys=True), flush=True)
        return 0
    if arguments.independent_reduce is not None:
        result = independent_reduction(arguments.independent_reduce, root=arguments.root)
        print(json.dumps({"mode": "independent_reduction", **result}, sort_keys=True), flush=True)
        return 0
    run_experiment(arguments.root, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
