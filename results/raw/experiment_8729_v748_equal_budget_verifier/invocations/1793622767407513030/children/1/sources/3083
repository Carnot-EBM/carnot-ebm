"""Reduce inherited ARC supervisor outcomes without running new gameplay.

REQ-ARC-7625 keeps shadow proposals separate from applied redirects. This
module reads authenticated public-run receipts, deduplicates copied episodes,
and reports observational support. It never loads a model or changes policy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.agentic.arc_supervisor_refinement import classify_receipt
from carnot.experiment_6844_supervisor_action_outcome_credit_audit import sha256_json
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

Json = dict[str, Any]
RUN_DATE = "20260924"
EXPERIMENT_ID = 7625
MILESTONE = "2026.09.665"
REQUIREMENT_ID = "REQ-ARC-7625"
SCHEMA = "carnot.experiment_7625_v665_arc_supervisor_transfer.v1"
RESULT_REL = Path("results/experiment_7625_v665_arc_supervisor_transfer.json")
MODULE_REL = Path("python/carnot/experiment_7625_v665_arc_supervisor_transfer.py")
TEST_REL = Path("tests/python/test_experiment_7625_v665_arc_supervisor_transfer.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7625_v665_arc_supervisor_transfer.py")
SPEC_REL = Path("openspec/capabilities/arc-agi/spec.md")
REGISTRY_REL = Path("ops/arc_solve_registry.yaml")
NOTE_REL = Path("docs/research-notes/v665-arc-supervisor-transfer.md")
MODEL_SPECS: list[Json] = []
GAMES = ("dc22", "ft09", "g50t", "sb26", "sp80", "su15")
MIN_UNCENSORED_FIRINGS = 20
MIN_GAMES = 3


@dataclass(frozen=True)
class ProducerSpec:
    """Name one producer artifact and its raw episode directory."""

    name: str
    result_path: Path
    episode_dir: Path


DEFAULT_PRODUCERS = (
    ProducerSpec(
        "experiment_7611_v664",
        Path("results/experiment_7611_v664_arc_matched_support.json"),
        Path("results/raw/experiment_7611_v664_arc_matched_support/episodes"),
    ),
    ProducerSpec(
        "experiment_7597_v663",
        Path("results/experiment_7597_v663_arc_history_generalization.json"),
        Path("results/raw/experiment_7597_v663_arc_history_generalization/episodes"),
    ),
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print one flushed phase boundary with real monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7625] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def check_row(
    check: str,
    *,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
) -> Json:
    """Build a complete gate row so a block always names exact operands."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def logical_episode_hash(payload: Mapping[str, Any]) -> str:
    """Hash behavior while excluding seed labels and copied receipt metadata."""

    material = {
        key: deepcopy(payload.get(key))
        for key in (
            "game",
            "policy",
            "action_limit",
            "induction_disabled",
            "adapter_withheld",
            "stored_solutions_withheld",
            "game_source_read",
            "hidden_state_read",
            "offline_ground_truth_bfs",
            "live_llm_invoked",
            "steps",
            "trajectory_supervisor",
            "termination",
        )
    }
    return sha256_json(material)


def _authenticated_hashes(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    hashes = payload.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return {}
    nested = hashes.get("authenticated_sources")
    return nested if isinstance(nested, Mapping) else hashes


def select_authenticated_receipts(
    root: Path,
    producers: Sequence[ProducerSpec] = DEFAULT_PRODUCERS,
) -> Json:
    """Select preferred receipts and collapse identical logical episodes."""

    resolved_root = Path(root).resolve()
    selected: dict[str, Json] = {}
    errors: list[str] = []
    source_hashes: dict[str, str] = {}
    observed = 0
    selection_started = time.monotonic()
    last_heartbeat = selection_started
    for producer in producers:
        result_path = resolved_root / producer.result_path
        if not result_path.is_file():
            errors.append(f"producer_missing:{producer.result_path.as_posix()}")
            continue
        producer_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_label = _relative(result_path, resolved_root)
        source_hashes[result_label] = sha256_file(result_path)
        if producer_payload.get("flagged_adversarial") is True:
            errors.append(f"producer_flagged:{result_label}")
            continue
        authenticated = _authenticated_hashes(producer_payload)
        episode_dir = resolved_root / producer.episode_dir
        if not episode_dir.is_dir():
            errors.append(f"episode_directory_missing:{producer.episode_dir.as_posix()}")
            continue
        for path in sorted(episode_dir.glob("*.json")):
            observed += 1
            now = time.monotonic()
            if now - last_heartbeat >= 60.0:  # pragma: no cover - large production inputs.
                print(
                    "[exp7625] phase=receipt_selection event=heartbeat "
                    f"completed_units={observed - 1} "
                    f"elapsed_s={now - selection_started:.3f} pending_operation=hash_and_parse",
                    flush=True,
                )
                last_heartbeat = now
            label = _relative(path, resolved_root)
            actual_hash = sha256_file(path)
            if authenticated.get(label) != actual_hash:
                errors.append(f"source_hash_mismatch:{label}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("policy") not in {"E3AgentPolicy", "make_carnot_agent"}:
                errors.append(f"policy_not_live_path:{label}")
                continue
            if payload.get("adapter_withheld") is not True:
                errors.append(f"adapter_not_withheld:{label}")
                continue
            source_hashes[label] = actual_hash
            identity = logical_episode_hash(payload)
            if identity in selected:
                selected[identity]["duplicate_sources"].append(label)
                continue
            selected[identity] = {
                "logical_episode_sha256": identity,
                "payload": payload,
                "path": label,
                "sha256": actual_hash,
                "producer": producer.name,
                "duplicate_sources": [],
            }
    receipts = sorted(selected.values(), key=lambda row: (row["payload"].get("game"), row["path"]))
    return {
        "receipts": receipts,
        "schema_errors": list(dict.fromkeys(errors)),
        "source_hashes": source_hashes,
        "observed_receipts": observed,
        "deduplicated_receipts": len(receipts),
        "duplicate_receipts": observed - len(receipts),
    }


def _schema_fields_present(rows: Any, fields: set[str]) -> bool:
    return isinstance(rows, list) and all(
        isinstance(row, Mapping) and fields <= set(row) for row in rows
    )


def _episode_censored(payload: Mapping[str, Any], redirect: Mapping[str, Any]) -> bool:
    if redirect.get("resolved_by_levelup") is True:
        return False
    termination = payload.get("termination")
    reason = termination.get("reason") if isinstance(termination, Mapping) else None
    return str(reason) in {"action_limit", "time_limit", "timeout", "collection_cap"}


def _actual_redirect_row(
    payload: Mapping[str, Any], receipt_record: Mapping[str, Any], redirect: Mapping[str, Any]
) -> Json:
    resolved = redirect.get("resolved_by_levelup") is True
    censored = _episode_censored(payload, redirect)
    row: Json = {
        "unit": str(payload.get("game")),
        "game": str(payload.get("game")),
        "episode_id": payload.get("episode_id"),
        "seed": payload.get("seed"),
        "arm": str(redirect.get("arm")),
        "action_index": redirect.get("action_index"),
        "level": redirect.get("level"),
        "stretch_level": redirect.get("stretch_level"),
        "resolved_by_levelup": resolved,
        "actions_to_levelup": redirect.get("actions_to_levelup"),
        "co_credited_count": redirect.get("co_credited_count"),
        "numerator": int(resolved and not censored),
        "denominator": int(not censored),
        "rate": 1.0 if resolved and not censored else (0.0 if not censored else None),
        "direction": "helped" if resolved else ("censored" if censored else "not_helped"),
        "censoring": {
            "censored": censored,
            "reason": "episode_horizon" if censored else None,
        },
        "raw_provenance": {
            "path": receipt_record.get("path"),
            "sha256": receipt_record.get("sha256"),
            "policy": payload.get("policy"),
            "receipt_mode": "applied",
        },
    }
    row["row_sha256"] = sha256_json(row)
    return row


def _join_matches_outcomes(
    receipt: Mapping[str, Any], redirects: Sequence[Mapping[str, Any]]
) -> bool:
    expected: dict[str, dict[str, int]] = defaultdict(lambda: {"fired": 0, "helped": 0})
    for redirect in redirects:
        arm = str(redirect.get("arm"))
        expected[arm]["fired"] += 1
        expected[arm]["helped"] += int(redirect.get("resolved_by_levelup") is True)
    outcomes = receipt.get("arm_outcomes")
    if not isinstance(outcomes, Mapping):
        return False
    for arm in set(expected) | {str(name) for name in outcomes}:
        row = outcomes.get(arm, {})
        if not isinstance(row, Mapping):
            return False
        observed = {"fired": int(row.get("fired", 0)), "helped": int(row.get("helped", 0))}
        if observed != expected[arm]:
            return False
    return True


def _arm_statistics(actual_rows: Sequence[Mapping[str, Any]]) -> Json:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in actual_rows:
        grouped[str(row["arm"])].append(row)
    output: Json = {}
    for arm, arm_rows in sorted(grouped.items()):
        uncensored = [row for row in arm_rows if row["censoring"]["censored"] is False]
        by_game: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in uncensored:
            by_game[str(row["game"])].append(row)
        helped = sum(int(row["numerator"]) for row in uncensored)
        total = len(uncensored)
        per_game = {
            game: sum(int(row["numerator"]) for row in rows) / len(rows)
            for game, rows in sorted(by_game.items())
        }
        loo: list[Json] = []
        for omitted in sorted(by_game):
            kept = [row for row in uncensored if row["game"] != omitted]
            loo.append(
                {
                    "omitted_game": omitted,
                    "helped": sum(int(row["numerator"]) for row in kept),
                    "firings": len(kept),
                    "help_rate": (
                        sum(int(row["numerator"]) for row in kept) / len(kept) if kept else None
                    ),
                }
            )
        eligible = total >= MIN_UNCENSORED_FIRINGS and len(by_game) >= MIN_GAMES
        loo_rates = [row["help_rate"] for row in loo if row["help_rate"] is not None]
        stable = bool(eligible and loo_rates) and (
            all(rate >= 0.5 for rate in loo_rates) or all(rate < 0.5 for rate in loo_rates)
        )
        output[arm] = {
            "actual_firings": len(arm_rows),
            "uncensored_actual_firings": total,
            "censored_actual_firings": len(arm_rows) - total,
            "helped": helped,
            "game_count": len(by_game),
            "help_rate": helped / total if total else None,
            "per_game_rates": per_game,
            "leave_one_game_out": loo,
            "leave_one_game_out_stable": stable,
            "never_helped_binomial_upper_95": (
                1.0 - math.pow(0.05, 1.0 / total) if eligible and total and helped == 0 else None
            ),
            "eligible": eligible,
            "interpretation": "observational_association_not_causal_treatment_effect",
        }
    return output


def _new_arm_specification(receipts: Sequence[Mapping[str, Any]]) -> Json:
    for record in receipts:
        payload = record["payload"]
        if not isinstance(payload, Mapping):
            continue
        receipt = payload.get("trajectory_supervisor") or {}
        if classify_receipt({"trajectory_supervisor": receipt}) != "applied":
            continue
        enabled = {str(arm) for arm in receipt.get("arms_enabled") or []}
        used = {str(arm) for arm in receipt.get("arms_used") or []}
        if enabled and enabled <= used and int(receipt.get("stagnations_unredirected") or 0) > 0:
            windows = [
                row for row in receipt.get("unredirected_windows") or [] if isinstance(row, Mapping)
            ]
            signals = sorted(
                key
                for key in (
                    "attempt_cap_reached",
                    "diversity_active",
                    "evidence_floor_met",
                    "goal_bias_installed",
                )
                if any(row.get(key) is True for row in windows)
            )
            return {
                "status": "specified_from_receipts",
                "trigger": "all_enabled_arms_used_and_stagnation_persisted",
                "required_capability": "a game-blind arm that changes a still-unmodified search constraint",
                "grounding_signals": signals,
                "game_specific_route": False,
                "model_generated": False,
            }
    return {
        "status": "not_supported",
        "reason": "no_applied_all-arms-exhausted_stagnation_receipt",
        "game_specific_route": False,
        "model_generated": False,
    }


def reduce_receipts(receipts: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce authenticated ledgers while preserving receipt mode boundaries."""

    errors: list[str] = []
    actual_rows: list[Json] = []
    episode_rows: list[Json] = []
    proposed_total = would_total = applied_total = 0
    for record in receipts:
        payload = record.get("payload")
        path = str(record.get("path"))
        if not isinstance(payload, Mapping):
            errors.append(f"episode_payload_invalid:{path}")
            continue
        receipt = payload.get("trajectory_supervisor")
        kind = classify_receipt({"trajectory_supervisor": receipt})
        if not isinstance(receipt, Mapping):
            errors.append(f"missing_outcome_schema:{path}")
            continue
        actual: list[Mapping[str, Any]] = []
        would_have: list[Mapping[str, Any]] = []
        outcomes: Mapping[str, Any] = {}
        if kind == "applied":
            raw_redirects = receipt.get("redirects")
            if not _schema_fields_present(
                raw_redirects,
                {"arm", "action_index", "level", "resolved_by_levelup", "actions_to_levelup"},
            ) or not isinstance(receipt.get("arm_outcomes"), Mapping):
                errors.append(f"missing_applied_outcome_schema:{path}")
                continue
            actual = list(raw_redirects)
            outcomes = receipt["arm_outcomes"]
            if not _join_matches_outcomes(receipt, actual):
                errors.append(f"arm_outcome_join_mismatch:{path}")
                continue
            proposed_total += len(actual)
            applied_total += len(actual)
            for redirect in actual:
                actual_rows.append(_actual_redirect_row(payload, record, redirect))
        elif kind == "shadow":
            raw_would = receipt.get("would_have_redirects")
            if not _schema_fields_present(
                raw_would,
                {
                    "arm",
                    "action_index",
                    "level",
                    "levelup_followed_without_redirect",
                    "actions_to_levelup_without_redirect",
                },
            ) or not isinstance(receipt.get("would_have_arm_outcomes"), Mapping):
                errors.append(f"missing_shadow_outcome_schema:{path}")
                continue
            would_have = list(raw_would)
            outcomes = receipt["would_have_arm_outcomes"]
            expected: dict[str, dict[str, int]] = defaultdict(lambda: {"fired": 0, "helped": 0})
            for redirect in would_have:
                arm = str(redirect["arm"])
                expected[arm]["fired"] += 1
                expected[arm]["helped"] += int(
                    redirect.get("levelup_followed_without_redirect") is True
                )
            joined = True
            for arm in set(expected) | {str(name) for name in outcomes}:
                outcome = outcomes.get(arm, {})
                if (
                    not isinstance(outcome, Mapping)
                    or {
                        "fired": int(outcome.get("fired", 0)),
                        "helped": int(outcome.get("helped", 0)),
                    }
                    != expected[arm]
                ):
                    joined = False
                    break
            if not joined:
                errors.append(f"would_have_outcome_join_mismatch:{path}")
                continue
            proposed_total += len(would_have)
            would_total += len(would_have)
        else:
            errors.append(f"missing_outcome_schema:{path}")
            continue

        game = str(payload.get("game"))
        arms = sorted(
            {str(arm) for arm in receipt.get("arms_enabled") or []}
            | {str(arm) for arm in outcomes}
            | {str(row.get("arm")) for row in (*actual, *would_have)}
        )
        per_arm: list[Json] = []
        for arm in arms:
            actual_for_arm = [
                row
                for row in actual_rows
                if row["game"] == game
                and row["arm"] == arm
                and row["raw_provenance"]["path"] == path
            ]
            proposed = sum(int(row.get("arm") == arm) for row in (*actual, *would_have))
            would_count = sum(int(row.get("arm") == arm) for row in would_have)
            uncensored = sum(int(row["censoring"]["censored"] is False) for row in actual_for_arm)
            helped = sum(int(row["numerator"]) for row in actual_for_arm)
            row: Json = {
                "unit": game,
                "arm": arm,
                "absolute_metric": "actual_redirect_help_rate",
                "proposed_redirects": proposed,
                "would_have_redirects": would_count,
                "applied_redirects": len(actual_for_arm),
                "actual_firings": len(actual_for_arm),
                "helped": helped,
                "numerator": helped,
                "denominator": uncensored,
                "rate": helped / uncensored if uncensored else None,
                "seed": payload.get("seed"),
                "direction": "observational_help_association",
                "censoring": {
                    "actual_firings_censored": len(actual_for_arm) - uncensored,
                    "episode_end_reason": (payload.get("termination") or {}).get("reason")
                    if isinstance(payload.get("termination"), Mapping)
                    else None,
                },
                "raw_provenance": {
                    "path": path,
                    "sha256": record.get("sha256"),
                    "receipt_mode": kind,
                    "logical_episode_sha256": record.get("logical_episode_sha256"),
                },
            }
            row["operand_checksum"] = sha256_json(
                {
                    key: row[key]
                    for key in (
                        "unit",
                        "arm",
                        "numerator",
                        "denominator",
                        "seed",
                        "direction",
                        "censoring",
                    )
                }
            )
            per_arm.append(row)
        episode_rows.append(
            {
                "game": game,
                "episode_id": payload.get("episode_id"),
                "seed": payload.get("seed"),
                "receipt_mode": kind,
                "proposed_redirects": len(actual) + len(would_have),
                "would_have_redirects": len(would_have),
                "applied_redirects": len(actual),
                "actual_firings": len(actual),
                "helped": sum(int(row.get("resolved_by_levelup") is True) for row in actual),
                "stagnations_unredirected": int(receipt.get("stagnations_unredirected") or 0),
                "headroom": {
                    "actual_outcome_classes": sorted(
                        {
                            row["direction"]
                            for row in actual_rows
                            if row["game"] == game and row["raw_provenance"]["path"] == path
                        }
                    ),
                    "nonzero": len(
                        {
                            row["direction"]
                            for row in actual_rows
                            if row["game"] == game and row["raw_provenance"]["path"] == path
                        }
                    )
                    > 1,
                },
                "censoring": {
                    "actual_firings_censored": sum(
                        int(row["censoring"]["censored"] is True)
                        for row in actual_rows
                        if row["game"] == game and row["raw_provenance"]["path"] == path
                    )
                },
                "arms": per_arm,
                "raw_provenance": {"path": path, "sha256": record.get("sha256")},
            }
        )

    stats = _arm_statistics(actual_rows)
    rows = [row for episode in episode_rows for row in episode["arms"]]
    ready = int(bool(receipts) and not errors)
    return {
        "schema_errors": list(dict.fromkeys(errors)),
        "supervisor_outcome_ledger_ready_score": ready,
        "proposed_redirect_count": proposed_total,
        "would_have_redirect_count": would_total,
        "applied_redirect_count": applied_total,
        "actual_firing_count": len(actual_rows),
        "helped_count": sum(int(row["numerator"]) for row in actual_rows),
        "censored_firing_count": sum(
            int(row["censoring"]["censored"] is True) for row in actual_rows
        ),
        "actual_redirects": actual_rows,
        "rows": rows,
        "per_game_results": episode_rows,
        "arm_statistics": stats,
        "new_arm_specification": _new_arm_specification(receipts),
    }


FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "The schema lets readers reject incompatible reductions.",
    "experiment_id": "The identifier binds the result to REQ-ARC-7625.",
    "milestone": "The milestone fixes the governance context.",
    "run_date": "The date states when this aggregation executed.",
    "honest_verdict": "A complete_ prefix separates execution from benefit.",
    "verdict_class": "The closed class prevents partial or positive drift.",
    "flagged_adversarial": "Flagged evidence cannot open a downstream gate.",
    "gate_check_summary": "A block names the exact failed check and operands.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
    "rows": "Each game and arm keeps absolute operands and provenance.",
    "sample_size_budget": "Games are units; seeds and copied receipts do not multiply them.",
    "preconditions_checked": "Actual input checks prevent fabricated work.",
    "planned_inference_substrate_class": "The plan is explicit before execution.",
    "inference_substrate_class": "The actual aggregation class sets the correct duration rule.",
    "inference_substrate": "Historical runs are not current model inference.",
    "MODEL_SPECS": "No current model call requires an empty model list.",
    "historical_model_identity": "Inherited identities remain distinct from current calls.",
    "model_invoked": "False records that no load or generation was attempted.",
    "execution_venue": "The current host is not an inherited GPU venue.",
    "execution_venue_details": "The hostname identifies the actual aggregation host.",
    "phase_spans": "Disjoint measured stages expose skipped work.",
    "invocation_counts": "Loads, forwards, generations, and tokens are separate current counters.",
    "duration_s": "Monotonic time is measured and never inherited or padded.",
    "random_seed": "An empty list states that the reduction is deterministic.",
    "reproducibility_checksum": "The digest binds inputs, configuration, and reductions.",
    "source_artifact_hashes": "Producer, pre-gate, and missing sources remain distinct.",
    "validation_receipts": "Commands, exits, and log hashes support validation claims.",
    "verifier_is_oracle": "Receipt parsing is not a learned advantage.",
    "field_principles": "Each field carries its audit reason.",
    "supervisor_outcome_ledger_ready_score": "One requires an authenticated complete outcome schema.",
    "per_game_results": "Game-local outcomes preserve headroom and censoring.",
    "selection_recommendation": "No live change follows from unstable or observational evidence.",
    "solve_provenance": "Inherited live attempts receive no new solve credit.",
    "solve_claim": "False prevents this reducer from becoming a solve claim.",
    "arc_generalization_activity": "The activity names the standing outcome-ledger amendment.",
    "actual_redirects": "Each actual firing keeps its exact joined outcome.",
    "redirect_counts": "Proposed, shadow, applied, fired, helped, and censored counts stay distinct.",
    "arm_statistics": "Support and leave-one-game-out summaries are recomputed from rows.",
    "new_arm_specification": "A new arm requires recorded all-arm exhaustion.",
    "historical_verdicts": "Exp7611 and Exp7612 remain unchanged historical facts.",
    "production_defaults_changed": "False prevents observational evidence from mutating policy.",
    "generator_weights_changed": "This aggregation cannot train or alter the generator.",
    "collector_rerun": "False records compliance with the no-rerun constraint.",
    "hidden_game_source_inspected": "False protects the hidden-game boundary.",
    "offline_bfs_run": "False records that no oracle search was used.",
    "affected_file_validation_manifest": "The manifest freezes validation scope.",
    "terminal_reader_outcomes": "Independent terminal readers remain visible after publication.",
    "current_work_receipt": "Owned zero-call counters prevent inherited model activity.",
}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding measured runtime fields."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "phase_spans", "reproducibility_checksum"}
        }
    )


def _recommendation(reduction: Mapping[str, Any]) -> Json:
    if reduction.get("supervisor_outcome_ledger_ready_score") != 1:
        return {
            "action": "no_change",
            "reason": "outcome_schema_not_ready",
            "live_defaults_changed": False,
            "causal_claim": False,
        }
    if reduction.get("actual_firing_count") == 0:
        return {
            "action": "no_change",
            "reason": "no_firings_nothing_to_refine",
            "live_defaults_changed": False,
            "causal_claim": False,
        }
    stable = [
        arm
        for arm, row in (reduction.get("arm_statistics") or {}).items()
        if row.get("eligible") is True and row.get("leave_one_game_out_stable") is True
    ]
    return {
        "action": "curated_priority_review" if stable else "no_change",
        "reason": "stable_observational_association" if stable else "insufficient_support",
        "candidate_arms": stable,
        "live_defaults_changed": False,
        "causal_claim": False,
    }


def _blocked_schema_gate(errors: Sequence[str]) -> list[Json]:
    first = errors[0] if errors else "no_authenticated_receipts"
    path = first.split(":", 1)[1] if ":" in first else "selected_receipts"
    return [
        check_row(
            "supervisor_outcome_schema",
            upstream="selected_trajectory_supervisor_receipts",
            path=path,
            field="trajectory_supervisor",
            operator="has_complete_outcome_schema",
            expected=True,
            observed=False,
        )
    ]


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    reduction: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    sample_counts: Mapping[str, Any],
    terminal_reader_outcomes: Mapping[str, Any] | None = None,
    blocking_checks: Sequence[Mapping[str, Any]] = (),
    validation_passed: bool = True,
    flagged_adversarial: bool = False,
) -> Json:
    """Build one terminal artifact from authenticated reduction operands."""

    ready = int(reduction.get("supervisor_outcome_ledger_ready_score") == 1)
    firing_count = int(reduction.get("actual_firing_count") or 0)
    if blocking_checks:
        verdict = "complete_blocked_precondition"
        verdict_class = "blocked"
        gate_summary = deepcopy([dict(row) for row in blocking_checks])
    elif not validation_passed:
        verdict = "complete_disqualified_required_validation_failure"
        verdict_class = "disqualified"
        gate_summary = {"passed": False, "failed_check": "required_validation"}
    elif flagged_adversarial:
        verdict = "complete_disqualified_adversarial_reader"
        verdict_class = "disqualified"
        gate_summary = {"passed": False, "failed_check": "adversarial_verify"}
    elif not ready:
        verdict = "complete_blocked_supervisor_outcome_schema"
        verdict_class = "blocked"
        gate_summary: Any = _blocked_schema_gate(reduction.get("schema_errors") or [])
    elif firing_count == 0:
        verdict = "complete_null_no_firings_nothing_to_refine"
        verdict_class = "null"
        gate_summary = {"passed": True, "failed_check": None}
    elif any(row.get("eligible") for row in (reduction.get("arm_statistics") or {}).values()):
        verdict = "complete_null_observational_supervisor_association"
        verdict_class = "null"
        gate_summary = {"passed": True, "failed_check": None}
    else:
        verdict = "complete_null_insufficient_actual_firing_support"
        verdict_class = "null"
        gate_summary = {"passed": True, "failed_check": None}
    preconditions_pass = all(row.get("passed") is True for row in preconditions_checked)
    retention_pass = not (source_hashes.get("missing_artifacts") or [])
    current_receipt = build_current_work_receipt(
        run_id=f"exp7625-{run_date}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={
            "planned_class": "aggregation",
            "actual_class": "aggregation",
            "historical_gpu_evidence_is_current_invocation": False,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=max(0, int(float(duration_s) * 1_000_000_000)),
        phase_spans=phase_spans,
    )
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "gate_check_summary": gate_summary,
        "acceptance_gate_results": {
            "validity": {
                "condition": "all preconditions and outcome schemas pass",
                "principle": "Invalid custody or schema cannot support a reduction.",
                "observed": preconditions_pass
                and bool(ready)
                and validation_passed
                and not flagged_adversarial,
                "result": preconditions_pass
                and bool(ready)
                and validation_passed
                and not flagged_adversarial,
            },
            "readiness": {
                "condition": "supervisor_outcome_ledger_ready_score == 1",
                "principle": "A valid empty ledger is ready without proving benefit.",
                "observed": ready,
                "result": ready == 1,
            },
            "benefit": {
                "condition": "causal treatment benefit established",
                "principle": "Observational follow rates cannot prove treatment benefit.",
                "observed": False,
                "result": False,
            },
            "retention": {
                "condition": "all selected producer bytes remain hash-bound",
                "principle": "Missing retained evidence prevents future audit.",
                "observed": retention_pass,
                "result": retention_pass,
            },
            "freshness": {
                "condition": "current reduction reads frozen source hashes",
                "principle": "Fresh reduction must not invent or inherit current execution.",
                "observed": bool(source_hashes.get("producer_artifacts")),
                "result": bool(source_hashes.get("producer_artifacts")),
            },
        },
        "rows": deepcopy(list(reduction.get("rows") or [])),
        "sample_size_budget": deepcopy(dict(sample_counts)),
        "preconditions_checked": deepcopy([dict(row) for row in preconditions_checked]),
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "MODEL_SPECS": [],
        "historical_model_identity": [],
        "model_invoked": False,
        "execution_venue": "host",
        "execution_venue_details": {"hostname": socket.gethostname(), "gpu_used": False},
        "phase_spans": deepcopy([dict(row) for row in phase_spans]),
        "invocation_counts": {
            **deepcopy(current_receipt["invocation_counts"]),
            "forward_passes_attempted": 0,
            "forward_passes_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "duration_s": float(duration_s),
        "random_seed": [],
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "validation_receipts": deepcopy([dict(row) for row in validation_receipts]),
        "verifier_is_oracle": False,
        "field_principles": {},
        "supervisor_outcome_ledger_ready_score": ready,
        "per_game_results": deepcopy(list(reduction.get("per_game_results") or [])),
        "selection_recommendation": _recommendation(reduction),
        "solve_provenance": "live_agent_self_discovery",
        "solve_claim": False,
        "arc_generalization_activity": (
            "outcome-ledger supervisor refinement under the explicit standing amendment"
        ),
        "actual_redirects": deepcopy(list(reduction.get("actual_redirects") or [])),
        "redirect_counts": {
            key: int(reduction.get(key) or 0)
            for key in (
                "proposed_redirect_count",
                "would_have_redirect_count",
                "applied_redirect_count",
                "actual_firing_count",
                "helped_count",
                "censored_firing_count",
            )
        },
        "arm_statistics": deepcopy(dict(reduction.get("arm_statistics") or {})),
        "new_arm_specification": deepcopy(dict(reduction.get("new_arm_specification") or {})),
        "historical_verdicts": {
            "experiment_7611": "complete_null_matched_prefix_fixture_ready_empirical_benefit_not_established",
            "experiment_7612": "complete_blocked_exp7611_protocol",
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "collector_rerun": False,
        "hidden_game_source_inspected": False,
        "offline_bfs_run": False,
        "affected_file_validation_manifest": {
            "test_paths": [TEST_REL.as_posix()],
            "changed_modules": [MODULE_REL.as_posix()],
            "static_paths": [WRAPPER_REL.as_posix()],
            "spec_path": SPEC_REL.as_posix(),
            "research_note": NOTE_REL.as_posix(),
        },
        "terminal_reader_outcomes": deepcopy(dict(terminal_reader_outcomes or {})),
        "current_work_receipt": current_receipt,
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, operands, gates, and checksum without source access."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field_principles")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim")
    if artifact.get("production_defaults_changed") is not False:
        errors.append("production_defaults_changed")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class")
    receipt = artifact.get("current_work_receipt")
    if not isinstance(receipt, Mapping) or validate_current_work_receipt(receipt):
        errors.append("current_work_receipt")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, Mapping) or set(gates) != {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }:
        errors.append("acceptance_gate_results")
    elif any(
        not isinstance(row, Mapping) or not row.get("condition") or not row.get("principle")
        for row in gates.values()
    ):
        errors.append("acceptance_gate_principles")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        required = {
            "check",
            "upstream",
            "path",
            "field",
            "operator",
            "expected",
            "observed",
        }
        if (
            not isinstance(summary, list)
            or not summary
            or any(not isinstance(row, Mapping) or not required <= set(row) for row in summary)
        ):
            errors.append("blocked_gate_check_summary")
    for index, row in enumerate(artifact.get("rows") or []):
        if not isinstance(row, Mapping):
            errors.append(f"row_invalid:{index}")
            continue
        numerator = row.get("numerator")
        denominator = row.get("denominator")
        expected_rate = numerator / denominator if denominator else None
        if row.get("rate") != expected_rate:
            errors.append(f"row_rate_mismatch:{index}")
        operand = sha256_json(
            {
                key: row.get(key)
                for key in (
                    "unit",
                    "arm",
                    "numerator",
                    "denominator",
                    "seed",
                    "direction",
                    "censoring",
                )
            }
        )
        if row.get("operand_checksum") != operand:
            errors.append(f"row_operand_checksum:{index}")
    for index, row in enumerate(artifact.get("actual_redirects") or []):
        if not isinstance(row, Mapping):
            errors.append(f"actual_redirect_invalid:{index}")
            continue
        expected_rate = (
            row.get("numerator") / row.get("denominator") if row.get("denominator") else None
        )
        if row.get("rate") != expected_rate:
            errors.append(f"actual_redirect_rate_mismatch:{index}")
        material = {key: value for key, value in row.items() if key != "row_sha256"}
        if row.get("row_sha256") != sha256_json(material):
            errors.append(f"actual_redirect_checksum:{index}")
    counts = artifact.get("redirect_counts") or {}
    actual = list(artifact.get("actual_redirects") or [])
    valid_actual = [row for row in actual if isinstance(row, Mapping)]
    expected_counts = {
        "actual_firing_count": len(actual),
        "helped_count": sum(int(row.get("numerator") or 0) for row in valid_actual),
        "censored_firing_count": sum(
            int((row.get("censoring") or {}).get("censored") is True) for row in valid_actual
        ),
    }
    for name, value in expected_counts.items():
        if counts.get(name) != value:
            errors.append(f"redirect_count_mismatch:{name}")
    ready = artifact.get("supervisor_outcome_ledger_ready_score")
    if ready not in {0, 1}:
        errors.append("supervisor_outcome_ledger_ready_score")
    if (
        ready == 1
        and not actual
        and artifact.get("honest_verdict") != ("complete_null_no_firings_nothing_to_refine")
    ):
        errors.append("zero_firing_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def cold_reduction(path: Path) -> list[str]:
    """Reload exact bytes and run the self-contained artifact validator."""

    return validate_artifact(json.loads(Path(path).read_text(encoding="utf-8")))


def independent_reduction(path: Path) -> list[str]:
    """Recompute comparative counts and rates without trusting summaries."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    rows = list(artifact.get("rows") or [])
    proposed = sum(int(row.get("proposed_redirects") or 0) for row in rows)
    would_have = sum(int(row.get("would_have_redirects") or 0) for row in rows)
    applied = sum(int(row.get("applied_redirects") or 0) for row in rows)
    counts = artifact.get("redirect_counts") or {}
    for name, observed in (
        ("proposed_redirect_count", proposed),
        ("would_have_redirect_count", would_have),
        ("applied_redirect_count", applied),
    ):
        if counts.get(name) != observed:
            errors.append(f"independent_count_mismatch:{name}")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze focused tests, coverage, formatting, typing, and spec coverage."""

    coverage_file = private_root / "coverage" / ".coverage"
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        root,
        [TEST_REL.as_posix()],
        [MODULE_REL.as_posix()],
        static_paths=[WRAPPER_REL.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=coverage_file,
    )
    output: list[CommandSpec] = []
    for command in commands:
        argv = tuple(part for part in command.argv if not part.startswith("--data-file="))
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}:
            argv = ("/usr/bin/env", f"COVERAGE_FILE={coverage_file}", *argv)
        output.append(CommandSpec(command.name, argv, command.scope, command.timeout_s))
    return output


def build_e2e_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Declare the scoped E2E-011 and E2E-013 CPU telemetry checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    pytest = str(root / ".venv/bin/pytest")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    return [
        CommandSpec(
            "e2e_011",
            (
                pytest,
                *common,
                f"--basetemp={private_root / 'e2e-011'}",
                "tests/python/test_arc_decision_telemetry.py",
                "-q",
            ),
            "E2E-011 telemetry parity",
        ),
        CommandSpec(
            "e2e_013",
            (
                pytest,
                *common,
                f"--basetemp={private_root / 'e2e-013'}",
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
                "-q",
            ),
            "E2E-013 telemetry reduction",
        ),
    ]


def build_terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Declare fresh-process and independent readers for one candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_REL)
    return [
        CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, "--validate", str(candidate)),
            "exact candidate",
        ),
        CommandSpec(
            "fresh_process_cold_reduction",
            (python, "-u", wrapper, "--cold-reduce", str(candidate)),
            "exact candidate",
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "exact candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact candidate",
        ),
    ]


PREREQUISITE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_6844_supervisor_action_outcome_credit_audit.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_solver_kit.py"),
    Path("python/carnot/agentic/arc_supervisor_refinement.py"),
    REGISTRY_REL,
    Path("results/experiment_7611_v664_arc_matched_support.json"),
    Path("results/experiment_7612_v664_arc_history_measurement.json"),
    SPEC_REL,
)


def repo_root() -> Path:
    """Resolve this module's checkout instead of trusting the caller CWD."""

    return Path(__file__).resolve().parents[2]


def _tool_version(argv: Sequence[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed worktree executable vectors.
            list(argv),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, type(exc).__name__
    value = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0 and bool(value), value


def collect_preconditions(root: Path) -> tuple[list[Json], Json]:
    """Authenticate named inputs, tools, requirement, and solved-game history."""

    resolved = Path(root).resolve()
    checks: list[Json] = []
    hashes: dict[str, str] = {}
    for relative in PREREQUISITE_PATHS:
        path = resolved / relative
        present = path.is_file()
        checks.append(
            check_row(
                "source_custody",
                upstream="worktree",
                path=relative.as_posix(),
                field="is_file",
                operator="==",
                expected=True,
                observed=present,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)
    spec = resolved / SPEC_REL
    requirement_present = spec.is_file() and REQUIREMENT_ID in spec.read_text(encoding="utf-8")
    checks.append(
        check_row(
            "capability_requirement",
            upstream="OpenSpec",
            path=SPEC_REL.as_posix(),
            field=REQUIREMENT_ID,
            operator="contains",
            expected=True,
            observed=requirement_present,
        )
    )
    versions: dict[str, str] = {}
    for name, argv in (
        ("python", (str(resolved / ".venv/bin/python"), "--version")),
        ("pytest", (str(resolved / ".venv/bin/pytest"), "--version")),
        ("ruff", (str(resolved / ".venv/bin/ruff"), "--version")),
        ("mypy", (str(resolved / ".venv/bin/mypy"), "--version")),
    ):
        available, version = _tool_version(argv)
        versions[name] = version
        row = check_row(
            "declared_tool_version",
            upstream="worktree_virtualenv",
            path=argv[0],
            field=f"{name}_version_nonempty",
            operator="==",
            expected=True,
            observed=available,
        )
        row["version"] = version
        checks.append(row)
    registry_path = resolved / REGISTRY_REL
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
    )
    entries = {
        str(row.get("game")): row for row in registry.get("games", []) if isinstance(row, Mapping)
    }
    observed_games = sorted(game for game in GAMES if game in entries)
    checks.append(
        check_row(
            "registry_precheck",
            upstream="ops/arc_solve_registry.yaml",
            path=REGISTRY_REL.as_posix(),
            field="games_present_as_development_history_no_new_solve_proposed",
            operator="==",
            expected=sorted(GAMES),
            observed=observed_games,
        )
    )
    checks.append(
        check_row(
            "model_call_declaration",
            upstream="current_exp7625_task",
            path=MODULE_REL.as_posix(),
            field="MODEL_SPECS",
            operator="==",
            expected=[],
            observed=MODEL_SPECS,
        )
    )
    return checks, {"hashes": hashes, "tool_versions": versions, "registry_games": observed_games}


def _phase_span(phase: str, phase_started: float, started: float, *, completed_units: int) -> Json:
    end = time.monotonic() - started
    begin = phase_started - started
    return {
        "phase": phase,
        "start_s": begin,
        "end_s": end,
        "duration_s": end - begin,
        "completed_units": completed_units,
        "pending_operations": [],
        "checkpoint_position": completed_units,
    }


def _receipt_summary(receipts: Sequence[Mapping[str, Any]]) -> Json:
    return {
        str(row["name"]): {
            "exit_code": row.get("exit_code"),
            "passed": row.get("passed"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
    }


def _all_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(row.get("passed") is True for row in receipts)


def _sample_counts(selection: Mapping[str, Any], reduction: Mapping[str, Any]) -> Json:
    games = {str(row["payload"].get("game")) for row in selection.get("receipts") or []}
    censored_games = {
        str(row.get("game"))
        for row in reduction.get("per_game_results") or []
        if row.get("actual_firings", 0) > 0
        and row.get("censoring", {}).get("actual_firings_censored") == row.get("actual_firings")
    }
    return {
        "intended_independent_units": len(GAMES),
        "observed_independent_units": len(games),
        "excluded_independent_units": max(0, len(GAMES) - len(games)),
        "censored_independent_units": len(censored_games),
        "observed_receipts": int(selection.get("observed_receipts") or 0),
        "deduplicated_logical_episodes": int(selection.get("deduplicated_receipts") or 0),
        "duplicate_receipts": int(selection.get("duplicate_receipts") or 0),
        "independent_unit": "game",
        "seeds_views_replays_multiply_independent_samples": False,
    }


def _source_hash_groups(
    root: Path, context: Mapping[str, Any], selection: Mapping[str, Any]
) -> Json:
    pre_gate_paths = (
        Path("results/experiment_7611_v664_arc_matched_support.json"),
        Path("results/experiment_7612_v664_arc_history_measurement.json"),
    )
    current_paths = (MODULE_REL, TEST_REL, WRAPPER_REL, SPEC_REL, NOTE_REL)
    return {
        "producer_artifacts": deepcopy(dict(selection.get("source_hashes") or {})),
        "actual_producers": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in selection.get("receipts") or []
        ],
        "pre_gate_receipts": {
            path.as_posix(): sha256_file(root / path)
            for path in pre_gate_paths
            if (root / path).is_file()
        },
        "current_sources": {
            path.as_posix(): sha256_file(root / path)
            for path in current_paths
            if (root / path).is_file()
        },
        "precondition_sources": deepcopy(dict(context.get("hashes") or {})),
        "missing_artifacts": [
            error for error in selection.get("schema_errors") or [] if "missing" in error
        ],
    }


def run_experiment(root: Path, run_date: str, output_path: Path) -> Json:  # pragma: no cover
    """Execute aggregation, validation, exact readers, and atomic publication."""

    started = time.monotonic()
    resolved_root = Path(root).resolve()
    if resolved_root != repo_root():
        raise ValueError(f"root_mismatch:{resolved_root}:{repo_root()}")
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    target = output_path if output_path.is_absolute() else resolved_root / output_path
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7625-v665-", dir="/tmp"))
    raw_root = resolved_root / "results/raw/experiment_7625_v665_arc_supervisor_transfer"
    spans: list[Json] = []
    progress(started, "startup", "begin", root=resolved_root, private_root=private_root)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    checks, context = collect_preconditions(resolved_root)
    failed_checks = [row for row in checks if row.get("passed") is not True]
    spans.append(_phase_span("preconditions", phase_started, started, completed_units=len(checks)))
    progress(started, "preconditions", "after", checks=len(checks), failed=len(failed_checks))
    if failed_checks:
        reduction = reduce_receipts([])
        reduction["schema_errors"] = ["precondition_failed"]
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            reduction=reduction,
            preconditions_checked=checks,
            source_hashes={
                "producer_artifacts": {},
                "actual_producers": [],
                "pre_gate_receipts": {},
                "current_sources": {},
                "precondition_sources": context["hashes"],
                "missing_artifacts": [row["path"] for row in failed_checks],
            },
            validation_receipts=[],
            phase_spans=spans,
            sample_counts=_sample_counts({}, reduction),
            blocking_checks=failed_checks,
        )
        progress(started, "publication", "before_atomic_blocked", path=target)
        atomic_json(target, artifact)
        progress(started, "publication", "after_atomic_blocked", path=target)
        return artifact

    phase_started = time.monotonic()
    progress(started, "receipt_selection", "before", producers=len(DEFAULT_PRODUCERS))
    selection = select_authenticated_receipts(resolved_root)
    checkpoint = {
        "observed_receipts": selection["observed_receipts"],
        "deduplicated_receipts": selection["deduplicated_receipts"],
        "duplicate_receipts": selection["duplicate_receipts"],
        "selected": [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "logical_episode_sha256": row["logical_episode_sha256"],
                "duplicate_sources": row["duplicate_sources"],
            }
            for row in selection["receipts"]
        ],
    }
    atomic_json(raw_root / "selection-checkpoint.json", checkpoint)
    spans.append(
        _phase_span(
            "receipt_selection",
            phase_started,
            started,
            completed_units=selection["deduplicated_receipts"],
        )
    )
    progress(
        started,
        "receipt_selection",
        "after",
        observed=selection["observed_receipts"],
        deduplicated=selection["deduplicated_receipts"],
    )

    phase_started = time.monotonic()
    progress(started, "outcome_reduction", "before", units=len(selection["receipts"]))
    reduction = reduce_receipts(selection["receipts"])
    if selection["schema_errors"]:
        reduction["schema_errors"] = list(
            dict.fromkeys([*selection["schema_errors"], *reduction["schema_errors"]])
        )
        reduction["supervisor_outcome_ledger_ready_score"] = 0
    spans.append(
        _phase_span(
            "outcome_reduction",
            phase_started,
            started,
            completed_units=len(selection["receipts"]),
        )
    )
    progress(
        started,
        "outcome_reduction",
        "after",
        actual_firings=reduction["actual_firing_count"],
        schema_errors=len(reduction["schema_errors"]),
    )

    phase_started = time.monotonic()
    validation_commands = build_validation_commands(resolved_root, private_root / "validation")
    progress(started, "scoped_validation", "before_subprocesses", units=len(validation_commands))
    validation = run_commands(
        resolved_root,
        validation_commands,
        log_dir=raw_root / "logs/scoped",
    )
    spans.append(
        _phase_span("scoped_validation", phase_started, started, completed_units=len(validation))
    )
    progress(started, "scoped_validation", "after_subprocesses", passed=_all_passed(validation))

    phase_started = time.monotonic()
    e2e_commands = build_e2e_commands(resolved_root, private_root / "e2e")
    progress(started, "task_e2e", "before_subprocesses", units=len(e2e_commands))
    e2e = run_commands(resolved_root, e2e_commands, log_dir=raw_root / "logs/e2e")
    spans.append(_phase_span("task_e2e", phase_started, started, completed_units=len(e2e)))
    progress(started, "task_e2e", "after_subprocesses", passed=_all_passed(e2e))

    source_hashes = _source_hash_groups(resolved_root, context, selection)
    samples = _sample_counts(selection, reduction)
    base_receipts = [*validation, *e2e]
    required_validation_passed = _all_passed(validation) and _all_passed(e2e)
    preliminary = build_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        reduction=reduction,
        preconditions_checked=checks,
        source_hashes=source_hashes,
        validation_receipts=base_receipts,
        phase_spans=spans,
        sample_counts=samples,
        validation_passed=required_validation_passed,
    )
    errors = validate_artifact(preliminary)
    if errors:
        raise RuntimeError("preliminary_candidate_invalid:" + ",".join(errors))
    preliminary_path = private_root / "terminal/preliminary-candidate.json"
    atomic_json(preliminary_path, preliminary)

    phase_started = time.monotonic()
    progress(
        started,
        "terminal_readers_preliminary",
        "before_subprocesses",
        units=5,
        candidate_sha256=sha256_file(preliminary_path),
    )
    preliminary_readers = run_commands(
        resolved_root,
        build_terminal_commands(resolved_root, preliminary_path),
        log_dir=raw_root / "logs/terminal-preliminary",
    )
    spans.append(
        _phase_span(
            "terminal_readers_preliminary",
            phase_started,
            started,
            completed_units=len(preliminary_readers),
        )
    )
    progress(
        started,
        "terminal_readers_preliminary",
        "after_subprocesses",
        passed=_all_passed(preliminary_readers),
    )
    adversarial = next(
        (row for row in preliminary_readers if row.get("name") == "adversarial_verify"), {}
    )
    flagged = "CRITICAL" in str(adversarial.get("output_tail") or "")
    final = build_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        reduction=reduction,
        preconditions_checked=checks,
        source_hashes=source_hashes,
        validation_receipts=[*base_receipts, *preliminary_readers],
        phase_spans=spans,
        sample_counts=samples,
        terminal_reader_outcomes=_receipt_summary(preliminary_readers),
        validation_passed=required_validation_passed and _all_passed(preliminary_readers),
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError("terminal_candidate_invalid:" + ",".join(errors))
    final_path = private_root / "terminal/final-candidate.json"
    atomic_json(final_path, final)

    phase_started = time.monotonic()
    progress(
        started,
        "terminal_readers_exact",
        "before_subprocesses",
        units=5,
        candidate_sha256=sha256_file(final_path),
    )
    exact_readers = run_commands(
        resolved_root,
        build_terminal_commands(resolved_root, final_path),
        log_dir=raw_root / "logs/terminal-exact",
    )
    spans.append(
        _phase_span(
            "terminal_readers_exact",
            phase_started,
            started,
            completed_units=len(exact_readers),
        )
    )
    exact_passed = _all_passed(exact_readers)
    progress(started, "terminal_readers_exact", "after_subprocesses", passed=exact_passed)
    atomic_json(raw_root / "terminal-exact-reader-outcomes.json", {"receipts": exact_readers})
    if not exact_passed:
        raise RuntimeError("exact_terminal_reader_failed")
    if sha256_file(resolved_root / REGISTRY_REL) != context["hashes"][REGISTRY_REL.as_posix()]:
        raise RuntimeError("solve_registry_changed_during_reduction")

    progress(started, "publication", "before_atomic", path=target)
    atomic_json(target, final)
    progress(
        started,
        "publication",
        "after_atomic",
        path=target,
        verdict=final["honest_verdict"],
        sha256=sha256_file(target),
    )
    return final


def _date_argument(value: str) -> str:
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_REL)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--cold-reduce", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(json.loads(args.validate.read_text(encoding="utf-8")))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.cold_reduce is not None:
        errors = cold_reduction(args.cold_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_reduction(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
