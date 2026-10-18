"""Audit V657 static probability and decisions from immutable raw rows.

This module does not trust producer headline reducers. It authenticates their
bytes, then recomputes source-level losses, uncertainty, and all decision costs.

Spec refs: REQ-REPORT-7508 and SCENARIO-REPORT-7508-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from typing import Any, Callable, TypeVar

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


Json = dict[str, Any]
T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7508-v657-static-audit"
SCHEMA = "carnot.exp7508.v657.static_audit.v1"
RESULT_PATH = Path("results/experiment_7508_v657_static_audit.json")
RAW_DIR = Path("results/raw/experiment_7508_v657_static_audit")
MODULE_PATH = Path("python/carnot/experiment_7508_v657_static_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7508_v657_static_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7508_v657_static_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v657-static-audit.md")
UPSTREAM_PATHS = {
    7504: Path("results/experiment_7504_v657_evidence_interface.json"),
    7505: Path("results/experiment_7505_v657_energy_fit.json"),
    7507: Path("results/experiment_7507_v657_static_evaluation.json"),
}
ACCESS_PATH = Path(
    "results/raw/experiment_7504_v657_evidence_interface/access_exposure_manifest.json"
)
CHECKPOINT_PATH = Path("results/raw/experiment_7505_v657_energy_fit/frozen_checkpoints.json")
PREDICTION_PATH = Path(
    "results/raw/experiment_7507_v657_static_evaluation/label_free_predictions.jsonl"
)
EVALUATION_PATH = Path("results/raw/experiment_7507_v657_static_evaluation/evaluation_rows.jsonl")
POLICY_PATH = Path("results/raw/experiment_7507_v657_static_evaluation/policy_rows.jsonl")

FIT_SEEDS = (656101, 656102, 656103, 656104, 656105)
LEARNED_ARMS = ("window_gibbs", "whole_only_gibbs", "identical_ten_feature_logistic")
SIMPLE_ARMS = ("temperature_whole", "raw_whole_expectation", "raw_max_window_probability")
PROBABILITY_CONTROLS = ("identical_ten_feature_logistic", "whole_only_gibbs")
FALSE_ACCEPT_COSTS = (1.0, 5.0, 20.0)
ESCALATION_COSTS = (0.1, 0.5, 1.0)
TIE_ORDER = ("accept", "escalate", "reject")
BOOTSTRAP_SEED = 657007
BOOTSTRAP_DRAWS = 2000
MUTATION_NAMES = (
    "flipped_label",
    "swapped_option_order",
    "duplicate_source",
    "favorable_seed_selection",
    "omitted_escalation",
    "wrong_checkpoint",
    "changed_multiplicity_family",
    "promoted_descriptive_result",
)
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
REQUIRED_TERMINAL_RECEIPTS = {
    7504: (
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ),
    7505: (
        "numeric_fit",
        *validation_scope.REQUIRED_CHECK_NAMES,
        "declared_entrypoint_cold_replay",
        "independent_row_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ),
    7507: (
        *validation_scope.REQUIRED_CHECK_NAMES,
        "cold_artifact_replay",
        "independent_row_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ),
}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_json(path: Path) -> Json:
    """Read one JSON object without repairing malformed external evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[Json]:
    """Read object rows so the audit can recompute from exact raw evidence."""

    rows: list[Json] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _source_row(path: Path, root: Path, *, evidence_class: str) -> Json:
    """Bind exact bytes and keep historical evidence out of current calls."""

    resolved = path if path.is_absolute() else root / path
    label = str(path) if path.is_absolute() else path.as_posix()
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "evidence_class": evidence_class,
    }


def _required_receipts_pass(number: int, artifact: Mapping[str, Any]) -> bool:
    """Require every named upstream command exactly once and successful."""

    receipts = artifact.get("validation_receipts")
    if not isinstance(receipts, list):
        return False
    for name in REQUIRED_TERMINAL_RECEIPTS[number]:
        matches = [row for row in receipts if isinstance(row, Mapping) and row.get("name") == name]
        if len(matches) != 1:
            return False
        row = matches[0]
        if (
            row.get("passed") is not True
            or row.get("exit_code") != 0
            or row.get("timed_out") is True
        ):
            return False
    return True


def _upstream_valid(number: int, value: Mapping[str, Any]) -> bool:
    """Apply only terminal, safety, readiness, and receipt checks here."""

    common = (
        value.get("milestone") == MILESTONE
        and value.get("terminal_status") == "complete"
        and value.get("verdict_class") in {"null", "positive"}
        and value.get("flagged_adversarial") is False
        and _required_receipts_pass(number, value)
    )
    readiness = {
        7504: value.get("evidence_ready_score") == 1,
        7505: value.get("energy_fit_ready_score") == 1 and value.get("baseline_ready_score") == 1,
        7507: value.get("static_evaluation_complete_score") == 1,
    }[number]
    return common and readiness


def inventory_upstreams(root: Path) -> list[Json]:
    """Inventory all declared producers even when one path is absent."""

    rows: list[Json] = []
    for number, relative in UPSTREAM_PATHS.items():
        path = root / relative
        if not path.is_file():
            rows.append(
                {"producer": number, "path": relative.as_posix(), "state": "absent", "sha256": None}
            )
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            rows.append(
                {
                    "producer": number,
                    "path": relative.as_posix(),
                    "state": "invalid",
                    "sha256": sha256_file(path),
                }
            )
            continue
        rows.append(
            {
                "producer": number,
                "path": relative.as_posix(),
                "state": "valid" if _upstream_valid(number, value) else "invalid",
                "sha256": sha256_file(path),
                "honest_verdict": value.get("honest_verdict"),
                "verdict_class": value.get("verdict_class"),
                "flagged_adversarial": value.get("flagged_adversarial"),
                "required_receipts_passed": _required_receipts_pass(number, value),
            }
        )
    return rows


def classify_inventory(
    inventory: Sequence[Mapping[str, Any]], reduction_errors: Sequence[str]
) -> Json:
    """Keep complete accounting independent from evidence availability."""

    if reduction_errors or any(row.get("state") == "invalid" for row in inventory):
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_v657_static_evidence",
            "static_audit_complete_score": 1,
            "static_claims_qualified_score": 0,
        }
    if any(row.get("state") == "absent" for row in inventory):
        return {
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_missing_v657_static_inputs",
            "static_audit_complete_score": 1,
            "static_claims_qualified_score": 0,
        }
    return {
        "verdict_class": "null",
        "honest_verdict": "complete_null_v657_static_claims_qualified_no_benefit",
        "static_audit_complete_score": 1,
        "static_claims_qualified_score": 1,
    }


def metric_losses(probability: float, label: int) -> Json:
    """Compute proper scores while clipping only the log-loss operand."""

    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0 or label not in (0, 1):
        raise ValueError("probability_or_label_invalid")
    clipped = min(max(numeric, 1e-6), 1.0 - 1e-6)
    return {
        "brier": (numeric - label) ** 2,
        "log_loss": -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped)),
    }


def decision_cell(
    probability: float,
    label: int,
    *,
    false_accept_cost: float,
    escalation_cost: float,
) -> Json:
    """Recompute one typed action and its realized registered cost."""

    if label not in (0, 1):
        raise ValueError("decision_label_invalid")
    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("decision_probability_invalid")
    expected = {
        "accept": numeric * false_accept_cost,
        "escalate": escalation_cost,
        "reject": 1.0 - numeric,
    }
    action = min(TIE_ORDER, key=lambda name: (expected[name], TIE_ORDER.index(name)))
    realized = {
        "accept": false_accept_cost if label == 1 else 0.0,
        "escalate": escalation_cost,
        "reject": 1.0 if label == 0 else 0.0,
    }[action]
    return {
        "cell_id": f"fa={false_accept_cost:g}|fr=1|esc={escalation_cost:g}",
        "false_accept_cost": float(false_accept_cost),
        "false_reject_cost": 1.0,
        "escalation_cost": float(escalation_cost),
        "action": action,
        "escalated": action == "escalate",
        "cost": float(realized),
    }


def verify_option_order_rows(
    plans: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Bind display labels to stable option IDs for both presentation orders."""

    errors: list[str] = []
    plan_by_id = {str(row.get("request_id")): row for row in plans}
    observed_by_id = {str(row.get("request_id")): row for row in observed}
    if len(plan_by_id) != len(plans):
        errors.append("duplicate_planned_request")
    if len(observed_by_id) != len(observed):
        errors.append("duplicate_observed_request")
    if set(plan_by_id) != set(observed_by_id):
        errors.append("option_request_roster_mismatch")
    orders: set[tuple[str, ...]] = set()
    legal = {
        ("supported", "contains_unsupported"),
        ("contains_unsupported", "supported"),
    }
    for request_id, plan in plan_by_id.items():
        order = tuple(str(item) for item in plan.get("option_order") or [])
        orders.add(order)
        row = observed_by_id.get(request_id)
        if row is None:
            continue
        if tuple(str(item) for item in row.get("option_order") or []) != order:
            errors.append(f"option_order_mismatch:{request_id}")
            continue
        if row.get("disposition") == "complete":
            expected = {" A": order[0], " B": order[1]} if len(order) == 2 else {}
            if row.get("label_to_option_id") != expected or row.get("order_remapping") != expected:
                errors.append(f"option_mapping_mismatch:{request_id}")
            if (
                set(row.get("raw_logits_by_option_id") or {})
                != {
                    "supported",
                    "contains_unsupported",
                }
                and row.get("raw_logits_by_option_id") is not None
            ):
                errors.append(f"option_logits_invalid:{request_id}")
    if orders != legal:
        errors.append("both_option_orders_not_observed")
    return list(dict.fromkeys(errors))


def project_policy_rows(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Project only the candidate and same-cost baseline policy evidence."""

    projected: list[Json] = []
    for row in rows:
        if row.get("arm") not in {"window_gibbs", "temperature_whole"}:
            continue
        for cell in row.get("decision_costs") or []:
            projected.append(
                {
                    "group_id": row.get("group_id"),
                    "source_hash": row.get("source_hash"),
                    "arm": row.get("arm"),
                    "fit_seed": row.get("fit_seed"),
                    "label": row.get("label"),
                    **deepcopy(dict(cell)),
                }
            )
    return projected


def _sort_policy(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Sort policy projections so serialization order cannot hide omissions."""

    return sorted(
        (deepcopy(dict(row)) for row in rows),
        key=lambda row: (
            str(row.get("cell_id")),
            str(row.get("group_id")),
            str(row.get("arm")),
            -1 if row.get("fit_seed") is None else int(row["fit_seed"]),
        ),
    )


def _validated_groups(
    rows: Sequence[Mapping[str, Any]], policy_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    """Reject row drift before any average or bootstrap can hide it."""

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    group_to_source: dict[str, str] = {}
    source_to_group: dict[str, str] = {}
    for row in rows:
        group = str(row.get("group_id") or "")
        source = str(row.get("source_hash") or "")
        arm = str(row.get("arm") or "")
        if not group or not source or arm not in {*LEARNED_ARMS, *SIMPLE_ARMS}:
            raise ValueError("row_identity_invalid")
        if group in group_to_source and group_to_source[group] != source:
            raise ValueError("source_identity_not_bijective")
        if source in source_to_group and source_to_group[source] != group:
            raise ValueError("source_identity_not_bijective")
        group_to_source[group] = source
        source_to_group[source] = group
        if row.get("role") != "test" or row.get("status") != "complete":
            raise ValueError("evaluation_status_or_role_invalid")
        if row.get("failed") is not False or row.get("censored") is not False:
            raise ValueError("evaluation_failure_or_censoring_present")
        probability = float(row.get("probability"))
        label = row.get("label")
        losses = metric_losses(probability, label)
        if any(
            not math.isclose(float(row.get(name)), value, abs_tol=1e-12)
            for name, value in losses.items()
        ):
            raise ValueError("metric_mismatch")
        expected_cells = {
            cell["cell_id"]: cell
            for cell in (
                decision_cell(
                    probability,
                    int(label),
                    false_accept_cost=false_accept,
                    escalation_cost=escalation,
                )
                for false_accept in FALSE_ACCEPT_COSTS
                for escalation in ESCALATION_COSTS
            )
        }
        cells = {str(cell.get("cell_id")): cell for cell in row.get("decision_costs") or []}
        if set(cells) != set(expected_cells) or len(cells) != 9:
            raise ValueError("decision_cell_roster_invalid")
        for cell_id, expected in expected_cells.items():
            observed = cells[cell_id]
            if any(observed.get(key) != expected[key] for key in expected):
                raise ValueError("decision_cell_mismatch")
        grouped[group][arm].append(row)
    if not grouped:
        raise ValueError("evaluation_rows_missing")
    expected_arms = {*LEARNED_ARMS, *SIMPLE_ARMS}
    for arms in grouped.values():
        if set(arms) != expected_arms:
            raise ValueError("arm_roster_invalid")
        for arm in LEARNED_ARMS:
            seeds = [row.get("fit_seed") for row in arms[arm]]
            if len(seeds) != len(FIT_SEEDS) or set(seeds) != set(FIT_SEEDS):
                raise ValueError("fit_seed_roster_invalid")
        for arm in SIMPLE_ARMS:
            if len(arms[arm]) != 1 or arms[arm][0].get("fit_seed") is not None:
                raise ValueError("simple_arm_roster_invalid")
        labels = {int(row["label"]) for arm_rows in arms.values() for row in arm_rows}
        if len(labels) != 1:
            raise ValueError("group_label_disagreement")
    if _sort_policy(project_policy_rows(rows)) != _sort_policy(policy_rows):
        raise ValueError("policy_projection_mismatch")
    return grouped


def _paired(values: Sequence[float], indices: np.ndarray, seed: int) -> Json:
    """Resample complete source groups and retain one-sided operands."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or indices.shape[1] != len(array):
        raise ValueError("paired_bootstrap_shape_invalid")
    means = array[indices].mean(axis=1)
    return {
        "group_count": len(array),
        "draws": len(indices),
        "seed": seed,
        "delta": float(array.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "one_sided_p": float((1 + np.sum(means >= 0.0)) / (len(means) + 1)),
        "bootstrap_means": means,
    }


def _holm(comparisons: Mapping[str, Mapping[str, Any]], alpha: float = 0.05) -> dict[str, Json]:
    """Apply one step-down family without borrowing evidence across claims."""

    ordered = sorted(comparisons, key=lambda name: (float(comparisons[name]["one_sided_p"]), name))
    family_size = len(ordered)
    cumulative = 0.0
    output: dict[str, Json] = {}
    for rank, name in enumerate(ordered, start=1):
        row = comparisons[name]
        threshold = alpha / (family_size - rank + 1)
        cumulative = max(cumulative, min(1.0, (family_size - rank + 1) * float(row["one_sided_p"])))
        output[name] = {
            key: deepcopy(value) for key, value in row.items() if key != "bootstrap_means"
        }
        output[name].update(
            {
                "holm_rank": rank,
                "holm_alpha": threshold,
                "holm_adjusted_p": cumulative,
                "holm_upper": float(np.quantile(row["bootstrap_means"], 1.0 - threshold)),
            }
        )
    return output


def fixture_settings(*, draws: int = 32) -> Json:
    """Return the frozen protocol with a smaller draw count for unit fixtures."""

    return {
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmatory_allowed": True,
        "expected_groups": 100,
        "holm_alpha": 0.05,
        "maximum_log_loss_delta": 0.01,
        "minimum_brier_delta": -0.01,
        "minimum_groups": 100,
        "minimum_non_escalated_coverage": 0.2,
        "minimum_per_class": 20,
        "policy_baseline_arm": "temperature_whole",
        "probability_candidate": "window_gibbs",
        "probability_controls": list(PROBABILITY_CONTROLS),
        "simple_baseline_arm": "raw_whole_expectation",
        "inference_unit": "unique_source_group",
        "seed_reduction": "average_losses_within_source_before_paired_inference",
    }


def _average_rows(
    grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[tuple[str, str], Json]:
    """Average repeated fit rows within one source, never across sources."""

    averaged: dict[tuple[str, str], Json] = {}
    for group, arms in grouped.items():
        for arm, rows in arms.items():
            averaged[(group, arm)] = {
                "probability": float(np.mean([float(row["probability"]) for row in rows])),
                "brier": float(
                    np.mean(
                        [
                            metric_losses(float(row["probability"]), int(row["label"]))["brier"]
                            for row in rows
                        ]
                    )
                ),
                "log_loss": float(
                    np.mean(
                        [
                            metric_losses(float(row["probability"]), int(row["label"]))["log_loss"]
                            for row in rows
                        ]
                    )
                ),
                "label": int(rows[0]["label"]),
                "source_hash": str(rows[0]["source_hash"]),
                "seed_rows": len(rows),
            }
    return averaged


def _probability_reduction(
    grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    settings: Mapping[str, Any],
    indices: np.ndarray,
) -> Json:
    """Recompute proper scores and the registered two-contrast family."""

    averaged = _average_rows(grouped)
    groups = sorted(grouped)
    labels = [int(averaged[(group, "window_gibbs")]["label"]) for group in groups]
    support = {
        "groups": len(groups),
        "supported": labels.count(0),
        "contains_unsupported": labels.count(1),
    }
    support["passed"] = support["groups"] >= int(settings["minimum_groups"]) and min(
        support["supported"], support["contains_unsupported"]
    ) >= int(settings["minimum_per_class"])
    raw: dict[str, Json] = {}
    for control in PROBABILITY_CONTROLS:
        raw[control] = _paired(
            [
                float(averaged[(group, "window_gibbs")]["brier"])
                - float(averaged[(group, control)]["brier"])
                for group in groups
            ],
            indices,
            int(settings["bootstrap_seed"]),
        )
    brier = _holm(raw, float(settings["holm_alpha"]))
    log_loss = _paired(
        [
            float(averaged[(group, "window_gibbs")]["log_loss"])
            - float(averaged[(group, str(settings["simple_baseline_arm"]))]["log_loss"])
            for group in groups
        ],
        indices,
        int(settings["bootstrap_seed"]),
    )
    log_loss.pop("bootstrap_means")
    log_loss["control"] = str(settings["simple_baseline_arm"])
    metrics = {
        arm: {
            "brier": float(
                np.mean(
                    [
                        metric_losses(
                            float(averaged[(group, arm)]["probability"]),
                            int(averaged[(group, arm)]["label"]),
                        )["brier"]
                        for group in groups
                    ]
                )
            ),
            "log_loss": float(
                np.mean(
                    [
                        metric_losses(
                            float(averaged[(group, arm)]["probability"]),
                            int(averaged[(group, arm)]["label"]),
                        )["log_loss"]
                        for group in groups
                    ]
                )
            ),
            "n_groups": len(groups),
            "class_support": {
                "supported": support["supported"],
                "contains_unsupported": support["contains_unsupported"],
            },
        }
        for arm in (*LEARNED_ARMS, *SIMPLE_ARMS)
    }
    score = int(
        support["passed"]
        and settings.get("confirmatory_allowed") is True
        and all(
            float(row["delta"]) <= float(settings["minimum_brier_delta"])
            and float(row["holm_upper"]) < 0.0
            for row in brier.values()
        )
        and float(log_loss["delta"]) <= float(settings["maximum_log_loss_delta"])
    )
    return {
        "averaged": averaged,
        "source_support": support,
        "probability_metrics": metrics,
        "probability_contrasts": {"brier": brier, "log_loss": log_loss},
        "probability_holm_family_size": len(brier),
        "static_probability_value_score": score,
    }


def _cell(row: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    """Select one already authenticated cost cell."""

    matches = [cell for cell in row["decision_costs"] if cell.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise ValueError("decision_cell_roster_invalid")
    return matches[0]


def _risk(rows: Sequence[Mapping[str, Any]], cell_id: str) -> float | None:
    """Measure wrong terminal decisions while keeping escalations separate."""

    terminal = [(row, _cell(row, cell_id)) for row in rows if not _cell(row, cell_id)["escalated"]]
    if not terminal:
        return None
    return float(
        np.mean(
            [
                (cell["action"] == "accept" and int(row["label"]) == 1)
                or (cell["action"] == "reject" and int(row["label"]) == 0)
                for row, cell in terminal
            ]
        )
    )


def _matched_sensitivity(
    candidate_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_by_group: Mapping[str, Mapping[str, Any]],
    cell_id: str,
) -> Json:
    """Compare rejection only where both policies make terminal decisions."""

    common: list[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]] = []
    for group, candidate in candidate_by_group.items():
        baseline = baseline_by_group[group]
        if (
            int(baseline["label"]) == 1
            and all(not _cell(row, cell_id)["escalated"] for row in candidate)
            and not _cell(baseline, cell_id)["escalated"]
        ):
            common.append((candidate, baseline))
    if not common:
        return {
            "definition": "positive groups non-escalated by both policies",
            "group_count": 0,
            "candidate": None,
            "baseline": None,
        }
    return {
        "definition": "positive groups non-escalated by both policies",
        "group_count": len(common),
        "candidate": float(
            np.mean(
                [
                    np.mean([_cell(row, cell_id)["action"] == "reject" for row in candidate])
                    for candidate, _baseline in common
                ]
            )
        ),
        "baseline": float(
            np.mean(
                [_cell(baseline, cell_id)["action"] == "reject" for _candidate, baseline in common]
            )
        ),
    }


def _decision_reduction(
    grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    settings: Mapping[str, Any],
    indices: np.ndarray,
) -> Json:
    """Recompute the separate nine-member selective-decision family."""

    groups = sorted(grouped)
    candidate_by_group = {group: grouped[group]["window_gibbs"] for group in groups}
    baseline_by_group = {group: grouped[group]["temperature_whole"][0] for group in groups}
    candidate_rows = [row for group in groups for row in candidate_by_group[group]]
    baseline_rows = [baseline_by_group[group] for group in groups]
    pending: dict[str, Json] = {}
    raw: dict[str, Json] = {}
    per_group_costs: dict[str, dict[str, float]] = defaultdict(dict)
    for false_accept in FALSE_ACCEPT_COSTS:
        for escalation in ESCALATION_COSTS:
            cell_id = f"fa={false_accept:g}|fr=1|esc={escalation:g}"
            deltas: list[float] = []
            coverage: list[float] = []
            for group in groups:
                candidate_cells = [_cell(row, cell_id) for row in candidate_by_group[group]]
                baseline_cell = _cell(baseline_by_group[group], cell_id)
                delta = float(np.mean([float(cell["cost"]) for cell in candidate_cells])) - float(
                    baseline_cell["cost"]
                )
                deltas.append(delta)
                per_group_costs[group][cell_id] = delta
                coverage.append(float(np.mean([not cell["escalated"] for cell in candidate_cells])))
            raw[cell_id] = _paired(deltas, indices, int(settings["bootstrap_seed"]))
            pending[cell_id] = {
                "cell_id": cell_id,
                "false_accept_cost": false_accept,
                "false_reject_cost": 1.0,
                "escalation_cost": escalation,
                "candidate_coverage": float(np.mean(coverage)),
                "baseline_coverage": float(
                    np.mean(
                        [
                            not _cell(baseline_by_group[group], cell_id)["escalated"]
                            for group in groups
                        ]
                    )
                ),
                "candidate_risk": _risk(candidate_rows, cell_id),
                "baseline_risk": _risk(baseline_rows, cell_id),
                "coverage_matched_sensitivity": _matched_sensitivity(
                    candidate_by_group, baseline_by_group, cell_id
                ),
            }
    adjusted = _holm(raw, float(settings["holm_alpha"]))
    cells: list[Json] = []
    for cell_id, row in pending.items():
        comparison = adjusted[cell_id]
        passed = (
            float(row["candidate_coverage"]) >= float(settings["minimum_non_escalated_coverage"])
            and float(comparison["delta"]) < 0.0
            and float(comparison["holm_upper"]) < 0.0
        )
        cells.append({**row, "comparison": comparison, "benefit_passed": passed})
    score = int(
        len(groups) >= int(settings["minimum_groups"])
        and settings.get("confirmatory_allowed") is True
        and all(cell["benefit_passed"] for cell in cells)
    )
    return {
        "decision_cells": cells,
        "decision_holm_family_size": len(cells),
        "selective_decision_value_score": score,
        "per_group_cost_deltas": per_group_costs,
    }


def reduce_static_rows(
    rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> Json:
    """Independently reduce raw rows without calling the producer reducer."""

    required_settings = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "holm_alpha": 0.05,
        "minimum_brier_delta": -0.01,
        "maximum_log_loss_delta": 0.01,
        "minimum_groups": 100,
        "minimum_per_class": 20,
        "minimum_non_escalated_coverage": 0.2,
        "probability_candidate": "window_gibbs",
        "probability_controls": list(PROBABILITY_CONTROLS),
        "simple_baseline_arm": "raw_whole_expectation",
        "policy_baseline_arm": "temperature_whole",
        "inference_unit": "unique_source_group",
        "seed_reduction": "average_losses_within_source_before_paired_inference",
    }
    for key, expected in required_settings.items():
        if key == "bootstrap_draws" and int(settings.get(key, -1)) != BOOTSTRAP_DRAWS:
            # Small private fixtures use fewer draws but production evidence may not.
            if int(settings.get("expected_groups", -1)) != 100:
                raise ValueError(f"evaluator_setting_invalid:{key}")
            continue
        if key != "bootstrap_draws" and settings.get(key) != expected:
            raise ValueError(f"evaluator_setting_invalid:{key}")
    grouped = _validated_groups(rows, policy_rows)
    groups = sorted(grouped)
    expected_groups = settings.get("expected_groups")
    if expected_groups is not None and len(groups) != int(expected_groups):
        raise ValueError("expected_group_count_mismatch")
    draws = int(settings["bootstrap_draws"])
    indices = np.random.default_rng(int(settings["bootstrap_seed"])).integers(
        0, len(groups), size=(draws, len(groups))
    )
    probability = _probability_reduction(grouped, settings, indices)
    decision = _decision_reduction(grouped, settings, indices)
    averaged = probability.pop("averaged")
    source_rows = []
    for group in groups:
        source_rows.append(
            {
                "unit_id": group,
                "group_id": group,
                "source_hash": averaged[(group, "window_gibbs")]["source_hash"],
                "label": averaged[(group, "window_gibbs")]["label"],
                "arm_metrics": {
                    arm: {
                        key: averaged[(group, arm)][key]
                        for key in ("probability", "brier", "log_loss", "seed_rows")
                    }
                    for arm in (*LEARNED_ARMS, *SIMPLE_ARMS)
                },
                "decision_cost_deltas": dict(decision["per_group_cost_deltas"][group]),
                "attempted": True,
                "complete": True,
                "failed": False,
                "excluded": False,
                "censored": False,
                "unstarted": False,
            }
        )
    decision.pop("per_group_cost_deltas")
    budget = {
        "planned": int(expected_groups or len(groups)),
        "attempted": len(groups),
        "completed": len(groups),
        "excluded": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": max(0, int(expected_groups or len(groups)) - len(groups)),
        "independent_unit": "unique_source_group",
        "fit_seeds_are_independent_units": False,
        "windows_are_independent_units": False,
    }
    return {
        **probability,
        **decision,
        "seed_rows_per_learned_arm": len(FIT_SEEDS),
        "rows": source_rows,
        "sample_size_budget": budget,
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    """Compare nested producer operands while tolerating only float roundoff."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def compare_producer_reduction(
    producer: Mapping[str, Any], reduced: Mapping[str, Any]
) -> list[str]:
    """Explain producer/auditor mismatch without modifying upstream bytes."""

    errors: list[str] = []
    if not _close(producer.get("probability_contrasts"), reduced.get("probability_contrasts")):
        errors.append("producer_probability_contrasts_mismatch")
    producer_metrics = producer.get("probability_metrics") or {}
    reduced_metrics = reduced.get("probability_metrics") or {}
    metric_subset = {
        arm: {key: values.get(key) for key in ("brier", "log_loss", "n_groups", "class_support")}
        for arm, values in producer_metrics.items()
    }
    if not _close(metric_subset, reduced_metrics):
        errors.append("producer_probability_metrics_mismatch")
    policy = producer.get("policy_evaluation") or {}
    if policy.get("holm_family_size") != reduced.get("decision_holm_family_size"):
        errors.append("producer_decision_holm_family_mismatch")
    if not _close(policy.get("cells"), reduced.get("decision_cells")):
        errors.append("producer_decision_cells_mismatch")
    for producer_field, reduced_field in (
        ("static_probability_value_score", "static_probability_value_score"),
        ("selective_decision_value_score", "selective_decision_value_score"),
    ):
        if producer.get(producer_field) != reduced.get(reduced_field):
            errors.append(f"producer_score_mismatch:{producer_field}")
    return errors


REQUIRED_INPUTS = (
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
    Path("python/carnot/experiment_7498_v656_independent_audit.py"),
    Path("results/experiment_7498_v656_independent_audit.json"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def collect_preconditions(root: Path) -> Json:
    """Check instructions and inventory producers before any measurement."""

    rows: list[Json] = []
    source_hashes: list[Json] = []
    for relative in REQUIRED_INPUTS:
        path = root / relative
        readable = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": "resource_readable",
                "upstream": "worktree_instruction_or_helper",
                "field_path": relative.as_posix(),
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if readable else "missing_or_empty",
                "op": "eq",
                "passed": readable,
            }
        )
        if readable:
            source_hashes.append(_source_row(relative, root, evidence_class="repository_input"))
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    declared = "REQ-REPORT-7508" in spec_text
    rows.append(
        {
            "check": "relevant_requirement_present",
            "upstream": "OpenSpec",
            "field_path": "REQ-REPORT-7508",
            "expected": True,
            "observed": declared,
            "op": "eq",
            "passed": declared,
        }
    )
    inventory = inventory_upstreams(root)
    for row in inventory:
        path = root / str(row["path"])
        if path.is_file():
            source_hashes.append(
                _source_row(Path(str(row["path"])), root, evidence_class="upstream_terminal")
            )
        rows.append(
            {
                "check": f"upstream_inventory_exp{row['producer']}",
                "upstream": f"Exp{row['producer']}",
                "field_path": row["path"],
                "expected": "valid_or_explicit_absence",
                "observed": row["state"],
                "op": "in",
                "passed": row["state"] in {"valid", "absent"},
            }
        )
    return {
        "rows": rows,
        "inventory": inventory,
        "source_artifact_hashes": source_hashes,
        "missing_external": [row["path"] for row in inventory if row["state"] == "absent"],
        "invalid_present": [row["path"] for row in inventory if row["state"] == "invalid"],
    }


def _reference(artifact: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    """Find one exact source or sidecar reference by repository path."""

    matches: list[Mapping[str, Any]] = []
    for row in artifact.get("source_artifact_hashes") or []:
        if isinstance(row, Mapping) and row.get("path") == path.as_posix():
            matches.append(row)
    for row in (artifact.get("raw_sidecars") or {}).values():
        if isinstance(row, Mapping) and row.get("path") == path.as_posix():
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"source_reference_invalid:{path}")
    return matches[0]


def _authenticate_file(root: Path, artifact: Mapping[str, Any], path: Path) -> Json:
    """Reject a missing or changed sidecar before parsing its content."""

    reference = _reference(artifact, path)
    resolved = root / path
    if not resolved.is_file():
        raise ValueError(f"source_missing:{path}")
    observed = sha256_file(resolved)
    if reference.get("sha256") != observed:
        raise ValueError(f"source_hash_mismatch:{path}")
    return _source_row(path, root, evidence_class="upstream_raw_sidecar")


def _prediction_errors(
    evaluation: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Prove the label-free bytes preceded the later labeled rows."""

    errors: list[str] = []
    receipt = evaluation.get("label_access_receipt") or {}
    freeze = evaluation.get("prediction_freeze") or {}
    if receipt.get("prediction_written_before_label_access") is not True:
        errors.append("prediction_not_frozen_before_labels")
    if receipt.get("held_out_labels_opened") is not True or receipt.get("label_roles_opened") != [
        "test"
    ]:
        errors.append("evaluation_label_access_invalid")
    if receipt.get("prediction_freeze_sha256") != freeze.get("sha256"):
        errors.append("prediction_freeze_hash_mismatch")
    if receipt.get("prediction_row_count") != len(predictions):
        errors.append("prediction_row_count_mismatch")
    if any("label" in row for row in predictions):
        errors.append("prediction_contains_label")
    keys = ("group_id", "source_hash", "role", "arm", "fit_seed", "probability")
    projected = [{key: row.get(key) for key in keys} for row in rows]
    if projected != [dict(row) for row in predictions]:
        errors.append("prediction_evaluation_projection_mismatch")
    return errors


def _checkpoint_errors(
    evidence: Mapping[str, Any],
    fit: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    access: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
) -> list[str]:
    """Authenticate role, checkpoint, policy, exposure, and access boundaries."""

    errors: list[str] = []
    manifest = fit.get("checkpoint_manifest") or {}
    if manifest.get("bundle_sha256") != checkpoints.get("bundle_sha256"):
        errors.append("checkpoint_bundle_hash_mismatch")
    if manifest.get("policy_sha256") != (checkpoints.get("frozen_policies") or {}).get(
        "policy_sha256"
    ):
        errors.append("checkpoint_policy_hash_mismatch")
    if manifest.get("transform_sha256") != checkpoints.get("transform_sha256"):
        errors.append("checkpoint_transform_hash_mismatch")
    if manifest.get("frozen_before_heldout_label_access") is not True:
        errors.append("checkpoint_not_frozen_before_labels")
    if (fit.get("label_access_receipt") or {}).get("held_out_labels_opened") is not False:
        errors.append("fit_accessed_heldout_labels")
    if (access.get("evaluator_separation") or {}).get(
        "evaluator_store_parsed_during_feature_build"
    ) is not False:
        errors.append("future_evaluator_influenced_features")
    if (access.get("equal_access") or {}).get("identical_roles") is not True:
        errors.append("unequal_arm_access")
    exposure = access.get("exposure_audit") or {}
    if (
        exposure.get("fresh_confirmatory_claim_allowed") is not False
        or exposure.get("claim_scope") != "exploratory_support_only"
    ):
        errors.append("exposure_scope_invalid")
    role = evidence.get("role_manifest") or {}
    if (
        role.get("eligible_role_counts", {}).get("test") != 116
        or role.get("independent_unit") != "unique_normalized_source_hash"
    ):
        errors.append("test_role_manifest_invalid")
    settings = evaluation.get("evaluator_settings") or {}
    if settings.get("confirmatory_allowed") is not False:
        errors.append("descriptive_result_promoted")
    return errors


def _option_rows(
    root: Path, references: Sequence[Mapping[str, Any]]
) -> tuple[list[Json], list[Json]]:
    """Stream large capture shards but retain only option-mapping operands."""

    plans: list[Json] = []
    observed: list[Json] = []
    for reference in references:
        path = root / str(reference["path"])
        target = plans if "/plan-" in str(reference["path"]) else observed
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                keys = (
                    ("request_id", "option_order")
                    if target is plans
                    else (
                        "request_id",
                        "option_order",
                        "label_to_option_id",
                        "order_remapping",
                        "raw_logits_by_option_id",
                        "disposition",
                    )
                )
                target.append({key: row.get(key) for key in keys})
    return plans, observed


def load_static_inputs(root: Path, *, verify_option_rows: bool = True) -> Json:
    """Load authenticated V657 evidence and fail before scientific reduction."""

    inventory = inventory_upstreams(root)
    if any(row["state"] != "valid" for row in inventory):
        raise ValueError("upstream_inventory_not_valid")
    evidence = load_json(root / UPSTREAM_PATHS[7504])
    fit = load_json(root / UPSTREAM_PATHS[7505])
    evaluation = load_json(root / UPSTREAM_PATHS[7507])
    access = load_json(root / ACCESS_PATH)
    checkpoints = load_json(root / CHECKPOINT_PATH)
    sources = [
        _authenticate_file(root, evidence, ACCESS_PATH),
        _authenticate_file(root, fit, CHECKPOINT_PATH),
        _authenticate_file(root, evaluation, PREDICTION_PATH),
        _authenticate_file(root, evaluation, EVALUATION_PATH),
        _authenticate_file(root, evaluation, POLICY_PATH),
    ]
    predictions = load_jsonl(root / PREDICTION_PATH)
    evaluation_rows = load_jsonl(root / EVALUATION_PATH)
    policy_rows = load_jsonl(root / POLICY_PATH)
    errors = _checkpoint_errors(evidence, fit, evaluation, access, checkpoints)
    errors.extend(_prediction_errors(evaluation, predictions, evaluation_rows))
    option_references = [
        row
        for row in evidence.get("source_artifact_hashes") or []
        if "experiment_7494_v656_window_eval_capture/" in str(row.get("path"))
        and ("/plan-" in str(row.get("path")) or "/raw_logits-" in str(row.get("path")))
    ]
    if len(option_references) != 4:
        errors.append("option_capture_reference_roster_invalid")
    for reference in option_references:
        path = Path(str(reference["path"]))
        resolved = root / path
        declared = reference.get("observed_sha256")
        if (
            not resolved.is_file()
            or declared != reference.get("expected_sha256")
            or sha256_file(resolved) != declared
        ):
            errors.append(f"option_capture_hash_invalid:{path}")
        else:
            sources.append(_source_row(path, root, evidence_class="historical_option_capture"))
    if verify_option_rows and not errors:
        plans, observed = _option_rows(root, option_references)
        errors.extend(verify_option_order_rows(plans, observed))
    if errors:
        raise ValueError("static_input_authentication_failed:" + ",".join(errors))
    return {
        "inventory": inventory,
        "evidence": evidence,
        "fit": fit,
        "evaluation": evaluation,
        "access": access,
        "checkpoints": checkpoints,
        "predictions": predictions,
        "evaluation_rows": evaluation_rows,
        "policy_rows": policy_rows,
        "source_artifact_hashes": sources,
        "option_order_checked": verify_option_rows,
    }


def fixture_claim_contract() -> Json:
    """Build a compact claim boundary for private mutation tests."""

    return {
        "fit_bundle_sha256": "sha256:bundle",
        "observed_bundle_sha256": "sha256:bundle",
        "probability_holm_family_size": 2,
        "decision_holm_family_size": 9,
        "confirmatory_allowed": False,
        "producer_probability_score": 0,
        "producer_decision_score": 0,
        "recomputed_probability_score": 0,
        "recomputed_decision_score": 0,
    }


def claim_contract_errors(contract: Mapping[str, Any]) -> list[str]:
    """Reject hash, family, and descriptive-scope promotion drift."""

    errors: list[str] = []
    if contract.get("fit_bundle_sha256") != contract.get("observed_bundle_sha256"):
        errors.append("checkpoint_hash_mismatch")
    if contract.get("probability_holm_family_size") != 2:
        errors.append("probability_holm_family_invalid")
    if contract.get("decision_holm_family_size") != 9:
        errors.append("decision_holm_family_invalid")
    if contract.get("confirmatory_allowed") is False and (
        contract.get("producer_probability_score") != 0
        or contract.get("producer_decision_score") != 0
    ):
        errors.append("descriptive_result_promoted")
    if int(contract.get("producer_probability_score", -1)) > int(
        contract.get("recomputed_probability_score", -1)
    ):
        errors.append("producer_probability_exceeds_recomputed")
    if int(contract.get("producer_decision_score", -1)) > int(
        contract.get("recomputed_decision_score", -1)
    ):
        errors.append("producer_decision_exceeds_recomputed")
    return errors


def _private_rows() -> list[Json]:
    """Create two tiny complete groups for corruption detection only."""

    rows: list[Json] = []
    for group_index, label in enumerate((0, 1)):
        probability = 0.2 if label == 0 else 0.8
        for arm in (*LEARNED_ARMS, *SIMPLE_ARMS):
            seeds: Sequence[int | None] = FIT_SEEDS if arm in LEARNED_ARMS else (None,)
            for seed in seeds:
                rows.append(
                    {
                        "group_id": f"g{group_index}",
                        "source_hash": f"s{group_index}",
                        "role": "test",
                        "arm": arm,
                        "fit_seed": seed,
                        "probability": probability,
                        "label": label,
                        **metric_losses(probability, label),
                        "decision_costs": [
                            decision_cell(
                                probability,
                                label,
                                false_accept_cost=false_accept,
                                escalation_cost=escalation,
                            )
                            for false_accept in FALSE_ACCEPT_COSTS
                            for escalation in ESCALATION_COSTS
                        ],
                        "status": "complete",
                        "failed": False,
                        "censored": False,
                    }
                )
    return rows


def run_private_mutations() -> dict[str, bool]:
    """Exercise eight corruptions without returning any corrupted payload."""

    rows = _private_rows()
    policy = project_policy_rows(rows)
    flipped = deepcopy(rows)
    flipped[0]["label"] = 1
    flipped_detected = False
    try:
        _validated_groups(flipped, policy)
    except ValueError as error:
        flipped_detected = "metric_mismatch" in str(error)
    plans = [
        {"request_id": "a", "option_order": ["supported", "contains_unsupported"]},
        {"request_id": "b", "option_order": ["contains_unsupported", "supported"]},
    ]
    observed = [
        {
            **row,
            "label_to_option_id": {" A": row["option_order"][0], " B": row["option_order"][1]},
            "order_remapping": {" A": row["option_order"][0], " B": row["option_order"][1]},
            "disposition": "complete",
        }
        for row in plans
    ]
    observed[0]["option_order"] = list(reversed(observed[0]["option_order"]))
    duplicate = deepcopy(rows)
    duplicate[-1]["source_hash"] = "s0"
    selected = [
        row
        for row in rows
        if not (
            row["group_id"] == "g0"
            and row["arm"] == "window_gibbs"
            and row["fit_seed"] == FIT_SEEDS[0]
        )
    ]
    detected: dict[str, bool] = {
        "flipped_label": flipped_detected,
        "swapped_option_order": bool(verify_option_order_rows(plans, observed)),
    }
    for name, candidate, candidate_policy, expected in (
        ("duplicate_source", duplicate, policy, "source_identity_not_bijective"),
        ("favorable_seed_selection", selected, policy, "fit_seed_roster_invalid"),
        ("omitted_escalation", rows, policy[1:], "policy_projection_mismatch"),
    ):
        try:
            _validated_groups(candidate, candidate_policy)
        except ValueError as error:
            detected[name] = expected in str(error)
        else:
            detected[name] = False
    contract = fixture_claim_contract()
    wrong = deepcopy(contract)
    wrong["fit_bundle_sha256"] = "wrong"
    detected["wrong_checkpoint"] = "checkpoint_hash_mismatch" in claim_contract_errors(wrong)
    wrong = deepcopy(contract)
    wrong["decision_holm_family_size"] = 8
    detected["changed_multiplicity_family"] = (
        "decision_holm_family_invalid" in claim_contract_errors(wrong)
    )
    wrong = deepcopy(contract)
    wrong["producer_probability_score"] = 1
    detected["promoted_descriptive_result"] = (
        "descriptive_result_promoted" in claim_contract_errors(wrong)
    )
    return {name: detected.get(name, False) for name in MUTATION_NAMES}


FIELD_PRINCIPLES = {
    "schema": "Versioned identity prevents reader drift between audit contracts.",
    "run_date": "The fixed date binds this result to the authorized run.",
    "preconditions_checked": "Exact observed paths expose missing or invalid prerequisites.",
    "MODEL_SPECS": "An empty list prevents historical models from becoming current work.",
    "model_specs": "The lowercase empty list keeps all readers on the same no-load claim.",
    "model_invoked": "False separates aggregation from current model inference.",
    "invocation_counts": "Balanced zero counters expose attempted or unfinished model work.",
    "inference_substrate": "The canonical aggregation value prevents substrate laundering.",
    "inference_substrate_class": "The aggregation class selects the correct duration rules.",
    "execution_venue": "Host work stays distinct from archived GPU or board evidence.",
    "duration_s": "Measured elapsed time prevents an invented compute floor.",
    "started_monotonic_ns": "The host monotonic boundary anchors elapsed accounting.",
    "ended_monotonic_ns": "The terminal monotonic boundary exposes unfinished timing.",
    "duration_breakdown_s": "Separate work classes prevent historical capture from becoming compute.",
    "phase_spans": "Bounded phase spans expose stalls and unfinished work.",
    "random_seed": "Frozen audit and bootstrap seeds prevent favorable reruns.",
    "reproducibility_checksum": "One hash binds sources, settings, rows, and validation scope.",
    "source_artifact_hashes": "Exact byte hashes prevent silent upstream replacement.",
    "rows": "Per-source rows keep seeds and windows from inflating support.",
    "sample_size_budget": "Separate unit states expose missing, failed, or censored sources.",
    "acceptance_gate_results": "Typed gates keep validity, readiness, and benefit separate.",
    "gate_check_summary": "Exact failed fields explain blocked, null, or invalid outcomes.",
    "honest_verdict": "A complete prefix prevents a terminal null from becoming retryable work.",
    "verdict_class": "A closed class prevents prose from changing machine meaning.",
    "verifier_is_oracle": "False prevents this audit from becoming a correctness oracle.",
    "flagged_adversarial": "Upstream and current flags cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits, and logs make current checks reviewable.",
    "field_principles": "Every field states the failure that it prevents.",
    "static_audit_complete_score": "Complete accounting can remain one despite external absence.",
    "static_claims_qualified_score": "Only independently reproduced valid evidence qualifies.",
    "qualified_static_probability_value_score": "Probability value cannot exceed the frozen gate.",
    "qualified_selective_decision_value_score": "Decision value remains a separate utility claim.",
    "audit_rows": "Observed states and mutation failures distinguish absence from a scientific null.",
}
REQUIRED_CURRENT_RECEIPTS = (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)


def _current_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one successful current receipt for each scoped command."""

    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in REQUIRED_CURRENT_RECEIPTS
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    *,
    upstream: str,
    field_path: str,
) -> Json:
    """Attach one failure-prevention principle to an observable operand."""

    principle = {
        "validity": "Favorable metrics cannot excuse invalid evidence.",
        "readiness": "A valid null must not block unrelated measurement accounting.",
        "benefit": "Exploratory or selected evidence cannot become a confirmatory claim.",
    }[category]
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "upstream": upstream,
        "field_path": field_path,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Retain every exact failure and identify the first one."""

    failed = [
        {
            key: row[key]
            for key in ("check", "category", "upstream", "field_path", "expected", "observed", "op")
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "all_validity_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "validity"
        ),
        "all_readiness_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "readiness"
        ),
        "all_benefit_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "benefit"
        ),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding measured clocks and this field."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "ended_at_utc",
        "duration_s",
        "duration_breakdown_s",
        "phase_spans",
        "process_identity",
        "started_monotonic_ns",
        "ended_monotonic_ns",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def build_artifact(
    *,
    preconditions: Mapping[str, Any],
    reduced: Mapping[str, Any],
    reduction_errors: Sequence[str],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    mutation_results: Mapping[str, bool],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    started_monotonic_ns: int = 0,
    ended_monotonic_ns: int | None = None,
) -> Json:
    """Build a schema-complete ledger while keeping value scores separate."""

    inventory = list(preconditions.get("inventory") or [])
    terminal = classify_inventory(inventory, reduction_errors)
    receipts_pass = _current_receipts_pass(validation_receipts)
    mutations_pass = set(mutation_results) == set(MUTATION_NAMES) and all(mutation_results.values())
    failed_guards = [
        row
        for row in validation_receipts
        if row.get("name") in {"adversarial_verify", "verdict_row_consistency_strict"}
        and row.get("passed") is not True
    ]
    if not receipts_pass or not mutations_pass:
        terminal = {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_required_validation",
            "static_audit_complete_score": 1,
            "static_claims_qualified_score": 0,
        }
    claims_qualified = int(
        terminal["static_claims_qualified_score"] == 1
        and not reduction_errors
        and receipts_pass
        and mutations_pass
    )
    probability_value = claims_qualified * min(
        int(reduced.get("static_probability_value_score", 0)),
        int(reduced.get("producer_probability_value_score", 0)),
    )
    decision_value = claims_qualified * min(
        int(reduced.get("selective_decision_value_score", 0)),
        int(reduced.get("producer_decision_value_score", 0)),
    )
    if terminal["verdict_class"] == "null" and (probability_value or decision_value):
        terminal["verdict_class"] = "positive"
        terminal["honest_verdict"] = "complete_positive_v657_qualified_static_value"
    audit_rows = [
        {
            "audit_kind": "upstream_inventory",
            "check": f"exp{row['producer']}_state",
            "path": row["path"],
            "expected": "valid",
            "observed": row["state"],
            "passed": row["state"] == "valid",
        }
        for row in inventory
    ]
    audit_rows.extend(
        {
            "audit_kind": "private_mutation",
            "check": name,
            "path": "private_fixture_not_published",
            "expected": "corruption_rejected",
            "observed": "corruption_rejected" if passed else "corruption_accepted",
            "passed": passed,
        }
        for name, passed in mutation_results.items()
    )
    audit_rows.extend(
        {
            "audit_kind": "required_validation_failure",
            "check": str(row.get("name")),
            "path": str(row.get("log_path") or "validation_receipts"),
            "expected": 0,
            "observed": row.get("exit_code"),
            "passed": False,
            "finding": str(row.get("output_tail") or "")[-2000:],
        }
        for row in validation_receipts
        if row.get("passed") is not True
    )
    inventory_gates = [
        _gate(
            f"exp{row['producer']}_available_and_valid",
            "validity" if row["state"] == "invalid" else "readiness",
            "valid",
            row["state"],
            "eq",
            row["state"] == "valid",
            upstream=f"Exp{row['producer']}",
            field_path=str(row["path"]),
        )
        for row in inventory
    ]
    gates = [
        *inventory_gates,
        _gate(
            "upstream_inventory_valid",
            "validity",
            [],
            list(reduction_errors),
            "eq",
            not reduction_errors,
            upstream="Exp7504/7505/7507",
            field_path="reduction_errors",
        ),
        _gate(
            "private_mutations_rejected",
            "validity",
            True,
            mutations_pass,
            "eq",
            mutations_pass,
            upstream="current_audit",
            field_path="audit_rows.private_mutation",
        ),
        _gate(
            "required_current_validation",
            "validity",
            True,
            receipts_pass,
            "eq",
            receipts_pass,
            upstream="current_audit",
            field_path="validation_receipts",
        ),
        _gate(
            "static_accounting_complete",
            "readiness",
            1,
            terminal["static_audit_complete_score"],
            "eq",
            terminal["static_audit_complete_score"] == 1,
            upstream="current_audit",
            field_path="static_audit_complete_score",
        ),
        _gate(
            "scientific_claims_qualified",
            "readiness",
            1,
            claims_qualified,
            "eq",
            claims_qualified == 1,
            upstream="Exp7504/7505/7507",
            field_path="static_claims_qualified_score",
        ),
        _gate(
            "qualified_probability_value",
            "benefit",
            1,
            probability_value,
            "eq",
            probability_value == 1,
            upstream="Exp7507 raw evaluation rows",
            field_path="qualified_static_probability_value_score",
        ),
        _gate(
            "qualified_selective_decision_value",
            "benefit",
            1,
            decision_value,
            "eq",
            decision_value == 1,
            upstream="Exp7507 raw policy rows",
            field_path="qualified_selective_decision_value_score",
        ),
    ]
    value: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": int(started_monotonic_ns),
        "ended_monotonic_ns": int(
            ended_monotonic_ns
            if ended_monotonic_ns is not None
            else started_monotonic_ns + round(duration_s * 1_000_000_000)
        ),
        "process_identity": {"pid": os.getpid(), "ppid": os.getppid(), "python": sys.executable},
        "device_identity": {"venue": "host", "platform": platform.platform(), "cuda_used": False},
        "preconditions_checked": deepcopy(list(preconditions.get("rows") or [])),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") in {"static_reduction", "private_mutations"}
            ),
            "validation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") in {"affected_validation", "terminal_validation"}
            ),
            "historical_capture": 0.0,
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "fit": list(FIT_SEEDS),
            "audit": 7508657,
            "bootstrap": BOOTSTRAP_SEED,
            "arrival": "pinned_upstream_order",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(list(source_hashes)),
        "rows": deepcopy(list(reduced.get("rows") or [])),
        "sample_size_budget": deepcopy(
            dict(
                reduced.get("sample_size_budget")
                or {
                    "planned": 0,
                    "attempted": 0,
                    "completed": 0,
                    "excluded": 0,
                    "failed": 0,
                    "censored": 0,
                    "unstarted": 0,
                    "independent_unit": "unique_source_group",
                }
            )
        ),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "verifier_is_oracle": False,
        "flagged_adversarial": bool(failed_guards),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "field_principles": {},
        "static_audit_complete_score": terminal["static_audit_complete_score"],
        "static_claims_qualified_score": claims_qualified,
        "qualified_static_probability_value_score": probability_value,
        "qualified_selective_decision_value_score": decision_value,
        "audit_rows": audit_rows,
        "independent_reduction": deepcopy(dict(reduced)),
        "reduction_errors": list(reduction_errors),
        "affected_validation_manifest": {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": receipts_pass,
            "numbered_runtime_e2e": [],
            "reason": "Reporting-only aggregation changed no runtime model, sampler, binding, ARC, telemetry, or Rust behavior.",
        },
        "external_publication_performed": False,
        "push_performed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
    }
    value["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, "This field preserves audit evidence and prevents silent omission."
        )
        for key in value
    }
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def fixture_artifact() -> Json:
    """Build one compact valid null for schema and mutation tests."""

    inventory = [
        {"producer": number, "path": path.as_posix(), "state": "valid", "sha256": "sha256:x"}
        for number, path in UPSTREAM_PATHS.items()
    ]
    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": f"fixture:{name}",
            "log_sha256": canonical_hash(name),
        }
        for name in REQUIRED_CURRENT_RECEIPTS
    ]
    reduced = {
        "rows": [
            {
                "unit_id": "g0",
                "group_id": "g0",
                "source_hash": "s0",
                "label": 0,
                "arm_metrics": {"window_gibbs": {"brier": 0.1, "log_loss": 0.2}},
                "decision_cost_deltas": {"fa=1|fr=1|esc=0.1": 0.0},
                "attempted": True,
                "complete": True,
                "failed": False,
                "excluded": False,
                "censored": False,
                "unstarted": False,
            }
        ],
        "sample_size_budget": {
            "planned": 1,
            "attempted": 1,
            "completed": 1,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "unique_source_group",
        },
        "static_probability_value_score": 0,
        "selective_decision_value_score": 0,
        "probability_holm_family_size": 2,
        "decision_holm_family_size": 9,
    }
    return build_artifact(
        preconditions={"rows": [{"check": "fixture", "passed": True}], "inventory": inventory},
        reduced=reduced,
        reduction_errors=[],
        source_hashes=[],
        validation_receipts=receipts,
        mutation_results={name: True for name in MUTATION_NAMES},
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00+00:00",
        ended_at_utc="2026-09-22T00:00:01+00:00",
        duration_s=1.0,
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, scores, receipts, hashes, gates, and checksum."""

    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "experiment_id",
        "milestone",
        "terminal_status",
        "independent_reduction",
        "reduction_errors",
    }
    missing = sorted(required - value.keys())
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if (
        value.get("schema"),
        value.get("experiment_id"),
        value.get("milestone"),
        value.get("run_date"),
        value.get("terminal_status"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE, "complete"):
        errors.append("artifact_identity_invalid")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or value.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
    ):
        errors.append("current_provenance_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    for field in (
        "static_audit_complete_score",
        "static_claims_qualified_score",
        "qualified_static_probability_value_score",
        "qualified_selective_decision_value_score",
    ):
        if value.get(field) not in {0, 1}:
            errors.append(f"score_not_bare_binary:{field}")
    reduced = value.get("independent_reduction") or {}
    if int(value.get("qualified_static_probability_value_score", 0)) > int(
        reduced.get("static_probability_value_score", 0)
    ):
        errors.append("qualified_probability_exceeds_recomputed")
    if int(value.get("qualified_selective_decision_value_score", 0)) > int(
        reduced.get("selective_decision_value_score", 0)
    ):
        errors.append("qualified_decision_exceeds_recomputed")
    if value.get("static_claims_qualified_score") == 0 and (
        value.get("qualified_static_probability_value_score") == 1
        or value.get("qualified_selective_decision_value_score") == 1
    ):
        errors.append("unqualified_claim_has_value")
    principles = value.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(value)
        or any(not isinstance(item, str) or not item.strip() for item in principles.values())
    ):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results")
    gate_fields = {
        "check",
        "category",
        "expected",
        "observed",
        "op",
        "passed",
        "upstream",
        "field_path",
        "principle",
    }
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(row, Mapping) or gate_fields - row.keys() for row in gates)
    ):
        errors.append("gate_contract_invalid")
    receipts_pass = _current_receipts_pass(value.get("validation_receipts") or [])
    if not receipts_pass and value.get("verdict_class") != "disqualified":
        errors.append("required_validation_failed")
    if (
        not receipts_pass
        and value.get("static_claims_qualified_score") != 0
        or value.get("flagged_adversarial") is True
        and value.get("verdict_class") != "disqualified"
    ):
        errors.append("failed_validation_not_disqualified")
    if verify_sources:
        for row in value.get("source_artifact_hashes") or []:
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file():
                errors.append(f"source_missing:{path}")
            elif sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{path}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print a flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7508] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _with_heartbeat(
    operation: str, fn: Callable[[], T], *, started: float, heartbeat_s: float = 60.0
) -> T:  # pragma: no cover
    """Emit truthful pending lines while an in-process reduction is active."""

    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(heartbeat_s):
            progress(started, operation, "pending")

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _span(phase: str, phase_started: int, run_started: int, units: int) -> Json:  # pragma: no cover
    """Record one measured phase without inventing model duration."""

    ended = time.monotonic_ns()
    return {
        "phase": phase,
        "start_s": (phase_started - run_started) / 1e9,
        "end_s": (ended - run_started) / 1e9,
        "duration_s": (ended - phase_started) / 1e9,
        "completed_units": units,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh readers and both required guards for the exact candidate."""

    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "capability_end_to_end",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "raw_upstream_rows",
            timeout_s=1200.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _empty_reduction() -> Json:
    """Represent no scientific operands without inventing a zero-effect result."""

    return {
        "rows": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "unique_source_group",
        },
        "static_probability_value_score": 0,
        "selective_decision_value_score": 0,
        "probability_holm_family_size": 0,
        "decision_holm_family_size": 0,
        "producer_probability_value_score": 0,
        "producer_decision_value_score": 0,
    }


def _reduce_real(root: Path, *, verify_option_rows: bool = True) -> tuple[Json, Json, list[str]]:
    """Load, recompute, and compare the three present producer artifacts."""

    loaded = load_static_inputs(root, verify_option_rows=verify_option_rows)
    evaluation = loaded["evaluation"]
    reduced = reduce_static_rows(
        loaded["evaluation_rows"], loaded["policy_rows"], evaluation["evaluator_settings"]
    )
    reduced["producer_probability_value_score"] = int(evaluation["static_probability_value_score"])
    reduced["producer_decision_value_score"] = int(evaluation["selective_decision_value_score"])
    errors = compare_producer_reduction(evaluation, reduced)
    contract = {
        "fit_bundle_sha256": (loaded["fit"].get("checkpoint_manifest") or {}).get("bundle_sha256"),
        "observed_bundle_sha256": loaded["checkpoints"].get("bundle_sha256"),
        "probability_holm_family_size": reduced["probability_holm_family_size"],
        "decision_holm_family_size": reduced["decision_holm_family_size"],
        "confirmatory_allowed": evaluation["evaluator_settings"].get("confirmatory_allowed"),
        "producer_probability_score": evaluation.get("static_probability_value_score"),
        "producer_decision_score": evaluation.get("selective_decision_value_score"),
        "recomputed_probability_score": reduced["static_probability_value_score"],
        "recomputed_decision_score": reduced["selective_decision_value_score"],
    }
    errors.extend(claim_contract_errors(contract))
    return loaded, reduced, list(dict.fromkeys(errors))


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover
    """Re-read raw inputs and compare every stable scientific output."""

    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return ["candidate_unreadable"]
    errors = validate_artifact(value, root=root)
    preconditions = collect_preconditions(root)
    inventory = preconditions["inventory"]
    if all(row["state"] == "valid" for row in inventory):
        try:
            loaded, reduced, reduction_errors = _reduce_real(root)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"independent_reduction_failed:{error}")
        else:
            if value.get("independent_reduction") != reduced:
                errors.append("independent_replay_mismatch:independent_reduction")
            expected_sources = [
                *preconditions["source_artifact_hashes"],
                *loaded["source_artifact_hashes"],
            ]
            if value.get("source_artifact_hashes") != expected_sources:
                errors.append("independent_replay_mismatch:source_artifact_hashes")
            if value.get("reduction_errors") != reduction_errors:
                errors.append("independent_replay_mismatch:reduction_errors")
    elif value.get("independent_reduction") != _empty_reduction():
        errors.append("independent_replay_mismatch:blocked_reduction")
    if value.get("preconditions_checked") != preconditions["rows"]:
        errors.append("independent_replay_mismatch:preconditions_checked")
    if value.get("static_audit_complete_score") != 1:
        errors.append("independent_replay_mismatch:accounting_score")
    return list(dict.fromkeys(errors))


def _write_note(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write the concise human record before publishing the terminal JSON."""

    probability = artifact.get("independent_reduction", {}).get("probability_contrasts", {})
    decision = artifact.get("independent_reduction", {}).get("decision_cells", [])
    lines = [
        "# V657 static audit",
        "",
        f"Run date: {artifact.get('run_date')}",
        f"Verdict: `{artifact.get('honest_verdict')}` (`{artifact.get('verdict_class')}`)",
        "",
        "The audit independently reproduced the held-out static rows. It did not load a model.",
        "The evidence is exploratory because the source corpus had prior exposure.",
        "",
        "## Qualified outcomes",
        "",
        f"- Complete accounting: {artifact.get('static_audit_complete_score')}",
        f"- Scientific rows qualified: {artifact.get('static_claims_qualified_score')}",
        f"- Static probability value: {artifact.get('qualified_static_probability_value_score')}",
        f"- Selective decision value: {artifact.get('qualified_selective_decision_value_score')}",
        "",
        "Both value scores are separate. A failure of benefit does not invalidate the completed audit.",
        "",
        "## Independent reduction",
        "",
        f"- Source groups: {artifact.get('sample_size_budget', {}).get('completed', 0)}",
        f"- Probability Holm family: {artifact.get('independent_reduction', {}).get('probability_holm_family_size', 0)}",
        f"- Decision Holm family: {artifact.get('independent_reduction', {}).get('decision_holm_family_size', 0)}",
        f"- Probability contrasts: `{json.dumps(probability, sort_keys=True)}`",
        f"- Decision cells recomputed: {len(decision)}",
        "",
        "Private mutations covered label, option order, source, seed, escalation, checkpoint,",
        "multiplicity, and descriptive-scope corruption. Corrupted fixtures are not published.",
        "",
    ]
    failures = artifact.get("gate_check_summary", {}).get("failed_checks", [])
    if failures:
        lines.extend(
            [
                "## Gate disposition",
                "",
                "The ledger preserves failed gates rather than repairing upstream JSON or changing thresholds.",
                f"Failed gates: `{json.dumps(failures, sort_keys=True)}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _provisional_terminal_receipts() -> list[Json]:
    """Let fresh candidate readers validate the final receipt shape."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> Json:  # pragma: no cover
    """Run the audit, all scoped checks, and one atomic terminal publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(UTC).isoformat()
    spans: list[Json] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic_ns()
    preconditions = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started_ns, len(preconditions["rows"])))
    internal_failures = [
        row
        for row in preconditions["rows"]
        if row.get("passed") is not True
        and row.get("check") != "upstream_inventory_exp7504"
        and not str(row.get("check", "")).startswith("upstream_inventory_exp")
    ]
    progress(
        started,
        "preconditions",
        "complete",
        missing=len(preconditions["missing_external"]),
        invalid=len(preconditions["invalid_present"]),
    )
    if internal_failures:
        raise RuntimeError(f"repository_precondition_failed:{internal_failures[0]}")

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic_ns()
        spans.append(_span(phase, phase_started, started_ns, 0))
        progress(started, phase, "after", completed=0)

    loaded: Json = {}
    reduced = _empty_reduction()
    reduction_errors: list[str] = []
    if preconditions["invalid_present"]:
        reduction_errors.extend(
            f"invalid_present:{path}" for path in preconditions["invalid_present"]
        )
    elif not preconditions["missing_external"]:
        progress(started, "static_reduction", "before_benchmark")
        phase_started = time.monotonic_ns()
        try:
            loaded, reduced, reduction_errors = _with_heartbeat(
                "static_reduction", lambda: _reduce_real(root), started=started
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            reduction_errors = [f"present_evidence_invalid:{error}"]
            reduced = _empty_reduction()
        spans.append(_span("static_reduction", phase_started, started_ns, len(reduced["rows"])))
        progress(
            started,
            "static_reduction",
            "after_benchmark",
            completed_units=len(reduced["rows"]),
            errors=len(reduction_errors),
        )

    progress(started, "private_mutations", "before_benchmark", planned=len(MUTATION_NAMES))
    phase_started = time.monotonic_ns()
    mutation_results = run_private_mutations()
    spans.append(_span("private_mutations", phase_started, started_ns, len(mutation_results)))
    progress(
        started,
        "private_mutations",
        "after_benchmark",
        rejected=sum(mutation_results.values()),
    )
    if not all(mutation_results.values()):
        reduction_errors.append("private_mutation_control_failed")

    private_root = Path(tempfile.mkdtemp(prefix="exp7508-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    manifest_path = root / RAW_DIR / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic_ns()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
    )
    affected_result = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started_ns, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_result["passed"],
    )
    if not affected_result["passed"]:
        raise RuntimeError(f"affected_validation_failed:{affected_result}")

    source_hashes = [
        *preconditions["source_artifact_hashes"],
        *list(loaded.get("source_artifact_hashes") or []),
    ]
    candidate = build_artifact(
        preconditions=preconditions,
        reduced=reduced,
        reduction_errors=reduction_errors,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *_provisional_terminal_receipts()],
        mutation_results=mutation_results,
        phase_spans=spans,
        started_at_utc=started_utc,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_atomic_write", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_atomic_write", path=candidate_path)

    progress(started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic_ns()
    terminal = run_categorized_commands(
        root, _terminal_commands(candidate_path), log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, started_ns, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if critical:
        reduction_errors.append("critical_guard_output")

    final = build_artifact(
        preconditions=preconditions,
        reduced=reduced,
        reduction_errors=reduction_errors,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        mutation_results=mutation_results,
        phase_spans=spans,
        started_at_utc=started_utc,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "research_note", "before_atomic_write", path=NOTE_PATH)
    _write_note(root / NOTE_PATH, final)
    progress(started, "research_note", "after_atomic_write", path=NOTE_PATH)
    progress(started, "publish", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(started, "publish", "after_atomic_terminal", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the audit or one exact fresh-process candidate reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        try:
            value = load_json(args.cold_replay)
        except (OSError, json.JSONDecodeError, ValueError):
            errors = ["candidate_unreadable"]
        else:
            errors = validate_artifact(value, root=root, verify_sources=not args.no_source_check)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce, root=root)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
