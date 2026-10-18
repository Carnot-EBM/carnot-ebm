"""Evaluate a frozen learning protocol on exposed cached Qwen evidence.

The module reads historical native option logits and real evaluator labels. It
does not load a model. All effects are descriptive because the corpus was
examined before this protocol was registered.

Spec refs: REQ-CL-7575 and SCENARIO-CL-7575-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7534_v659_count_memory import CountArm, CountConfig, bin_index
from carnot.experiment_7561_v661_recalibration_prototype import (
    KNOTS,
    SolverConfig,
    SufficientStatisticMap,
    map_probability,
    solve_constrained_map,
    solve_unconstrained_map,
    statistics_from_examples,
    typed_decision,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7575-v662-cached-learning-protocol"
SCHEMA = "carnot.exp7575.v662.cached_learning_protocol.v1"
RESULT_PATH = Path("results/experiment_7575_v662_cached_learning_protocol.json")
RAW_DIR = Path("results/raw/experiment_7575_v662_cached_learning_protocol")
MODULE_PATH = Path("python/carnot/experiment_7575_v662_cached_learning_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7575_v662_cached_learning_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7575_v662_cached_learning_protocol.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
FIT_CAPTURE_PATH = Path("results/experiment_7564_v661_fit_capture.json")
EVAL_CAPTURE_PATH = Path("results/experiment_7565_v661_test_online_capture.json")
PROTOCOL_SOURCE_PATH = Path("results/experiment_7533_v659_tool_protocol.json")
SOURCE_EVALUATION_PATH = Path("results/experiment_7567_v661_source_evaluation.json")
PINNED_DATASET_REVISION = "866a7c5392c3cf87e4fbc2b3808815d524f54331"
PINNED_DATASET_ID = "KRLabsOrg/lettucedetect-code-hallucination"
HISTORICAL_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
ROLE_COUNTS = {"fit": 160, "tune": 40, "policy": 40, "online": 160, "test": 80}
ORDER_SEEDS = (7_578_001, 7_578_002, 7_578_003, 7_578_004, 7_578_005)
ONLINE_ARMS = ("bounded", "raw", "global_count", "local_count", "shuffled_feedback")
ZERO_INVOCATION_COUNTS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
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
    "verifier_is_oracle",
    "cached_roles_ready_score",
    "online_protocol_ready_score",
    "fresh_confirmatory_claim_allowed",
    "protocol_sha256",
    "historical_model_id",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def sigmoid(value: float) -> float:
    """Convert one finite logit difference to a stable probability."""

    if not math.isfinite(value):
        raise ValueError("logit_not_finite")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _semantic_logit(row: Mapping[str, Any]) -> float:
    """Authenticate display order before returning the unsupported logit."""

    order = row.get("option_order")
    display = row.get("display_logits")
    semantic = row.get("full_logits_by_option_id")
    if not isinstance(order, list) or set(order) != {"supported", "contains_unsupported"}:
        raise ValueError("option_mapping_invalid")
    if not isinstance(display, Mapping) or not isinstance(semantic, Mapping):
        raise ValueError("option_mapping_invalid")
    mapped = dict(zip(order, (float(display.get(" A")), float(display.get(" B"))), strict=True))
    expected = {name: float(semantic.get(name)) for name in order}
    if any(not math.isclose(mapped[name], expected[name], abs_tol=1e-12) for name in order):
        raise ValueError("option_mapping_invalid")
    return mapped["contains_unsupported"] - mapped["supported"]


def reduce_native_group(
    native_rows: Sequence[Mapping[str, Any]], label_row: Mapping[str, Any], official_split: str
) -> JsonDict:
    """Reduce six complete native cells without using diagnostics as truth."""

    if len(native_rows) != 6:
        raise ValueError("native_group_incomplete")
    identities = {
        (str(row.get("component_hash")), str(row.get("group_hash")), str(row.get("role")))
        for row in native_rows
    }
    if len(identities) != 1:
        raise ValueError("native_group_identity_invalid")
    source_id, group_id, role = next(iter(identities))
    if label_row.get("role") != role or label_row.get("label") not in (0, 1):
        raise ValueError("label_identity_invalid")
    label_component = label_row.get("component_hash")
    if label_component is not None and str(label_component) != source_id:
        raise ValueError("label_identity_invalid")
    if any(
        row.get("disposition") != "complete"
        or row.get("generated_tokens") != 0
        or row.get("readout_kind") != "option_logits"
        for row in native_rows
    ):
        raise ValueError("native_group_incomplete")
    cells: dict[tuple[tuple[str, ...], str], float] = {}
    for row in native_rows:
        order = tuple(str(item) for item in row.get("option_order", []))
        condition = str(row.get("condition"))
        key = (order, condition)
        if key in cells or condition not in {"original", "absent", "mismatched"}:
            raise ValueError("native_cell_duplicate_or_invalid")
        cells[key] = _semantic_logit(row)
    orders = {
        ("supported", "contains_unsupported"),
        ("contains_unsupported", "supported"),
    }
    if set(cells) != {
        (order, condition) for order in orders for condition in ("original", "absent", "mismatched")
    }:  # pragma: no cover - six unique allowed cells prove equality.
        raise ValueError("native_cells_invalid")
    probabilities = {
        condition: float(np.mean([sigmoid(cells[(order, condition)]) for order in orders]))
        for condition in ("original", "absent", "mismatched")
    }
    context_map = {
        f"{row['condition']}|{'-'.join(str(item) for item in row['option_order'])}": str(
            row.get("complete_source_window") or ""
        )
        for row in native_rows
    }
    if len(context_map) != 6 or any(not value for value in context_map.values()):
        raise ValueError("whole_context_invalid")
    responses = {str(row.get("complete_response_window") or "") for row in native_rows}
    if len(responses) != 1:
        raise ValueError("whole_response_invalid")
    context = context_map["original|supported-contains_unsupported"]
    response = next(iter(responses))
    return {
        "source_id": source_id,
        "group_id": group_id,
        "role": role,
        "official_split": official_split,
        "probability": probabilities["original"],
        "label": int(label_row["label"]),
        "context": context,
        "contexts_by_condition": context_map,
        "response": response,
        "context_sha256": canonical_hash(context),
        "response_sha256": canonical_hash(response),
        "request_hashes": sorted(str(row.get("request_id")) for row in native_rows),
        "prompt_hashes": sorted(str(row.get("prompt_sha256")) for row in native_rows),
        "source_removal_probability": probabilities["absent"],
        "source_mismatch_probability": probabilities["mismatched"],
        "diagnostics_are_labels": False,
    }


def build_exposure_manifest(
    roles: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_counts: Mapping[str, int] = ROLE_COUNTS,
) -> JsonDict:
    """Preserve every prior exposure and official role without reinterpretation."""

    groups: list[JsonDict] = []
    seen: set[str] = set()
    for role, expected in expected_counts.items():
        rows = list(roles.get(role, []))
        if len(rows) != expected:
            raise ValueError(f"role_count_invalid:{role}")
        for row in rows:
            source_id = str(row.get("source_id") or "")
            if not source_id or source_id in seen:
                raise ValueError("source_id_duplicate")
            seen.add(source_id)
            if row.get("role") != role or not row.get("official_split"):
                raise ValueError(f"role_identity_invalid:{role}")
            groups.append(
                {
                    "source_id": source_id,
                    "role": role,
                    "official_split": row["official_split"],
                    "context_sha256": row["context_sha256"],
                    "historically_exposed": True,
                }
            )
    groups.sort(key=lambda row: (str(row["role"]), str(row["source_id"])))
    return {
        "dataset_id": PINNED_DATASET_ID,
        "dataset_revision": PINNED_DATASET_REVISION,
        "groups": groups,
        "manifest_sha256": canonical_hash(groups),
        "all_candidate_groups_previously_exposed": True,
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse",
        "runtime_label_isolation_erases_historical_exposure": False,
    }


def freeze_protocol(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Freeze identities, orders, costs, controls, and gates before fitting."""

    online_ids = [str(row["source_id"]) for row in roles["online"]]
    retention_ids = [str(row["source_id"]) for row in roles["test"]]
    orders: dict[str, list[str]] = {}
    for seed in ORDER_SEEDS:
        order = online_ids.copy()
        random.Random(seed).shuffle(order)
        orders[str(seed)] = order
    payload: JsonDict = {
        "role_counts": {name: len(roles[name]) for name in ROLE_COUNTS},
        "role_ids": {name: [str(row["source_id"]) for row in roles[name]] for name in ROLE_COUNTS},
        "order_seeds": list(ORDER_SEEDS),
        "orders": orders,
        "feedback_delay": 8,
        "release_block_size": 8,
        "retention_ids": retention_ids,
        "controls": [
            "raw",
            "temperature",
            "unconstrained_nine_knot",
            "global_count",
            "local_count",
            "shuffled_feedback",
        ],
        "costs": {"accept": "5q", "reject": "1-q", "escalate": 0.2, "tie_breaker": "escalate"},
        "minimum_non_escalation_fraction": 0.10,
        "bootstrap_replays_per_order": 1000,
        "resampling_unit": "source_component",
        "holm_adjusted_static_contrasts": True,
        "retention_max_upper95_degradation": 0.005,
        "stopping_rule": "all_five_orders_and_all_registered_replays",
    }
    return {
        **payload,
        "hash_payload": deepcopy(payload),
        "protocol_sha256": canonical_hash(payload),
    }


def _brier(probability: float, label: int) -> float:
    """Return one binary squared error after strict input checks."""

    value = float(probability)
    if label not in (0, 1) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability_or_label_invalid")
    return (value - label) ** 2


def _temperature_probability(probability: float, temperature: float) -> float:
    """Apply one positive scalar temperature to a probability."""

    value = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return sigmoid(math.log(value / (1.0 - value)) / temperature)


def _mean_brier(rows: Sequence[Mapping[str, Any]], predictor: Any) -> float:
    return float(np.mean([_brier(float(predictor(row)), int(row["label"])) for row in rows]))


def fit_static_controls(
    fit_rows: Sequence[Mapping[str, Any]],
    tune_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Fit registered controls in role order and select only on policy labels."""

    for name, rows in (("fit", fit_rows), ("tune", tune_rows), ("policy", policy_rows)):
        if not rows or any(row.get("role") != name for row in rows):
            raise ValueError(f"static_role_invalid:{name}")
    fit_probabilities = [float(row["probability"]) for row in fit_rows]
    fit_labels = [int(row["label"]) for row in fit_rows]
    gram, target = statistics_from_examples(fit_probabilities, fit_labels)
    temperatures = np.geomspace(0.25, 4.0, 65)
    selected_temperature = min(
        temperatures,
        key=lambda value: _mean_brier(
            tune_rows, lambda row: _temperature_probability(float(row["probability"]), float(value))
        ),
    )
    tune_gram, tune_target = statistics_from_examples(
        [float(row["probability"]) for row in tune_rows],
        [int(row["label"]) for row in tune_rows],
    )
    combined_gram = gram + tune_gram
    combined_target = target + tune_target
    bounded_theta, bounded_receipt = solve_constrained_map(combined_gram, combined_target)
    unconstrained_theta, unconstrained_receipt = solve_unconstrained_map(gram, target)
    predictors = {
        "raw": lambda row: float(row["probability"]),
        "temperature": lambda row: _temperature_probability(
            float(row["probability"]), float(selected_temperature)
        ),
        "unconstrained_nine_knot": lambda row: map_probability(
            float(row["probability"]), unconstrained_theta
        ),
    }
    tune_losses = {
        name: _mean_brier(tune_rows, predictor) for name, predictor in predictors.items()
    }
    strongest = min(tune_losses, key=tune_losses.get)
    policy_comparison_rows = [
        comparison_row(
            str(row["source_id"]),
            arm,
            predictor(row),
            int(row["label"]),
            seed=0,
            phase="static",
        )
        for row in policy_rows
        for arm, predictor in {
            **predictors,
            "bounded": lambda item: map_probability(float(item["probability"]), bounded_theta),
        }.items()
    ]
    count_config = _count_config(fit_rows)
    return {
        "temperature": {
            "value": float(selected_temperature),
            "selected_on_role": "tune",
            "grid_size": len(temperatures),
        },
        "bounded": {
            "theta": bounded_theta.tolist(),
            "gram": combined_gram.tolist(),
            "target": combined_target.tolist(),
            "sample_count": len(fit_rows) + len(tune_rows),
            "processed_event_ids": [str(row["source_id"]) for row in (*fit_rows, *tune_rows)],
            "fit_roles": ["fit", "tune"],
            "solver_receipt": bounded_receipt,
        },
        "unconstrained_nine_knot": {
            "theta": unconstrained_theta.tolist(),
            "capacity": 9,
            "fit_role": "fit",
            "solver_receipt": unconstrained_receipt,
        },
        "count_config": {
            "bin_means": list(count_config.bin_means),
            "global_mean": count_config.global_mean,
            "kappa": count_config.kappa,
        },
        "strongest_control": {
            "arm": strongest,
            "tune_brier": tune_losses[strongest],
            "all_tune_brier": tune_losses,
            "selected_on_role": "tune",
        },
        "protocol_frozen_before_policy_labels": True,
        "static_rows_by_role": {"policy": policy_comparison_rows},
    }


def comparison_row(
    unit_id: str,
    arm: str,
    probability: float,
    label: int,
    *,
    seed: int,
    phase: str,
    censored: bool = False,
) -> JsonDict:
    """Build one independently reducible proper-loss and decision row."""

    brier = _brier(probability, label)
    decision = typed_decision(probability)
    action = str(decision["action"])
    realized_cost = {
        "accept": 5.0 if label else 0.0,
        "reject": 0.0 if label else 1.0,
        "escalate": 0.2,
    }[action]
    return {
        "unit_id": unit_id,
        "arm": arm,
        "phase": phase,
        "probability": float(probability),
        "label": int(label),
        "raw_squared_error_numerator": brier,
        "raw_squared_error_denominator": 1,
        "brier": brier,
        "action": action,
        "realized_cost": realized_cost,
        "non_escalated": action != "escalate",
        "metric_direction": "lower_brier_and_cost_are_better",
        "seed": int(seed),
        "censored": bool(censored),
        "provenance": "cached_authentic_qwen_option_logits",
    }


def _count_config(fit_rows: Sequence[Mapping[str, Any]]) -> CountConfig:
    """Freeze eight label-free bin means from fit-role probabilities."""

    values = [min(0.9999, max(0.0001, float(row["probability"]))) for row in fit_rows]
    bins: list[list[float]] = [[] for _ in range(8)]
    for value in values:
        bins[bin_index(value)].append(value)
    means = tuple(
        float(np.mean(bucket)) if bucket else (index + 0.5) / 8 for index, bucket in enumerate(bins)
    )
    return CountConfig(means, float(np.mean(values)))


def seeded_bounded_learner(static: Mapping[str, Any]) -> SufficientStatisticMap:
    """Restore the fit-and-tune bounded map without retaining raw examples."""

    value = static["bounded"]
    return SufficientStatisticMap(
        np.asarray(value["gram"], dtype=float),
        np.asarray(value["target"], dtype=float),
        np.asarray(value["theta"], dtype=float),
        int(value["sample_count"]),
        {str(item) for item in value["processed_event_ids"]},
        SolverConfig(),
        True,
        value["solver_receipt"],
    )


def _state_payload(
    bounded: SufficientStatisticMap,
    shuffled: SufficientStatisticMap,
    global_arm: CountArm,
    local_arm: CountArm,
) -> JsonDict:
    return {
        "bounded": bounded.to_payload(),
        "shuffled": shuffled.to_payload(),
        "global": global_arm.to_payload(),
        "local": local_arm.to_payload(),
    }


def replay_order(
    online_rows: Sequence[Mapping[str, Any]],
    order: Sequence[str],
    static: Mapping[str, Any],
    *,
    seed: int,
    state_path: Path,
) -> JsonDict:
    """Predict, release, update, persist, and reload one registered order."""

    by_id = {str(row["source_id"]): row for row in online_rows}
    if len(by_id) != len(online_rows) or set(order) != set(by_id):
        raise ValueError("online_order_roster_invalid")
    count_payload = static["count_config"]
    config = CountConfig(
        tuple(float(value) for value in count_payload["bin_means"]),
        float(count_payload["global_mean"]),
        float(count_payload["kappa"]),
    )
    bounded = seeded_bounded_learner(static)
    shuffled = seeded_bounded_learner(static)
    global_arm = CountArm.create("global", config)
    local_arm = CountArm.create("local", config)
    rows: list[JsonDict] = []
    releases: list[JsonDict] = []
    chronology_violations = 0
    restart_mismatches = 0
    blocks = [
        {
            "block_id": start // 8,
            "event_ids": list(order[start : start + 8]),
            "release_time": start + len(order[start : start + 8]) - 1 + 8,
        }
        for start in range(0, len(order), 8)
    ]
    due_by_time = {
        block["release_time"]: block for block in blocks if block["release_time"] < len(order)
    }
    for time_index, source_id in enumerate(order):
        source = by_id[source_id]
        base = float(source["probability"])
        probabilities = {
            "bounded": bounded.predict(base),
            "raw": base,
            "global_count": global_arm.predict(base).probability,
            "local_count": local_arm.predict(base).probability,
            "shuffled_feedback": shuffled.predict(base),
        }
        for arm, probability in probabilities.items():
            row = comparison_row(
                source_id,
                arm,
                probability,
                int(source["label"]),
                seed=seed,
                phase="online",
                censored=time_index >= len(order) - 8,
            )
            row["prediction_time"] = time_index
            row["state_hash_before_prediction"] = canonical_hash(
                _state_payload(bounded, shuffled, global_arm, local_arm)
            )
            rows.append(row)
        due = due_by_time.get(time_index)
        if due is None:
            continue
        release_rows = [by_id[event_id] for event_id in due["event_ids"]]
        if any(
            next(row for row in rows if row["unit_id"] == item["source_id"])["prediction_time"]
            > time_index
            for item in release_rows
        ):  # pragma: no cover - release lookup occurs after prediction.
            chronology_violations += 1
        matched = [
            (str(item["source_id"]), float(item["probability"]), int(item["label"]))
            for item in release_rows
        ]
        rotated_labels = [int(item["label"]) for item in release_rows[1:] + release_rows[:1]]
        permuted = [
            (str(item["source_id"]), float(item["probability"]), label)
            for item, label in zip(release_rows, rotated_labels, strict=True)
        ]
        bounded.update_batch(matched)
        shuffled.update_batch(permuted)
        for item in release_rows:
            global_arm.update(
                str(item["source_id"]), float(item["probability"]), int(item["label"])
            )
            local_arm.update(str(item["source_id"]), float(item["probability"]), int(item["label"]))
        before = _state_payload(bounded, shuffled, global_arm, local_arm)
        atomic_json(state_path, before)
        restored = json.loads(state_path.read_text(encoding="utf-8"))
        bounded = SufficientStatisticMap.from_payload(restored["bounded"])
        shuffled = SufficientStatisticMap.from_payload(restored["shuffled"])
        global_arm = CountArm.from_payload(restored["global"], config)
        local_arm = CountArm.from_payload(restored["local"], config)
        after = _state_payload(bounded, shuffled, global_arm, local_arm)
        if before != after:  # pragma: no cover - serializer mutation belongs to its upstream tests.
            restart_mismatches += 1
        releases.append(
            {
                "block_id": due["block_id"],
                "event_ids": due["event_ids"],
                "label_origins": due["event_ids"][1:] + due["event_ids"][:1],
            }
        )
    return {
        "rows": rows,
        "release_rows": releases,
        "release_count": len(releases),
        "chronology_violations": chronology_violations,
        "restart_mismatches": restart_mismatches,
        "shuffled_feedback_within_release_block": all(
            set(row["event_ids"]) == set(row["label_origins"]) for row in releases
        ),
        "final_state": _state_payload(bounded, shuffled, global_arm, local_arm),
    }


def evaluate_retention(
    retention_rows: Sequence[Mapping[str, Any]], learner: SufficientStatisticMap, *, seed: int
) -> JsonDict:
    """Score evaluator-only rows and prove the learner state is unchanged."""

    before = learner.state_hash()
    rows = [
        comparison_row(
            str(source["source_id"]),
            arm,
            probability,
            int(source["label"]),
            seed=seed,
            phase="retention",
        )
        for source in retention_rows
        for arm, probability in (
            ("raw", float(source["probability"])),
            ("bounded", learner.predict(float(source["probability"]))),
        )
    ]
    after = learner.state_hash()
    return {
        "rows": rows,
        "state_hash_before": before,
        "state_hash_after": after,
        "retention_labels_used_for_update": 0,
        "passed": before == after,
    }


def reduce_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Recompute each arm from raw unit numerators and denominators."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("raw_squared_error_denominator") != 1:
            raise ValueError("row_denominator_invalid")
        expected = _brier(float(row["probability"]), int(row["label"]))
        if not math.isclose(float(row["raw_squared_error_numerator"]), expected, abs_tol=1e-12):
            raise ValueError("row_numerator_invalid")
        grouped[str(row["arm"])].append(row)
    return {
        arm: {
            "brier_numerator": float(
                sum(float(row["raw_squared_error_numerator"]) for row in arm_rows)
            ),
            "denominator": len(arm_rows),
            "mean_brier": float(
                np.mean([float(row["raw_squared_error_numerator"]) for row in arm_rows])
            ),
            "mean_realized_cost": float(np.mean([float(row["realized_cost"]) for row in arm_rows])),
            "non_escalation_fraction": float(
                np.mean([bool(row["non_escalated"]) for row in arm_rows])
            ),
        }
        for arm, arm_rows in grouped.items()
    }


def paired_improvements(
    rows: Sequence[Mapping[str, Any]], candidate: str, comparator: str
) -> list[float]:
    """Return comparator-minus-candidate Brier for every matched unit."""

    by_key: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["unit_id"]), int(row["seed"]), str(row["phase"]))
        by_key[key][str(row["arm"])] = float(row["raw_squared_error_numerator"])
    pairs = [
        values[comparator] - values[candidate]
        for values in by_key.values()
        if {candidate, comparator} <= set(values)
    ]
    if not pairs:
        raise ValueError("paired_contrast_empty")
    return pairs


def _realized_cost(probability: float, label: int) -> tuple[float, bool]:
    decision = typed_decision(probability)
    action = str(decision["action"])
    return (
        {"accept": 5.0 if label else 0.0, "reject": 0.0 if label else 1.0, "escalate": 0.2}[action],
        action != "escalate",
    )


def _control_probability(static: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    arm = str(static["strongest_control"]["arm"])
    base = float(row["probability"])
    if arm == "raw":
        return base
    if arm == "temperature":
        return _temperature_probability(base, float(static["temperature"]["value"]))
    return map_probability(base, static["unconstrained_nine_knot"]["theta"])


def causal_uncertainty_replays(
    online_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    static: Mapping[str, Any],
    *,
    replays_per_order: int = 1000,
    progress_callback: Any | None = None,
) -> JsonDict:
    """Retrain every source-resampled stream in causal release order."""

    if replays_per_order < 1:
        raise ValueError("replays_per_order_invalid")
    by_id = {str(row["source_id"]): row for row in online_rows}
    distributions: dict[str, list[float]] = defaultdict(list)
    order_summaries: list[JsonDict] = []
    for seed in ORDER_SEEDS:
        base_order = list(by_id)
        random.Random(seed).shuffle(base_order)
        rng = np.random.default_rng(seed)
        order_metrics: list[JsonDict] = []
        for replay_index in range(replays_per_order):
            sampled_indices = rng.integers(0, len(base_order), size=len(base_order))
            stream = [by_id[base_order[int(index)]] for index in sampled_indices]
            learner = seeded_bounded_learner(static)
            losses = defaultdict(float)
            costs = defaultdict(float)
            non_escalated = 0
            pending: list[tuple[str, float, int]] = []
            for event_index, source in enumerate(stream):
                base = float(source["probability"])
                label = int(source["label"])
                candidate = learner.predict(base)
                control = _control_probability(static, source)
                losses["candidate"] += _brier(candidate, label)
                losses["raw"] += _brier(base, label)
                losses["control"] += _brier(control, label)
                candidate_cost, covered = _realized_cost(candidate, label)
                raw_cost, _unused = _realized_cost(base, label)
                costs["candidate"] += candidate_cost
                costs["raw"] += raw_cost
                non_escalated += int(covered)
                pending.append((f"{seed}:{replay_index}:{event_index}", base, label))
                if event_index >= 15 and (event_index - 15) % 8 == 0:
                    learner.update_batch(pending[:8])
                    pending = pending[8:]
            denominator = len(stream)
            final_retention = _mean_brier(
                retention_rows, lambda row: learner.predict(float(row["probability"]))
            )
            frozen_retention = _mean_brier(
                retention_rows,
                lambda row: map_probability(float(row["probability"]), static["bounded"]["theta"]),
            )
            metrics = {
                "brier_vs_raw": (losses["raw"] - losses["candidate"]) / denominator,
                "brier_vs_control": (losses["control"] - losses["candidate"]) / denominator,
                "decision_cost": (costs["raw"] - costs["candidate"]) / denominator,
                "coverage": non_escalated / denominator,
                "retention_degradation": final_retention - frozen_retention,
            }
            order_metrics.append(metrics)
            for name, value in metrics.items():
                distributions[name].append(float(value))
            if progress_callback is not None:
                progress_callback(seed, replay_index + 1, replays_per_order)
        order_summaries.append(
            {
                "seed": seed,
                "replays": replays_per_order,
                "mean_brier_vs_raw": float(np.mean([row["brier_vs_raw"] for row in order_metrics])),
            }
        )
    intervals = {
        name: {
            "lower95": float(np.quantile(values, 0.025)),
            "upper95": float(np.quantile(values, 0.975)),
            "mean": float(np.mean(values)),
        }
        for name, values in distributions.items()
    }
    return {
        "replays_per_order": replays_per_order,
        "orders": len(ORDER_SEEDS),
        "full_causal_replays": replays_per_order * len(ORDER_SEEDS),
        "each_replay_retrained_in_causal_order": True,
        "real_released_labels_only": True,
        "order_dependence_disclosed": True,
        "order_summaries": order_summaries,
        "intervals": intervals,
    }


def _load_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_object_required:{path}")
            rows.append(value)
    return rows


def _resolved(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def _precondition(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path),
        "field": field,
        "op": "eq",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check exact producer artifacts and revisions before measurement."""

    root = root.resolve()
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    artifacts: dict[str, JsonDict] = {}
    for label, relative in (
        ("exp7564", FIT_CAPTURE_PATH),
        ("exp7565", EVAL_CAPTURE_PATH),
        ("exp7533", PROTOCOL_SOURCE_PATH),
        ("exp7567", SOURCE_EVALUATION_PATH),
    ):
        path = root / relative
        exists = path.is_file()
        checks.append(
            _precondition(
                "artifact_exists",
                label,
                path,
                "path",
                "readable_file",
                "readable_file" if exists else "missing",
            )
        )
        if exists:
            try:
                artifacts[label] = _load_json(path)
                hashes.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
            except (OSError, json.JSONDecodeError, ValueError):
                checks.append(
                    _precondition("artifact_json", label, path, "json_object", True, False)
                )
    if set(artifacts) != {"exp7564", "exp7565", "exp7533", "exp7567"}:
        return checks, hashes
    field_checks = (
        ("exp7564", "fit_capture_ready_score", 1),
        ("exp7565", "test_capture_ready_score", 1),
        ("exp7565", "online_capture_ready_score", 1),
        ("exp7567", "static_measurement_complete_score", 1),
    )
    for source, field, expected in field_checks:
        path = (
            root
            / {
                "exp7564": FIT_CAPTURE_PATH,
                "exp7565": EVAL_CAPTURE_PATH,
                "exp7567": SOURCE_EVALUATION_PATH,
            }[source]
        )
        checks.append(
            _precondition(
                "upstream_gate", source, path, field, expected, artifacts[source].get(field)
            )
        )
    revision_row = next(
        (
            row
            for row in artifacts["exp7533"].get("preconditions_checked", [])
            if isinstance(row, Mapping) and row.get("check") == "dataset_revision"
        ),
        {},
    )
    checks.append(
        _precondition(
            "dataset_revision",
            PINNED_DATASET_ID,
            Path(str(revision_row.get("path") or root / "missing-dataset")),
            "revision",
            PINNED_DATASET_REVISION,
            revision_row.get("observed"),
        )
    )
    for source, expected in (
        ("exp7564", {"fit": 160, "tune": 40, "policy": 40}),
        ("exp7565", {"online": 160, "test": 80}),
    ):
        path = root / (FIT_CAPTURE_PATH if source == "exp7564" else EVAL_CAPTURE_PATH)
        checks.append(
            _precondition(
                "role_counts",
                source,
                path,
                "role_counts",
                expected,
                artifacts[source].get("role_counts"),
            )
        )
    for source in ("exp7564", "exp7565"):
        path = root / (FIT_CAPTURE_PATH if source == "exp7564" else EVAL_CAPTURE_PATH)
        checks.append(
            _precondition(
                "historical_model",
                source,
                path,
                "MODEL_SPECS",
                [HISTORICAL_MODEL_ID],
                artifacts[source].get("MODEL_SPECS"),
            )
        )
    return checks, hashes


def _read_receipt(root: Path, receipt: Mapping[str, Any], hashes: list[JsonDict]) -> list[JsonDict]:
    """Read one JSONL sidecar only after exact hash and row-count checks."""

    path = _resolved(root, str(receipt.get("path") or ""))
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"sidecar_hash_invalid:{path}")
    rows = _read_jsonl(path)
    if len(rows) != receipt.get("rows"):
        raise ValueError(f"sidecar_count_invalid:{path}")
    hashes.append(
        {"path": str(receipt["path"]), "sha256": receipt["sha256"], "bytes": path.stat().st_size}
    )
    return rows


def load_cached_roles(root: Path, source_hashes: list[JsonDict]) -> dict[str, list[JsonDict]]:
    """Join raw capture cells to separately stored evaluator labels."""

    root = root.resolve()
    fit_capture = _load_json(root / FIT_CAPTURE_PATH)
    eval_capture = _load_json(root / EVAL_CAPTURE_PATH)
    protocol = _load_json(root / PROTOCOL_SOURCE_PATH)
    official = {
        str(row["component_hash"]): str(row["official_split"])
        for row in protocol["role_manifest"]["groups"]
    }
    labels: list[JsonDict] = []
    for name in ("fit_labels", "tune_labels", "policy_labels"):
        labels.extend(_read_receipt(root, protocol["sealed_shards"][name], source_hashes))
    for name in ("online_labels", "test_labels"):
        labels.extend(
            _read_receipt(root, eval_capture["raw_manifest"]["role_sidecars"][name], source_hashes)
        )
    label_by_identity = {(str(row["component_hash"]), str(row["role"])): row for row in labels}
    if (
        len(label_by_identity) != len(labels)
    ):  # pragma: no cover - producer receipts are immutable; private duplicates are tested upstream.
        raise ValueError("label_identity_duplicate")
    native: list[JsonDict] = []
    for capture in (fit_capture, eval_capture):
        for receipt in capture["raw_manifest"]["native_row_shards"]:
            native.extend(_read_receipt(root, receipt, source_hashes))
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in native:
        grouped[str(row["group_hash"])].append(row)
    roles: dict[str, list[JsonDict]] = {name: [] for name in ROLE_COUNTS}
    for group in grouped.values():
        identity = (str(group[0]["component_hash"]), str(group[0]["role"]))
        label = label_by_identity.get(identity)
        if (
            label is None or identity[0] not in official
        ):  # pragma: no cover - authenticated producer rosters establish this join.
            raise ValueError("authorized_label_or_split_missing")
        reduced = reduce_native_group(group, label, official[identity[0]])
        roles[identity[1]].append(reduced)
    for rows in roles.values():
        rows.sort(key=lambda row: str(row["source_id"]))
    build_exposure_manifest(roles)
    return roles


def static_bootstrap(
    rows: Sequence[Mapping[str, Any]], strongest_control: str, *, draws: int = 1000
) -> JsonDict:
    """Resample policy source components and Holm-adjust two contrasts."""

    candidate = "bounded"
    contrasts = {"vs_raw": "raw", "vs_strongest": strongest_control}
    by_unit: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_unit[str(row["unit_id"])][str(row["arm"])] = float(row["brier"])
    units = sorted(by_unit)
    if any(not {candidate, *contrasts.values()} <= set(by_unit[unit]) for unit in units):
        raise ValueError("static_contrast_roster_invalid")
    rng = np.random.default_rng(7_575_101)
    samples = {name: [] for name in contrasts}
    for _draw in range(draws):
        indices = rng.integers(0, len(units), size=len(units))
        for name, arm in contrasts.items():
            samples[name].append(
                float(
                    np.mean(
                        [
                            by_unit[units[int(index)]][arm] - by_unit[units[int(index)]][candidate]
                            for index in indices
                        ]
                    )
                )
            )
    raw = {
        name: {
            "mean_improvement": float(np.mean(values)),
            "lower95": float(np.quantile(values, 0.025)),
            "upper95": float(np.quantile(values, 0.975)),
            "one_sided_p": float((1 + sum(value <= 0.0 for value in values)) / (len(values) + 1)),
        }
        for name, values in samples.items()
    }
    ordered = sorted(raw, key=lambda name: raw[name]["one_sided_p"])
    running = 0.0
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * raw[name]["one_sided_p"])
        running = max(running, adjusted)
        raw[name]["holm_adjusted_p"] = running
        raw[name]["holm_passed"] = running < 0.05 and raw[name]["lower95"] > 0.0
        raw[name]["comparator_arm"] = contrasts[name]
    return {"draws": draws, "resampling_unit": "source_component", "contrasts": raw}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], root: Path) -> JsonDict:
    """Write a deterministic sidecar atomically and return its byte receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(path.resolve())
    return {
        "path": label,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def field_principles() -> dict[str, str]:
    """Return the one-line meaning of every required terminal field."""

    return {
        "honest_verdict": "Use a complete_ terminal prefix; completion does not establish benefit.",
        "verdict_class": "Exactly one closed verdict class describes the terminal evidence.",
        "flagged_adversarial": "Persist the exact terminal verification outcome; flagged evidence cannot open readiness.",
        "gate_check_summary": "Blocked work names the failed check and its exact expected and observed operands.",
        "acceptance_gate_results": "Validity, readiness, and benefit checks remain separate so a valid null stays usable.",
        "rows": "Each comparison unit and arm carries raw arithmetic, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Actual and planned substrates stay distinct; no model call means no live inference claim.",
        "MODEL_SPECS": "Cached-only work records no current model specification.",
        "invocation_counts": "Current loads, forwards, generations, and tokens are counted independently from history.",
        "duration_s": "Monotonic current work excludes inherited timing and artificial sleeps.",
        "source_artifact_hashes": "Every conclusion binds to exact source bytes and names absent inputs before gates.",
        "validation_receipts": "Each check binds its command, worktree, exit code, and log hash.",
        "verifier_is_oracle": "Label-accessing controls cannot support an oracle-distinct positive claim.",
        "cached_roles_ready_score": "One requires complete authentic role-separated cached evidence.",
        "online_protocol_ready_score": "One requires frozen causal schedules, controls, and retention isolation.",
        "fresh_confirmatory_claim_allowed": "This field is always false on the exposed corpus.",
        "protocol_sha256": "The hash freezes roles, orders, costs, controls, uncertainty, and stopping rules before fitting.",
        "historical_model_id": "The historical Qwen identity does not claim a current load.",
    }


def _gate(check: str, category: str, expected: Any, observed: Any, *, op: str = "eq") -> JsonDict:
    if op == "eq":
        passed = observed == expected
    elif op == "gt":
        passed = float(observed) > float(expected)
    elif op == "ge":
        passed = float(observed) >= float(expected)
    elif op == "le":
        passed = float(observed) <= float(expected)
    else:
        raise ValueError(f"gate_operator_invalid:{op}")
    return {
        "check": check,
        "category": category,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row["check"] for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    stable = deepcopy(dict(value))
    stable.pop("reproducibility_checksum", None)
    return canonical_hash(stable)


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is not True
        for row in receipts
    )


def build_artifact(
    evidence: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    require_validation: bool = True,
) -> JsonDict:
    """Build a schema-complete descriptive result from raw reduced evidence."""

    protocol = evidence["protocol"]
    exposure = evidence["exposure_manifest"]
    effects = evidence["effect_gates"]
    validation_ok = _validation_passed(validation_receipts) if require_validation else True
    gates = [
        _gate(
            "preconditions",
            "validity",
            True,
            all(row.get("passed") is True for row in preconditions),
        ),
        _gate("cached_custody", "validity", True, evidence.get("custody_passed")),
        _gate("scoped_and_terminal_validation", "validity", True, validation_ok),
        _gate(
            "historical_exposure_preserved",
            "validity",
            False,
            exposure.get("fresh_confirmatory_claim_allowed"),
        ),
        _gate("cached_roles_ready", "readiness", True, evidence.get("custody_passed")),
        _gate("causal_lifecycle", "readiness", True, evidence.get("lifecycle_passed")),
        _gate("retention_isolation", "readiness", True, evidence.get("retention_isolated")),
        _gate("brier_vs_raw_lower95", "benefit", True, effects.get("brier_vs_raw")),
        _gate(
            "brier_vs_strongest_control_lower95", "benefit", True, effects.get("brier_vs_control")
        ),
        _gate("decision_cost_lower95_and_coverage", "benefit", True, effects.get("decision_cost")),
        _gate("retention_upper95_degradation", "benefit", True, effects.get("retention")),
    ]
    readiness = all(row["passed"] for row in gates if row["category"] in {"validity", "readiness"})
    benefit = all(row["passed"] for row in gates if row["category"] == "benefit")
    verdict_class = "null" if readiness else "disqualified"
    verdict = (
        "complete_null_exploratory_cached_learning_protocol_frozen"
        if readiness
        else "complete_disqualified_cached_learning_protocol_validation_failed"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "V662 cached learning protocol",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "positive_claim": False,
        "exploratory_effect_gates_passed": benefit,
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse",
        "deployment_promotion_available": False,
        "verifier_is_oracle": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "historical_model_id": HISTORICAL_MODEL_ID,
        "model_invoked": False,
        "no_model_load": True,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_protocol": deepcopy(protocol),
        "exposure_manifest": deepcopy(exposure),
        "static_controls": deepcopy(evidence.get("static_controls", {})),
        "uncertainty": deepcopy(evidence["uncertainty"]),
        "effect_gates": deepcopy(effects),
        "analytical_positive_control": deepcopy(evidence["analytical_positive_control"]),
        "rows": [deepcopy(dict(row)) for row in evidence["rows"]],
        "independent_row_reduction": reduce_comparison_rows(evidence["rows"]),
        "cached_roles_ready_score": int(readiness),
        "online_protocol_ready_score": int(
            readiness
            and evidence.get("lifecycle_passed") is True
            and evidence.get("retention_isolated") is True
        ),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "capability_e2e": deepcopy(evidence["capability_e2e"]),
        "applicable_numbered_e2e": [],
        "private_llm_off_real_environment_smoke": {
            "applicable": False,
            "reason": "No ARC runtime, binding, or sampler changed.",
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "raw_sidecars": deepcopy(evidence.get("raw_sidecars", {})),
        "historical_hash_authentication": deepcopy(
            evidence.get("historical_hash_authentication", {})
        ),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "duration_s": float(duration_s),
        "field_principles": field_principles(),
        "external_publication_authorized": False,
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Publish missing external evidence without invented dependent rows."""

    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else {
            "check": "unknown_precondition",
            "upstream": "unknown",
            "path": "unknown",
            "field": "unknown",
            "op": "eq",
            "expected": True,
            "observed": False,
            "passed": False,
        }
    )
    reason = str(first["check"]).replace(" ", "_")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "positive_claim": False,
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_reuse",
        "verifier_is_oracle": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "historical_model_id": HISTORICAL_MODEL_ID,
        "model_invoked": False,
        "no_model_load": True,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "blocked_before_cached_aggregation",
        "protocol_sha256": None,
        "rows": [],
        "cached_roles_ready_score": 0,
        "online_protocol_ready_score": 0,
        "acceptance_gate_results": failed,
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failed),
            "failed_checks": [row["check"] for row in failed],
            "first_failure": first,
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "validation_receipts": [],
        "duration_s": float(duration_s),
        "field_principles": field_principles(),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_validation: bool = True
) -> JsonDict:
    """Cold-check identity, raw arithmetic, hashes, scope, and readiness."""

    if not isinstance(value, Mapping):
        raise ValueError("artifact_object_required")
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if artifact.get("historical_model_id") != HISTORICAL_MODEL_ID:
        errors.append("historical_model_invalid")
    if (
        artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocations_not_zero")
    if (
        artifact.get("fresh_confirmatory_claim_allowed") is not False
        or artifact.get("claim_scope") != "descriptive_reuse"
    ):
        errors.append("claim_scope_invalid")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if not str(artifact.get("honest_verdict") or "").startswith("complete_blocked_"):
            errors.append("blocked_verdict_invalid")
        first = artifact.get("gate_check_summary", {}).get("first_failure")
        if not isinstance(first, Mapping) or not {
            "check",
            "upstream",
            "path",
            "field",
            "op",
            "expected",
            "observed",
        } <= set(first):
            errors.append("blocked_gate_summary_invalid")
        if (
            artifact.get("rows") != []
            or artifact.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_measurement_invalid")
    else:
        if artifact.get("inference_substrate_class") != "no_model_load":
            errors.append("substrate_invalid")
        protocol = artifact.get("frozen_protocol")
        if not isinstance(protocol, Mapping) or protocol.get("protocol_sha256") != canonical_hash(
            protocol.get("hash_payload")
        ):
            errors.append("protocol_hash_invalid")
        if artifact.get("protocol_sha256") != (protocol or {}).get("protocol_sha256"):
            errors.append("protocol_identity_invalid")
        rows = artifact.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append("rows_missing")
        else:
            try:
                reduced = reduce_comparison_rows(rows)
                if reduced != artifact.get("independent_row_reduction"):
                    errors.append("row_reduction_mismatch")
            except (KeyError, TypeError, ValueError):
                errors.append("row_reduction_invalid")
        if artifact.get("cached_roles_ready_score") not in (0, 1) or artifact.get(
            "online_protocol_ready_score"
        ) not in (0, 1):
            errors.append("score_invalid")
        if artifact.get("verdict_class") not in {"null", "disqualified"}:
            errors.append("verdict_mismatch")
        if require_validation and not _validation_passed(artifact.get("validation_receipts", [])):
            errors.append("validation_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_PRINCIPLE_FIELDS) <= set(principles):
        errors.append("field_principles_invalid")
    for receipt in (
        artifact.get("raw_sidecars", {}).values()
        if isinstance(artifact.get("raw_sidecars"), Mapping)
        else []
    ):
        if not isinstance(receipt, Mapping):
            errors.append("raw_sidecar_invalid")
            continue
        path = _resolved(root, str(receipt.get("path") or ""))
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            errors.append("raw_sidecar_hash_invalid")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "valid": True,
        "blocked": blocked,
        "row_arms": len(artifact.get("independent_row_reduction", {})),
    }


def cold_replay(path: Path, *, root: Path = REPO_ROOT, require_validation: bool = True) -> JsonDict:
    """Reload one exact artifact in a fresh-process compatible path."""

    return validate_artifact(_load_json(path), root=root, require_validation=require_validation)


def independent_reduce_artifact(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Independently reduce all comparative rows from one candidate."""

    artifact = _load_json(path)
    validation = validate_artifact(artifact, root=root, require_validation=False)
    return {
        "passed": validation["valid"],
        "reduction_sha256": canonical_hash(reduce_comparison_rows(artifact.get("rows", [])))
        if artifact.get("rows")
        else None,
        "protocol_sha256": artifact.get("protocol_sha256"),
    }


def measure_cached_protocol(  # pragma: no cover - exercised by the declared entrypoint.
    root: Path,
    raw_dir: Path,
    *,
    replays_per_order: int = 1000,
    replay_progress: Any | None = None,
) -> JsonDict:
    """Measure the frozen protocol from authenticated historical bytes."""

    source_hashes: list[JsonDict] = []
    roles = load_cached_roles(root, source_hashes)
    exposure = build_exposure_manifest(roles)
    protocol = freeze_protocol(roles)
    static = fit_static_controls(roles["fit"], roles["tune"], roles["policy"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    role_rows = [row for role in ROLE_COUNTS for row in roles[role]]
    role_receipt = _write_jsonl(raw_dir / "cached_roles.jsonl", role_rows, root)
    protocol_path = raw_dir / "frozen_protocol.json"
    exposure_path = raw_dir / "exposure_manifest.json"
    atomic_json(protocol_path, protocol)
    atomic_json(exposure_path, exposure)
    protocol_receipt = {
        "path": protocol_path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(protocol_path),
        "bytes": protocol_path.stat().st_size,
    }
    exposure_receipt = {
        "path": exposure_path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(exposure_path),
        "bytes": exposure_path.stat().st_size,
    }
    all_rows: list[JsonDict] = list(static["static_rows_by_role"]["policy"])
    replays: list[JsonDict] = []
    retention_receipts: list[JsonDict] = []
    for seed in ORDER_SEEDS:
        replay = replay_order(
            roles["online"],
            protocol["orders"][str(seed)],
            static,
            seed=seed,
            state_path=raw_dir / "states" / f"seed-{seed}.json",
        )
        replays.append(replay)
        all_rows.extend(replay["rows"])
        learner = SufficientStatisticMap.from_payload(replay["final_state"]["bounded"])
        retention = evaluate_retention(roles["test"], learner, seed=seed)
        all_rows.extend(retention["rows"])
        retention_receipts.append(
            {
                "seed": seed,
                "state_hash_before": retention["state_hash_before"],
                "state_hash_after": retention["state_hash_after"],
                "labels_used_for_update": retention["retention_labels_used_for_update"],
            }
        )
    static_uncertainty = static_bootstrap(
        static["static_rows_by_role"]["policy"],
        str(static["strongest_control"]["arm"]),
    )
    causal_uncertainty = causal_uncertainty_replays(
        roles["online"],
        roles["test"],
        static,
        replays_per_order=replays_per_order,
        progress_callback=replay_progress,
    )
    intervals = causal_uncertainty["intervals"]
    static_contrasts = static_uncertainty["contrasts"]
    effect_gates = {
        "brier_vs_raw": static_contrasts["vs_raw"]["holm_passed"],
        "brier_vs_control": static_contrasts["vs_strongest"]["holm_passed"],
        "decision_cost": intervals["decision_cost"]["lower95"] > 0.0
        and intervals["coverage"]["lower95"] >= 0.10,
        "retention": intervals["retention_degradation"]["upper95"] <= 0.005,
        "thresholds": {
            "brier_improvement_lower95": 0.0,
            "decision_cost_improvement_lower95": 0.0,
            "minimum_non_escalation_fraction": 0.10,
            "retention_brier_upper95_degradation": 0.005,
        },
    }
    lifecycle_passed = all(
        replay["chronology_violations"] == 0
        and replay["restart_mismatches"] == 0
        and replay["shuffled_feedback_within_release_block"] is True
        for replay in replays
    )
    retention_isolated = all(
        row["state_hash_before"] == row["state_hash_after"] and row["labels_used_for_update"] == 0
        for row in retention_receipts
    )
    request_hashes = sorted({value for row in role_rows for value in row["request_hashes"]})
    response_hashes = sorted({str(row["response_sha256"]) for row in role_rows})
    return {
        "rows": all_rows,
        "protocol": protocol,
        "exposure_manifest": exposure,
        "static_controls": static,
        "custody_passed": len(role_rows) == sum(ROLE_COUNTS.values()),
        "lifecycle_passed": lifecycle_passed,
        "retention_isolated": retention_isolated,
        "uncertainty": {
            "static": static_uncertainty,
            "causal": causal_uncertainty,
            **{
                key: causal_uncertainty[key]
                for key in ("replays_per_order", "orders", "order_dependence_disclosed")
            },
        },
        "effect_gates": effect_gates,
        "capability_e2e": {
            "passed": lifecycle_passed,
            "operations": ["predict", "release", "update", "persist", "reload"],
            "orders_exercised": len(replays),
        },
        "analytical_positive_control": {
            "verdict_class": "null",
            "row_contrast_supported": False,
            "readiness_alone_is_positive": False,
        },
        "raw_sidecars": {
            "cached_roles": role_receipt,
            "frozen_protocol": protocol_receipt,
            "exposure_manifest": exposure_receipt,
        },
        "historical_hash_authentication": {
            "request_hash_count": len(request_hashes),
            "request_hashes_sha256": canonical_hash(request_hashes),
            "response_hash_count": len(response_hashes),
            "response_hashes_sha256": canonical_hash(response_hashes),
            "current_inference_calls": 0,
        },
        "source_hashes": source_hashes,
    }


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary with monotonic elapsed time."""

    payload = {
        "phase": phase,
        "event": event,
        "elapsed_s": round(time.monotonic() - started, 3),
        **details,
    }
    print("[exp7575-progress] " + json.dumps(payload, sort_keys=True), flush=True)


def _terminal_commands(candidate: Path) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build bounded fresh readers for the exact candidate."""

    python = ".venv/bin/python"
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--verify-artifact", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_row_reduction",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300.0,
        ),
    ]


def _run_specs(  # pragma: no cover - subprocess boundary.
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
) -> list[JsonDict]:
    planned = [PlannedCommand(command, "required_validation", True) for command in commands]
    rows = run_categorized_commands(root, planned, log_dir=log_dir, heartbeat_s=60.0)
    for row in rows:
        row["worktree"] = str(root.resolve())
    return rows


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - declared capability E2E.
    """Run cached measurement, scoped checks, exact readers, and publication."""

    started = time.monotonic()
    root = root.resolve()
    if root != REPO_ROOT.resolve() or run_date != RUN_DATE:
        raise ValueError("root_or_date_invalid")
    progress(started, "preconditions", "before")
    checks, source_hashes = collect_preconditions(root)
    progress(started, "preconditions", "after", completed=len(checks))
    result_path = root / RESULT_PATH
    raw_dir = root / RAW_DIR
    candidate_path = raw_dir / "terminal_candidate.json"
    manifest_path = raw_dir / "affected_validation_manifest.json"
    if any(row["passed"] is not True for row in checks):
        artifact = build_blocked_artifact(
            checks, source_hashes, duration_s=time.monotonic() - started
        )
        atomic_json(result_path, artifact)
        progress(started, "publish", "blocked", path=str(result_path))
        return 0
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7575-"))
    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    command_errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if command_errors:
        raise RuntimeError("validation_plan_invalid:" + ",".join(command_errors))
    progress(started, "affected_validation", "before", commands=len(commands))
    affected = _run_specs(root, commands, raw_dir / "validation" / "affected")
    progress(started, "affected_validation", "after", passed=_validation_passed(affected))
    if not _validation_passed(affected):
        raise RuntimeError("affected_validation_failed")
    heartbeat = {"last": time.monotonic()}

    def replay_progress(seed: int, completed: int, total: int) -> None:
        now = time.monotonic()
        if now - heartbeat["last"] >= 60.0 or completed == total:
            progress(
                started,
                "causal_uncertainty",
                "unit_progress",
                seed=seed,
                completed=completed,
                total=total,
            )
            heartbeat["last"] = now

    progress(started, "measurement", "before")
    evidence = measure_cached_protocol(root, raw_dir, replay_progress=replay_progress)
    source_hashes.extend(evidence.pop("source_hashes"))
    progress(started, "measurement", "after", rows=len(evidence["rows"]))
    provisional = build_artifact(
        evidence,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
    )
    atomic_json(candidate_path, provisional)
    progress(started, "terminal_validation", "before")
    terminal = _run_specs(
        root, _terminal_commands(candidate_path), raw_dir / "validation" / "terminal"
    )
    if not _validation_passed(terminal):
        raise RuntimeError("terminal_validation_failed")
    final = build_artifact(
        evidence,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
    )
    atomic_json(candidate_path, final)
    exact = _run_specs(
        root, _terminal_commands(candidate_path), raw_dir / "validation" / "exact_terminal"
    )
    if not _validation_passed(exact):
        raise RuntimeError("exact_terminal_validation_failed")
    atomic_json(result_path, final)
    progress(started, "publish", "after", path=str(result_path), verdict=final["honest_verdict"])
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin producer and two read-only verification modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path, root: Path) -> Path:  # pragma: no cover - CLI path boundary.
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Dispatch production or read-only verification without model work."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.verify_artifact is not None:
        result = cold_replay(
            _argument_path(args.verify_artifact, root), root=root, require_validation=True
        )
        print(json.dumps({"mode": "cold_replay", **result}, sort_keys=True), flush=True)
        return 0
    if args.independent_reduce is not None:
        result = independent_reduce_artifact(
            _argument_path(args.independent_reduce, root), root=root
        )
        print(json.dumps({"mode": "independent_reduce", **result}, sort_keys=True), flush=True)
        return int(result["passed"] is not True)
    return run_experiment(root, args.date)
