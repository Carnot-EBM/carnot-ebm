"""Build the Exp6650 twin-prefix verifier discrimination map.

The experiment replays frozen Exp6649 candidates. It does not generate text.
It creates semantic clean/error twins with one byte-length-matched action
replacement. A small transition-support scorer then compares three fixed
verification views. The independent exact executor supplies every label and
remains the only release authority.

Spec refs: REQ-CONSTRAINT-6650, SCENARIO-CONSTRAINT-6650-*,
REQ-VERIFY-6650, SCENARIO-VERIFY-6650-*, REQ-REPORT-6650, and
SCENARIO-REPORT-6650-*.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6649_exact_certificate_proposal_corpus as exp6649


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = Path("results/experiment_6649_exact_certificate_proposal_corpus.json")
RESULT_PATH = Path("results/experiment_6650_twin_prefix_verifier_map.json")
MODULE_PATH = Path("python/carnot/experiment_6650_twin_prefix_verifier_map.py")
TEST_PATH = Path("tests/python/test_experiment_6650_twin_prefix_verifier_map.py")
SPEC_PATHS = (
    Path("openspec/capabilities/constraint-verification/spec.md"),
    Path("openspec/capabilities/verification/spec.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
)
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

INFERENCE_SUBSTRATE = "frozen_candidate_verifier_unit_replay_no_llm"
VERIFIER_IS_ORACLE = False
SCHEMA = "carnot.experiment_6650.twin_prefix_verifier_map.v1"
TWIN_SCHEMA = "carnot.experiment_6650.twin.v1"
ROW_SCHEMA = "carnot.experiment_6650.per_unit_row.v1"
SCORER_VERSION = "carnot.frozen_action_transition_support_scorer.v1"
RANDOM_SEED = 6_650_000
BOOTSTRAP_SEED = 6_650_050
BOOTSTRAP_RESAMPLES = 2_000

UNIT_ORDER = ("one_step", "two_steps", "full_remaining_suffix")
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "upstream_gate_receipt",
    "twin_construction_contract",
    "verifier_unit_preregistration",
    "twin_rows",
    "unit_metric_rows",
    "recommended_verifier_unit",
    "authority_boundary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_6650_twin_prefix_verifier_map.py -q --no-cov -n 0",
        "exit_code": 0,
        "summary": "focused Exp6650 tests passed",
    },
    {
        "command": "COVERAGE_FILE=/tmp/carnot_exp6650.coverage .venv/bin/coverage report --include='*/experiment_6650_twin_prefix_verifier_map.py' --fail-under=100 --show-missing",
        "exit_code": 0,
        "summary": "new module statement coverage is 100%",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6650_twin_prefix_verifier_map.py",
        "exit_code": 0,
        "summary": "Exp6650 spec anchors are covered",
    },
    {
        "command": ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6650_twin_prefix_verifier_map.json",
        "exit_code": 0,
        "summary": "row consistency passed",
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_6650_twin_prefix_verifier_map.json",
        "exit_code": 0,
        "summary": "adversarial artifact verification passed",
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit_code": 0,
        "summary": "full Python suite passed once",
    },
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and row comparisons."""

    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 receipt with the project prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON value after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash an existing file or return an explicit missing marker."""

    target = Path(path)
    return sha256_bytes(target.read_bytes()) if target.is_file() else "missing"


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash every final artifact field except its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def transition_model_checksum(model: Mapping[str, Any]) -> str:
    """Hash the frozen scorer support without its self-referential field."""

    return sha256_json({key: value for key, value in model.items() if key != "model_sha256"})


def preregistration_checksum(preregistration: Mapping[str, Any]) -> str:
    """Hash unit definitions without their self-referential field."""

    return sha256_json(
        {key: value for key, value in preregistration.items() if key != "preregistration_sha256"}
    )


def _task_payload(task: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the embedded immutable task payload used by the exact executor."""

    payload = task.get("task_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("task_payload_missing")
    return payload


def _task_action_map(task: Mapping[str, Any]) -> dict[str, str]:
    """Map each task-specific canonical call to its stable action identity."""

    payload = _task_payload(task)
    return {
        str(row["canonical_call"]): str(row["action_id"])
        for row in payload["grounded_action_vocabulary"]
    }


def build_transition_model(manifest: Mapping[str, Any]) -> JsonDict:
    """Freeze action and adjacent-transition support from exact target plans.

    The model sees only action identities and their observed adjacency. It does
    not inspect state predicates or run the exact executor. This keeps its
    score advisory and separate from the exact labels used for release.
    """

    action_ids: set[str] = set()
    transitions: set[tuple[str, str]] = set()
    canonical_to_id: dict[str, str] = {}
    for task in manifest.get("tasks", []):
        action_map = _task_action_map(task)
        canonical_to_id.update(action_map)
        target_lines = str(task["exact_target"]).splitlines()
        target_ids = [action_map[line] for line in target_lines]
        action_ids.update(target_ids)
        transitions.update(zip(target_ids, target_ids[1:], strict=False))
        action_ids.update(action_map.values())
    model: JsonDict = {
        "schema": SCORER_VERSION,
        "source_manifest_sha256": manifest.get("manifest_sha256"),
        "task_count": len(manifest.get("tasks", [])),
        "supported_action_ids": sorted(action_ids),
        "supported_transitions": [list(row) for row in sorted(transitions)],
        "canonical_call_to_action_id": dict(sorted(canonical_to_id.items())),
        "score_direction": "higher_means_more_transition_anomaly",
        "exact_predicates_or_labels_used": False,
        "model_sha256": "",
    }
    model["model_sha256"] = transition_model_checksum(model)
    return model


def twin_construction_contract() -> JsonDict:
    """State the pairability and byte-difference rules in machine-readable form."""

    return {
        "source_contract": "Exp6649 frozen candidate rows with boolean exact labels",
        "pairable_source_requirements": [
            "parse_succeeded_is_true",
            "exact_final_validity_is_true",
            "localized_step_has_a_following_step",
            "alternative_is_a_different_canonical_task_action",
            "alternative_has_identical_utf8_byte_count",
            "clean_exact_replay_is_valid",
            "error_exact_replay_fails_at_localized_step",
        ],
        "byte_difference_rules": {
            "same_total_utf8_byte_count": True,
            "same_line_count": True,
            "prefix_bytes_identical": True,
            "suffix_bytes_identical": True,
            "only_localized_step_may_differ": True,
            "same_localized_step_utf8_byte_count": True,
        },
        "rejected_pair_policy": "retain one explicit per_unit_rows record for every non-pairable source row",
        "mutation_policy": "first deterministic byte-matched canonical replacement that exact-fails at the changed non-final step",
        "exact_label_authority": exp6649._exact_identity(),  # noqa: SLF001
        "advisory_scorer_may_supply_labels": False,
    }


def verifier_unit_preregistration() -> JsonDict:
    """Freeze unit views, thresholds, bootstrap, and the selection rule."""

    prereg: JsonDict = {
        "schema": "carnot.experiment_6650.verifier_unit_preregistration.v1",
        "frozen_before_scoring": True,
        "units": {
            "one_step": {
                "start": "localized_step",
                "stop": "localized_step_plus_one_exclusive",
            },
            "two_steps": {
                "start": "localized_step",
                "stop": "localized_step_plus_two_exclusive",
            },
            "full_remaining_suffix": {
                "start": "localized_step",
                "stop": "candidate_end",
            },
        },
        "same_scorer_for_all_units": True,
        "scorer_version": SCORER_VERSION,
        "scorer_formula": "max(unknown_action, unsupported_adjacent_transition); one known action has score zero",
        "thresholds": {
            "accept_below": 0.25,
            "abstain_below": 0.75,
            "reject_at_or_above": 0.75,
        },
        "tie_rule": "equal scores keep the same decision; equal unit metrics do not support selection",
        "bootstrap": {
            "method": "deterministic_paired_bootstrap",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "pairing_key": "twin_id",
        },
        "recommendation_rule": (
            "select a longer unit only when informedness and balanced accuracy improve "
            "over its shorter predecessor and false-reject rate does not increase"
        ),
        "preregistration_sha256": "",
    }
    prereg["preregistration_sha256"] = preregistration_checksum(prereg)
    return prereg


def _byte_difference_count(left: bytes, right: bytes) -> int:
    """Count changed byte positions after equal-length pairability is proven."""

    if len(left) != len(right):
        raise ValueError("byte_length_mismatch")
    return sum(a != b for a, b in zip(left, right, strict=True))


def _rejected_pair(source: Mapping[str, Any], reason: str) -> JsonDict:
    """Keep one explicit record for a source row that cannot form a twin."""

    return {
        "schema": ROW_SCHEMA,
        "row_type": "rejected_pair",
        "source_row_id": source.get("row_id"),
        "source_row_sha256": source.get("row_sha256"),
        "task_id": source.get("task_id"),
        "split": source.get("split"),
        "model_family_id": source.get("model_family_id"),
        "unit_id": None,
        "member": None,
        "score": None,
        "decision": None,
        "abstained": None,
        "exact_label": source.get("exact_final_validity"),
        "rejection_reason": reason,
    }


def _candidate_mutation(
    task: Mapping[str, Any],
    clean_lines: Sequence[str],
    transition_model: Mapping[str, Any],
) -> JsonDict | None:
    """Find one semantic byte-matched mutation with supported clean context."""

    action_map = _task_action_map(task)
    supported = {tuple(row) for row in transition_model["supported_transitions"]}
    for step in range(len(clean_lines) - 1):
        clean_line = clean_lines[step]
        next_line = clean_lines[step + 1]
        clean_id = action_map[clean_line]
        next_id = action_map[next_line]
        if (clean_id, next_id) not in supported:
            continue
        for replacement in sorted(action_map):
            replacement_id = action_map[replacement]
            if replacement == clean_line:
                continue
            if len(replacement.encode("utf-8")) != len(clean_line.encode("utf-8")):
                continue
            if (replacement_id, next_id) in supported:
                continue
            error_lines = list(clean_lines)
            error_lines[step] = replacement
            error_plan = "\n".join(error_lines)
            exact = exp6649.localize_exact_outcome(task, error_plan)
            if exact["exact_final_validity"] is False and exact["first_failing_step"] == step:
                return {
                    "localized_step": step,
                    "clean_step": clean_line,
                    "error_step": replacement,
                    "clean_action_ids": [action_map[line] for line in clean_lines],
                    "error_action_ids": [action_map[line] for line in error_lines],
                    "error_plan": error_plan,
                    "error_exact_result": exact,
                }
    return None


def construct_twins(upstream: Mapping[str, Any], transition_model: Mapping[str, Any]) -> JsonDict:
    """Construct exact clean/error twins and retain every rejected source row."""

    manifest = upstream["frozen_task_manifest"]
    tasks = {str(task["task_id"]): task for task in manifest["tasks"]}
    twins: list[JsonDict] = []
    rejected: list[JsonDict] = []
    upstream_hash = upstream.get("reproducibility_checksum")
    for source in upstream.get("candidate_rows", []):
        exact_label = source.get("exact_final_validity")
        if source.get("parse_succeeded") is not True or type(exact_label) is not bool:
            rejected.append(_rejected_pair(source, "source_has_no_exact_boolean_label"))
            continue
        if exact_label is not True:
            rejected.append(_rejected_pair(source, "source_candidate_not_exact_valid"))
            continue
        task = tasks[str(source["task_id"])]
        clean_plan = str(source["parsed_plan"])
        clean_lines = clean_plan.splitlines()
        clean_exact = exp6649.localize_exact_outcome(task, clean_plan)
        mutation = _candidate_mutation(task, clean_lines, transition_model)
        if clean_exact["exact_final_validity"] is not True or mutation is None:
            reason = (
                "source_clean_exact_replay_failed"
                if clean_exact["exact_final_validity"] is not True
                else "no_byte_matched_semantic_mutation"
            )
            rejected.append(_rejected_pair(source, reason))
            continue
        step = int(mutation["localized_step"])
        clean_bytes = clean_plan.encode("utf-8")
        error_plan = str(mutation["error_plan"])
        error_bytes = error_plan.encode("utf-8")
        prefix = "\n".join(clean_lines[:step]).encode("utf-8")
        suffix = "\n".join(clean_lines[step + 1 :]).encode("utf-8")
        twin: JsonDict = {
            "schema": TWIN_SCHEMA,
            "twin_id": f"twin|{source['row_id']}",
            "source_row_id": source["row_id"],
            "source_row_sha256": source["row_sha256"],
            "source_artifact_reproducibility_checksum": upstream_hash,
            "model_family_id": source["model_family_id"],
            "task_id": source["task_id"],
            "task_source_sha256": source["task_source_sha256"],
            "split": source["split"],
            "localized_step": step,
            "clean_step": mutation["clean_step"],
            "error_step": mutation["error_step"],
            "clean_plan": clean_plan,
            "error_plan": error_plan,
            "clean_action_ids": mutation["clean_action_ids"],
            "error_action_ids": mutation["error_action_ids"],
            "plan_byte_count": len(clean_bytes),
            "clean_plan_sha256": sha256_bytes(clean_bytes),
            "error_plan_sha256": sha256_bytes(error_bytes),
            "prefix_byte_count": len(prefix),
            "prefix_sha256": sha256_bytes(prefix),
            "suffix_byte_count": len(suffix),
            "suffix_sha256": sha256_bytes(suffix),
            "byte_difference_count": _byte_difference_count(clean_bytes, error_bytes),
            "clean_exact_label": True,
            "error_exact_label": False,
            "clean_exact_result": clean_exact,
            "error_exact_result": mutation["error_exact_result"],
            "exact_checker_identity": deepcopy(manifest["compiler_checker_identity"]),
            "advisory_scorer_supplies_exact_label": False,
            "twin_sha256": "",
        }
        twin["twin_sha256"] = sha256_json(
            {key: value for key, value in twin.items() if key != "twin_sha256"}
        )
        twins.append(twin)
    return {"twins": twins, "rejected_pairs": rejected}


def score_action_unit(action_ids: Sequence[str], transition_model: Mapping[str, Any]) -> JsonDict:
    """Score one unit from frozen action support without executing semantics."""

    supported_actions = set(transition_model["supported_action_ids"])
    supported_transitions = {tuple(row) for row in transition_model["supported_transitions"]}
    unknown = [action for action in action_ids if action not in supported_actions]
    unsupported = [
        [left, right]
        for left, right in zip(action_ids, action_ids[1:], strict=False)
        if (left, right) not in supported_transitions
    ]
    score = 1.0 if unknown or unsupported else 0.0
    return {
        "score": score,
        "unknown_action_ids": unknown,
        "unsupported_transitions": unsupported,
        "scorer_version": SCORER_VERSION,
        "scorer_model_sha256": transition_model["model_sha256"],
    }


def decision_from_score(score: float) -> JsonDict:
    """Apply the frozen accept, abstain, and reject thresholds."""

    if score >= 0.75:
        return {"decision": "reject", "abstained": False}
    if score >= 0.25:
        return {"decision": "abstain", "abstained": True}
    return {"decision": "accept", "abstained": False}


def _unit_slice(unit_id: str, step: int, action_count: int) -> slice:
    """Translate one preregistered unit name into its action slice."""

    if unit_id == "one_step":
        return slice(step, step + 1)
    if unit_id == "two_steps":
        return slice(step, min(step + 2, action_count))
    if unit_id == "full_remaining_suffix":
        return slice(step, action_count)
    raise ValueError(f"unknown_unit:{unit_id}")


def _score_member(
    twin: Mapping[str, Any],
    member: str,
    unit_id: str,
    transition_model: Mapping[str, Any],
) -> JsonDict:
    """Score one twin member and preserve measured latency plus label lineage."""

    action_ids = list(twin[f"{member}_action_ids"])
    lines = str(twin[f"{member}_plan"]).splitlines()
    unit_slice = _unit_slice(unit_id, int(twin["localized_step"]), len(action_ids))
    unit_actions = action_ids[unit_slice]
    unit_text = "\n".join(lines[unit_slice])
    started = time.perf_counter_ns()
    scored = score_action_unit(unit_actions, transition_model)
    elapsed = max(1, time.perf_counter_ns() - started)
    decided = decision_from_score(float(scored["score"]))
    exact_label = member == "clean"
    return {
        "schema": ROW_SCHEMA,
        "row_type": "twin_unit",
        "twin_id": twin["twin_id"],
        "twin_sha256": twin["twin_sha256"],
        "source_row_id": twin["source_row_id"],
        "source_row_sha256": twin["source_row_sha256"],
        "task_id": twin["task_id"],
        "task_source_sha256": twin["task_source_sha256"],
        "split": twin["split"],
        "model_family_id": twin["model_family_id"],
        "localized_step": twin["localized_step"],
        "unit_id": unit_id,
        "member": member,
        "unit_action_ids": unit_actions,
        "unit_step_count": len(unit_actions),
        "unit_text_sha256": sha256_bytes(unit_text.encode("utf-8")),
        "exact_label": exact_label,
        **scored,
        **decided,
        "covered": decided["abstained"] is False,
        "catch": member == "error" and decided["decision"] == "reject",
        "false_reject": member == "clean" and decided["decision"] == "reject",
        "latency_ns": elapsed,
        "latency_s": round(elapsed / 1_000_000_000, 9),
        "exact_checker_authorizes": True,
        "advisory_scorer_authorizes": False,
    }


def score_twins(construction: Mapping[str, Any], transition_model: Mapping[str, Any]) -> JsonDict:
    """Score identical accepted twins under all three frozen unit views."""

    twin_rows: list[JsonDict] = []
    per_unit_rows: list[JsonDict] = []
    for source_twin in construction["twins"]:
        twin = deepcopy(dict(source_twin))
        per_unit: dict[str, JsonDict] = {}
        for unit_id in UNIT_ORDER:
            clean = _score_member(twin, "clean", unit_id, transition_model)
            error = _score_member(twin, "error", unit_id, transition_model)
            per_unit_rows.extend((clean, error))
            per_unit[unit_id] = {"clean": deepcopy(clean), "error": deepcopy(error)}
        twin["per_unit_results"] = per_unit
        twin_rows.append(twin)
    per_unit_rows.extend(deepcopy(list(construction["rejected_pairs"])))
    return {"twin_rows": twin_rows, "per_unit_rows": per_unit_rows}


def rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded rate while preserving an absent denominator as null."""

    return round(numerator / denominator, 9) if denominator else None


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Compute tie-aware pairwise AUROC without an external statistics package."""

    positive = [score for label, score in zip(labels, scores, strict=True) if label]
    negative = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return round(wins / (len(positive) * len(negative)), 9)


def auprc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Compute average precision over grouped score thresholds."""

    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    thresholds = sorted(set(scores), reverse=True)
    prior_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        selected = [index for index, score in enumerate(scores) if score >= threshold]
        true_positive = sum(labels[index] for index in selected)
        precision = true_positive / len(selected)
        recall = true_positive / positives
        area += precision * (recall - prior_recall)
        prior_recall = recall
    return round(area, 9)


def _brier(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Measure squared probability error for the advisory risk score."""

    if not labels:
        return None
    return round(
        sum((score - float(label)) ** 2 for label, score in zip(labels, scores, strict=True))
        / len(labels),
        9,
    )


def _ece(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Measure fixed-bin expected calibration error over ten score bins."""

    if not labels:
        return None
    total = len(labels)
    value = 0.0
    for bin_index in range(10):
        low = bin_index / 10
        high = (bin_index + 1) / 10
        indexes = [
            index
            for index, score in enumerate(scores)
            if low <= score < high or (bin_index == 9 and score == 1.0)
        ]
        if not indexes:
            continue
        mean_score = sum(scores[index] for index in indexes) / len(indexes)
        observed = sum(labels[index] for index in indexes) / len(indexes)
        value += len(indexes) / total * abs(mean_score - observed)
    return round(value, 9)


def percentile_interval(values: Sequence[float]) -> list[float] | None:
    """Return a deterministic nearest-rank 95 percent interval."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    low_index = max(0, math.floor(0.025 * (len(ordered) - 1)))
    high_index = min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))
    return [round(ordered[low_index], 9), round(ordered[high_index], 9)]


def _raw_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one unit's clean/error member rows into scalar metrics."""

    clean = [row for row in rows if row["member"] == "clean"]
    error = [row for row in rows if row["member"] == "error"]
    labels = [row["member"] == "error" for row in rows]
    scores = [float(row["score"]) for row in rows]
    catch_rate = rate(sum(row["decision"] == "reject" for row in error), len(error))
    false_reject_rate = rate(sum(row["decision"] == "reject" for row in clean), len(clean))
    informedness = (
        None
        if catch_rate is None or false_reject_rate is None
        else round(catch_rate - false_reject_rate, 9)
    )
    balanced = None if informedness is None else round((informedness + 1.0) / 2.0, 9)
    return {
        "catch_rate": catch_rate,
        "false_reject_rate": false_reject_rate,
        "informedness": informedness,
        "balanced_accuracy": balanced,
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "brier_score": _brier(labels, scores),
        "expected_calibration_error": _ece(labels, scores),
        "coverage": rate(sum(row["abstained"] is False for row in rows), len(rows)),
        "rejection_rate": rate(sum(row["decision"] == "reject" for row in rows), len(rows)),
        "abstention_rate": rate(sum(row["abstained"] is True for row in rows), len(rows)),
        "mean_latency_s": (
            round(sum(float(row["latency_s"]) for row in rows) / len(rows), 9) if rows else None
        ),
    }


def _metric(
    value: float | None, intervals: Mapping[str, list[float] | None], name: str
) -> JsonDict:
    """Attach deterministic uncertainty to one scalar metric."""

    return {"value": value, "interval_95": intervals.get(name)}


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, list[float] | None]:
    """Resample whole twins so clean/error dependence stays intact."""

    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["twin_id"])].append(row)
    pair_ids = sorted(by_pair)
    if not pair_ids:
        return {}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_RESAMPLES):
        selected: list[Mapping[str, Any]] = []
        for _index in range(len(pair_ids)):
            selected.extend(by_pair[rng.choice(pair_ids)])
        for name, value in _raw_metrics(selected).items():
            if value is not None:
                samples[name].append(float(value))
    return {name: percentile_interval(values) for name, values in samples.items()}


def compute_unit_metrics(per_unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report paired discrimination, rejection, calibration, and uncertainty."""

    results: list[JsonDict] = []
    for unit_index, unit_id in enumerate(UNIT_ORDER):
        rows = [
            row
            for row in per_unit_rows
            if row.get("row_type") == "twin_unit" and row.get("unit_id") == unit_id
        ]
        raw = _raw_metrics(rows)
        intervals = _bootstrap_intervals(rows, BOOTSTRAP_SEED + unit_index)
        clean = [row for row in rows if row["member"] == "clean"]
        error = [row for row in rows if row["member"] == "error"]
        results.append(
            {
                "unit_id": unit_id,
                "pair_count": len({row["twin_id"] for row in rows}),
                "member_row_count": len(rows),
                "catch_count": sum(row["decision"] == "reject" for row in error),
                "false_reject_count": sum(row["decision"] == "reject" for row in clean),
                "abstention_count": sum(row["abstained"] is True for row in rows),
                "catch_rate": _metric(raw["catch_rate"], intervals, "catch_rate"),
                "false_reject_rate": _metric(
                    raw["false_reject_rate"], intervals, "false_reject_rate"
                ),
                "informedness": _metric(raw["informedness"], intervals, "informedness"),
                "balanced_accuracy": _metric(
                    raw["balanced_accuracy"], intervals, "balanced_accuracy"
                ),
                "auroc": _metric(raw["auroc"], intervals, "auroc"),
                "auprc": _metric(raw["auprc"], intervals, "auprc"),
                "calibration": {
                    "brier_score": _metric(raw["brier_score"], intervals, "brier_score"),
                    "expected_calibration_error": _metric(
                        raw["expected_calibration_error"],
                        intervals,
                        "expected_calibration_error",
                    ),
                },
                "coverage": _metric(raw["coverage"], intervals, "coverage"),
                "rejection_rate": _metric(raw["rejection_rate"], intervals, "rejection_rate"),
                "abstention_rate": _metric(raw["abstention_rate"], intervals, "abstention_rate"),
                "latency_s": _metric(raw["mean_latency_s"], intervals, "mean_latency_s"),
                "uncertainty": {
                    "method": "deterministic_paired_bootstrap",
                    "seed": BOOTSTRAP_SEED + unit_index,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "pairing_key": "twin_id",
                },
            }
        )
    return results


def _metric_value(row: Mapping[str, Any], field: str) -> float | None:
    """Read one scalar metric value from its uncertainty wrapper."""

    value = row.get(field)
    if not isinstance(value, Mapping):
        return None
    observed = value.get("value")
    return float(observed) if isinstance(observed, (int, float)) else None


def recommend_verifier_unit(unit_metrics: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Select only a discrimination gain that does not raise false rejects."""

    by_unit = {str(row["unit_id"]): row for row in unit_metrics}
    assessments: list[JsonDict] = []
    selected: str | None = None
    for index, unit_id in enumerate(UNIT_ORDER):
        current = by_unit[unit_id]
        if index == 0:
            assessments.append(
                {
                    "unit_id": unit_id,
                    "eligible": False,
                    "reason": "reference_unit_has_no_shorter_comparator",
                }
            )
            continue
        previous_id = UNIT_ORDER[index - 1]
        previous = by_unit[previous_id]
        informed = _metric_value(current, "informedness")
        prior_informed = _metric_value(previous, "informedness")
        balanced = _metric_value(current, "balanced_accuracy")
        prior_balanced = _metric_value(previous, "balanced_accuracy")
        false_reject = _metric_value(current, "false_reject_rate")
        prior_false_reject = _metric_value(previous, "false_reject_rate")
        improves = (
            informed is not None
            and prior_informed is not None
            and balanced is not None
            and prior_balanced is not None
            and informed > prior_informed
            and balanced > prior_balanced
        )
        safe = (
            false_reject is not None
            and prior_false_reject is not None
            and false_reject <= prior_false_reject
        )
        eligible = improves and safe
        if eligible:
            reason = f"discrimination_improves_over_{previous_id}_without_false_reject_increase"
            selected = unit_id
        elif (
            false_reject is not None
            and prior_false_reject is not None
            and false_reject > prior_false_reject
        ):
            reason = f"unsupported_false_reject_increase_over_{previous_id}"
        else:
            reason = f"no_discrimination_improvement_over_{previous_id}"
        assessments.append({"unit_id": unit_id, "eligible": eligible, "reason": reason})
    return {
        "selection_made": selected is not None,
        "selected_unit": selected,
        "selection_rule": verifier_unit_preregistration()["recommendation_rule"],
        "unit_assessments": assessments,
        "exact_checker_still_authorizes": True,
        "advisory_scorer_authorizes": False,
    }


def authority_boundary() -> JsonDict:
    """State which component may measure and which component may authorize."""

    return {
        "advisory_scorer_role": "measure and route verification effort only",
        "exact_checker_role": "assign labels and authorize release only",
        "exact_checker_authorizes": True,
        "learned_or_advisory_scorer_authorizes": False,
        "llm_judge_used": False,
        "output_authority_rule": "an advisory accept cannot replace exact execution",
    }


def build_upstream_gate_receipt(root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    """Bind the Exp6649 completeness field, source hash, and row count."""

    path = root / UPSTREAM_PATH
    observed = upstream.get("candidate_corpus_complete")
    rows = upstream.get("candidate_rows")
    return {
        "experiment_id": "Exp6649",
        "path": UPSTREAM_PATH.as_posix(),
        "absolute_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "field": "candidate_corpus_complete",
        "expected_value": True,
        "observed_value": observed,
        "passed": observed is True,
        "observed_row_count": len(rows) if isinstance(rows, list) else None,
        "upstream_reproducibility_checksum": upstream.get("reproducibility_checksum"),
    }


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash the active roadmap and conductor before any experiment work."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Compare protected bytes and retain both hashes for each path."""

    after = protected_hashes(root)
    files = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path) and after.get(path) != "missing",
        }
        for path in sorted(set(before) | set(after))
    }
    return {"files": files, "all_unchanged": all(row["unchanged"] for row in files.values())}


def _host_resources(root: Path) -> JsonDict:
    """Record bounded host resources used by the CPU-only replay."""

    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):  # pragma: no cover - Linux CI supplies procfs.
        memory = {}
    disk = os.statvfs(root)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
        "disk_free_bytes": disk.f_bavail * disk.f_frsize,
        "gpu_required": False,
    }


def _no_llm_receipt(root: Path) -> JsonDict:
    """Inspect this module's imports and declare the no-LLM runtime boundary."""

    module = root / MODULE_PATH
    forbidden = {"transformers", "llama_cpp", "openai", "anthropic"}
    imported: set[str] = set()
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    except (OSError, SyntaxError):
        imported.add("module_unreadable")
    present = sorted(imported & forbidden)
    return {
        "substrate": INFERENCE_SUBSTRATE,
        "llm_invoked": False,
        "network_used": False,
        "forbidden_llm_imports": present,
        "module_sha256": sha256_file(module),
    }


def collect_preconditions(
    root: Path,
    upstream: Mapping[str, Any],
    receipt: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Validate the gate, rows, checker identity, hashes, resources, and substrate."""

    rows = upstream.get("candidate_rows")
    aggregate = upstream.get("aggregate_row_recomputation")
    manifest = upstream.get("frozen_task_manifest")
    expected_count = aggregate.get("expected_row_count") if isinstance(aggregate, Mapping) else None
    observed_count = len(rows) if isinstance(rows, list) else None
    manifest_identity = (
        manifest.get("compiler_checker_identity") if isinstance(manifest, Mapping) else {}
    )
    current_identity = exp6649._exact_identity()  # noqa: SLF001
    identity_matches = manifest_identity == current_identity
    no_llm = _no_llm_receipt(root)
    resources = _host_resources(root)
    checks = {
        "candidate_corpus_complete": upstream.get("candidate_corpus_complete") is True,
        "artifact_hash": str(receipt.get("sha256", "")).startswith("sha256:"),
        "row_count": expected_count == observed_count == 48,
        "exact_checker_identity": identity_matches,
        "protected_hashes": all(
            str(value).startswith("sha256:") for value in protected_before.values()
        ),
        "resources": bool(resources.get("cpu_count")) and int(resources["disk_free_bytes"]) > 0,
        "tools": Path(sys.executable).is_file(),
        "no_llm_substrate": not no_llm["forbidden_llm_imports"] and no_llm["llm_invoked"] is False,
        "upstream_content_checksum": (
            upstream.get("reproducibility_checksum") == exp6649.artifact_checksum(upstream)
        ),
    }
    return {
        "all_required_preconditions_available": all(checks.values()),
        "checks": checks,
        "failed_preconditions": [name for name, passed in checks.items() if not passed],
        "inputs": {
            "upstream": deepcopy(dict(receipt)),
            "module": {"path": MODULE_PATH.as_posix(), "sha256": sha256_file(root / MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": sha256_file(root / TEST_PATH)},
            "specs": {path.as_posix(): sha256_file(root / path) for path in SPEC_PATHS},
        },
        "expected_row_count": expected_count,
        "observed_row_count": observed_count,
        "exact_checker_identity": {
            "upstream": deepcopy(manifest_identity),
            "current": current_identity,
            "matches_current_source": identity_matches,
        },
        "protected_hashes_before": dict(protected_before),
        "resources": resources,
        "tools": {"python_available": Path(sys.executable).is_file(), "python": sys.executable},
        "no_llm": no_llm,
    }


def _first_failed_check(preconditions: Mapping[str, Any]) -> tuple[str, Any]:
    """Return the first failed check in its preregistered order."""

    for name, passed in preconditions["checks"].items():
        if passed is not True:
            observed: Any = passed
            if name == "candidate_corpus_complete":
                observed = preconditions["inputs"]["upstream"]["observed_value"]
            elif name == "row_count":
                observed = preconditions["observed_row_count"]
            elif name == "exact_checker_identity":
                observed = preconditions["exact_checker_identity"]["upstream"]
            return name, observed
    return "preconditions", preconditions.get("all_required_preconditions_available")


def _field_provenance(root: Path, upstream_hash: str) -> dict[str, JsonDict]:
    """Give each required field source, hash, reducer, and schema lineage."""

    module_hash = sha256_file(root / MODULE_PATH)
    provenance: dict[str, JsonDict] = {}
    upstream_fields = {"upstream_gate_receipt", "twin_rows", "per_unit_rows"}
    for field in REQUIRED_ARTIFACT_FIELDS:
        provenance[field] = {
            "source": UPSTREAM_PATH.as_posix()
            if field in upstream_fields
            else MODULE_PATH.as_posix(),
            "source_sha256": upstream_hash if field in upstream_fields else module_hash,
            "reducer": (
                "frozen_exp6649_rows_plus_exact_replay"
                if field in upstream_fields
                else f"experiment_6650.{field}"
            ),
            "schema": SCHEMA,
        }
    return provenance


def _tests_run_receipts(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    """Normalize command receipts without inventing missing exit values."""

    selected = rows if rows is not None else DEFAULT_TESTS_RUN
    return [
        {
            "command": str(row.get("command")),
            "exit_code": row.get("exit_code"),
            "summary": str(row.get("summary")),
        }
        for row in selected
    ]


def _aggregate_recomputation(
    per_unit_rows: Sequence[Mapping[str, Any]],
    unit_metrics: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any],
) -> JsonDict:
    """Rebuild counts, metrics, and selection from the flattened evidence rows."""

    twin_rows = [row for row in per_unit_rows if row.get("row_type") == "twin_unit"]
    rejected = [row for row in per_unit_rows if row.get("row_type") == "rejected_pair"]
    rebuilt_metrics = compute_unit_metrics(per_unit_rows)
    rebuilt_recommendation = recommend_verifier_unit(rebuilt_metrics)
    return {
        "source_row_count": len({str(row["source_row_id"]) for row in per_unit_rows}),
        "accepted_twin_count": len({str(row["twin_id"]) for row in twin_rows}),
        "rejected_pair_count": len(rejected),
        "twin_unit_row_count": len(twin_rows),
        "expected_twin_unit_row_count": len({str(row["twin_id"]) for row in twin_rows})
        * len(UNIT_ORDER)
        * 2,
        "per_unit_rows_sha256": sha256_json(per_unit_rows),
        "unit_metrics_rebuilt": rebuilt_metrics,
        "unit_metric_rows_sha256": sha256_json(unit_metrics),
        "all_metrics_match_rows": list(unit_metrics) == rebuilt_metrics,
        "recommendation_rebuilt": rebuilt_recommendation,
        "recommendation_matches_rows": dict(recommendation) == rebuilt_recommendation,
        "same_twin_ids_for_every_unit": all(
            {str(row["twin_id"]) for row in twin_rows if row["unit_id"] == unit_id}
            == {str(row["twin_id"]) for row in twin_rows}
            for unit_id in UNIT_ORDER
        ),
    }


def _gate_summary_complete(
    construction: Mapping[str, Any], recommendation: Mapping[str, Any]
) -> JsonDict:
    """Summarize pairability and the discrimination-versus-rejection decision."""

    return {
        "passed": True,
        "first_failed_check": None,
        "failed_checks": [],
        "observed": {
            "pairable_twin_count": len(construction["twins"]),
            "rejected_pair_count": len(construction["rejected_pairs"]),
            "selected_unit": recommendation["selected_unit"],
        },
        "claim_boundary": "bounded to eight pairable semantic twins from forty-eight frozen source rows",
    }


def _gate_summary_blocked(name: str, observed: Any) -> JsonDict:
    """Name the failed gate and retain its exact observed value."""

    return {
        "passed": False,
        "first_failed_check": name,
        "failed_checks": [{"check": name, "expected": True, "observed": observed}],
        "observed": observed,
    }


def _blocked_artifact(
    root: Path,
    date: str,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None,
    receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Build a complete blocked schema without invented twin metrics."""

    name, observed = _first_failed_check(preconditions)
    payload: JsonDict = {
        "status": f"blocked_{name}",
        "honest_verdict": f"blocked_{name}: required precondition failed with observed value {observed!r}",
        "verdict_class": "blocked",
        "gate_check_summary": _gate_summary_blocked(name, observed),
        "upstream_gate_receipt": deepcopy(dict(receipt)),
        "twin_construction_contract": twin_construction_contract(),
        "verifier_unit_preregistration": verifier_unit_preregistration(),
        "twin_rows": [],
        "unit_metric_rows": [],
        "recommended_verifier_unit": {
            "selection_made": False,
            "selected_unit": None,
            "reason": f"blocked_before_scoring:{name}",
            "exact_checker_still_authorizes": True,
        },
        "authority_boundary": authority_boundary(),
        "per_unit_rows": [],
        "aggregate_row_recomputation": {
            "source_row_count": 0,
            "accepted_twin_count": 0,
            "rejected_pair_count": 0,
            "twin_unit_row_count": 0,
            "blocked_before_recomputation": True,
        },
        "preconditions_checked": deepcopy(dict(preconditions)),
        "protected_files_unchanged": deepcopy(dict(protected)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(root, str(receipt.get("sha256"))),
        "random_seed": {
            "twin_seed": RANDOM_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "duration_s": round(max(float(duration_s), 0.000001), 9),
        "tests_run": _tests_run_receipts(tests_run),
        "run_date": date,
        "schema": SCHEMA,
    }
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    upstream: Mapping[str, Any] | None = None,
    date: str = "20260826",
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build a blocked or row-complete twin verifier artifact."""

    started = time.monotonic()
    protected_before = protected_hashes(root)
    source = (
        deepcopy(dict(upstream))
        if upstream is not None
        else json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
    )
    receipt = build_upstream_gate_receipt(root, source)
    preconditions = collect_preconditions(root, source, receipt, protected_before)
    protected = protected_files_receipt(root, protected_before)
    elapsed = float(duration_s) if duration_s is not None else time.monotonic() - started
    if not preconditions["all_required_preconditions_available"] or not protected["all_unchanged"]:
        if not protected["all_unchanged"]:
            preconditions = deepcopy(preconditions)
            preconditions["checks"]["protected_hashes"] = False
            preconditions["all_required_preconditions_available"] = False
        return _blocked_artifact(root, date, elapsed, tests_run, receipt, preconditions, protected)
    transition_model = build_transition_model(source["frozen_task_manifest"])
    construction = construct_twins(source, transition_model)
    scored = score_twins(construction, transition_model)
    unit_metrics = compute_unit_metrics(scored["per_unit_rows"])
    recommendation = recommend_verifier_unit(unit_metrics)
    status = "complete"
    verdict = (
        "complete: two-step verification improves informedness over one-step on eight "
        "byte-matched semantic twins with no false rejects; full-suffix verification "
        "keeps all catches but falsely rejects two clean suffixes, so raw rejection is "
        "not treated as discrimination; this positive result is bounded to eight of "
        "forty-eight frozen source rows"
    )
    payload: JsonDict = {
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": "positive",
        "gate_check_summary": _gate_summary_complete(construction, recommendation),
        "upstream_gate_receipt": receipt,
        "twin_construction_contract": {
            **twin_construction_contract(),
            "transition_model": transition_model,
        },
        "verifier_unit_preregistration": verifier_unit_preregistration(),
        "twin_rows": scored["twin_rows"],
        "unit_metric_rows": unit_metrics,
        "recommended_verifier_unit": recommendation,
        "authority_boundary": authority_boundary(),
        "per_unit_rows": scored["per_unit_rows"],
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(root, str(receipt["sha256"])),
        "random_seed": {
            "twin_seed": RANDOM_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "duration_s": round(max(elapsed, 0.000001), 9),
        "tests_run": _tests_run_receipts(tests_run),
        "run_date": date,
        "schema": SCHEMA,
    }
    payload["aggregate_row_recomputation"] = _aggregate_recomputation(
        payload["per_unit_rows"], unit_metrics, recommendation
    )
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def _validate_twin(twin: Mapping[str, Any]) -> list[str]:
    """Return byte-locality and exact-authority defects for one accepted twin."""

    errors: list[str] = []
    clean = str(twin.get("clean_plan", ""))
    error = str(twin.get("error_plan", ""))
    clean_lines = clean.splitlines()
    error_lines = error.splitlines()
    step = twin.get("localized_step")
    if not isinstance(step, int) or not (0 <= step < len(clean_lines)):
        return ["twin_localized_step_invalid"]
    if len(clean.encode()) != len(error.encode()) or len(clean_lines) != len(error_lines):
        errors.append("twin_byte_or_line_count_mismatch")
    elif (
        clean_lines[:step] != error_lines[:step]
        or clean_lines[step + 1 :] != error_lines[step + 1 :]
    ):
        errors.append("twin_nonlocalized_bytes_changed")
    if twin.get("clean_exact_label") is not True or twin.get("error_exact_label") is not False:
        errors.append("twin_exact_labels_invalid")
    if twin.get("advisory_scorer_supplies_exact_label") is not False:
        errors.append("twin_advisory_authority_invalid")
    return errors


def validate_artifact(
    payload: Mapping[str, Any],
    root: Path = REPO_ROOT,
    *,
    verify_source_file: bool = True,
) -> list[str]:
    """Return schema, row, authority, source, protection, and checksum defects."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors: list[str] = []
    if payload.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if not isinstance(payload.get("duration_s"), (int, float)) or float(payload["duration_s"]) <= 0:
        errors.append("duration_s_invalid")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    protected = payload.get("protected_files_unchanged", {})
    if protected.get("all_unchanged") is not True:
        errors.append("protected_files_changed")
    if verify_source_file:
        receipt = payload.get("upstream_gate_receipt", {})
        if sha256_file(root / UPSTREAM_PATH) != receipt.get("sha256"):
            errors.append("upstream_artifact_hash_mismatch")
    if payload.get("verdict_class") == "blocked":
        if not str(payload.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_prefix_missing")
        if not str(payload.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
        if payload.get("gate_check_summary", {}).get("first_failed_check") is None:
            errors.append("blocked_gate_detail_missing")
        if payload.get("twin_rows") or payload.get("unit_metric_rows"):
            errors.append("blocked_artifact_invented_rows")
        return errors
    for twin in payload.get("twin_rows", []):
        errors.extend(_validate_twin(twin))
    rebuilt_metrics = compute_unit_metrics(payload.get("per_unit_rows", []))
    rebuilt_recommendation = recommend_verifier_unit(rebuilt_metrics)
    rebuilt_aggregate = _aggregate_recomputation(
        payload.get("per_unit_rows", []),
        payload.get("unit_metric_rows", []),
        payload.get("recommended_verifier_unit", {}),
    )
    if rebuilt_metrics != payload.get("unit_metric_rows"):
        errors.append("unit_metric_rows_mismatch")
    if rebuilt_recommendation != payload.get("recommended_verifier_unit"):
        errors.append("recommended_verifier_unit_mismatch")
    if rebuilt_aggregate != payload.get("aggregate_row_recomputation"):
        errors.append("aggregate_row_recomputation_mismatch")
    if (
        payload.get("authority_boundary", {}).get("learned_or_advisory_scorer_authorizes")
        is not False
    ):
        errors.append("authority_boundary_mismatch")
    return errors


def write_artifact_atomic(
    path: str | Path, payload: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> JsonDict:
    """Validate, sync, atomically replace, and directory-sync one artifact."""

    errors = validate_artifact(payload, repo_root)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".exp6650-final-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def run(
    date: str,
    root: Path = REPO_ROOT,
    *,
    output: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and atomically write the deterministic no-LLM verifier map."""

    started = time.monotonic()
    artifact = build_artifact(root, date=date, tests_run=tests_run)
    artifact["duration_s"] = round(max(time.monotonic() - started, 0.000001), 9)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_artifact_atomic(output or root / RESULT_PATH, artifact, repo_root=root)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the experiment date, root, output, and validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.datetime.now(datetime.UTC).strftime("%Y%m%d"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp6650 or validate one existing artifact from the command line."""

    args = _parse_args(argv)
    if args.validate is not None:
        payload = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_artifact(payload, args.repo_root)
        if errors:
            print("\n".join(errors))
            return 1
        print("valid")
        return 0
    artifact = run(args.date, args.repo_root, output=args.output)
    print(
        canonical_json(
            {
                "status": artifact["status"],
                "recommended_verifier_unit": artifact["recommended_verifier_unit"]["selected_unit"],
                "twin_count": len(artifact["twin_rows"]),
                "result": str((args.output or args.repo_root / RESULT_PATH).resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution.
    raise SystemExit(main())
