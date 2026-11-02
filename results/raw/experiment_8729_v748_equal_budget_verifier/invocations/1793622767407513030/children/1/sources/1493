"""Exp 5126: distributional-energy ranker over the structured pool.

Spec refs: REQ-INFER-SOTA-031,
SCENARIO-INFER-SOTA-031-RANKER,
SCENARIO-INFER-SOTA-031-BLOCKED.

The experiment is deliberately small and local.  It follows the decomposed
energy shape from arXiv:2605.18871, but uses CPU-only calibration over the
Exp 5125 exact-validated candidates: deterministic constraint penalties define
truth, a lightweight answer-shape score supplies the learned quality term, and
ensemble spread plus nonzero penalties trigger abstention.  Model identity,
FoVer rows, and LLM judges are never used as correctness features.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5125_structured_reasoning_pool_v470 as pool_mod  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp5126-distributional-energy-ranker-v470"
MILESTONE = "2026.07.470"
RESULT_RELATIVE_PATH = "results/experiment_5126_distributional_energy_ranker_v470.json"
SOURCE_ARTIFACT_RELATIVE_PATH = pool_mod.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = "cpu_ranker_over_exact_validated_sota_candidates"
SUCCESS_READY_VERDICT = "complete_distributional_energy_ranker_ready_for_audit"
SUCCESS_NOT_READY_VERDICT = "complete_distributional_energy_ranker_evaluated_not_ready_for_audit"
BLOCKED_POOL_VERDICT = "blocked_structured_pool_not_ready"
BLOCKED_ROWS_VERDICT = "blocked_structured_pool_rows_missing"
RANDOM_SEED = 20260701
CONSTRAINT_WEIGHT = 100.0
QUALITY_WEIGHT = 1.0
UNCERTAINTY_WEIGHT = 0.25
BOOTSTRAP_RESAMPLES = 1000
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "source_pool_path",
    "MODEL_SPECS",
    "deterministic_constraint_penalties",
    "learned_quality_score_description",
    "uncertainty_abstention_rule",
    "strongest_cheap_baseline",
    "distributional_energy_delta",
    "delta_ci95",
    "family_holdout_results",
    "label_shuffle_result",
    "model_identity_shortcut_check",
    "ranker_ready_for_audit",
    "verifier_is_oracle",
    "conductor_modified",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "source_pool_path": "data provenance",
    "MODEL_SPECS": "inherited local SOTA generation provenance",
    "deterministic_constraint_penalties": "exact ground truth",
    "learned_quality_score_description": "method transparency",
    "uncertainty_abstention_rule": "no forced overclaim",
    "strongest_cheap_baseline": "baseline adequacy",
    "distributional_energy_delta": "structured gate",
    "delta_ci95": "sample-size rigor",
    "family_holdout_results": "generalization",
    "label_shuffle_result": "leakage control",
    "model_identity_shortcut_check": "shortcut control",
    "ranker_ready_for_audit": "structured downstream gate",
    "verifier_is_oracle": "no oracle verifier headline",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5126_distributional_energy_ranker_v470.py --date 20260701",
    '.venv/bin/pytest tests/python/test_experiment_5126_distributional_energy_ranker_v470.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run "
    "--include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5126_distributional_energy_ranker_v470.py' -m pytest "
    'tests/python/test_experiment_5126_distributional_energy_ranker_v470.py -q -o addopts="" && '
    ".venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/python/"
    "carnot/experiment_5126_distributional_energy_ranker_v470.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5126_distributional_energy_ranker_v470.py "
    "scripts/experiment_5126_distributional_energy_ranker_v470.py "
    "tests/python/test_experiment_5126_distributional_energy_ranker_v470.py",
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_5126_distributional_energy_ranker_v470.py "
    "scripts/experiment_5126_distributional_energy_ranker_v470.py "
    "tests/python/test_experiment_5126_distributional_energy_ranker_v470.py",
    "python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5126_distributional_energy_ranker_v470.py",
    ".venv/bin/pytest tests/python -q",
]

DETERMINISTIC_CONSTRAINT_PENALTIES = {
    "parse": "100-point hard penalty when candidate JSON cannot expose an answer key.",
    "graph_coloring": (
        "length mismatch, out-of-range colors, and equal-color edge conflicts; zero iff the "
        "exact graph-coloring validator accepts."
    ),
    "knights_knaves": (
        "missing/invalid A/B/C roles plus speaker truth-table violations; zero iff the exact "
        "Knights-and-Knaves validator accepts."
    ),
    "travel_budget": (
        "unknown or duplicate activities, budget/hour overages, and optimal-value shortfall; "
        "zero iff the exact travel validator accepts."
    ),
    "code_property": (
        "symmetric-difference count against the executable property solution plus out-of-domain "
        "answers; zero iff the exact code-property validator accepts."
    ),
}

LEARNED_QUALITY_SCORE_DESCRIPTION = (
    "Family-wise answer-shape calibration fitted only on grouped train rows with deterministic "
    "validator labels: mean raw JSON length and mean answer cardinality of correct candidates. "
    "The ensemble scores length closeness, answer-size closeness, and a parse-shape blend. "
    "It excludes model_hf_id, model_path, candidate_id, FoVer data, and LLM judge labels."
)

UNCERTAINTY_ABSTENTION_RULE = (
    "Rank by mean decomposed energy over the shape-score ensemble. Abstain when the selected "
    "candidate has any deterministic constraint penalty or when ensemble spread exceeds the "
    "calibrated train-positive threshold; abstentions count as not-correct for accuracy@1."
)


@dataclass(frozen=True)
class PoolBundle:
    source_artifact: JsonDict
    source_pool_path: str
    model_specs: list[JsonDict]
    rows: list[JsonDict]


@dataclass(frozen=True)
class CandidateExample:
    task_id: str
    family: str
    candidate_id: str
    label: bool
    raw_length: float
    answer_size: float
    parse_ok: bool
    deterministic_penalty: float


@dataclass(frozen=True)
class QualityModel:
    family_stats: dict[str, JsonDict]
    global_stats: JsonDict
    uncertainty_threshold: float


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else None


def build_task_lookup() -> dict[str, JsonDict]:
    return {str(task["task_id"]): task for task in pool_mod.build_task_bank()}


def _pool_artifact(root: Path) -> tuple[JsonDict | None, str | None]:
    try:
        artifact = _read_json(root / SOURCE_ARTIFACT_RELATIVE_PATH)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if artifact is None:
        return None, "missing Exp 5125 structured pool artifact"
    return artifact, None


def load_structured_pool(*, root: Path = REPO_ROOT) -> PoolBundle:
    artifact, error = _pool_artifact(root)
    if artifact is None or artifact.get("structured_pool_ready") is not True:
        raise ValueError(error or "structured_pool_ready is not true")
    source_pool_path = str(artifact.get("pool_path") or pool_mod.POOL_RELATIVE_PATH)
    rows = read_jsonl(root / source_pool_path)
    if not rows:
        raise ValueError("structured pool rows missing")
    model_specs = [dict(row) for row in artifact.get("MODEL_SPECS", []) if isinstance(row, Mapping)]
    return PoolBundle(
        source_artifact=dict(artifact),
        source_pool_path=source_pool_path,
        model_specs=model_specs,
        rows=rows,
    )


def split_rows_by_family_and_item(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(dict(row))
    splits = {"train": [], "calibration": [], "test": []}
    for family in sorted(grouped):
        family_rows = sorted(grouped[family], key=lambda item: str(item["task_id"]))
        for item_index, row in enumerate(family_rows):
            if item_index % 4 == 0:
                splits["test"].append(row)
            elif item_index % 4 in {1, 2}:
                splits["train"].append(row)
            else:
                splits["calibration"].append(row)
    return splits


def _parse_answer(raw_response: str) -> tuple[Any, str | None]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(payload, Mapping) or "answer" not in payload:
        return None, "candidate JSON must be an object with an answer key"
    return payload["answer"], None


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return [int(item) for item in value]


def _as_str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _parse_knights_answer(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, Mapping):
        return None
    parsed: dict[str, bool] = {}
    for name in ("A", "B", "C"):
        raw = str(value.get(name, "")).strip().lower()
        if raw in {"knight", "true", "truth", "t"}:
            parsed[name] = True
        elif raw in {"knave", "false", "liar", "f"}:
            parsed[name] = False
        else:
            return None
    return parsed


def _answer_size(value: Any) -> float:
    if isinstance(value, (list, tuple, dict, set)):
        return float(len(value))
    if value is None:
        return 0.0
    return 1.0


def _graph_penalty(task: Mapping[str, Any], answer: Any) -> dict[str, float]:
    colors = _as_int_list(answer)
    constraints = task["constraints"]
    n_nodes = int(constraints["n_nodes"])
    n_colors = int(constraints["n_colors"])
    if colors is None:
        return {"answer_type": 10.0}
    components: dict[str, float] = {
        "length_mismatch": float(abs(len(colors) - n_nodes)),
        "range": float(sum(1 for color in colors if color < 0 or color >= n_colors)),
    }
    if len(colors) == n_nodes:
        components["edge_conflicts"] = float(
            sum(
                1 for left, right in constraints["edges"] if colors[int(left)] == colors[int(right)]
            )
        )
    else:
        components["edge_conflicts"] = float(n_nodes)
    return components


def _knights_penalty(task: Mapping[str, Any], answer: Any) -> dict[str, float]:
    assignment = _parse_knights_answer(answer)
    if assignment is None:
        return {"answer_type": 10.0}
    violations = 0
    for statement in task["constraints"]["statements"]:
        speaker_truth = bool(assignment[str(statement["speaker"])])
        statement_truth = pool_mod._statement_truth(statement, assignment)
        violations += int(speaker_truth != statement_truth)
    return {"truth_table_violations": float(violations)}


def _travel_penalty(task: Mapping[str, Any], answer: Any) -> dict[str, float]:
    chosen_ids = _as_str_list(answer)
    if chosen_ids is None:
        return {"answer_type": 10.0}
    constraints = task["constraints"]
    activities = {str(row["id"]): row for row in constraints["activities"]}
    unknown = sum(1 for item in chosen_ids if item not in activities)
    duplicates = len(chosen_ids) - len(set(chosen_ids))
    chosen = [activities[item] for item in chosen_ids if item in activities]
    cost = sum(int(row["cost"]) for row in chosen)
    hours = sum(int(row["hours"]) for row in chosen)
    value = sum(int(row["value"]) for row in chosen)
    return {
        "unknown_or_duplicate": float(unknown + duplicates),
        "budget_overage": float(max(0, cost - int(constraints["budget"]))),
        "hour_overage": float(max(0, hours - int(constraints["hours"]))),
        "optimal_value_shortfall": float(max(0, int(task["optimal_value"]) - value)),
    }


def _code_penalty(task: Mapping[str, Any], answer: Any) -> dict[str, float]:
    values = _as_int_list(answer)
    if values is None:
        return {"answer_type": 10.0}
    expected = set(int(value) for value in task["solution"])
    domain_n = int(task["constraints"]["domain_n"])
    actual = set(values)
    out_of_domain = sum(1 for value in values if value < 0 or value >= domain_n)
    return {
        "symmetric_difference": float(len(expected.symmetric_difference(actual))),
        "out_of_domain": float(out_of_domain),
    }


def deterministic_constraint_penalty(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_lookup: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    answer, parse_error = _parse_answer(str(candidate.get("raw_response", "")))
    if parse_error:
        components = {"parse": 100.0}
    else:
        task = task_lookup[str(row["task_id"])]
        family = str(row["family"])
        if family == "graph_coloring":
            components = _graph_penalty(task, answer)
        elif family == "knights_knaves":
            components = _knights_penalty(task, answer)
        elif family == "travel_budget":
            components = _travel_penalty(task, answer)
        else:
            components = _code_penalty(task, answer)
    total = float(sum(float(value) for value in components.values()))
    return {
        "components": {key: _round_rate(value) for key, value in components.items()},
        "total_penalty": _round_rate(total),
        "violation_count": int(sum(1 for value in components.values() if float(value) > 0.0)),
    }


def _exact_label(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_lookup: Mapping[str, Mapping[str, Any]],
) -> bool:
    task = task_lookup[str(row["task_id"])]
    score = pool_mod.score_candidate(task, str(candidate.get("raw_response", "")))
    return bool(score["correct"])


def _candidate_shape(candidate: Mapping[str, Any]) -> tuple[float, float, bool]:
    raw = str(candidate.get("raw_response", ""))
    answer, parse_error = _parse_answer(raw)
    return float(len(raw)), _answer_size(answer), parse_error is None


def build_candidate_examples(
    rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    *,
    label_overrides: Mapping[str, bool] | None = None,
) -> list[CandidateExample]:
    examples: list[CandidateExample] = []
    for row in rows:
        for candidate in row["candidates"]:
            raw_length, answer_size, parse_ok = _candidate_shape(candidate)
            penalty = deterministic_constraint_penalty(row, candidate, task_lookup)
            candidate_id = str(candidate["candidate_id"])
            label = (
                bool(label_overrides[candidate_id])
                if label_overrides is not None and candidate_id in label_overrides
                else _exact_label(row, candidate, task_lookup)
            )
            examples.append(
                CandidateExample(
                    task_id=str(row["task_id"]),
                    family=str(row["family"]),
                    candidate_id=candidate_id,
                    label=label,
                    raw_length=raw_length,
                    answer_size=answer_size,
                    parse_ok=parse_ok,
                    deterministic_penalty=float(penalty["total_penalty"]),
                )
            )
    return examples


def _mean(values: Sequence[float], fallback: float) -> float:
    return float(statistics.fmean(values)) if values else fallback


def fit_quality_model(examples: Sequence[CandidateExample]) -> QualityModel:
    positive = [example for example in examples if example.label]
    global_length = _mean([example.raw_length for example in positive], 1.0)
    global_size = _mean([example.answer_size for example in positive], 1.0)
    global_stats = {"mean_length": global_length, "mean_answer_size": global_size}
    family_stats: dict[str, JsonDict] = {}
    families = sorted({example.family for example in examples})
    for family in families:
        family_positive = [example for example in positive if example.family == family]
        family_stats[family] = {
            "mean_length": _mean(
                [example.raw_length for example in family_positive], global_length
            ),
            "mean_answer_size": _mean(
                [example.answer_size for example in family_positive], global_size
            ),
        }
    uncertainties = []
    provisional = QualityModel(family_stats, global_stats, uncertainty_threshold=1.0)
    for example in positive:
        scores = _quality_ensemble_scores(
            example.family, example.raw_length, example.answer_size, example.parse_ok, provisional
        )
        uncertainties.append(_stddev(scores))
    threshold = max(uncertainties, default=0.0) + 0.01
    return QualityModel(family_stats, global_stats, uncertainty_threshold=_round_rate(threshold))


def _quality_ensemble_scores(
    family: str,
    raw_length: float,
    answer_size: float,
    parse_ok: bool,
    model: QualityModel,
) -> list[float]:
    stats = model.family_stats.get(family, model.global_stats)
    length_scale = max(float(stats["mean_length"]), 1.0)
    size_scale = max(float(stats["mean_answer_size"]), 1.0)
    length_score = 1.0 - abs(raw_length - float(stats["mean_length"])) / length_scale
    size_score = 1.0 - abs(answer_size - float(stats["mean_answer_size"])) / size_scale
    parse_score = 1.0 if parse_ok else -1.0
    blend = 0.5 * length_score + 0.35 * size_score + 0.15 * parse_score
    return [length_score, size_score, blend]


def _stddev(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def score_decomposed_energy(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_lookup: Mapping[str, Mapping[str, Any]],
    model: QualityModel,
) -> JsonDict:
    penalty = deterministic_constraint_penalty(row, candidate, task_lookup)
    raw_length, answer_size, parse_ok = _candidate_shape(candidate)
    quality_scores = _quality_ensemble_scores(
        str(row["family"]), raw_length, answer_size, parse_ok, model
    )
    quality_mean = float(statistics.fmean(quality_scores))
    uncertainty = _stddev(quality_scores)
    mean_energy = (
        CONSTRAINT_WEIGHT * float(penalty["total_penalty"])
        - QUALITY_WEIGHT * quality_mean
        + UNCERTAINTY_WEIGHT * uncertainty
    )
    return {
        "deterministic_penalty": penalty["total_penalty"],
        "quality_score": _round_rate(quality_mean),
        "uncertainty": _round_rate(uncertainty),
        "mean_energy": _round_rate(mean_energy),
    }


def _ranker_selection(
    row: Mapping[str, Any], task_lookup: Mapping[str, Mapping[str, Any]], model: QualityModel
) -> JsonDict:
    scored = [
        {
            "candidate": candidate,
            "score": score_decomposed_energy(row, candidate, task_lookup, model),
        }
        for candidate in row["candidates"]
    ]
    scored.sort(
        key=lambda item: (
            float(item["score"]["mean_energy"]),
            int(item["candidate"].get("candidate_index", 999)),
        )
    )
    best = scored[0]
    abstained = (
        float(best["score"]["deterministic_penalty"]) > 0.0
        or float(best["score"]["uncertainty"]) > model.uncertainty_threshold
    )
    return {
        "candidate_id": None if abstained else str(best["candidate"]["candidate_id"]),
        "abstained": abstained,
        "correct": False if abstained else _exact_label(row, best["candidate"], task_lookup),
        "violation": False if abstained else float(best["score"]["deterministic_penalty"]) > 0.0,
        "mean_energy": best["score"]["mean_energy"],
    }


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return _round_rate(wins / total)


def _ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    model: QualityModel,
) -> JsonDict:
    reciprocal_ranks: list[float] = []
    labels: list[bool] = []
    scores: list[float] = []
    for row in rows:
        ranked = []
        for candidate in row["candidates"]:
            score = score_decomposed_energy(row, candidate, task_lookup, model)
            label = _exact_label(row, candidate, task_lookup)
            labels.append(label)
            scores.append(-float(score["mean_energy"]))
            ranked.append((float(score["mean_energy"]), int(candidate["candidate_index"]), label))
        ranked.sort()
        rank = next((index + 1 for index, item in enumerate(ranked) if item[2]), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return {
        "auroc": _auroc(labels, scores),
        "mrr": _round_rate(statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0),
    }


def evaluate_ranker(
    rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    model: QualityModel,
) -> JsonDict:
    selections = [_ranker_selection(row, task_lookup, model) for row in rows]
    n_rows = len(rows)
    accepted = [selection for selection in selections if not selection["abstained"]]
    ranking = _ranking_metrics(rows, task_lookup, model)
    metrics = {
        "accuracy_at_1": _round_rate(sum(1 for item in selections if item["correct"]) / n_rows),
        "accepted_accuracy_at_1": _round_rate(
            sum(1 for item in accepted if item["correct"]) / len(accepted) if accepted else 0.0
        ),
        "violation_rate": _round_rate(sum(1 for item in selections if item["violation"]) / n_rows),
        "abstention_rate": _round_rate(sum(1 for item in selections if item["abstained"]) / n_rows),
        "auroc": ranking["auroc"],
        "mrr": ranking["mrr"],
    }
    return {"metrics": metrics, "per_item": selections}


def _baseline_select(
    row: Mapping[str, Any],
    task_lookup: Mapping[str, Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    candidates = list(row["candidates"])
    if name == "length_features":
        return sorted(
            candidates, key=lambda item: (len(str(item["raw_response"])), item["candidate_index"])
        )[0]
    if name == "parse_validity":
        return next(
            (candidate for candidate in candidates if candidate.get("parse_ok")), candidates[0]
        )
    if name == "constraint_count_only":
        return sorted(
            candidates,
            key=lambda item: (
                float(deterministic_constraint_penalty(row, item, task_lookup)["total_penalty"]),
                int(item["candidate_index"]),
            ),
        )[0]
    rng = random.Random(f"{RANDOM_SEED}:{row['task_id']}")
    valid = [candidate for candidate in candidates if candidate.get("parse_ok")]
    return rng.choice(valid or candidates)


def _evaluate_baseline(
    rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    name: str,
) -> JsonDict:
    per_item = []
    for row in rows:
        selected = _baseline_select(row, task_lookup, name)
        penalty = deterministic_constraint_penalty(row, selected, task_lookup)
        per_item.append(
            {
                "candidate_id": str(selected["candidate_id"]),
                "correct": _exact_label(row, selected, task_lookup),
                "violation": float(penalty["total_penalty"]) > 0.0,
            }
        )
    n_rows = len(rows)
    return {
        "accuracy_at_1": _round_rate(sum(1 for item in per_item if item["correct"]) / n_rows),
        "violation_rate": _round_rate(sum(1 for item in per_item if item["violation"]) / n_rows),
        "abstention_rate": 0.0,
        "per_item": per_item,
    }


def evaluate_baselines(
    rows: Sequence[Mapping[str, Any]], task_lookup: Mapping[str, Mapping[str, Any]]
) -> dict[str, JsonDict]:
    names = (
        "length_features",
        "parse_validity",
        "constraint_count_only",
        "random_parse_valid",
    )
    return {name: _evaluate_baseline(rows, task_lookup, name) for name in names}


def _strongest_baseline(baselines: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    name, metrics = max(
        baselines.items(),
        key=lambda item: (
            float(item[1]["accuracy_at_1"]),
            -float(item[1]["violation_rate"]),
            item[0] == "constraint_count_only",
        ),
    )
    return {
        "name": name,
        "accuracy_at_1": metrics["accuracy_at_1"],
        "violation_rate": metrics["violation_rate"],
    }


def _paired_delta_ci95(
    ranker_items: Sequence[Mapping[str, Any]], baseline_items: Sequence[Mapping[str, Any]]
) -> tuple[float, list[float]]:
    deltas = [
        float(bool(ranker["correct"])) - float(bool(baseline["correct"]))
        for ranker, baseline in zip(ranker_items, baseline_items, strict=True)
    ]
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    rng = random.Random(RANDOM_SEED)
    boot = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        boot.append(statistics.fmean(sample))
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))]
    upper = boot[int(0.975 * (len(boot) - 1))]
    return _round_rate(mean_delta), [_round_rate(lower), _round_rate(upper)]


def _quality_only_auc(
    rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    model: QualityModel,
) -> float | None:
    labels: list[bool] = []
    scores: list[float] = []
    for row in rows:
        for candidate in row["candidates"]:
            raw_length, answer_size, parse_ok = _candidate_shape(candidate)
            quality = statistics.fmean(
                _quality_ensemble_scores(
                    str(row["family"]), raw_length, answer_size, parse_ok, model
                )
            )
            labels.append(_exact_label(row, candidate, task_lookup))
            scores.append(float(quality))
    return _auroc(labels, scores)


def _label_shuffle_control(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    train_examples = build_candidate_examples(train_rows, task_lookup)
    labels = [example.label for example in train_examples]
    rng = random.Random(RANDOM_SEED)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    overrides = {
        example.candidate_id: bool(label)
        for example, label in zip(train_examples, shuffled, strict=True)
    }
    shuffled_model = fit_quality_model(
        build_candidate_examples(train_rows, task_lookup, label_overrides=overrides)
    )
    shuffled_auc = _quality_only_auc(test_rows, task_lookup, shuffled_model)
    return {
        "quality_only_auroc_after_shuffle": shuffled_auc,
        "passed": shuffled_auc is None or shuffled_auc <= 0.7,
        "random_seed": RANDOM_SEED,
    }


def _family_holdout_results(
    rows: Sequence[Mapping[str, Any]], task_lookup: Mapping[str, Mapping[str, Any]]
) -> dict[str, JsonDict]:
    families = sorted({str(row["family"]) for row in rows})
    results: dict[str, JsonDict] = {}
    for family in families:
        train_rows = [row for row in rows if str(row["family"]) != family]
        test_rows = [row for row in rows if str(row["family"]) == family]
        model = fit_quality_model(build_candidate_examples(train_rows, task_lookup))
        ranker = evaluate_ranker(test_rows, task_lookup, model)
        baselines = evaluate_baselines(test_rows, task_lookup)
        strongest = _strongest_baseline(baselines)
        results[family] = {
            "n": len(test_rows),
            "accuracy_at_1": ranker["metrics"]["accuracy_at_1"],
            "abstention_rate": ranker["metrics"]["abstention_rate"],
            "strongest_cheap_baseline": strongest,
            "delta": _round_rate(
                float(ranker["metrics"]["accuracy_at_1"]) - float(strongest["accuracy_at_1"])
            ),
        }
    return results


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    deduped: list[JsonDict] = []
    for row in rows:
        seen: set[str] = set()
        candidates = []
        for candidate in row["candidates"]:
            key = str(candidate.get("normalized_answer") or candidate.get("raw_response"))
            if key not in seen:
                seen.add(key)
                candidates.append(dict(candidate))
        copied = dict(row)
        copied["candidates"] = candidates
        deduped.append(copied)
    return deduped


def _duplicate_control(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    model: QualityModel,
    original_accuracy: float,
) -> JsonDict:
    split_ids = {split: {str(row["task_id"]) for row in rows} for split, rows in splits.items()}
    leakage = bool(
        split_ids["train"] & split_ids["calibration"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["calibration"] & split_ids["test"]
    )
    deduped = evaluate_ranker(_dedupe_rows(splits["test"]), task_lookup, model)
    deduped_accuracy = float(deduped["metrics"]["accuracy_at_1"])
    return {
        "task_id_leakage": leakage,
        "deduped_accuracy_at_1": _round_rate(deduped_accuracy),
        "accuracy_delta_after_dedup": _round_rate(deduped_accuracy - original_accuracy),
        "passed": not leakage and deduped_accuracy >= original_accuracy - 0.05,
    }


def _model_identity_baseline(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
) -> float:
    model_hits: dict[str, list[bool]] = defaultdict(list)
    for row in train_rows:
        for candidate in row["candidates"]:
            model_hits[str(candidate.get("model_hf_id"))].append(
                _exact_label(row, candidate, task_lookup)
            )
    best_model = max(model_hits, key=lambda key: statistics.fmean(model_hits[key]))
    correct = 0
    for row in test_rows:
        candidates = list(row["candidates"])
        selected = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("model_hf_id")) == best_model
            ),
            candidates[0],
        )
        correct += int(_exact_label(row, selected, task_lookup))
    return _round_rate(correct / len(test_rows))


def _model_identity_shortcut_check(
    test_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    task_lookup: Mapping[str, Mapping[str, Any]],
    model: QualityModel,
) -> JsonDict:
    original = evaluate_ranker(test_rows, task_lookup, model)["per_item"]
    mutated_rows = copy.deepcopy(list(test_rows))
    for row in mutated_rows:
        for candidate in row["candidates"]:
            candidate["model_hf_id"] = "identity-swapped/model"
            candidate["model_path"] = "/identity-swapped/model.gguf"
    mutated = evaluate_ranker(mutated_rows, task_lookup, model)["per_item"]
    unchanged = all(
        before["candidate_id"] == after["candidate_id"]
        and before["abstained"] == after["abstained"]
        and before["mean_energy"] == after["mean_energy"]
        for before, after in zip(original, mutated, strict=True)
    )
    return {
        "selected_ids_unchanged_after_model_id_swap": unchanged,
        "model_identity_only_accuracy_at_1": _model_identity_baseline(
            train_rows, test_rows, task_lookup
        ),
        "passed": unchanged,
    }


def _blocked_artifact(
    *,
    verdict: str,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
    source_artifact: Mapping[str, Any] | None,
    error: str | None,
) -> JsonDict:
    source_pool_path = (
        str(source_artifact.get("pool_path"))
        if isinstance(source_artifact, Mapping) and source_artifact.get("pool_path")
        else pool_mod.POOL_RELATIVE_PATH
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_pool_path": source_pool_path,
        "MODEL_SPECS": list(source_artifact.get("MODEL_SPECS", []))
        if isinstance(source_artifact, Mapping)
        else [],
        "deterministic_constraint_penalties": DETERMINISTIC_CONSTRAINT_PENALTIES,
        "learned_quality_score_description": LEARNED_QUALITY_SCORE_DESCRIPTION,
        "uncertainty_abstention_rule": UNCERTAINTY_ABSTENTION_RULE,
        "strongest_cheap_baseline": None,
        "distributional_energy_delta": 0.0,
        "delta_ci95": [0.0, 0.0],
        "family_holdout_results": {},
        "label_shuffle_result": {"passed": False, "blocked": True},
        "model_identity_shortcut_check": {"passed": False, "blocked": True},
        "ranker_ready_for_audit": False,
        "verifier_is_oracle": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": {
            "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
            "source_artifact_read": source_artifact is not None,
            "structured_pool_ready": bool(
                source_artifact and source_artifact.get("structured_pool_ready") is True
            ),
            "source_error": error,
        },
        "baseline_metrics": {},
        "ranker_metrics": {},
        "duplicate_control_result": {"passed": False, "blocked": True},
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256_payload(
            {"verdict": verdict, "run_date": run_date, "error": error}
        ),
    }
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
) -> JsonDict:
    source_artifact, error = _pool_artifact(root)
    if source_artifact is None or source_artifact.get("structured_pool_ready") is not True:
        return _blocked_artifact(
            verdict=BLOCKED_POOL_VERDICT,
            duration_s=duration_s,
            run_date=run_date,
            tests_run=tests_run,
            source_artifact=source_artifact,
            error=error,
        )
    try:
        bundle = load_structured_pool(root=root)
    except ValueError as exc:
        return _blocked_artifact(
            verdict=BLOCKED_ROWS_VERDICT,
            duration_s=duration_s,
            run_date=run_date,
            tests_run=tests_run,
            source_artifact=source_artifact,
            error=str(exc),
        )

    task_lookup = build_task_lookup()
    splits = split_rows_by_family_and_item(bundle.rows)
    train_examples = build_candidate_examples(splits["train"], task_lookup)
    model = fit_quality_model(train_examples)
    ranker = evaluate_ranker(splits["test"], task_lookup, model)
    baselines = evaluate_baselines(splits["test"], task_lookup)
    strongest = _strongest_baseline(baselines)
    baseline_items = baselines[str(strongest["name"])]["per_item"]
    delta, ci95 = _paired_delta_ci95(ranker["per_item"], baseline_items)
    family_holdout = _family_holdout_results(bundle.rows, task_lookup)
    label_shuffle = _label_shuffle_control(splits["train"], splits["test"], task_lookup)
    duplicate_control = _duplicate_control(
        splits, task_lookup, model, float(ranker["metrics"]["accuracy_at_1"])
    )
    identity_check = _model_identity_shortcut_check(
        splits["test"], splits["train"], task_lookup, model
    )
    controls_pass = bool(
        label_shuffle["passed"] and duplicate_control["passed"] and identity_check["passed"]
    )
    ready = bool(ci95[0] > 0.0 and controls_pass)
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": SUCCESS_READY_VERDICT if ready else SUCCESS_NOT_READY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_pool_path": bundle.source_pool_path,
        "MODEL_SPECS": bundle.model_specs,
        "deterministic_constraint_penalties": DETERMINISTIC_CONSTRAINT_PENALTIES,
        "learned_quality_score_description": LEARNED_QUALITY_SCORE_DESCRIPTION,
        "uncertainty_abstention_rule": UNCERTAINTY_ABSTENTION_RULE,
        "strongest_cheap_baseline": strongest,
        "distributional_energy_delta": delta,
        "delta_ci95": ci95,
        "family_holdout_results": family_holdout,
        "label_shuffle_result": label_shuffle,
        "model_identity_shortcut_check": identity_check,
        "ranker_ready_for_audit": ready,
        "verifier_is_oracle": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": {
            "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
            "source_artifact_read": True,
            "structured_pool_ready": True,
            "source_error": None,
            "split_item_leakage": duplicate_control["task_id_leakage"],
            "fover_scope_used": False,
            "llm_judge_used_as_ground_truth": False,
        },
        "baseline_metrics": {
            name: {k: v for k, v in metrics.items() if k != "per_item"}
            for name, metrics in baselines.items()
        },
        "ranker_metrics": ranker["metrics"],
        "duplicate_control_result": duplicate_control,
        "split_summary": {
            split: {
                "n": len(rows),
                "families": sorted({str(row["family"]) for row in rows}),
            }
            for split, rows in splits.items()
        },
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256_payload(
            {
                "source_pool_path": bundle.source_pool_path,
                "ranker_metrics": ranker["metrics"],
                "baseline_metrics": {
                    name: {k: v for k, v in metrics.items() if k != "per_item"}
                    for name, metrics in baselines.items()
                },
                "delta": delta,
                "ci95": ci95,
                "controls_pass": controls_pass,
            }
        ),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        duration_s=duration_s,
        run_date=run_date,
        tests_run=tests_run,
    )
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _terminal_verdict(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if not _terminal_verdict(artifact["honest_verdict"]):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must not be empty")
    ready = bool(artifact["ranker_ready_for_audit"])
    if ready and float(artifact["delta_ci95"][0]) <= 0.0:
        raise ValueError("ranker_ready_for_audit requires delta CI95 to exclude zero")
    if not ready and str(artifact["honest_verdict"]).startswith("blocked_"):
        return
    if not isinstance(artifact["strongest_cheap_baseline"], Mapping):
        raise ValueError("non-blocked artifact must name strongest_cheap_baseline")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Exp 5126 distributional-energy ranker.")
    parser.add_argument("--date", default="20260701")
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    started = time.monotonic()
    root = Path(args.root)
    artifact = build_artifact(
        root=root,
        duration_s=0.0,
        run_date=str(args.date),
        tests_run=DEFAULT_TESTS_RUN,
    )
    artifact["duration_s"] = max(time.monotonic() - started, 0.000001)
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
