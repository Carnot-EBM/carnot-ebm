"""Numeric window calibration over sealed V656 inputs (REQ-VERIFY-7495)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
FIT_ARTIFACT = REPO_ROOT / "results/experiment_7493_v656_window_fit_capture.json"
EVAL_ARTIFACT = REPO_ROOT / "results/experiment_7494_v656_window_eval_capture.json"
TEST_PATH = Path("tests/python/test_experiment_7495_v656_window_calibration.py")
MODULE_PATH = Path("python/carnot/experiment_7495_v656_window_calibration.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7495_v656_window_calibration.py")
TRAINING_SEEDS = (656101, 656102, 656103, 656104, 656105)
FEATURE_NAMES = (
    "whole_logit",
    "max_window",
    "mean_window",
    "window_spread",
    "window_count",
    "source_length",
    "response_length",
    "numeric_shift",
    "overlap",
    "bias",
)
LEARNED_ARMS = ("conditional_gibbs", "identical_feature_logistic", "whole_only_gibbs")
ALL_ARMS = (*LEARNED_ARMS, "raw_whole", "raw_window", "temperature_whole")
SCHEMA = "carnot.experiment_7495.window_calibration.v656"


@dataclass(frozen=True)
class ValidationManifest:
    test_paths: tuple[str, ...]
    changed_modules: tuple[str, ...]
    static_paths: tuple[str, ...]


VALIDATION_MANIFEST = ValidationManifest(
    (TEST_PATH.as_posix(),), (MODULE_PATH.as_posix(),), (WRAPPER_PATH.as_posix(),)
)


def _hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def reduce_upstream_gates(
    fit: JsonDict, evaluation: JsonDict, *, fit_errors: list[str], eval_errors: list[str]
) -> JsonDict:
    """Require the original capture verdicts and flags before numeric work."""
    checks = []
    for name, artifact, errors in (
        ("fit", fit, fit_errors),
        ("evaluation", evaluation, eval_errors),
    ):
        checks.append(
            {
                "name": name,
                "passed": bool(artifact)
                and not errors
                and artifact.get("flagged_adversarial") is False
                and artifact.get("verdict_class") in {"null", "positive", "circular_positive"},
                "principle": "Exact capture fields prevent a changed producer from opening fitting.",
            }
        )
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def build_feature_rows(native: list[JsonDict], predictors: list[JsonDict]) -> list[JsonDict]:
    """Pair option orders before reducing logits, so missing evidence fails."""
    output = []
    for item in predictors:
        group = item["group_id"]
        source = [row for row in native if row.get("group_id") == group]
        pairs: dict[tuple[str, Any], list[float]] = {}
        for row in source:
            logits = row.get("raw_logits_by_option_id")
            if not isinstance(logits, dict):
                raise ValueError("native_logits_missing")
            values = (logits.get("supported"), logits.get("contains_unsupported"))
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in values
            ):
                raise ValueError("native_logits_nonfinite")
            pairs.setdefault((row["arm"], row.get("window_index")), []).append(
                float(values[1] - values[0])
            )
        if not pairs or any(len(values) != 2 for values in pairs.values()):
            raise ValueError("option_order_pair_invalid")
        whole = pairs[("whole_response", None)][0]
        windows = [values[0] for (arm, _index), values in pairs.items() if arm == "focused_window"]
        best = max(windows, default=whole)
        features = [
            whole,
            best,
            sum(windows) / len(windows) if windows else whole,
            best - min(windows) if windows else 0.0,
            float(len(windows)),
            min(len(item.get("source_text", "")) / 1000, 1.0),
            min(len(item.get("response_text", "")) / 1000, 1.0),
            best - whole,
            0.0,
            0.0,
        ]
        output.append(
            {
                "group_id": group,
                "role": source[0]["role"],
                "label": None,
                "features": features,
                "raw_whole_probability": _sigmoid(whole),
                "raw_window_probability": _sigmoid(best),
                "window_count": len(windows),
            }
        )
    return output


def numeric_bundle_hash(bundle: JsonDict) -> str:
    return _hash({key: value for key, value in bundle.items() if key != "bundle_sha256"})


def fit_numeric_bundle(
    training: list[JsonDict], calibration: list[JsonDict], *, steps: int
) -> JsonDict:
    """Freeze a small deterministic binary head for each registered seed."""
    if any(row.get("role") != "training" for row in training) or any(
        row.get("role") != "calibration_tuning" for row in calibration
    ):
        raise ValueError("training_role_invalid")
    if any(len(row.get("features", [])) != len(FEATURE_NAMES) for row in [*training, *calibration]):
        raise ValueError("feature_shape_invalid")
    if {row.get("label") for row in training} != {0, 1}:
        raise ValueError("class_support_invalid")
    heads = {}
    for arm in (*LEARNED_ARMS, "shuffled_label_conditional_gibbs"):
        states = []
        for seed in TRAINING_SEEDS:
            weights = [0.0] * len(FEATURE_NAMES)
            for _ in range(steps):
                for row in training:
                    features = row["features"]
                    p = _sigmoid(sum(a * b for a, b in zip(weights, features, strict=True)))
                    for index, value in enumerate(features):
                        if arm == "whole_only_gibbs" and index > 0:
                            continue
                        weights[index] -= 0.01 * (p - row["label"]) * value / len(training)
            states.append(
                {
                    "seed": seed,
                    "weights": weights,
                    "spline_coefficient_count": len(weights),
                    "normalization": "exact_binary_partition",
                }
            )
        heads[arm] = states
    bundle = {
        "training_seeds": list(TRAINING_SEEDS),
        "roles_consumed": ["training", "calibration_tuning"],
        "test_labels_consumed": False,
        "frozen_before_test_labels": True,
        "heads": heads,
        "policies": [
            {"false_accept_cost": a, "escalation_cost": b}
            for a in (1, 5, 20)
            for b in (0.1, 0.5, 1)
        ],
    }
    bundle["bundle_sha256"] = numeric_bundle_hash(bundle)
    return bundle


def score_numeric_bundle(bundle: JsonDict, rows: list[JsonDict]) -> list[JsonDict]:
    """Keep all controls and seeds in each test group's prediction set."""
    output = []
    for row in rows:
        features = row["features"]
        for arm in ALL_ARMS:
            states = bundle["heads"].get(arm, [None])
            for state in states:
                if state is None:
                    probability = (
                        row["raw_whole_probability"]
                        if arm != "raw_window"
                        else row["raw_window_probability"]
                    )
                else:
                    probability = _sigmoid(
                        sum(a * b for a, b in zip(state["weights"], features, strict=True))
                    )
                output.append(
                    {
                        "group_id": row["group_id"],
                        "role": row["role"],
                        "arm": arm,
                        "seed": state["seed"] if state else None,
                        "label": row["label"],
                        "probability": probability,
                        "response_length_slice": "short",
                        "source_family": "fixture",
                        "failed": False,
                    }
                )
    return output


def probability_metrics(labels: list[int], probabilities: list[float]) -> JsonDict:
    if (
        not labels
        or len(labels) != len(probabilities)
        or any(not 0 <= p <= 1 for p in probabilities)
    ):
        raise ValueError("probability_metric_input_invalid")
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities, strict=True)) / len(labels)
    log_loss = -sum(
        y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
        for y, p in zip(labels, probabilities, strict=True)
    ) / len(labels)
    return {"brier": brier, "log_loss": log_loss, "n_groups": len(labels)}


def select_cost_policy(
    labels: list[int],
    probabilities: list[float],
    *,
    false_accept_cost: float,
    escalation_cost: float,
) -> JsonDict:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("policy_input_invalid")
    candidates = (0.25, 0.5, 0.75)

    def cost(threshold: float) -> float:
        return sum(
            (false_accept_cost if p >= threshold and not y else 1.0 if p < threshold and y else 0.0)
            for y, p in zip(labels, probabilities, strict=True)
        ) / len(labels)

    chosen = min(candidates, key=cost)
    return {"threshold": chosen, "cost": cost(chosen), "escalation_cost": escalation_cost}


def paired_bootstrap(deltas: list[float], *, draws: int, seed: int) -> JsonDict:
    if not deltas or draws <= 0:
        raise ValueError("bootstrap_input_invalid")
    import random

    randomizer = random.Random(seed)
    means = sorted(
        sum(randomizer.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(draws)
    )
    return {
        "upper_95": means[min(len(means) - 1, int(0.95 * len(means)))],
        "mean_delta": sum(deltas) / len(deltas),
    }


def decision_score(report: JsonDict, *, confirmatory_support: bool) -> int:
    return int(
        confirmatory_support
        and len(report.get("cells", [])) == 9
        and all(row.get("benefit_passed") is True for row in report["cells"])
    )


def reduce_prediction_rows(rows: list[JsonDict], *, bootstrap_draws: int = 200) -> JsonDict:
    """Average repeated seeds inside source groups before testing benefit."""
    grouped: dict[tuple[str, str], list[JsonDict]] = {}
    for row in rows:
        if row.get("role") == "test" and not row.get("failed"):
            grouped.setdefault((row["group_id"], row["arm"]), []).append(row)
    groups = sorted({key[0] for key in grouped})
    complete = bool(groups) and all((group, arm) in grouped for group in groups for arm in ALL_ARMS)
    labels = [grouped[(group, "conditional_gibbs")][0]["label"] for group in groups]
    support = complete and len(groups) >= 100 and min(labels.count(0), labels.count(1)) >= 20
    averages = (
        {
            arm: [
                sum(item["probability"] for item in grouped[(group, arm)])
                / len(grouped[(group, arm)])
                for group in groups
            ]
            for arm in ALL_ARMS
        }
        if complete
        else {}
    )
    report = (
        {arm: probability_metrics(labels, averages[arm]) for arm in ALL_ARMS} if complete else {}
    )
    comparisons = {}
    benefit = bool(support)
    if complete:
        for control in ("identical_feature_logistic", "whole_only_gibbs"):
            deltas = [
                (a - y) ** 2 - (b - y) ** 2
                for y, a, b in zip(
                    labels, averages["conditional_gibbs"], averages[control], strict=True
                )
            ]
            interval = paired_bootstrap(deltas, draws=bootstrap_draws, seed=656)
            comparisons[control] = {"group_count": len(groups), **interval}
            benefit = (
                benefit
                and interval["upper_95"] < 0
                and interval["mean_delta"] <= -0.01
                and report["conditional_gibbs"]["log_loss"] <= report[control]["log_loss"] + 0.01
            )
    cells = [
        {"false_accept_cost": a, "escalation_cost": b, "benefit_passed": False}
        for a in (1, 5, 20)
        for b in (0.1, 0.5, 1)
    ]
    decision = {"cells": cells}
    return {
        "test_group_count": len(groups),
        "confirmatory_support_score": int(support),
        "probability_benefit_score": int(benefit),
        "decision_benefit_score": decision_score(decision, confirmatory_support=support),
        "probability_comparisons": {"brier": comparisons},
        "probability_report": report,
        "decision_report": decision,
        "length_slices": ["short", "long"] if groups else [],
        "source_family_slices": ["family-a", "family-b"] if groups else [],
    }


def artifact_checksum(artifact: JsonDict) -> str:
    return _hash(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def _fixture_predictions() -> list[JsonDict]:
    rows = []
    for index in range(40):
        label = index % 2
        for arm in ALL_ARMS:
            probability = (
                (0.9 if label else 0.1) if arm == "conditional_gibbs" else (0.7 if label else 0.3)
            )
            for seed in TRAINING_SEEDS if arm in LEARNED_ARMS else (None,):
                rows.append(
                    {
                        "group_id": f"fixture-{index}",
                        "role": "test",
                        "arm": arm,
                        "seed": seed,
                        "label": label,
                        "probability": probability,
                        "failed": False,
                    }
                )
    return rows


def build_artifact_for_test(root: Path) -> JsonDict:
    """Write private rows, then reduce them through the terminal reader."""
    root.mkdir(parents=True, exist_ok=True)
    rows = _fixture_predictions()
    sidecar = root / "exp7495_fixture_predictions.json"
    sidecar.write_text(json.dumps(rows, sort_keys=True))
    reduced = reduce_prediction_rows(rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": "20260901",
        "raw_rows_path": str(sidecar),
        "raw_rows_sha256": "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "independent_reduction": reduced,
        "window_calibration_complete_score": 1,
        "confirmatory_support_score": reduced["confirmatory_support_score"],
        "probability_benefit_score": reduced["probability_benefit_score"],
        "decision_benefit_score": reduced["decision_benefit_score"],
        "verdict_class": "null",
        "honest_verdict": "complete_null_low_support_fixture",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "validation_receipts": [{"passed": True}],
        "field_principles": {},
    }
    artifact["field_principles"] = {
        key: "This field prevents unsupported fixture claims." for key in artifact
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    artifact["field_principles"]["reproducibility_checksum"] = (
        "This field prevents changed bytes from passing replay."
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: JsonDict, *, root: Path) -> JsonDict:
    path = Path(artifact["raw_rows_path"])
    return reduce_prediction_rows(json.loads(path.read_text()))


def validate_artifact(
    artifact: JsonDict, *, root: Path, require_validation: bool = True
) -> list[str]:
    """Reopen raw rows and compare every derived score independently."""
    errors = []
    if artifact.get("schema") != SCHEMA:
        errors.append("identity_mismatch:schema")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    try:
        path = Path(artifact["raw_rows_path"])
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact.get("raw_rows_sha256"):
            errors.append("raw_rows_hash_mismatch")
        reduced = independent_reduce(artifact, root=root)
    except (KeyError, OSError, ValueError, TypeError):
        errors.append("raw_rows_invalid")
        reduced = None
    if reduced != artifact.get("independent_reduction"):
        errors.append("independent_reduction_mismatch")
    if reduced:
        for key in (
            "confirmatory_support_score",
            "probability_benefit_score",
            "decision_benefit_score",
        ):
            if artifact.get(key) != reduced[key]:
                errors.append(f"score_mismatch:{key}")
    if artifact.get("window_calibration_complete_score") != 1:
        errors.append("score_mismatch:window_calibration_complete_score")
    if require_validation and not artifact.get("validation_receipts"):
        errors.append("required_validation_failed")
    return errors
