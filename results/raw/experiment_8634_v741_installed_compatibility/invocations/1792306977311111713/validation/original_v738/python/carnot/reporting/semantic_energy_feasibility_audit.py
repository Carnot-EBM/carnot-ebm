"""Exp 1481 Semantic Energy feasibility audit.

This module keeps the Semantic Energy follow-up deliberately bounded: it reads
the Exp 1480 top-k telemetry that already exists, computes small deterministic
proxies, and refuses headline telemetry claims unless those proxies beat the
surface baselines recorded on the same rows.

Spec: REQ-VERIFY-1481, SCENARIO-VERIFY-1481.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from carnot.reporting.halt_spilled_energy_telemetry_diagnostic import binary_auroc


MANDATED_MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "telemetry_rows_loaded",
    "semantic_energy_features_computed",
    "baseline_features_computed",
    "semantic_energy_audit_complete",
    "best_semantic_signal",
    "best_superficial_baseline",
    "signal_beats_superficial_baselines",
    "diagnostic_path",
    "claim_allowed",
    "diagnostic_lineage_retired",
    "honest_verdict",
)

SEMANTIC_FEATURE_FIELDS: tuple[str, ...] = (
    "answer_choice_energy_gap",
    "final_logit_entropy",
    "topk_semantic_cluster_proxy",
    "per_case_uncertainty_spread",
)

NUMERIC_BASELINE_FIELDS: tuple[str, ...] = (
    "response_length",
    "token_count",
    "json_valid",
    "schema_valid",
    "answer_lexical_overlap",
)

CATEGORICAL_BASELINE_FIELDS: tuple[str, ...] = ("prompt_family", "model_family")

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _topk_items(topk: dict[str, Any]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for token, value in topk.items():
        score = _coerce_float(value)
        if score is not None:
            items.append((str(token), score))
    return items


def _entropy_from_logprobs(items: Sequence[tuple[str, float]]) -> float | None:
    if not items:
        return None
    max_logprob = max(score for _, score in items)
    weights = [math.exp(score - max_logprob) for _, score in items]
    total = sum(weights)
    probs = [weight / total for weight in weights]
    return -sum(prob * math.log(prob) for prob in probs if prob > 0.0)


def _semantic_cluster(token: str) -> str:
    stripped = token.strip().lower()
    if not stripped:
        return "whitespace"
    if _NUMBER_RE.fullmatch(stripped):
        return "number"
    if "think" in stripped or "<" in stripped or ">" in stripped:
        return "markup"
    if stripped in {"yes", "no", "true", "false"}:
        return "truth_word"
    return "prose"


def _semantic_cluster_proxy(items: Sequence[tuple[str, float]]) -> float | None:
    if not items:
        return None
    max_logprob = max(score for _, score in items)
    cluster_mass: dict[str, float] = {}
    for token, score in items:
        cluster = _semantic_cluster(token)
        cluster_mass[cluster] = cluster_mass.get(cluster, 0.0) + math.exp(score - max_logprob)
    total = sum(cluster_mass.values())
    dominant_mass = max(cluster_mass.values()) / total
    return 1.0 - dominant_mass


def _answer_logprob(topk_rows: Sequence[dict[str, Any]], answer: Any) -> float | None:
    answer_text = str(answer).strip()
    if not answer_text:
        return None
    matches = [
        score
        for topk in topk_rows
        for token, score in _topk_items(topk)
        if token.strip() == answer_text
    ]
    return max(matches) if matches else None


def _usable_rows(
    rows: Sequence[dict[str, Any]], source_artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    if not source_artifact.get("logits_available"):
        return []
    return [row for row in rows if row.get("logits_available") and row.get("top_logprobs")]


def extract_semantic_energy_features(row: dict[str, Any]) -> dict[str, float | None]:
    """Compute bounded Semantic Energy proxies from one Exp 1480 telemetry row.

    The input is top-k logprob telemetry rather than full logits, so each proxy
    is explicitly a feasibility diagnostic.  Answer-choice gap uses the energy
    convention: higher values mean the wrong answer has lower energy than the
    expected answer, which is a failure signal.
    """

    topk_rows = [topk for topk in row.get("top_logprobs", []) if isinstance(topk, dict)]
    final_items = _topk_items(topk_rows[-1]) if topk_rows else []
    entropy_values = [
        entropy
        for topk in topk_rows
        if (entropy := _entropy_from_logprobs(_topk_items(topk))) is not None
    ]
    expected_logprob = _answer_logprob(topk_rows, row.get("expected_answer"))
    wrong_logprob = _answer_logprob(topk_rows, row.get("adversarial_wrong_answer"))
    answer_gap = (
        wrong_logprob - expected_logprob
        if expected_logprob is not None and wrong_logprob is not None
        else None
    )
    spread = max(entropy_values) - min(entropy_values) if entropy_values else None

    return {
        "final_logit_entropy": _entropy_from_logprobs(final_items),
        "topk_semantic_cluster_proxy": _semantic_cluster_proxy(final_items),
        "answer_choice_energy_gap": answer_gap,
        "per_case_uncertainty_spread": spread,
    }


def _label_from_row(row: dict[str, Any]) -> tuple[int | None, str]:
    label = row.get("known_verifier_label")
    if label in {0, 1}:
        return 1 - int(label), "known_binary_verifier_label"
    if isinstance(row.get("correct"), bool):
        return 0 if row["correct"] else 1, "response_correctness_label"
    return None, "unlabeled"


def build_case_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build the per-case diagnostic row used for REQ-VERIFY-1481 auditing."""

    semantic_features = extract_semantic_energy_features(row)
    baseline_scores = dict(row.get("superficial_baselines") or {})
    failure_label, label_source = _label_from_row(row)
    feature_row: dict[str, Any] = {
        "case_id": row.get("case_id"),
        "hf_id": row.get("hf_id"),
        "semantic_failure_label": failure_label,
        "label_source": label_source,
        "semantic_energy_features": semantic_features,
        "baseline_scores": baseline_scores,
    }
    feature_row.update(semantic_features)
    for field in NUMERIC_BASELINE_FIELDS:
        feature_row[field] = _coerce_float(baseline_scores.get(field))
    return feature_row


def _add_categorical_baseline_indicators(feature_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for field in CATEGORICAL_BASELINE_FIELDS:
        values = sorted(
            {
                str(row["baseline_scores"].get(field))
                for row in feature_rows
                if row["baseline_scores"].get(field) is not None
            }
        )
        for value in values:
            name = f"{field}={value}"
            names.append(name)
            for row in feature_rows:
                row[name] = 1.0 if str(row["baseline_scores"].get(field)) == value else 0.0
    return tuple(names)


def _best_threshold_accuracy(
    labels: Sequence[int], scores: Sequence[float], direction: str
) -> float:
    sense = 1.0 if direction == "higher" else -1.0
    oriented_scores = [sense * score for score in scores]
    best = 0.0
    for threshold in sorted(set(oriented_scores)):
        correct = sum(
            int(label == (1 if score >= threshold else 0))
            for label, score in zip(labels, oriented_scores, strict=True)
        )
        best = max(best, correct / len(labels))
    return best


def evaluate_feature_signals(
    feature_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    *,
    feature_source: str,
) -> dict[str, Any]:
    """Evaluate oriented AUROC and best threshold accuracy for feature names."""

    signals: list[dict[str, Any]] = []
    for name in feature_names:
        pairs = [
            (row.get("semantic_failure_label"), _coerce_float(row.get(name)))
            for row in feature_rows
        ]
        usable = [
            (int(label), float(score))
            for label, score in pairs
            if label in {0, 1} and score is not None
        ]
        labels = [label for label, _ in usable]
        scores = [score for _, score in usable]
        auroc = binary_auroc(labels, scores)
        if auroc is None:
            continue
        direction = "higher" if auroc >= 0.5 else "lower"
        oriented_auroc = auroc if auroc >= 0.5 else 1.0 - auroc
        signals.append(
            {
                "name": name,
                "feature_source": feature_source,
                "raw_auroc": auroc,
                "oriented_auroc": oriented_auroc,
                "direction": direction,
                "best_accuracy": _best_threshold_accuracy(labels, scores, direction),
                "n": len(usable),
            }
        )
    best_signal = max(
        signals,
        key=lambda signal: (signal["oriented_auroc"], signal["best_accuracy"]),
        default=None,
    )
    return {
        "feature_source": feature_source,
        "signals": signals,
        "best_signal": best_signal,
    }


def _diagnostic_rows(
    rows: Sequence[dict[str, Any]],
    source_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_rows = [build_case_feature_row(row) for row in _usable_rows(rows, source_artifact)]
    _add_categorical_baseline_indicators(feature_rows)
    return feature_rows


def _empty_payload(
    *,
    source_artifact: dict[str, Any],
    diagnostic_path: Path,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "model_specs": source_artifact.get("model_specs") or list(MANDATED_MODEL_SPECS),
        "telemetry_rows_loaded": 0,
        "semantic_energy_features_computed": False,
        "baseline_features_computed": False,
        "semantic_energy_audit_complete": False,
        "best_semantic_signal": None,
        "best_superficial_baseline": None,
        "signal_beats_superficial_baselines": False,
        "diagnostic_path": str(diagnostic_path),
        "claim_allowed": False,
        "diagnostic_lineage_retired": True,
        "honest_verdict": "blocked_no_reusable_exp1480_logits",
    }


def build_semantic_energy_payload(
    rows: Sequence[dict[str, Any]],
    *,
    source_artifact: dict[str, Any],
    run_date: str,
    diagnostic_path: Path,
) -> dict[str, Any]:
    """Build the terminal Exp 1481 artifact without writing files."""

    feature_rows = _diagnostic_rows(rows, source_artifact)
    if not feature_rows:
        return _empty_payload(source_artifact=source_artifact, diagnostic_path=diagnostic_path)

    categorical_names = tuple(
        name
        for row in feature_rows
        for name in row
        if name.startswith("prompt_family=") or name.startswith("model_family=")
    )
    baseline_names = NUMERIC_BASELINE_FIELDS + tuple(sorted(set(categorical_names)))
    semantic_summary = evaluate_feature_signals(
        feature_rows,
        SEMANTIC_FEATURE_FIELDS,
        feature_source="semantic_energy",
    )
    baseline_summary = evaluate_feature_signals(
        feature_rows,
        baseline_names,
        feature_source="superficial_baseline",
    )
    best_semantic = semantic_summary["best_signal"]
    best_baseline = baseline_summary["best_signal"]
    nontrivial_semantic = bool(best_semantic and best_semantic["oriented_auroc"] > 0.5)
    signal_beats = bool(
        nontrivial_semantic
        and best_baseline
        and best_semantic["oriented_auroc"] > best_baseline["oriented_auroc"]
        and best_semantic["best_accuracy"] >= best_baseline["best_accuracy"]
    )
    audit_complete = bool(best_semantic and best_baseline)
    honest_verdict = (
        "semantic_energy_signal_survives_superficial_baselines"
        if signal_beats
        else "retired_semantic_energy_confounded_by_superficial_baseline"
    )
    return {
        "status": "complete",
        "schema_version": 1,
        "run_date": run_date,
        "model_specs": source_artifact.get("model_specs") or list(MANDATED_MODEL_SPECS),
        "telemetry_rows_loaded": len(feature_rows),
        "semantic_energy_features_computed": bool(best_semantic),
        "baseline_features_computed": bool(best_baseline),
        "semantic_energy_audit_complete": audit_complete,
        "best_semantic_signal": best_semantic,
        "best_superficial_baseline": best_baseline,
        "signal_beats_superficial_baselines": signal_beats,
        "diagnostic_path": str(diagnostic_path),
        "claim_allowed": signal_beats,
        "diagnostic_lineage_retired": not signal_beats,
        "honest_verdict": honest_verdict,
    }


def _diagnostic_payload(
    rows: Sequence[dict[str, Any]],
    *,
    source_artifact: dict[str, Any],
    run_date: str,
) -> dict[str, Any]:
    feature_rows = _diagnostic_rows(rows, source_artifact)
    return {
        "schema_version": 1,
        "run_date": run_date,
        "case_features": feature_rows,
        "semantic_signal_summary": evaluate_feature_signals(
            feature_rows,
            SEMANTIC_FEATURE_FIELDS,
            feature_source="semantic_energy",
        ),
        "baseline_signal_summary": evaluate_feature_signals(
            feature_rows,
            NUMERIC_BASELINE_FIELDS,
            feature_source="superficial_baseline",
        ),
    }


def run_experiment(
    *,
    project_root: Path | str,
    run_date: str = "20260507",
    source_artifact_path: Path | None = None,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    diagnostic_path: Path | None = None,
) -> dict[str, Any]:
    """Run Exp 1481 from existing Exp 1480 artifacts and write JSON outputs."""

    root = Path(project_root)
    source_path = (
        source_artifact_path
        or root / "results" / "experiment_1480_live_sota_balanced_telemetry_v2.json"
    )
    source_manifest = (
        manifest_path or root / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"
    )
    artifact_path = (
        output_path or root / "results" / "experiment_1481_semantic_energy_feasibility_audit.json"
    )
    features_path = diagnostic_path or root / "results" / "semantic_energy_features_1481.json"

    _write_json(
        artifact_path,
        {
            "status": "in_progress",
            "model_specs": list(MANDATED_MODEL_SPECS),
            "telemetry_rows_loaded": 0,
            "semantic_energy_features_computed": False,
            "baseline_features_computed": False,
            "semantic_energy_audit_complete": False,
            "best_semantic_signal": None,
            "best_superficial_baseline": None,
            "signal_beats_superficial_baselines": False,
            "diagnostic_path": str(features_path),
            "claim_allowed": False,
            "diagnostic_lineage_retired": False,
            "honest_verdict": "in_progress",
        },
    )
    source_artifact = _read_json(source_path)
    rows = _read_jsonl(source_manifest)
    artifact = build_semantic_energy_payload(
        rows,
        source_artifact=source_artifact,
        run_date=run_date,
        diagnostic_path=features_path,
    )
    _write_json(
        features_path, _diagnostic_payload(rows, source_artifact=source_artifact, run_date=run_date)
    )
    _write_json(artifact_path, artifact)
    return artifact
