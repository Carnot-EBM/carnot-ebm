"""Exp 1469 HALT and Spilled-Energy telemetry diagnostic.

The Exp 1468 manifest is a rare useful case where local SOTA GGUF inference
already paid the GPU cost and persisted token logprobs plus top-k alternatives.
This module keeps Exp 1469 deliberately CPU-only: it reuses those rows, derives
small time-series features, and records whether any logprob feature separates
the bounded verifier labels after checking length and formatting confounds.

Spec: REQ-VERIFY-1469, SCENARIO-VERIFY-1469
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_RUN_DATE = "20260507"
DEFAULT_SOURCE_ARTIFACT_PATH = Path(
    "results/experiment_1468_live_sota_logprob_telemetry_preflight.json"
)
DEFAULT_MANIFEST_PATH = Path("results/live_sota_telemetry_manifest_1468.jsonl")
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1469_halt_spilled_energy_telemetry_diagnostic.json"
)
DEFAULT_DIAGNOSTIC_PATH = Path("results/live_sota_halt_spilled_diagnostics_1469.json")
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "telemetry_rows_loaded",
    "halt_features_computed",
    "spilled_energy_features_computed",
    "telemetry_diagnostic_complete",
    "auroc_or_rank_signal",
    "best_signal_name",
    "length_or_format_confound_checked",
    "diagnostic_path",
    "diagnostic_lineage_preserved",
    "diagnostic_lineage_retired",
    "honest_verdict",
)
HALT_FEATURES: tuple[str, ...] = (
    "token_logprob_mean",
    "token_logprob_min",
    "token_logprob_trend",
    "token_nll_mean",
    "token_nll_trend",
    "topk_entropy_mean",
    "topk_entropy_trend",
    "topk_gap_mean",
    "topk_gap_trend",
)
SPILLED_FEATURES: tuple[str, ...] = (
    "spilled_energy_proxy_mean",
    "spilled_energy_proxy_trend",
    "marginal_energy_proxy_mean",
    "marginal_energy_proxy_trend",
    "topk_mass_mean",
)
CONFOUND_FEATURES: tuple[str, ...] = (
    "completion_tokens",
    "response_char_length",
    "json_like_response",
    "exact_answer_format",
)
SIGNAL_FLOOR = 0.65
CONFOUND_MARGIN = 0.10
SMALL_SAMPLE_N = 30

JsonDict = dict[str, Any]
WriteJsonFn = Callable[[Path, JsonDict], None]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:  # pragma: no cover - only hit for caller-supplied outside paths.
        return str(path)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_series(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _finite_float(value)) is not None]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / len(values)))


def _linear_slope(values: Sequence[float]) -> float:
    """Return the least-squares slope against token position.

    The HALT paper treats logprob-derived values as a time series.  A centered
    one-line regression is enough here because Exp 1469 is a diagnostic, not a
    trained detector.

    Spec: REQ-VERIFY-1469
    """
    if len(values) < 2:
        return 0.0
    x_center = (len(values) - 1) / 2.0
    y_center = _mean(values)
    numerator = sum((idx - x_center) * (value - y_center) for idx, value in enumerate(values))
    denominator = sum((idx - x_center) ** 2 for idx in range(len(values)))
    return float(numerator / denominator) if denominator else 0.0


def _logsumexp(values: Sequence[float]) -> float:
    if not values:  # pragma: no cover - callers filter empty top-k positions first.
        return float("-inf")
    maximum = max(values)
    return float(maximum + math.log(sum(math.exp(value - maximum) for value in values)))


def _topk_position_stats(top_logprobs: dict[str, Any], chosen_logprob: float | None) -> JsonDict:
    logprobs = sorted(_finite_series(top_logprobs.values()), reverse=True)
    if not logprobs:
        return {
            "entropy": 0.0,
            "gap": 0.0,
            "spilled_proxy": 0.0,
            "marginal_energy_proxy": 0.0,
            "topk_mass": 0.0,
        }

    local_logsum = _logsumexp(logprobs)
    renormalized_probs = [math.exp(value - local_logsum) for value in logprobs]
    entropy = -sum(prob * math.log(prob) for prob in renormalized_probs if prob > 0.0)
    gap = logprobs[0] - logprobs[1] if len(logprobs) > 1 else 0.0
    selected_logprob = chosen_logprob if chosen_logprob is not None else logprobs[0]
    topk_mass = min(max(sum(math.exp(value) for value in logprobs), 0.0), 1.0)
    marginal_energy = -math.log(max(topk_mass, 1e-12))
    spilled_proxy = max(0.0, entropy + selected_logprob)
    return {
        "entropy": float(entropy),
        "gap": float(gap),
        "spilled_proxy": float(spilled_proxy),
        "marginal_energy_proxy": float(marginal_energy),
        "topk_mass": float(topk_mass),
    }


def _topk_series(row: JsonDict, token_logprobs: Sequence[float]) -> JsonDict:
    top_logprobs = row.get("top_logprobs") or []
    stats: list[JsonDict] = []
    for idx, alternatives in enumerate(top_logprobs):
        if not isinstance(alternatives, dict):
            continue
        chosen = token_logprobs[idx] if idx < len(token_logprobs) else None
        stats.append(_topk_position_stats(alternatives, chosen))
    return {
        "entropy": [entry["entropy"] for entry in stats],
        "gap": [entry["gap"] for entry in stats],
        "spilled_proxy": [entry["spilled_proxy"] for entry in stats],
        "marginal_energy_proxy": [entry["marginal_energy_proxy"] for entry in stats],
        "topk_mass": [entry["topk_mass"] for entry in stats],
    }


def extract_final_answer(response_text: str) -> str | None:
    """Extract a terminal integer answer from a bounded Exp 1468 response.

    This is intentionally conservative: answer markers win, then post-``think``
    text is scanned from the end.  Internal arithmetic in long reasoning is not
    treated as an answer unless it appears at the end of a line or sentence.

    Spec: REQ-VERIFY-1469
    """
    text = str(response_text or "")
    marker_match = re.search(r"(?:answer|final)\D*(-?\d+)\b", text, flags=re.IGNORECASE)
    if marker_match:
        return marker_match.group(1)

    answer_region = text.split("</think>")[-1] if "</think>" in text else text
    lines = [line.strip() for line in answer_region.splitlines() if line.strip()]
    for line in reversed(lines):
        exact = re.fullmatch(r"[-+]?\d+", line)
        if exact:
            return exact.group(0).lstrip("+")
        terminal = re.search(r"(?:^|\s)([-+]?\d+)\s*$", line)
        if terminal:
            return terminal.group(1).lstrip("+")
    return None


def _known_verifier_label(row: JsonDict) -> int | None:
    expected = str(row.get("expected_answer", "")).strip()
    family = str(row.get("family", ""))
    prompt = str(row.get("prompt", ""))
    is_binary_verifier_case = family == "fover_style" or "Return 1" in prompt
    if expected in {"0", "1"} and is_binary_verifier_case:
        return 1 if expected == "0" else 0
    return None


def _response_correct_label(row: JsonDict) -> int | None:
    expected = str(row.get("expected_answer", "")).strip()
    if not expected:
        return None
    extracted = extract_final_answer(str(row.get("response_text", "")))
    if extracted is None:
        return None
    return 1 if extracted == expected else 0


def extract_telemetry_features(row: JsonDict) -> JsonDict:
    """Compute deterministic HALT and Spilled-Energy features for one row.

    The Exp 1468 telemetry gives chosen-token logprobs plus top-k alternatives,
    not full logits.  The spill and marginal-energy fields are therefore labeled
    as proxies: they are derived from the observed top-k mass and never promoted
    as full-logit Spilled Energy.

    Spec: REQ-VERIFY-1469
    """
    token_logprobs = _finite_series(row.get("token_logprobs") or [])
    token_nll = [-value for value in token_logprobs]
    topk = _topk_series(row, token_logprobs)
    response_text = str(row.get("response_text", ""))
    expected = str(row.get("expected_answer", "")).strip()
    stripped = response_text.strip()
    exact_answer_format = 1.0 if expected and stripped == expected else 0.0
    json_like = 1.0 if stripped.startswith("{") or stripped.startswith("[") else 0.0

    return {
        "completion_tokens": float(row.get("completion_tokens") or len(token_logprobs)),
        "response_char_length": float(len(response_text)),
        "json_like_response": json_like,
        "exact_answer_format": exact_answer_format,
        "token_logprob_mean": _mean(token_logprobs),
        "token_logprob_min": min(token_logprobs) if token_logprobs else 0.0,
        "token_logprob_stdev": _stdev(token_logprobs),
        "token_logprob_trend": _linear_slope(token_logprobs),
        "token_nll_mean": _mean(token_nll),
        "token_nll_trend": _linear_slope(token_nll),
        "topk_entropy_mean": _mean(topk["entropy"]),
        "topk_entropy_trend": _linear_slope(topk["entropy"]),
        "topk_gap_mean": _mean(topk["gap"]),
        "topk_gap_trend": _linear_slope(topk["gap"]),
        "spilled_energy_proxy_mean": _mean(topk["spilled_proxy"]),
        "spilled_energy_proxy_trend": _linear_slope(topk["spilled_proxy"]),
        "marginal_energy_proxy_mean": _mean(topk["marginal_energy_proxy"]),
        "marginal_energy_proxy_trend": _linear_slope(topk["marginal_energy_proxy"]),
        "topk_mass_mean": _mean(topk["topk_mass"]),
    }


def build_case_diagnostic(row: JsonDict) -> JsonDict:
    """Return one JSON-ready per-case diagnostic row.

    Spec: REQ-VERIFY-1469
    """
    known_label = _known_verifier_label(row)
    response_correct = _response_correct_label(row)
    response_failure = None if response_correct is None else 1 - response_correct
    label_source = (
        "known_binary_verifier_label"
        if known_label is not None
        else "response_correctness_label"
        if response_correct is not None
        else "unlabeled"
    )
    return {
        "case_id": row.get("case_id"),
        "family": row.get("family"),
        "expected_answer": row.get("expected_answer"),
        "known_verifier_label": known_label,
        "response_correct_label": response_correct,
        "response_failure_label": response_failure,
        "label_source": label_source,
        "features": extract_telemetry_features(row),
    }


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC by pairwise rank comparison.

    A pure-Python Mann-Whitney calculation keeps this diagnostic independent of
    sklearn.  Ties receive half credit, matching standard AUROC behavior.

    Spec: REQ-VERIFY-1469
    """
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def _feature_signal(
    diagnostics: Sequence[JsonDict],
    feature_name: str,
    label_key: str,
) -> JsonDict | None:
    pairs: list[tuple[int, float]] = []
    for row in diagnostics:
        label = row.get(label_key)
        score = _finite_float((row.get("features") or {}).get(feature_name))
        if label in {0, 1} and score is not None:
            pairs.append((int(label), score))
    if not pairs:
        return None
    labels = [label for label, _score in pairs]
    scores = [score for _label, score in pairs]
    auroc = binary_auroc(labels, scores)
    if auroc is None:
        return None
    oriented = max(auroc, 1.0 - auroc)
    return {
        "name": feature_name,
        "label_key": label_key,
        "auroc": float(auroc),
        "oriented_auroc": float(oriented),
        "direction": "higher" if auroc >= 0.5 else "lower",
        "n": len(pairs),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
    }


def evaluate_rank_signals(
    diagnostics: Sequence[JsonDict],
    *,
    candidate_features: Sequence[str] = HALT_FEATURES + SPILLED_FEATURES,
    label_key: str = "known_verifier_label",
) -> JsonDict:
    """Compare feature rank signals against binary labels.

    Spec: REQ-VERIFY-1469, SCENARIO-VERIFY-1469
    """
    signals = [
        signal
        for feature in candidate_features
        if (signal := _feature_signal(diagnostics, feature, label_key)) is not None
    ]
    signals.sort(key=lambda item: (-item["oriented_auroc"], item["name"]))
    best = signals[0] if signals else None
    n = int(best["n"]) if best else 0
    return {
        "label_key": label_key,
        "signals": signals,
        "best_signal": best,
        "best_signal_name": best["name"] if best else None,
        "small_sample_caveat": n < SMALL_SAMPLE_N,
        "bootstrap_caveat": "n<30; AUROC is descriptive rank evidence, not a stable estimator"
        if n < SMALL_SAMPLE_N
        else None,
    }


def _choose_label_key(diagnostics: Sequence[JsonDict]) -> str:
    known = [row["known_verifier_label"] for row in diagnostics if row["known_verifier_label"] in {0, 1}]
    if len(set(known)) == 2:
        return "known_verifier_label"
    response = [
        row["response_failure_label"]
        for row in diagnostics
        if row["response_failure_label"] in {0, 1}
    ]
    if len(set(response)) == 2:
        return "response_failure_label"
    return "known_verifier_label"


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_source_artifact(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def build_diagnostic_payload(rows: Sequence[JsonDict], *, run_date: str) -> JsonDict:
    """Build the per-case diagnostic JSON payload.

    Spec: REQ-VERIFY-1469
    """
    cases = [build_case_diagnostic(row) for row in rows]
    label_key = _choose_label_key(cases)
    rank_summary = evaluate_rank_signals(cases, label_key=label_key)
    confound_summary = evaluate_rank_signals(
        cases,
        candidate_features=CONFOUND_FEATURES,
        label_key=label_key,
    )
    return {
        "schema_version": 1,
        "run_date": run_date,
        "case_count": len(cases),
        "label_key": label_key,
        "cases": cases,
        "rank_signal": rank_summary,
        "confound_signal": confound_summary,
    }


def _lineage_decision(rank_signal: JsonDict, confound_signal: JsonDict) -> JsonDict:
    best = rank_signal.get("best_signal") or {}
    confound_best = confound_signal.get("best_signal") or {}
    best_score = float(best.get("oriented_auroc") or 0.0)
    confound_score = float(confound_best.get("oriented_auroc") or 0.0)
    preserved = best_score >= SIGNAL_FLOOR and best_score > confound_score + CONFOUND_MARGIN
    return {
        "diagnostic_lineage_preserved": preserved,
        "diagnostic_lineage_retired": not preserved,
        "honest_verdict": "preserved_nontrivial_logprob_signal_small_n"
        if preserved
        else "retired_non_headline_telemetry_flat_or_confounded",
        "best_oriented_auroc": best_score,
        "best_confound_oriented_auroc": confound_score,
    }


def build_artifact(
    *,
    project_root: Path,
    run_date: str,
    source_artifact_path: Path,
    manifest_path: Path,
    diagnostic_path: Path,
    write_json_fn: WriteJsonFn = _write_json,
) -> JsonDict:
    """Build and write the final Exp 1469 artifact.

    Spec: REQ-VERIFY-1469, SCENARIO-VERIFY-1469
    """
    source = _load_source_artifact(source_artifact_path)
    rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
    model_specs = source.get("model_specs") or list(MANDATED_MODEL_IDS)
    topk_ready = bool(source.get("topk_logprobs_available", True))
    diagnostic = build_diagnostic_payload(rows, run_date=run_date)
    write_json_fn(diagnostic_path, diagnostic)

    rank_signal = diagnostic["rank_signal"]
    confound_signal = diagnostic["confound_signal"]
    lineage = _lineage_decision(rank_signal, confound_signal)
    telemetry_ready = bool(rows) and topk_ready
    artifact = {
        "schema_version": 1,
        "run_date": run_date,
        "status": "complete",
        "model_specs": model_specs,
        "source_artifact_path": _display_path(project_root, source_artifact_path),
        "source_manifest_path": _display_path(project_root, manifest_path),
        "telemetry_rows_loaded": len(rows),
        "halt_features_computed": telemetry_ready,
        "spilled_energy_features_computed": telemetry_ready,
        "telemetry_diagnostic_complete": telemetry_ready,
        "auroc_or_rank_signal": {
            "rank_signal": rank_signal,
            "confound_signal": confound_signal,
            "small_sample_caveat": rank_signal.get("small_sample_caveat"),
            "bootstrap_caveat": rank_signal.get("bootstrap_caveat"),
            "best_oriented_auroc": lineage["best_oriented_auroc"],
            "best_confound_oriented_auroc": lineage["best_confound_oriented_auroc"],
        },
        "best_signal_name": rank_signal.get("best_signal_name"),
        "length_or_format_confound_checked": True,
        "diagnostic_path": _display_path(project_root, diagnostic_path),
        "diagnostic_lineage_preserved": lineage["diagnostic_lineage_preserved"],
        "diagnostic_lineage_retired": lineage["diagnostic_lineage_retired"],
        "honest_verdict": lineage["honest_verdict"] if telemetry_ready else "blocked_no_reusable_topk_telemetry",
    }
    if not telemetry_ready:
        artifact["diagnostic_lineage_preserved"] = False
        artifact["diagnostic_lineage_retired"] = True
    return artifact


def run_experiment(
    *,
    project_root: str | Path = Path("."),
    run_date: str = DEFAULT_RUN_DATE,
    source_artifact_path: str | Path = DEFAULT_SOURCE_ARTIFACT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    diagnostic_path: str | Path = DEFAULT_DIAGNOSTIC_PATH,
    write_json_fn: WriteJsonFn = _write_json,
) -> JsonDict:
    """Write the in-progress marker, compute diagnostics, then write final JSON.

    Spec: REQ-VERIFY-1469
    """
    root = Path(project_root)
    source_artifact = _resolve_path(root, source_artifact_path)
    manifest = _resolve_path(root, manifest_path)
    output = _resolve_path(root, output_path)
    diagnostic = _resolve_path(root, diagnostic_path)

    write_json_fn(
        output,
        {
            "status": "in_progress",
            "model_specs": list(MANDATED_MODEL_IDS),
            "telemetry_rows_loaded": 0,
            "halt_features_computed": False,
            "spilled_energy_features_computed": False,
            "telemetry_diagnostic_complete": False,
            "auroc_or_rank_signal": None,
            "best_signal_name": None,
            "length_or_format_confound_checked": False,
            "diagnostic_path": _display_path(root, diagnostic),
            "diagnostic_lineage_preserved": False,
            "diagnostic_lineage_retired": False,
            "honest_verdict": "in_progress",
        },
    )
    artifact = build_artifact(
        project_root=root,
        run_date=run_date,
        source_artifact_path=source_artifact,
        manifest_path=manifest,
        diagnostic_path=diagnostic,
        write_json_fn=write_json_fn,
    )
    write_json_fn(output, artifact)
    return artifact


def _parse_args() -> argparse.Namespace:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--source-artifact-path", default=str(DEFAULT_SOURCE_ARTIFACT_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--diagnostic-path", default=str(DEFAULT_DIAGNOSTIC_PATH))
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper.
    args = _parse_args()
    run_experiment(
        project_root=args.project_root,
        run_date=args.run_date,
        source_artifact_path=args.source_artifact_path,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        diagnostic_path=args.diagnostic_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
