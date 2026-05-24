"""Exp 2978 repair/formalization telemetry triage panel.

This module reads existing repair and solver artifacts and asks a narrow
question: do early-prefix, first-step, confidence, or semantic-energy proxy
features separate candidates that later fail verification?  It does not run a
verifier and it does not change acceptance policy.

Spec: REQ-VERIFY-2978, SCENARIO-VERIFY-2978
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

ARTIFACT_FILENAME = "experiment_2978_first_step_semantic_energy_repair_telemetry_v1.json"
EXP2964_FILENAME = "experiment_2964_sota_dccd_repair_replication_v1.json"
EXP2967_FILENAME = "experiment_2967_sota_nl_to_z3_dccd_formalization_v1.json"
EXP2968_FILENAME = "experiment_2968_interwhen_partial_monitor_harness_v1.json"
EXP2977_FILENAME = "experiment_2977_sota_intent_preserving_code_repair_v1.json"
RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "mixed_artifact_and_optional_live_llm"
SIGNAL_USABLE_AUROC_FLOOR = 0.60
MANDATORY_HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "telemetry_panel_ready",
    "first_step_signal_usable",
    "semantic_energy_signal_usable",
    "logprob_unavailable",
    "no_headline_verifier_claim",
    "models_used",
    "mandatory_headline_model_ids",
    "candidate_rows",
    "calibration_metrics",
    "triage_examples",
    "failure_modes_explained",
    "inference_substrate",
    "duration_s",
)


@dataclass(frozen=True)
class TelemetryPanelConfig:
    """Runtime paths for the artifact-only telemetry panel."""

    repo_root: Path = Path(__file__).resolve().parents[3]
    output_path: Path | None = None
    started_at: float | None = None
    clock: ClockFn = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.repo_root / "results" / ARTIFACT_FILENAME


def build_artifact(config: TelemetryPanelConfig | None = None) -> JsonDict:
    """Build the Exp 2978 JSON payload without writing it."""
    active = config or TelemetryPanelConfig()
    started_at = active.start_time()
    source_artifacts = _load_source_artifacts(active.repo_root)
    rows = _candidate_rows(active.repo_root, source_artifacts)
    model_specs, models_used = _model_provenance(source_artifacts, rows)

    if not rows:
        return _blocked_artifact(active, started_at, source_artifacts, model_specs, models_used)

    calibration = _calibration_metrics(rows)
    first_step_metric = calibration["first_step_proxy_failure_score"]["auroc"]
    semantic_metric = calibration["semantic_energy_proxy_failure_score"]["auroc"]
    logprob_unavailable = not any(row["logprob_available"] for row in rows)
    artifact = {
        "honest_verdict": (
            "complete: telemetry panel ready for diagnostic triage only; "
            "no verifier or headline claim made"
        ),
        "telemetry_panel_ready": True,
        "first_step_signal_usable": _metric_clears_floor(first_step_metric),
        "semantic_energy_signal_usable": _metric_clears_floor(semantic_metric),
        "logprob_unavailable": logprob_unavailable,
        "no_headline_verifier_claim": True,
        "models_used": models_used,
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "candidate_rows": len(rows),
        "calibration_metrics": calibration,
        "triage_examples": _triage_examples(rows),
        "failure_modes_explained": _failure_modes(rows, source_artifacts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": active.clock() - started_at,
        "run_date": RUN_DATE,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "source_artifacts": source_artifacts,
        "model_specs": model_specs,
        "MODEL_SPECS": model_specs,
        "candidate_level_rows": rows,
        "methodology_notes": [
            "No fresh live inference was run by this panel.",
            "Direct token-logprob features remain unavailable unless an upstream artifact records them.",
            "All confidence and semantic-energy fields are artifact-derived proxies for triage.",
            "Telemetry alone is not an acceptance gate and is not a verifier claim.",
        ],
    }
    return artifact


def write_artifact(config: TelemetryPanelConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2978 terminal artifact."""
    active = config or TelemetryPanelConfig()
    payload = build_artifact(active)
    output_path = active.resolved_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return tie-aware AUROC for scores where larger means more likely failure."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        wins += sum(1.0 for negative in negatives if positive > negative)
        wins += sum(0.5 for negative in negatives if positive == negative)
    return wins / (len(positives) * len(negatives))


def _load_source_artifacts(repo_root: Path) -> JsonDict:
    sources: JsonDict = {}
    for filename in (EXP2964_FILENAME, EXP2967_FILENAME, EXP2968_FILENAME, EXP2977_FILENAME):
        rel_path = f"results/{filename}"
        path = repo_root / rel_path
        sources[rel_path] = {
            "available": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "payload": _read_json(path) if path.is_file() else {},
        }
    return sources


def _blocked_artifact(
    active: TelemetryPanelConfig,
    started_at: float,
    source_artifacts: JsonDict,
    model_specs: list[JsonDict],
    models_used: list[str],
) -> JsonDict:
    return {
        "honest_verdict": "blocked_no_candidate_artifacts_available",
        "telemetry_panel_ready": False,
        "first_step_signal_usable": False,
        "semantic_energy_signal_usable": False,
        "logprob_unavailable": True,
        "no_headline_verifier_claim": True,
        "models_used": models_used,
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "candidate_rows": 0,
        "calibration_metrics": _empty_calibration_metrics(),
        "triage_examples": [],
        "failure_modes_explained": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": active.clock() - started_at,
        "run_date": RUN_DATE,
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "source_artifacts": source_artifacts,
        "model_specs": model_specs,
        "MODEL_SPECS": model_specs,
        "candidate_level_rows": [],
    }


def _candidate_rows(repo_root: Path, sources: JsonDict) -> list[JsonDict]:
    rows: list[JsonDict] = []
    exp2964 = sources[f"results/{EXP2964_FILENAME}"]["payload"]
    rows.extend(_code_repair_rows(repo_root, exp2964, EXP2964_FILENAME, "raw_response_ref"))
    exp2967 = sources[f"results/{EXP2967_FILENAME}"]["payload"]
    rows.extend(_formalization_rows(repo_root, exp2967, EXP2967_FILENAME))
    exp2977 = sources[f"results/{EXP2977_FILENAME}"]["payload"]
    rows.extend(_code_repair_rows(repo_root, exp2977, EXP2977_FILENAME, "raw_candidate_ref"))
    return rows


def _code_repair_rows(
    repo_root: Path,
    payload: JsonDict,
    filename: str,
    raw_field: str,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, candidate in enumerate(payload.get("candidate_evaluations") or []):
        if not isinstance(candidate, dict):
            continue
        raw_ref = str(candidate.get(raw_field) or "")
        raw_text = _raw_text(repo_root, raw_ref)
        schema_valid = bool(candidate.get("schema_valid", True))
        syntax_success = bool(candidate.get("syntax_success", False))
        final_outcome = bool(candidate.get("passed", False))
        first_step = _first_step_features(raw_text, "code_repair")
        failure_category = _code_failure_category(candidate, schema_valid, syntax_success, final_outcome)
        rows.append(
            _candidate_row(
                source_filename=filename,
                candidate_kind="code_repair",
                row_index=index,
                candidate=candidate,
                final_outcome=final_outcome,
                schema_status="valid" if schema_valid else "invalid",
                syntax_status="valid" if syntax_success else "invalid",
                failure_category=failure_category,
                first_step=first_step,
                semantic_energy=_semantic_energy_proxy(
                    schema_valid=schema_valid,
                    syntax_success=syntax_success,
                    first_step_failure=first_step["first_step_proxy_failure_score"],
                    tokens_generated=_finite_float(candidate.get("tokens_generated")),
                ),
                raw_ref=raw_ref,
            )
        )
    return rows


def _formalization_rows(repo_root: Path, payload: JsonDict, filename: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, item in enumerate(payload.get("per_item_results") or []):
        if not isinstance(item, dict):
            continue
        raw_ref = str(item.get("raw_response_path") or "")
        raw_text = _raw_text(repo_root, raw_ref)
        parseable = bool(item.get("parseable", False))
        z3_executed = bool(item.get("z3_executed", False))
        final_outcome = bool(item.get("answer_correct")) and bool(item.get("solver_formula_correct"))
        first_step = _first_step_features(raw_text, "nl_to_z3_formalization")
        rows.append(
            _candidate_row(
                source_filename=filename,
                candidate_kind="nl_to_z3_formalization",
                row_index=index,
                candidate=item,
                final_outcome=final_outcome,
                schema_status="valid" if parseable else "invalid",
                syntax_status="valid" if z3_executed else "invalid",
                failure_category=str(item.get("failure_category") or "unknown"),
                first_step=first_step,
                semantic_energy=_semantic_energy_proxy(
                    schema_valid=parseable,
                    syntax_success=z3_executed,
                    first_step_failure=first_step["first_step_proxy_failure_score"],
                    tokens_generated=None,
                ),
                raw_ref=raw_ref,
            )
        )
    return rows


def _candidate_row(
    *,
    source_filename: str,
    candidate_kind: str,
    row_index: int,
    candidate: JsonDict,
    final_outcome: bool,
    schema_status: str,
    syntax_status: str,
    failure_category: str,
    first_step: JsonDict,
    semantic_energy: float,
    raw_ref: str,
) -> JsonDict:
    task_id = str(candidate.get("task_id") or candidate.get("item_id") or candidate.get("sample_id"))
    row = {
        "candidate_id": (
            f"{source_filename}:{candidate_kind}:{task_id}:"
            f"{candidate.get('mode', 'single')}:{candidate.get('seed', row_index)}"
        ),
        "source_artifact": f"results/{source_filename}",
        "candidate_kind": candidate_kind,
        "task_id": task_id,
        "mode": str(candidate.get("mode") or "single"),
        "model_hf_id": str(candidate.get("model_hf_id") or candidate.get("model_name") or "unknown"),
        "final_verifier_outcome": final_outcome,
        "downstream_failed": not final_outcome,
        "schema_status": schema_status,
        "syntax_status": syntax_status,
        "failure_category": failure_category,
        "raw_response_ref": raw_ref,
        "logprob_available": _has_logprobs(candidate),
        "artifact_confidence_proxy": _artifact_confidence_proxy(candidate, final_outcome),
        "semantic_energy_proxy_failure_score": semantic_energy,
        "tokens_generated": _finite_float(candidate.get("tokens_generated")),
    }
    row.update(first_step)
    row["artifact_confidence_failure_score"] = _bounded01(1.0 - row["artifact_confidence_proxy"])
    return row


def _code_failure_category(
    candidate: JsonDict,
    schema_valid: bool,
    syntax_success: bool,
    final_outcome: bool,
) -> str:
    if final_outcome:
        return "passed"
    if not schema_valid:
        return "schema_failure"
    if not syntax_success:
        return "syntax_failure"
    if candidate.get("false_accept"):
        return "false_accept"
    if candidate.get("execution_error_type"):
        return str(candidate["execution_error_type"])
    categories = candidate.get("original_failure_categories")
    if isinstance(categories, list) and categories:
        return str(categories[0])
    return str(candidate.get("test_status") or "verifier_rejected")


def _first_step_features(raw_text: str, candidate_kind: str) -> JsonDict:
    prefix = str(raw_text or "")[:512]
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[{}()[\]:;]", prefix)
    first_token = tokens[0] if tokens else ""
    token_count = len(tokens)
    repetition = _repetition_ratio(tokens[:40])
    structure = _structure_confidence(prefix, first_token, candidate_kind)
    confidence = _bounded01(structure * (1.0 - 0.45 * repetition))
    first_step_failure = _bounded01(1.0 - confidence)
    prefix_confidence = _bounded01(confidence + (0.10 if token_count >= 4 else -0.10))
    return {
        "first_token_text": first_token,
        "first_token_proxy_confidence": confidence,
        "first_step_proxy_failure_score": first_step_failure,
        "prefix_confidence_proxy": prefix_confidence,
        "prefix_proxy_failure_score": _bounded01(1.0 - prefix_confidence),
        "early_prefix_token_count": token_count,
        "early_prefix_repetition_ratio": repetition,
    }


def _structure_confidence(prefix: str, first_token: str, candidate_kind: str) -> float:
    lowered = prefix.lstrip().lower()
    token = first_token.lower()
    if candidate_kind == "nl_to_z3_formalization":
        if lowered.startswith("{") or "(check-sat" in lowered[:240]:
            return 0.85
        if "declare-" in lowered[:240] or "z3" in lowered[:160]:
            return 0.70
        return 0.35
    if token == "def" or lowered.startswith("```python"):
        return 0.85
    if lowered.startswith("{") or "repaired_code" in lowered[:120]:
        return 0.70
    if token in {"public", "function", "signature"}:
        return 0.20
    return 0.45


def _semantic_energy_proxy(
    *,
    schema_valid: bool,
    syntax_success: bool,
    first_step_failure: float,
    tokens_generated: float | None,
) -> float:
    """Compute a pre-verifier artifact-shape energy proxy.

    This intentionally avoids final verifier acceptance, pass/fail labels, and
    verifier scores.  It is a triage proxy over structured-output health and
    early-prefix shape, not a ground-truth or reward signal.
    """
    score = 0.0
    if not schema_valid:
        score += 0.45
    if not syntax_success:
        score += 0.35
    score += 0.20 * _bounded01(first_step_failure)
    if tokens_generated is not None and (tokens_generated <= 4 or tokens_generated >= 384):
        score += 0.10
    return _bounded01(score)


def _artifact_confidence_proxy(candidate: JsonDict, final_outcome: bool) -> float:
    if isinstance(candidate.get("verifier_output"), dict):
        score = _finite_float(candidate["verifier_output"].get("score"))
        if score is not None:
            return _bounded01(score)
    score = _finite_float(candidate.get("verifier_score"))
    if score is not None:
        return _bounded01(score)
    return 1.0 if final_outcome else 0.0


def _calibration_metrics(rows: Sequence[JsonDict]) -> JsonDict:
    metrics = _empty_calibration_metrics()
    labels = [1 if row["downstream_failed"] else 0 for row in rows]
    for feature in (
        "first_step_proxy_failure_score",
        "prefix_proxy_failure_score",
        "artifact_confidence_failure_score",
        "semantic_energy_proxy_failure_score",
    ):
        scores = [_finite_float(row.get(feature)) for row in rows]
        finite_pairs = [(label, score) for label, score in zip(labels, scores, strict=True) if score is not None]
        finite_labels = [label for label, _score in finite_pairs]
        finite_scores = [score for _label, score in finite_pairs]
        metrics[feature] = {
            "metric": "auroc",
            "auroc": compute_auroc(finite_labels, finite_scores),
            "sample_count": len(finite_pairs),
            "positive_failures": sum(finite_labels),
            "negative_successes": len(finite_labels) - sum(finite_labels),
            "score_direction": "larger_score_predicts_downstream_failure",
            "proxy_only": True,
            "false_positive_examples": _threshold_examples(rows, feature, False),
            "false_negative_examples": _threshold_examples(rows, feature, True),
        }
    return metrics


def _empty_calibration_metrics() -> JsonDict:
    return {
        "direct_logprob_features": {
            "available": False,
            "reason": "source artifacts did not record token logprobs or top-k logprobs",
        },
        "first_step_proxy_failure_score": _empty_metric(),
        "prefix_proxy_failure_score": _empty_metric(),
        "artifact_confidence_failure_score": _empty_metric(),
        "semantic_energy_proxy_failure_score": _empty_metric(),
    }


def _empty_metric() -> JsonDict:
    return {
        "metric": "auroc",
        "auroc": None,
        "sample_count": 0,
        "positive_failures": 0,
        "negative_successes": 0,
        "score_direction": "larger_score_predicts_downstream_failure",
        "proxy_only": True,
        "false_positive_examples": [],
        "false_negative_examples": [],
    }


def _threshold_examples(rows: Sequence[JsonDict], feature: str, want_false_negative: bool) -> list[JsonDict]:
    examples: list[JsonDict] = []
    for row in rows:
        score = _finite_float(row.get(feature))
        if score is None:
            continue
        predicted_failure = score >= 0.50
        actual_failure = bool(row["downstream_failed"])
        if want_false_negative and actual_failure and not predicted_failure:
            examples.append(_compact_example(row, feature, "false_negative"))
        if not want_false_negative and not actual_failure and predicted_failure:
            examples.append(_compact_example(row, feature, "false_positive"))
        if len(examples) == 3:
            return examples
    return examples


def _triage_examples(rows: Sequence[JsonDict]) -> list[JsonDict]:
    examples: list[JsonDict] = []
    examples.extend(_threshold_examples(rows, "first_step_proxy_failure_score", False)[:2])
    examples.extend(_threshold_examples(rows, "first_step_proxy_failure_score", True)[:2])
    high_risk = sorted(
        rows,
        key=lambda row: row["semantic_energy_proxy_failure_score"],
        reverse=True,
    )[:2]
    examples.extend(_compact_example(row, "semantic_energy_proxy_failure_score", "high_risk") for row in high_risk)
    seen: set[tuple[str, str]] = set()
    deduped: list[JsonDict] = []
    for example in examples:
        key = (str(example["candidate_id"]), str(example["example_type"]))
        if key not in seen:
            deduped.append(example)
            seen.add(key)
    return deduped


def _compact_example(row: JsonDict, feature: str, example_type: str) -> JsonDict:
    return {
        "example_type": example_type,
        "candidate_id": row["candidate_id"],
        "source_artifact": row["source_artifact"],
        "candidate_kind": row["candidate_kind"],
        "task_id": row["task_id"],
        "feature": feature,
        "score": row.get(feature),
        "downstream_failed": row["downstream_failed"],
        "schema_status": row["schema_status"],
        "syntax_status": row["syntax_status"],
        "failure_category": row["failure_category"],
    }


def _failure_modes(rows: Sequence[JsonDict], sources: JsonDict) -> JsonDict:
    schema_count = sum(1 for row in rows if row["schema_status"] == "invalid")
    syntax_count = sum(1 for row in rows if row["syntax_status"] == "invalid")
    categories = Counter(str(row["failure_category"]) for row in rows if row["downstream_failed"])
    monitor_payload = sources[f"results/{EXP2968_FILENAME}"]["payload"]
    monitor_results = monitor_payload.get("monitor_results") or []
    monitor_failures = sum(1 for row in monitor_results if isinstance(row, dict) and not row.get("checks_passed", False))
    return {
        "schema_failure": {
            "count": schema_count,
            "explanation": "Candidate-level schema or structured-output parsing failed.",
        },
        "syntax_failure": {
            "count": syntax_count,
            "explanation": "Code syntax or solver execution failed before final semantic acceptance.",
        },
        "final_verifier_failure": {
            "count": sum(1 for row in rows if row["downstream_failed"]),
            "explanation": "Final repair test, verifier, answer, or solver-formula outcome failed.",
        },
        "failure_category_counts": dict(sorted(categories.items())),
        "partial_monitor_context": {
            "source_available": bool(sources[f"results/{EXP2968_FILENAME}"]["available"]),
            "monitor_failure_count": monitor_failures,
            "used_as_verifier": False,
        },
    }


def _model_provenance(sources: JsonDict, rows: Sequence[JsonDict]) -> tuple[list[JsonDict], list[str]]:
    specs: list[JsonDict] = []
    seen_specs: set[str] = set()
    models = {row["model_hf_id"] for row in rows if row.get("model_hf_id") != "unknown"}
    for source in sources.values():
        payload = source.get("payload") or {}
        for key in ("models_used", "headline_models_used"):
            values = payload.get(key) or []
            if isinstance(values, list):
                models.update(str(value) for value in values)
        for spec in payload.get("model_specs") or []:
            if isinstance(spec, dict):
                marker = json.dumps(spec, sort_keys=True)
                if marker not in seen_specs:
                    specs.append(spec)
                    seen_specs.add(marker)
    return specs, sorted(models)


def _metric_clears_floor(value: Any) -> bool:
    number = _finite_float(value)
    return bool(number is not None and number >= SIGNAL_USABLE_AUROC_FLOOR)


def _has_logprobs(candidate: JsonDict) -> bool:
    for key in ("token_logprobs", "top_logprobs", "logprobs", "first_token_logprob"):
        value = candidate.get(key)
        if isinstance(value, list) and value:
            return True
        if _finite_float(value) is not None:
            return True
    return False


def _raw_text(repo_root: Path, raw_ref: str) -> str:
    if not raw_ref:
        return ""
    path = Path(raw_ref)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.is_file():
        return ""
    if resolved.suffix == ".json":
        payload = _read_json(resolved)
        return "\n".join(str(payload.get(key) or "") for key in ("draft_text", "structured_output"))
    return resolved.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _repetition_ratio(tokens: Iterable[str]) -> float:
    token_list = [token.lower() for token in tokens]
    if not token_list:
        return 1.0
    counts = Counter(token_list)
    return max(counts.values()) / len(token_list)
