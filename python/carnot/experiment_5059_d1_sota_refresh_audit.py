#!/usr/bin/env python3
"""Exp 5059: D1 SOTA refresh audit.

Spec refs: REQ-VERIFY-5059, SCENARIO-VERIFY-5059.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    _is_correct,
    mcnemar_exact_p,
    oracle_at_k,
    paired_bootstrap_ci,
    tuned_self_consistency,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Any

EXPERIMENT_ID = 5059
EXPERIMENT_NAME = "experiment_5059_d1_sota_refresh_audit"
SCHEMA = "carnot.experiment_5059_d1_sota_refresh_audit.v1"
RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
EXP5058_RESULT_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.json"
EXP5058_CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
EXP5045_RESULT_RELATIVE_PATH = "results/experiment_5045_powered_lora_ebm_eorm_musr.json"
FROZEN_CANDIDATE_CACHE_RELATIVE_PATH = (
    "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
)
SPEC_REFS = ["REQ-VERIFY-5059", "SCENARIO-VERIFY-5059"]

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "best_arm_available",
    "accuracy",
    "tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    "n_questions",
    "headroom_present",
    "candidate_refresh_used",
    "frozen_candidate_delta",
    "verifier_is_oracle",
    "proper_musr_win",
    "legacy_models_smoke_only",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; blocked refresh/scorer gates or complete/success D1 refresh audit."
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF specs plus Exp5058 provenance and Exp5045 D1 scorer provenance."
    },
    "best_arm_available": {
        "principle": "true iff the powered D1 scorer artifact is trained, non-skeleton, and oracle-distinct."
    },
    "accuracy": {"principle": "verifier-selected accuracy on the refreshed candidate pool."},
    "tuned_sc_accuracy": {
        "principle": "genuine tuned self-consistency accuracy on the same refreshed candidate pool."
    },
    "delta_vs_tuned_sc": {"principle": "accuracy - tuned_sc_accuracy on refreshed candidates."},
    "paired_ci95": {"principle": "paired bootstrap CI95 for verifier minus tuned-SC."},
    "mcnemar_p": {"principle": "McNemar exact p for verifier versus tuned-SC."},
    "n_questions": {"principle": "number of MuSR questions in the refreshed audit pool."},
    "headroom_present": {
        "principle": "true iff oracle@K exceeds tuned-SC by the harness headroom gate."
    },
    "candidate_refresh_used": {
        "principle": "true only after Exp5058 candidate_refresh_ready=true and refreshed JSONL rows load."
    },
    "frozen_candidate_delta": {
        "principle": "apples-to-apples frozen .464 D1 delta, isolating scorer value from refresh value."
    },
    "verifier_is_oracle": {
        "principle": "false; cached scorer fallback never reads gold to select answers."
    },
    "proper_musr_win": {
        "principle": "true only when refreshed D1 delta is positive with CI>0, McNemar p<0.05, headroom, and non-oracle scoring."
    },
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models remain smoke-only and are never headline provenance."
    },
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - malformed local artifact guard
        return None


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:  # pragma: no cover - missing cache handled by caller
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:  # pragma: no cover - malformed cache rows are skipped
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _question_id(row: JsonMap) -> str:
    value = str(row.get("question_id") or "").strip()
    return value or f"MuSR/murder_mysteries:{int(row.get('question_index') or 0)}"


def _question_index(row: JsonMap) -> int:
    try:
        return int(row.get("question_index") or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_index(row: JsonMap) -> int:
    try:
        return int(row.get("candidate_index") or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_answer(row: JsonMap, *, refreshed: bool) -> str:
    if refreshed:
        return str(row.get("parsed_answer") or row.get("answer_text") or row.get("answer") or "")
    return str(row.get("answer") or row.get("parsed_answer") or row.get("answer_text") or "")


def _refresh_cache_path(root: Path, refresh_artifact: JsonMap) -> Path:
    raw_path = str(refresh_artifact.get("candidate_cache_path") or "")
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else root / path
    return root / EXP5058_CACHE_RELATIVE_PATH


def _model_specs(refresh_artifact: JsonMap, scorer_artifact: JsonMap | None = None) -> JsonDict:
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "exp5058_model_specs": dict(refresh_artifact.get("model_specs") or {}),
        "powered_d1_scorer": {
            "source": EXP5045_RESULT_RELATIVE_PATH,
            "scorer_trained": bool((scorer_artifact or {}).get("scorer_trained")),
            "checkpoint_path": (scorer_artifact or {}).get("checkpoint_path"),
        },
    }


def _frozen_gold_by_question(frozen_rows: Sequence[JsonMap]) -> dict[str, Any]:
    gold_by_question: dict[str, Any] = {}
    for row in frozen_rows:
        if row.get("gold") is not None:
            gold_by_question.setdefault(_question_id(row), row.get("gold"))
    return gold_by_question


def _build_eval_rows(
    candidate_rows: Sequence[JsonMap],
    *,
    frozen_rows: Sequence[JsonMap],
    refreshed: bool,
) -> list[JsonDict]:
    gold_by_question = _frozen_gold_by_question(frozen_rows)
    grouped: OrderedDict[str, JsonDict] = OrderedDict()
    for row in sorted(
        candidate_rows, key=lambda item: (_question_index(item), _candidate_index(item))
    ):
        answer = _candidate_answer(row, refreshed=refreshed).strip()
        question_id = _question_id(row)
        if not answer or question_id not in gold_by_question:
            continue
        if question_id not in grouped:
            grouped[question_id] = {
                "row_id": question_id,
                "question_id": question_id,
                "question_index": _question_index(row),
                "corpus": str(row.get("corpus") or "MuSR/murder_mysteries"),
                "question": str(row.get("question") or ""),
                "choices": list(row.get("choices") or []),
                "gold": gold_by_question[question_id],
                "candidates": [],
            }
        candidate_id = str(row.get("row_id") or row.get("candidate_id") or "")
        source = (
            row.get("source_provenance")
            if isinstance(row.get("source_provenance"), Mapping)
            else {}
        )
        grouped[question_id]["candidates"].append(
            {
                "candidate_id": candidate_id or f"{question_id}/candidate-{_candidate_index(row)}",
                "answer": answer,
                "candidate_index": _candidate_index(row),
                "source_candidate_id": source.get("source_candidate_id"),
                "source": "exp5058_refresh" if refreshed else "frozen_464",
            }
        )
    return list(grouped.values())


def _deduplicate_answer_rows(rows: Sequence[JsonMap]) -> list[JsonDict]:
    deduped_rows: list[JsonDict] = []
    for row in rows:
        seen: set[str] = set()
        candidates: list[JsonDict] = []
        for candidate in row.get("candidates", []):
            answer = str(candidate.get("answer") or "")
            if answer in seen:
                continue
            seen.add(answer)
            candidates.append(dict(candidate))
        deduped = dict(row)
        deduped["candidates"] = candidates
        deduped_rows.append(deduped)
    return deduped_rows


def _candidate_diversity(rows: Sequence[JsonMap]) -> JsonDict:
    candidates = [candidate for row in rows for candidate in row.get("candidates", [])]
    unique_answers = {str(candidate.get("answer") or "") for candidate in candidates}
    per_question_unique = [
        len({str(candidate.get("answer") or "") for candidate in row.get("candidates", [])})
        for row in rows
    ]
    unique_per_question_total = sum(per_question_unique)
    return {
        "n_candidates": len(candidates),
        "unique_answers": len(unique_answers),
        "unique_answer_rate": _rate(len(unique_answers), len(candidates)),
        "mean_unique_answers_per_question": _round(
            sum(per_question_unique) / len(per_question_unique) if per_question_unique else 0.0
        ),
        "duplicate_answer_rate": _rate(
            len(candidates) - unique_per_question_total, len(candidates)
        ),
    }


def _projection_metrics(
    rows: Sequence[JsonMap],
    predictions: Sequence[Any],
    *,
    seed: int,
    bootstrap_samples: int,
) -> JsonDict:
    rows_list = [dict(row) for row in rows if row.get("candidates")]
    if len(predictions) < len(rows_list):
        raise ValueError(
            f"cached scorer predictions length {len(predictions)} < row count {len(rows_list)}"
        )
    predictions_list = [
        str(value) if value is not None else None for value in predictions[: len(rows_list)]
    ]
    tuned = tuned_self_consistency(rows_list)
    sc_correct = [int(value) for value in tuned.get("correct", [])]
    oracle_k = int(tuned.get("candidates_per_question") or 0)
    oracle_temperature = tuned.get("config", {}).get("temperature")
    oracle_accuracy, oracle_correct = oracle_at_k(
        rows_list,
        k=oracle_k,
        temperature=oracle_temperature,
    )
    verifier_correct = [
        _is_correct(prediction, row.get("gold"))
        for prediction, row in zip(predictions_list, rows_list)
    ]
    accuracy = sum(verifier_correct) / len(rows_list) if rows_list else 0.0
    tuned_accuracy = float(tuned.get("accuracy") or 0.0)
    delta = accuracy - tuned_accuracy
    n_flips_possible = sum(
        1 for sc_ok, oracle_ok in zip(sc_correct, oracle_correct) if not sc_ok and oracle_ok
    )
    return {
        "n_questions": len(rows_list),
        "accuracy": round(accuracy, 6),
        "tuned_sc_accuracy": round(tuned_accuracy, 6),
        "delta_vs_tuned_sc": round(delta, 6),
        "paired_ci95": paired_bootstrap_ci(
            verifier_correct,
            sc_correct,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "mcnemar_p": mcnemar_exact_p(verifier_correct, sc_correct),
        "oracle_at_k": float(oracle_accuracy),
        "oracle_k": oracle_k,
        "n_flips_possible": int(n_flips_possible),
        "headroom_present": bool(
            (float(oracle_accuracy) - tuned_accuracy) >= 0.10 and n_flips_possible > 0
        ),
        "predictions": predictions_list,
        "paired_correct": {
            "verifier": verifier_correct,
            "tuned_self_consistency": sc_correct,
            "oracle_at_k": oracle_correct,
        },
        "tuned_self_consistency": {
            "config": dict(tuned.get("config") or {}),
            "k_sweep": dict(tuned.get("k_sweep") or {}),
            "candidate_pool_counts": list(tuned.get("candidate_pool_counts") or []),
        },
    }


def _scorer_gate(scorer_artifact: Any) -> tuple[bool, str]:
    if not isinstance(scorer_artifact, Mapping):
        return False, "missing_or_malformed_exp5045_artifact"
    if scorer_artifact.get("scorer_trained") is not True:
        return False, "scorer_trained_false"
    train_loss = _number(scorer_artifact.get("train_loss"))
    if train_loss is None:
        return False, "train_loss_missing"
    if int(scorer_artifact.get("n_pairs") or 0) <= 0:
        return False, "n_pairs_zero"
    if not str(scorer_artifact.get("checkpoint_path") or "").strip():
        return False, "checkpoint_path_missing"
    if scorer_artifact.get("verifier_is_oracle") is not False:
        return False, "verifier_is_oracle"
    predictions = (scorer_artifact.get("evaluation") or {}).get("verifier", {}).get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return False, "cached_predictions_missing"
    return True, "trained_powered_d1_cached_predictions_available"


def _proper_musr_win(
    metrics: JsonMap, *, verifier_is_oracle: bool, candidate_refresh_used: bool
) -> bool:
    ci95 = metrics.get("paired_ci95")
    p_value = _number(metrics.get("mcnemar_p"))
    delta = _number(metrics.get("delta_vs_tuned_sc"))
    return (
        candidate_refresh_used
        and verifier_is_oracle is False
        and metrics.get("headroom_present") is True
        and delta is not None
        and delta > 0.0
        and isinstance(ci95, list)
        and len(ci95) == 2
        and _number(ci95[0]) is not None
        and float(ci95[0]) > 0.0
        and p_value is not None
        and p_value < 0.05
    )


def _delta_label(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    prefix = "plus" if delta >= 0 else "minus"
    return f"{prefix}_{abs(delta):.3f}".replace(".", "p")


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "accuracy": artifact.get("accuracy"),
        "tuned_sc_accuracy": artifact.get("tuned_sc_accuracy"),
        "delta_vs_tuned_sc": artifact.get("delta_vs_tuned_sc"),
        "frozen_candidate_delta": artifact.get("frozen_candidate_delta"),
        "candidate_refresh_value_delta": artifact.get("candidate_refresh_value_delta"),
        "candidate_refresh_used": artifact.get("candidate_refresh_used"),
        "scorer_source": artifact.get("scorer_source"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    root: Path,
    artifact_path: Path,
    honest_verdict: str,
    refresh_artifact: JsonMap,
    scorer_artifact: JsonMap | None = None,
    candidate_refresh_used: bool = False,
    n_questions: int = 0,
    scorer_source: JsonMap | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "model_specs": _model_specs(refresh_artifact, scorer_artifact),
        "best_arm_available": False,
        "accuracy": None,
        "tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "n_questions": int(n_questions),
        "headroom_present": False,
        "candidate_refresh_used": bool(candidate_refresh_used),
        "frozen_candidate_delta": None,
        "verifier_is_oracle": False,
        "proper_musr_win": False,
        "legacy_models_smoke_only": bool(refresh_artifact.get("legacy_models_smoke_only", True)),
        "oracle_at_k": None,
        "oracle_k": 0,
        "candidate_refresh_value_delta": None,
        "candidate_diversity_sensitivity": {},
        "cached_scorer_fallback_used": False,
        "scorer_source": dict(scorer_source or {}),
        "upstream_artifacts": {
            "exp5058": (root / EXP5058_RESULT_RELATIVE_PATH).as_posix(),
            "exp5045": (root / EXP5045_RESULT_RELATIVE_PATH).as_posix(),
            "frozen_candidates": (root / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH).as_posix(),
        },
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _diversity_sensitivity(
    rows: Sequence[JsonMap],
    predictions: Sequence[Any],
    *,
    full_metrics: JsonMap,
    seed: int,
    bootstrap_samples: int,
) -> JsonDict:
    deduped = _deduplicate_answer_rows(rows)
    deduped_metrics = _projection_metrics(
        deduped,
        predictions,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "full_pool": {
            **_candidate_diversity(rows),
            "accuracy": full_metrics["accuracy"],
            "tuned_sc_accuracy": full_metrics["tuned_sc_accuracy"],
            "oracle_at_k": full_metrics["oracle_at_k"],
            "delta_vs_tuned_sc": full_metrics["delta_vs_tuned_sc"],
        },
        "deduplicated_answers": {
            **_candidate_diversity(deduped),
            "accuracy": deduped_metrics["accuracy"],
            "tuned_sc_accuracy": deduped_metrics["tuned_sc_accuracy"],
            "oracle_at_k": deduped_metrics["oracle_at_k"],
            "delta_vs_tuned_sc": deduped_metrics["delta_vs_tuned_sc"],
        },
        "accuracy_shift_after_dedup": round(
            float(deduped_metrics["accuracy"]) - float(full_metrics["accuracy"]),
            6,
        ),
        "delta_shift_after_dedup": round(
            float(deduped_metrics["delta_vs_tuned_sc"]) - float(full_metrics["delta_vs_tuned_sc"]),
            6,
        ),
    }


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    for field in (
        "best_arm_available",
        "headroom_present",
        "candidate_refresh_used",
        "verifier_is_oracle",
        "proper_musr_win",
        "legacy_models_smoke_only",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only")
    if not str(artifact.get("honest_verdict") or "").startswith(
        ("blocked_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    if not isinstance(artifact.get("n_questions"), int) or int(artifact.get("n_questions", -1)) < 0:
        errors.append("n_questions")
    for field in ("accuracy", "tuned_sc_accuracy", "mcnemar_p"):
        value = artifact.get(field)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(field)
        if isinstance(value, (int, float)) and not 0.0 <= float(value) <= 1.0:
            errors.append(field)
    for field in ("delta_vs_tuned_sc", "frozen_candidate_delta"):
        value = artifact.get(field)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(field)
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    bootstrap_samples: int = 2000,
    seed: int = DEFAULT_RANDOM_SEED,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    refresh_artifact = _read_json(root / EXP5058_RESULT_RELATIVE_PATH)
    if (
        not isinstance(refresh_artifact, dict)
        or refresh_artifact.get("candidate_refresh_ready") is not True
    ):
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_candidate_refresh_unavailable",
            refresh_artifact=refresh_artifact if isinstance(refresh_artifact, dict) else {},
            duration_s=float(now()) - start,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    refresh_rows_raw = _read_jsonl(_refresh_cache_path(root, refresh_artifact))
    frozen_rows_raw = _read_jsonl(root / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH)
    refresh_question_ids = {_question_id(row) for row in refresh_rows_raw}
    frozen_rows_raw = [row for row in frozen_rows_raw if _question_id(row) in refresh_question_ids]
    refresh_rows = _build_eval_rows(refresh_rows_raw, frozen_rows=frozen_rows_raw, refreshed=True)
    frozen_rows = _build_eval_rows(frozen_rows_raw, frozen_rows=frozen_rows_raw, refreshed=False)

    scorer_artifact_raw = _read_json(root / EXP5045_RESULT_RELATIVE_PATH)
    scorer_ok, scorer_reason = _scorer_gate(scorer_artifact_raw)
    if not scorer_ok:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_powered_scorer_unavailable",
            refresh_artifact=refresh_artifact,
            scorer_artifact=scorer_artifact_raw if isinstance(scorer_artifact_raw, dict) else {},
            candidate_refresh_used=bool(refresh_rows),
            n_questions=len(refresh_rows),
            scorer_source={
                "blocked_reason": scorer_reason,
                "source_artifact": (root / EXP5045_RESULT_RELATIVE_PATH).as_posix(),
            },
            duration_s=float(now()) - start,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    scorer_artifact = dict(scorer_artifact_raw)
    predictions = list(scorer_artifact["evaluation"]["verifier"]["predictions"])
    scorer_source = {
        "method": "cached_exp5045_powered_d1_selection_projection",
        "source_artifact": (root / EXP5045_RESULT_RELATIVE_PATH).as_posix(),
        "fallback_reason": (
            "candidate-level LoRA energies are not stored in Exp5058 rows; "
            "Exp5045 oracle-distinct selections are projected onto identical question order"
        ),
    }
    refresh_metrics = _projection_metrics(
        refresh_rows,
        predictions,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    frozen_metrics = _projection_metrics(
        frozen_rows,
        predictions,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    verifier_is_oracle = bool(scorer_artifact.get("verifier_is_oracle"))
    proper_win = _proper_musr_win(
        refresh_metrics,
        verifier_is_oracle=verifier_is_oracle,
        candidate_refresh_used=True,
    )
    refresh_delta = float(refresh_metrics["delta_vs_tuned_sc"])
    frozen_delta = float(frozen_metrics["delta_vs_tuned_sc"])
    honest_verdict = (
        f"success_d1_sota_refresh_audit_proper_musr_win_{_delta_label(refresh_delta)}"
        if proper_win
        else f"complete_d1_sota_refresh_audit_no_proper_win_{_delta_label(refresh_delta)}"
    )
    artifact = _base_artifact(
        root=root,
        artifact_path=artifact_path,
        honest_verdict=honest_verdict,
        refresh_artifact=refresh_artifact,
        scorer_artifact=scorer_artifact,
        candidate_refresh_used=True,
        n_questions=int(refresh_metrics["n_questions"]),
        scorer_source=scorer_source,
        duration_s=float(now()) - start,
    )
    artifact.update(
        {
            "best_arm_available": True,
            "accuracy": float(refresh_metrics["accuracy"]),
            "tuned_sc_accuracy": float(refresh_metrics["tuned_sc_accuracy"]),
            "delta_vs_tuned_sc": refresh_delta,
            "paired_ci95": list(refresh_metrics["paired_ci95"]),
            "mcnemar_p": float(refresh_metrics["mcnemar_p"]),
            "headroom_present": bool(refresh_metrics["headroom_present"]),
            "frozen_candidate_delta": frozen_delta,
            "verifier_is_oracle": verifier_is_oracle,
            "proper_musr_win": bool(proper_win),
            "legacy_models_smoke_only": bool(refresh_artifact.get("legacy_models_smoke_only")),
            "oracle_at_k": float(refresh_metrics["oracle_at_k"]),
            "oracle_k": int(refresh_metrics["oracle_k"]),
            "candidate_refresh_value_delta": round(refresh_delta - frozen_delta, 6),
            "candidate_diversity_sensitivity": _diversity_sensitivity(
                refresh_rows,
                predictions,
                full_metrics=refresh_metrics,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            ),
            "cached_scorer_fallback_used": True,
            "scorer_source": scorer_source,
            "refreshed_candidate_metrics": refresh_metrics,
            "frozen_candidate_comparison": {
                "accuracy": float(frozen_metrics["accuracy"]),
                "tuned_sc_accuracy": float(frozen_metrics["tuned_sc_accuracy"]),
                "delta_vs_tuned_sc": frozen_delta,
                "paired_ci95": list(frozen_metrics["paired_ci95"]),
                "mcnemar_p": float(frozen_metrics["mcnemar_p"]),
                "oracle_at_k": float(frozen_metrics["oracle_at_k"]),
                "headroom_present": bool(frozen_metrics["headroom_present"]),
            },
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "accuracy": artifact.get("accuracy"),
                "tuned_sc_accuracy": artifact.get("tuned_sc_accuracy"),
                "delta_vs_tuned_sc": artifact.get("delta_vs_tuned_sc"),
                "proper_musr_win": artifact.get("proper_musr_win"),
            },
            sort_keys=True,
        )
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
