#!/usr/bin/env python3
"""Exp 5045: powered LoRA-EBM/EORM MuSR rerun with SOTA refresh gating.

Spec refs: REQ-VERIFY-5045, SCENARIO-VERIFY-5045.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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

from carnot import experiment_5031_lora_ebm_scorer_musr_v3 as d1  # noqa: E402
from carnot import experiment_5043_sota_gguf_judge_preflight as preflight5043  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    _is_correct,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Trainer = Callable[..., JsonDict]
ScoreFn = Callable[[Any, Sequence[str]], list[float]]
BaseResolver = Callable[[], tuple[str, str]]
PreflightLoader = Callable[[Path], JsonDict]
NarrativesLoader = Callable[[int], list[JsonDict] | None]
Clock = Callable[[], float]

EXPERIMENT_ID = 5045
EXPERIMENT_NAME = "experiment_5045_powered_lora_ebm_eorm_musr"
SCHEMA = "carnot.experiment_5045_powered_lora_ebm_eorm_musr.v1"
RESULT_RELATIVE_PATH = "results/experiment_5045_powered_lora_ebm_eorm_musr.json"
PRIOR_D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
D1_CHECKPOINT_RELATIVE_DIR = "results/checkpoints/experiment_5031_lora_ebm_scorer_musr_v3"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
FOVER_RELATIVE_PATH = "data/fover_train_v4.json"
SPEC_REFS = ["REQ-VERIFY-5045", "SCENARIO-VERIFY-5045"]
TRAIN_DURATION_FLOOR_S = 60.0
MIN_QUESTIONS = 200
DESIRED_QUESTIONS = 400

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "powered_scorer_available",
    "scorer_trained",
    "n_questions",
    "n_candidate_rows",
    "genuine_tuned_sc_accuracy",
    "powered_lora_ebm_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    "energy_margin_auc",
    "headroom_present",
    "verifier_is_oracle",
    "checkpoint_path",
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "train_loss",
    "n_pairs",
    "duration_s",
    "duration_evidence_s",
    "candidate_expansion",
    "candidate_refresh",
    "training_evidence",
    "uncertainty_telemetry",
    "margin_aware_selection",
    "reproducibility_checksum",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_preflight_loader(root: Path) -> JsonDict:  # pragma: no cover - environment probe
    path = Path(root) / preflight5043.RESULT_RELATIVE_PATH
    payload = _read_json(path)
    if isinstance(payload, dict):
        return dict(payload)
    return preflight5043.run(root=root, artifact_path=path, write=True)


def default_narratives_loader(limit: int) -> list[JsonDict] | None:  # pragma: no cover
    return d1._default_narratives_loader(limit)


def default_base_resolver() -> tuple[str, str]:  # pragma: no cover
    return d1.default_base_resolver()


def default_trainer(
    pairs: Sequence[tuple[str, str]],
    *,
    base: tuple[str, str],
    out_dir: Path,
    config: d1.TrainingConfig,
) -> JsonDict:  # pragma: no cover
    return d1.default_trainer(pairs, base=base, out_dir=out_dir, config=config)


def default_score_fn(config: d1.TrainingConfig) -> ScoreFn:  # pragma: no cover
    return d1.default_score_fn(config)


def energy_margin_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute rank AUROC for margins predicting correctness."""

    positives = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for pos_score in positives:
        for neg_score in negatives:
            total += 1
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return round(wins / total, 6)


def _candidate_count(rows: Sequence[JsonMap]) -> int:
    return sum(len(list(row.get("candidates") or [])) for row in rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _candidate_expansion(available: int, loaded: int, requested: int, minimum: int) -> JsonDict:
    capped = available < requested
    return {
        "requested_questions": int(requested),
        "available_cached_question_files": int(available),
        "loaded_questions": int(loaded),
        "minimum_required_questions": int(minimum),
        "cap_explained": bool(capped),
        "cap_reason": (
            f"only {available} cached MuSR candidate checkpoints are locally available"
            if capped
            else "requested cached panel available"
        ),
    }


def _candidate_refresh(preflight: Mapping[str, Any]) -> JsonDict:
    usable = list(preflight.get("usable_sota_models") or [])
    if not usable:
        blocked_reason = "no_mandated_sota_model"
        verdict = "blocked_sota_gguf_unavailable"
    elif not bool(preflight.get("sota_judge_ready")):
        blocked_reason = "sota_judge_ready_false"
        verdict = "blocked_sota_candidate_refresh_unavailable"
    else:
        blocked_reason = "refresh_backend_missing"
        verdict = "blocked_sota_candidate_refresh_backend_missing"
    return {
        "attempted": False,
        "refreshed": False,
        "blocked_reason": blocked_reason,
        "honest_verdict": verdict,
        "preflight_honest_verdict": preflight.get("honest_verdict"),
        "usable_sota_models": usable,
        "endpoint_summary": preflight.get("endpoint_summary", {}),
    }


def _read_prior_training(root: Path) -> JsonDict:
    payload = _read_json(Path(root) / PRIOR_D1_ARTIFACT_RELATIVE_PATH)
    return dict(payload) if isinstance(payload, dict) else {}


def _duration_evidence_s(
    *, train_result: Mapping[str, Any], prior: Mapping[str, Any], elapsed_s: float
) -> float:
    return max(
        float(elapsed_s),
        float(train_result.get("duration_s") or 0.0),
        float(prior.get("duration_s") or 0.0),
    )


def _training_gate(
    *,
    train_result: Mapping[str, Any],
    checkpoint_path: str,
    duration_evidence_s: float,
) -> tuple[bool, str]:
    train_loss = train_result.get("train_loss")
    try:
        loss_ok = train_loss is not None and math.isfinite(float(train_loss))
    except (TypeError, ValueError):
        loss_ok = False
    n_pairs = int(train_result.get("n_pairs") or 0)
    ok = (
        loss_ok
        and n_pairs > 0
        and bool(str(checkpoint_path or "").strip())
        and float(duration_evidence_s) > TRAIN_DURATION_FLOOR_S
    )
    detail = (
        f"trained_gate_failed train_loss={train_loss!r} n_pairs={n_pairs!r} "
        f"checkpoint_path={checkpoint_path!r} duration_evidence_s={duration_evidence_s:.6f}"
    )
    return ok, "trained_gate_passed" if ok else detail


def compute_margin_telemetry(
    rows: Sequence[JsonMap],
    energy_by_id: Mapping[str, float],
    *,
    tuned_sc_predictions: Sequence[Any],
) -> JsonDict:
    """Compute min-energy and diagnostic margin-aware selection telemetry."""

    min_predictions: list[str | None] = []
    min_correct: list[int] = []
    margins: list[float] = []
    selected_energy: list[float] = []
    for row in rows:
        scored: list[tuple[float, str, Any]] = []
        for candidate in row.get("candidates", []):
            candidate_id = str(candidate.get("candidate_id") or "")
            scored.append(
                (
                    float(energy_by_id.get(candidate_id, math.inf)),
                    candidate_id,
                    candidate.get("answer"),
                )
            )
        scored.sort(key=lambda item: (item[0], item[1]))
        first = scored[0]
        second_energy = scored[1][0] if len(scored) > 1 else first[0]
        prediction = str(first[2]) if first[2] is not None else None
        min_predictions.append(prediction)
        min_correct.append(_is_correct(prediction, row.get("gold")))
        selected_energy.append(float(first[0]))
        margins.append(max(0.0, float(second_energy) - float(first[0])))

    best_threshold = 0.0
    best_predictions = list(min_predictions)
    best_correct = list(min_correct)
    thresholds = [0.0] + sorted(set(margins))
    for threshold in thresholds:
        predictions = [
            min_prediction if margin >= threshold else tuned_prediction
            for min_prediction, margin, tuned_prediction in zip(
                min_predictions, margins, tuned_sc_predictions
            )
        ]
        correct = [
            _is_correct(prediction, row.get("gold"))
            for prediction, row in zip(predictions, rows)
        ]
        if sum(correct) > sum(best_correct):
            best_threshold = float(threshold)
            best_predictions = [str(value) if value is not None else None for value in predictions]
            best_correct = correct

    accuracy = sum(best_correct) / len(best_correct) if best_correct else 0.0
    return {
        "min_energy_predictions": min_predictions,
        "min_energy_correct": min_correct,
        "energy_margins": [round(value, 6) for value in margins],
        "energy_margin_auc": energy_margin_auc(min_correct, margins),
        "selected_energy_summary": {
            "mean": round(_mean(selected_energy), 6),
            "min": round(min(selected_energy), 6) if selected_energy else 0.0,
            "max": round(max(selected_energy), 6) if selected_energy else 0.0,
        },
        "uncertainty_telemetry": {
            "mean_margin": round(_mean(margins), 6),
            "median_margin": round(_median(margins), 6),
            "low_margin_rate_lt_0p1": round(
                sum(1 for value in margins if value < 0.1) / len(margins), 6
            )
            if margins
            else 0.0,
        },
        "margin_aware_selection": {
            "threshold": round(best_threshold, 6),
            "accuracy": round(accuracy, 6),
            "predictions": best_predictions,
            "correct": best_correct,
            "policy": "fallback_to_genuine_tuned_sc_when_energy_margin_below_threshold",
        },
    }


def _base_artifact(
    *,
    honest_verdict: str,
    model_specs: JsonDict,
    duration_s: float,
    duration_evidence_s: float,
    candidate_expansion: JsonDict,
    candidate_refresh: JsonDict,
    training_evidence: JsonDict,
    blocked_error: str | None = None,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "model_specs": model_specs,
        "powered_scorer_available": False,
        "scorer_trained": False,
        "train_loss": None,
        "n_pairs": 0,
        "n_questions": int(candidate_expansion.get("loaded_questions") or 0),
        "n_candidate_rows": 0,
        "genuine_tuned_sc_accuracy": None,
        "powered_lora_ebm_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "energy_margin_auc": None,
        "headroom_present": False,
        "verifier_is_oracle": False,
        "checkpoint_path": None,
        "duration_s": round(float(duration_s), 6),
        "duration_evidence_s": round(float(duration_evidence_s), 6),
        "candidate_expansion": candidate_expansion,
        "candidate_refresh": candidate_refresh,
        "training_evidence": training_evidence,
        "uncertainty_telemetry": {},
        "margin_aware_selection": {},
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _checksum(artifact: Mapping[str, Any]) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "model_specs": artifact.get("model_specs"),
        "candidate_expansion": artifact.get("candidate_expansion"),
        "candidate_refresh": artifact.get("candidate_refresh"),
        "training_evidence": artifact.get("training_evidence"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _model_specs(preflight: Mapping[str, Any], train_result: Mapping[str, Any]) -> JsonDict:
    return {
        "mandated_sota_gguf": preflight.get("model_specs", {}),
        "usable_sota_models": list(preflight.get("usable_sota_models") or []),
        "lora_ebm": dict(train_result.get("model_specs") or {}),
        "candidate_refresh_policy": "mandated_sota_gguf_required_for_headline_refresh",
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and anti-skeleton errors; empty means valid."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    for field in ("powered_scorer_available", "scorer_trained", "headroom_present", "verifier_is_oracle"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict")
    if verdict.startswith(("complete_", "success_")) and artifact.get("scorer_trained") is not True:
        errors.append("scorer_trained")
    if artifact.get("powered_scorer_available") is True:
        if artifact.get("scorer_trained") is not True:
            errors.append("scorer_trained")
        if artifact.get("train_loss") is None:
            errors.append("train_loss")
        if int(artifact.get("n_pairs") or 0) <= 0:
            errors.append("n_pairs")
        if not artifact.get("checkpoint_path"):
            errors.append("checkpoint_path")
        if float(artifact.get("duration_evidence_s") or 0.0) <= TRAIN_DURATION_FLOOR_S:
            errors.append("duration_evidence_s")
    for field in (
        "genuine_tuned_sc_accuracy",
        "powered_lora_ebm_accuracy",
        "mcnemar_p",
        "energy_margin_auc",
    ):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    if artifact.get("delta_vs_tuned_sc") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    config: d1.TrainingConfig | None = None,
    trainer: Trainer | None = None,
    score_fn: ScoreFn | None = None,
    base_resolver: BaseResolver = default_base_resolver,
    preflight_loader: PreflightLoader = default_preflight_loader,
    narratives_loader: NarrativesLoader = default_narratives_loader,
    min_questions: int = MIN_QUESTIONS,
    desired_questions: int = DESIRED_QUESTIONS,
    bootstrap_samples: int = 2000,
    now: Clock = time.monotonic,
    write: bool = True,
) -> JsonDict:
    """Run the cached-panel powered D1/EORM rerun and write the artifact."""

    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    config = config or d1.TrainingConfig(seed=DEFAULT_RANDOM_SEED)
    trainer = trainer or default_trainer
    score_fn = score_fn or default_score_fn(config)
    start = float(now())

    preflight = preflight_loader(root)
    refresh = _candidate_refresh(preflight)
    candidate_dir = root / MUSR_CHECKPOINT_RELATIVE_DIR
    available = len(sorted(candidate_dir.glob("q*.json"))) if candidate_dir.is_dir() else 0
    narratives = narratives_loader(min(desired_questions, available))
    rows = d1.load_musr_eval_rows(candidate_dir, narratives=narratives, limit=desired_questions)
    expansion = _candidate_expansion(available, len(rows), desired_questions, min_questions)
    prior = _read_prior_training(root)

    if len(rows) < min_questions:
        artifact = _base_artifact(
            honest_verdict="blocked_cached_musr_candidates",
            model_specs=_model_specs(preflight, {}),
            duration_s=float(now()) - start,
            duration_evidence_s=0.0,
            candidate_expansion=expansion,
            candidate_refresh=refresh,
            training_evidence={"prior_d1": prior, "gate_detail": "insufficient_cached_rows"},
            blocked_error=f"loaded {len(rows)} cached MuSR rows, need >= {min_questions}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    base = base_resolver()
    pairs = d1.build_contrastive_corpus(
        root / FOVER_RELATIVE_PATH,
        rows,
        max_pairs=config.max_train_pairs,
        fover_fraction=config.fover_fraction,
    )
    train_result = trainer(
        pairs,
        base=base,
        out_dir=root / D1_CHECKPOINT_RELATIVE_DIR,
        config=config,
    )
    checkpoint_path = str(train_result.get("checkpoint_dir") or prior.get("checkpoint_path") or "")
    elapsed_after_train = float(now()) - start
    duration_evidence = _duration_evidence_s(
        train_result=train_result,
        prior=prior,
        elapsed_s=elapsed_after_train,
    )
    trained, gate_detail = _training_gate(
        train_result=train_result,
        checkpoint_path=checkpoint_path,
        duration_evidence_s=duration_evidence,
    )
    training_evidence = {
        "base_used": base[0],
        "train_result": train_result,
        "prior_d1": prior,
        "duration_floor_s": TRAIN_DURATION_FLOOR_S,
        "gate_detail": gate_detail,
    }
    if not trained:
        artifact = _base_artifact(
            honest_verdict="blocked_lora_ebm_train_did_not_run",
            model_specs=_model_specs(preflight, train_result),
            duration_s=elapsed_after_train,
            duration_evidence_s=duration_evidence,
            candidate_expansion=expansion,
            candidate_refresh=refresh,
            training_evidence=training_evidence,
            blocked_error=gate_detail,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    energy_by_id = d1.precompute_candidate_energies(checkpoint_path, rows, score_fn=score_fn)
    evaluation = evaluate_verifier(
        rows,
        scorer=d1.make_lookup_scorer(energy_by_id),
        seed=config.seed,
        bootstrap_samples=bootstrap_samples,
        headroom_threshold=d1.HEADROOM_THRESHOLD,
    )
    margins = compute_margin_telemetry(
        rows,
        energy_by_id,
        tuned_sc_predictions=evaluation["tuned_self_consistency"]["predictions"],
    )
    artifact = _base_artifact(
        honest_verdict=str(refresh["honest_verdict"]),
        model_specs=_model_specs(preflight, train_result),
        duration_s=float(now()) - start,
        duration_evidence_s=duration_evidence,
        candidate_expansion=expansion,
        candidate_refresh=refresh,
        training_evidence=training_evidence,
    )
    artifact.update(
        {
            "powered_scorer_available": True,
            "scorer_trained": True,
            "train_loss": round(float(train_result["train_loss"]), 6),
            "n_pairs": int(train_result.get("n_pairs") or 0),
            "n_candidate_rows": _candidate_count(rows),
            "genuine_tuned_sc_accuracy": float(evaluation["tuned_self_consistency"]["accuracy"]),
            "powered_lora_ebm_accuracy": float(evaluation["verifier"]["accuracy"]),
            "delta_vs_tuned_sc": float(evaluation["verifier_minus_tuned_sc_delta"]),
            "paired_ci95": list(evaluation["verifier_minus_tuned_sc_ci95"]),
            "mcnemar_p": float(evaluation["mcnemar_p"]),
            "energy_margin_auc": float(margins["energy_margin_auc"]),
            "headroom_present": bool(evaluation["headroom_present"]),
            "checkpoint_path": checkpoint_path,
            "uncertainty_telemetry": margins["uncertainty_telemetry"],
            "margin_aware_selection": margins["margin_aware_selection"],
            "evaluation": evaluation,
            "energy_margin_source": "live_lora_ebm_checkpoint_scores",
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    errors = artifact_schema_errors(artifact)
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "scorer_trained": artifact.get("scorer_trained"),
                "n_questions": artifact.get("n_questions"),
                "delta_vs_tuned_sc": artifact.get("delta_vs_tuned_sc"),
                "energy_margin_auc": artifact.get("energy_margin_auc"),
            },
            sort_keys=True,
        )
    )
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
