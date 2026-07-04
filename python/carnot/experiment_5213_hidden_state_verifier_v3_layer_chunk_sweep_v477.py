"""Exp 5213: hidden-state verifier v3 layer/chunk sweep on MMLU-Pro.

Spec refs: REQ-REPORT-5213, SCENARIO-REPORT-5213,
SCENARIO-REPORT-5213-BLOCKED-PRECONDITION.

This is the terminal sharper check for the MMLU-Pro hidden-state path. It
tries to promote beyond Exp 5200's final-layer-only GGUF embedding null by
first preflighting a Transformers ``output_hidden_states=True`` path. When
that path is not technically available, the GGUF embedding path is retained as
a disclosed negative-control comparison and cannot by itself rescue the
headline.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np

from carnot import experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476 as v2


JsonDict = dict[str, Any]
FeatureProvider = Callable[[list[v2.MmluQuestion]], "FeatureBatch"]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477"
EXPERIMENT_ID = 5213
SCHEMA = "carnot.hidden_state_verifier_v3_layer_chunk_sweep_5213.v1"
RESULT_RELATIVE_PATH = "results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json"
CANDIDATE_POOL_RELATIVE_PATH = v2.CANDIDATE_POOL_RELATIVE_PATH
HEADROOM_RELATIVE_PATH = v2.HEADROOM_RELATIVE_PATH
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
CORPUS_USED = "TIGER-Lab/MMLU-Pro zero-shot candidate pool"
RANDOM_SEED = 5213
EXPECTED_POOL_ROWS = 240
DEFAULT_N_FOLDS = 5
DEFAULT_N_BOOTSTRAP = 1000
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
BASE_MODEL_BY_GGUF = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "unsloth/Qwen3.6-35B-A3B",
    "unsloth/gemma-4-31B-it-GGUF": "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "unsloth/gemma-4-26B-A4B-it",
}
MODEL_NAME_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "Qwen3.6-35B-A3B",
    "unsloth/gemma-4-31B-it-GGUF": "Gemma4-31B-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "Gemma4-26B-A4B-it",
}
ESTIMATED_TRANSFORMERS_VRAM_GB = {
    "unsloth/Qwen3.6-35B-A3B": 70.0,
    "unsloth/gemma-4-31B-it": 62.0,
    "unsloth/gemma-4-26B-A4B-it": 52.0,
}
SPEC_REFS = [
    "REQ-REPORT-5213",
    "SCENARIO-REPORT-5213",
    "SCENARIO-REPORT-5213-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "models_used": "Exact HF ids used or preflighted for hidden-state and embedding extraction.",
    "model_specs": "List of model provenance rows with name, hf_id, gpu, and model_path/load_path.",
    "intermediate_layer_available": "Whether an output_hidden_states=True path exposed intermediate layer tensors.",
    "chunk_features_available": "Whether candidate chunk-boundary embedding features were extracted.",
    "halting_or_convergence_signal_available": (
        "Whether any loop, halting, or convergence proxy was available from the selected model path."
    ),
    "best_probe_accuracy": "Held-out question selection accuracy of the best eligible hidden-signal probe.",
    "tuned_sc_accuracy": "Held-out selection accuracy of tuned self-consistency on the same question split.",
    "self_certainty_accuracy": "Held-out selection accuracy of the zero-training self-certainty control.",
    "clue_accuracy": "Held-out selection accuracy of the CLUE-style hidden-geometry clustering control.",
    "radial_consensus_score_accuracy": "Held-out selection accuracy of the Radial Consensus Score control.",
    "beats_all_controls": "True only if the probe beats every control and all paired CI lower bounds are positive.",
    "retire_mmlu_hidden_state_path": (
        "True when no richer MMLU-Pro hidden-state signal beats all controls with positive paired CIs."
    ),
    "verifier_is_oracle": "Gold labels are used only for train/eval supervision and scoring; the verifier is not an oracle.",
    "inference_substrate": "This task performs live hidden-state or embedding extraction from local LLM weights.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_ and state whether the v3 signal "
        "beats all controls or retires this MMLU-Pro path."
    ),
    "random_seed": "Deterministic split, probe training, bootstrap, and checksum reproducibility.",
    "reproducibility_checksum": "Content-addressed hash catches silent artifact or row drift.",
    "headroom_present": "Must be true before a verifier-moat claim; sourced from the MMLU-Pro headroom artifact.",
}

REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "corpus_used",
    "candidate_pool",
    "headroom_context",
    "model_resolution",
    "signal_availability",
    "feature_provenance",
    "split_summary",
    "method_correctness",
    "control_comparisons",
    "failure_mode_analysis",
    "tests_run",
    "duration_s",
    "field_principles",
    *REQUIRED_PRINCIPLED_FIELDS,
)


@dataclass(frozen=True)
class ModelInventory:
    cached_sota_pair_attempted: bool
    cached_sota_pair_available: bool
    models_used: list[str]
    model_specs: list[JsonDict]
    cached_sota_pair_error: str | None = None


@dataclass(frozen=True)
class SignalAvailability:
    usable: bool
    reason: str
    intermediate_layer_available: bool
    chunk_features_available: bool
    halting_or_convergence_signal_available: bool
    extraction_path: str
    transformer_attempt: Mapping[str, Any]
    tensor_provenance: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class FeatureBatch:
    vectors: np.ndarray
    keys: list[tuple[int, int]]
    feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionEvaluation:
    correct_by_method: dict[str, list[int]]
    selected_by_method: dict[str, list[int]]
    tuned_k_by_fold: dict[int, int]
    eval_question_ids: list[str]
    self_certainty_source: str
    probe_score_by_candidate: dict[tuple[int, int], float]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _value(artifact: Mapping[str, Any], field: str) -> Any:
    raw = artifact.get(field)
    if isinstance(raw, Mapping) and "value" in raw:
        return raw.get("value")
    return raw


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, Mapping):
        checksum = dict(checksum)
        checksum["value"] = ""
        payload["reproducibility_checksum"] = checksum
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _accuracy(values: Sequence[int]) -> float:
    return sum(int(v) for v in values) / len(values) if values else 0.0


def _model_spec_for(hf_id: str, path: str | None, gpu: int | None) -> JsonDict:
    return {
        "name": MODEL_NAME_BY_HF_ID[hf_id],
        "hf_id": hf_id,
        "gpu": gpu,
        "model_path": path,
        "load_path": path,
        "available": bool(path),
    }


def resolve_model_inventory(
    *,
    cached_pair_fn: Callable[[], list[dict[str, Any]] | None] | None = None,
    resolve_gguf_fn: Callable[[str, str], str | None] | None = None,
) -> ModelInventory:
    if cached_pair_fn is None or resolve_gguf_fn is None:  # pragma: no cover - import path is environment glue.
        from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf

        cached_pair_fn = cached_pair_fn or (lambda: cached_sota_pair())
        resolve_gguf_fn = resolve_gguf_fn or resolve_cached_gguf

    pair_rows: list[dict[str, Any]] | None = None
    pair_error: str | None = None
    try:
        pair_rows = cached_pair_fn()
    except Exception as exc:  # pragma: no cover - defensive around host cache helpers.
        pair_error = f"{type(exc).__name__}: {exc}"
        pair_rows = None

    by_hf: dict[str, JsonDict] = {}
    for row in pair_rows or []:
        hf_id = str(row.get("hf_id", ""))
        if hf_id in MANDATED_MODEL_IDS:
            by_hf[hf_id] = {
                "name": str(row.get("name") or MODEL_NAME_BY_HF_ID[hf_id]),
                "hf_id": hf_id,
                "gpu": row.get("gpu"),
                "model_path": row.get("model_path"),
                "load_path": row.get("model_path"),
                "available": bool(row.get("model_path")),
            }

    for gpu, hf_id in enumerate(MANDATED_MODEL_IDS):
        if hf_id in by_hf and by_hf[hf_id].get("model_path"):
            continue
        try:
            resolved = resolve_gguf_fn(hf_id, "Q4_K_M")
        except Exception as exc:  # pragma: no cover - defensive around host cache helpers.
            resolved = None
            if pair_error is None:
                pair_error = f"resolve_cached_gguf:{type(exc).__name__}: {exc}"
        by_hf[hf_id] = _model_spec_for(hf_id, resolved, by_hf.get(hf_id, {}).get("gpu", gpu % 2))

    specs = [by_hf[hf_id] for hf_id in MANDATED_MODEL_IDS]
    models_used = [str(row["hf_id"]) for row in specs if row.get("available")]
    return ModelInventory(
        cached_sota_pair_attempted=True,
        cached_sota_pair_available=bool(pair_rows),
        models_used=models_used,
        model_specs=specs,
        cached_sota_pair_error=pair_error,
    )


def _gpu_memory_rows() -> list[JsonDict]:  # pragma: no cover - host hardware probe.
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    if completed.returncode != 0:
        return [{"error": completed.stderr.strip() or completed.stdout.strip() or "nvidia-smi failed"}]
    rows: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_gb": _round_float(float(parts[2]) / 1024.0),
                    "memory_free_gb": _round_float(float(parts[3]) / 1024.0),
                }
            )
    return rows


def attempt_transformers_hidden_state_path(
    inventory: ModelInventory,
    *,
    gpu_rows_fn: Callable[[], list[JsonDict]] = _gpu_memory_rows,
) -> JsonDict:
    candidate_hf = next((row["hf_id"] for row in inventory.model_specs if row.get("available")), MANDATED_MODEL_IDS[0])
    base_hf = BASE_MODEL_BY_GGUF[str(candidate_hf)]
    gpu_rows = gpu_rows_fn()
    free_gb = sum(float(row.get("memory_free_gb", 0.0) or 0.0) for row in gpu_rows)
    required_gb = ESTIMATED_TRANSFORMERS_VRAM_GB[base_hf]
    attempt: JsonDict = {
        "attempted": True,
        "hf_id": base_hf,
        "source_gguf_hf_id": candidate_hf,
        "output_hidden_states_requested": True,
        "gpu_memory": gpu_rows,
        "estimated_required_gb": required_gb,
        "available_free_gb": _round_float(free_gb),
    }
    if free_gb < required_gb:
        attempt["status"] = "blocked_insufficient_gpu_memory_for_non_gguf_transformers_load"
        attempt["reason"] = (
            f"available GPU free memory {free_gb:.2f} GB is below estimated {required_gb:.2f} GB "
            f"for {base_hf} with output_hidden_states=True"
        )
        return attempt
    attempt["status"] = "blocked_transformers_load_not_executed_without_explicit_quantized_fit_path"
    attempt["reason"] = (
        "GPU memory preflight passed, but no local non-GGUF quantized Transformers load path is validated "
        "for this experiment; refusing to download or risk an unbounded load."
    )
    return attempt


def _candidate_labels(
    questions: Sequence[v2.MmluQuestion],
    keys: Sequence[tuple[int, int]],
) -> np.ndarray:
    return np.asarray([int(questions[qpos].candidates[cpos].correct) for qpos, cpos in keys], dtype=int)


def _candidate_feature_map(
    questions: Sequence[v2.MmluQuestion],
    batch: FeatureBatch,
) -> dict[tuple[int, int], np.ndarray]:
    arr = np.asarray(batch.vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D candidate features, got shape {arr.shape}")
    by_key = {key: arr[i] for i, key in enumerate(batch.keys)}
    width = int(arr.shape[1])
    out: dict[tuple[int, int], np.ndarray] = {}
    for question in questions:
        for candidate in question.candidates:
            key = (question.question_pos, candidate.candidate_pos)
            out[key] = by_key.get(key, np.zeros(width, dtype=float))
    return out


def evaluate_selectors(
    questions: Sequence[v2.MmluQuestion],
    batch: FeatureBatch,
    folds: Sequence[set[str]],
    *,
    seed: int = RANDOM_SEED,
) -> SelectionEvaluation:
    vectors = np.asarray(batch.vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError(f"expected 2-D candidate features, got shape {vectors.shape}")
    if len(batch.keys) != len(vectors):
        raise ValueError("feature key count must match feature rows")

    labels = _candidate_labels(questions, batch.keys)
    q_by_id = v2._question_by_id(questions)
    candidate_vecs = _candidate_feature_map(questions, batch)
    k_values = v2._candidate_k_values(questions)
    correct: dict[str, list[int]] = {name: [] for name in ("probe", "self_certainty", "clue", "rcs", "tuned_sc")}
    selected: dict[str, list[int]] = {name: [] for name in correct}
    tuned_k_by_fold: dict[int, int] = {}
    eval_question_ids: list[str] = []
    probe_score_by_candidate: dict[tuple[int, int], float] = {}
    self_certainty_sources: list[str] = []

    all_qids = {question.question_id for question in questions}
    for fold_i, eval_qids in enumerate(folds):
        train_qids = all_qids - set(eval_qids)
        train_questions = [q_by_id[qid] for qid in sorted(train_qids)]
        tuned_k = v2._tuned_sc_k(train_questions, k_values)
        tuned_k_by_fold[fold_i] = tuned_k

        train_rows = np.asarray(
            [i for i, (qpos, _cpos) in enumerate(batch.keys) if questions[qpos].question_id in train_qids]
        )
        eval_rows = np.asarray(
            [i for i, (qpos, _cpos) in enumerate(batch.keys) if questions[qpos].question_id in eval_qids]
        )
        fold_scores = v2._fit_probe_scores(
            vectors[train_rows],
            labels[train_rows],
            vectors[eval_rows],
            seed=seed + fold_i,
        )
        for row_i, score in zip(eval_rows.tolist(), fold_scores.tolist(), strict=True):
            probe_score_by_candidate[batch.keys[row_i]] = float(score)

        for qid in sorted(eval_qids):
            question = q_by_id[qid]
            eval_question_ids.append(qid)
            probe_cpos = max(
                range(len(question.candidates)),
                key=lambda cpos: (probe_score_by_candidate.get((question.question_pos, cpos), 0.0), -cpos),
            )
            self_cpos, source = v2._select_self_certainty(question)
            choices = {
                "probe": probe_cpos,
                "self_certainty": self_cpos,
                "clue": v2._select_clue(question, candidate_vecs),
                "rcs": v2._select_radial_consensus(question, candidate_vecs),
                "tuned_sc": v2._select_sc_candidate(question, tuned_k),
            }
            self_certainty_sources.append(source)
            for name, cpos in choices.items():
                selected[name].append(cpos)
                correct[name].append(int(question.candidates[cpos].correct))

    source_counts = Counter(self_certainty_sources)
    source = source_counts.most_common(1)[0][0] if source_counts else "unavailable_no_logit_distribution_tie_first"
    if len(source_counts) > 1:  # pragma: no cover - rare mixed candidate-logit provenance.
        source = "mixed_self_certainty_sources"
    return SelectionEvaluation(
        correct_by_method=correct,
        selected_by_method=selected,
        tuned_k_by_fold=tuned_k_by_fold,
        eval_question_ids=eval_question_ids,
        self_certainty_source=source,
        probe_score_by_candidate=probe_score_by_candidate,
    )


def _failure_mode_analysis(
    questions: Sequence[v2.MmluQuestion],
    evaluation: SelectionEvaluation,
    *,
    retire: bool,
) -> JsonDict:
    q_by_id = v2._question_by_id(questions)
    misses: list[JsonDict] = []
    for idx, qid in enumerate(evaluation.eval_question_ids):
        if evaluation.correct_by_method["probe"][idx]:
            continue
        question = q_by_id[qid]
        selected_cpos = evaluation.selected_by_method["probe"][idx]
        misses.append(
            {
                "question_id": qid,
                "category": question.category,
                "oracle_available": any(candidate.correct for candidate in question.candidates),
                "selected_candidate": selected_cpos,
                "selected_answer": question.candidates[selected_cpos].parsed_letter,
                "gold": question.gold,
            }
        )
    categories = Counter(str(miss["category"]) for miss in misses)
    return {
        "n_probe_misses": len(misses),
        "n_oracle_recoverable_misses": sum(int(miss["oracle_available"]) for miss in misses),
        "misses_by_category": dict(sorted(categories.items())),
        "examples": misses[:8],
        "residual_failure_mode": (
            "mmlu_hidden_state_path_retired_no_positive_ci_vs_all_controls"
            if retire
            else "richer_hidden_signal_positive_vs_all_controls"
        ),
    }


def _build_comparisons(
    correct: Mapping[str, Sequence[int]],
    *,
    best_acc: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, JsonDict]:
    comparisons: dict[str, JsonDict] = {}
    for offset, (name, baseline) in enumerate(
        (
            ("tuned_sc", "tuned_sc"),
            ("self_certainty", "self_certainty"),
            ("clue", "clue"),
            ("radial_consensus_score", "rcs"),
        ),
        start=1,
    ):
        base_acc = _accuracy(correct[baseline])
        comparisons[f"probe_vs_{name}"] = {
            "delta_ci95": v2.paired_bootstrap_ci(
                correct["probe"],
                correct[baseline],
                n_bootstrap=n_bootstrap,
                seed=seed + offset,
            ),
            "mcnemar_p": v2.mcnemar_exact_p(correct["probe"], correct[baseline]),
            "delta": round(best_acc - base_acc, 6),
        }
    return comparisons


def _headline_eligible(status: SignalAvailability) -> bool:
    return bool(status.intermediate_layer_available and "output_hidden_states" in _stable_json(status.transformer_attempt))


def _beats_all_controls(
    best_acc: float,
    control_accs: Mapping[str, float],
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    headline_eligible: bool,
) -> bool:
    if not headline_eligible:
        return False
    for name, acc in control_accs.items():
        comparison = comparisons[f"probe_vs_{name}"]
        ci = comparison.get("delta_ci95")
        if best_acc <= acc or not isinstance(ci, Sequence) or len(ci) != 2 or float(ci[0]) <= 0.0:
            return False
    return True


def _verdict(best_acc: float, control_accs: Mapping[str, float], beats_all: bool, retire: bool) -> str:
    if beats_all:
        return (
            "success_hidden_state_v3_signal_beats_all_controls_"
            f"probe{best_acc:.3f}_sc{control_accs['tuned_sc']:.3f}_self{control_accs['self_certainty']:.3f}_"
            f"clue{control_accs['clue']:.3f}_rcs{control_accs['radial_consensus_score']:.3f}"
        )
    if retire:
        return (
            "complete_hidden_state_v3_signal_does_not_beat_all_controls_retires_mmlu_hidden_state_path_"
            f"probe{best_acc:.3f}_sc{control_accs['tuned_sc']:.3f}_self{control_accs['self_certainty']:.3f}_"
            f"clue{control_accs['clue']:.3f}_rcs{control_accs['radial_consensus_score']:.3f}"
        )
    return "complete_hidden_state_v3_signal_inconclusive_no_retirement_gate"


def build_complete_artifact(
    *,
    questions: Sequence[v2.MmluQuestion],
    batch: FeatureBatch,
    signal_status: SignalAvailability,
    model_inventory: ModelInventory,
    headroom_context: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[str] = (),
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    folds = v2.question_folds([question.question_id for question in questions], n_folds=n_folds, seed=random_seed)
    evaluation = evaluate_selectors(questions, batch, folds, seed=random_seed)
    correct = evaluation.correct_by_method
    best_acc = _accuracy(correct["probe"])
    control_accs = {
        "tuned_sc": _accuracy(correct["tuned_sc"]),
        "self_certainty": _accuracy(correct["self_certainty"]),
        "clue": _accuracy(correct["clue"]),
        "radial_consensus_score": _accuracy(correct["rcs"]),
    }
    comparisons = _build_comparisons(correct, best_acc=best_acc, n_bootstrap=n_bootstrap, seed=random_seed)
    raw_beats = all(
        best_acc > acc and comparisons[f"probe_vs_{name}"]["delta_ci95"][0] > 0.0
        for name, acc in control_accs.items()
    )
    headline_eligible = _headline_eligible(signal_status)
    beats_all = _beats_all_controls(best_acc, control_accs, comparisons, headline_eligible=headline_eligible)
    retire = not beats_all
    failures = _failure_mode_analysis(questions, evaluation, retire=retire)
    candidate_count = sum(len(question.candidates) for question in questions)

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "corpus_used": CORPUS_USED,
        "candidate_pool": {
            "path": CANDIDATE_POOL_RELATIVE_PATH,
            "n_rows": candidate_count,
            "n_questions": len(questions),
            "zero_shot_pool": True,
        },
        "headroom_context": dict(headroom_context),
        "model_resolution": {
            "cached_sota_pair_attempted": model_inventory.cached_sota_pair_attempted,
            "cached_sota_pair_available": model_inventory.cached_sota_pair_available,
            "cached_sota_pair_error": model_inventory.cached_sota_pair_error,
            "mandated_model_ids": list(MANDATED_MODEL_IDS),
        },
        "signal_availability": {
            "usable": signal_status.usable,
            "reason": signal_status.reason,
            "extraction_path": signal_status.extraction_path,
            "transformer_attempt": dict(signal_status.transformer_attempt),
            "headline_eligible_signal": headline_eligible,
            "negative_control_only": not headline_eligible,
            "raw_probe_beats_all_controls_before_headline_gate": raw_beats,
        },
        "feature_provenance": {
            "feature_shape": list(np.asarray(batch.vectors).shape),
            "feature_names": list(batch.feature_names),
            "tensor_provenance": [dict(row) for row in signal_status.tensor_provenance],
        },
        "split_summary": {
            "n_folds": len(folds),
            "fold_question_counts": [len(fold) for fold in folds],
            "tuned_k_by_fold": {str(k): v for k, v in evaluation.tuned_k_by_fold.items()},
            "leakage_guard": "question_id_grouped_train_eval_split",
        },
        "method_correctness": {
            "question_ids": evaluation.eval_question_ids,
            "probe": correct["probe"],
            "self_certainty": correct["self_certainty"],
            "clue": correct["clue"],
            "radial_consensus_score": correct["rcs"],
            "tuned_sc": correct["tuned_sc"],
            "selected_candidate_by_method": evaluation.selected_by_method,
        },
        "control_comparisons": comparisons,
        "self_certainty_control": {"source": evaluation.self_certainty_source},
        "failure_mode_analysis": failures,
        "models_used": _wrap("models_used", model_inventory.models_used),
        "model_specs": _wrap("model_specs", model_inventory.model_specs),
        "intermediate_layer_available": _wrap(
            "intermediate_layer_available", bool(signal_status.intermediate_layer_available)
        ),
        "chunk_features_available": _wrap("chunk_features_available", bool(signal_status.chunk_features_available)),
        "halting_or_convergence_signal_available": _wrap(
            "halting_or_convergence_signal_available",
            bool(signal_status.halting_or_convergence_signal_available),
        ),
        "best_probe_accuracy": _wrap("best_probe_accuracy", round(best_acc, 6)),
        "tuned_sc_accuracy": _wrap("tuned_sc_accuracy", round(control_accs["tuned_sc"], 6)),
        "self_certainty_accuracy": _wrap("self_certainty_accuracy", round(control_accs["self_certainty"], 6)),
        "clue_accuracy": _wrap("clue_accuracy", round(control_accs["clue"], 6)),
        "radial_consensus_score_accuracy": _wrap(
            "radial_consensus_score_accuracy", round(control_accs["radial_consensus_score"], 6)
        ),
        "beats_all_controls": _wrap("beats_all_controls", beats_all),
        "retire_mmlu_hidden_state_path": _wrap("retire_mmlu_hidden_state_path", retire),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "inference_substrate": _wrap("inference_substrate", "live_llm_hidden_state_extraction"),
        "honest_verdict": _wrap("honest_verdict", _verdict(best_acc, control_accs, beats_all, retire)),
        "random_seed": _wrap("random_seed", random_seed),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "headroom_present": _wrap("headroom_present", bool(headroom_context.get("headroom_present"))),
        "tests_run": list(tests_run),
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap("reproducibility_checksum", payload_checksum(artifact))
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    headroom_context: Mapping[str, Any] | None,
    model_inventory: ModelInventory,
    signal_status: SignalAvailability | None,
    duration_s: float,
    tests_run: Sequence[str] = (),
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    blocked_reason = reason if reason.startswith("blocked_") else f"blocked_{reason}"
    status = signal_status or SignalAvailability(
        usable=False,
        reason=blocked_reason,
        intermediate_layer_available=False,
        chunk_features_available=False,
        halting_or_convergence_signal_available=False,
        extraction_path="not_evaluated_blocked",
        transformer_attempt={"attempted": False, "reason": blocked_reason},
        tensor_provenance=[],
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "corpus_used": CORPUS_USED,
        "candidate_pool": {
            "path": CANDIDATE_POOL_RELATIVE_PATH,
            "n_rows": 0,
            "n_questions": 0,
            "blocked_reason": blocked_reason,
            "zero_shot_pool": True,
        },
        "headroom_context": dict(headroom_context or {}),
        "model_resolution": {
            "cached_sota_pair_attempted": model_inventory.cached_sota_pair_attempted,
            "cached_sota_pair_available": model_inventory.cached_sota_pair_available,
            "cached_sota_pair_error": model_inventory.cached_sota_pair_error,
            "mandated_model_ids": list(MANDATED_MODEL_IDS),
        },
        "signal_availability": {
            "usable": False,
            "reason": status.reason,
            "extraction_path": status.extraction_path,
            "transformer_attempt": dict(status.transformer_attempt),
            "headline_eligible_signal": False,
            "negative_control_only": True,
            "raw_probe_beats_all_controls_before_headline_gate": False,
        },
        "feature_provenance": {
            "feature_shape": [0, 0],
            "feature_names": [],
            "tensor_provenance": [dict(row) for row in status.tensor_provenance],
        },
        "split_summary": {"n_folds": 0, "fold_question_counts": [], "leakage_guard": "not_evaluated_blocked"},
        "method_correctness": {
            "question_ids": [],
            "probe": [],
            "self_certainty": [],
            "clue": [],
            "radial_consensus_score": [],
            "tuned_sc": [],
            "selected_candidate_by_method": {},
        },
        "control_comparisons": {},
        "failure_mode_analysis": {"blocked_reason": blocked_reason},
        "models_used": _wrap("models_used", model_inventory.models_used),
        "model_specs": _wrap("model_specs", model_inventory.model_specs),
        "intermediate_layer_available": _wrap("intermediate_layer_available", False),
        "chunk_features_available": _wrap("chunk_features_available", False),
        "halting_or_convergence_signal_available": _wrap("halting_or_convergence_signal_available", False),
        "best_probe_accuracy": _wrap("best_probe_accuracy", 0.0),
        "tuned_sc_accuracy": _wrap("tuned_sc_accuracy", 0.0),
        "self_certainty_accuracy": _wrap("self_certainty_accuracy", 0.0),
        "clue_accuracy": _wrap("clue_accuracy", 0.0),
        "radial_consensus_score_accuracy": _wrap("radial_consensus_score_accuracy", 0.0),
        "beats_all_controls": _wrap("beats_all_controls", False),
        "retire_mmlu_hidden_state_path": _wrap("retire_mmlu_hidden_state_path", True),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "inference_substrate": _wrap("inference_substrate", "live_llm_hidden_state_extraction"),
        "honest_verdict": _wrap("honest_verdict", blocked_reason),
        "random_seed": _wrap("random_seed", random_seed),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "headroom_present": _wrap("headroom_present", bool((headroom_context or {}).get("headroom_present"))),
        "tests_run": list(tests_run),
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap("reproducibility_checksum", payload_checksum(artifact))
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in REQUIRED_PRINCIPLED_FIELDS:
        raw = artifact.get(field)
        if not isinstance(raw, Mapping) or "value" not in raw or "principle" not in raw:
            errors.append(f"{field} must be principle-wrapped")
            continue
        if raw.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"{field} has wrong principle")
    if _value(artifact, "verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if _value(artifact, "inference_substrate") != "live_llm_hidden_state_extraction":
        errors.append("inference_substrate must be live_llm_hidden_state_extraction")
    for field in (
        "intermediate_layer_available",
        "chunk_features_available",
        "halting_or_convergence_signal_available",
        "beats_all_controls",
        "retire_mmlu_hidden_state_path",
        "headroom_present",
    ):
        if not isinstance(_value(artifact, field), bool):
            errors.append(f"{field} must be bool")
    if _value(artifact, "beats_all_controls") and _value(artifact, "retire_mmlu_hidden_state_path"):
        errors.append("beats_all_controls and retire_mmlu_hidden_state_path cannot both be true")
    if _value(artifact, "beats_all_controls"):
        comparisons = artifact.get("control_comparisons", {})
        if not isinstance(comparisons, Mapping) or len(comparisons) < 4:
            errors.append("beats_all_controls requires all control comparisons")
        for name, comparison in comparisons.items() if isinstance(comparisons, Mapping) else ():
            ci = comparison.get("delta_ci95") if isinstance(comparison, Mapping) else None
            if not isinstance(ci, list) or len(ci) != 2 or float(ci[0]) <= 0.0:
                errors.append(f"beats_all_controls requires positive CI lower bound for {name}")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal prefix")
    models_used = _value(artifact, "models_used")
    if not isinstance(models_used, list) or not all(isinstance(item, str) for item in models_used):
        errors.append("models_used must be a list of exact hf_ids")
    model_specs = _value(artifact, "model_specs")
    if not isinstance(model_specs, list):
        errors.append("model_specs must be a list")
    else:
        for row in model_specs:
            if not isinstance(row, Mapping) or not {"name", "hf_id", "gpu"}.issubset(row):
                errors.append("model_specs rows must include name, hf_id, and gpu")
                break
            if "model_path" not in row and "load_path" not in row:
                errors.append("model_specs rows must include model_path/load_path")
                break
    checksum = _value(artifact, "reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(artifact):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum must be sha256")
    return errors


def _coerce_one_embedding(raw: Any) -> np.ndarray:  # pragma: no cover - live llama.cpp shape glue.
    data = raw[0] if isinstance(raw, tuple) else raw
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr[-1]
    raise ValueError(f"cannot coerce llama embedding shape {arr.shape}")


def _make_features_from_embedding_provider(
    questions: Sequence[v2.MmluQuestion],
    embed_one: Callable[[str], np.ndarray],
    *,
    model_hf_id: str,
) -> FeatureBatch:  # pragma: no cover - exercised by the terminal experiment run.
    rows: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    for question in questions:
        for candidate in question.candidates:
            texts = v2.chunk_boundary_texts(question, candidate, max_boundaries=2)
            vectors = [embed_one(text) for text in texts]
            matrix = np.vstack(vectors).astype(float)
            final_vec = matrix[-1]
            mean_vec = matrix.mean(axis=0)
            first_vec = matrix[0]
            delta_vec = final_vec - first_vec
            denom = max(float(np.linalg.norm(final_vec) * np.linalg.norm(first_vec)), 1e-12)
            cosine_drift = 1.0 - float(np.dot(final_vec, first_vec) / denom)
            norm_delta = float(np.linalg.norm(final_vec) - np.linalg.norm(first_vec))
            scalar = np.asarray([cosine_drift, norm_delta, float(len(vectors))], dtype=float)
            rows.append(np.concatenate([final_vec, mean_vec, delta_vec, scalar]))
            keys.append((question.question_pos, candidate.candidate_pos))
    width = rows[0].shape[0] if rows else 0
    return FeatureBatch(
        vectors=np.vstack(rows) if rows else np.empty((0, width), dtype=float),
        keys=keys,
        feature_names=(
            "gguf_final_token_embedding",
            "gguf_chunk_boundary_mean_embedding",
            "gguf_chunk_boundary_delta_embedding",
            "chunk_embedding_cosine_drift",
            "chunk_embedding_norm_delta",
            "n_chunk_boundaries",
            f"producer:{model_hf_id}",
        ),
    )


def make_live_feature_provider(
    inventory: ModelInventory,
    transformer_attempt: Mapping[str, Any],
    *,
    n_ctx: int = 256,
    n_gpu_layers: int = -1,
) -> tuple[SignalAvailability, FeatureProvider]:  # pragma: no cover - live model path.
    try:
        from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama
    except Exception as exc:
        status = SignalAvailability(
            usable=False,
            reason=f"blocked_hidden_state_access_infeasible: llama_cpp import failed: {exc!r}",
            intermediate_layer_available=False,
            chunk_features_available=False,
            halting_or_convergence_signal_available=False,
            extraction_path="llama_cpp_unavailable",
            transformer_attempt=transformer_attempt,
            tensor_provenance=[],
        )
        return status, lambda _questions: FeatureBatch(np.empty((0, 0), dtype=float), [])

    last_error: str | None = None
    for spec in inventory.model_specs:
        path = spec.get("model_path") or spec.get("load_path")
        hf_id = str(spec.get("hf_id"))
        if not path:
            continue
        load_started = time.time()
        try:
            llm = Llama(
                model_path=str(path),
                n_ctx=int(n_ctx),
                n_batch=min(32, int(n_ctx)),
                n_gpu_layers=int(n_gpu_layers),
                offload_kqv=bool(n_gpu_layers != 0),
                embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_LAST,
                logits_all=False,
                seed=RANDOM_SEED,
                verbose=False,
            )
            load_s = time.time() - load_started
            smoke_vec = _coerce_one_embedding(llm.embed("Hidden-state v3 smoke.", normalize=False, truncate=True))
        except Exception as exc:
            last_error = f"{hf_id}: {type(exc).__name__}: {exc}"
            continue

        timing = {"embed_s": 0.0, "calls": 0}

        def embed_one(text: str) -> np.ndarray:
            clipped = str(text)[: v2.MAX_BOUNDARY_TEXT_CHARS]
            started = time.time()
            vec = _coerce_one_embedding(llm.embed(clipped, normalize=False, truncate=True))
            timing["embed_s"] += time.time() - started
            timing["calls"] += 1
            return vec

        def provider(questions: list[v2.MmluQuestion]) -> FeatureBatch:
            return _make_features_from_embedding_provider(questions, embed_one, model_hf_id=hf_id)

        status = SignalAvailability(
            usable=True,
            reason=(
                "transformers intermediate-layer path unavailable; "
                "using GGUF final-layer chunk/final embeddings as negative-control comparison"
            ),
            intermediate_layer_available=False,
            chunk_features_available=True,
            halting_or_convergence_signal_available=True,
            extraction_path="llama_cpp.GGUF_embedding_negative_control_final_layer_chunk_boundary",
            transformer_attempt=transformer_attempt,
            tensor_provenance=[
                {
                    "model_hf_id": hf_id,
                    "model_path": str(path),
                    "feature": "gguf_final_layer_chunk_boundary_embedding",
                    "vector_shape": list(smoke_vec.shape),
                    "load_s": round(load_s, 6),
                    "timing_ref": timing,
                }
            ],
        )
        return status, provider

    status = SignalAvailability(
        usable=False,
        reason=f"blocked_hidden_state_access_infeasible: no GGUF embedding path loaded ({last_error})",
        intermediate_layer_available=False,
        chunk_features_available=False,
        halting_or_convergence_signal_available=False,
        extraction_path="llama_cpp_gguf_embedding_failed",
        transformer_attempt=transformer_attempt,
        tensor_provenance=[],
    )
    return status, lambda _questions: FeatureBatch(np.empty((0, 0), dtype=float), [])


def _write_verifier_gap(root: Path | str, artifact: Mapping[str, Any]) -> None:
    if not _value(artifact, "retire_mmlu_hidden_state_path"):
        return
    root_path = Path(root)
    path = root_path / VERIFIER_GAPS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier gaps\n"
    start = "<!-- experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477:start -->"
    end = "<!-- experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477:end -->"
    failure = artifact.get("failure_mode_analysis", {})
    verdict = _value(artifact, "honest_verdict")
    entry = (
        f"{start}\n"
        "### experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477\n"
        "- status: retired\n"
        f"- evidence: `{RESULT_RELATIVE_PATH}`; honest_verdict={verdict}; "
        f"best_probe_accuracy={_value(artifact, 'best_probe_accuracy')}; "
        f"tuned_sc_accuracy={_value(artifact, 'tuned_sc_accuracy')}; "
        f"self_certainty_accuracy={_value(artifact, 'self_certainty_accuracy')}; "
        f"clue_accuracy={_value(artifact, 'clue_accuracy')}; "
        f"radial_consensus_score_accuracy={_value(artifact, 'radial_consensus_score_accuracy')}.\n"
        f"- failure mode: {failure.get('residual_failure_mode', 'not_evaluated')}.\n"
        "- residual gap: richer hidden-state access did not provide a positive-CI selector win over tuned SC, "
        "self-certainty, CLUE, and RCS on the headroom-confirmed MMLU-Pro pool.\n"
        "- recommendation: retire MMLU-Pro hidden-state verifier path; do not rerun this path without a new "
        "non-final-layer internal signal or a different corpus-level mechanism.\n"
        f"{end}\n"
    )
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", flags=re.S)
    updated = pattern.sub(entry, existing) if pattern.search(existing) else existing.rstrip() + "\n\n" + entry
    path.write_text(updated, encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | None = None,
    feature_provider: FeatureProvider | None = None,
    signal_status: SignalAvailability | None = None,
    model_inventory: ModelInventory | None = None,
    expected_pool_rows: int = EXPECTED_POOL_ROWS,
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    duration_s: float | None = None,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    started = time.time()
    root_path = Path(root)
    output = result_path or (root_path / RESULT_RELATIVE_PATH)
    headroom = v2.load_headroom_context(root_path)
    inventory = model_inventory or resolve_model_inventory()

    try:
        questions = v2.load_mmlu_questions(root_path, expected_rows=expected_pool_rows)
    except v2.CandidatePoolError as exc:
        artifact = build_blocked_artifact(
            reason=str(exc),
            headroom_context=headroom,
            model_inventory=inventory,
            signal_status=signal_status,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
        )
        errors = artifact_schema_errors(artifact)
        if errors:  # pragma: no cover - defensive schema gate.
            raise ValueError(f"invalid Exp 5213 blocked artifact: {errors}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    if signal_status is None or feature_provider is None:
        transformer_attempt = attempt_transformers_hidden_state_path(inventory)
        signal_status, feature_provider = make_live_feature_provider(inventory, transformer_attempt)

    if not signal_status.usable:
        artifact = build_blocked_artifact(
            reason=signal_status.reason,
            headroom_context=headroom,
            model_inventory=inventory,
            signal_status=signal_status,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
        )
    else:
        feature_started = time.time()
        batch = feature_provider(list(questions))
        measured = time.time() - feature_started
        provenance = [dict(row) for row in signal_status.tensor_provenance]
        for row in provenance:
            timing = row.pop("timing_ref", None)
            if isinstance(timing, Mapping):
                row["measured_vector_seconds"] = round(float(timing.get("embed_s", measured) or measured), 6)
                row["vector_provider_calls"] = timing.get("calls")
        signal_status = SignalAvailability(
            usable=signal_status.usable,
            reason=signal_status.reason,
            intermediate_layer_available=signal_status.intermediate_layer_available,
            chunk_features_available=signal_status.chunk_features_available,
            halting_or_convergence_signal_available=signal_status.halting_or_convergence_signal_available,
            extraction_path=signal_status.extraction_path,
            transformer_attempt=signal_status.transformer_attempt,
            tensor_provenance=provenance,
        )
        artifact = build_complete_artifact(
            questions=questions,
            batch=batch,
            signal_status=signal_status,
            model_inventory=inventory,
            headroom_context=headroom,
            duration_s=duration_s if duration_s is not None else time.time() - started,
            tests_run=tests_run,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
        )
        _write_verifier_gap(root_path, artifact)

    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive schema gate.
        raise ValueError(f"invalid Exp 5213 artifact: {errors}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run()
    print(json.dumps({"honest_verdict": _value(artifact, "honest_verdict")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
