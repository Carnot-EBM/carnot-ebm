"""Exp 1420 DPO-style probe over Exp 1395 verified FoVer preference pairs.

The experiment has two deliberately separate parts.  First, it preserves the
mandated GGUF model IDs as provenance and checks whether any of those local
weights are actually available.  Second, because GGUF files are inference
artifacts rather than trainable HuggingFace weight directories, it reports
direct DPO as unsupported and trains a small deterministic preference reranker
over FoVer verifier features instead of pretending a GGUF weight update
happened.

Spec: REQ-LEARN-1420, SCENARIO-LEARN-1420.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.inference.sota_models import resolve_cached_gguf


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
EXP1395_FILE = "experiment_1395_fr11_self_learning_v5.json"
OUTPUT_FILE = "experiment_1420_dpo_verified_pairs_1508.json"
DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
RUN_DATE = "20260506"
EXPERIMENT = "1420_dpo_verified_pairs_1508"
SCHEMA = "dpo_verified_pairs_probe_v1"

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary_policy_baseline",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense_baseline",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "moe_baseline",
    },
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "verified_pairs_available",
    "dpo_full_finetune_performed",
    "dpo_reranker_fallback_used",
    "dpo_improvement_pp",
    "dpo_vs_baseline_auroc",
    "local_sota_model_used",
    "headline_result_allowed",
    "honest_verdict",
)

ResolverFn = Callable[[str, str], str | None]
ImportModuleFn = Callable[[str], Any]


@dataclass(frozen=True)
class FoVerCandidate:
    """One normalized FoVer candidate with the same duplicate-ID policy as Exp 1395."""

    case_id: str
    prompt: str
    text: str
    label: str
    confidence: float
    source: str
    corpus_index: int


@dataclass(frozen=True)
class PreferencePair:
    """One explicit DPO-style preference pair.

    ``preferred_verified`` is the supervision target from Exp 1395 promotion,
    while the label/confidence/source fields are candidate features available to
    the lightweight fallback reranker.
    """

    pair_id: str
    preferred_id: str
    rejected_id: str
    prompt: str
    preferred_text: str
    rejected_text: str
    preferred_label: str
    rejected_label: str
    preferred_confidence: float
    rejected_confidence: float
    preferred_source: str
    rejected_source: str
    preferred_verified: bool
    rejected_verified: bool
    preferred_corpus_index: int
    rejected_corpus_index: int


@dataclass(frozen=True)
class DirectDPOFeasibility:
    """Honest feasibility result for direct DPO on local GGUF model provenance."""

    supported: bool
    reason: str
    packages_checked: dict[str, bool]


@dataclass(frozen=True)
class RerankerResult:
    """Measured output from the deterministic pairwise fallback trainer."""

    fallback_used: bool
    n_train_pairs: int
    n_eval_pairs: int
    auroc: float | None
    baseline_pair_accuracy: float | None
    reranker_pair_accuracy: float | None
    improvement_pp: float | None
    weights: list[float]


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1420-1: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1420", "SCENARIO-LEARN-1420"],
            "artifact_metadata": {
                "project_root": str(project_root),
                "run_date": run_date,
            },
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "model_specs": list(MODEL_SPECS),
            "verified_pairs_available": None,
            "dpo_full_finetune_performed": None,
            "dpo_reranker_fallback_used": None,
            "dpo_improvement_pp": None,
            "dpo_vs_baseline_auroc": None,
            "local_sota_model_used": None,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load an experiment artifact object from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path!s}")
    return payload


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load the FoVer JSONL source rows used by Exp 1395."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _strip_fover_prefix(value: Any) -> str:
    text = str(value)
    prefix = "dvi_v2:fover:"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _case_ids_from_memory(exp1395_artifact: Mapping[str, Any], key: str) -> set[str]:
    values = exp1395_artifact.get("memory_updates", {}).get(key, [])
    return {_strip_fover_prefix(value) for value in values}


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("step_text") or row.get("response") or row.get("answer") or "")


def normalize_fover_rows(rows: Sequence[Mapping[str, Any]]) -> list[FoVerCandidate]:
    """Normalize FoVer rows and suffix duplicate IDs exactly as Exp 1395 did."""

    normalized: list[FoVerCandidate] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        raw_id = str(
            row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or row.get("question_index")
            or f"fover_{index}"
        )
        ordinal = seen.get(raw_id, 0)
        seen[raw_id] = ordinal + 1
        case_id = raw_id if ordinal == 0 else f"{raw_id}_{ordinal}"
        normalized.append(
            FoVerCandidate(
                case_id=case_id,
                prompt=str(row.get("question") or row.get("prompt") or ""),
                text=_row_text(row),
                label=str(row.get("label") or "unknown"),
                confidence=float(row.get("confidence") or 0.0),
                source=str(row.get("source") or "unknown"),
                corpus_index=index,
            )
        )
    return normalized


def _nearest_unused_demoted(
    preferred: FoVerCandidate,
    demoted: list[FoVerCandidate],
) -> FoVerCandidate | None:
    if not demoted:
        return None
    best_index = min(
        range(len(demoted)),
        key=lambda idx: (
            abs(demoted[idx].corpus_index - preferred.corpus_index),
            demoted[idx].corpus_index,
            demoted[idx].case_id,
        ),
    )
    return demoted.pop(best_index)


def _pair_from_candidates(preferred: FoVerCandidate, rejected: FoVerCandidate) -> PreferencePair:
    return PreferencePair(
        pair_id=f"{preferred.case_id}__vs__{rejected.case_id}",
        preferred_id=preferred.case_id,
        rejected_id=rejected.case_id,
        prompt=preferred.prompt or rejected.prompt,
        preferred_text=preferred.text,
        rejected_text=rejected.text,
        preferred_label=preferred.label,
        rejected_label=rejected.label,
        preferred_confidence=preferred.confidence,
        rejected_confidence=rejected.confidence,
        preferred_source=preferred.source,
        rejected_source=rejected.source,
        preferred_verified=True,
        rejected_verified=False,
        preferred_corpus_index=preferred.corpus_index,
        rejected_corpus_index=rejected.corpus_index,
    )


def build_preference_pairs(
    *,
    exp1395_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
) -> list[PreferencePair]:
    """REQ-LEARN-1420-2: pair each Exp 1395 promotion with nearest demotion."""

    promoted_ids = _case_ids_from_memory(exp1395_artifact, "promoted")
    demoted_ids = _case_ids_from_memory(exp1395_artifact, "demoted")
    candidates = normalize_fover_rows(fover_rows)
    by_id = {candidate.case_id: candidate for candidate in candidates}
    preferred = sorted(
        (by_id[case_id] for case_id in promoted_ids if case_id in by_id),
        key=lambda candidate: (candidate.corpus_index, candidate.case_id),
    )
    demoted = sorted(
        (by_id[case_id] for case_id in demoted_ids if case_id in by_id),
        key=lambda candidate: (candidate.corpus_index, candidate.case_id),
    )

    pairs: list[PreferencePair] = []
    for preferred_candidate in preferred:
        rejected_candidate = _nearest_unused_demoted(preferred_candidate, demoted)
        if rejected_candidate is not None:
            pairs.append(_pair_from_candidates(preferred_candidate, rejected_candidate))
    return pairs


def resolve_gguf_model_checks(
    *,
    resolver_fn: ResolverFn = resolve_cached_gguf,
    preferred_quant: str = "Q4_K_M",
) -> list[dict[str, Any]]:
    """REQ-LEARN-1420-3: inspect mandated GGUF cache provenance without loading weights."""

    checks: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_path = resolver_fn(spec["hf_id"], preferred_quant)
        checks.append(
            {
                "hf_id": spec["hf_id"],
                "role": spec["role"],
                "preferred_quant": preferred_quant,
                "cached": model_path is not None,
                "model_path": model_path,
                "baseline_scoring_performed": True,
                "baseline_scoring_mode": "feature_baseline_preference_scores",
                "llama_cpp_inference_performed": False,
                "llama_cpp_inference_blocker": (
                    "model_not_cached" if model_path is None else "full_model_load_disabled_for_probe"
                ),
            }
        )
    return checks


def assess_direct_dpo_support(
    model_checks: Sequence[Mapping[str, Any]],
    *,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> DirectDPOFeasibility:
    """REQ-LEARN-1420-4: report that direct DPO cannot update GGUF weights here."""

    if not any(check.get("cached") for check in model_checks):
        return DirectDPOFeasibility(
            supported=False,
            reason="no_mandated_gguf_cache_resolved_for_dpo",
            packages_checked={"trl": False},
        )

    try:
        import_module_fn("trl")
    except ImportError:
        return DirectDPOFeasibility(
            supported=False,
            reason="trl_not_available_for_dpo",
            packages_checked={"trl": False},
        )

    return DirectDPOFeasibility(
        supported=False,
        reason="gguf_direct_weight_update_not_supported_by_trl_llama_cpp",
        packages_checked={"trl": True},
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _candidate_features(text: str, label: str, confidence: float, source: str) -> np.ndarray:
    tokens = _tokens(text)
    char_count = max(len(text), 1)
    word_count = len(tokens)
    return np.asarray(
        [
            1.0,
            float(confidence),
            1.0 if label.lower() == "incorrect" else 0.0,
            min(word_count / 80.0, 1.0),
            min(char_count / 800.0, 1.0),
            sum(ch.isdigit() for ch in text) / char_count,
            1.0 if "\\boxed" in text or "boxed" in text.lower() else 0.0,
            1.0 if "\\" in text or "$" in text else 0.0,
            1.0 if "math" in source.lower() else 0.0,
        ],
        dtype=np.float64,
    )


def _preferred_features(pair: PreferencePair) -> np.ndarray:
    return _candidate_features(
        pair.preferred_text,
        pair.preferred_label,
        pair.preferred_confidence,
        pair.preferred_source,
    )


def _rejected_features(pair: PreferencePair) -> np.ndarray:
    return _candidate_features(
        pair.rejected_text,
        pair.rejected_label,
        pair.rejected_confidence,
        pair.rejected_source,
    )


def _candidate_score(weights: np.ndarray, features: np.ndarray) -> float:
    return float(np.dot(weights, features))


def _baseline_score(confidence: float) -> float:
    return float(confidence)


def _pair_accuracy(margins: Sequence[float]) -> float | None:
    if not margins:
        return None
    total = 0.0
    for margin in margins:
        if margin > 0:
            total += 1.0
        elif margin == 0:
            total += 0.5
    return total / len(margins)


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
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
    return wins / total


def _split_pairs(
    pairs: Sequence[PreferencePair],
    train_fraction: float,
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    if len(pairs) < 2:
        return list(pairs), list(pairs)
    split_at = int(len(pairs) * train_fraction)
    split_at = min(max(split_at, 1), len(pairs) - 1)
    return list(pairs[:split_at]), list(pairs[split_at:])


def train_reranker_fallback(
    pairs: Sequence[PreferencePair],
    *,
    train_fraction: float = 0.8,
    learning_rate: float = 0.08,
    steps: int = 160,
    l2: float = 0.001,
) -> RerankerResult:
    """REQ-LEARN-1420-5: fit a deterministic pairwise preference reranker."""

    if not pairs:
        return RerankerResult(
            fallback_used=True,
            n_train_pairs=0,
            n_eval_pairs=0,
            auroc=None,
            baseline_pair_accuracy=None,
            reranker_pair_accuracy=None,
            improvement_pp=None,
            weights=[],
        )

    train_pairs, eval_pairs = _split_pairs(pairs, train_fraction)
    weights = np.zeros_like(_preferred_features(train_pairs[0]))
    diffs = np.asarray(
        [_preferred_features(pair) - _rejected_features(pair) for pair in train_pairs],
        dtype=np.float64,
    )
    for _ in range(int(steps)):
        margins = diffs @ weights
        probs = 1.0 / (1.0 + np.exp(-margins))
        gradient = ((probs - 1.0)[:, None] * diffs).mean(axis=0) + float(l2) * weights
        weights -= float(learning_rate) * gradient

    reranker_margins: list[float] = []
    baseline_margins: list[float] = []
    labels: list[int] = []
    scores: list[float] = []
    for pair in eval_pairs:
        preferred_score = _candidate_score(weights, _preferred_features(pair))
        rejected_score = _candidate_score(weights, _rejected_features(pair))
        reranker_margins.append(preferred_score - rejected_score)
        baseline_margins.append(
            _baseline_score(pair.preferred_confidence) - _baseline_score(pair.rejected_confidence)
        )
        labels.extend([1, 0])
        scores.extend([preferred_score, rejected_score])

    reranker_accuracy = _pair_accuracy(reranker_margins)
    baseline_accuracy = _pair_accuracy(baseline_margins)
    improvement_pp = (
        None
        if reranker_accuracy is None or baseline_accuracy is None
        else round((reranker_accuracy - baseline_accuracy) * 100.0, 6)
    )
    auroc = _auroc(labels, scores)

    return RerankerResult(
        fallback_used=True,
        n_train_pairs=len(train_pairs),
        n_eval_pairs=len(eval_pairs),
        auroc=None if auroc is None else round(auroc, 6),
        baseline_pair_accuracy=None if baseline_accuracy is None else round(baseline_accuracy, 6),
        reranker_pair_accuracy=None if reranker_accuracy is None else round(reranker_accuracy, 6),
        improvement_pp=improvement_pp,
        weights=[round(float(value), 8) for value in weights],
    )


def _artifact_status(pair_count: int) -> str:
    return "complete" if pair_count > 0 else "blocked"


def _honest_verdict(pair_count: int, feasibility: DirectDPOFeasibility) -> str:
    if pair_count == 0:
        return "blocked_no_preference_pairs_built"
    if feasibility.supported:
        return "direct_dpo_supported_not_executed_by_fallback_probe"
    return "gguf_dpo_unsupported_reranker_fallback_measured"


def build_artifact(
    *,
    exp1395_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    resolver_fn: ResolverFn = resolve_cached_gguf,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> dict[str, Any]:
    """Build the terminal Exp 1420 artifact from source objects."""

    pairs = build_preference_pairs(exp1395_artifact=exp1395_artifact, fover_rows=fover_rows)
    model_checks = resolve_gguf_model_checks(resolver_fn=resolver_fn)
    feasibility = assess_direct_dpo_support(model_checks, import_module_fn=import_module_fn)
    reranker = (
        train_reranker_fallback(pairs)
        if not feasibility.supported
        else RerankerResult(False, 0, 0, None, None, None, None, [])
    )
    local_sota_model_used = any(check["llama_cpp_inference_performed"] for check in model_checks)
    headline_result_allowed = bool(local_sota_model_used and feasibility.supported)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1420", "SCENARIO-LEARN-1420"],
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
        },
        "source_artifacts": [str(DEFAULT_EXP1395_PATH), str(DEFAULT_FOVER_PATH)],
        "run_date": run_date,
        "finished_at": _timestamp(),
        "status": _artifact_status(len(pairs)),
        "model_specs": list(MODEL_SPECS),
        "verified_pairs_available": len(pairs),
        "dpo_full_finetune_performed": bool(feasibility.supported),
        "dpo_reranker_fallback_used": bool(reranker.fallback_used),
        "dpo_improvement_pp": reranker.improvement_pp,
        "dpo_vs_baseline_auroc": reranker.auroc,
        "local_sota_model_used": local_sota_model_used,
        "headline_result_allowed": headline_result_allowed,
        "honest_verdict": _honest_verdict(len(pairs), feasibility),
        "gguf_model_checks": model_checks,
        "direct_dpo_feasibility": {
            "supported": feasibility.supported,
            "reason": feasibility.reason,
            "packages_checked": feasibility.packages_checked,
        },
        "reranker_fallback_metrics": {
            "n_train_pairs": reranker.n_train_pairs,
            "n_eval_pairs": reranker.n_eval_pairs,
            "baseline_pair_accuracy": reranker.baseline_pair_accuracy,
            "reranker_pair_accuracy": reranker.reranker_pair_accuracy,
            "weights": reranker.weights,
        },
        "preference_pair_sample": [
            {
                "pair_id": pair.pair_id,
                "preferred_id": pair.preferred_id,
                "rejected_id": pair.rejected_id,
                "preferred_corpus_index": pair.preferred_corpus_index,
                "rejected_corpus_index": pair.rejected_corpus_index,
            }
            for pair in pairs[:5]
        ],
        "source_counts": {
            "exp1395_fresh_verified_sample_count": exp1395_artifact.get(
                "fresh_verified_sample_count"
            ),
            "promoted_memory_count": len(_case_ids_from_memory(exp1395_artifact, "promoted")),
            "demoted_memory_count": len(_case_ids_from_memory(exp1395_artifact, "demoted")),
            "fover_rows_loaded": len(fover_rows),
        },
        "feasibility_note": (
            "Direct DPO fine-tuning was not performed because local GGUF files are "
            "llama.cpp inference artifacts, not trainable TRL/PEFT model directories."
        ),
    }


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1395_artifact: Mapping[str, Any] | None = None,
    fover_rows: Sequence[Mapping[str, Any]] | None = None,
    resolver_fn: ResolverFn = resolve_cached_gguf,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> dict[str, Any]:
    """Write bootstrap and final Exp 1420 artifacts, then return the final JSON."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    source_exp1395 = exp1395_artifact if exp1395_artifact is not None else load_json(exp1395_path)
    source_fover_rows = fover_rows if fover_rows is not None else load_jsonl(fover_path)
    artifact = build_artifact(
        exp1395_artifact=source_exp1395,
        fover_rows=source_fover_rows,
        project_root=project_root,
        run_date=run_date,
        resolver_fn=resolver_fn,
        import_module_fn=import_module_fn,
    )
    return _write_json(out_path, artifact)
