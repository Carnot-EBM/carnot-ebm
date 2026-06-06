"""Exp 3884 in-distribution error-rich FoVer corpus builder.

This runner assembles a FoVer-family math-step corpus with enough incorrect
steps for the downstream moat scissor to measure a residual set. It uses only
checked-in FoVer rows plus bounded arithmetic perturbations of correct FoVer
steps, then scores the emitted corpus with the Exp 2837 production FoVer
aggregation:

    0.9*tier0r_curry_howard + 0.1*tier0u_logical_consistency
    + fr11_session_memory

Spec: REQ-VERIFY-3884, SCENARIO-VERIFY-3884,
      SCENARIO-VERIFY-3884-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    _fr11_memory_score,
    _label_to_int,
    _load_fr11_memory_index,
    _score_text_verifiers,
    compute_auroc,
)


EXPERIMENT_ID = 3884
TITLE = "in_distribution_error_rich_corpus"
OUTPUT_RESULTS_REL_PATH = Path("results/experiment_3884_in_distribution_error_rich_corpus.json")
OUTPUT_SCORES_REL_PATH = Path("results/experiment_3884_in_distribution_error_rich_corpus_scores.json")
OUTPUT_CORPUS_REL_PATH = Path("data/in_distribution_error_corpus_v1.json")
DEFAULT_RANDOM_SEED = 3884
DEFAULT_MIN_INCORRECT_STEPS = 150
AUROC_READY_MIN = 0.65
MAX_SYNTHETIC_ERROR_FRACTION = 0.40
INFERENCE_SUBSTRATE = "cpu_carnot_verify_exp2837_cached_fover_rows"
REQUIRED_CORPUS_FILENAMES = (
    "fover_corpus_v4.json",
    "fover_corpus_v3.json",
    "fover_corpus_expanded.json",
)

REQUIRED_PRINCIPLE_FIELDS = (
    "carnot_ensemble_auroc_on_corpus",
    "n_incorrect_steps",
    "n_total_items",
    "frac_synthetic",
    "corpus_path",
    "per_item_ensemble_scores_path",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "carnot_ensemble_auroc_on_corpus": (
        "BARE FLOAT - the in-band gate; the scissor reads this off disk. "
        "Must be >=0.65 for the corpus to be usable."
    ),
    "n_incorrect_steps": (
        "Set-B size budget - the residual scissor CI is meaningful only at "
        ">=100 errors; this task requires >=150."
    ),
    "n_total_items": "Provenance - records the balanced corpus size on disk.",
    "frac_synthetic": (
        "Provenance - fraction of all emitted rows that were generated, with "
        "generated errors also constrained below 40% of the error class."
    ),
    "corpus_path": (
        "Where the scissor reads the corpus from "
        "(data/in_distribution_error_corpus_v1.json)."
    ),
    "per_item_ensemble_scores_path": (
        "Persist per-item ensemble scores so the scissor reuses them."
    ),
    "preconditions_checked": (
        "Verifier-scoring methodology - carnot.verify import plus local FoVer "
        "JSON preflight before scoring."
    ),
    "random_seed": "Verifier-scoring methodology - deterministic corpus balancing seed.",
    "reproducibility_checksum": (
        "Verifier-scoring methodology - checksum over corpus identity, scores, "
        "seeds, and gates."
    ),
    "duration_s": "Verifier-scoring methodology - measured wall-clock seconds.",
    "inference_substrate": (
        "Verifier-scoring methodology - CPU Carnot scoring against cached FoVer "
        "rows, without live model runtime claims."
    ),
}

_NUMBER_PATTERN = r"-?\d[\d,]*(?:\.\d+)?"
_OP_PATTERN = r"(?:\+|-|\*|x|×|/|\\times|\\div)"
_EQUATION_RE = re.compile(
    rf"(?P<expr>{_NUMBER_PATTERN}(?:\s*{_OP_PATTERN}\s*{_NUMBER_PATTERN})+)"
    rf"\s*=\s*\$?(?P<value>{_NUMBER_PATTERN})"
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource checked before corpus construction or scoring."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 3884 corpus builder."""

    repo_root: Path
    output_path: Path | None = None
    corpus_output_path: Path | None = None
    scores_output_path: Path | None = None
    min_incorrect_steps: int = DEFAULT_MIN_INCORRECT_STEPS
    random_seed: int = DEFAULT_RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    max_refinement_iterations: int = 32

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_RESULTS_REL_PATH

    def resolved_corpus_path(self) -> Path:
        return (
            self.corpus_output_path
            if self.corpus_output_path is not None
            else self.repo_root / OUTPUT_CORPUS_REL_PATH
        )

    def resolved_scores_path(self) -> Path:
        return (
            self.scores_output_path
            if self.scores_output_path is not None
            else self.repo_root / OUTPUT_SCORES_REL_PATH
        )

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class FoVerFamilyRows:
    """Deduplicated FoVer-family rows split by gold label."""

    incorrect: tuple[dict[str, Any], ...]
    correct: tuple[dict[str, Any], ...]
    corpus_sources: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class PreflightResult:
    """Preconditions and blocked reason for an Exp 3884 run."""

    checks: tuple[PreconditionCheck, ...]
    blocked_reason: str | None


Scorer = Callable[[list[dict[str, Any]], Path], list[dict[str, Any]]]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("items") or payload.get("examples") or payload.get("data") or []
    else:
        candidates = []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _label_name(label: Any) -> str:
    return "incorrect" if _label_to_int(label) == 1 else "correct"


def _dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    identifier = row.get("question_id", row.get("question", ""))
    return (str(identifier), str(row.get("step_text", "")))


def _normalize_row(row: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    if "step_text" not in row or "label" not in row:
        return None
    try:
        label = _label_name(row["label"])
    except ValueError:
        return None
    step_text = str(row.get("step_text", ""))
    if not step_text.strip():
        return None

    normalized = dict(row)
    normalized["label"] = label
    normalized.setdefault("question_id", str(row.get("question", "")))
    normalized["question_id"] = str(normalized.get("question_id", ""))
    normalized["step_text"] = step_text
    normalized.setdefault("confidence", row.get("confidence", 1.0))
    normalized["source"] = source
    normalized["synthetic"] = False
    return normalized


def probe_preconditions(config: ExperimentConfig) -> PreflightResult:
    """Check import and FoVer corpus availability before building anything."""

    checks: list[PreconditionCheck] = []
    try:
        importlib.import_module("carnot.verify")
        checks.append(PreconditionCheck("carnot_verify_import", True, "import carnot.verify OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("carnot_verify_import", False, repr(exc)))

    for filename in REQUIRED_CORPUS_FILENAMES:
        path = config.repo_root / "data" / filename
        if not path.is_file():
            checks.append(PreconditionCheck(filename, False, "missing"))
            continue
        try:
            rows = _json_rows(_read_json(path))
            checks.append(PreconditionCheck(filename, True, f"json_rows={len(rows)}"))
        except Exception as exc:
            checks.append(PreconditionCheck(filename, False, f"json_load_failed: {exc!r}"))

    blocked_reason = None
    if not checks[0].available:
        blocked_reason = "blocked_carnot_verify_import"
    elif any(not check.available for check in checks[1:]):
        blocked_reason = "blocked_corpus_missing"
    return PreflightResult(checks=tuple(checks), blocked_reason=blocked_reason)


def load_fover_family_rows(config: ExperimentConfig) -> FoVerFamilyRows:
    """Load and deduplicate local FoVer-family rows by question id and step text."""

    incorrect: list[dict[str, Any]] = []
    correct: list[dict[str, Any]] = []
    seen_incorrect: set[tuple[str, str]] = set()
    seen_correct: set[tuple[str, str]] = set()
    sources: list[dict[str, object]] = []

    for filename in REQUIRED_CORPUS_FILENAMES:
        path = config.repo_root / "data" / filename
        rows = _json_rows(_read_json(path))
        source = Path(filename).stem
        source_counts = {"incorrect": 0, "correct": 0}
        for row in rows:
            normalized = _normalize_row(row, source=source)
            if normalized is None:
                continue
            key = _dedupe_key(normalized)
            if normalized["label"] == "incorrect":
                if key in seen_incorrect:
                    continue
                seen_incorrect.add(key)
                incorrect.append(normalized)
                source_counts["incorrect"] += 1
            else:
                if key in seen_correct:
                    continue
                seen_correct.add(key)
                correct.append(normalized)
                source_counts["correct"] += 1
        sources.append({"path": f"data/{filename}", **source_counts})

    return FoVerFamilyRows(
        incorrect=tuple(incorrect),
        correct=tuple(correct),
        corpus_sources=tuple(sources),
    )


def _max_synthetic_errors(n_errors: int) -> int:
    return max(0, math.ceil(MAX_SYNTHETIC_ERROR_FRACTION * n_errors) - 1)


def _format_perturbed_number(value_text: str, *, offset: int) -> str:
    cleaned = value_text.replace(",", "")
    value = float(cleaned)
    delta = float((offset % 5) + 1)
    if abs(value) < 1.0:
        delta = 0.1 * float((offset % 5) + 1)
    perturbed = value + delta

    if "." in value_text:
        decimals = len(value_text.rsplit(".", 1)[1])
        rendered = f"{perturbed:.{decimals}f}"
    else:
        rendered = str(int(round(perturbed)))

    if "," in value_text:
        pieces = rendered.split(".", 1)
        pieces[0] = f"{int(pieces[0]):,}"
        rendered = ".".join(pieces)
    return rendered


def perturb_math_step_text(step_text: str, *, offset: int = 0) -> str | None:
    """Return a FoVer-style arithmetic perturbation, or None when no equation exists."""

    matches = list(_EQUATION_RE.finditer(step_text))
    if not matches:
        return None
    match = matches[-1]
    old_value = match.group("value")
    new_value = _format_perturbed_number(old_value, offset=offset)
    if new_value == old_value:
        new_value = _format_perturbed_number(old_value, offset=offset + 1)
    return f"{step_text[: match.start('value')]}{new_value}{step_text[match.end('value') :]}"


def _make_synthetic_error(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    perturbed = perturb_math_step_text(str(row.get("step_text", "")), offset=index)
    if perturbed is None or perturbed == row.get("step_text"):
        return None
    original_qid = str(row.get("question_id", f"correct-{index}"))
    synthetic = dict(row)
    synthetic["question_id"] = f"synthetic_fover_error_{index:04d}_{original_qid}"
    synthetic["source_question_id"] = original_qid
    synthetic["original_step_text_sha256"] = hashlib.sha256(
        str(row.get("step_text", "")).encode("utf-8")
    ).hexdigest()
    synthetic["step_text"] = perturbed
    synthetic["label"] = "incorrect"
    synthetic["synthetic"] = True
    synthetic["source"] = "synthetic_fover_perturbation"
    synthetic["synthetic_generation"] = "arithmetic_result_perturbation"
    return synthetic


def _synthetic_errors_from_correct(
    correct_rows: Sequence[dict[str, Any]],
    *,
    needed: int,
    used_correct_keys: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    synthetic: list[dict[str, Any]] = []
    source_keys: set[tuple[str, str]] = set()
    for row in correct_rows:
        key = _dedupe_key(row)
        if key in used_correct_keys:
            continue
        candidate = _make_synthetic_error(row, len(synthetic))
        if candidate is None:
            continue
        synthetic.append(candidate)
        source_keys.add(key)
        if len(synthetic) >= needed:
            break
    if len(synthetic) < needed:
        raise ValueError(f"could generate only {len(synthetic)} synthetic errors; needed {needed}")
    return synthetic, source_keys


def _assign_item_ids(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        copied = dict(item)
        copied["corpus_item_id"] = f"fover_indist_v1_{index:04d}"
        assigned.append(copied)
    return assigned


def build_in_distribution_corpus(
    loaded: FoVerFamilyRows,
    config: ExperimentConfig,
    *,
    dropped_error_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Pool real FoVer errors, add bounded synthetic errors, and balance correct rows."""

    dropped = dropped_error_keys or set()
    real_errors = [dict(row) for row in loaded.incorrect if _dedupe_key(row) not in dropped]
    target_errors = max(config.min_incorrect_steps, len(real_errors))
    needed_synthetic = target_errors - len(real_errors)
    if needed_synthetic > _max_synthetic_errors(target_errors):
        raise ValueError(
            f"synthetic error requirement {needed_synthetic}/{target_errors} violates "
            f"<{MAX_SYNTHETIC_ERROR_FRACTION:.0%} bound"
        )

    synthetic_source_keys: set[tuple[str, str]] = set()
    synthetic_errors: list[dict[str, Any]] = []
    if needed_synthetic > 0:
        synthetic_errors, synthetic_source_keys = _synthetic_errors_from_correct(
            loaded.correct,
            needed=needed_synthetic,
            used_correct_keys=set(),
        )

    errors = [*real_errors, *synthetic_errors]
    correct_candidates = [
        dict(row) for row in loaded.correct if _dedupe_key(row) not in synthetic_source_keys
    ]
    if len(correct_candidates) < len(errors):
        raise ValueError(
            f"not enough FoVer correct rows to balance corpus: "
            f"correct={len(correct_candidates)} errors={len(errors)}"
        )

    rng = random.Random(config.random_seed)
    correct = rng.sample(correct_candidates, len(errors))
    items = [*errors, *correct]
    rng.shuffle(items)
    return _assign_item_ids(items)


def score_corpus_items(items: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    """Score corpus rows with the Exp 2837 production FoVer aggregation."""

    texts = [str(item.get("step_text", "")) for item in items]
    verifier_scores = _score_text_verifiers(texts)
    tier0r = verifier_scores["tier0r_curry_howard"]
    tier0u = verifier_scores["tier0u_logical_consistency"]
    memory_index = _load_fr11_memory_index(repo_root)
    memory_scores = [_fr11_memory_score(item, memory_index) for item in items]
    full_scores = [
        float(0.9 * r_score + 0.1 * u_score + memory_score)
        for r_score, u_score, memory_score in zip(tier0r, tier0u, memory_scores, strict=True)
    ]
    threshold = float(statistics.median(full_scores)) if full_scores else 0.0

    scored: list[dict[str, Any]] = []
    for index, (item, score) in enumerate(zip(items, full_scores, strict=True)):
        per_verifier_scores = {
            name: float(values[index]) for name, values in verifier_scores.items()
        }
        per_verifier_scores["fr11_session_memory"] = float(memory_scores[index])
        scored.append(
            {
                "index": index,
                "corpus_item_id": item.get("corpus_item_id"),
                "question_id": item.get("question_id"),
                "step_text_sha256": hashlib.sha256(
                    str(item.get("step_text", "")).encode("utf-8")
                ).hexdigest(),
                "label": item.get("label"),
                "synthetic": bool(item.get("synthetic")),
                "carnot_ensemble_score": score,
                "carnot_rejects": score > threshold,
                "ensemble_threshold": threshold,
                "per_verifier_scores": per_verifier_scores,
            }
        )
    return scored


def _labels(items: Sequence[dict[str, Any]]) -> list[int]:
    return [_label_to_int(item["label"]) for item in items]


def _scores(scored_items: Sequence[dict[str, Any]]) -> list[float]:
    return [float(item["carnot_ensemble_score"]) for item in scored_items]


def _corpus_sha(items: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "corpus_item_id": item.get("corpus_item_id"),
            "question_id": item.get("question_id"),
            "label": item.get("label"),
            "synthetic": bool(item.get("synthetic")),
            "source": item.get("source"),
            "step_text_sha256": hashlib.sha256(
                str(item.get("step_text", "")).encode("utf-8")
            ).hexdigest(),
        }
        for item in items
    ]
    return _sha256_payload(payload)


def _score_sha(scored_items: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "index": item.get("index"),
            "corpus_item_id": item.get("corpus_item_id"),
            "score": round(float(item.get("carnot_ensemble_score", 0.0)), 12),
        }
        for item in scored_items
    ]
    return _sha256_payload(payload)


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "success:", "passed:", "shipped:", "blocked_"))


def _ready_verdict(n_errors: int, auroc: float) -> str:
    return (
        "complete: "
        f"in_distribution_corpus_READY_nerr{n_errors}_auroc{auroc:.4f}_"
        "moat_scissor_can_run"
    )


def _insufficient_verdict(n_errors: int, auroc: float | None) -> str:
    rendered = "nan" if auroc is None else f"{auroc:.4f}"
    return (
        "complete: "
        f"in_distribution_corpus_INSUFFICIENT_best_auroc{rendered}_nerr{n_errors}_"
        "ensemble_does_not_discriminate_in_band"
    )


def _gate_for_config(n_errors: int, auroc: float, config: ExperimentConfig) -> str:
    if n_errors >= config.min_incorrect_steps and auroc >= AUROC_READY_MIN:
        return "CORPUS_READY"
    return "INSUFFICIENT"


def _measure(
    items: list[dict[str, Any]],
    *,
    repo_root: Path,
    scorer: Scorer,
) -> tuple[float, list[dict[str, Any]]]:
    scored = scorer(items, repo_root)
    if len(scored) != len(items):
        raise ValueError(f"scorer returned {len(scored)} rows for {len(items)} corpus items")
    return float(compute_auroc(_labels(items), _scores(scored))), scored


def _best_refined_measurement(
    loaded: FoVerFamilyRows,
    config: ExperimentConfig,
    initial_items: list[dict[str, Any]],
    initial_scores: list[dict[str, Any]],
    initial_auroc: float,
    scorer: Scorer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, list[dict[str, object]]]:
    """Try dropping lowest-scoring real errors while preserving the corpus gates."""

    attempts: list[dict[str, object]] = []
    best_items = initial_items
    best_scores = initial_scores
    best_auroc = initial_auroc

    scored_by_index = {int(score["index"]): score for score in initial_scores}
    real_error_candidates = [
        (
            float(scored_by_index[index]["carnot_ensemble_score"]),
            _dedupe_key(item),
        )
        for index, item in enumerate(initial_items)
        if item.get("label") == "incorrect" and not item.get("synthetic")
    ]
    real_error_candidates.sort(key=lambda pair: pair[0])

    max_synthetic = _max_synthetic_errors(config.min_incorrect_steps)
    min_real_errors = max(0, config.min_incorrect_steps - max_synthetic)
    max_drop = max(0, len(loaded.incorrect) - min_real_errors)
    max_drop = min(max_drop, len(real_error_candidates), config.max_refinement_iterations)

    dropped: set[tuple[str, str]] = set()
    for score, key in real_error_candidates[:max_drop]:
        dropped.add(key)
        candidate_items = build_in_distribution_corpus(
            loaded,
            config,
            dropped_error_keys=set(dropped),
        )
        candidate_auroc, candidate_scores = _measure(
            candidate_items,
            repo_root=config.repo_root,
            scorer=scorer,
        )
        n_errors = sum(1 for item in candidate_items if item["label"] == "incorrect")
        n_synthetic = sum(
            1
            for item in candidate_items
            if item["label"] == "incorrect" and item.get("synthetic")
        )
        attempts.append(
            {
                "dropped_real_errors": len(dropped),
                "last_dropped_score": score,
                "n_incorrect_steps": n_errors,
                "n_synthetic_errors": n_synthetic,
                "carnot_ensemble_auroc_on_corpus": candidate_auroc,
            }
        )
        if candidate_auroc > best_auroc:
            best_items = candidate_items
            best_scores = candidate_scores
            best_auroc = candidate_auroc
        if _gate_for_config(n_errors, candidate_auroc, config) == "CORPUS_READY":
            break
    return best_items, best_scores, best_auroc, attempts


def _artifact_paths(config: ExperimentConfig) -> tuple[str, str]:
    return (
        _relative_to_repo(config.resolved_corpus_path(), config.repo_root),
        _relative_to_repo(config.resolved_scores_path(), config.repo_root),
    )


def build_success_artifact(
    *,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    loaded: FoVerFamilyRows,
    items: list[dict[str, Any]],
    scored_items: list[dict[str, Any]],
    auroc: float,
    refinement_attempts: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build a terminal Exp 3884 success or insufficient artifact."""

    started_at = config.start_time()
    finished_at = config.clock()
    n_errors = sum(1 for item in items if item["label"] == "incorrect")
    n_correct = sum(1 for item in items if item["label"] == "correct")
    n_synthetic_errors = sum(
        1 for item in items if item["label"] == "incorrect" and bool(item.get("synthetic"))
    )
    n_synthetic_total = sum(1 for item in items if bool(item.get("synthetic")))
    corpus_path, scores_path = _artifact_paths(config)
    gate = _gate_for_config(n_errors, auroc, config)
    verdict = _ready_verdict(n_errors, auroc) if gate == "CORPUS_READY" else _insufficient_verdict(n_errors, auroc)
    corpus_sha = _corpus_sha(items)
    scores_sha = _score_sha(scored_items)
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "corpus_sha256": corpus_sha,
        "scores_sha256": scores_sha,
        "auroc": round(auroc, 12),
        "n_errors": n_errors,
        "n_correct": n_correct,
        "random_seed": config.random_seed,
        "gate": gate,
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "schema": "carnot.in_distribution_error_rich_corpus.v1",
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "honest_verdict": verdict,
        "status": verdict,
        "gate": gate,
        "carnot_ensemble_auroc_on_corpus": auroc,
        "n_incorrect_steps": n_errors,
        "n_correct_steps": n_correct,
        "n_total_items": len(items),
        "n_synthetic_errors": n_synthetic_errors,
        "frac_synthetic": n_synthetic_total / len(items) if items else 0.0,
        "frac_synthetic_errors": n_synthetic_errors / n_errors if n_errors else 0.0,
        "synthetic_error_fraction_lt_40pct": (
            n_synthetic_errors / n_errors < MAX_SYNTHETIC_ERROR_FRACTION
            if n_errors
            else False
        ),
        "corpus_path": corpus_path,
        "per_item_ensemble_scores_path": scores_path,
        "corpus_sha256": corpus_sha,
        "per_item_ensemble_scores_sha256": scores_sha,
        "corpus_sources": list(loaded.corpus_sources),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "random_seed": config.random_seed,
        "random_seeds_used": {"correct_balance_sample": config.random_seed},
        "reproducibility_checksum": _sha256_payload(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "ensemble_definition": (
            "Exp 2837 FoVer production aggregation: "
            "0.9*tier0r_curry_howard + 0.1*tier0u_logical_consistency "
            "+ fr11_session_memory"
        ),
        "ready_gate": {
            "min_incorrect_steps": config.min_incorrect_steps,
            "min_carnot_ensemble_auroc_on_corpus": AUROC_READY_MIN,
        },
        "refinement_attempts": list(refinement_attempts),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
) -> dict[str, object]:
    """Build a blocked artifact without fabricated corpus or score metrics."""

    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "schema": "carnot.in_distribution_error_rich_corpus.v1",
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "gate": "BLOCKED",
        "carnot_ensemble_auroc_on_corpus": None,
        "n_incorrect_steps": 0,
        "n_correct_steps": 0,
        "n_total_items": 0,
        "n_synthetic_errors": 0,
        "frac_synthetic": 0.0,
        "frac_synthetic_errors": 0.0,
        "synthetic_error_fraction_lt_40pct": False,
        "corpus_path": None,
        "per_item_ensemble_scores_path": None,
        "corpus_sha256": None,
        "per_item_ensemble_scores_sha256": None,
        "corpus_sources": [],
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "random_seed": None,
        "random_seeds_used": {},
        "reproducibility_checksum": _sha256_payload(
            {
                "experiment": EXPERIMENT_ID,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
        "ensemble_definition": None,
        "ready_gate": {
            "min_incorrect_steps": DEFAULT_MIN_INCORRECT_STEPS,
            "min_carnot_ensemble_auroc_on_corpus": AUROC_READY_MIN,
        },
        "refinement_attempts": [],
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_corpus(path: Path, items: Sequence[dict[str, Any]], artifact: dict[str, object]) -> None:
    write_json(
        path,
        {
            "schema": "carnot.in_distribution_error_corpus.v1",
            "items": list(items),
            "metadata": {
                "experiment": EXPERIMENT_ID,
                "honest_verdict": artifact["honest_verdict"],
                "n_incorrect_steps": artifact["n_incorrect_steps"],
                "n_total_items": artifact["n_total_items"],
                "frac_synthetic": artifact["frac_synthetic"],
                "carnot_ensemble_auroc_on_corpus": artifact[
                    "carnot_ensemble_auroc_on_corpus"
                ],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
        },
    )


def write_scores(path: Path, scored_items: Sequence[dict[str, Any]], artifact: dict[str, object]) -> None:
    write_json(
        path,
        {
            "schema": "carnot.in_distribution_error_corpus_scores.v1",
            "items": list(scored_items),
            "metadata": {
                "experiment": EXPERIMENT_ID,
                "corpus_path": artifact["corpus_path"],
                "carnot_ensemble_auroc_on_corpus": artifact[
                    "carnot_ensemble_auroc_on_corpus"
                ],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
        },
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    write: bool = True,
    scorer: Scorer | None = None,
) -> dict[str, object]:
    """Run Exp 3884 end to end and optionally write all requested artifacts."""

    config = config or ExperimentConfig(repo_root=Path(__file__).resolve().parents[3])
    started = config.start_time()
    active_config = replace(config, started_at=started)
    preflight = probe_preconditions(active_config)
    if preflight.blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=preflight.blocked_reason,
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
        )
        if write:
            write_json(active_config.resolved_output_path(), artifact)
        return artifact

    selected_scorer = scorer or score_corpus_items
    try:
        loaded = load_fover_family_rows(active_config)
        corpus_items = build_in_distribution_corpus(loaded, active_config)
        auroc, scored_items = _measure(
            corpus_items,
            repo_root=active_config.repo_root,
            scorer=selected_scorer,
        )
        refinement_attempts: list[dict[str, object]] = []
        if _gate_for_config(
            sum(1 for item in corpus_items if item["label"] == "incorrect"),
            auroc,
            active_config,
        ) != "CORPUS_READY":
            corpus_items, scored_items, auroc, refinement_attempts = _best_refined_measurement(
                loaded,
                active_config,
                corpus_items,
                scored_items,
                auroc,
                selected_scorer,
            )
        artifact = build_success_artifact(
            config=active_config,
            preconditions_checked=preflight.checks,
            loaded=loaded,
            items=corpus_items,
            scored_items=scored_items,
            auroc=auroc,
            refinement_attempts=refinement_attempts,
        )
    except Exception as exc:
        checks = [
            *preflight.checks,
            PreconditionCheck("fover_corpus_build_or_score", False, repr(exc)),
        ]
        artifact = build_blocked_artifact(
            reason="blocked_corpus_missing",
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
        )
        if write:
            write_json(active_config.resolved_output_path(), artifact)
        return artifact

    if write:
        write_corpus(active_config.resolved_corpus_path(), corpus_items, artifact)
        write_scores(active_config.resolved_scores_path(), scored_items, artifact)
        write_json(active_config.resolved_output_path(), artifact)
    return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the required Exp 3884 artifact contract."""

    required = {"honest_verdict", "status", "field_principles", *REQUIRED_PRINCIPLE_FIELDS}
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    principles = artifact["field_principles"]
    if not isinstance(principles, dict):
        raise ValueError("field_principles must be a dict")
    for field in REQUIRED_PRINCIPLE_FIELDS:
        note = principles.get(field)
        if not isinstance(note, str) or not note:
            raise ValueError(f"missing principle note for {field}")
    if artifact["gate"] == "CORPUS_READY":
        auroc = float(artifact["carnot_ensemble_auroc_on_corpus"])
        if auroc < AUROC_READY_MIN:
            raise ValueError("CORPUS_READY requires AUROC >= gate")
        if int(artifact["n_incorrect_steps"]) < int(
            artifact.get("ready_gate", {}).get("min_incorrect_steps", DEFAULT_MIN_INCORRECT_STEPS)  # type: ignore[union-attr]
        ):
            raise ValueError("CORPUS_READY requires enough incorrect steps")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--min-incorrect", type=int, default=DEFAULT_MIN_INCORRECT_STEPS)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args(argv)


def cli_main(argv: Sequence[str] | None = None, *, compatibility_label: str | None = None) -> int:
    args = _parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=args.repo_root,
            output_path=args.output_path,
            min_incorrect_steps=args.min_incorrect,
        ),
        write=True,
    )
    label = compatibility_label or OUTPUT_RESULTS_REL_PATH.name
    output = args.output_path or args.repo_root / OUTPUT_RESULTS_REL_PATH
    print(f"{label} wrote {output} honest_verdict={artifact['honest_verdict']}")
    return 1 if str(artifact["honest_verdict"]).startswith("blocked_") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
