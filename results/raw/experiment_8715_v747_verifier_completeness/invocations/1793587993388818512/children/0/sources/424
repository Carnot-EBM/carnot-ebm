"""Exp 2841 HaluEval + FEVER factuality readiness pilot.

This runner is deliberately scoped as a readiness pilot, not a headline
factuality benchmark. It gates on Exp 2836's SOTA GGUF runtime preflight, loads
small labeled candidate-answer rows from HaluEval and FEVER when available, and
scores those rows with Carnot's existing text verifier ensemble. The reported
AUROC values are exploratory sample diagnostics for deciding whether either
corpus deserves a future N>=500 milestone.

Spec: REQ-VERIFY-2841,
      SCENARIO-VERIFY-2841-BLOCKED,
      SCENARIO-VERIFY-2841-PILOT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2841_halueval_fever_pilot.json"
EXP2836_FILENAME = "experiment_2836_sota_runtime_preflight.json"
RUN_DATE = "20260522"
DEFAULT_SAMPLE_PER_DATASET = 25
DEFAULT_BOOTSTRAP_REPS = 500
REPO_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_SOTA_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY = ("Qwen3.5-0.8B", "gemma-4-E4B-it")

FIELD_PRINCIPLES = {
    "honest_verdict": 'MUST start with "complete:" / "success:" or "blocked_".',
    "pilot_only": "Prevents accidental headline use.",
    "datasets_loaded": "Readiness depends on real datasets.",
    "n_examples": "Pilot sample size transparency.",
    "pilot_auroc_by_dataset": "Exploratory only; not headline.",
    "recommendation": "States whether to scale HaluEval/FEVER later.",
    "model_specs": "Mandated SOTA GGUF recorded.",
    "preconditions_checked": "Explains blocks honestly.",
    "duration_s": "Real compute wall-time; no sleep padding.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before any pilot scoring begins."""

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
class PilotExample:
    """One labeled factuality candidate for verifier-energy scoring."""

    dataset: str
    example_id: str
    prompt: str
    candidate: str
    label: int
    source: str
    reference: str | None = None

    @property
    def score_text(self) -> str:
        if self.reference:
            return f"{self.prompt}\nReference: {self.reference}\nCandidate: {self.candidate}"
        return f"{self.prompt}\nCandidate: {self.candidate}"

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "label": self.label,
            "source": self.source,
            "candidate_len": len(self.candidate),
            "has_reference": self.reference is not None,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 2841."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    exp2836_path: Path | None = None
    sample_per_dataset: int = DEFAULT_SAMPLE_PER_DATASET
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
    random_seed: int = 42
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    probe_timeout_s: int = 60

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def preflight_path(self) -> Path:
        if self.exp2836_path is not None:
            return self.exp2836_path
        return self.output_dir() / EXP2836_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


ScorePayload = dict[str, object]
Scorer = Callable[[PilotExample], ScorePayload]
DatasetLoader = Callable[[ExperimentConfig], dict[str, list[PilotExample]]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def load_exp2836_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_model_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model_path", "path", "resolved_gguf", "resolved_path"} and item:
                paths.append(str(item))
            else:
                paths.extend(_extract_model_paths(item))
    elif isinstance(value, list | tuple):
        for item in value:
            paths.extend(_extract_model_paths(item))
    return paths


def model_specs_from_exp2836(preflight: dict[str, Any]) -> dict[str, object]:
    """Normalize Exp 2836 model/runtime evidence for the pilot artifact."""

    cached_pair = dict(preflight.get("cached_sota_pair_result") or {})
    cached_pair_paths = _extract_model_paths(cached_pair.get("result"))
    smoke_results = [
        dict(row)
        for row in preflight.get("smoke_load_results", [])
        if row.get("load_success") and row.get("headline_usable") and row.get("model_path")
    ]
    cached_models = [
        dict(row)
        for row in preflight.get("sota_models_cached", [])
        if row.get("hf_id") in PRIMARY_SOTA_MODEL_IDS and row.get("path")
    ]
    smoke_paths = [str(row["model_path"]) for row in smoke_results]
    cached_model_paths = [str(row["path"]) for row in cached_models]
    selected_path = (cached_pair_paths or smoke_paths or cached_model_paths or [None])[0]
    selected_hf_id = None
    for row in [*smoke_results, *cached_models]:
        if row.get("model_path") == selected_path or row.get("path") == selected_path:
            selected_hf_id = str(row.get("hf_id"))
            break
    raw_specs = dict(preflight.get("model_specs") or {})
    return {
        "headline_required_any_of": list(raw_specs.get("primary") or PRIMARY_SOTA_MODEL_IDS),
        "legacy_cpu_smoke_only": list(
            raw_specs.get("legacy_cpu_smoke_only") or LEGACY_CPU_SMOKE_ONLY
        ),
        "sota_runtime_ready": bool(preflight.get("sota_runtime_ready")),
        "selected_python": preflight.get("selected_python"),
        "cached_sota_pair_result": cached_pair,
        "cached_sota_pair_model_paths": cached_pair_paths,
        "smoke_model_paths": smoke_paths,
        "sota_models_cached": cached_models,
        "selected_model_path": selected_path,
        "selected_model_hf_id": selected_hf_id,
        "scorer_or_generator_model_paths_used": [],
    }


def _run_json_probe(
    *,
    selected_python: str,
    repo_root: Path,
    script: str,
    resource: str,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 60,
) -> PreconditionCheck:
    if not selected_python:
        return PreconditionCheck(resource, False, "selected_python missing")
    env = os.environ.copy()
    python_dir = str(repo_root / "python")
    env["PYTHONPATH"] = python_dir + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        proc = command_runner(
            [selected_python, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host dependent.
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        return PreconditionCheck(
            resource,
            False,
            (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip(),
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return PreconditionCheck(resource, False, f"invalid JSON probe output: {proc.stdout[:200]}")
    return PreconditionCheck(
        resource,
        bool(payload.get("available")),
        str(payload.get("detail", "")),
    )


def _dataset_probe_check(
    selected_python: str,
    repo_root: Path,
    *,
    resource: str,
    script: str,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 60,
) -> PreconditionCheck:
    return _run_json_probe(
        selected_python=selected_python,
        repo_root=repo_root,
        script=script,
        resource=resource,
        command_runner=command_runner,
        timeout_s=timeout_s,
    )


def examples_from_halueval_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[PilotExample]:
    """Convert HaluEval QA-style right/hallucinated answer pairs to labels."""

    examples: list[PilotExample] = []
    for idx, row in enumerate(rows):
        knowledge = str(row.get("knowledge") or row.get("document") or "").strip()
        question = str(row.get("question") or row.get("user_query") or "").strip()
        prompt = f"Context: {knowledge}\nQuestion: {question}".strip()
        right = row.get("right_answer") or row.get("right_response") or row.get("right_summary")
        hallucinated = (
            row.get("hallucinated_answer")
            or row.get("hallucinated_response")
            or row.get("hallucinated_summary")
        )
        if right:
            examples.append(
                PilotExample(
                    dataset="HaluEval",
                    example_id=f"halueval-{idx}-right",
                    prompt=prompt,
                    candidate=str(right),
                    label=0,
                    source="pminervini/HaluEval:qa:data",
                    reference=str(right),
                )
            )
        if hallucinated:
            examples.append(
                PilotExample(
                    dataset="HaluEval",
                    example_id=f"halueval-{idx}-hallucinated",
                    prompt=prompt,
                    candidate=str(hallucinated),
                    label=1,
                    source="pminervini/HaluEval:qa:data",
                    reference=str(right) if right else None,
                )
            )
        if len(examples) >= limit:
            break
    return examples[:limit]


def examples_from_fever_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[PilotExample]:
    """Convert FEVER claim/evidence rows to hallucination-positive labels."""

    examples: list[PilotExample] = []
    for idx, row in enumerate(rows):
        label = _fever_label_to_int(row.get("label") or row.get("fever_gold_label"))
        if label is None:
            continue
        claim = str(row.get("claim") or row.get("premise") or "").strip()
        evidence = _stringify_evidence(row.get("evidence") or row.get("hypothesis") or "")
        if not claim:
            continue
        examples.append(
            PilotExample(
                dataset="FEVER",
                example_id=f"fever-{row.get('id') or row.get('cid') or idx}",
                prompt=f"Evidence: {evidence}",
                candidate=claim,
                label=label,
                source="maxzoech/fever:train",
                reference=evidence or None,
            )
        )
        if len(examples) >= limit:
            break
    return examples


def _fever_label_to_int(value: Any) -> int | None:
    label = str(value).strip().upper().replace("_", " ")
    if label == "SUPPORTS":
        return 0
    if label in {"REFUTES", "NOT ENOUGH INFO", "NEI"}:
        return 1
    if label in {"0", "1"}:
        return int(label)
    return None


def _stringify_evidence(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, list | tuple):
                parts.append(" ".join(str(part) for part in item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def load_pilot_datasets(
    config: ExperimentConfig,
) -> dict[str, list[PilotExample]]:  # pragma: no cover - network/cache dependent.
    """Load up to 25 HaluEval and 25 FEVER rows from HuggingFace datasets."""

    from datasets import load_dataset

    loaded: dict[str, list[PilotExample]] = {}
    errors: dict[str, str] = {}
    try:
        halueval = load_dataset("pminervini/HaluEval", "qa", split="data")
        rows = [dict(halueval[idx]) for idx in range(min(len(halueval), config.sample_per_dataset))]
        examples = examples_from_halueval_rows(rows, limit=config.sample_per_dataset)
        if examples:
            loaded["HaluEval"] = examples
    except Exception as exc:
        errors["HaluEval"] = f"{type(exc).__name__}: {exc}"

    try:
        fever = load_dataset("maxzoech/fever", split="train")
        rows = _balanced_fever_rows(fever, limit=config.sample_per_dataset)
        examples = examples_from_fever_rows(rows, limit=config.sample_per_dataset)
        if examples:
            loaded["FEVER"] = examples
    except Exception as exc:
        errors["FEVER"] = f"{type(exc).__name__}: {exc}"

    load_pilot_datasets.last_errors = errors  # type: ignore[attr-defined]
    return loaded


def _balanced_fever_rows(
    dataset: Any, *, limit: int
) -> list[dict[str, Any]]:  # pragma: no cover - exercised by live dataset only.
    need_pos = limit // 2
    need_neg = limit - need_pos
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for row in dataset:
        label = _fever_label_to_int(row.get("label") if isinstance(row, dict) else None)
        if label == 0 and len(negatives) < need_neg:
            negatives.append(dict(row))
        elif label == 1 and len(positives) < need_pos:
            positives.append(dict(row))
        if len(positives) >= need_pos and len(negatives) >= need_neg:
            break
    rows: list[dict[str, Any]] = []
    for pair in zip(negatives, positives, strict=False):
        rows.extend(pair)
    rows.extend(negatives[len(positives) :])
    rows.extend(positives[len(negatives) :])
    return rows[:limit]


def _base_precondition_checks(
    config: ExperimentConfig,
    model_specs: dict[str, object],
) -> list[PreconditionCheck]:
    selected_python = str(model_specs.get("selected_python") or "")
    selected_model_path = str(model_specs.get("selected_model_path") or "")
    model_path_ok = bool(selected_model_path and Path(selected_model_path).is_file())
    return [
        PreconditionCheck(
            "exp2836_artifact",
            config.preflight_path().is_file(),
            str(config.preflight_path()) if config.preflight_path().is_file() else "missing",
        ),
        PreconditionCheck(
            "exp2836_sota_runtime_ready",
            bool(model_specs.get("sota_runtime_ready")),
            f"sota_runtime_ready={model_specs.get('sota_runtime_ready')}",
        ),
        PreconditionCheck(
            "exp2836_selected_python",
            bool(selected_python),
            selected_python if selected_python else "missing",
        ),
        PreconditionCheck(
            "mandated_sota_model_path",
            model_path_ok,
            selected_model_path if selected_model_path else "missing",
        ),
    ]


def _dataset_precondition_checks(
    loaded: dict[str, list[PilotExample]],
    errors: dict[str, str] | None = None,
) -> list[PreconditionCheck]:
    errors = errors or {}
    checks: list[PreconditionCheck] = []
    for dataset in ("HaluEval", "FEVER"):
        count = len(loaded.get(dataset, []))
        detail = f"loaded_examples={count}" if count else errors.get(dataset, "not loaded")
        checks.append(PreconditionCheck(f"{dataset.lower()}_dataset", count > 0, detail))
    total = sum(len(examples) for examples in loaded.values())
    checks.append(
        PreconditionCheck(
            "labeled_pilot_datasets",
            total > 0,
            f"datasets={sorted(loaded)}; n_examples={total}",
        )
    )
    return checks


def _blocked_verdict(checks: Sequence[PreconditionCheck]) -> str | None:
    verdict_by_resource = {
        "exp2836_artifact": "blocked_exp2836_missing",
        "exp2836_sota_runtime_ready": "blocked_sota_runtime_not_ready",
        "exp2836_selected_python": "blocked_selected_python_missing",
        "mandated_sota_model_path": "blocked_model_path",
        "labeled_pilot_datasets": "blocked_labeled_pilot_datasets",
    }
    for check in checks:
        if not check.available and check.resource in verdict_by_resource:
            return verdict_by_resource[check.resource]
    for check in checks:
        if not check.available and check.resource not in {"halueval_dataset", "fever_dataset"}:
            return f"blocked_{check.resource}"
    return None


def default_score_example(example: PilotExample) -> ScorePayload:
    """Score a candidate with the existing text verifier ensemble."""

    from carnot.verify.tier0r_curry_howard import Tier0rVerifier
    from carnot.verify.tier0s_halluguard import Tier0sVerifier
    from carnot.verify.tier0u_logical_consistency import Tier0uVerifier

    score_text = example.score_text
    tier0r = float(Tier0rVerifier().score(score_text))
    tier0u = float(Tier0uVerifier().score(score_text))
    tier0s = min(1.0, float(Tier0sVerifier().halluguard_ntk_score(score_text)) / 100.0)
    per_verifier = {
        "tier0r_curry_howard": tier0r,
        "tier0u_logical_consistency": tier0u,
        "tier0s_arithmetic_gap": tier0s,
    }
    return {
        "ensemble_energy": sum(per_verifier.values()) / len(per_verifier),
        "per_verifier_energy": per_verifier,
    }


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC where label 1 is the hallucinated/unsupported class."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_pos = sum(1 for label in labels if int(label) == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    negatives = [
        float(score) for label, score in zip(labels, scores, strict=True) if int(label) == 0
    ]
    for label, score in zip(labels, scores, strict=True):
        if int(label) != 1:
            continue
        positive = float(score)
        wins += sum(1.0 for negative in negatives if positive > negative)
        wins += 0.5 * sum(1.0 for negative in negatives if positive == negative)
    return wins / (n_pos * n_neg)


def bootstrap_auroc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap confidence interval."""

    base = compute_auroc(labels, scores)
    if reps <= 0:
        return (base, base)
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    samples: list[float] = []
    for _ in range(reps):
        chosen = [rng.choice(indices) for _idx in indices]
        sampled_labels = [labels[idx] for idx in chosen]
        if len(set(sampled_labels)) < 2:
            continue
        sampled_scores = [scores[idx] for idx in chosen]
        samples.append(compute_auroc(sampled_labels, sampled_scores))
    if not samples:
        return (base, base)
    samples.sort()
    lo_idx = max(0, int(0.025 * (len(samples) - 1)))
    hi_idx = min(len(samples) - 1, int(0.975 * (len(samples) - 1)))
    return (samples[lo_idx], samples[hi_idx])


def evaluate_dataset(
    examples: Sequence[PilotExample],
    *,
    scorer: Scorer,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, object]:
    labels: list[int] = []
    scores: list[float] = []
    per_verifier_scores: dict[str, list[float]] = {}
    failures = 0
    for example in examples:
        try:
            payload = scorer(example)
            score = float(payload["ensemble_energy"])
            per_verifier = dict(payload.get("per_verifier_energy") or {})
        except (KeyError, TypeError, ValueError):
            failures += 1
            continue
        if not math.isfinite(score):
            failures += 1
            continue
        labels.append(int(example.label))
        scores.append(score)
        for name, value in per_verifier.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                per_verifier_scores.setdefault(str(name), []).append(numeric)

    label_counts = {str(label): count for label, count in sorted(Counter(labels).items())}
    all_finite = bool(scores) and all(math.isfinite(score) for score in scores)
    score_min = min(scores) if scores else None
    score_max = max(scores) if scores else None
    score_mean = sum(scores) / len(scores) if scores else None
    try:
        auroc = compute_auroc(labels, scores)
        ci_low, ci_high = bootstrap_auroc_ci(labels, scores, reps=bootstrap_reps, seed=seed)
    except ValueError:
        auroc = None
        ci_low = None
        ci_high = None
    ready = bool(len(set(labels)) == 2 and all_finite and auroc is not None)
    return {
        "n_examples": len(examples),
        "n_scored": len(scores),
        "score_failures": failures,
        "label_counts": label_counts,
        "auroc": auroc,
        "auroc_ci95": [ci_low, ci_high],
        "ci_method": "bootstrap_percentile",
        "energy_stability": {
            "all_finite": all_finite,
            "score_min": score_min,
            "score_max": score_max,
            "score_mean": score_mean,
            "unique_score_count": len(set(scores)),
        },
        "per_verifier_energy_summary": {
            name: {
                "n": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in sorted(per_verifier_scores.items())
            if values
        },
        "ready_for_full_benchmark": ready,
    }


def _recommendation(dataset_metrics: dict[str, dict[str, object]]) -> str:
    ready = sorted(
        dataset
        for dataset, metrics in dataset_metrics.items()
        if metrics["ready_for_full_benchmark"]
    )
    blocked = sorted(set(dataset_metrics) - set(ready))
    if ready and not blocked:
        return (
            "Scale " + ", ".join(ready) + " to a future N>=500 milestone, keeping "
            "pilot_only=true results out of headline tables."
        )
    if ready:
        return (
            "Scale "
            + ", ".join(ready)
            + " to a future N>=500 milestone; keep "
            + ", ".join(blocked)
            + " in readiness until labels and verifier energies are stable."
        )
    return (
        "Do not scale HaluEval or FEVER yet; this pilot did not establish stable "
        "two-class labels and finite verifier energies for an N>=500 milestone."
    )


def _base_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    loaded: dict[str, list[PilotExample]],
) -> dict[str, object]:
    return {
        "artifact": "experiment_2841_halueval_fever_pilot",
        "schema": "carnot.halueval_fever_pilot",
        "run_date": RUN_DATE,
        "pilot_only": True,
        "datasets_loaded": sorted(loaded),
        "n_examples": sum(len(examples) for examples in loaded.values()),
        "sample_per_dataset": config.sample_per_dataset,
        "random_seed": config.random_seed,
        "model_specs": model_specs,
        "preconditions_checked": [check.as_dict() for check in checks],
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "candidate_generation": {
            "mode": "loaded_dataset_candidate_answers",
            "mandated_sota_model_hf_id": model_specs.get("selected_model_hf_id"),
            "mandated_sota_model_path": model_specs.get("selected_model_path"),
            "loaded_candidate_count": sum(len(examples) for examples in loaded.values()),
            "fresh_generation_count": 0,
            "caveat": (
                "This readiness pilot uses corpus-provided candidate answers/claims. "
                "Exp2836 supplies the mandated local GGUF runtime provenance; a future "
                "full generator benchmark should run fresh autoregressive candidates."
            ),
        },
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    loaded: dict[str, list[PilotExample]],
) -> dict[str, object]:
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        checks=checks,
        model_specs=model_specs,
        loaded={},
    )
    artifact.update(
        {
            "honest_verdict": _blocked_verdict(checks) or "blocked_unknown_resource",
            "datasets_loaded": [],
            "n_examples": 0,
            "blocked_resources": [check.resource for check in checks if not check.available],
            "pilot_auroc_by_dataset": {},
            "recommendation": (
                "Do not scale HaluEval or FEVER yet; required pilot resources were missing."
            ),
            "dataset_sample_manifest": [],
            "methodology_note": (
                "Blocked before pilot AUROC scoring. No dataset metric, confidence interval, "
                "candidate generation stability claim, or verifier-energy result was inferred."
            ),
        }
    )
    artifact["candidate_generation"]["loaded_candidate_count"] = 0
    return artifact


def _success_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    loaded: dict[str, list[PilotExample]],
    scorer: Scorer,
) -> dict[str, object]:
    metrics = {
        dataset: evaluate_dataset(
            examples,
            scorer=scorer,
            bootstrap_reps=config.bootstrap_reps,
            seed=config.random_seed,
        )
        for dataset, examples in sorted(loaded.items())
    }
    selected_model_path = str(model_specs.get("selected_model_path") or "")
    model_specs = {
        **model_specs,
        "scorer_or_generator_model_paths_used": [selected_model_path]
        if selected_model_path
        else [],
    }
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        checks=checks,
        model_specs=model_specs,
        loaded=loaded,
    )
    artifact.update(
        {
            "honest_verdict": (
                "complete: HaluEval/FEVER readiness pilot measured with dataset labels "
                "and mandated SOTA GGUF provenance"
            ),
            "pilot_auroc_by_dataset": metrics,
            "recommendation": _recommendation(metrics),
            "dataset_sample_manifest": [
                example.as_dict() for dataset in sorted(loaded) for example in loaded[dataset]
            ],
            "methodology_note": (
                "AUROC is pilot/readiness only. Label orientation is hallucinated, "
                "refuted, or unsupported = 1 and grounded/supports = 0. The pilot "
                "checks label balance, finite verifier energies, and corpus loading "
                "before recommending any future N>=500 benchmark."
            ),
        }
    )
    return artifact


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    dataset_loader: DatasetLoader = load_pilot_datasets,
    scorer: Scorer = default_score_example,
    write: bool = True,
) -> dict[str, object]:
    """Run Exp 2841 and write either a blocked or complete pilot artifact."""

    config = config or ExperimentConfig()
    start = config.start_time()
    preflight = load_exp2836_preflight(config.preflight_path())
    model_specs = model_specs_from_exp2836(preflight)
    checks = _base_precondition_checks(config, model_specs)
    loaded: dict[str, list[PilotExample]] = {}
    if _blocked_verdict(checks) is None:
        try:
            loaded = dataset_loader(config)
        except Exception as exc:
            errors = {
                "HaluEval": f"{type(exc).__name__}: {exc}",
                "FEVER": f"{type(exc).__name__}: {exc}",
            }
        else:
            errors = getattr(dataset_loader, "last_errors", {})
        checks.extend(_dataset_precondition_checks(loaded, errors))

    verdict = _blocked_verdict(checks)
    if verdict is not None:
        artifact = _blocked_artifact(
            config=config,
            duration_s=config.clock() - start,
            checks=checks,
            model_specs=model_specs,
            loaded=loaded,
        )
    else:
        artifact = _success_artifact(
            config=config,
            duration_s=config.clock() - start,
            checks=checks,
            model_specs=model_specs,
            loaded=loaded,
            scorer=scorer,
        )

    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--sample-per-dataset", type=int, default=DEFAULT_SAMPLE_PER_DATASET)
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    run_experiment(
        ExperimentConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            sample_per_dataset=args.sample_per_dataset,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
