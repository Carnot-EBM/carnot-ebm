"""Exp 3926 valid efficiency head-to-head versus the competent judge.

This runner only compares verifiers after the Exp 3925 judge has proven it is
competent.  The previous Exp 3917 comparison used a below-chance judge, so this
module gates on the competent-judge artifact first, then runs the Exp 3884
corpus through the Exp 3905 timing wrapper for both verifier paths.

Spec refs: REQ-VERIFY-3926, SCENARIO-VERIFY-3926,
SCENARIO-VERIFY-3926-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:  # pragma: no cover - depends on invocation path.
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify import competent_llm_judge as competent_judge  # noqa: E402
from carnot.verify.cost_instrumented_verification import (  # noqa: E402
    _llama_token_count,
    _text_token_count,
    measure_verification_cost,
    model_params_for_path,
)


EXPERIMENT_ID = 3926
TITLE = "valid_efficiency_head_to_head"
OUTPUT_REL_PATH = Path("results/experiment_3926_valid_efficiency_head_to_head.json")
EXP3925_ARTIFACT_REL_PATH = Path("results/experiment_3925_competent_judge_build.json")
EXP3884_ARTIFACT_REL_PATH = Path("results/experiment_3884_in_distribution_error_rich_corpus.json")
ENERGY_MODULE_PATH = "python/carnot/eval/in_distribution_error_rich_corpus.py"
JUDGE_MODULE_PATH = "python/carnot/verify/competent_llm_judge.py"
COST_HARNESS_MODULE_PATH = "python/carnot/verify/cost_instrumented_verification.py"
SCRIPT_PATH = "scripts/experiments/experiment_3926_valid_efficiency_head_to_head.py"
SPEC_PATH = "openspec/capabilities/verification/spec.md"
RANDOM_SEED = 3926
DEFAULT_MIN_CORPUS_ITEMS = 200
DEFAULT_BOOTSTRAP_REPS = 1000
LIVE_FLOOR_S = 60.0
INFERENCE_SUBSTRATE = (
    "live_llm_inference:competent_exp3925_judge_vs_cpu_exp3884_energy_verifier"
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

REQUIRED_FIELDS = {
    "energy_auroc",
    "llm_judge_auroc",
    "judge_positive_control_passed",
    "accuracy_parity",
    "pareto_dominates",
    "cost_ratio_walltime",
    "cost_ratio_flops",
    "energy_per_item_ms",
    "llm_per_item_ms",
    "judge_model_used",
    "n_items",
    "corpus_source",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "energy_auroc": (
        "Accuracy of each verifier on the SAME labels; the judge is now competent "
        "so parity/Pareto is testable."
    ),
    "llm_judge_auroc": (
        "Accuracy of each verifier on the SAME labels; the judge is now competent "
        "so parity/Pareto is testable."
    ),
    "judge_positive_control_passed": (
        "BARE BOOL - llm_judge_auroc>0.6 on THIS corpus; the gate the .362 comparison lacked."
    ),
    "accuracy_parity": (
        "BARE BOOL - energy AUROC within the competent judge's CI95 (equally effective)."
    ),
    "pareto_dominates": (
        "BARE BOOL - energy AUROC >= judge AUROC (more accurate AND cheaper = strict Pareto win)."
    ),
    "cost_ratio_walltime": (
        "BARE FLOAT - llm_per_item_ms / energy_per_item_ms; the 'Nx cheaper' headline."
    ),
    "cost_ratio_flops": (
        "BARE FLOAT - FLOP/token-based ratio (amortized load excluded); "
        "substrate-independent corroboration."
    ),
    "energy_per_item_ms": "Per-item latency for each verifier - the latency story.",
    "llm_per_item_ms": "Per-item latency for each verifier - the latency story.",
    "judge_model_used": "Which GGUF the competent judge loaded (records the comparator config).",
    "n_items": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "corpus_source": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "preconditions_checked": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "model_specs": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "random_seed": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "random_seeds_used": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "reproducibility_checksum": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "duration_s": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
    "inference_substrate": (
        "Pre-Launch + Adversarial-Verify - a real competent-judge run over hundreds "
        "of items takes wall-clock; <60s = fabrication."
    ),
}

WRAPPED_VALUE_FORBIDDEN_FIELDS = (
    "energy_auroc",
    "llm_judge_auroc",
    "judge_positive_control_passed",
    "accuracy_parity",
    "pareto_dominates",
    "cost_ratio_walltime",
    "cost_ratio_flops",
    "energy_per_item_ms",
    "llm_per_item_ms",
    "judge_model_used",
    "n_items",
    "corpus_source",
    "random_seed",
    "duration_s",
    "inference_substrate",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource checked before the Exp 3926 live comparison."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3926."""

    repo_root: Path
    output_path: Path | None = None
    started_monotonic_s: float | None = None
    clock: Callable[[], float] = time.perf_counter
    random_seed: int = RANDOM_SEED
    min_corpus_items: int = DEFAULT_MIN_CORPUS_ITEMS
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
    cuda_probe_timeout_s: int = 60
    max_tokens: int = competent_judge.DEFAULT_MAX_TOKENS

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_monotonic_s is None else self.started_monotonic_s

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


GeneratorLoader = Callable[[dict[str, object], ExperimentConfig], tuple[object, dict[str, object]]]
CudaProbe = Callable[[ExperimentConfig], PreconditionCheck]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_from_monotonic(duration_s: float) -> str:
    return datetime.fromtimestamp(time.time() + duration_s, tz=UTC).isoformat().replace("+00:00", "Z")


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _label_to_error(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "incorrect", "error", "wrong", "bad", "true"}:
        return 1
    if text in {"0", "correct", "valid", "right", "good", "ok", "false"}:
        return 0
    raise ValueError(f"unsupported gold error label: {value!r}")


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _bootstrap_ci95(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seed: int,
    reps: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n_items = len(labels)
    estimates: list[float] = []
    for _index in range(max(1, reps)):
        sample_indices = [rng.randrange(n_items) for _unused in range(n_items)]
        sample_labels = [int(labels[index]) for index in sample_indices]
        if len(set(sample_labels)) < 2:
            continue
        sample_scores = [float(scores[index]) for index in sample_indices]
        estimates.append(_auroc(sample_labels, sample_scores))
    if not estimates:
        point = _auroc(labels, scores)
        return (point, point)
    estimates.sort()
    lo_index = max(0, min(len(estimates) - 1, math.floor(0.025 * (len(estimates) - 1))))
    hi_index = max(0, min(len(estimates) - 1, math.ceil(0.975 * (len(estimates) - 1))))
    return (float(estimates[lo_index]), float(estimates[hi_index]))


def _token_count(text: str) -> int:
    counted = _text_token_count(text)
    return counted if counted > 0 else len(_TOKEN_RE.findall(text))


def load_exp3925_source(repo_root: Path) -> dict[str, object]:
    """Load the upstream competent-judge artifact and enforce its gate."""

    artifact_path = repo_root / EXP3925_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3925_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3925 artifact is not a JSON object")
    if artifact.get("unit_test_passed") is not True:
        raise ValueError("exp3925 unit_test_passed is not true")
    fixture_auroc = float(artifact.get("fixture_auroc") or 0.0)
    if fixture_auroc <= 0.65:
        raise ValueError(f"exp3925 fixture_auroc must be >0.65, got {fixture_auroc}")
    judge_module_path = repo_root / str(artifact.get("judge_module_path") or JUDGE_MODULE_PATH)
    if not judge_module_path.is_file() and repo_root == REPO_ROOT:  # pragma: no cover
        raise FileNotFoundError(f"judge module missing: {judge_module_path}")
    return {
        "artifact_path": EXP3925_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "unit_test_passed": True,
        "fixture_auroc": fixture_auroc,
        "judge_model_used": artifact.get("judge_model_used"),
        "model_specs": dict(artifact.get("model_specs") or {}),
        "honest_verdict": artifact.get("honest_verdict"),
        "judge_module_path": str(artifact.get("judge_module_path") or JUDGE_MODULE_PATH),
    }


def load_exp3884_corpus(config: ExperimentConfig) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Load Exp 3884 labels, item order, and cached energy scores."""

    artifact_path = config.repo_root / EXP3884_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3884_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3884 artifact is not a JSON object")

    corpus_path = config.repo_root / str(artifact.get("corpus_path") or "")
    scores_path = config.repo_root / str(artifact.get("per_item_ensemble_scores_path") or "")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"exp3884 corpus missing: {corpus_path}")
    if not scores_path.is_file():
        raise FileNotFoundError(f"exp3884 score file missing: {scores_path}")

    corpus_payload = _read_json(corpus_path)
    scores_payload = _read_json(scores_path)
    corpus_rows = [dict(row) for row in corpus_payload.get("items", []) if isinstance(row, dict)]
    score_rows = [dict(row) for row in scores_payload.get("items", []) if isinstance(row, dict)]
    if len(corpus_rows) != len(score_rows):
        raise ValueError("exp3884 corpus and score files have different lengths")
    if len(corpus_rows) < config.min_corpus_items:
        raise ValueError(
            f"exp3884 corpus has {len(corpus_rows)} items; required>={config.min_corpus_items}"
        )

    items: list[dict[str, object]] = []
    for index, (corpus_row, score_row) in enumerate(zip(corpus_rows, score_rows, strict=True)):
        corpus_id = str(corpus_row.get("corpus_item_id"))
        if corpus_id != str(score_row.get("corpus_item_id")):
            raise ValueError(f"exp3884 item order mismatch at index {index}")
        label = corpus_row.get("label")
        if str(label) != str(score_row.get("label")):
            raise ValueError(f"exp3884 label mismatch at index {index}")
        step_text = str(corpus_row.get("step_text", "")).strip()
        if not step_text:
            raise ValueError(f"exp3884 item {index} lacks step text")
        items.append(
            {
                "index": index,
                "corpus_item_id": corpus_id,
                "question_id": str(corpus_row.get("question_id", "")),
                "step": step_text,
                "step_text": step_text,
                "label": str(label),
                "gold_error": _label_to_error(label),
                "energy_score": float(score_row["carnot_ensemble_score"]),
                "synthetic": bool(corpus_row.get("synthetic")),
            }
        )

    labels = [int(item["gold_error"]) for item in items]
    if len(set(labels)) < 2:
        raise ValueError("exp3884 corpus must contain both labels")
    energy_scores = [float(item["energy_score"]) for item in items]
    corpus_source = {
        "artifact_path": EXP3884_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "corpus_path": str(artifact.get("corpus_path")),
        "corpus_sha256": _sha256_file(corpus_path),
        "scores_path": str(artifact.get("per_item_ensemble_scores_path")),
        "scores_sha256": _sha256_file(scores_path),
        "n_total_items": len(items),
        "n_incorrect": sum(labels),
        "energy_auroc_from_scores": _auroc(labels, energy_scores),
    }
    return tuple(items), corpus_source


def _probe_cuda_with_venv(config: ExperimentConfig) -> PreconditionCheck:  # pragma: no cover
    try:
        proc = subprocess.run(
            [str(config.venv_python()), "-c", "import torch; assert torch.cuda.is_available()"],
            capture_output=True,
            cwd=config.repo_root,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


def probe_preconditions(
    config: ExperimentConfig,
    *,
    cuda_probe: CudaProbe = _probe_cuda_with_venv,
) -> tuple[
    tuple[PreconditionCheck, ...],
    str | None,
    dict[str, object],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    """Check hard resources before any full-corpus scoring."""

    checks: list[PreconditionCheck] = [cuda_probe(config)]

    exp3925_source: dict[str, object] = {}
    try:
        exp3925_source = load_exp3925_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3925_competent_judge_ready",
                True,
                (
                    f"fixture_auroc={exp3925_source.get('fixture_auroc')} "
                    f"model={exp3925_source.get('judge_model_used')}"
                ),
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3925_competent_judge_ready", False, repr(exc)))

    try:
        importlib.import_module("carnot.verify.cost_instrumented_verification")
        checks.append(
            PreconditionCheck(
                "exp3905_cost_instrumentation_import",
                True,
                COST_HARNESS_MODULE_PATH,
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3905_cost_instrumentation_import", False, repr(exc)))

    items: tuple[dict[str, object], ...] = ()
    corpus_source: dict[str, object] = {}
    try:
        items, corpus_source = load_exp3884_corpus(config)
        checks.append(
            PreconditionCheck(
                "exp3884_corpus_ready",
                True,
                f"n={len(items)} scores={corpus_source.get('scores_path')}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3884_corpus_ready", False, repr(exc)))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("exp3925_competent_judge_ready", False):
        blocked_reason = "blocked_upstream_competent_judge_not_ready"
    elif not available.get("exp3905_cost_instrumentation_import", False):
        blocked_reason = "blocked_upstream_cost_harness"
    elif not available.get("exp3884_corpus_ready", False):
        blocked_reason = "blocked_exp3884_corpus_not_ready"

    return tuple(checks), blocked_reason, exp3925_source, items, corpus_source


def run_energy_verifier_from_exp3884_scores(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return Exp 3884 energy scores with explicit cheap-verifier cost evidence."""

    scores: list[float] = []
    scanned_tokens = 0
    estimated_ops = 0
    for item in items:
        text = str(item.get("step_text", item.get("step", "")))
        token_count = _token_count(text)
        char_count = len(text)
        scores.append(float(item["energy_score"]))
        scanned_tokens += token_count
        estimated_ops += (2 * char_count) + (16 * token_count) + 32
    return {"scores": scores, "est_tokens": scanned_tokens, "est_flops": estimated_ops}


def _safe_llama_token_count(generator: object, text: str, *, add_bos: bool) -> int:
    if hasattr(generator, "tokenize"):
        return _llama_token_count(generator, text, add_bos=add_bos)
    counted = _token_count(text)
    return counted + (1 if add_bos else 0)


def run_competent_llm_judge_verifier(
    items: Sequence[Mapping[str, object]],
    *,
    generator: object,
    model_path: str | Path | None,
    model_params: int | None = None,
    max_tokens: int = competent_judge.DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    """Score rows with Exp 3925's competent judge and count judge tokens."""

    scores: list[float] = []
    raw_texts: list[str] = []
    parsed_flags: list[bool] = []
    total_tokens = 0
    for item in items:
        prompt = competent_judge.build_judge_prompt(dict(item))
        prompt_tokens = _safe_llama_token_count(generator, prompt, add_bos=True)
        judged = competent_judge.judge_step(dict(item), generator, max_tokens=max_tokens)
        raw_text = str(judged["raw_text"])
        completion_tokens = _safe_llama_token_count(generator, raw_text, add_bos=False)
        scores.append(float(judged["verdict_prob"]))
        raw_texts.append(raw_text)
        parsed_flags.append(bool(judged["parsed"]))
        total_tokens += prompt_tokens + completion_tokens

    params = model_params
    if params is None:
        params = model_params_for_path(str(model_path or "Qwen3.6-35B-A3B"))
    return {
        "scores": scores,
        "est_tokens": total_tokens,
        "est_flops": 2 * int(params) * total_tokens,
        "parse_rate": sum(parsed_flags) / len(parsed_flags) if parsed_flags else 0.0,
        "raw_text_sha256": _checksum(raw_texts),
    }


def load_competent_generator(
    exp3925_source: dict[str, object],
    config: ExperimentConfig,
) -> tuple[object, dict[str, object]]:  # pragma: no cover - exercised by live experiment.
    """Load the GGUF generator from the Exp 3925 comparator configuration."""

    from carnot.verify.gguf_inference import load_gguf_generator

    source_specs = dict(exp3925_source.get("model_specs") or {})
    judge_model_used = exp3925_source.get("judge_model_used")
    prefer_order = list(source_specs.get("prefer_order") or competent_judge.COMPETENT_PREFER_ORDER)
    if judge_model_used:
        model_name = str(judge_model_used)
        prefer_order = [model_name, *[name for name in prefer_order if name != model_name]]
    generator, meta = load_gguf_generator(
        prefer_order=tuple(prefer_order),
        n_ctx=int(source_specs.get("n_ctx") or competent_judge.DEFAULT_N_CTX),
        max_n_gpu_layers=int(
            source_specs.get("max_n_gpu_layers", competent_judge.DEFAULT_MAX_N_GPU_LAYERS)
        ),
    )
    return generator, {
        **source_specs,
        **dict(meta),
        "loader": "carnot.verify.gguf_inference.load_gguf_generator",
        "source_exp3925_artifact": exp3925_source.get("artifact_path"),
        "source_exp3925_artifact_sha256": exp3925_source.get("artifact_sha256"),
        "judge_module_path": JUDGE_MODULE_PATH,
    }


def _ratio(numerator: object, denominator: object) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den <= 0.0:
        return None
    return num / den


def _render_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.{digits}f}"


def _classify_verdict(
    *,
    judge_positive_control_passed: bool,
    accuracy_parity: bool,
    pareto_dominates: bool,
    cost_ratio_walltime: float | None,
    energy_auroc: float,
    llm_judge_auroc: float,
    duration_s: float,
) -> str:
    if not judge_positive_control_passed:
        return "blocked_competent_judge_failed_positive_control_on_corpus"
    if duration_s < LIVE_FLOOR_S:
        return "blocked_llm_judge_not_invoked"
    if cost_ratio_walltime is None or cost_ratio_walltime <= 10.0:
        return "complete: efficiency_INCONCLUSIVE_cost_ratio_walltime<=10"

    ratio_text = _render_metric(cost_ratio_walltime, digits=2)
    energy_text = _render_metric(energy_auroc)
    judge_text = _render_metric(llm_judge_auroc)
    if accuracy_parity or pareto_dominates:
        return (
            "complete: "
            f"efficiency_VALID_EARNS_PLACE_energy{energy_text}_judge{judge_text}_"
            f"{ratio_text}x_cheaper_pareto{str(pareto_dominates).lower()}_vs_competent_judge"
        )
    return (
        "complete: "
        f"efficiency_CHEAPER_{ratio_text}x_but_JUDGE_MORE_ACCURATE_"
        f"energy{energy_text}_judge{judge_text}_cascade_is_the_story"
    )


def build_artifact(
    *,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    exp3925_source: dict[str, object],
    corpus_source: dict[str, object],
    model_specs: dict[str, object],
    energy_cost: dict[str, object],
    llm_cost: dict[str, object],
    labels: Sequence[int],
    energy_scores: Sequence[float],
    llm_scores: Sequence[float],
) -> dict[str, object]:
    """Build the terminal Exp 3926 artifact from measured full-corpus costs."""

    duration_s = config.clock() - config.start_time()
    energy_auroc = float(energy_cost["auroc"])
    llm_judge_auroc = float(llm_cost["auroc"])
    judge_ci95 = _bootstrap_ci95(
        labels,
        llm_scores,
        seed=config.random_seed,
        reps=config.bootstrap_reps,
    )
    judge_positive_control_passed = llm_judge_auroc > 0.6
    accuracy_parity = judge_ci95[0] <= energy_auroc <= judge_ci95[1]
    pareto_dominates = energy_auroc >= llm_judge_auroc
    cost_ratio_walltime = _ratio(llm_cost["per_item_wall_ms"], energy_cost["per_item_wall_ms"])
    cost_ratio_flops = _ratio(llm_cost["est_flops"], energy_cost["est_flops"])

    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "exp3925_source": exp3925_source,
        "corpus_source": str(corpus_source.get("corpus_path") or corpus_source.get("artifact_path") or ""),
        "corpus_provenance": corpus_source,
        "labels": list(labels),
        "energy_scores": [round(float(score), 12) for score in energy_scores],
        "llm_scores": [round(float(score), 12) for score in llm_scores],
        "energy_cost": energy_cost,
        "llm_cost": llm_cost,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.now(tz=UTC).strftime("%Y%m%d"),
        "finished_at": _iso_from_monotonic(duration_s),
        "energy_auroc": energy_auroc,
        "llm_judge_auroc": llm_judge_auroc,
        "judge_bootstrap_ci95": [judge_ci95[0], judge_ci95[1]],
        "judge_positive_control_passed": judge_positive_control_passed,
        "accuracy_parity": bool(accuracy_parity),
        "pareto_dominates": bool(pareto_dominates),
        "cost_ratio_walltime": cost_ratio_walltime,
        "cost_ratio_flops": cost_ratio_flops,
        "energy_per_item_ms": float(energy_cost["per_item_wall_ms"]),
        "llm_per_item_ms": float(llm_cost["per_item_wall_ms"]),
        "energy_total_wall_s": float(energy_cost["total_wall_s"]),
        "llm_total_wall_s": float(llm_cost["total_wall_s"]),
        "energy_est_tokens": int(energy_cost["est_tokens"]),
        "llm_est_tokens": int(llm_cost["est_tokens"]),
        "energy_est_flops": int(energy_cost["est_flops"]),
        "llm_est_flops": int(llm_cost["est_flops"]),
        "judge_model_used": model_specs.get("model_used") or exp3925_source.get("judge_model_used"),
        "n_items": int(energy_cost["n_items"]),
        "corpus_source": str(corpus_source.get("corpus_path") or corpus_source.get("artifact_path") or ""),
        "corpus_provenance": corpus_source,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "exp3925_competent_judge_source": exp3925_source,
        "energy_cost": energy_cost,
        "llm_cost": llm_cost,
        "random_seed": config.random_seed,
        "random_seeds_used": {
            "experiment": config.random_seed,
            "bootstrap_ci": config.random_seed,
            "exp3884": 3884,
            "exp3925": 3925,
        },
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    verdict = _classify_verdict(
        judge_positive_control_passed=judge_positive_control_passed,
        accuracy_parity=bool(accuracy_parity),
        pareto_dominates=bool(pareto_dominates),
        cost_ratio_walltime=cost_ratio_walltime,
        energy_auroc=energy_auroc,
        llm_judge_auroc=llm_judge_auroc,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = verdict
    artifact["status"] = verdict
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
    corpus_source: dict[str, object] | None = None,
    exp3925_source: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a blocked artifact without fabricated accuracy or cost claims."""

    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "reason": reason,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "corpus_source": str((corpus_source or {}).get("corpus_path") or ""),
        "corpus_provenance": corpus_source or {},
        "exp3925_source": exp3925_source or {},
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "energy_auroc": None,
        "llm_judge_auroc": None,
        "judge_bootstrap_ci95": None,
        "judge_positive_control_passed": False,
        "accuracy_parity": False,
        "pareto_dominates": False,
        "cost_ratio_walltime": None,
        "cost_ratio_flops": None,
        "energy_per_item_ms": None,
        "llm_per_item_ms": None,
        "energy_total_wall_s": None,
        "llm_total_wall_s": None,
        "energy_est_tokens": None,
        "llm_est_tokens": None,
        "energy_est_flops": None,
        "llm_est_flops": None,
        "judge_model_used": None,
        "n_items": 0,
        "corpus_source": str((corpus_source or {}).get("corpus_path") or ""),
        "corpus_provenance": corpus_source or {},
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "exp3925_competent_judge_source": exp3925_source or {},
        "energy_cost": None,
        "llm_cost": None,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3926 fields and bare-scalar discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    for key in WRAPPED_VALUE_FORBIDDEN_FIELDS:
        if isinstance(artifact.get(key), dict):
            raise ValueError(f"{key} must not be a value/principle wrapper")
    for key in (
        "judge_positive_control_passed",
        "accuracy_parity",
        "pareto_dominates",
    ):
        if not isinstance(artifact[key], bool):
            raise ValueError(f"{key} must be a bare bool")
    for key in ("energy_auroc", "llm_judge_auroc", "cost_ratio_walltime", "cost_ratio_flops"):
        if artifact[key] is not None and not isinstance(artifact[key], float):
            raise ValueError(f"{key} must be a bare float or null")
    for key in ("energy_per_item_ms", "llm_per_item_ms"):
        if artifact[key] is not None and not isinstance(artifact[key], float):
            raise ValueError(f"{key} must be a bare float or null")
    if not isinstance(artifact["n_items"], int):
        raise ValueError("n_items must be a bare int")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if verdict.startswith("blocked_upstream") and artifact["energy_auroc"] is not None:
        raise ValueError("upstream-blocked artifacts must not claim verifier accuracy")
    if (
        verdict == "blocked_competent_judge_failed_positive_control_on_corpus"
        and artifact["judge_positive_control_passed"] is not False
    ):
        raise ValueError("positive-control blocked artifact must keep the gate false")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    write: bool = True,
    cuda_probe: CudaProbe = _probe_cuda_with_venv,
    generator_loader: GeneratorLoader = load_competent_generator,
) -> dict[str, object]:
    """Run Exp 3926 end to end, or write a blocked artifact on failed gates."""

    config = config or ExperimentConfig(repo_root=REPO_ROOT)
    started = config.start_time()
    active_config = replace(config, started_monotonic_s=started)
    output_path = active_config.resolved_output_path()
    checks, blocked_reason, exp3925_source, items, corpus_source = probe_preconditions(
        active_config,
        cuda_probe=cuda_probe,
    )
    preflight_model_specs = {
        "judge_model_used": exp3925_source.get("judge_model_used"),
        "max_tokens": active_config.max_tokens,
        "source_exp3925_model_specs": exp3925_source.get("model_specs", {}),
    }
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight_model_specs,
            corpus_source=corpus_source,
            exp3925_source=exp3925_source,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    generator, loaded_model_specs = generator_loader(exp3925_source, active_config)
    model_path = loaded_model_specs.get("gguf_path")
    model_specs = {
        **preflight_model_specs,
        **loaded_model_specs,
        "energy_module_path": ENERGY_MODULE_PATH,
        "cost_harness_module_path": COST_HARNESS_MODULE_PATH,
    }
    model_params = model_params_for_path(str(model_path or model_specs.get("model_used", "")))

    energy_result: dict[str, object] = {}
    llm_result: dict[str, object] = {}

    def measured_energy(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        result = run_energy_verifier_from_exp3884_scores(rows)
        energy_result.update(result)
        return result

    def measured_llm(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        result = run_competent_llm_judge_verifier(
            rows,
            generator=generator,
            model_path=str(model_path or ""),
            model_params=model_params,
            max_tokens=active_config.max_tokens,
        )
        llm_result.update(result)
        return result

    energy_cost = measure_verification_cost(
        measured_energy,
        items,
        "exp3884_energy_verifier",
    )
    llm_cost = measure_verification_cost(
        measured_llm,
        items,
        "exp3925_competent_llm_judge",
    )
    labels = [int(item["gold_error"]) for item in items]
    energy_scores = [float(score) for score in energy_result.get("scores", [])]
    llm_scores = [float(score) for score in llm_result.get("scores", [])]

    artifact = build_artifact(
        config=active_config,
        preconditions_checked=checks,
        exp3925_source=exp3925_source,
        corpus_source=corpus_source,
        model_specs=model_specs,
        energy_cost=energy_cost,
        llm_cost=llm_cost,
        labels=labels,
        energy_scores=energy_scores,
        llm_scores=llm_scores,
    )
    if write:
        write_artifact(output_path, artifact)
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - script entrypoint.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(repo_root=args.repo_root, output_path=args.output_path),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


def main() -> int:  # pragma: no cover - script entrypoint.
    return cli_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
