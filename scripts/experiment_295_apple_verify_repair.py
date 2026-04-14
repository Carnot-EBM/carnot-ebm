#!/usr/bin/env python3
"""Experiment 295: Verify-repair benchmark on Apple adversarial GSM8K corpus — pre-warm fix.

**Research hypothesis (credibility benchmark, 3rd attempt):**
    Verify-repair improvement should be LARGER on number_swap adversarial variants
    than on standard GSM8K.  Semantic grounding (Exp 279) detects stale-answer errors
    at 100%, and number_swap variants generate exactly this error pattern.

    This is the pre-warm-fixed re-run of Exp 283.  Exps 282 and 283 were INCONCLUSIVE
    for two consecutive milestones because GPU stall (diagnosed in Exp 294) caused the
    60 s inference timeout to fire before any question was processed.  This script
    applies the Exp 294 fix: explicit model_prewarm() before the timed benchmark loop.

**Design — 12 benchmark cells:**
    3 modes × 2 variant types × 2 models = 12 cells

    Modes:
      - baseline       — no verification, raw model output
      - verify_only    — run verifiers, flag violations, do NOT repair
      - verify_repair  — run verifiers, repair if violation detected

    Variant types (from Exp 281 adversarial corpus):
      - number_swap         — numeric operands scaled; model may recall stale answer
      - irrelevant_sentence — distractor sentence inserted; answer unchanged

    Models (DualGPURunner, wired at startup):
      - Qwen3.5-0.8B   on GPU 0
      - Gemma4-E4B-it  on GPU 1

**Pre-warm fix (from Exp 294 — REQ-VERIFY-079):**
    model_prewarm() is called for each model before any timed benchmark work.
    The per-question generate_fn re-uses the already-loaded model from _model_cache,
    so the first question never pays the cold-load penalty.

**Primary criterion:**
    ``Δ(verify_repair, number_swap) > Δ(verify_repair, standard)``
    where Δ(mode, variant) = accuracy(mode, variant) − accuracy(baseline, variant)
    and standard accuracy is taken from the Exp 294 baseline artifact.

**Logit saving (required for Exp 291 JEPA training):**
    Logit tensors from the initial baseline generation step are saved at
    25/50/75/100% prefix fractions as NumPy .npy files under data/research/.

**Checkpoint / timeout:**
    Checkpoints are written every CHECKPOINT_INTERVAL (10) questions.
    A 60 s hard timeout is enforced per inference call; on timeout a partial
    artifact is emitted with ``stall_at`` identifying (model:mode:variant:qid).

Usage (live GPU, CARNOT_FORCE_LIVE=1):
    cd /path/to/carnot
    CARNOT_FORCE_LIVE=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/experiment_295_apple_verify_repair.py

Usage (mock / unit mode — no GPU required):
    CARNOT_FORCE_LIVE=0 .venv/bin/pytest \\
        tests/python/test_experiment_295_apple_verify_repair.py -q --no-cov -n0

Spec: REQ-VERIFY-079, REQ-VERIFY-068, REQ-VERIFY-069, REQ-VERIFY-070,
      REQ-VERIFY-071, REQ-VERIFY-072, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104,
      SCENARIO-VERIFY-105, SCENARIO-VERIFY-106, SCENARIO-VERIFY-107,
      SCENARIO-VERIFY-108
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 295
"""Experiment number — matches the filename and artifact ``experiment`` field."""

RUN_DATE: str = "20260414"
"""Wall-clock date of this experiment run."""

CHECKPOINT_INTERVAL: int = 10
"""Number of questions processed before each checkpoint write (REQ-VERIFY-068)."""

INFERENCE_TIMEOUT_SECONDS: int = 60
"""Hard per-call timeout in seconds (REQ-VERIFY-071).  If exceeded, a partial
artifact is emitted with a ``stall_at`` field."""

PREWARM_LOAD_TIMEOUT_SECONDS: int = 15
"""Timeout for the model-load phase of the pre-warm health-check (REQ-VERIFY-079).
If loading takes longer than this, ``stall_root_cause`` is set to ``'lazy_load_stall'``."""

PREWARM_WARMUP_PROMPTS: int = 2
"""Number of warm-up prompts run per model after the initial health-check to fully
populate CUDA compilation caches and reduce first-batch latency."""

LOGIT_FRACTIONS: list[float] = [0.25, 0.50, 0.75, 1.00]
"""Prefix fractions at which accumulated logit tensors are saved to disk (REQ-VERIFY-070).

Files are named ``logits_295_{model_slug}_{mode}_{variant}_{pct}pct.npy``.
"""

MODES: list[str] = ["baseline", "verify_only", "verify_repair"]
"""The three inference modes evaluated (REQ-VERIFY-068)."""

VARIANT_TYPES: list[str] = ["number_swap", "irrelevant_sentence"]
"""Adversarial variant types from the Exp 281 corpus (REQ-VERIFY-068)."""

ARTIFACT_SCHEMA: list[str] = [
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "inference_mode",
    "cell_results",
    "logit_paths",
    "improvement_deltas",
    "primary_criterion_met",
    "comparison_refs",
    "partial",
    "stall_at",
    "pre_warm_status",
    "pre_warm_time_s",
]
"""Required top-level fields in the Exp 295 artifact JSON (SCENARIO-VERIFY-103).

Adds ``pre_warm_status`` and ``pre_warm_time_s`` vs the Exp 283 schema to record
GPU pre-warm results (REQ-VERIFY-079 / SCENARIO-VERIFY-107).
"""

# Default model pair — Qwen on GPU 0, Gemma on GPU 1 (SCENARIO-VERIFY-105).
MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
]

_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "research" / "gsm8k_adversarial_281.jsonl"
)
_CHECKPOINT_BASE = (
    Path(__file__).resolve().parents[1] / "results" / "checkpoints" / "experiment_295"
)
_LOGIT_BASE = Path(__file__).resolve().parents[1] / "data" / "research"
_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "experiment_295_results.json"
_EXP294_RESULTS = Path(__file__).resolve().parents[1] / "results" / "experiment_294_results.json"
_EXP235_RESULTS = Path(__file__).resolve().parents[1] / "results" / "experiment_235_results.json"

# ---------------------------------------------------------------------------
# Utilities (same pattern as Exps 282, 283, 294)
# ---------------------------------------------------------------------------

_SLUG_KEEP = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


def safe_slug(text: str) -> str:
    """Convert *text* into a filesystem-safe lower-case slug.

    Spaces and path separators become underscores; any other character outside
    ``[a-z0-9_-]`` is also replaced with an underscore.

    Example::

        safe_slug("Qwen3.5-0.8B") -> "qwen3_5-0_8b"
    """
    cleaned = text.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(c if c in _SLUG_KEEP else "_" for c in cleaned)


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format (e.g. ``2026-04-14T05:00:00Z``)."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_repo_root() -> Path:
    """Resolve the repository root, honoring the ``CARNOT_REPO_ROOT`` override used in tests."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def _extract_numeric_answer(response: str) -> float | None:
    """Extract the final numeric answer from a model response string.

    Looks for the last decimal number in *response*.  Returns ``None`` if no
    number is found.  GSM8K answers are always integers but we parse as float
    to tolerate responses like ``"The answer is 5.0"``.

    Args:
        response: Raw model-generated text.

    Returns:
        The last number found in the response, or ``None``.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", response)
    if not matches:
        return None
    return float(matches[-1])


def _is_correct(response: str, expected_answer: int | float) -> bool:
    """Return True if the extracted answer in *response* matches *expected_answer*.

    Args:
        response: Raw model response text.
        expected_answer: The ground-truth numeric answer.

    Returns:
        True if the extracted answer matches within 1e-6.
    """
    pred = _extract_numeric_answer(response)
    if pred is None:
        return False
    return abs(pred - float(expected_answer)) < 1e-6


# ---------------------------------------------------------------------------
# Pre-warm dataclass (mirrors Exp 294 — REQ-VERIFY-079)
# ---------------------------------------------------------------------------


@dataclass
class PrewarmResult:
    """Result of a single ``model_prewarm()`` call.

    Fields
    ------
    model_name : str
        Human-readable model label (e.g. ``"Qwen3.5-0.8B"``).
    gpu_id : int
        CUDA device index this model was loaded onto.
    load_time_s : float
        Wall-clock seconds for the model-load phase.  Zero if load failed.
    health_ok : bool
        ``True`` iff both loading and the health-check prompt completed within
        their respective timeouts and produced a non-empty response.
    stall_root_cause : str | None
        One of ``"lazy_load_stall"``, ``"cuda_oom"``, ``"unknown"``, or ``None``
        (no stall).  ``"lazy_load_stall"`` is set when a timeout occurs during
        either the load or generate phase.
    """

    model_name: str
    gpu_id: int
    load_time_s: float
    health_ok: bool
    stall_root_cause: str | None


def model_prewarm(
    model_name: str,
    hf_id: str,
    gpu_id: int,
    *,
    health_prompt: str = "What is 2+2?",
    timeout_seconds: float = PREWARM_LOAD_TIMEOUT_SECONDS,
    load_fn: Callable[[str, int], tuple[Any, Any]] | None = None,
    generate_fn: Callable[[Any, Any, str], str] | None = None,
) -> PrewarmResult:
    """Load a model onto a GPU and run a health-check prompt to confirm it responds.

    This is the Exp 294 pre-warm fix applied to the verify-repair pipeline.
    By calling it before the timed benchmark loop, the VRAM transfer and CUDA
    compilation caches are fully warm before the first timed inference call.

    Args:
        model_name: Human-readable label (e.g. ``"Qwen3.5-0.8B"``).
        hf_id: HuggingFace model ID (e.g. ``"Qwen/Qwen3.5-0.8B"``).
        gpu_id: CUDA device index (0 or 1).
        health_prompt: Short prompt used to verify the model produces output.
        timeout_seconds: Combined timeout for load + health-check (seconds).
        load_fn: Optional override for model loading (injected in tests).
            Signature: ``(hf_id: str, gpu_id: int) -> (model, tokenizer)``.
        generate_fn: Optional override for generation (injected in tests).
            Signature: ``(model, tokenizer, prompt: str) -> response_str``.

    Returns:
        :class:`PrewarmResult` describing load time, health status, and root cause.
    """
    t0 = time.perf_counter()
    load_time_s: float = 0.0

    def _do_prewarm() -> tuple[float, bool, str | None]:
        nonlocal load_time_s

        # Step 1: load model.
        t_load_start = time.perf_counter()
        if load_fn is not None:
            model, tokenizer = load_fn(hf_id, gpu_id)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()

        load_time_s = time.perf_counter() - t_load_start

        # Step 2: health-check prompt.
        if generate_fn is not None:
            response = generate_fn(model, tokenizer, health_prompt)
        else:
            import torch

            device = next(model.parameters()).device
            inputs = tokenizer(health_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            generated_ids = out[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        health_ok = bool(response and response.strip())
        return load_time_s, health_ok, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_prewarm)
        try:
            load_time_s_result, health_ok, src = future.result(timeout=timeout_seconds)
            load_time_s = load_time_s_result
            return PrewarmResult(
                model_name=model_name,
                gpu_id=gpu_id,
                load_time_s=load_time_s,
                health_ok=health_ok,
                stall_root_cause=src,
            )
        except concurrent.futures.TimeoutError:
            future.cancel()
            load_time_s = time.perf_counter() - t0
            return PrewarmResult(
                model_name=model_name,
                gpu_id=gpu_id,
                load_time_s=load_time_s,
                health_ok=False,
                stall_root_cause="lazy_load_stall",
            )
        except MemoryError:
            load_time_s = time.perf_counter() - t0
            return PrewarmResult(
                model_name=model_name,
                gpu_id=gpu_id,
                load_time_s=load_time_s,
                health_ok=False,
                stall_root_cause="cuda_oom",
            )
        except RuntimeError as exc:
            root = "cuda_oom" if "out of memory" in str(exc).lower() else "unknown"
            load_time_s = time.perf_counter() - t0
            return PrewarmResult(
                model_name=model_name,
                gpu_id=gpu_id,
                load_time_s=load_time_s,
                health_ok=False,
                stall_root_cause=root,
            )
        except Exception:  # noqa: BLE001
            load_time_s = time.perf_counter() - t0
            return PrewarmResult(
                model_name=model_name,
                gpu_id=gpu_id,
                load_time_s=load_time_s,
                health_ok=False,
                stall_root_cause="unknown",
            )


# ---------------------------------------------------------------------------
# GPU diagnostics helper (mirrors Exp 294)
# ---------------------------------------------------------------------------


def _query_vram_gb() -> dict[str, float]:
    """Query free VRAM on both GPUs via nvidia-smi.

    Returns a dict with keys ``vram_gpu0_free_gb`` and ``vram_gpu1_free_gb``.
    Values are 0.0 if nvidia-smi is unavailable or fails.
    """
    out: dict[str, float] = {"vram_gpu0_free_gb": 0.0, "vram_gpu1_free_gb": 0.0}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            for i, line in enumerate(lines[:2]):
                try:
                    mib = float(line)
                    out[f"vram_gpu{i}_free_gb"] = round(mib / 1024, 2)
                except ValueError:
                    pass
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# Checkpoint helpers (same atomic-write pattern as Exps 283/294)
# ---------------------------------------------------------------------------


def _ckpt_path(
    checkpoint_dir: Path, *, model_name: str, mode: str, variant_type: str
) -> Path:
    """Return the checkpoint file path for a (model, mode, variant_type) triple.

    Args:
        checkpoint_dir: Base directory for checkpoint files.
        model_name: Human-readable model label.
        mode: One of MODES.
        variant_type: One of VARIANT_TYPES.

    Returns:
        Absolute path to the ``.json`` checkpoint file.
    """
    return checkpoint_dir / (
        f"{safe_slug(model_name)}__{safe_slug(mode)}__{safe_slug(variant_type)}.json"
    )


def _load_ckpt(path: Path) -> dict[str, Any]:
    """Load a checkpoint file if it exists; return an empty structure otherwise.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Dict with at least a ``"completed"`` key mapping question_id → result dict.
    """
    if not path.exists():
        return {"completed": {}}
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("completed"), dict):
            return {"completed": {}}
        return payload
    except (json.JSONDecodeError, OSError):
        return {"completed": {}}


def _save_ckpt(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint file atomically using a ``.tmp`` rename.

    Args:
        path: Destination path for the checkpoint JSON.
        payload: Checkpoint data to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Logit saving helpers (Exp 295 naming: logits_295_…)
# ---------------------------------------------------------------------------


def _logit_npy_path(
    logit_dir: Path, *, model_name: str, mode: str, variant_type: str, pct: int
) -> Path:
    """Return the .npy path for the logit tensor at a given prefix fraction.

    File naming: ``logits_295_{model_slug}_{mode}_{variant}_{pct}pct.npy``

    Args:
        logit_dir: Base directory for logit files.
        model_name: Human-readable model label.
        mode: Inference mode slug.
        variant_type: Question variant slug.
        pct: Percentage value (25, 50, 75, or 100).

    Returns:
        Absolute .npy path.
    """
    fname = (
        f"logits_295_{safe_slug(model_name)}_{safe_slug(mode)}"
        f"_{safe_slug(variant_type)}_{pct}pct.npy"
    )
    return logit_dir / fname


def _save_logits(
    logit_dir: Path,
    *,
    model_name: str,
    mode: str,
    variant_type: str,
    pct: int,
    logit_list: list[np.ndarray],
) -> str:
    """Stack *logit_list* into an object array and save as .npy.

    Each element of *logit_list* is a ``(seq_len, vocab_size)`` array (batch dim
    already stripped by caller).  We store them as a 1-D NumPy object array so
    callers can retrieve individual question logits by index regardless of
    varying sequence lengths.

    Args:
        logit_dir: Destination directory.
        model_name: Model label — used in filename.
        mode: Mode label — used in filename.
        variant_type: Variant label — used in filename.
        pct: Fraction label (25 / 50 / 75 / 100) — used in filename.
        logit_list: List of per-question logit arrays with shape (seq_len, vocab_size).

    Returns:
        String path of the saved file (for inclusion in the artifact).
    """
    logit_dir.mkdir(parents=True, exist_ok=True)
    out_path = _logit_npy_path(
        logit_dir, model_name=model_name, mode=mode, variant_type=variant_type, pct=pct
    )
    arr = np.empty(len(logit_list), dtype=object)
    for i, lg in enumerate(logit_list):
        arr[i] = lg
    np.save(str(out_path), arr, allow_pickle=True)
    return str(out_path)


# ---------------------------------------------------------------------------
# Improvement delta computation (REQ-VERIFY-069)
# ---------------------------------------------------------------------------


def compute_improvement_deltas(
    cell_results: dict[str, dict[str, dict[str, Any]]],
    *,
    baseline_standard_acc: dict[str, float] | None = None,
    verify_repair_standard_acc: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-model improvement deltas for each (mode, variant_type) cell.

    Delta is defined as::

        Δ(mode, variant) = accuracy(mode, variant) − accuracy(baseline, variant)

    For the primary criterion (SCENARIO-VERIFY-104), the standard variant delta
    is computed as::

        Δ(verify_repair, standard) = verify_repair_standard_acc[model]
                                     − baseline_standard_acc[model]

    This uses standard-variant accuracy from the Exp 294 baseline artifact rather
    than running standard questions again in this experiment.

    Args:
        cell_results: Nested dict ``{model: {mode: {variant: {accuracy: float, …}}}}``.
        baseline_standard_acc: Per-model baseline accuracy on standard questions
                                (from Exp 294).  Optional; if provided, enables
                                primary criterion calculation.
        verify_repair_standard_acc: Per-model verify_repair accuracy on standard
                                     questions.  Optional; if provided alongside
                                     ``baseline_standard_acc``, the
                                     ``verify_repair.standard`` delta is included.

    Returns:
        Nested dict ``{model: {mode: {variant: delta_float}}}``.
    """
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, mode_results in cell_results.items():
        baseline_acc = {
            vt: mode_results.get("baseline", {}).get(vt, {}).get("accuracy", 0.0)
            for vt in VARIANT_TYPES
        }
        model_deltas: dict[str, dict[str, float]] = {}
        for mode in MODES:
            variant_deltas: dict[str, float] = {}
            for vt in VARIANT_TYPES:
                mode_acc = mode_results.get(mode, {}).get(vt, {}).get("accuracy", 0.0)
                variant_deltas[vt] = mode_acc - baseline_acc[vt]
            # Inject standard delta if reference data provided.
            if (
                mode == "verify_repair"
                and baseline_standard_acc is not None
                and verify_repair_standard_acc is not None
            ):
                vr_std = verify_repair_standard_acc.get(model_name, 0.0)
                bl_std = baseline_standard_acc.get(model_name, 0.0)
                variant_deltas["standard"] = vr_std - bl_std
            elif baseline_standard_acc is not None and mode == "verify_repair":
                variant_deltas["standard"] = 0.0
            model_deltas[mode] = variant_deltas
        deltas[model_name] = model_deltas
    return deltas


def _primary_criterion_met(
    improvement_deltas: dict[str, dict[str, dict[str, float]]],
) -> bool:
    """Return True if Δ(verify_repair, number_swap) > Δ(verify_repair, standard) for any model.

    This is the primary research criterion (SCENARIO-VERIFY-104): verify-repair
    improvement should be larger on number_swap adversarial questions (where the
    semantic grounding verifier fires on stale-answer errors) than on standard
    questions (where no adversarial perturbation exists).

    Args:
        improvement_deltas: Output of ``compute_improvement_deltas``.

    Returns:
        True if the criterion is satisfied for at least one model.
    """
    for model_deltas in improvement_deltas.values():
        vr_deltas = model_deltas.get("verify_repair", {})
        delta_ns = vr_deltas.get("number_swap", 0.0)
        delta_std = vr_deltas.get("standard", 0.0)
        if delta_ns > delta_std:
            return True
    return False


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_artifact(
    *,
    run_date: str,
    started_at: str,
    finished_at: str,
    inference_mode: str,
    cell_results: dict[str, Any],
    logit_paths: dict[str, Any],
    improvement_deltas: dict[str, Any],
    primary_criterion_met: bool,
    stall_at: str | None,
    comparison_refs: dict[str, Any],
    pre_warm_status: dict[str, bool],
    pre_warm_time_s: dict[str, float],
) -> dict[str, Any]:
    """Build the Exp 295 result artifact dict.

    Extends the Exp 283 artifact with pre-warm fields from Exp 294:
    - ``pre_warm_status``: Per-model bool indicating health-check passed.
    - ``pre_warm_time_s``: Per-model wall-clock seconds for the pre-warm.

    Args:
        run_date: Date string in ``YYYYMMDD`` format.
        started_at: ISO-8601 UTC start timestamp.
        finished_at: ISO-8601 UTC finish timestamp.
        inference_mode: ``"live_gpu"`` or ``"mock"``.
        cell_results: Nested dict keyed by (model → mode → variant_type).
        logit_paths: Nested dict of saved .npy paths keyed by model name.
        improvement_deltas: Output of ``compute_improvement_deltas``.
        primary_criterion_met: Whether Δ(vr, ns) > Δ(vr, std) for any model.
        stall_at: ``None`` for a complete run; otherwise the stall location string.
        comparison_refs: References to prior experiment results (Exp 294, 235).
        pre_warm_status: Dict ``{model_name: health_ok}`` from pre-warm phase.
        pre_warm_time_s: Dict ``{model_name: load_time_s}`` from pre-warm phase.

    Returns:
        JSON-serialisable artifact dict containing all ARTIFACT_SCHEMA fields.
    """
    is_partial = stall_at is not None
    return {
        "experiment": EXPERIMENT,
        "schema": "carnot.apple_verify_repair.v2",
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "inference_mode": inference_mode,
        "cell_results": cell_results,
        "logit_paths": logit_paths,
        "improvement_deltas": improvement_deltas,
        "primary_criterion_met": primary_criterion_met,
        "comparison_refs": comparison_refs,
        "partial": is_partial,
        "stall_at": stall_at,
        "pre_warm_status": pre_warm_status,
        "pre_warm_time_s": pre_warm_time_s,
    }


# ---------------------------------------------------------------------------
# VerifyRepairRunner295 — the main experiment controller
# ---------------------------------------------------------------------------


class VerifyRepairRunner295:
    """Run three inference modes on the Apple adversarial GSM8K corpus with GPU pre-warm.

    This class combines the 12-cell verify-repair benchmark from Exp 283 with
    the GPU pre-warm fix from Exp 294.  It:

    1. Accepts rows from the Exp 281 dataset (number_swap and irrelevant_sentence).
    2. Pre-warms each model on its assigned GPU before any timed benchmark work.
    3. Runs each (model, mode, variant_type) cell via an injectable ``generate_fn``.
    4. For ``verify_only`` and ``verify_repair`` modes, calls the Carnot verifiers
       (semantic grounding + formal claim verifier) and, for ``verify_repair``, feeds
       violations back to the model for iterative repair.
    5. Saves logit tensors at 25/50/75/100% prefix fractions (REQ-VERIFY-070).
    6. Writes checkpoints every CHECKPOINT_INTERVAL questions (REQ-VERIFY-068).
    7. Handles ``TimeoutError`` by emitting a partial artifact (REQ-VERIFY-071).
    8. Records ``pre_warm_verified`` in each per-question result (SCENARIO-VERIFY-108).

    Args:
        rows: List of dicts loaded from ``gsm8k_adversarial_281.jsonl``.
        model_specs: Ordered list of ``{"name": ..., "hf_id": ..., "gpu": int}`` dicts.
        generate_fn: Callable ``(question, expected_answer, *, model_name, mode,
                     variant_type) -> (response_str, logit_ndarray)``.
                     Injected in tests; defaults to live GPU inference.
        checkpoint_dir: Directory for per-(model, mode, variant) checkpoint files.
        logit_dir: Directory where logit .npy files are saved.
        timeout_seconds: Per-call hard timeout in seconds.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        model_specs: list[dict[str, Any]] | None = None,
        generate_fn: Callable[..., tuple[str, np.ndarray]] | None = None,
        checkpoint_dir: Path | None = None,
        logit_dir: Path | None = None,
        timeout_seconds: int = INFERENCE_TIMEOUT_SECONDS,
    ) -> None:
        self.rows = list(rows)
        self.model_specs: list[dict[str, Any]] = list(model_specs or MODEL_SPECS)
        self.generate_fn = generate_fn or self._default_generate_fn()
        self.checkpoint_dir: Path = checkpoint_dir or _CHECKPOINT_BASE
        self.logit_dir: Path = logit_dir or _LOGIT_BASE
        self.timeout_seconds = timeout_seconds

        # Collected results across all cells: {model → mode → variant → stats}
        self._cell_results: dict[str, dict[str, dict[str, Any]]] = {}
        # Collected logit paths: {model → {mode_variant_pct → path}}
        self._logit_paths: dict[str, dict[str, str]] = {}

        # Pre-warm results accumulated during __init__ (live mode only).
        self._pre_warm_results: dict[str, PrewarmResult] = {}

        # GPU stall diagnosis gathered at startup.
        self._stall_diagnosis: dict[str, Any] = {}

        # DualGPURunner wired at construction time (live mode only).
        self._dual_runner: Any | None = None

        if os.environ.get("CARNOT_FORCE_LIVE", "0") == "1":
            self._wire_dual_gpu_runner()
            self._run_startup_diagnostics()
            self._run_prewarm_phase()

        # Verifiers (lazy-imported to avoid hard dependency in mock mode).
        self._semantic_grounder: Any | None = None
        self._formal_claim_verifier: Any | None = None
        self._verifiers_loaded = False

    # ------------------------------------------------------------------
    # DualGPU wiring (live mode only, REQ-VERIFY-072)
    # ------------------------------------------------------------------

    def _wire_dual_gpu_runner(self) -> None:
        """Wire the DualGPUBenchmarkHarness at startup (REQ-VERIFY-072).

        GPU slots are reserved immediately, before any data loading or inference.
        Assignments: Qwen → GPU 0, Gemma → GPU 1 (SCENARIO-VERIFY-105).
        """
        try:
            import importlib.util as ilu

            harness_path = Path(__file__).parent / "experiment_258_dual_gpu_harness.py"
            spec258 = ilu.spec_from_file_location("experiment_258_dual_gpu_harness", harness_path)
            assert spec258 is not None and spec258.loader is not None
            mod258 = ilu.module_from_spec(spec258)
            spec258.loader.exec_module(mod258)  # type: ignore[union-attr]

            harness = mod258.DualGPUBenchmarkHarness(
                model_specs=[
                    {"name": s["name"], "hf_id": s.get("hf_id", s["name"])}
                    for s in self.model_specs
                ]
            )
            harness.verify_gpu_assignments()
            self._dual_runner = harness
            print(f"[Exp {EXPERIMENT}] DualGPURunner wired:")
            for i, spec in enumerate(self.model_specs):
                print(f"  GPU {spec.get('gpu', i)}: {spec['name']}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Exp {EXPERIMENT}] DualGPURunner unavailable ({exc}) "
                "— proceeding without GPU runner"
            )

    # ------------------------------------------------------------------
    # Startup diagnostics (mirrors Exp 294)
    # ------------------------------------------------------------------

    def _run_startup_diagnostics(self) -> None:
        """Capture VRAM readings at script start to diagnose GPU stall root cause.

        Both RTX 3090s should show ≥ 20 GiB free at startup.  If either GPU
        shows < 2 GiB free, that confirms residual VRAM from a previous run.
        """
        self._stall_diagnosis = _query_vram_gb()
        print(
            f"[Exp {EXPERIMENT}] GPU diagnostics at startup:\n"
            f"  GPU 0 free VRAM: {self._stall_diagnosis.get('vram_gpu0_free_gb', 0):.1f} GiB\n"
            f"  GPU 1 free VRAM: {self._stall_diagnosis.get('vram_gpu1_free_gb', 0):.1f} GiB"
        )

    # ------------------------------------------------------------------
    # Pre-warm phase (REQ-VERIFY-079)
    # ------------------------------------------------------------------

    def _run_prewarm_phase(self) -> None:
        """Load each model and run a health-check prompt before any benchmarking.

        This is the core fix for the GPU stall (diagnosed in Exp 294).  By explicitly
        loading models here, the per-question ``generate_fn`` closure never pays the
        cold-load penalty.

        For each model spec this calls :func:`model_prewarm` and stores the
        :class:`PrewarmResult`.  If any model fails its health-check we print a
        warning — benchmarking proceeds anyway (the generate_fn will hit the 60 s
        timeout on the first question if the GPU is truly broken, producing a
        partial artifact with stall_at set).
        """
        for spec in self.model_specs:
            name = spec["name"]
            hf_id = spec.get("hf_id", name)
            gpu_id = spec.get("gpu", 0)
            print(f"[Exp {EXPERIMENT}] Pre-warming {name} on GPU {gpu_id}…")
            result = model_prewarm(
                name,
                hf_id,
                gpu_id,
                timeout_seconds=PREWARM_LOAD_TIMEOUT_SECONDS,
            )
            self._pre_warm_results[name] = result
            status = "OK" if result.health_ok else f"FAILED ({result.stall_root_cause})"
            print(f"  {name}: {status}  load_time={result.load_time_s:.1f}s")

    # ------------------------------------------------------------------
    # Verifier loading (lazy, mock-safe)
    # ------------------------------------------------------------------

    def _ensure_verifiers(self) -> None:
        """Load semantic grounding and formal claim verifiers if not yet loaded.

        Verifiers are only imported when needed (verify_only / verify_repair
        modes) so the module can be loaded without triggering carnot imports in
        pure baseline runs or mock tests.
        """
        if self._verifiers_loaded:
            return
        try:
            from carnot.pipeline.semantic_grounding import SemanticGroundingVerifier
            from carnot.pipeline.formal_claim_verifier import FormalClaimVerifier

            self._semantic_grounder = SemanticGroundingVerifier()
            self._formal_claim_verifier = FormalClaimVerifier()
        except ImportError:
            pass
        self._verifiers_loaded = True

    # ------------------------------------------------------------------
    # Default generate fn (live GPU mode — reuses pre-warm cache)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_generate_fn() -> Callable[..., tuple[str, np.ndarray]]:
        """Return the live GPU inference callable.

        In mock mode (``CARNOT_FORCE_LIVE=0``) returns a simple stub that always
        produces the correct answer.  In live mode loads the transformers pipeline
        and returns real logits, re-using the ``_model_cache`` populated during
        pre-warm so the first question never triggers a fresh model load.

        Returns:
            Callable ``(question, expected_answer, *, model_name, mode, variant_type)
            -> (response_text, logits_ndarray)`` where logits has shape
            ``(1, seq_len, vocab_size)``.
        """
        if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
            def _mock(question: str, expected_answer: int, **kw: Any) -> tuple[str, np.ndarray]:
                """Mock inference: always returns the correct answer."""
                logits = np.zeros((1, 8, 100), dtype=np.float32)
                return str(expected_answer), logits
            return _mock

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _model_cache: dict[str, Any] = {}

            def _live(
                question: str,
                expected_answer: int,
                *,
                model_name: str = "Qwen3.5-0.8B",
                mode: str = "baseline",
                variant_type: str = "number_swap",
                **kw: Any,
            ) -> tuple[str, np.ndarray]:
                """Live GPU inference: greedy decode with logit capture.

                Model is loaded into ``_model_cache`` on the first call (or re-used
                if already cached by the pre-warm phase).
                """
                hf_id = next(
                    (s.get("hf_id", s["name"]) for s in MODEL_SPECS if s["name"] == model_name),
                    model_name,
                )
                gpu_id = next(
                    (s.get("gpu", 0) for s in MODEL_SPECS if s["name"] == model_name),
                    0,
                )
                device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

                if hf_id not in _model_cache:
                    tokenizer = AutoTokenizer.from_pretrained(hf_id)
                    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
                    model = model.to(device).eval()
                    _model_cache[hf_id] = (model, tokenizer)

                model, tokenizer = _model_cache[hf_id]
                prompt = (
                    f"Solve the following math problem step by step. "
                    f"State the final answer as a single number.\n\n"
                    f"Problem: {question}\n\nAnswer:"
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                if output.scores:
                    logits_2d = torch.stack(output.scores, dim=0).cpu().float().numpy()
                    logits = np.expand_dims(logits_2d, axis=0)
                else:
                    logits = np.zeros((1, 1, model.config.vocab_size), dtype=np.float32)

                return response, logits

            return _live

        except ImportError as exc:
            raise RuntimeError(
                f"[Exp {EXPERIMENT}] Live inference requires 'torch' and 'transformers' "
                f"but import failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _build_variant_questions(self, variant_type: str) -> list[dict[str, Any]]:
        """Build the question list for one variant type from the Exp 281 rows.

        For each variant type, returns the variant questions and answers.
        Only rows whose ``variant_type`` field matches are included.

        Args:
            variant_type: One of VARIANT_TYPES (``"number_swap"`` or
                          ``"irrelevant_sentence"``).

        Returns:
            List of ``{"question_id", "question", "expected_answer"}`` dicts.
        """
        return [
            {
                "question_id": row["question_id"],
                "question": row["variant_question"],
                "expected_answer": row["variant_answer"],
            }
            for row in self.rows
            if row.get("variant_type") == variant_type
        ]

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _run_verifiers(
        self, *, question: str, response: str
    ) -> tuple[bool, bool, bool]:
        """Run semantic grounding and formal claim verifiers.

        Args:
            question: The question text.
            response: The model's current response text.

        Returns:
            Tuple of (violation_detected, semantic_grounding_fired, formal_claim_fired).
        """
        self._ensure_verifiers()
        semantic_grounding_fired = False
        formal_claim_fired = False
        violation_detected = False

        if self._semantic_grounder is not None:
            try:
                sg_result = self._semantic_grounder.verify(question, response, typed_reasoning=None)
                if not sg_result.verified:
                    semantic_grounding_fired = True
                    violation_detected = True
            except Exception:  # noqa: BLE001 — verifier failure is non-fatal
                pass

        if self._formal_claim_verifier is not None:
            try:
                numeric_answer = _extract_numeric_answer(response)
                if numeric_answer is not None:
                    claim = {
                        "type": "arithmetic",
                        "operands": [],
                        "result": numeric_answer,
                        "operator": "identity",
                    }
                    result = self._formal_claim_verifier.verify_batch([claim])
                    if result.any_failed:
                        formal_claim_fired = True
                        violation_detected = True
            except Exception:  # noqa: BLE001
                pass

        return violation_detected, semantic_grounding_fired, formal_claim_fired

    def _repair_response(
        self,
        *,
        question: str,
        response: str,
        expected_answer: int,
        model_name: str,
        variant_type: str,
    ) -> tuple[str, np.ndarray]:
        """Run one repair iteration: re-generate with violation feedback in the prompt.

        This simplified repair loop appends a violation notice to the original
        question and regenerates.  A production pipeline would use the full
        VerifyRepairPipeline with constraint-guided repair prompts.

        Args:
            question: Original question text.
            response: Current (incorrect) model response.
            expected_answer: Ground truth (used for logit saving, not cheating).
            model_name: Model label for dispatch.
            variant_type: Variant type for dispatch.

        Returns:
            Tuple of (repaired_response, logits).
        """
        repair_prompt = (
            f"{question}\n\n"
            f"Your previous answer was: {response}\n"
            f"Please check your arithmetic carefully and provide the correct answer."
        )
        return self.generate_fn(
            repair_prompt,
            expected_answer,
            model_name=model_name,
            mode="verify_repair",
            variant_type=variant_type,
        )

    # ------------------------------------------------------------------
    # Core inference loop (per-cell)
    # ------------------------------------------------------------------

    def run_mode_variant(
        self,
        *,
        model_name: str,
        mode: str,
        variant_type: str,
    ) -> list[dict[str, Any]]:
        """Run inference for one (model, mode, variant_type) cell.

        Implements:
        - Checkpoint resume (REQ-VERIFY-068)
        - Logit saving at prefix fractions (REQ-VERIFY-070)
        - Verification for verify_only / verify_repair modes (REQ-VERIFY-068)
        - Iterative repair for verify_repair mode (REQ-VERIFY-068)
        - pre_warm_verified field in each record (SCENARIO-VERIFY-108)
        - logit_path field in each record (SCENARIO-VERIFY-106)

        Args:
            model_name: Human-readable model label.
            mode: One of MODES.
            variant_type: One of VARIANT_TYPES.

        Returns:
            List of per-question result dicts.

        Raises:
            TimeoutError: Propagated from generate_fn on hard timeout.
        """
        questions = self._build_variant_questions(variant_type)
        total = len(questions)

        ckpt_path = _ckpt_path(
            self.checkpoint_dir, model_name=model_name, mode=mode, variant_type=variant_type
        )
        ckpt = _load_ckpt(ckpt_path)
        completed: dict[str, Any] = dict(ckpt.get("completed", {}))

        results: list[dict[str, Any]] = []
        # Logit buffer: list of (seq_len, vocab_size) arrays for prefix-fraction saving.
        logit_buffer: list[np.ndarray] = []
        # Tracks which prefix fractions have already been saved.
        saved_fractions: set[int] = set()
        # Logit path index for this cell.
        cell_logit_paths: dict[str, str] = {}

        # Whether this model passed the pre-warm health check (False in mock mode).
        pre_warm_verified: bool = (
            self._pre_warm_results.get(model_name, None) is not None
            and self._pre_warm_results[model_name].health_ok
        )

        for idx, item in enumerate(questions):
            qid = item["question_id"]
            question = item["question"]
            expected = item["expected_answer"]

            if qid in completed:
                # Resume: reconstruct from checkpoint without calling generate_fn.
                results.append(completed[qid])
                continue

            # ---- Baseline generation (all modes) ----
            t0 = time.perf_counter()
            response, logits = self.generate_fn(
                question, expected, model_name=model_name, mode=mode, variant_type=variant_type
            )
            elapsed = time.perf_counter() - t0

            # ---- Verification (verify_only and verify_repair) ----
            violation_detected = False
            semantic_grounding_fired = False
            formal_claim_fired = False
            repaired = False

            if mode in ("verify_only", "verify_repair"):
                violation_detected, semantic_grounding_fired, formal_claim_fired = (
                    self._run_verifiers(question=question, response=response)
                )

            # ---- Repair (verify_repair only, when violation found) ----
            if mode == "verify_repair" and violation_detected:
                try:
                    repaired_response, _repair_logits = self._repair_response(
                        question=question,
                        response=response,
                        expected_answer=expected,
                        model_name=model_name,
                        variant_type=variant_type,
                    )
                    response = repaired_response
                    repaired = True
                except TimeoutError:
                    repaired = False
                    raise  # Propagate so caller emits partial artifact.

            correct = _is_correct(response, expected)
            result: dict[str, Any] = {
                "question_id": qid,
                "mode": mode,
                "variant_type": variant_type,
                "model": model_name,
                "correct": correct,
                "response": response,
                "violation_detected": violation_detected,
                "repaired": repaired,
                "logit_path": None,  # Updated below when fraction file is saved.
                "semantic_grounding_fired": semantic_grounding_fired,
                "formal_claim_fired": formal_claim_fired,
                "pre_warm_verified": pre_warm_verified,
                "elapsed_s": round(elapsed, 3),
            }
            results.append(result)
            completed[qid] = result

            # Accumulate logits (strip batch dim: (1, seq_len, vocab) → (seq_len, vocab)).
            if logits.ndim == 3:
                logit_buffer.append(logits[0])
            else:
                logit_buffer.append(logits)

            # Checkpoint every CHECKPOINT_INTERVAL questions.
            n_done = idx + 1
            if n_done % CHECKPOINT_INTERVAL == 0 or n_done == total:
                _save_ckpt(ckpt_path, {
                    "model_name": model_name,
                    "mode": mode,
                    "variant_type": variant_type,
                    "completed": completed,
                })

            # Save logits at prefix fractions (REQ-VERIFY-070 / SCENARIO-VERIFY-106).
            if total > 0:
                progress = n_done / total
                for frac in LOGIT_FRACTIONS:
                    pct = int(round(frac * 100))
                    if pct not in saved_fractions and progress >= frac:
                        out_path = _save_logits(
                            self.logit_dir,
                            model_name=model_name,
                            mode=mode,
                            variant_type=variant_type,
                            pct=pct,
                            logit_list=logit_buffer,
                        )
                        cell_logit_paths[f"{pct}pct"] = out_path
                        saved_fractions.add(pct)

                        # Back-fill logit_path on all results that don't have one yet.
                        for r in results:
                            if r.get("logit_path") is None:
                                r["logit_path"] = out_path
                                completed[r["question_id"]]["logit_path"] = out_path

        # Store logit paths for this cell.
        self._logit_paths.setdefault(model_name, {}).update({
            f"{mode}__{variant_type}__{k}": v
            for k, v in cell_logit_paths.items()
        })

        return results

    # ------------------------------------------------------------------
    # Per-model aggregation
    # ------------------------------------------------------------------

    def run_all_modes(self, *, model_name: str) -> dict[str, dict[str, Any]]:
        """Run all modes and variant types for one model.

        Args:
            model_name: Human-readable model label.

        Returns:
            Nested dict ``{mode: {variant_type: {correct, total, accuracy,
            violation_detected_count, repaired_count}}}``.

        Raises:
            TimeoutError: Propagated from run_mode_variant on inference timeout.
        """
        mode_stats: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            variant_stats: dict[str, Any] = {}
            for vt in VARIANT_TYPES:
                records = self.run_mode_variant(
                    model_name=model_name, mode=mode, variant_type=vt
                )
                n_correct = sum(1 for r in records if r.get("correct"))
                n_total = len(records)
                n_violations = sum(1 for r in records if r.get("violation_detected"))
                n_repaired = sum(1 for r in records if r.get("repaired"))
                variant_stats[vt] = {
                    "correct": n_correct,
                    "total": n_total,
                    "accuracy": n_correct / n_total if n_total else 0.0,
                    "violation_detected_count": n_violations,
                    "repaired_count": n_repaired,
                }
            mode_stats[mode] = variant_stats
        self._cell_results[model_name] = mode_stats
        return mode_stats

    def run_all(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Run all configured models across all modes and variants.

        Returns:
            Nested dict ``{model_name: mode_stats}`` (see ``run_all_modes``).

        Raises:
            TimeoutError: Propagated from run_all_modes on inference timeout.
        """
        for spec in self.model_specs:
            self.run_all_modes(model_name=spec["name"])
        return dict(self._cell_results)

    # ------------------------------------------------------------------
    # Timeout-aware top-level entry point
    # ------------------------------------------------------------------

    def run_with_timeout_handling(
        self,
        *,
        output_path: Path,
        run_date: str = RUN_DATE,
        inference_mode: str | None = None,
    ) -> dict[str, Any]:
        """Run all cells, writing a partial artifact on timeout (REQ-VERIFY-071).

        Args:
            output_path: Destination JSON path for the experiment artifact.
            run_date: Date string in ``YYYYMMDD`` format.
            inference_mode: Override for the artifact field.

        Returns:
            The artifact dict (may be partial if a timeout occurred).
        """
        if inference_mode is None:
            inference_mode = (
                "live_gpu" if os.environ.get("CARNOT_FORCE_LIVE", "0") == "1" else "mock"
            )

        started_at = utc_now()
        stall_at: str | None = None

        try:
            for spec in self.model_specs:
                model_name = spec["name"]
                for mode in MODES:
                    for vt in VARIANT_TYPES:
                        questions = self._build_variant_questions(vt)
                        for item in questions:
                            qid = item["question_id"]
                            ckpt_path = _ckpt_path(
                                self.checkpoint_dir,
                                model_name=model_name,
                                mode=mode,
                                variant_type=vt,
                            )
                            ckpt = _load_ckpt(ckpt_path)
                            if qid in ckpt.get("completed", {}):
                                continue
                            question = item["question"]
                            expected = item["expected_answer"]
                            try:
                                self.generate_fn(
                                    question, expected,
                                    model_name=model_name,
                                    mode=mode,
                                    variant_type=vt,
                                )
                            except TimeoutError:
                                stall_at = f"{model_name}:{mode}:{vt}:{qid}"
                                raise
        except TimeoutError:
            pass  # Partial run — fall through to artifact construction.

        # Run the full benchmark (fast-path above handles early-exit detection only).
        if stall_at is None:
            try:
                self.run_all()
            except TimeoutError as exc:
                stall_at = str(exc)

        finished_at = utc_now()

        # Load comparison references (Exp 294 baseline, Exp 235 semantic v2).
        comparison_refs = _load_comparison_refs()

        # Compute improvement deltas using Exp 294 standard baseline if available.
        baseline_standard_acc: dict[str, float] = {}
        for spec in self.model_specs:
            mn = spec["name"]
            ref294 = comparison_refs.get("exp294", {}).get("model_results", {}).get(mn, {})
            std_entry = ref294.get("standard", {})
            if std_entry.get("accuracy") is not None:
                baseline_standard_acc[mn] = std_entry["accuracy"]

        improvement_deltas = compute_improvement_deltas(
            self._cell_results,
            baseline_standard_acc=baseline_standard_acc or None,
        )
        criterion_met = _primary_criterion_met(improvement_deltas)

        # Build pre-warm summary dicts.
        pre_warm_status = {
            name: result.health_ok
            for name, result in self._pre_warm_results.items()
        }
        pre_warm_time_s = {
            name: result.load_time_s
            for name, result in self._pre_warm_results.items()
        }

        artifact = build_artifact(
            run_date=run_date,
            started_at=started_at,
            finished_at=finished_at,
            inference_mode=inference_mode,
            cell_results=self._cell_results,
            logit_paths=self._logit_paths,
            improvement_deltas=improvement_deltas,
            primary_criterion_met=criterion_met,
            stall_at=stall_at,
            comparison_refs=comparison_refs,
            pre_warm_status=pre_warm_status,
            pre_warm_time_s=pre_warm_time_s,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        tmp.replace(output_path)

        return artifact


# ---------------------------------------------------------------------------
# Comparison reference loader
# ---------------------------------------------------------------------------


def _load_comparison_refs() -> dict[str, Any]:
    """Load prior experiment artifacts for comparison (REQ-VERIFY-069).

    Loads Exp 294 (Apple pre-warm baseline) and Exp 235 (semantic v2 same cohort).
    Missing files are silently skipped.

    Returns:
        Dict with keys ``"exp294"`` and ``"exp235"`` where each value is the
        loaded artifact dict or ``{}`` if not found.
    """
    refs: dict[str, Any] = {}
    for key, path in (
        ("exp294", _EXP294_RESULTS),
        ("exp235", _EXP235_RESULTS),
    ):
        if path.exists():
            try:
                refs[key] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                refs[key] = {}
        else:
            refs[key] = {}
    return refs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp 295.

    Environment variables
    ---------------------
    CARNOT_FORCE_LIVE : "1" enables live GPU execution (default "0" = mock).
    JAX_PLATFORMS     : Set to "cpu" to keep JAX off the GPU.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Exp 295: Verify-repair benchmark on Apple adversarial GSM8K corpus "
            "(pre-warm fix)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DATASET_PATH,
        help="Path to gsm8k_adversarial_281.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_CHECKPOINT_BASE,
        help="Directory for incremental checkpoint files.",
    )
    parser.add_argument(
        "--logit-dir",
        type=Path,
        default=_LOGIT_BASE,
        help="Directory for logit .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help="Path for the result artifact JSON.",
    )
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[Exp {EXPERIMENT}] Loaded {len(rows)} rows from {args.dataset}")

    runner = VerifyRepairRunner295(
        rows=rows,
        checkpoint_dir=args.checkpoint_dir,
        logit_dir=args.logit_dir,
    )

    print(
        f"[Exp {EXPERIMENT}] Running 12-cell benchmark "
        "(3 modes × 2 variants × 2 models)…"
    )
    artifact = runner.run_with_timeout_handling(
        output_path=args.output,
        run_date=RUN_DATE,
    )

    print(f"\n[Exp {EXPERIMENT}] Result written to {args.output}")
    print(f"  partial: {artifact.get('partial')}")
    print(f"  primary_criterion_met: {artifact.get('primary_criterion_met')}")
    print(f"  pre_warm_status: {artifact.get('pre_warm_status')}")
    if artifact.get("stall_at"):
        print(f"  stall_at: {artifact['stall_at']}")
    for model_name, mode_stats in artifact.get("cell_results", {}).items():
        print(f"\n  Model: {model_name}")
        for mode in MODES:
            for vt in VARIANT_TYPES:
                cell = mode_stats.get(mode, {}).get(vt, {})
                acc = cell.get("accuracy", 0.0)
                print(f"    [{mode:12s}][{vt:21s}] accuracy={acc:.3f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
