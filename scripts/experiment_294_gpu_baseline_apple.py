#!/usr/bin/env python3
"""Experiment 294: GPU stall diagnosis + Apple adversarial baseline re-run.

Root-cause analysis and fix for the recurring GPU stall that left Exps 282/283
INCONCLUSIVE for two consecutive milestones (2026.04.20, 2026.04.21).

Stall root cause (diagnosed): ``AutoModelForCausalLM.from_pretrained()`` is called
*inside* the per-question inference closure (Exp 282's ``_default_generate_fn``).
The first question of each (model, variant) pair triggers a full model download +
VRAM transfer while the 60 s inference-timeout clock is already running.  On a cold
filesystem cache (conductor runs start clean) this routinely exceeds the timeout,
yielding 0% GPU utilisation at milestone end.

The fix (implemented here):
  1. **Explicit pre-warm** — ``model_prewarm()`` loads each model onto its assigned
     GPU before the timed benchmark loop starts, then runs a single health-check
     prompt to confirm the model responds.
  2. **Health-check gating** — if either model fails its health-check, we emit a
     ``blocked`` partial artifact with ``stall_root_cause`` set and abort.
  3. **Warm generate_fn** — the per-question ``generate_fn`` re-uses the already-
     loaded model from ``_model_cache`` (same as Exp 282) but because the cache is
     populated during pre-warm, the first question never pays the load penalty.

This script is a self-contained re-run of the Exp 282 benchmark with the pre-warm
fix applied.  All checkpoint / logit / artifact paths use the ``294`` prefix to
avoid colliding with existing Exp 282 artefacts.

Usage (live GPU, requires CARNOT_FORCE_LIVE=1):
    CARNOT_FORCE_LIVE=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/experiment_294_gpu_baseline_apple.py

Usage (mock / unit mode — no GPU required):
    CARNOT_FORCE_LIVE=0 .venv/bin/pytest \\
        tests/python/test_experiment_294_gpu_baseline_apple.py -q --no-cov -n0

Spec: REQ-VERIFY-079, REQ-VERIFY-064, REQ-VERIFY-065, REQ-VERIFY-066, REQ-VERIFY-067,
      SCENARIO-VERIFY-101, SCENARIO-VERIFY-102
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

EXPERIMENT: int = 294
"""Experiment number — matches the filename and artifact ``experiment`` field."""

RUN_DATE: str = "20260414"
"""Wall-clock date of this experiment run."""

CHECKPOINT_INTERVAL: int = 10
"""Number of questions processed before each checkpoint write (REQ-VERIFY-065)."""

INFERENCE_TIMEOUT_SECONDS: int = 60
"""Hard per-call timeout in seconds (REQ-VERIFY-066).  If exceeded, a partial
artifact is emitted with a ``stall_at`` field identifying where inference stalled."""

PREWARM_LOAD_TIMEOUT_SECONDS: int = 300
"""Timeout for the model-load phase of the pre-warm health-check (REQ-VERIFY-079).
If loading takes longer than this, ``stall_root_cause`` is set to ``'lazy_load_stall'``.

2026-04-17: bumped from 15s to 300s. The original 15s was calibrated for the
small-model prewarm scenario where weights were already in page cache. On cold
first-run with 4B-parameter models like gemma-4-E4B-it, weight load + CUDA init
takes 30-120s (per the docstring of model_prewarm itself). The old 15s value
caused every live GPU experiment in milestone 2026.04.33 to fail at prewarm,
cascading into 7+ consecutive scaffolding-only deliverables. 300s matches the
check_model_loadable timeout set earlier (python/carnot/pipeline/live_gpu_diagnostic.py).
"""

PREWARM_HEALTH_TIMEOUT_SECONDS: int = 60
"""Timeout for the health-check prompt phase (REQ-VERIFY-079).

2026-04-17: bumped from 10s to 60s. First inference call on a freshly-loaded
model compiles CUDA kernels and can take 10-30s before any token is produced.
The old 10s was too tight for larger models."""

PREWARM_WARMUP_PROMPTS: int = 2
"""Number of warm-up prompts run per model after the initial health-check to fully
populate CUDA compilation caches and reduce first-batch latency."""

LOGIT_FRACTIONS: list[float] = [0.25, 0.50, 0.75, 1.00]
"""Prefix fractions at which accumulated logit tensors are saved to disk (REQ-VERIFY-067).

Files are written to ``data/research/logits_294_{model_slug}_{variant}_{pct}pct.npy``.
"""

ARTIFACT_SCHEMA: list[str] = [
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "inference_mode",
    "model_results",
    "logit_paths",
    "partial",
    "stall_at",
    "apple_2410_05229_check",
    "stall_diagnosis",
    "pre_warm_status",
    "pre_warm_time_s",
]
"""Required top-level fields in the artifact JSON (SCENARIO-VERIFY-080, REQ-VERIFY-079)."""

# Default model pair — Qwen on GPU 0, Gemma on GPU 1 (mirrors Exp 282 MODEL_SPECS).
MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
]

# Minimum accuracy drop (pp) that constitutes a positive result for the
# Apple 2410.05229 number_swap hypothesis (§4 predicts ≥ 15 pp drop).
_APPLE_DROP_THRESHOLD_PP: float = 15.0

_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "research" / "gsm8k_adversarial_281.jsonl"
)
_CHECKPOINT_BASE = (
    Path(__file__).resolve().parents[1] / "results" / "checkpoints" / "experiment_294"
)
_LOGIT_BASE = Path(__file__).resolve().parents[1] / "data" / "research"
_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "experiment_294_results.json"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_SLUG_KEEP = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


def safe_slug(text: str) -> str:
    """Convert *text* into a filesystem-safe lower-case slug.

    Spaces and path separators become underscores; any other character outside
    ``[a-z0-9_-]`` is replaced with an underscore.

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
        True if the extracted answer matches the expected answer within 1e-6.
    """
    pred = _extract_numeric_answer(response)
    if pred is None:
        return False
    return abs(pred - float(expected_answer)) < 1e-6


# ---------------------------------------------------------------------------
# Pre-warm result dataclass (REQ-VERIFY-079)
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


# ---------------------------------------------------------------------------
# model_prewarm — explicit GPU pre-warm with health-check (REQ-VERIFY-079)
# ---------------------------------------------------------------------------


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

    This function is the core fix for the GPU stall diagnosed in Exps 282/283.
    By calling it before the timed benchmark loop, the VRAM transfer and CUDA
    compilation caches are fully warm before the first timed inference call.

    The fix addresses the root cause: Exp 282's ``_default_generate_fn`` loaded
    models lazily inside the per-question closure.  On a cold filesystem cache
    (typical in conductor runs), ``from_pretrained()`` could take 30–120 s, which
    exhausted the 60 s inference timeout on the very first question, leaving both
    GPUs idle for the rest of the milestone.

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
    stall_root_cause: str | None = None

    def _do_prewarm() -> tuple[float, bool, str | None]:
        """Run load + health-check in the calling thread; raises on error."""
        nonlocal load_time_s

        # --- Step 1: load model ---
        t_load_start = time.perf_counter()
        if load_fn is not None:
            model, tokenizer = load_fn(hf_id, gpu_id)
        else:
            # Live GPU path: load via transformers.
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16)
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()

        load_time_s = time.perf_counter() - t_load_start

        # --- Step 2: health-check prompt ---
        if generate_fn is not None:
            response = generate_fn(model, tokenizer, health_prompt)
        else:
            # Live GPU path: greedy decode with short output.
            import torch

            device = next(model.parameters()).device
            inputs = tokenizer(health_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            generated_ids = out[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        health_ok = bool(response and response.strip())
        return load_time_s, health_ok, None

    # Run in a thread so we can enforce a wall-clock timeout.
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
            # Force-cancel — best-effort; Python threads can't be truly killed.
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
            # torch.cuda.OutOfMemoryError is a subclass of RuntimeError, but
            # we also catch the generic MemoryError for safety.
            load_time_s = time.perf_counter() - t0
            return PrewarmResult(
                model_name=model_name,
                gpu_id=gpu_id,
                load_time_s=load_time_s,
                health_ok=False,
                stall_root_cause="cuda_oom",
            )
        except RuntimeError as exc:
            # torch.cuda.OutOfMemoryError is a RuntimeError subclass.
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
# GPU diagnostics helper
# ---------------------------------------------------------------------------


def _query_vram_gb() -> dict[str, float]:
    """Query free VRAM on both GPUs via nvidia-smi.

    Returns a dict with keys ``vram_gpu0_free_gb`` and ``vram_gpu1_free_gb``.
    Values are 0.0 if nvidia-smi is unavailable or fails.
    """
    out: dict[str, float] = {"vram_gpu0_free_gb": 0.0, "vram_gpu1_free_gb": 0.0}
    try:
        # nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
        # Output: one line per GPU, value in MiB.
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
# Checkpoint helpers (mirrors Exp 282)
# ---------------------------------------------------------------------------


def _ckpt_path(checkpoint_dir: Path, *, model_name: str, variant_type: str) -> Path:
    """Return the checkpoint file path for a given (model, variant_type) pair.

    Args:
        checkpoint_dir: Base directory for checkpoint files.
        model_name: Human-readable model label (e.g. ``"Qwen3.5-0.8B"``).
        variant_type: One of ``"standard"``, ``"number_swap"``, ``"irrelevant_sentence"``.

    Returns:
        Absolute path to the ``.json`` checkpoint file.
    """
    return checkpoint_dir / f"{safe_slug(model_name)}__{safe_slug(variant_type)}.json"


def _load_ckpt(path: Path) -> dict[str, Any]:
    """Load a checkpoint file if it exists; return an empty structure otherwise."""
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
    """Write a checkpoint file atomically using a ``.tmp`` rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Logit saving helpers (mirrors Exp 282, using 294 filename prefix)
# ---------------------------------------------------------------------------


def _logit_npy_path(logit_dir: Path, *, model_name: str, variant_type: str, pct: int) -> Path:
    """Return the .npy path for the logit tensor at a given prefix fraction.

    File naming: ``logits_294_{model_slug}_{variant}_{pct}pct.npy``
    """
    fname = f"logits_294_{safe_slug(model_name)}_{safe_slug(variant_type)}_{pct}pct.npy"
    return logit_dir / fname


def _save_logits(
    logit_dir: Path,
    *,
    model_name: str,
    variant_type: str,
    pct: int,
    logit_list: list[np.ndarray],
) -> str:
    """Stack *logit_list* into a 1-D object array and save as .npy.

    Each element of *logit_list* is a ``(seq_len, vocab_size)`` array from one
    question.  We use an object array so variable ``seq_len`` values are allowed.

    Returns:
        String path of the saved file (for inclusion in the artifact).
    """
    logit_dir.mkdir(parents=True, exist_ok=True)
    out_path = _logit_npy_path(logit_dir, model_name=model_name, variant_type=variant_type, pct=pct)
    arr = np.empty(len(logit_list), dtype=object)
    for i, logits in enumerate(logit_list):
        arr[i] = logits[0] if logits.ndim == 3 else logits
    np.save(str(out_path), arr, allow_pickle=True)
    return str(out_path)


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_artifact(
    *,
    run_date: str,
    started_at: str,
    finished_at: str,
    inference_mode: str,
    model_results: dict[str, dict[str, Any]],
    logit_paths: dict[str, dict[str, str]],
    stall_at: str | None,
    stall_diagnosis: dict[str, Any],
    pre_warm_status: dict[str, bool],
    pre_warm_time_s: dict[str, float],
) -> dict[str, Any]:
    """Build the Exp 294 result artifact dict.

    Extends the Exp 282 artifact with three new fields:
    - ``stall_diagnosis``: VRAM readings + nvidia-smi output at script start.
    - ``pre_warm_status``: Per-model bool indicating health-check passed.
    - ``pre_warm_time_s``: Per-model wall-clock seconds for the pre-warm.

    Args:
        run_date: Date string in ``YYYYMMDD`` format.
        started_at: ISO-8601 UTC start timestamp.
        finished_at: ISO-8601 UTC finish timestamp.
        inference_mode: ``"live_gpu"`` or ``"mock"``.
        model_results: Nested dict ``{model_name: {variant_type: {correct, total, accuracy}}}``.
        logit_paths: Nested dict ``{model_name: {fraction_label: npy_path}}``.
        stall_at: ``None`` for a complete run; otherwise ``"model:variant:question_id"``.
        stall_diagnosis: Dict of VRAM readings and GPU state at start (from ``_query_vram_gb``).
        pre_warm_status: Dict ``{model_name: health_ok}`` from pre-warm phase.
        pre_warm_time_s: Dict ``{model_name: load_time_s}`` from pre-warm phase.

    Returns:
        JSON-serialisable artifact dict containing all fields in ``ARTIFACT_SCHEMA``.
    """
    is_partial = stall_at is not None

    # Apple 2410.05229 hypothesis check per model.
    apple_check: dict[str, Any] = {}
    for model_name, variants in model_results.items():
        std_entry = variants.get("standard", {})
        ns_entry = variants.get("number_swap", {})
        std_acc = std_entry.get("accuracy")
        ns_acc = ns_entry.get("accuracy")
        if std_acc is not None and ns_acc is not None:
            drop_pp = (std_acc - ns_acc) * 100.0
            apple_check[model_name] = {
                "standard_accuracy": std_acc,
                "number_swap_accuracy": ns_acc,
                "drop_pp": round(drop_pp, 2),
                "hypothesis_confirmed": drop_pp >= _APPLE_DROP_THRESHOLD_PP,
            }

    return {
        "experiment": EXPERIMENT,
        "schema": "carnot.apple_baseline.v2",
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "inference_mode": inference_mode,
        "model_results": model_results,
        "logit_paths": logit_paths,
        "partial": is_partial,
        "stall_at": stall_at,
        "apple_2410_05229_check": apple_check,
        "stall_diagnosis": stall_diagnosis,
        "pre_warm_status": pre_warm_status,
        "pre_warm_time_s": pre_warm_time_s,
    }


# ---------------------------------------------------------------------------
# AppleBaselineRunner294 — pre-warm-aware runner (REQ-VERIFY-079)
# ---------------------------------------------------------------------------


class AppleBaselineRunner294:
    """Run baseline inference on the Apple adversarial GSM8K corpus with GPU pre-warm.

    This class mirrors :class:`~experiment_282_apple_baseline_gpu.AppleBaselineRunner`
    from Exp 282 but adds:

    1. **model_prewarm() phase** — called at construction for each model in live mode,
       before any timed benchmark work starts.  Both GPU slots are loaded + health-checked
       before the first question is processed.

    2. **stall_diagnosis** — VRAM readings from ``nvidia-smi`` are captured at startup
       so we can confirm GPUs are idle (≥ 20 GiB free) before proceeding.

    3. **pre_warm_status / pre_warm_time_s** — included in the final artifact so analysts
       can see whether pre-warm succeeded and how long it took.

    Args:
        rows: List of dicts loaded from ``gsm8k_adversarial_281.jsonl``.
        model_specs: Ordered list of ``{"name": ..., "hf_id": ..., "gpu": int}`` dicts.
        generate_fn: Callable ``(question, expected_answer, *, model_name, variant_type)
                     -> (response_str, logit_ndarray)``.  Injected in tests.
        checkpoint_dir: Directory for per-(model, variant) checkpoint files.
        logit_dir: Directory where logit .npy files are saved.
        timeout_seconds: Per-call hard timeout in seconds (default 60 s).
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

        # Collected across run: {model_name: {variant: {…}}}
        self._model_results: dict[str, dict[str, Any]] = {}
        # Logit paths: {model_name: {fraction_label: path}}
        self._logit_paths: dict[str, dict[str, str]] = {}

        # Pre-warm results accumulated during __init__ (live mode only).
        self._pre_warm_results: dict[str, PrewarmResult] = {}

        # GPU stall diagnosis gathered at startup.
        self._stall_diagnosis: dict[str, Any] = {}

        if os.environ.get("CARNOT_FORCE_LIVE", "0") == "1":
            self._run_startup_diagnostics()
            self._run_prewarm_phase()

    # ------------------------------------------------------------------
    # Startup diagnostics
    # ------------------------------------------------------------------

    def _run_startup_diagnostics(self) -> None:
        """Capture VRAM readings at script start to diagnose GPU stall root cause.

        Both RTX 3090s should show ≥ 20 GiB free at startup.  If either GPU
        shows < 2 GiB free the conductor's stall hypothesis (residual VRAM from
        a previous run) is confirmed.
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

        This is the core fix for the GPU stall.  By explicitly loading models here,
        the per-question ``generate_fn`` closure never pays the load penalty.

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
    # Default generate fn (live GPU mode — re-uses pre-warm cache)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_generate_fn() -> Callable[..., tuple[str, np.ndarray]]:
        """Return the live GPU inference function.

        In live mode (``CARNOT_FORCE_LIVE=1``) this loads models the same way as
        Exp 282 but relies on ``_model_cache`` being populated by ``_run_prewarm_phase()``
        at construction time, so the first question never triggers a fresh load.

        In mock mode (``CARNOT_FORCE_LIVE=0``) returns the expected answer with a
        dummy logit tensor.
        """
        if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
            def _mock(question: str, expected_answer: int, **kw: Any) -> tuple[str, np.ndarray]:
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
                variant_type: str = "standard",
                **kw: Any,
            ) -> tuple[str, np.ndarray]:
                """Live GPU inference with greedy decode and logit capture.

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
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_id, torch_dtype=torch.float16
                    )
                    model = model.to(device).eval()
                    _model_cache[hf_id] = (model, tokenizer)

                model, tokenizer = _model_cache[hf_id]
                prompt = (
                    "Solve the following math problem step by step. "
                    "State the final answer as a single number.\n\n"
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
                f"[Exp {EXPERIMENT}] Live inference requires 'torch' and 'transformers': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _build_variant_questions(self, variant_type: str) -> list[dict[str, Any]]:
        """Build the question list for one variant type.

        ``"standard"`` uses ``original_question`` / ``original_answer`` de-duplicated
        by ``question_id``.  ``"number_swap"`` and ``"irrelevant_sentence"`` use
        ``variant_question`` / ``variant_answer`` filtered by ``variant_type``.

        Args:
            variant_type: One of ``"standard"``, ``"number_swap"``,
                          ``"irrelevant_sentence"``.

        Returns:
            List of ``{"question_id", "question", "expected_answer"}`` dicts.
        """
        if variant_type == "standard":
            seen: set[str] = set()
            out: list[dict[str, Any]] = []
            for row in self.rows:
                qid = row["question_id"]
                if qid not in seen:
                    seen.add(qid)
                    out.append({
                        "question_id": qid,
                        "question": row["original_question"],
                        "expected_answer": row["original_answer"],
                    })
            return out

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
    # Core inference loop (REQ-VERIFY-064, REQ-VERIFY-065, REQ-VERIFY-067)
    # ------------------------------------------------------------------

    def run_variant(
        self,
        *,
        model_name: str,
        variant_type: str,
    ) -> list[dict[str, Any]]:
        """Run baseline inference for one (model, variant_type) pair.

        Checkpoints every ``CHECKPOINT_INTERVAL`` questions (REQ-VERIFY-065).
        Saves logit tensors at prefix fractions defined by ``LOGIT_FRACTIONS``
        (REQ-VERIFY-067).

        Args:
            model_name: Human-readable model label (e.g. ``"Qwen3.5-0.8B"``).
            variant_type: One of ``"standard"``, ``"number_swap"``,
                          ``"irrelevant_sentence"``.

        Returns:
            List of per-question result dicts with keys
            ``question_id``, ``correct``, ``response``, ``elapsed_s``.

        Raises:
            TimeoutError: Re-raised if ``generate_fn`` raises ``TimeoutError``
                          (caller should catch this and emit a partial artifact).
        """
        questions = self._build_variant_questions(variant_type)
        total = len(questions)

        ckpt_path = _ckpt_path(
            self.checkpoint_dir, model_name=model_name, variant_type=variant_type
        )
        ckpt = _load_ckpt(ckpt_path)
        completed: dict[str, Any] = dict(ckpt.get("completed", {}))

        results: list[dict[str, Any]] = []
        logit_buffer: list[np.ndarray] = []
        saved_fractions: set[int] = set()
        variant_logit_paths: dict[str, str] = {}

        for idx, item in enumerate(questions):
            qid = item["question_id"]
            question = item["question"]
            expected = item["expected_answer"]

            if qid in completed:
                # Resume from checkpoint — skip generate_fn call.
                r = completed[qid]
                results.append(r)
                continue

            t0 = time.perf_counter()
            response, logits = self.generate_fn(
                question,
                expected,
                model_name=model_name,
                variant_type=variant_type,
            )
            elapsed = time.perf_counter() - t0

            correct = _is_correct(response, expected)
            result = {
                "question_id": qid,
                "correct": correct,
                "response": response,
                "elapsed_s": round(elapsed, 3),
            }
            results.append(result)
            completed[qid] = result

            # Strip batch dim: (1, seq_len, vocab) → (seq_len, vocab).
            if logits.ndim == 3:
                logit_buffer.append(logits[0])
            else:
                logit_buffer.append(logits)

            # Checkpoint every CHECKPOINT_INTERVAL questions.
            n_done = idx + 1
            if n_done % CHECKPOINT_INTERVAL == 0 or n_done == total:
                _save_ckpt(ckpt_path, {
                    "model_name": model_name,
                    "variant_type": variant_type,
                    "completed": completed,
                })

            # Save logits at prefix fractions (REQ-VERIFY-067).
            if total > 0:
                progress = n_done / total
                for frac in LOGIT_FRACTIONS:
                    pct = int(round(frac * 100))
                    if pct not in saved_fractions and progress >= frac:
                        out_path = _save_logits(
                            self.logit_dir,
                            model_name=model_name,
                            variant_type=variant_type,
                            pct=pct,
                            logit_list=logit_buffer,
                        )
                        variant_logit_paths[f"{pct}pct"] = out_path
                        saved_fractions.add(pct)

        self._logit_paths.setdefault(model_name, {}).update(variant_logit_paths)
        return results

    # ------------------------------------------------------------------
    # Per-model aggregation
    # ------------------------------------------------------------------

    def run_all_variants(self, *, model_name: str) -> dict[str, Any]:
        """Run all three variant types for one model and return accuracy stats.

        Args:
            model_name: Human-readable model label.

        Returns:
            Dict ``{variant_type: {"correct": int, "total": int, "accuracy": float}}``.

        Raises:
            TimeoutError: Propagated from ``run_variant`` on inference timeout.
        """
        variant_stats: dict[str, Any] = {}
        for vt in ("standard", "number_swap", "irrelevant_sentence"):
            results = self.run_variant(model_name=model_name, variant_type=vt)
            n_correct = sum(1 for r in results if r.get("correct"))
            n_total = len(results)
            variant_stats[vt] = {
                "correct": n_correct,
                "total": n_total,
                "accuracy": n_correct / n_total if n_total else 0.0,
            }
        self._model_results[model_name] = variant_stats
        return variant_stats

    # ------------------------------------------------------------------
    # Timeout-aware top-level runner — returns stall_at string or None
    # ------------------------------------------------------------------

    def run_with_timeout_handling(self) -> str | None:
        """Run all models/variants, catching TimeoutError to record the stall location.

        Returns:
            ``None`` for a complete run without timeout, or a ``"model:variant:question_id"``
            string identifying where the first inference stall was detected.

        Unlike Exp 282's ``run_with_timeout_handling``, this method returns only the
        stall location so callers can decide whether to build a full or partial artifact.
        Use :meth:`run_and_save` for the complete end-to-end flow with artifact writing.
        """
        stall_at: str | None = None
        model_name = ""
        vt = ""
        qid = ""

        try:
            for spec in self.model_specs:
                model_name = spec["name"]
                for variant_type in ("standard", "number_swap", "irrelevant_sentence"):
                    vt = variant_type
                    questions = self._build_variant_questions(vt)
                    for item in questions:
                        qid = item["question_id"]
                        question = item["question"]
                        expected = item["expected_answer"]

                        ckpt_path = _ckpt_path(
                            self.checkpoint_dir,
                            model_name=model_name,
                            variant_type=vt,
                        )
                        ckpt = _load_ckpt(ckpt_path)
                        if qid in ckpt.get("completed", {}):
                            continue

                        # Enforce 60 s per-call hard timeout (REQ-VERIFY-066).
                        # We wrap generate_fn in a thread so wall-clock limit is
                        # applied even if generate_fn never raises TimeoutError itself.
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                            _fut = _ex.submit(
                                self.generate_fn,
                                question,
                                expected,
                                model_name=model_name,
                                variant_type=vt,
                            )
                            try:
                                _response, _logits = _fut.result(
                                    timeout=self.timeout_seconds
                                )
                            except concurrent.futures.TimeoutError:
                                raise TimeoutError(
                                    f"Inference stalled >{self.timeout_seconds}s "
                                    f"({model_name}:{vt}:{qid})"
                                )

                        ckpt.setdefault("completed", {})[qid] = {
                            "question_id": qid,
                            "correct": _is_correct(_response, expected),
                            "response": _response,
                        }
                        _save_ckpt(ckpt_path, ckpt)

        except TimeoutError:
            stall_at = f"{model_name}:{vt}:{qid}"

        self._rebuild_results_from_checkpoints()
        return stall_at

    # ------------------------------------------------------------------
    # Full end-to-end run + artifact write
    # ------------------------------------------------------------------

    def run_and_save(
        self,
        *,
        output_path: Path,
        run_date: str = RUN_DATE,
        inference_mode: str | None = None,
    ) -> dict[str, Any]:
        """Run all models/variants, write the artifact JSON, and return the artifact.

        Args:
            output_path: Destination JSON path for the experiment artifact.
            run_date: Date string in ``YYYYMMDD`` format.
            inference_mode: Override for the ``inference_mode`` field.
                            Defaults to ``"live_gpu"`` if ``CARNOT_FORCE_LIVE=1``,
                            otherwise ``"mock"``.

        Returns:
            The artifact dict (may be partial if a timeout occurred).
        """
        if inference_mode is None:
            inference_mode = (
                "live_gpu"
                if os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
                else "mock"
            )

        started_at = utc_now()
        stall_at = self.run_with_timeout_handling()
        finished_at = utc_now()

        pre_warm_status = {
            name: r.health_ok for name, r in self._pre_warm_results.items()
        }
        pre_warm_time_s = {
            name: round(r.load_time_s, 3) for name, r in self._pre_warm_results.items()
        }

        artifact = build_artifact(
            run_date=run_date,
            started_at=started_at,
            finished_at=finished_at,
            inference_mode=inference_mode,
            model_results=self._model_results,
            logit_paths=self._logit_paths,
            stall_at=stall_at,
            stall_diagnosis=self._stall_diagnosis,
            pre_warm_status=pre_warm_status,
            pre_warm_time_s=pre_warm_time_s,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        tmp.replace(output_path)

        return artifact

    # ------------------------------------------------------------------
    # Checkpoint rebuild helper
    # ------------------------------------------------------------------

    def _rebuild_results_from_checkpoints(self) -> None:
        """Recompute ``self._model_results`` from checkpoint files written so far.

        Called after a complete or partial run to ensure the artifact reflects
        the latest checkpoint state rather than any partially-updated in-memory
        state.
        """
        for spec in self.model_specs:
            model_name = spec["name"]
            variant_stats: dict[str, Any] = {}
            for vt in ("standard", "number_swap", "irrelevant_sentence"):
                ckpt_path = _ckpt_path(
                    self.checkpoint_dir, model_name=model_name, variant_type=vt
                )
                ckpt = _load_ckpt(ckpt_path)
                completed = ckpt.get("completed", {})
                results = list(completed.values())
                n_correct = sum(1 for r in results if r.get("correct"))
                n_total = len(results)
                variant_stats[vt] = {
                    "correct": n_correct,
                    "total": n_total,
                    "accuracy": n_correct / n_total if n_total else 0.0,
                }
            self._model_results[model_name] = variant_stats


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------


def _print_report(artifact: dict[str, Any]) -> None:
    """Print a human-readable accuracy report to stdout."""
    print(f"\n[Exp {EXPERIMENT}] Baseline accuracy report")
    print(f"  inference_mode : {artifact.get('inference_mode')}")
    print(f"  partial        : {artifact.get('partial')}")
    if artifact.get("stall_at"):
        print(f"  stall_at       : {artifact['stall_at']}")

    diag = artifact.get("stall_diagnosis", {})
    if diag:
        print(
            f"  GPU 0 free VRAM: {diag.get('vram_gpu0_free_gb', 0):.1f} GiB  "
            f"GPU 1 free VRAM: {diag.get('vram_gpu1_free_gb', 0):.1f} GiB"
        )

    pre_warm = artifact.get("pre_warm_status", {})
    if pre_warm:
        print("  Pre-warm results:")
        for name, ok in pre_warm.items():
            t = artifact.get("pre_warm_time_s", {}).get(name, 0.0)
            print(f"    {name}: {'OK' if ok else 'FAILED'}  ({t:.1f} s)")
    print()

    apple_check = artifact.get("apple_2410_05229_check", {})
    model_results = artifact.get("model_results", {})

    for model_name, variants in model_results.items():
        print(f"  Model: {model_name}")
        std_acc = variants.get("standard", {}).get("accuracy", float("nan"))
        ns_acc = variants.get("number_swap", {}).get("accuracy", float("nan"))
        ir_acc = variants.get("irrelevant_sentence", {}).get("accuracy", float("nan"))
        print(f"    standard            : {std_acc:.1%}")
        print(
            f"    number_swap         : {ns_acc:.1%}  "
            f"(drop: {(std_acc - ns_acc) * 100:.1f} pp)"
        )
        print(
            f"    irrelevant_sentence : {ir_acc:.1%}  "
            f"(drop: {(std_acc - ir_acc) * 100:.1f} pp)"
        )

        chk = apple_check.get(model_name, {})
        if chk:
            confirmed = chk.get("hypothesis_confirmed", False)
            symbol = "✓ CONFIRMED" if confirmed else "✗ NOT CONFIRMED"
            print(f"    Apple 2410.05229 (≥15pp number_swap drop): {symbol}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp 294.

    Environment variables
    ---------------------
    CARNOT_FORCE_LIVE : Set to ``"1"`` for live GPU execution, ``"0"`` for mock.
    JAX_PLATFORMS     : Set to ``"cpu"`` to prevent JAX from claiming the GPU.

    Returns:
        Exit code (0 = success).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Exp {EXPERIMENT}: GPU stall diagnosis + Apple adversarial baseline re-run",
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
        help="Path for the experiment result artifact.",
    )
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"[Exp {EXPERIMENT}] ERROR: Dataset not found at {args.dataset}")
        print(
            f"  Run scripts/experiment_281_apple_adversarial_dataset.py first to generate it."
        )
        return 1

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[Exp {EXPERIMENT}] Loaded {len(rows)} rows from {args.dataset}")

    runner = AppleBaselineRunner294(
        rows,
        checkpoint_dir=args.checkpoint_dir,
        logit_dir=args.logit_dir,
    )

    artifact = runner.run_and_save(output_path=args.output)
    _print_report(artifact)

    print(f"[Exp {EXPERIMENT}] Artifact written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PrewarmResult",
    "model_prewarm",
    "AppleBaselineRunner294",
    "build_artifact",
    "ARTIFACT_SCHEMA",
    "EXPERIMENT",
    "LOGIT_FRACTIONS",
    "CHECKPOINT_INTERVAL",
    "MODEL_SPECS",
    "RUN_DATE",
]
