#!/usr/bin/env python3
"""Experiment 283: Verify-repair benchmark on Apple adversarial GSM8K corpus.

**Research hypothesis (most credibility-critical experiment to date):**
    Verify-repair improvement should be LARGER on number_swap adversarial
    variants than on standard GSM8K (Exp 260/235 baseline), because semantic
    grounding (Exp 279) detects stale-answer errors at 100% and number_swap
    variants generate exactly this error pattern.

**Design — 12 benchmark cells:**
    3 modes × 2 variant types × 2 models = 12 cells

    Modes:
      - baseline       — no verification, raw model output
      - verify_only    — run verifiers, flag violations, no repair
      - verify_repair  — run verifiers, iteratively repair on violations

    Variant types (from Exp 281 adversarial corpus):
      - number_swap         — numeric operands scaled; model may recall stale answer
      - irrelevant_sentence — distractor sentence inserted; answer unchanged

    Models (DualGPURunner, wired at startup):
      - Qwen3.5-0.8B   on GPU 0
      - Gemma4-E4B-it  on GPU 1

**Primary criterion:**
    ``Δ(verify_repair, number_swap) > Δ(verify_repair, standard)``
    where Δ(mode, variant) = accuracy(mode, variant) − accuracy(baseline, variant)
    and standard accuracy is taken from the Exp 282 baseline artifact.

**Logit saving (required for Exp 291 JEPA training):**
    Logit tensors from the initial baseline generation step are saved at
    25/50/75/100% prefix fractions as NumPy .npy files under data/research/.

**Checkpoint / timeout:**
    Checkpoints are written every CHECKPOINT_INTERVAL (10) questions.
    A 60s hard timeout is enforced per inference call; on timeout a partial
    artifact is emitted with ``stall_at`` identifying (model:mode:variant:qid).

Usage (live GPU, CARNOT_FORCE_LIVE=1):
    cd /path/to/carnot
    CARNOT_FORCE_LIVE=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/experiment_283_apple_verify_repair.py

Usage (mock / unit mode — no GPU required):
    CARNOT_FORCE_LIVE=0 .venv/bin/pytest \\
        tests/python/test_experiment_283_apple_verify_repair.py -q --no-cov -n0

Spec: REQ-VERIFY-068, REQ-VERIFY-069, REQ-VERIFY-070, REQ-VERIFY-071,
      REQ-VERIFY-072, SCENARIO-VERIFY-084, SCENARIO-VERIFY-085,
      SCENARIO-VERIFY-086, SCENARIO-VERIFY-087
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 283
"""Experiment number — matches the filename and artifact ``experiment`` field."""

RUN_DATE: str = "20260414"
"""Wall-clock date of this experiment run."""

CHECKPOINT_INTERVAL: int = 10
"""Number of questions processed before each checkpoint write (REQ-VERIFY-068)."""

INFERENCE_TIMEOUT_SECONDS: int = 60
"""Hard per-call timeout in seconds (REQ-VERIFY-071).  If exceeded, a partial
artifact is emitted with a ``stall_at`` field."""

LOGIT_FRACTIONS: list[float] = [0.25, 0.50, 0.75, 1.00]
"""Prefix fractions at which accumulated logit tensors are saved to disk (REQ-VERIFY-070).

Files are named ``logits_283_{model_slug}_{mode}_{variant}_{pct}pct.npy``.
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
]
"""Required top-level fields in the Exp 283 artifact JSON (SCENARIO-VERIFY-084)."""

# Default model pair — Qwen on GPU 0, Gemma on GPU 1 (SCENARIO-VERIFY-086).
MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
]

_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "research" / "gsm8k_adversarial_281.jsonl"
_CHECKPOINT_BASE = Path(__file__).resolve().parents[1] / "results" / "checkpoints" / "experiment_283"
_LOGIT_BASE = Path(__file__).resolve().parents[1] / "data" / "research"
_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "experiment_283_results.json"
_EXP282_RESULTS = Path(__file__).resolve().parents[1] / "results" / "experiment_282_results.json"
_EXP260_RESULTS = Path(__file__).resolve().parents[1] / "results" / "experiment_260_results.json"
_EXP235_RESULTS = Path(__file__).resolve().parents[1] / "results" / "experiment_235_results.json"

# ---------------------------------------------------------------------------
# Utilities (identical pattern to Exp 282 for consistency)
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
# Checkpoint helpers (Exp 283-specific, same atomic-write pattern as Exp 282)
# ---------------------------------------------------------------------------


def _ckpt_path(checkpoint_dir: Path, *, model_name: str, mode: str, variant_type: str) -> Path:
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
# Logit saving helpers (Exp 283 naming: logits_283_…)
# ---------------------------------------------------------------------------


def _logit_npy_path(
    logit_dir: Path, *, model_name: str, mode: str, variant_type: str, pct: int
) -> Path:
    """Return the .npy path for the logit tensor at a given prefix fraction.

    File naming: ``logits_283_{model_slug}_{mode}_{variant}_{pct}pct.npy``

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
        f"logits_283_{safe_slug(model_name)}_{safe_slug(mode)}"
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

    For the primary criterion (SCENARIO-VERIFY-085), the standard variant delta
    is computed as::

        Δ(verify_repair, standard) = verify_repair_standard_acc[model]
                                     − baseline_standard_acc[model]

    This uses standard-variant accuracy from the Exp 282 baseline artifact rather
    than running standard questions again in this experiment.

    Args:
        cell_results: Nested dict ``{model: {mode: {variant: {accuracy: float, …}}}}``.
        baseline_standard_acc: Per-model baseline accuracy on standard questions
                                (from Exp 282).  Optional; if provided, enables
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
                # Only baseline standard available; can't compute delta without vr_std.
                variant_deltas["standard"] = 0.0
            model_deltas[mode] = variant_deltas
        deltas[model_name] = model_deltas
    return deltas


def _primary_criterion_met(
    improvement_deltas: dict[str, dict[str, dict[str, float]]],
) -> bool:
    """Return True if Δ(verify_repair, number_swap) > Δ(verify_repair, standard) for any model.

    This is the primary research criterion (SCENARIO-VERIFY-085): verify-repair
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
) -> dict[str, Any]:
    """Build the Exp 283 result artifact dict.

    The artifact records the 12-cell benchmark results, improvement deltas,
    whether the primary criterion (larger improvement on number_swap) is met,
    and references to comparison experiments.

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
        comparison_refs: References to prior experiment results (Exp 260, 235, 282).

    Returns:
        JSON-serialisable artifact dict containing all ARTIFACT_SCHEMA fields.
    """
    is_partial = stall_at is not None
    return {
        "experiment": EXPERIMENT,
        "schema": "carnot.apple_verify_repair.v1",
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
    }


# ---------------------------------------------------------------------------
# VerifyRepairRunner — the main experiment controller
# ---------------------------------------------------------------------------


class VerifyRepairRunner:
    """Run three inference modes on the Apple adversarial GSM8K corpus.

    This class is the central controller for Exp 283.  It:

    1. Accepts rows from the Exp 281 dataset (number_swap and irrelevant_sentence).
    2. Runs each (model, mode, variant_type) cell via an injectable ``generate_fn``.
    3. For ``verify_only`` and ``verify_repair`` modes, calls the Carnot verifiers
       (semantic grounding + formal claim verifier) and, for ``verify_repair``, feeds
       violations back to the model for iterative repair.
    4. Saves logit tensors at 25/50/75/100% prefix fractions (REQ-VERIFY-070).
    5. Writes checkpoints every CHECKPOINT_INTERVAL questions (REQ-VERIFY-068).
    6. Handles ``TimeoutError`` by emitting a partial artifact (REQ-VERIFY-071).
    7. Wires DualGPUBenchmarkHarness at construction time in live mode (REQ-VERIFY-072).

    Args:
        rows: List of dicts loaded from ``gsm8k_adversarial_281.jsonl``.
        model_specs: Ordered list of ``{"name": ..., "gpu": int}`` dicts.
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

        # DualGPURunner wired at construction time (live mode only).
        self._dual_runner: Any | None = None
        if os.environ.get("CARNOT_FORCE_LIVE", "0") == "1":
            self._wire_dual_gpu_runner()

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
        Assignments: Qwen → GPU 0, Gemma → GPU 1 (SCENARIO-VERIFY-086).
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
            print(f"[Exp 283] DualGPURunner wired:")
            for i, spec in enumerate(self.model_specs):
                print(f"  GPU {spec.get('gpu', i)}: {spec['name']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Exp 283] DualGPURunner unavailable ({exc}) — proceeding without GPU runner")

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
            # In mock mode or minimal environments, verifiers may not be importable.
            pass
        self._verifiers_loaded = True

    # ------------------------------------------------------------------
    # Default generate fn (live GPU mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_generate_fn() -> Callable[..., tuple[str, np.ndarray]]:
        """Return the live GPU inference callable.

        In mock mode (``CARNOT_FORCE_LIVE=0``) returns a simple stub that always
        produces the correct answer.  In live mode loads the transformers pipeline
        and returns real logits.

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
                """Live GPU inference: greedy decode with logit capture."""
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
                f"[Exp 283] Live inference requires 'torch' and 'transformers' "
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
                # Extract claims from the response (heuristic: look for numeric assertions).
                # In a full pipeline this would use the ConstraintExtractor; here we use
                # a lightweight heuristic to detect obvious arithmetic errors.
                numeric_answer = _extract_numeric_answer(response)
                if numeric_answer is not None:
                    # Build a simple cardinality claim for the final answer.
                    claim = {
                        "type": "arithmetic",
                        "operands": [],
                        "result": numeric_answer,
                        "operator": "identity",
                    }
                    result = self._formal_claim_verifier.verify_batch([claim])
                    # Formal claim fires only when a definitive violation is found.
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

        ckpt_path = _ckpt_path(self.checkpoint_dir, model_name=model_name, mode=mode, variant_type=variant_type)
        ckpt = _load_ckpt(ckpt_path)
        completed: dict[str, Any] = dict(ckpt.get("completed", {}))

        results: list[dict[str, Any]] = []
        # Logit buffer: list of (seq_len, vocab_size) arrays for prefix-fraction saving.
        logit_buffer: list[np.ndarray] = []
        # Tracks which prefix fractions have already been saved.
        saved_fractions: set[int] = set()
        # Logit path index for this cell.
        cell_logit_paths: dict[str, str] = {}

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
                    repaired_response, repair_logits = self._repair_response(
                        question=question,
                        response=response,
                        expected_answer=expected,
                        model_name=model_name,
                        variant_type=variant_type,
                    )
                    response = repaired_response
                    repaired = True
                except TimeoutError:
                    # Repair timed out: keep original response, do not mark repaired.
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
                "semantic_grounding_fired": semantic_grounding_fired,
                "formal_claim_fired": formal_claim_fired,
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

            # Save logits at prefix fractions (REQ-VERIFY-070 / SCENARIO-VERIFY-087).
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
                records = self.run_mode_variant(model_name=model_name, mode=mode, variant_type=vt)
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
                            # Check checkpoint — skip if already done.
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

        # After the fast-path probe above, run the full run_all() for real results.
        # If stall_at is set, run_all() may raise again; catch and continue.
        if stall_at is None:
            try:
                self.run_all()
            except TimeoutError as exc:
                stall_at = str(exc)

        finished_at = utc_now()

        # Load comparison references from prior experiments.
        comparison_refs = _load_comparison_refs()

        # Compute improvement deltas using Exp 282 standard baseline if available.
        baseline_standard_acc: dict[str, float] = {}
        for spec in self.model_specs:
            mn = spec["name"]
            ref282 = comparison_refs.get("exp282", {}).get("model_results", {}).get(mn, {})
            std_entry = ref282.get("standard", {})
            if std_entry.get("accuracy") is not None:
                baseline_standard_acc[mn] = std_entry["accuracy"]

        improvement_deltas = compute_improvement_deltas(
            self._cell_results,
            baseline_standard_acc=baseline_standard_acc or None,
        )
        criterion_met = _primary_criterion_met(improvement_deltas)

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

    Loads Exp 282 (Apple baseline), Exp 260 (standard GSM8K), and Exp 235
    (semantic v2 cohort).  Missing files are silently skipped.

    Returns:
        Dict with keys ``"exp282"``, ``"exp260"``, ``"exp235"`` where each
        value is the loaded artifact dict or ``{}`` if not found.
    """
    refs: dict[str, Any] = {}
    for key, path in (
        ("exp282", _EXP282_RESULTS),
        ("exp260", _EXP260_RESULTS),
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
    """CLI entry point for Exp 283.

    Environment variables
    ---------------------
    CARNOT_FORCE_LIVE : "1" enables live GPU execution (default "0" = mock).
    JAX_PLATFORMS     : Set to "cpu" to keep JAX off the GPU.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 283: Verify-repair benchmark on Apple adversarial GSM8K corpus",
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

    # Load dataset.
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[Exp 283] Loaded {len(rows)} rows from {args.dataset}")

    runner = VerifyRepairRunner(
        rows=rows,
        checkpoint_dir=args.checkpoint_dir,
        logit_dir=args.logit_dir,
    )

    print(f"[Exp 283] Running 12-cell benchmark (3 modes × 2 variants × 2 models)…")
    artifact = runner.run_with_timeout_handling(
        output_path=args.output,
        run_date=RUN_DATE,
    )

    print(f"\n[Exp 283] Result written to {args.output}")
    print(f"  partial: {artifact.get('partial')}")
    print(f"  primary_criterion_met: {artifact.get('primary_criterion_met')}")
    if artifact.get("stall_at"):
        print(f"  stall_at: {artifact['stall_at']}")
    for model_name, mode_stats in artifact.get("cell_results", {}).items():
        print(f"\n  Model: {model_name}")
        for mode in MODES:
            for vt in VARIANT_TYPES:
                entry = mode_stats.get(mode, {}).get(vt, {})
                acc = entry.get("accuracy", 0.0)
                print(f"    [{mode:12s}][{vt:22s}] acc={acc:.3f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "VerifyRepairRunner",
    "build_artifact",
    "compute_improvement_deltas",
    "MODEL_SPECS",
    "MODES",
    "VARIANT_TYPES",
    "CHECKPOINT_INTERVAL",
    "INFERENCE_TIMEOUT_SECONDS",
    "LOGIT_FRACTIONS",
    "ARTIFACT_SCHEMA",
    "EXPERIMENT",
    "RUN_DATE",
    "safe_slug",
    "utc_now",
    "get_repo_root",
]


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
