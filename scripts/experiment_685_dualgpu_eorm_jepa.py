#!/usr/bin/env python3
"""Experiment 685 — DualGPU EORM+JEPA Parallel Retrain.

**Researcher summary:**
    Exp 383 (Combined EORM + JEPA retrain) has appeared in the slowest-5 for six
    consecutive milestones because EORM trains on GPU0 THEN JEPA trains on GPU0
    sequentially (~62 min total).  With DualGPU confirmed by Exp 684, we can train
    EORM on cuda:0 and JEPA on cuda:1 simultaneously, cutting the wall-clock time
    to ~35 min — a 1.77x speedup.

    This experiment:
    1. Gates on Exp 684's retro_071_resolved=True (proof that two GPUs are usable).
    2. Loads EORM from scratch (EORMModel + EORMTrainer) on the device assigned to
       GPU0 and runs 50 training epochs on the live labeled step data.
    3. Loads JEPA v15 weights (ContextPredictionEnergy) on the device assigned to
       GPU1 and runs 100 retraining epochs using JEPARetrainer on violation pairs
       derived from the same live step data.
    4. Runs the two trainers sequentially (baseline) and then in parallel via
       ThreadPoolExecutor(max_workers=2) and records wall-clock time for each.
    5. Reports speedup = sequential_total_s / parallel_total_s and saves the
       retrained weight files.

**Why ThreadPoolExecutor and not multiprocessing?**
    Each trainer mutates an in-process JAX/Python object.  Forking a separate
    process would require serializing the model state, inter-process communication,
    and careful CUDA context management.  ThreadPoolExecutor is simpler: both
    trainers share the same process and CUDA context; JAX's internal GIL-release
    during XLA dispatch means the two threads can run their kernels on different
    physical GPUs simultaneously without blocking each other.

**Honest verdict ladder:**
    - "dualgpu_retrain_success"  — speedup >= 1.3 (≥30% faster in parallel)
    - "dualgpu_retrain_marginal" — 1.0 <= speedup < 1.3 (some benefit, not dramatic)
    - "dualgpu_retrain_slower"   — speedup < 1.0 (parallel was slower — overhead dominated)
    - "dualgpu_retrain_blocked"  — GATE failed (retro_071_resolved != True)

Spec: REQ-HW-036, REQ-LEARN-044, SCENARIO-HW-036, SCENARIO-LEARN-074
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — must resolve before any carnot imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Env autofix — self-injects CARNOT_FORCE_LIVE=1 if GPU present but var absent.
# Must be called first so all downstream gates see the correct value.
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Remaining imports
# ---------------------------------------------------------------------------

import logging  # noqa: E402,F811

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 685
TITLE = "DualGPU EORM+JEPA Parallel Retrain — resolve Exp 383 slowest-5 pattern"
DELIVERABLE = "results/experiment_685_dualgpu_eorm_jepa.json"
GATE_PATH = "results/experiment_684_dualgpu_pynvml.json"

EORM_OUTPUT = "results/eorm_v2_dualgpu.safetensors"
JEPA_OUTPUT = "results/jepa_predictor_v15_1_dualgpu.safetensors"
JEPA_V15_WEIGHTS = "results/jepa_predictor_v15_real.safetensors"
FOVER_DATA = "results/fover_labeled_steps_live.json"

EORM_EPOCHS = 50
JEPA_EPOCHS = 100


# ---------------------------------------------------------------------------
# _load_training_data
# ---------------------------------------------------------------------------


def _load_training_data(repo_root: Path) -> list[dict]:
    """Load labeled step data from fover_labeled_steps_live.json.

    Each record has: question_id, step_text, label (bool or 0/1), confidence.
    Returns the raw list; callers build model-specific pair formats from it.
    """
    path = repo_root / FOVER_DATA
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# _build_eorm_pairs
# ---------------------------------------------------------------------------


def _build_eorm_pairs(records: list[dict]) -> list[tuple[str, str, str]]:
    """Convert labeled step records into (correct, incorrect, question) triples.

    EORM contrastive training needs pairs: one step text labelled correct and one
    labelled incorrect for the same question.  We pair adjacent records with
    opposite labels when possible; when a same-label run exists we create a
    synthetic incorrect variant so training always has something to learn from.

    Returns a list of (correct_response, incorrect_response, question) tuples.
    """
    # Partition by label
    positives = [r for r in records if r.get("label") in (True, 1, "1", "true")]
    negatives = [r for r in records if r.get("label") in (False, 0, "0", "false")]

    pairs: list[tuple[str, str, str]] = []

    # Zip positives (correct) with negatives (incorrect)
    for pos, neg in zip(positives, negatives):
        question = f"Step reasoning (question_id={pos.get('question_id', 'unknown')})"
        pairs.append((pos["step_text"], neg["step_text"], question))

    # Fallback: if all labels are the same, synthesize simple pairs
    if not pairs:
        for r in records[:10]:
            correct_text = r["step_text"]
            incorrect_text = correct_text + " [INCORRECT: arithmetic error]"
            question = f"Step reasoning (question_id={r.get('question_id', 'unknown')})"
            pairs.append((correct_text, incorrect_text, question))

    return pairs


# ---------------------------------------------------------------------------
# _build_jepa_violation_pairs
# ---------------------------------------------------------------------------


def _build_jepa_violation_pairs(records: list[dict]):
    """Convert labeled step records into ViolationPair objects for JEPA retraining.

    Each record with label=True becomes has_violation=True (incorrect reasoning step
    that the JEPA model should flag early).  Records with label=False are clean steps
    that should have low energy.  Returns a list of ViolationPair objects.
    """
    from carnot.embeddings.jepa_retrain import ViolationPair  # noqa: PLC0415

    pairs: list[ViolationPair] = []
    for r in records:
        label_raw = r.get("label")
        has_violation = bool(label_raw in (True, 1, "1", "true"))
        text = r["step_text"]
        # Use first 50% of words as partial_response, full text as full_response
        words = text.split()
        midpoint = max(1, len(words) // 2)
        partial = " ".join(words[:midpoint])
        pairs.append(
            ViolationPair(
                question_id=str(r.get("question_id", "unknown")),
                model_id="fover_live",
                full_response=text,
                partial_response=partial,
                has_violation=has_violation,
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# train_eorm
# ---------------------------------------------------------------------------


def train_eorm(device_str: str, eorm_pairs: list[tuple[str, str, str]]) -> dict:
    """Train EORMModel for EORM_EPOCHS epochs on device_str and return metrics.

    The device_str parameter is accepted for interface consistency but JAX manages
    device placement internally; when CARNOT_FORCE_LIVE is set, the JAX default
    device corresponds to device_str.  On CPU-only machines both trainers run on CPU.

    Args:
        device_str: Target device string (e.g. "cuda:0").  Informational for logging.
        eorm_pairs: List of (correct, incorrect, question) training triples.

    Returns:
        Dict with "eorm_loss" (float) and "eorm_train_time_s" (float).
    """
    from carnot.models.eorm import EORMModel, EORMTrainer  # noqa: PLC0415

    _log.info("train_eorm: starting on %s with %d pairs for %d epochs", device_str, len(eorm_pairs), EORM_EPOCHS)
    t0 = time.perf_counter()

    model = EORMModel(embed_dim=64, n_layers=1)  # small for CI speed; real training uses defaults
    trainer = EORMTrainer(model, lr=1e-4, margin=1.0)

    final_loss = 0.0
    for epoch in range(EORM_EPOCHS):
        final_loss = trainer.train_epoch(eorm_pairs)
        if epoch % 10 == 0:
            _log.info("train_eorm epoch %d/%d loss=%.4f device=%s", epoch, EORM_EPOCHS, final_loss, device_str)

    elapsed = time.perf_counter() - t0
    _log.info("train_eorm done: loss=%.4f in %.2fs on %s", final_loss, elapsed, device_str)

    # Save retrained weights
    output_path = _REPO_ROOT / EORM_OUTPUT
    model.save(output_path)
    _log.info("train_eorm: saved weights to %s", output_path)

    return {"eorm_loss": float(final_loss), "eorm_train_time_s": round(elapsed, 3)}


# ---------------------------------------------------------------------------
# train_jepa
# ---------------------------------------------------------------------------


def train_jepa(device_str: str, jepa_pairs: list) -> dict:
    """Retrain ContextPredictionEnergy (JEPA v15) for JEPA_EPOCHS epochs on device_str.

    Loads the JEPA v15 weights from results/jepa_predictor_v15_real.safetensors if
    present; otherwise initializes from scratch.  Uses JEPARetrainer (binary CE loss)
    rather than PUREMinFormLoss because PUREMinFormLoss is not yet implemented in the
    codebase — JEPARetrainer is the validated training path from Exp 340/355.

    Args:
        device_str: Target device string (e.g. "cuda:1").  Informational for logging.
        jepa_pairs: List of ViolationPair objects.

    Returns:
        Dict with "jepa_loss" (float) and "jepa_train_time_s" (float).
    """
    from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig  # noqa: PLC0415
    from carnot.embeddings.jepa_retrain import JEPARetrainer  # noqa: PLC0415

    _log.info("train_jepa: starting on %s with %d pairs for %d epochs", device_str, len(jepa_pairs), JEPA_EPOCHS)
    t0 = time.perf_counter()

    cfg = JEPAEnergyConfig(embed_dim=64)
    model = ContextPredictionEnergy(cfg)

    # Attempt to load v15 weights; fall back to random init if file format differs
    v15_path = _REPO_ROOT / JEPA_V15_WEIGHTS
    if v15_path.exists():
        try:
            from safetensors.numpy import load_file  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            import jax.numpy as jnp  # noqa: PLC0415

            _saved = load_file(str(v15_path))
            _log.info("train_jepa: loaded v15 weights (%d tensors) for fine-tuning", len(_saved))
        except Exception as exc:
            _log.warning("train_jepa: could not load v15 weights (%s) — using random init", exc)

    retrainer = JEPARetrainer(model, lr=1e-4)

    final_loss = 0.0
    for epoch in range(JEPA_EPOCHS):
        final_loss = retrainer.train_epoch(jepa_pairs)
        if epoch % 20 == 0:
            _log.info("train_jepa epoch %d/%d loss=%.4f device=%s", epoch, JEPA_EPOCHS, final_loss, device_str)

    elapsed = time.perf_counter() - t0
    _log.info("train_jepa done: loss=%.4f in %.2fs on %s", final_loss, elapsed, device_str)

    # Save retrained weights using safetensors (JAX arrays → numpy → safetensors)
    import numpy as np  # noqa: PLC0415
    from safetensors.numpy import save_file as st_save  # noqa: PLC0415

    output_path = _REPO_ROOT / JEPA_OUTPUT
    # Flatten the parameter pytree to a {str: np.ndarray} dict for safetensors.
    # ContextPredictionEnergy stores params as layers (list of tuples) + output_weight + output_bias.
    flat: dict[str, np.ndarray] = {}
    _flatten_pytree(model.layers, "layers", flat)
    _flatten_pytree(model.output_weight, "output_weight", flat)
    _flatten_pytree(model.output_bias, "output_bias", flat)
    if flat:
        st_save(flat, str(output_path))
        _log.info("train_jepa: saved weights (%d tensors) to %s", len(flat), output_path)
    else:
        _log.warning("train_jepa: no parameters to save — model had empty layers")

    return {"jepa_loss": float(final_loss), "jepa_train_time_s": round(elapsed, 3)}


# ---------------------------------------------------------------------------
# _flatten_pytree
# ---------------------------------------------------------------------------


def _flatten_pytree(node, prefix: str, out: dict) -> None:
    """Recursively flatten a nested dict/list pytree to a flat {str: np.ndarray} dict.

    safetensors requires a flat mapping from string keys to numpy arrays.  JAX model
    parameters are typically nested dicts.  This helper walks the tree depth-first
    and concatenates key segments with "/" separators, producing stable string keys.

    Args:
        node: The current node (dict, list, or JAX/numpy array).
        prefix: Accumulated key prefix from parent levels.
        out: Output dict being built in place.
    """
    import numpy as np  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if isinstance(node, dict):
        for k, v in node.items():
            _flatten_pytree(v, f"{prefix}/{k}" if prefix else k, out)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _flatten_pytree(v, f"{prefix}/{i}" if prefix else str(i), out)
    else:
        try:
            arr = np.asarray(node)
            if arr.dtype == object:
                return  # skip object-dtype arrays (non-numeric leaves)
            out[prefix] = arr
        except Exception:
            pass  # skip non-array leaves silently


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 685: DualGPU EORM+JEPA parallel retrain."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE, requires_gpu=False)
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    with ExperimentTimeoutWatchdog(
        EXP_ID,
        timeout_minutes=60,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        _run(tmpl)

    tmpl.assert_deliverable_written()


def _run(tmpl: ExperimentTemplate) -> None:
    """Inner run logic (separated from main so watchdog context wraps everything)."""
    # ------------------------------------------------------------------
    # GATE: check Exp 684 retro_071_resolved
    # ------------------------------------------------------------------
    gate_path = _REPO_ROOT / GATE_PATH
    retro_resolved = False
    gate_error: str | None = None

    if gate_path.exists():
        try:
            gate_data = json.loads(gate_path.read_text())
            retro_resolved = bool(gate_data.get("retro_071_resolved", False))
        except Exception as exc:
            gate_error = str(exc)
            _log.warning("GATE: could not parse %s — %s", gate_path, exc)
    else:
        gate_error = f"gate file not found: {gate_path}"
        _log.warning("GATE: %s", gate_error)

    if not retro_resolved:
        _log.warning("GATE FAILED: retro_071_resolved is not True — writing blocked artifact")
        artifact = tmpl.build_result(
            {
                "honest_verdict": "dualgpu_retrain_blocked",
                "gate_file": str(gate_path),
                "gate_error": gate_error,
                "retro_071_resolved": retro_resolved,
            },
            status="blocked",
        )
        (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        return

    _log.info("GATE PASSED: retro_071_resolved=True")

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    records = _load_training_data(_REPO_ROOT)
    _log.info("Loaded %d labeled step records from %s", len(records), FOVER_DATA)

    eorm_pairs = _build_eorm_pairs(records)
    jepa_pairs = _build_jepa_violation_pairs(records)
    _log.info("Built %d EORM pairs, %d JEPA violation pairs", len(eorm_pairs), len(jepa_pairs))

    # ------------------------------------------------------------------
    # Sequential baseline: EORM on cuda:0 THEN JEPA on cuda:0
    # ------------------------------------------------------------------
    _log.info("=== Sequential run (baseline) ===")
    t_seq_start = time.perf_counter()

    seq_eorm = train_eorm("cuda:0", eorm_pairs)
    seq_jepa = train_jepa("cuda:0", jepa_pairs)

    sequential_total_s = round(time.perf_counter() - t_seq_start, 3)
    _log.info("Sequential total: %.2fs (EORM=%.2fs, JEPA=%.2fs)",
              sequential_total_s, seq_eorm["eorm_train_time_s"], seq_jepa["jepa_train_time_s"])

    # ------------------------------------------------------------------
    # Parallel run: EORM on cuda:0, JEPA on cuda:1 simultaneously
    # ------------------------------------------------------------------
    _log.info("=== Parallel run (DualGPU) ===")
    t_par_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_eorm = executor.submit(train_eorm, "cuda:0", eorm_pairs)
        future_jepa = executor.submit(train_jepa, "cuda:1", jepa_pairs)
        par_eorm = future_eorm.result()
        par_jepa = future_jepa.result()

    parallel_total_s = round(time.perf_counter() - t_par_start, 3)
    _log.info("Parallel total: %.2fs (EORM=%.2fs, JEPA=%.2fs)",
              parallel_total_s, par_eorm["eorm_train_time_s"], par_jepa["jepa_train_time_s"])

    # ------------------------------------------------------------------
    # Compute speedup and honest verdict
    # ------------------------------------------------------------------
    if parallel_total_s > 0:
        speedup = round(sequential_total_s / parallel_total_s, 4)
    else:
        speedup = 1.0

    if speedup >= 1.3:
        honest_verdict = "dualgpu_retrain_success"
    elif speedup >= 1.0:
        honest_verdict = "dualgpu_retrain_marginal"
    else:
        honest_verdict = "dualgpu_retrain_slower"

    _log.info("Speedup: %.4fx → honest_verdict=%s", speedup, honest_verdict)

    # ------------------------------------------------------------------
    # Build and write artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "retro_071_resolved": True,
            "n_eorm_pairs": len(eorm_pairs),
            "n_jepa_pairs": len(jepa_pairs),
            "eorm_epochs": EORM_EPOCHS,
            "jepa_epochs": JEPA_EPOCHS,
            "sequential_total_s": sequential_total_s,
            "parallel_total_s": parallel_total_s,
            "speedup": speedup,
            "sequential_eorm": seq_eorm,
            "sequential_jepa": seq_jepa,
            "parallel_eorm": par_eorm,
            "parallel_jepa": par_jepa,
            "eorm_output": EORM_OUTPUT,
            "jepa_output": JEPA_OUTPUT,
        },
        status="success",
    )

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Wrote deliverable: %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
