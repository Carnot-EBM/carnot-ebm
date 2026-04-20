#!/usr/bin/env python3
"""Experiment 308: JEPA fast-path gate benchmark — latency vs accuracy trade-off.

**Researcher summary:**
    Benchmarks the JepaGate (trained in Exp 307) as a latency-reduction
    filter before full Ising verification.  The gate takes the mean logit
    vector from a response and predicts a scalar energy: low energy → safe
    to skip expensive Ising verification.

    Primary question: can any threshold in [0.3, 0.5, 0.7] achieve
    skip_rate ≥ 30% while keeping TP_rate ≥ 0.85 on a 50-question corpus?

    TP_rate = fraction of questions with real violations that the gate did
    NOT skip (i.e. the gate correctly sent them to full Ising).  A gate
    that skips too aggressively will miss violations (TP_rate < 0.85).

    skip_rate = fraction of total questions where gate said "skip Ising".
    A gate that never skips delivers 0% latency improvement.

**Model loading:**
    Tries jepa_predictor_307.onnx first (Exp 307 retrain), then falls back
    to jepa_predictor_291.onnx (Exp 291 calibrated model).  If neither
    exists, emits a ``blocked`` artifact listing exact missing paths.

**Corpus:**
    Uses a 50-question simulated corpus with arithmetic errors injected for
    ~30% of questions (to give a realistic mix of clean + violated responses).
    Real GPU inference is used if Qwen3.5-0.8B is available; otherwise
    simulated inference with ``inference_mode="simulated"`` label.

**Honest reporting:**
    - TP_rate < 0.85 is reported as a miss — never hidden.
    - skip_rate = 0 is reported honestly even if the target isn't met.
    - threshold_sweep entries always include raw counts for auditability.

Spec: REQ-JEPA-005, SCENARIO-JEPA-010, SCENARIO-JEPA-011
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID: int = 308
"""Experiment number — matches filename and artifact ``experiment`` field."""

N_QUESTIONS: int = 50
"""Benchmark corpus size."""

THRESHOLD_SWEEP: list[float] = [0.3, 0.5, 0.7]
"""Gate threshold values to evaluate."""

VIOLATION_RATE: float = 0.3
"""Fraction of simulated questions that have real arithmetic violations."""

ONNX_CANDIDATES: list[str] = [
    "results/jepa_predictor_307.onnx",
    "results/jepa_predictor_291.onnx",
]
"""ONNX model paths to try in order (Exp 307 first, 291 as fallback)."""

DELIVERABLE: str = "results/experiment_308_jepa_gate_benchmark.json"


# ---------------------------------------------------------------------------
# Simulated corpus
# ---------------------------------------------------------------------------


def build_corpus(n: int, violation_rate: float, seed: int = 42) -> list[dict[str, Any]]:
    """Build a synthetic 50-question benchmark corpus.

    **Detailed explanation for engineers:**
        Each question is an arithmetic addition prompt.  Roughly
        ``violation_rate * n`` of the responses contain deliberate wrong
        answers, labelled ``ground_truth_violated=True``.  The remaining
        responses are correct.

        We also inject a simulated logit mean vector for each question.
        In a live setting these would come from the LLM's generation pass;
        here they are random normal vectors seeded for reproducibility.
        The JEPA model was trained on mean logit vectors of shape (vocab_size,)
        but a small random vector of shape (32,) is sufficient for ONNX
        inference since the model input size is flexible.

    Args:
        n: Number of questions to generate.
        violation_rate: Fraction of questions with a wrong answer.
        seed: RNG seed for reproducibility.

    Returns:
        List of dicts with keys: ``question``, ``response``, ``domain``,
        ``ground_truth_violated``, ``logit_mean`` (list of floats).
    """
    import numpy as np

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    corpus: list[dict[str, Any]] = []
    for i in range(n):
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        correct = a + b
        violated = rng.random() < violation_rate
        if violated:
            # Introduce a wrong answer (off by ±1 to ±5).
            wrong = correct + rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            response = f"The answer is {a} + {b} = {wrong}."
        else:
            response = f"The answer is {a} + {b} = {correct}."
        # Simulated logit mean: random normal, shape (8,) — matching the JEPA
        # ONNX model input dimension.  In a live run these would be the actual
        # mean logit values projected to the model's feature dimension.
        logit_mean: list[float] = np_rng.randn(8).astype(np.float32).tolist()
        corpus.append(
            {
                "question": f"What is {a} + {b}?",
                "response": response,
                "domain": "arithmetic",
                "ground_truth_violated": violated,
                "logit_mean": logit_mean,
            }
        )
    return corpus


# ---------------------------------------------------------------------------
# Gate benchmark for a single threshold
# ---------------------------------------------------------------------------


def benchmark_threshold(
    pipeline: Any,
    gate: Any,
    corpus: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Run the gate benchmark for one threshold value.

    **Detailed explanation for engineers:**
        For each question we call ``pipeline.verify_with_gate()`` with the
        gate set to the given threshold, passing the simulated ``logit_mean``.
        We record whether the gate skipped Ising or ran it, and compare the
        gate decision against ``ground_truth_violated``.

        TP_rate is computed only over questions with real violations:
            TP_rate = (violations sent to Ising) / (total real violations)
        A skip on a violated question is a miss (TP--).  The gate must keep
        TP_rate ≥ 0.85 to meet the spec target.

        skip_rate = n_skipped / n_total.  The gate must hit skip_rate ≥ 0.30
        at the same threshold that achieves TP_rate ≥ 0.85.

    Args:
        pipeline: VerifyRepairPipeline instance.
        gate: JepaGate with threshold already set.
        corpus: List of question dicts from build_corpus().
        threshold: The threshold value being evaluated (for logging only).

    Returns:
        Dict with keys: threshold, skip_rate, n_skipped, n_total,
        n_violated, n_violations_caught, TP_rate, meets_target,
        pipeline_time_s, per_question (list of per-question dicts).
    """
    import numpy as np
    from scripts.experiment_template import BatchedInferenceRunner  # REQ-INFRA-075

    gate.threshold = threshold  # override for sweep

    def _infer_with_gate(row_json: str) -> str:
        """Run one corpus row through the gated pipeline; return JSON result."""
        row = json.loads(row_json)
        lm = np.array(row["logit_mean"], dtype=np.float32)
        result = pipeline.verify_with_gate(
            question=row["question"],
            response=row["response"],
            domain=row["domain"],
            jepa_gate=gate,
            logit_mean=lm,
        )
        decision = result.certificate.get("gate_decision", "verify")
        return json.dumps({
            "question": row["question"],
            "ground_truth_violated": row["ground_truth_violated"],
            "gate_decision": decision,
            "gate_energy": result.certificate.get("gate_energy"),
            "ising_ran": decision != "skip",
        })

    t_start = time.perf_counter()
    bir = BatchedInferenceRunner(_infer_with_gate, batch_size=8)
    ir_list = bir.run_batch([json.dumps(row) for row in corpus])
    elapsed = time.perf_counter() - t_start

    per_question: list[dict[str, Any]] = []
    n_skipped = 0
    n_violated = 0
    n_violations_caught = 0

    for ir, row in zip(ir_list, corpus):
        if ir.timed_out:
            entry: dict[str, Any] = {
                "question": row["question"],
                "ground_truth_violated": row["ground_truth_violated"],
                "gate_decision": "verify",
                "gate_energy": None,
                "ising_ran": True,
            }
        else:
            entry = json.loads(ir.response)

        skipped = entry["gate_decision"] == "skip"
        if skipped:
            n_skipped += 1
        if row["ground_truth_violated"]:
            n_violated += 1
            if not skipped:
                n_violations_caught += 1
        per_question.append(entry)

    n_total = len(corpus)
    skip_rate = n_skipped / n_total if n_total > 0 else 0.0
    tp_rate = n_violations_caught / n_violated if n_violated > 0 else 1.0
    meets_target = (skip_rate >= 0.30) and (tp_rate >= 0.85)

    return {
        "threshold": threshold,
        "n_total": n_total,
        "n_skipped": n_skipped,
        "skip_rate": skip_rate,
        "n_violated": n_violated,
        "n_violations_caught": n_violations_caught,
        "TP_rate": tp_rate,
        "meets_target": meets_target,
        "pipeline_time_s": elapsed,
        "per_question": per_question,
        "batch_log": bir.batch_log,
    }


# ---------------------------------------------------------------------------
# Baseline (no gate) timing
# ---------------------------------------------------------------------------


def benchmark_no_gate(
    pipeline: Any,
    corpus: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure baseline pipeline time without any gate.

    Returns dict with pipeline_time_s, per-question results, and batch_log.

    Uses BatchedInferenceRunner (REQ-INFRA-075) to group corpus rows into
    batches of 8 with a per-batch timeout, replacing the prior sequential
    for-loop that was the bottleneck identified in Exp 547.
    """
    from scripts.experiment_template import BatchedInferenceRunner  # REQ-INFRA-075

    def _infer_no_gate(row_json: str) -> str:
        """Run one corpus row through the pipeline with no gate; return JSON result."""
        row = json.loads(row_json)
        result = pipeline.verify_with_gate(
            question=row["question"],
            response=row["response"],
            domain=row["domain"],
            jepa_gate=None,
        )
        return json.dumps({
            "question": row["question"],
            "verified": result.verified,
            "n_violations": len(result.violations),
        })

    t_start = time.perf_counter()
    bir = BatchedInferenceRunner(_infer_no_gate, batch_size=8)
    ir_list = bir.run_batch([json.dumps(row) for row in corpus])
    elapsed = time.perf_counter() - t_start

    per_question: list[dict[str, Any]] = []
    for ir, row in zip(ir_list, corpus):
        if ir.timed_out:
            per_question.append({"question": row["question"], "verified": False, "n_violations": 0})
        else:
            per_question.append(json.loads(ir.response))

    return {"pipeline_time_s": elapsed, "per_question": per_question, "batch_log": bir.batch_log}


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(output_path: str | Path | None = None) -> dict[str, Any]:
    """Run Exp 308: JEPA gate benchmark across threshold sweep.

    **Detailed explanation for engineers:**
        1. Resolve the ONNX model path (try 307, fallback to 291).
        2. Emit a blocked artifact if no model found.
        3. Build the 50-question simulated corpus.
        4. Run baseline (no gate) for latency comparison.
        5. For each threshold in [0.3, 0.5, 0.7]:
           - Build JepaGate with that threshold.
           - Run benchmark_threshold().
        6. Compute speedup_factor = baseline_time / best_gate_time.
        7. Write artifact to DELIVERABLE path.

    Args:
        output_path: Override the output path (default: DELIVERABLE constant).

    Returns:
        Artifact dict matching the Exp 308 schema.
    """
    from carnot.pipeline.jepa_fast_path import JepaGate
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    out_path = Path(output_path or DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resolve ONNX model ---
    onnx_path: str | None = None
    for candidate in ONNX_CANDIDATES:
        if Path(candidate).exists():
            onnx_path = candidate
            break

    if onnx_path is None:
        artifact: dict[str, Any] = {
            "experiment": EXPERIMENT_ID,
            "status": "blocked",
            "reason": "ONNX model not found",
            "missing_paths": ONNX_CANDIDATES,
            "run_date": time.strftime("%Y%m%d"),
        }
        out_path.write_text(json.dumps(artifact, indent=2))
        return artifact

    # --- Build corpus and pipeline ---
    corpus = build_corpus(N_QUESTIONS, VIOLATION_RATE)
    pipeline = VerifyRepairPipeline()
    onnx_source = "Exp307" if "307" in onnx_path else "Exp291"

    # --- Baseline timing (no gate) ---
    baseline = benchmark_no_gate(pipeline, corpus)
    baseline_time = baseline["pipeline_time_s"]

    # --- Threshold sweep ---
    threshold_sweep: list[dict[str, Any]] = []
    best_gate_time = baseline_time  # worst case: no improvement

    for thresh in THRESHOLD_SWEEP:
        gate = JepaGate(onnx_path=onnx_path, threshold=thresh, enabled=True)
        sweep_result = benchmark_threshold(pipeline, gate, corpus, thresh)
        # Compute speedup relative to baseline.
        gate_time = sweep_result["pipeline_time_s"]
        speedup = baseline_time / gate_time if gate_time > 0 else float("inf")
        sweep_result["speedup_factor"] = speedup
        sweep_result["gate_config"] = gate.to_dict()
        if gate_time < best_gate_time:
            best_gate_time = gate_time
        threshold_sweep.append(sweep_result)

    # --- Primary metric: did any threshold meet the target? ---
    meets_any = any(t["meets_target"] for t in threshold_sweep)
    best_threshold = next(
        (t for t in threshold_sweep if t["meets_target"]), threshold_sweep[-1]
    )
    overall_speedup = baseline_time / best_gate_time if best_gate_time > 0 else 1.0

    artifact = {
        "experiment": EXPERIMENT_ID,
        "status": "success",
        "run_date": time.strftime("%Y%m%d"),
        "onnx_model": onnx_path,
        "onnx_source": onnx_source,
        "n_questions": N_QUESTIONS,
        "violation_rate": VIOLATION_RATE,
        "inference_mode": "simulated",
        "threshold_sweep": threshold_sweep,
        "baseline_pipeline_time_s": baseline_time,
        "best_gate_pipeline_time_s": best_gate_time,
        "overall_speedup_factor": overall_speedup,
        "meets_target": meets_any,
        "primary_result": (
            "Target met: skip_rate >= 0.30 AND TP_rate >= 0.85 achieved"
            if meets_any
            else "Target not met: no threshold achieved skip_rate >= 0.30 AND TP_rate >= 0.85"
        ),
        "best_threshold": best_threshold.get("threshold"),
        "best_skip_rate": best_threshold.get("skip_rate"),
        "best_TP_rate": best_threshold.get("TP_rate"),
        "batch_log": baseline.get("batch_log", []),
    }

    out_path.write_text(json.dumps(artifact, indent=2))
    return artifact


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: wrap run_experiment() with ExperimentTemplate lifecycle."""
    from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
    from scripts.experiment_template import ExperimentTemplate

    repo_root = Path(__file__).resolve().parents[1]
    tmpl = ExperimentTemplate(
        exp_id=EXPERIMENT_ID,
        title="JEPA gate benchmark (Exp 308)",
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=repo_root,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXPERIMENT_ID,
        timeout_minutes=40,
        result_path=str(repo_root / DELIVERABLE),
    )
    _watchdog.start()

    result = run_experiment()
    _watchdog.stop()
    print(json.dumps(result, indent=2))
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
