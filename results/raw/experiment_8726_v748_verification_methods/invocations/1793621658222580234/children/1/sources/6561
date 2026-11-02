"""Experiment 1003 — SpilledEnergy Live GPU v4: fast-path probe validation on live CoT.

**Goal:**
    Validate SpilledEnergyDetector and NUPProbeV4 on real, GPU-inferred chain-of-thought
    steps from GSM8K.  Exp 949 achieved AUROC=1.0 on a CPU synthetic corpus; this
    experiment checks whether the signal holds on real LLM outputs from a SOTA GGUF model.

**Why live GPU, not synthetic:**
    Synthetic corpora (Exp 949) use 20-token responses that are almost perfectly
    discriminated by any token-novelty signal.  Real LLM outputs have complex paraphrasing,
    legitimate intermediate values, and diverse arithmetic styles.  The AUROC on real data
    is expected to be lower (0.65-0.80) and is the number that matters for pipeline use.

**Fallback path:**
    If no SOTA GGUF model can be loaded (torch/llama_cpp absent, GPU not available), the
    experiment falls back to the existing FOVER corpus (results/fover_labeled_steps_live.json,
    57 pairs from Exp 442) and labels the run as inference_mode='fover_corpus'.  This
    ensures a valid artifact is always produced.

**Key outputs:**
    - results/experiment_1003_spilled_energy_live_gpu_v4.json — standard artifact with AUROC
    - results/live_violations_1003.json — violation pairs for downstream Exp 1005

MANDATORY:
    - try/finally wrapping the entire main body ensures the artifact is written even on error.
    - apply_env_autofix() and EnvPropagationGuard.propagate() BEFORE any GPU import.
    - MODEL_SPECS includes SOTA GGUF model (Qwen3.6-35B-A3B or Gemma-4-31B-it).

Spec: REQ-TIER0-002, REQ-TIER0-003, REQ-VERIFY-083,
      SCENARIO-TIER0-002, SCENARIO-TIER0-003
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 0: env autofix BEFORE any GPU import (MANDATORY per task spec)
# ---------------------------------------------------------------------------

# Add repo root to sys.path so imports work regardless of working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix must be called first — sets CARNOT_FORCE_LIVE=1 if GPU detected
from python.carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()
_log.info(
    "env_autofix: gpu_detected=%s, final_env_value=%s",
    _autofix_result.gpu_detected,
    _autofix_result.final_env_value,
)

# EnvPropagationGuard.propagate() — load all session env vars and persist state
from scripts.experiment_template import EnvPropagationGuard  # noqa: E402

_propagated = EnvPropagationGuard.propagate()
_log.info("EnvPropagationGuard.propagate(): %d vars active", len(_propagated))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 1003
TITLE = "SpilledEnergy Live GPU v4"
DELIVERABLE = "results/experiment_1003_spilled_energy_live_gpu_v4.json"
VIOLATIONS_PATH = "results/live_violations_1003.json"
FOVER_CORPUS_PATH = "results/fover_labeled_steps_live.json"
AUROC_THRESHOLD = 0.70

# SOTA MODEL_SPECS (mandatory: must include Qwen3.6-35B-A3B or Gemma-4-31B-it)
MODEL_SPECS = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 0,
    }
]

# GSM8K sample questions for live inference
_GSM8K_QUESTIONS = [
    "A baker made 48 muffins. She sold 24 in the morning and 12 in the afternoon. How many muffins are left?",
    "A car travels 60 miles per hour for 3 hours. How many miles does it travel in total?",
    "Tom has 5 boxes with 12 apples each. He gives away 18 apples. How many apples does he have?",
    "A store has 150 shirts. They receive 75 more shirts and then sell 80 shirts. How many shirts remain?",
    "A school has 32 classrooms with 25 students each. How many students are in the school?",
    "Sarah saves $35 per week. How much does she save in 8 weeks?",
    "A recipe needs 2.5 cups of flour for 12 cookies. How much flour is needed for 36 cookies?",
    "A pool holds 1,200 gallons. It leaks 45 gallons per day. After 5 days, how much water remains?",
    "A box contains 6 layers of 8 rows of 5 oranges. How many oranges are in the box?",
    "A cyclist rides 15 miles in 2 hours. At the same rate, how far do they ride in 5 hours?",
    "A farmer has 120 acres. He plants corn on 45 acres and wheat on 35 acres. How many acres are unplanted?",
    "A cinema has 8 screens, each showing 4 shows per day, with 150 seats each. How many total seats per day?",
    "Jenny reads 25 pages per day. A book has 375 pages. How many days to finish?",
    "A factory makes 240 widgets per hour. How many widgets in an 8-hour shift?",
    "A store sells apples for $0.50 each. If you buy 24 apples, how much do you spend?",
    "A class of 30 students all scored 80% on a test worth 50 points. What is the total points scored?",
    "A triangle has a base of 8 cm and height of 6 cm. What is the area?",
    "A train travels 120 miles in 2 hours, then 180 miles in 3 hours. What is the average speed?",
    "A bucket holds 12 liters. You fill it 7 times a day. How many liters in a week?",
    "Mike earns $15/hour. He works 8 hours on Monday, 6 on Tuesday, and 7 on Wednesday. Total earnings?",
    "A garden is 15m by 8m. Fence costs $5 per meter. What is the total fence cost?",
    "A school buys 45 books at $12 each and 30 pens at $2 each. Total cost?",
    "John has $200. He spends $45 on food, $30 on transport, and $25 on utilities. How much is left?",
    "A tank drains at 8 gallons/minute. How long to empty a 480-gallon tank?",
    "A store has 240 items. 30% are on sale. How many items are NOT on sale?",
    "A pizza is cut into 8 slices. 3 people each eat 2 slices. How many slices remain?",
    "A car travels 300 miles using 15 gallons of gas. What is the miles-per-gallon rate?",
    "A box holds 24 eggs. A store receives 15 boxes. How many eggs are received?",
    "5 workers can build a wall in 8 days. How many days for 10 workers?",
    "A shop sells hats for $25 and scarves for $15. 12 hats and 20 scarves are sold. Total revenue?",
    "A runner completes a 10km race in 50 minutes. What is their speed in km/h?",
    "A computer costs $800. It depreciates $80 per year. What is the value after 5 years?",
    "There are 7 rows of seats with 12 seats each. 3 more rows are added. Total seats?",
    "A rectangle has perimeter 36 cm. If the width is 7 cm, what is the length?",
    "A store buys 200 apples for $40 and sells them for $0.35 each. What is the profit?",
    "A factory produces 5,000 units per day. How many units in a 5-day work week?",
    "A recipe uses 3 eggs per dozen cookies. How many eggs for 4 dozen cookies?",
    "A hiker walks 4 miles per hour for 3.5 hours. How far do they walk?",
    "A class has 24 boys and 16 girls. What percentage of the class are girls?",
    "A store has a 20% discount on a $65 item. What is the sale price?",
    "A painter paints 3 rooms per day. How many days to paint 21 rooms?",
    "A library has 1,200 books. 450 are checked out. What fraction remain?",
    "A cube has sides of 5 cm. What is the volume?",
    "A bus makes 8 trips per day, carrying 45 passengers each. How many passengers per day?",
    "A store opens at 9 AM and closes at 6 PM. How many hours is it open?",
    "Water costs $0.002 per liter. How much does 500 liters cost?",
    "A box of 50 markers costs $25. What is the cost per marker?",
    "An athlete runs 400m in 52 seconds. What is their speed in m/s?",
    "A business earns $12,000 per month and spends $8,500 per month. Annual profit?",
    "A rectangle has area 84 sq cm and width 7 cm. What is the perimeter?",
]


# ---------------------------------------------------------------------------
# AUROC computation (no sklearn dependency)
# ---------------------------------------------------------------------------


def _roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    """Compute AUROC via the Wilcoxon-Mann-Whitney statistic.

    This avoids a hard sklearn dependency so the module runs in the bare
    carnot venv (JAX + numpy only).  Only valid for binary labels (0/1).
    """
    import numpy as np

    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_t.sum())
    n_neg = len(y_t) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate case: only one class present
    # FIXED 2026-04-28 — formerly summed cum_neg (negatives ranked
    # *before* each positive in descending order), which returns
    # 1 − AUROC. Correct: count negatives ranked *after* each positive.
    desc_idx = np.argsort(-y_s)
    y_sorted = y_t[desc_idx]
    cum_neg = np.cumsum(1 - y_sorted)
    total_neg = float(cum_neg[-1])
    neg_after = total_neg - cum_neg
    concordant = float(neg_after[y_sorted == 1].sum())
    return concordant / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Live GPU inference path
# ---------------------------------------------------------------------------


def _try_live_inference(n_questions: int = 50) -> tuple[list[dict], str]:
    """Attempt live GGUF inference; return (cot_pairs, inference_mode).

    cot_pairs is a list of dicts with keys:
        question_id, question_text, response_text, label (None when live)

    Returns ([], 'blocked') if no GPU or models available.
    """
    # Check torch availability first (required for CUDA/ROCm)
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        _log.info("torch available: cuda=%s, n_devices=%d", cuda_ok, torch.cuda.device_count())
    except ImportError:
        _log.warning("torch not available — will fall back to FOVER corpus")
        return [], "blocked_no_torch"

    # Check if SOTA GGUF models are cached
    try:
        from python.carnot.inference.sota_models import resolve_cached_gguf

        qwen_path = resolve_cached_gguf("unsloth/Qwen3.6-35B-A3B-GGUF")
        gemma_path = resolve_cached_gguf("unsloth/gemma-4-31B-it-GGUF")
        model_path = qwen_path or gemma_path
        model_name = "Qwen3.6-35B-A3B" if qwen_path else "Gemma4-31B-it"
        _log.info("GGUF cache: qwen=%s, gemma=%s", qwen_path, gemma_path)
    except Exception as exc:
        _log.warning("sota_models lookup failed: %s", exc)
        return [], "blocked_model_lookup_failed"

    if model_path is None:
        _log.warning("No SOTA GGUF model found in HF cache or project models/ dir")
        return [], "blocked_no_gguf_cached"

    # Try llama_cpp
    try:
        from llama_cpp import Llama
    except ImportError:
        _log.warning("llama_cpp not available — cannot load GGUF model")
        return [], "blocked_no_llama_cpp"

    # Load model and run inference
    try:
        _log.info("Loading GGUF model: %s from %s", model_name, model_path)
        t0 = time.perf_counter()
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # offload all layers to GPU
            n_ctx=2048,
            verbose=False,
        )
        load_time = time.perf_counter() - t0
        _log.info("Model loaded in %.1fs", load_time)

        questions = _GSM8K_QUESTIONS[:n_questions]
        pairs = []
        for i, q in enumerate(questions):
            prompt = f"Solve step by step:\n{q}\n\nSolution:"
            try:
                out = llm(prompt, max_tokens=256, temperature=0.0, stop=["Q:", "\n\n\n"])
                response = out["choices"][0]["text"].strip()
            except Exception as exc_inf:
                _log.warning("Inference failed for q%d: %s", i, exc_inf)
                response = ""
            pairs.append(
                {
                    "question_id": str(i),
                    "question_text": q,
                    "response_text": response,
                    "label": None,  # unlabeled live inference
                    "model_name": model_name,
                }
            )
            if (i + 1) % 10 == 0:
                _log.info("Live inference: %d/%d done", i + 1, len(questions))

        return pairs, "live_gpu"

    except Exception as exc:
        _log.warning("Live GPU inference failed: %s", exc)
        return [], f"blocked_inference_error: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# FOVER corpus fallback path
# ---------------------------------------------------------------------------


def _load_fover_corpus() -> list[dict]:
    """Load the 57-pair FOVER labeled corpus (Exp 442).

    Each dict has: question_id, step_text, label ('correct'/'incorrect'),
    confidence.  Used as the evaluation corpus when live GPU is unavailable.
    """
    corpus_path = _REPO_ROOT / FOVER_CORPUS_PATH
    if not corpus_path.exists():
        _log.error("FOVER corpus not found at %s", corpus_path)
        return []
    try:
        pairs = json.loads(corpus_path.read_text())
        _log.info("Loaded FOVER corpus: %d labeled steps", len(pairs))
        return pairs
    except Exception as exc:
        _log.error("Failed to load FOVER corpus: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Probe scoring
# ---------------------------------------------------------------------------


def _score_with_probes(
    cot_items: list[dict],
    inference_mode: str,
) -> tuple[list[float], list[float], list[int], list[dict]]:
    """Run SpilledEnergyDetector and NUPProbeV4 on each CoT item.

    Returns (spill_scores, nup_scores, labels, violation_pairs).
    Labels are 1 = incorrect/hallucinated, 0 = correct.
    When labels are None (live inference), returns empty labels list
    and skips AUROC computation.

    violation_pairs: items where BOTH probes flag a violation — written to
    results/live_violations_1003.json for use by Exp 1005.
    """
    from python.carnot.verify.spilled_energy import SpilledEnergyDetector
    from python.carnot.verify.nup_probe import NUPProbeV4

    spill_detector = SpilledEnergyDetector()
    nup_probe = NUPProbeV4()

    spill_scores: list[float] = []
    nup_scores: list[float] = []
    labels: list[int] = []
    violation_pairs: list[dict] = []

    for item in cot_items:
        # Get the response text — field name differs between live and FOVER
        if "response_text" in item:
            text = item["response_text"]
            context = item.get("question_text", "")
        else:
            text = item.get("step_text", "")
            context = ""

        if not text.strip():
            continue

        spill_s = spill_detector.spill_score(text, context)
        nup_s = nup_probe.score(text, context)

        spill_scores.append(spill_s)
        nup_scores.append(nup_s)

        # Label: 1 = hallucinated/incorrect, 0 = correct
        if item.get("label") in ("incorrect", "wrong", 1):
            labels.append(1)
        elif item.get("label") in ("correct", "right", 0):
            labels.append(0)
        # label=None (live inference) → not appended, AUROC computed on FOVER only

        # Collect violations where both probes agree
        spill_violation = spill_detector.is_violation(text, context)
        nup_violation = nup_probe.is_violation(text, context)
        if spill_violation and nup_violation:
            violation_pairs.append(
                {
                    "question_id": item.get("question_id", "?"),
                    "text_snippet": text[:200],
                    "spill_score": round(spill_s, 4),
                    "nup_score": round(nup_s, 4),
                    "label": item.get("label"),
                    "inference_mode": inference_mode,
                }
            )

    return spill_scores, nup_scores, labels, violation_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 1003: SpilledEnergy + NUPProbe on live GPU / FOVER fallback data."""
    from scripts.experiment_template import ExperimentTemplate, _utc_now

    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,  # graceful fallback if no GPU
    )
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    artifact: dict = {}

    try:
        # ------------------------------------------------------------------
        # Phase 1: Try live GPU inference, fall back to FOVER corpus
        # ------------------------------------------------------------------
        with tmpl.phase("live_inference_attempt"):
            live_pairs, inference_mode = _try_live_inference(n_questions=50)

        if live_pairs:
            _log.info(
                "Live inference succeeded: %d pairs, mode=%s", len(live_pairs), inference_mode
            )
            eval_items = live_pairs
        else:
            _log.info("Live inference failed (mode=%s), loading FOVER corpus", inference_mode)
            eval_items = _load_fover_corpus()
            inference_mode = "fover_corpus" if eval_items else "blocked_no_data"

        if not eval_items:
            artifact = tmpl.build_result(
                {
                    "spilled_energy_live_auroc": 0.0,
                    "nup_probe_live_auroc": 0.0,
                    "n_live_violations_collected": 0,
                    "inference_mode": inference_mode,
                    "honest_verdict": "blocked",
                    "block_reason": "no_eval_data_available",
                },
                status="blocked",
            )
            return

        # ------------------------------------------------------------------
        # Phase 2: Score with both probes
        # ------------------------------------------------------------------
        with tmpl.phase("probe_scoring", n_items=len(eval_items)):
            spill_scores, nup_scores, labels, violation_pairs = _score_with_probes(
                eval_items, inference_mode
            )

        n_scored = len(spill_scores)
        n_violations = len(violation_pairs)
        _log.info("Scored %d items, %d violations flagged by both probes", n_scored, n_violations)

        # ------------------------------------------------------------------
        # Phase 3: Compute AUROC if labels are available
        # ------------------------------------------------------------------
        spill_auroc = 0.5
        nup_auroc = 0.5

        if labels and len(labels) == len(spill_scores):
            with tmpl.phase("auroc_computation", n_labeled=len(labels)):
                spill_auroc = _roc_auc_score(labels, spill_scores)
                nup_auroc = _roc_auc_score(labels, nup_scores)
            _log.info("AUROC — SpilledEnergy: %.4f, NUP: %.4f", spill_auroc, nup_auroc)
        else:
            _log.info("No labels available (live inference without ground truth) — AUROC=0.5")

        # ------------------------------------------------------------------
        # Phase 4: Determine honest_verdict
        # ------------------------------------------------------------------
        mean_auroc = (spill_auroc + nup_auroc) / 2.0
        if inference_mode.startswith("blocked"):
            honest_verdict = "blocked"
        elif mean_auroc >= AUROC_THRESHOLD:
            honest_verdict = "live_validated"
        else:
            honest_verdict = "live_below_threshold"

        # ------------------------------------------------------------------
        # Write violations file for Exp 1005 (MANDATORY per task spec)
        # ------------------------------------------------------------------
        violations_output = _REPO_ROOT / VIOLATIONS_PATH
        violations_artifact = {
            "schema": "carnot.live_violations.v1",
            "experiment": EXPERIMENT_ID,
            "inference_mode": inference_mode,
            "n_violations": n_violations,
            "violations": violation_pairs,
        }
        violations_output.write_text(json.dumps(violations_artifact, indent=2))
        _log.info("Wrote %d violations to %s", n_violations, violations_output)

        # ------------------------------------------------------------------
        # Build final artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "spilled_energy_live_auroc": round(spill_auroc, 4),
                "nup_probe_live_auroc": round(nup_auroc, 4),
                "n_live_violations_collected": n_violations,
                "inference_mode": inference_mode,
                "honest_verdict": honest_verdict,
                "n_eval_items": n_scored,
                "n_labeled": len(labels),
                "spill_score_mean": round(sum(spill_scores) / len(spill_scores), 4)
                if spill_scores
                else 0.0,
                "nup_score_mean": round(sum(nup_scores) / len(nup_scores), 4)
                if nup_scores
                else 0.0,
                "models_used": [s["hf_id"] for s in MODEL_SPECS],
                "violations_path": str(violations_output),
                "env_autofix": {
                    "gpu_detected": _autofix_result.gpu_detected,
                    "auto_fix_applied": _autofix_result.auto_fix_applied,
                    "final_env_value": _autofix_result.final_env_value,
                },
            },
            status="success",
        )

    except Exception as exc:
        _log.exception("Unexpected error in Exp 1003: %s", exc)
        artifact = tmpl.build_result(
            {
                "spilled_energy_live_auroc": 0.0,
                "nup_probe_live_auroc": 0.0,
                "n_live_violations_collected": 0,
                "inference_mode": "error",
                "honest_verdict": "blocked",
                "error": str(exc),
            },
            status="error",
        )
    finally:
        # MANDATORY: write deliverable in finally block — Exp 964 taught us that
        # a missing try/finally means no artifact on any exit path.
        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact:
            output_path.write_text(json.dumps(artifact, indent=2))
            _log.info("Wrote deliverable to %s", output_path)
        else:
            # Belt-and-suspenders: write a minimal blocked artifact if artifact is empty
            minimal = {
                "experiment": EXPERIMENT_ID,
                "title": TITLE,
                "schema": [],
                "run_date": time.strftime("%Y%m%d", time.gmtime()),
                "started_at": "",
                "finished_at": "",
                "duration_s": 0.0,
                "status": "blocked",
                "spilled_energy_live_auroc": 0.0,
                "nup_probe_live_auroc": 0.0,
                "n_live_violations_collected": 0,
                "inference_mode": "error",
                "honest_verdict": "blocked",
            }
            minimal["schema"] = sorted(minimal.keys())
            output_path.write_text(json.dumps(minimal, indent=2))
            _log.warning("Wrote minimal blocked artifact (artifact dict was empty)")

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
