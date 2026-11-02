"""Experiment 654: HermesV2StructuredLoop — COMPUTE: forcing + live SymCodeVerifier.

RETRO-070 VR critical path attempt #18.

All 17 prior approaches are architecturally capped at ~12% recall because they attack the
extraction layer — they try to parse whatever arithmetic the model happened to write in
natural language prose.  The fundamental fix: change what the model writes, not how we
parse it.

Exp 653 proved that injecting a COMPUTE: system prompt forces the model to write every
arithmetic step in a parseable format (detection_rate_on_forced=1.0 on synthetic
responses).  This experiment wires that forcing system prompt into a live generation
loop and runs SymCodeVerifier on each COMPUTE: line to measure whether violations are
now detectable at a rate above the 12% post-hoc ceiling.

Gate: hermes_v2_structured_recall >= 0.30 on 25 known-incorrect live questions.

Spec: REQ-VERIFY-147, REQ-VERIFY-148,
      SCENARIO-VERIFY-197, SCENARIO-VERIFY-198, SCENARIO-VERIFY-199
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.hermes_v2_structured_loop import (
    HermesV2StructuredLoop,
    HermesV2StructuredResult,
)
from carnot.pipeline.structured_equation_forcer import StructuredEquationForcer
from carnot.pipeline.symcode_verifier import SymCodeVerifier
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# 1. env autofix FIRST — detects GPU hardware and injects CARNOT_FORCE_LIVE=1
# ---------------------------------------------------------------------------
apply_env_autofix()

# ---------------------------------------------------------------------------
# 2. Watchdog — abort if experiment exceeds 90 minutes
# ---------------------------------------------------------------------------
_watchdog = ExperimentTimeoutWatchdog(654, timeout_minutes=90)

# ---------------------------------------------------------------------------
# 3. Experiment setup
# ---------------------------------------------------------------------------
tmpl = ExperimentTemplate(
    654,
    "HERMES v2 Structured Live",
    "results/experiment_654_hermes_v2_structured.json",
    requires_gpu=True,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# 4. CI stub gate — exit 0 with honest_verdict='ci_stub_gpu_required' if not live
# ---------------------------------------------------------------------------
CARNOT_FORCE_LIVE = os.environ.get("CARNOT_FORCE_LIVE", "") == "1"

if not CARNOT_FORCE_LIVE:
    artifact = tmpl.build_result(
        {
            "schema": "carnot.hermes_v2_structured.v1",
            "n_questions": 25,
            "n_correct": 10,
            "hermes_v2_structured_recall": None,
            "hermes_v2_structured_fp_rate": None,
            "post_hoc_baseline": 0.12,
            "structured_improvement": None,
            "gate_contribution": False,
            "retro_070_partial": False,
            "inference_mode": "ci_stub",
            "honest_verdict": "ci_stub_gpu_required",
        },
        status="blocked",
    )
    output_path = _repo_root / "results" / "experiment_654_hermes_v2_structured.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print("CARNOT_FORCE_LIVE not set — writing ci_stub artifact and exiting.")
    print(f"Artifact written to {output_path}")
    tmpl.assert_deliverable_written()
    sys.exit(0)

# ---------------------------------------------------------------------------
# 5. GPU health check
# ---------------------------------------------------------------------------
MODEL_SPECS = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]
gpu_status = tmpl.setup_gpu(MODEL_SPECS)

if not gpu_status["all_healthy"]:
    artifact = tmpl.build_result(
        {
            "schema": "carnot.hermes_v2_structured.v1",
            "n_questions": 25,
            "n_correct": 10,
            "hermes_v2_structured_recall": None,
            "hermes_v2_structured_fp_rate": None,
            "post_hoc_baseline": 0.12,
            "structured_improvement": None,
            "gate_contribution": False,
            "retro_070_partial": False,
            "inference_mode": "live_gpu",
            "honest_verdict": "ci_stub_gpu_required",
            "blocked_reason": "gpu_not_healthy",
            "gpu_status": str(gpu_status.get("models", [])),
        },
        status="blocked",
    )
    output_path = _repo_root / "results" / "experiment_654_hermes_v2_structured.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"GPU not healthy — writing blocked artifact. Status: {gpu_status}")
    tmpl.assert_deliverable_written()
    sys.exit(0)

# ---------------------------------------------------------------------------
# 6. Build live llm_caller via transformers on cuda:0
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from transformers import pipeline as hf_pipeline  # noqa: E402

_pipe = hf_pipeline(
    "text-generation",
    model="Qwen/Qwen3.5-0.8B",
    device=0,
    torch_dtype=torch.float16,
    max_new_tokens=256,
    do_sample=False,
)


def llm_caller(prompt: str, system: str) -> str:
    """Call Qwen3.5-0.8B on cuda:0, returning the generated text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        out = _pipe(messages)
        if isinstance(out, list) and out:
            gen = out[0]
            if isinstance(gen, dict):
                text = gen.get("generated_text", "")
                if isinstance(text, list) and text:
                    last = text[-1]
                    if isinstance(last, dict):
                        return last.get("content", "")
                return str(text)
        return str(out)
    except Exception as exc:  # noqa: BLE001
        print(f"llm_caller error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# 7. Build pipeline objects
# ---------------------------------------------------------------------------
verifier = SymCodeVerifier(llm_caller=lambda p: llm_caller(p, ""))
forcer = StructuredEquationForcer(llm_caller=llm_caller, verifier=verifier)
loop = HermesV2StructuredLoop(llm_caller, verifier, forcer, max_sentences=12)

# ---------------------------------------------------------------------------
# 8. Load live questions
# ---------------------------------------------------------------------------
live_pairs_path = _repo_root / "results" / "live_pairs_578.json"
with open(live_pairs_path) as f:
    live_pairs = json.load(f)

# 25 known-incorrect for TP measurement, 10 known-correct for FP measurement.
incorrect_pairs = [p for p in live_pairs if not p.get("is_correct", True)]
correct_pairs = [p for p in live_pairs if p.get("is_correct", False)]

incorrect_questions = [p["question"] for p in incorrect_pairs[:25]]
correct_questions = [p["question"] for p in correct_pairs[:10]]

# ---------------------------------------------------------------------------
# 9. Run structured loop (batch_size=5 per task spec)
# ---------------------------------------------------------------------------
_watchdog.start()

incorrect_results: list[HermesV2StructuredResult] = []
for i in range(0, len(incorrect_questions), 5):
    batch = incorrect_questions[i : i + 5]
    for q in batch:
        incorrect_results.append(loop.generate_structured(q))
    tmpl.checkpoint_save(
        {"incorrect_done": i + len(batch), "incorrect_results_count": len(incorrect_results)},
        step=i + len(batch),
    )

correct_results: list[HermesV2StructuredResult] = []
for i in range(0, len(correct_questions), 5):
    batch = correct_questions[i : i + 5]
    for q in batch:
        correct_results.append(loop.generate_structured(q))

_watchdog.stop()

# ---------------------------------------------------------------------------
# 10. Compute metrics
# ---------------------------------------------------------------------------
tp = sum(1 for r in incorrect_results if r.recall_contribution)
fp = sum(1 for r in correct_results if r.recall_contribution)

n_incorrect = len(incorrect_questions)
n_correct_q = len(correct_questions)

hermes_v2_structured_recall = tp / max(n_incorrect, 1)
hermes_v2_structured_fp_rate = fp / max(n_correct_q, 1)

# Baseline from Exp 641 (live hermes_v2_recall field)
hermes_v2_baseline_path = _repo_root / "results" / "experiment_641_hermes_v2_live.json"
hermes_v2_baseline = 0.12  # Exp 633 post-hoc default
if hermes_v2_baseline_path.exists():
    with open(hermes_v2_baseline_path) as f:
        exp641 = json.load(f)
    hermes_v2_baseline = exp641.get("hermes_v2_recall", 0.12)

post_hoc_baseline = 0.12
structured_improvement = hermes_v2_structured_recall - post_hoc_baseline

# ---------------------------------------------------------------------------
# 11. Determine honest_verdict
# ---------------------------------------------------------------------------
if hermes_v2_structured_recall >= 0.30:
    honest_verdict = "hermes_v2_structured_breakthrough"
elif hermes_v2_structured_recall > 0.12:
    honest_verdict = "hermes_v2_structured_improved"
else:
    honest_verdict = "hermes_v2_structured_no_improvement"

# ---------------------------------------------------------------------------
# 12. Write artifact
# ---------------------------------------------------------------------------
artifact = tmpl.build_result(
    {
        "schema": "carnot.hermes_v2_structured.v1",
        "n_questions": n_incorrect,
        "n_correct": n_correct_q,
        "hermes_v2_structured_recall": hermes_v2_structured_recall,
        "hermes_v2_structured_fp_rate": hermes_v2_structured_fp_rate,
        "post_hoc_baseline": post_hoc_baseline,
        "hermes_v2_baseline": hermes_v2_baseline,
        "structured_improvement": structured_improvement,
        "gate_contribution": hermes_v2_structured_recall >= 0.30,
        "retro_070_partial": hermes_v2_structured_recall > 0.12,
        "inference_mode": "live_gpu",
        "honest_verdict": honest_verdict,
    },
    status="success",
)

output_path = _repo_root / "results" / "experiment_654_hermes_v2_structured.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"hermes_v2_structured_recall: {hermes_v2_structured_recall:.3f}")
print(f"hermes_v2_structured_fp_rate: {hermes_v2_structured_fp_rate:.3f}")
print(f"structured_improvement: {structured_improvement:+.3f}")
print(f"honest_verdict: {honest_verdict}")
print(f"Artifact written to {output_path}")

# FINAL LINE — raises FileNotFoundError if the deliverable was not written.
tmpl.assert_deliverable_written()
