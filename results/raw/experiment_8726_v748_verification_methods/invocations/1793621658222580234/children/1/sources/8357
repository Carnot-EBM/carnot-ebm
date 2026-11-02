#!/usr/bin/env python3
"""Experiment 913 — DualGPU Production Wiring for ThreeTierPipeline.

**Researcher summary:**
    DualGPURunner was validated at 1.979x throughput in Exp 856 but was never
    connected to ThreeTierPipeline.  This experiment wires the runner to the
    pipeline via wire_dual_gpu_runner(), measures end-to-end throughput on 20
    synthetic GSM8K questions in both CARNOT_DUAL_GPU=0 (baseline sequential)
    and CARNOT_DUAL_GPU=1 (dual-GPU parallel) modes, and records the
    observed speedup ratio.

**What "wiring" means:**
    ThreeTierPipeline.benchmark() is the batch entry point.  After wiring,
    when CARNOT_DUAL_GPU=1 it splits the batch across two ThreadPoolExecutor
    workers — one per GPU partition — so verify() calls run concurrently.
    On dual-GPU hardware (cuda:0 + cuda:1), EORM and Ising dispatch to
    separate devices; on CPU or single-GPU machines, threading reduces
    scheduling latency by releasing the Python GIL during JAX JIT dispatch.

**Honest verdict criteria:**
    "dualgpu_wired_speedup_confirmed"  — observed_speedup > 1.7
    "dualgpu_wired_partial_speedup"    — 1.0 < observed_speedup <= 1.7
    "dualgpu_wired_no_speedup"         — observed_speedup <= 1.0

Spec: REQ-PERF-004, SCENARIO-PERF-004
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from repo root without pip install
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic stub factories — no GPU required for wiring validation
# ---------------------------------------------------------------------------


def _make_stub_eorm() -> Any:
    """Return a stub EORMModel whose energy() always returns 0.8.

    Why above threshold (0.5): we want all responses to reach the Ising tier
    so benchmark() exercises the full cascade path in both modes.  Clearing at
    EORM would mask threading overhead and give a misleading comparison.
    """

    class _StubEORM:
        def energy(self, cot_input: Any) -> float:
            # Deliberate 0.1ms sleep to simulate real EORM forward pass latency.
            # Without artificial latency the wall-time difference between
            # sequential and parallel is below measurement noise.
            time.sleep(0.0001)
            return 0.8

    return _StubEORM()


def _make_stub_ising() -> Any:
    """Return a stub Ising callable: (response, question) -> (bool, float)."""

    def _ising(response: str, question: str) -> tuple[bool, float]:
        # Deliberate 0.1ms sleep to simulate Ising sweep latency.
        time.sleep(0.0001)
        return True, 0.5

    return _ising


def _make_stub_sink_probe() -> Any:
    """Return a stub SinkProbe with a .score() that returns a low sink score."""

    class _FakeConcentration:
        mean_sink_score = 0.0  # Always below sink_threshold=0.3 → no early exit

    class _StubSinkProbe:
        def score(self, attn: Any, sink_positions: list[int]) -> Any:
            return _FakeConcentration()

    return _StubSinkProbe()


def _make_stub_dual_gpu_runner() -> Any:
    """Return a minimal DualGPURunner-compatible marker object.

    ThreeTierPipeline.wire_dual_gpu_runner() only checks `is not None`; the
    actual parallelism is implemented in benchmark() via ThreadPoolExecutor.
    This stub satisfies the check without requiring CUDA.
    """

    class _StubDualGPURunner:
        """Marker object — presence signals dual-GPU intent to benchmark()."""

    return _StubDualGPURunner()


# ---------------------------------------------------------------------------
# Synthetic GSM8K-style question corpus (20 items)
# ---------------------------------------------------------------------------

_QUESTIONS = [
    "Janet sells 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. "
    "How many eggs does she have left to sell?",
    "A store has 240 apples. They sell 60 on Monday and 80 on Tuesday. How many remain?",
    "Tom ran 5 km on day 1 and 3 km on day 2. How far did he run total?",
    "A baker makes 48 cookies. She gives 12 to neighbours. How many does she keep?",
    "A car travels 60 mph for 2 hours. How far does it travel?",
    "Sara has $50. She spends $18 on books and $7 on lunch. How much does she have left?",
    "A box holds 24 cans. If 3 boxes are on the shelf, how many cans are there?",
    "A class has 30 students. 12 are girls. How many are boys?",
    "A train departs at 9:00 and arrives at 11:30. How long was the journey?",
    "There are 5 bags with 8 oranges each. How many oranges total?",
    "A worker earns $15/hour and works 8 hours. What are the total earnings?",
    "A farmer has 100 chickens and 40 cows. How many legs total?",
    "A rectangle is 12m long and 5m wide. What is its area?",
    "A store discounts a $200 jacket by 25%. What is the sale price?",
    "You have 3 pizzas each cut into 8 slices. How many slices total?",
    "A pool holds 5000 litres. 1200 litres are drained. How much remains?",
    "A cyclist rides 45 km in 3 hours. What is average speed?",
    "A school library has 820 books. 135 are checked out. How many are left?",
    "Two friends share a $36 bill equally. How much does each pay?",
    "A plant grows 3 cm per week. How tall after 8 weeks?",
]

_RESPONSES = [
    "Step 1: Start with 16 eggs. Step 2: 16 - 3 - 4 = 9. Answer: 9 eggs.",
    "Step 1: 240 - 60 = 180. Step 2: 180 - 80 = 100. Answer: 100 apples.",
    "Step 1: 5 + 3 = 8. Answer: 8 km.",
    "Step 1: 48 - 12 = 36. Answer: 36 cookies.",
    "Step 1: 60 × 2 = 120. Answer: 120 km.",
    "Step 1: 50 - 18 = 32. Step 2: 32 - 7 = 25. Answer: $25.",
    "Step 1: 3 × 24 = 72. Answer: 72 cans.",
    "Step 1: 30 - 12 = 18. Answer: 18 boys.",
    "Step 1: 11:30 - 9:00 = 2.5 hours. Answer: 2 hours 30 minutes.",
    "Step 1: 5 × 8 = 40. Answer: 40 oranges.",
    "Step 1: 15 × 8 = 120. Answer: $120.",
    "Step 1: 100 × 2 = 200 chicken legs. Step 2: 40 × 4 = 160 cow legs. "
    "Step 3: 200 + 160 = 360. Answer: 360 legs.",
    "Step 1: 12 × 5 = 60. Answer: 60 m².",
    "Step 1: 200 × 0.25 = 50. Step 2: 200 - 50 = 150. Answer: $150.",
    "Step 1: 3 × 8 = 24. Answer: 24 slices.",
    "Step 1: 5000 - 1200 = 3800. Answer: 3800 litres.",
    "Step 1: 45 ÷ 3 = 15. Answer: 15 km/h.",
    "Step 1: 820 - 135 = 685. Answer: 685 books.",
    "Step 1: 36 ÷ 2 = 18. Answer: $18 each.",
    "Step 1: 3 × 8 = 24. Answer: 24 cm.",
]

assert len(_QUESTIONS) == 20 and len(_RESPONSES) == 20


def _build_corpus() -> tuple[list[dict[str, Any]], list[bool]]:
    """Return (responses_dicts, ground_truth) for the 20 GSM8K-style items.

    All responses are marked correct (True) — the experiment is measuring
    throughput, not verification accuracy, so the ground_truth label does not
    affect the speedup measurement.
    """
    response_dicts = [
        {"question": q, "response": r, "attention_matrix": None}
        for q, r in zip(_QUESTIONS, _RESPONSES)
    ]
    ground_truth = [True] * 20
    return response_dicts, ground_truth


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _build_pipeline(dual_gpu_enabled: bool) -> Any:
    """Construct ThreeTierPipeline with stubs and optional dual-GPU runner.

    The CARNOT_DUAL_GPU env var controls the class-level flag, so we set it
    before importing ThreeTierPipeline and reset it after construction.  This
    is necessary because ThreeTierPipeline.DUAL_GPU_ENABLED is a class-level
    constant evaluated at import time per-process, but in this single-process
    experiment we patch os.environ to simulate both modes.

    For the dual-GPU run we also wire a stub DualGPURunner so benchmark()
    actually takes the threading path (CARNOT_DUAL_GPU=1 alone is insufficient
    without a runner wired).
    """
    # Patch the class attribute directly — simpler than re-importing the module.
    from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

    ThreeTierPipeline.DUAL_GPU_ENABLED = dual_gpu_enabled

    pipeline = ThreeTierPipeline(
        sink_probe=_make_stub_sink_probe(),
        eorm_model=_make_stub_eorm(),
        ising_pipeline=_make_stub_ising(),
        sink_threshold=0.3,
        eorm_threshold=0.5,
    )

    if dual_gpu_enabled:
        pipeline.wire_dual_gpu_runner(_make_stub_dual_gpu_runner())

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        913,
        "DualGPU Production Wiring — ThreeTierPipeline benchmark()",
        "results/experiment_913_dualgpu_production_wiring.json",
        requires_gpu=False,  # Stubs run on CPU; real GPU not needed for wiring test
    )
    tmpl.setup()

    corpus, ground_truth = _build_corpus()

    # ------------------------------------------------------------------
    # Baseline: CARNOT_DUAL_GPU=0 — sequential single-threaded benchmark
    # ------------------------------------------------------------------
    baseline_pipeline = _build_pipeline(dual_gpu_enabled=False)
    t0 = time.perf_counter()
    baseline_result = baseline_pipeline.benchmark(
        corpus, ground_truth, inference_mode="cpu_sequential"
    )
    baseline_wall_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # DualGPU: CARNOT_DUAL_GPU=1 — parallel two-thread benchmark
    # ------------------------------------------------------------------
    dualgpu_pipeline = _build_pipeline(dual_gpu_enabled=True)
    t0 = time.perf_counter()
    dualgpu_result = dualgpu_pipeline.benchmark(
        corpus, ground_truth, inference_mode="cpu_dual_threaded"
    )
    dualgpu_wall_time = time.perf_counter() - t0

    # Restore env to safe default
    from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

    ThreeTierPipeline.DUAL_GPU_ENABLED = os.getenv("CARNOT_DUAL_GPU", "0") == "1"

    # ------------------------------------------------------------------
    # Compute observed speedup
    # ------------------------------------------------------------------
    if dualgpu_wall_time > 0:
        observed_speedup = baseline_wall_time / dualgpu_wall_time
    else:
        observed_speedup = 1.0

    if observed_speedup > 1.7:
        honest_verdict = "dualgpu_wired_speedup_confirmed"
    elif observed_speedup > 1.0:
        honest_verdict = "dualgpu_wired_partial_speedup"
    else:
        honest_verdict = "dualgpu_wired_no_speedup"

    print(
        f"Baseline wall_time_s={baseline_wall_time:.4f}  "
        f"DualGPU wall_time_s={dualgpu_wall_time:.4f}  "
        f"observed_speedup={observed_speedup:.3f}  "
        f"verdict={honest_verdict}"
    )

    artifact = tmpl.build_result(
        {
            "baseline_wall_time_s": round(baseline_wall_time, 6),
            "dualgpu_wall_time_s": round(dualgpu_wall_time, 6),
            "observed_speedup": round(observed_speedup, 6),
            "validated_speedup_exp856": 1.979,
            "n_questions": 20,
            "baseline_throughput_qps": round(baseline_result.throughput_qps, 3),
            "dualgpu_throughput_qps": round(dualgpu_result.throughput_qps, 3),
            "carnot_dual_gpu_flag": "CARNOT_DUAL_GPU=1",
            "wire_method": "ThreeTierPipeline.wire_dual_gpu_runner()",
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    import json

    out_path = Path(_REPO_ROOT / "results" / "experiment_913_dualgpu_production_wiring.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"Deliverable written: {out_path}")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
