#!/usr/bin/env python3
"""Experiment 598: HISR Hindsight Credit Assignment + D-Wave Integration Validation.

**Researcher summary (REQ-LEARN-072, REQ-SAMPLE-034):**
    Two independent research threads combined into one slot:

    1. HISR (arXiv 2603.18683) — Hindsight Importance Score Reweighting applies
       a positional credit-assignment score to constraint violations: violations
       that immediately preceded a final incorrect answer score higher (closer
       to 1.0) than violations early in the chain.  Violations in correct chains
       score 0.0 (false positives).  This improves constraint-addition signal
       quality for ConstraintAdditionFromMemory.

    2. D-Wave integration — validates that the dwave-ocean-sdk package can be
       installed and that DWaveNealBackend (local SimulatedAnnealingSampler)
       runs within a comparable latency envelope to the JAX CPU baseline
       (ParallelIsingSampler).  This is the first Carnot experiment validating
       the D-Wave hardware path described in research-hardware-wishlist.md.

**Gate chain (every exit path writes the deliverable):**
    0. apply_env_autofix() FIRST.
    1. assert_live_or_ci_skip() — graceful skip in CI.
    2. ExperimentTimeoutWatchdog(598, timeout_minutes=25).
    3. HISR test: 20 synthetic violations (10 correct chain, 10 incorrect).
    4. D-Wave: try pip install dwave-ocean-sdk, then latency benchmark.
    5. Build artifact schema='carnot.hisr_dwave.v1'.
    6. tmpl.assert_deliverable_written() — FINAL LINE.

Spec: REQ-LEARN-072, REQ-SAMPLE-034, SCENARIO-LEARN-113, SCENARIO-LEARN-114,
      SCENARIO-SAMPLE-058, SCENARIO-SAMPLE-059
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix BEFORE any heavy imports (JAX, torch, etc.)
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step 1: assert_live_or_ci_skip — graceful skip in CI without live GPU.
# ---------------------------------------------------------------------------
from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

# ---------------------------------------------------------------------------
# Remaining imports (after env/live checks are established)
# ---------------------------------------------------------------------------
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(598, timeout_minutes=25)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.constraint_addition import ViolationPattern  # noqa: E402
from carnot.pipeline.hisr_weights import HISRWeighter  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402

_DELIVERABLE = "results/experiment_598_hisr_dwave.json"

tmpl = ExperimentTemplate(
    598,
    "HISR + D-Wave Cloud",
    _DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step 3: HISR test — 20 synthetic violations, 10 correct / 10 incorrect
# ---------------------------------------------------------------------------

# Build two chains of 10 violations each.
# Correct chain: carries and signs in a chain where the final answer was right.
_correct_violations = [
    ViolationPattern(type="carry", count=1, example_steps=["3+9=11 (carry)"]),
    ViolationPattern(type="sign", count=1, example_steps=["-(−3)=−3"]),
    ViolationPattern(type="unit", count=1, example_steps=["5km+500m"]),
    ViolationPattern(type="comparison", count=1, example_steps=["-3>-1"]),
    ViolationPattern(type="carry", count=1, example_steps=["17+5=21"]),
    ViolationPattern(type="sign", count=1, example_steps=["−7×−2=−14"]),
    ViolationPattern(type="unit", count=1, example_steps=["km vs m"]),
    ViolationPattern(type="comparison", count=1, example_steps=["x>y direction"]),
    ViolationPattern(type="carry", count=1, example_steps=["99+1=99"]),
    ViolationPattern(type="sign", count=1, example_steps=["−5+3=−8"]),
]

# Incorrect chain: same types but the chain ended in a wrong answer.
_incorrect_violations = [
    ViolationPattern(type="carry", count=1, example_steps=["48+13=50"]),
    ViolationPattern(type="sign", count=1, example_steps=["double neg error"]),
    ViolationPattern(type="unit", count=1, example_steps=["unit mixing"]),
    ViolationPattern(type="comparison", count=1, example_steps=["lt vs gt"]),
    ViolationPattern(type="carry", count=1, example_steps=["67+34=91"]),
    ViolationPattern(type="sign", count=1, example_steps=["sign flip"]),
    ViolationPattern(type="unit", count=1, example_steps=["m vs km"]),
    ViolationPattern(type="comparison", count=1, example_steps=["direction err"]),
    ViolationPattern(type="carry", count=1, example_steps=["carry not propagated"]),
    ViolationPattern(type="sign", count=1, example_steps=["negation missed"]),
]

_weighter = HISRWeighter()

_correct_weights = _weighter.compute_hindsight_score(
    _correct_violations, final_correct=True
)
_incorrect_weights = _weighter.compute_hindsight_score(
    _incorrect_violations, final_correct=False
)

# Correct chain: all scores must be 0.0 (false positives).
_correct_scores_all_zero = all(w.hindsight_score == 0.0 for w in _correct_weights)

# Incorrect chain: last violation must score > earliest violation.
_last_incorrect_score = _incorrect_weights[-1].hindsight_score
_first_incorrect_score = _incorrect_weights[0].hindsight_score
_incorrect_chain_ordered = _last_incorrect_score > _first_incorrect_score

hisr_credit_assignment_correct = _correct_scores_all_zero and _incorrect_chain_ordered

print(
    f"EXP-598 HISR: correct_zeros={_correct_scores_all_zero}, "
    f"last_score={_last_incorrect_score:.4f}, first_score={_first_incorrect_score:.4f}, "
    f"hisr_ok={hisr_credit_assignment_correct}"
)

# ---------------------------------------------------------------------------
# Step 4: D-Wave integration — try to install dwave-ocean-sdk, then benchmark
# ---------------------------------------------------------------------------

_pip_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "dwave-ocean-sdk", "--quiet"],
    capture_output=True,
    text=True,
    timeout=120,
)
dwave_installed = _pip_result.returncode == 0
print(
    f"EXP-598 D-Wave: pip install returncode={_pip_result.returncode}, "
    f"dwave_installed={dwave_installed}"
)

# Import AFTER pip attempt so the newly-installed package is importable.
from carnot.samplers.dwave_backend import DWaveNealBackend  # noqa: E402
from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_dwave = DWaveNealBackend()
dwave_available = _dwave.available

neal_latency_ms: float | None = None
cpu_latency_ms: float

# Benchmark D-Wave neal if available.
if dwave_available:
    neal_latency_ms = _dwave.latency_ms(100)
    print(f"EXP-598 D-Wave: neal_latency_ms={neal_latency_ms:.2f}")

# CPU baseline: ParallelIsingSampler.
import jax.random as jrandom  # noqa: E402

_rng = np.random.default_rng(1)
_J_np = _rng.standard_normal((100, 100)).astype(np.float32)
_J_np = (_J_np + _J_np.T) / 2.0
np.fill_diagonal(_J_np, 0.0)
_h_np = _rng.standard_normal(100).astype(np.float32)
_J_jax = jnp.asarray(_J_np)
_h_jax = jnp.asarray(_h_np)

_cpu_sampler = ParallelIsingSampler(
    n_warmup=200,
    n_samples=10,
    steps_per_sample=10,
    schedule=AnnealingSchedule(beta_init=0.1, beta_final=10.0),
    use_checkerboard=True,
)
_key = jrandom.PRNGKey(0)

_cpu_n_calls = 10
_cpu_elapsed = 0.0
for _ in range(_cpu_n_calls):
    _t0 = time.perf_counter()
    _cpu_sampler.sample(_key, _h_jax, _J_jax, beta=10.0)
    _cpu_elapsed += time.perf_counter() - _t0
cpu_latency_ms = (_cpu_elapsed / _cpu_n_calls) * 1000.0
print(f"EXP-598 CPU: cpu_latency_ms={cpu_latency_ms:.2f}")

speedup_ratio: float | None = None
if dwave_available and neal_latency_ms is not None and cpu_latency_ms > 0:
    speedup_ratio = cpu_latency_ms / neal_latency_ms

# ---------------------------------------------------------------------------
# Step 5: Determine honest verdict and build artifact
# ---------------------------------------------------------------------------

if dwave_available and speedup_ratio is not None and speedup_ratio > 1.0:
    honest_verdict = "dwave_faster"
elif dwave_available:
    honest_verdict = "dwave_installed_no_advantage"
else:
    honest_verdict = "dwave_not_installed"

print(
    f"EXP-598: dwave_available={dwave_available}, "
    f"speedup_ratio={speedup_ratio}, honest_verdict={honest_verdict}"
)

artifact = tmpl.build_result(
    {
        "hisr_credit_assignment_correct": hisr_credit_assignment_correct,
        "dwave_installed": dwave_installed,
        "dwave_available": dwave_available,
        "neal_latency_ms": neal_latency_ms,
        "cpu_latency_ms": cpu_latency_ms,
        "speedup_ratio": speedup_ratio,
        "honest_verdict": honest_verdict,
    },
    schema="carnot.hisr_dwave.v1",
    status="success",
)

AtomicResultWriter(str(_REPO_ROOT / _DELIVERABLE)).write(artifact)
print(f"EXP-598: artifact written, honest_verdict={honest_verdict}")

tmpl.assert_deliverable_written()
