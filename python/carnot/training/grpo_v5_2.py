"""GRPO v5.2 TinyV confidence-abstention helpers and artifact builder.

Background
----------
GRPO v4 (Exp 1159) shipped a structural warm-up that lifted the 35B
Qwen3.6-A3B baseline by +10pp on a 50-question GSM8K eval slice. The
arXiv 2506.10947 "Spurious Rewards" paper warns that random or
near-uniform reward signals can mimic real verifier signal at the
"Phase A structural warm-up" magnitude, so the v5 hypothesis must beat
v4 by *more than 3pp* (the paper's threshold) for the energy verifier
to count as adding real signal beyond structure.

Exp 1208 implements that test with a TinyV-style *abstention* rule:
when the ThinkPRM v2 verifier confidence sits in the uncertainty band
``[0.3, 0.7]``, the rollout's reward contribution is replaced with
zero (skipped) so it does not pollute the GRPO advantage estimate with
false-negative noise. v5.0 abstention failed before because of CPU-only
llama.cpp; this v5.2 attempt only runs after Exp 1207 verifies GPU
offload.

The honest_verdict label set differs from REQ-LEARN-1184's v5.1 set:
this experiment measures *improvement vs the v4 +10pp floor*, not raw
pass-rate delta, so its three "ran-to-completion" labels are
``improvement_above_v4`` / ``improvement_below_v4`` /
``improvement_equal_v4``. Two prereq labels (``blocked_no_gpu_offload``
and ``blocked_no_dualgpu``) plus ``training_wall_hit`` round out
REQ-LEARN-1208-7.

Spec: REQ-LEARN-1208, SCENARIO-LEARN-1208, SCENARIO-LEARN-1209,
      SCENARIO-LEARN-1210.
"""

from __future__ import annotations

from typing import Any

# v4's measured improvement floor over the .60 GRPO baseline. See
# REQ-LEARN-1208-4 and Exp 1159's trained_correct_count=13 vs
# baseline_correct_count=8 on a 50-question slice (10pp).
V4_BASELINE_IMPROVEMENT_PP = 10.0

# Per-paper Spurious-Reward threshold: a v5 attempt must improve by
# more than 3pp over v4 for the energy verifier signal to count as
# real beyond the structural warm-up's contribution.
SPURIOUS_REWARD_THRESHOLD_PP = 3.0

# Default TinyV uncertainty band. A verifier confidence inside this
# range is treated as "abstain" so its reward never enters the GRPO
# advantage. The band is inclusive on both ends — a confidence
# exactly at 0.3 or 0.7 still counts as uncertain.
TINYV_ABSTAIN_LOW = 0.3
TINYV_ABSTAIN_HIGH = 0.7

# Tolerance used to call an improvement "equal" to the v4 floor. A
# 50-question slice can shift +/-1pp by chance, so anything tighter
# than 0.5pp is statistically indistinguishable from the floor.
EQUAL_FLOOR_TOLERANCE_PP = 0.5

# DualGPU layout: half the 35B model on each RTX 3090. main_gpu=0
# means llama.cpp dispatches the first layer on GPU 0 and balances
# the rest evenly across both devices.
DUALGPU_TENSOR_SPLIT = (0.5, 0.5)
DUALGPU_MAIN_GPU = 0
DUALGPU_N_GPU_LAYERS = -1

REQUIRED_GRPO_V5_2_ARTIFACT_FIELDS = (
    "llama_cpp_gpu_offload",
    "cuda_device_count",
    "dualgpu_confirmed",
    "model_used",
    "training_completed",
    "tinyv_abstention_count",
    "tinyv_abstention_rate",
    "v4_baseline_improvement_pp",
    "v5_fraction_correct_before",
    "v5_fraction_correct_after",
    "improvement_over_baseline_pp",
    "beats_spurious_reward_threshold",
    "dualgpu_gpu0_utilization_pct",
    "dualgpu_gpu1_utilization_pct",
    "honest_verdict",
)

ALLOWED_HONEST_VERDICTS = frozenset(
    {
        "improvement_above_v4",
        "improvement_below_v4",
        "improvement_equal_v4",
        "blocked_no_gpu_offload",
        "blocked_no_dualgpu",
        "training_wall_hit",
    }
)


def tinyv_confidence_abstain(
    confidence: float,
    *,
    low: float = TINYV_ABSTAIN_LOW,
    high: float = TINYV_ABSTAIN_HIGH,
) -> bool:
    """Return True when ``confidence`` is inside the uncertainty band.

    Per REQ-LEARN-1208-2 the band is inclusive on both ends because a
    verifier whose confidence lands exactly at the boundary should
    still be treated as uncertain — being on the border is itself a
    signal that the verifier cannot decide. A non-finite input is
    treated as uncertain too: NaN means the verifier produced no
    decision, and abstaining is safer than passing a poisoned reward.
    """
    c = float(confidence)
    if c != c:  # NaN check without importing math
        return True
    return float(low) <= c <= float(high)


def apply_tinyv_abstention(
    confidences: list[float],
    rewards: list[float],
    *,
    low: float = TINYV_ABSTAIN_LOW,
    high: float = TINYV_ABSTAIN_HIGH,
) -> tuple[list[float], int]:
    """Replace rewards inside the uncertainty band with 0.0.

    Returns ``(filtered_rewards, abstention_count)``. The two input
    lists must have matching length per REQ-LEARN-1208-3 — a length
    mismatch silently aligning the wrong confidence to the wrong
    reward is exactly the failure mode this experiment is meant to
    surface, so we raise loudly instead.
    """
    if len(confidences) != len(rewards):
        raise ValueError(f"confidence/reward length mismatch: {len(confidences)} vs {len(rewards)}")
    filtered: list[float] = []
    abstention_count = 0
    for conf, rew in zip(confidences, rewards, strict=True):
        if tinyv_confidence_abstain(conf, low=low, high=high):
            filtered.append(0.0)
            abstention_count += 1
        else:
            filtered.append(float(rew))
    return filtered, abstention_count


def derive_grpo_v5_2_honest_verdict(
    *,
    llama_cpp_gpu_offload: bool,
    cuda_device_count: int,
    training_completed: bool,
    improvement_over_baseline_pp: float,
    equal_floor_tolerance_pp: float = EQUAL_FLOOR_TOLERANCE_PP,
) -> str:
    """Map Exp 1208 outcomes onto the REQ-LEARN-1208-7 verdict set.

    Order matters: prereq failures are reported even if a partial
    training number happens to exist, because reporting the training
    number would mask the upstream blocker. After prereqs, an
    incomplete training run is reported as ``training_wall_hit`` so
    the conductor can route the next milestone to a budget-extension
    fix instead of a "no progress" interpretation.
    """
    if not llama_cpp_gpu_offload:
        return "blocked_no_gpu_offload"
    if int(cuda_device_count) < 2:
        return "blocked_no_dualgpu"
    if not training_completed:
        return "training_wall_hit"
    delta = float(improvement_over_baseline_pp)
    if abs(delta) <= float(equal_floor_tolerance_pp):
        return "improvement_equal_v4"
    if delta > 0.0:
        return "improvement_above_v4"
    return "improvement_below_v4"


def build_grpo_v5_2_artifact_fields(
    *,
    llama_cpp_gpu_offload: bool,
    cuda_device_count: int,
    dualgpu_confirmed: bool,
    model_used: str,
    training_completed: bool,
    tinyv_abstention_count: int,
    tinyv_abstention_rate: float,
    v5_fraction_correct_before: float,
    v5_fraction_correct_after: float,
    dualgpu_gpu0_utilization_pct: float,
    dualgpu_gpu1_utilization_pct: float,
    v4_baseline_improvement_pp: float = V4_BASELINE_IMPROVEMENT_PP,
    spurious_reward_threshold_pp: float = SPURIOUS_REWARD_THRESHOLD_PP,
    equal_floor_tolerance_pp: float = EQUAL_FLOOR_TOLERANCE_PP,
) -> dict[str, Any]:
    """Return the REQ-LEARN-1208-6 required artifact fields.

    ``improvement_over_baseline_pp`` is computed as the v5 raw
    improvement (after - before, in pp) MINUS the v4 floor, so a
    positive number means v5 beats v4 by that many pp. The
    ``beats_spurious_reward_threshold`` flag is True only when v5
    beats the v4 floor by *more than* the Spurious-Reward threshold
    (default 3pp) — strict inequality because exactly equal to the
    threshold is still attributable to chance.
    """
    raw_improvement_pp = round(
        (float(v5_fraction_correct_after) - float(v5_fraction_correct_before)) * 100.0, 4
    )
    improvement_over_baseline_pp = round(raw_improvement_pp - float(v4_baseline_improvement_pp), 4)
    beats_threshold = bool(improvement_over_baseline_pp > float(spurious_reward_threshold_pp))
    verdict = derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=llama_cpp_gpu_offload,
        cuda_device_count=cuda_device_count,
        training_completed=training_completed,
        improvement_over_baseline_pp=improvement_over_baseline_pp,
        equal_floor_tolerance_pp=equal_floor_tolerance_pp,
    )
    if verdict not in ALLOWED_HONEST_VERDICTS:
        raise AssertionError(f"verdict {verdict!r} not in REQ-LEARN-1208-7 allowed set")
    return {
        "llama_cpp_gpu_offload": bool(llama_cpp_gpu_offload),
        "cuda_device_count": int(cuda_device_count),
        "dualgpu_confirmed": bool(dualgpu_confirmed),
        "model_used": str(model_used),
        "training_completed": bool(training_completed),
        "tinyv_abstention_count": int(tinyv_abstention_count),
        "tinyv_abstention_rate": float(tinyv_abstention_rate),
        "v4_baseline_improvement_pp": float(v4_baseline_improvement_pp),
        "v5_fraction_correct_before": float(v5_fraction_correct_before),
        "v5_fraction_correct_after": float(v5_fraction_correct_after),
        "improvement_over_baseline_pp": float(improvement_over_baseline_pp),
        "beats_spurious_reward_threshold": beats_threshold,
        "dualgpu_gpu0_utilization_pct": float(dualgpu_gpu0_utilization_pct),
        "dualgpu_gpu1_utilization_pct": float(dualgpu_gpu1_utilization_pct),
        "honest_verdict": verdict,
    }


def llama_cpp_supports_gpu_offload() -> bool:
    """Probe the active llama.cpp runtime for GPU offload capability.

    Returns False on any import or symbol error so callers can treat a
    missing optional dep identically to a CPU-only build. The
    ``llama_supports_gpu_offload`` C symbol is exposed only by CUDA-
    compiled llama.cpp, which makes it a reliable preflight before
    loading the 35B GGUF.
    """
    try:
        from llama_cpp import llama_cpp as llama_cpp_backend  # type: ignore

        return bool(llama_cpp_backend.llama_supports_gpu_offload())
    except Exception:
        return False


def detect_cuda_device_count() -> int:
    """Return ``torch.cuda.device_count()`` if torch+CUDA visible, else 0.

    We collapse "torch missing" / "driver mismatch" / "no GPUs visible"
    onto the same `0` sentinel because the experiment's blocked path
    is identical for any of those root causes — it cannot run a
    DualGPU split without two visible CUDA devices.
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0
