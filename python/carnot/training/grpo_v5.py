"""GRPO v5 continuous TinyV v2 reward + DualGPU split helpers.

Background
----------
Exp 1159 (GRPO v4) shipped a structural warm-up that beat the prior
mixed-reward baseline by +10.0pp on 47 FoVer questions; Exp 1173 then
attempted to layer TinyV-style false-negative *abstention* on top of v4
(reward set to 0 inside an uncertainty band) but blocked because the
active ``llama.cpp`` runtime had no GPU offload.

Exp 1184 takes a different bet: replace the binary correct / abstain
signal with a continuous ThinkPRM v2 energy score so every completion
contributes a graded reward, then split the 35B GGUF across both RTX
3090 GPUs (tensor_split=[0.5, 0.5]) so it fits in 48 GiB combined VRAM.

The full-phase reward is

    r_total = w_energy * r_energy + w_reflect * r_reflect

with default weights ``w_energy=0.6`` and ``w_reflect=0.4``. The warm-up
phase keeps Exp 1159's reflection-only schedule.

We MUST refuse to run on CPU. The 35B model on CPU is several seconds
per token, which would exceed every wall-budget the conductor sets and
produce honest_verdict=training_wall_hit instead of the gpu_offload
prerequisite signal we actually want. The
``gpu_offload_prerequisite_met`` field plus the
``honest_verdict="gpu_offload_prerequisite_not_met"`` label exist for
exactly that case so downstream planners can route the milestone to a
toolchain rebuild instead of re-attempting the same broken run.

Spec: REQ-LEARN-1184, SCENARIO-LEARN-1184, SCENARIO-LEARN-1185,
      SCENARIO-LEARN-1186.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default reward weights for the continuous TinyV v2 mix in the full phase.
# Exposed so tests can reference exactly the values the spec promises.
TINYV_V2_ENERGY_WEIGHT = 0.6
TINYV_V2_REFLECTION_WEIGHT = 0.4

# Exp 1159 reported trained_fraction_correct=0.26 on its 50-question GSM8K
# eval slice. Exp 1184 evaluates on the 47-question FoVer slice, but the v4
# baseline we compare against is the same 0.26 number from Exp 1159 so the
# delta is interpretable as "what does TinyV v2 buy us over the published
# v4 result". This is documented in REQ-LEARN-1184-4.
GRPO_V4_BASELINE_PASS_RATE = 0.26

# Evaluation slice size for Exp 1184. Held constant by REQ-LEARN-1184-3.
N_EVAL_QUESTIONS = 47

# DualGPU split — half the 35B model on each RTX 3090. main_gpu=0 means
# llama.cpp dispatches the first layer on GPU 0; tensor_split=[0.5,0.5]
# tells it to balance the rest of the model evenly across both devices.
DUALGPU_TENSOR_SPLIT = (0.5, 0.5)
DUALGPU_MAIN_GPU = 0
DUALGPU_N_GPU_LAYERS = -1

REQUIRED_GRPO_V5_ARTIFACT_FIELDS = (
    "gpu_offload_prerequisite_met",
    "training_completed",
    "dualgpu_confirmed",
    "training_tokens_per_sec",
    "grpo_v4_baseline_pass_rate",
    "grpo_v5_pass_rate",
    "grpo_v5_delta_pp",
    "tinyv_v2_mean_reward",
    "n_eval_questions",
    "honest_verdict",
)

# Exact set of allowed honest_verdict labels per REQ-LEARN-1184-6.
ALLOWED_HONEST_VERDICTS = frozenset(
    {
        "grpo_v5_above_v4",
        "grpo_v5_regression_vs_v4",
        "grpo_v5_no_delta",
        "gpu_offload_prerequisite_not_met",
        "training_wall_hit",
    }
)


@dataclass(frozen=True)
class TinyVV2Weights:
    """Continuous TinyV v2 reward weights for the GRPO v5 full phase."""

    energy_weight: float = TINYV_V2_ENERGY_WEIGHT
    reflection_weight: float = TINYV_V2_REFLECTION_WEIGHT

    def __post_init__(self) -> None:
        e = float(self.energy_weight)
        r = float(self.reflection_weight)
        # Per REQ-LEARN-1184-2 the weights must be non-negative and sum to 1.
        # We allow a small float-tolerance because callers may construct the
        # weights from rounded JSON values.
        if e < 0.0 or r < 0.0:
            raise ValueError(
                f"tinyv_v2 weights must be non-negative, got energy={e}, reflection={r}"
            )
        if abs((e + r) - 1.0) > 1e-9:
            raise ValueError(f"tinyv_v2 weights must sum to 1.0, got {e} + {r} = {e + r}")


def continuous_tinyv_v2_reward(
    energy_score: float,
    reflection_reward: float,
    *,
    weights: TinyVV2Weights | None = None,
) -> float:
    """Mix a ThinkPRM v2 energy score with the reflection reward.

    The energy score is a continuous value in ``[0, 1]`` from ThinkPRM v2
    (AUROC=0.9946 per Exp 1111). The reflection reward is the
    energy-before-minus-after signal from Exp 1159's verify-repair
    pipeline; it can be negative when repair makes things worse.

    Returns the linear combination
    ``w_energy * energy_score + w_reflect * reflection_reward`` which is
    exactly what GRPO v5 emits as the per-completion reward in the full
    phase. Round to 12 decimals so JSON artifacts stay byte-stable across
    runs (matches the Exp 1159 ``combine_phase_rewards`` convention).
    """
    w = weights or TinyVV2Weights()
    return float(
        round(
            w.energy_weight * float(energy_score) + w.reflection_weight * float(reflection_reward),
            12,
        )
    )


def continuous_tinyv_v2_reward_group(
    energy_scores: list[float],
    reflection_rewards: list[float],
    *,
    weights: TinyVV2Weights | None = None,
) -> list[float]:
    """Apply ``continuous_tinyv_v2_reward`` over an aligned reward group.

    Validates the two lists have matching length so a missing reflection
    reward never silently re-uses the prior completion's value — that bug
    would mute all GRPO advantages within the group and is exactly the
    failure mode REQ-LEARN-1184-3 wants to surface loudly.
    """
    if len(energy_scores) != len(reflection_rewards):
        raise ValueError(
            f"reward group length mismatch: {len(energy_scores)} vs {len(reflection_rewards)}"
        )
    return [
        continuous_tinyv_v2_reward(e, r, weights=weights)
        for e, r in zip(energy_scores, reflection_rewards, strict=True)
    ]


def derive_grpo_v5_honest_verdict(
    *,
    gpu_offload_prerequisite_met: bool,
    training_completed: bool,
    grpo_v5_delta_pp: float,
    no_delta_tolerance_pp: float = 0.005,
) -> str:
    """Map Exp 1184 outcomes to the canonical REQ-LEARN-1184-6 labels.

    Order matters: the GPU-offload prerequisite is checked before any
    training-related fields because if the prereq failed we never even
    started training, and reporting ``training_wall_hit`` would mask the
    real upstream blocker (a CPU-only ``llama.cpp`` build).

    The ``no_delta_tolerance_pp`` band exists because evaluating on 47
    questions can shift +/- 1pp by chance. A delta of exactly 0.0 is
    extremely unlikely on a 47-question slice, so we declare any delta
    inside ``+/- no_delta_tolerance_pp`` (default 0.005 = 0.5pp) as
    statistically indistinguishable from v4.
    """
    if not gpu_offload_prerequisite_met:
        return "gpu_offload_prerequisite_not_met"
    if not training_completed:
        return "training_wall_hit"
    delta = float(grpo_v5_delta_pp)
    if abs(delta) <= float(no_delta_tolerance_pp):
        return "grpo_v5_no_delta"
    if delta > 0.0:
        return "grpo_v5_above_v4"
    return "grpo_v5_regression_vs_v4"


def build_grpo_v5_artifact_fields(
    *,
    gpu_offload_prerequisite_met: bool,
    training_completed: bool,
    dualgpu_confirmed: bool,
    training_tokens_per_sec: float,
    grpo_v5_pass_rate: float,
    tinyv_v2_mean_reward: float,
    n_eval_questions: int,
    grpo_v4_baseline_pass_rate: float = GRPO_V4_BASELINE_PASS_RATE,
    no_delta_tolerance_pp: float = 0.005,
) -> dict[str, Any]:
    """Return the REQ-LEARN-1184-5 required artifact fields.

    Computes ``grpo_v5_delta_pp`` and ``honest_verdict`` from the inputs
    so the calling experiment script does not have to repeat the
    arithmetic and so the verdict is guaranteed to be one of the
    REQ-LEARN-1184-6 labels.
    """
    delta = float(round(float(grpo_v5_pass_rate) - float(grpo_v4_baseline_pass_rate), 4))
    verdict = derive_grpo_v5_honest_verdict(
        gpu_offload_prerequisite_met=gpu_offload_prerequisite_met,
        training_completed=training_completed,
        grpo_v5_delta_pp=delta,
        no_delta_tolerance_pp=no_delta_tolerance_pp,
    )
    if verdict not in ALLOWED_HONEST_VERDICTS:
        raise AssertionError(f"verdict {verdict!r} not in REQ-LEARN-1184-6 allowed set")
    return {
        "gpu_offload_prerequisite_met": bool(gpu_offload_prerequisite_met),
        "training_completed": bool(training_completed),
        "dualgpu_confirmed": bool(dualgpu_confirmed),
        "training_tokens_per_sec": float(training_tokens_per_sec),
        "grpo_v4_baseline_pass_rate": float(grpo_v4_baseline_pass_rate),
        "grpo_v5_pass_rate": float(grpo_v5_pass_rate),
        "grpo_v5_delta_pp": delta,
        "tinyv_v2_mean_reward": float(tinyv_v2_mean_reward),
        "n_eval_questions": int(n_eval_questions),
        "honest_verdict": verdict,
    }


def llama_cpp_supports_gpu_offload() -> bool:
    """Probe the active ``llama.cpp`` runtime for GPU offload capability.

    The C symbol ``llama_supports_gpu_offload`` is exposed by every CUDA-
    compiled llama.cpp build and absent from the CPU-only build, so this
    is a reliable preflight check before we try to load the 35B model.

    Wrapped in ``try/except`` because the ``llama_cpp`` package is an
    optional dependency in some test environments and we do not want a
    missing import to crash the entire script — the caller treats a
    ``False`` return identically to "GPU offload missing".
    """
    try:
        from llama_cpp import llama_cpp as llama_cpp_backend  # type: ignore

        return bool(llama_cpp_backend.llama_supports_gpu_offload())
    except Exception:
        return False


def detect_cuda_device_count() -> int:
    """Return ``torch.cuda.device_count()`` if torch+CUDA are visible, else 0.

    Used by REQ-LEARN-1184-1 / -7 to confirm the dual-RTX-3090 layout
    before loading the 35B model. ``0`` here is the canonical "no GPUs
    visible" sentinel even when the failure is something else (torch not
    installed, driver mismatch, etc.); the caller maps the same blocked
    artifact for any of those root causes.
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def gpu_offload_prerequisite_met(
    *,
    cuda_device_count: int,
    llama_cpp_gpu_offload: bool,
) -> bool:
    """Return True iff REQ-LEARN-1184-1 prerequisites both hold.

    Both checks are AND-combined: a CUDA-aware llama.cpp build with no
    visible GPUs is just as broken as two visible GPUs with a CPU-only
    llama.cpp build. The exp1184 script refuses to train in either case.
    """
    return bool(int(cuda_device_count) >= 2 and bool(llama_cpp_gpu_offload))
