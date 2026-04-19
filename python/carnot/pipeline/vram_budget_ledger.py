"""VRAMBudgetLedger — proactive VRAM feasibility forecasting before experiment launch.

**Why VRAMBudgetLedger exists (retro .37 improvement, RETRO-048 follow-up):**

    The conductor process (research_conductor.py) compiles a JAX computation graph
    at startup and holds ~9 GiB of GPU 0 VRAM for the entire session.  GPUVRAMGateV2
    (REQ-INFRA-049) checks *free* VRAM before each experiment launch, but it cannot
    account for the conductor's own footprint — the gate sees 15 GiB free (24 - 9),
    but the experiment declares "I need 16 GiB", and only learns this is impossible
    AFTER the subprocess is launched and hits OOM.

    The RETRO-048 fix (Exp 500) quantized Gemma4 to INT4 so it fits in 8-10 GiB.
    The RETRO-.37 improvement (Exp 501) goes one step further: the conductor should
    KNOW at milestone planning time which experiments will fit.  This converts the
    reactive 'deferred_to_gpu' / silent OOM pattern into a fast-fail with actionable
    root cause: "exp502 requires 18 GB but only 15 GB is available with conductor
    holding 9 GB — route conductor to CPU (JAX_PLATFORMS=cpu) to free full 24 GB."

**The CPU-routing insight:**
    If the conductor is started with JAX_PLATFORMS=cpu, its computation graph runs on
    CPU and it holds 0 GiB GPU VRAM.  All 24 GiB become available for inference models.
    VRAMBudgetLedger models this with conductor_vram_gb=0.0, letting the planner compare
    the GPU-routed vs CPU-routed feasibility sets before the milestone begins.

**Usage (conductor planning loop):**
    ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
    ledger.add_experiment("exp502", required_gb=18.0)
    ledger.add_experiment("exp503", required_gb=18.0)
    forecasts = ledger.check_all()
    # If not all feasible, restart conductor with JAX_PLATFORMS=cpu and re-plan.

Spec: REQ-INFRA-054, REQ-INFRA-055, REQ-INFRA-056,
      SCENARIO-INFRA-062, SCENARIO-INFRA-063, SCENARIO-INFRA-064
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# VRAMForecast
# ---------------------------------------------------------------------------


@dataclass
class VRAMForecast:
    """Feasibility forecast for a single experiment's VRAM requirement.

    Fields
    ------
    exp_id : str
        The experiment identifier (e.g. 'exp502').
    is_feasible : bool
        True iff required_gb <= available_gb (the experiment fits in free VRAM).
    required_gb : float
        VRAM the experiment model needs (from the ledger manifest).
    available_gb : float
        VRAM available for the experiment = gpu_total_gb - conductor_vram_gb.
    blocking_experiment : str | None
        None when is_feasible=True.  When is_feasible=False, set to exp_id itself
        — the identity of the blocking requirement.  Future extensions may carry
        the name of a prior experiment that consumed VRAM, but in the current
        single-experiment-at-a-time model, the experiment is its own blocker.

    Spec: REQ-INFRA-054, SCENARIO-INFRA-062/063/064
    """

    exp_id: str
    is_feasible: bool
    required_gb: float
    available_gb: float
    blocking_experiment: Optional[str]

    @property
    def headroom_gb(self) -> float:
        """Signed headroom: positive means the experiment fits with room to spare,
        negative means it overflows by that many GiB.

        Why signed rather than abs: negative headroom is the actionable signal that
        tells the operator exactly how many GiB to recover (e.g. by CPU-routing the
        conductor, which frees 9 GiB and often flips a -3 GiB overflow to +6 GiB).
        """
        return self.available_gb - self.required_gb

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict for embedding in experiment artifacts."""
        return {
            "exp_id": self.exp_id,
            "is_feasible": self.is_feasible,
            "required_gb": self.required_gb,
            "available_gb": self.available_gb,
            "headroom_gb": self.headroom_gb,
            "blocking_experiment": self.blocking_experiment,
        }


# ---------------------------------------------------------------------------
# VRAMBudgetLedger
# ---------------------------------------------------------------------------


class VRAMBudgetLedger:
    """Proactive VRAM feasibility forecaster for planned experiment sequences.

    The ledger maintains a registry of (exp_id -> required_gb) entries and
    computes, for each, whether it will fit given the GPU's total VRAM minus the
    conductor process's own footprint.

    **Why conductor_vram_gb is a first-class parameter:**
        The conductor holds ~9 GiB GPU 0 VRAM in default (JAX-GPU) mode.
        GPUVRAMGateV2 cannot subtract this: it reads pynvml free VRAM, which already
        has the conductor's allocation subtracted, but the gate's threshold is a
        fixed 'min_free_gb' that doesn't know how big the incoming model is.
        The ledger makes the conductor's footprint explicit so the planner can say
        "with conductor on GPU we have 15 GiB free; exp502 needs 18 GiB — not feasible.
        Switch conductor to CPU for 24 GiB free — now exp502 fits."

    Parameters
    ----------
    conductor_vram_gb : float
        VRAM (GiB) consumed by the conductor process itself.
        Default 9.0 based on JAX-compiled graph observations (Exp 500 retro).
        Set to 0.0 to model a CPU-routed conductor (JAX_PLATFORMS=cpu).
    gpu_total_gb : float
        Total VRAM (GiB) on the primary GPU.
        Default 24.0 (RTX 3090 / 3090 Ti observed in Exp 480+).

    Spec: REQ-INFRA-054, REQ-INFRA-055, REQ-INFRA-056,
          SCENARIO-INFRA-062, SCENARIO-INFRA-063, SCENARIO-INFRA-064
    """

    def __init__(
        self,
        conductor_vram_gb: float = 9.0,
        gpu_total_gb: float = 24.0,
    ) -> None:
        self.conductor_vram_gb = conductor_vram_gb
        self.gpu_total_gb = gpu_total_gb
        # Ordered registry: exp_id -> required_gb
        self._experiments: dict[str, float] = {}

    @property
    def available_gb(self) -> float:
        """VRAM available for experiment models = total - conductor footprint.

        Why property rather than stored field: conductor_vram_gb might be mutated
        by the planner after construction (e.g. switching from GPU to CPU routing),
        and available_gb should always reflect the current state.
        """
        return self.gpu_total_gb - self.conductor_vram_gb

    def add_experiment(self, exp_id: str, required_gb: float) -> None:
        """Register an experiment's peak VRAM requirement.

        Parameters
        ----------
        exp_id : str
            Unique identifier for the experiment (e.g. 'exp502').
        required_gb : float
            Peak GPU VRAM the experiment's model(s) will consume at inference time,
            in GiB.  Use the observed peak from prior runs or the model's published
            quantized size.

        Why 'peak' not 'average':
            VRAM is not shared — allocation is high-water-mark.  If you size for the
            average and the peak exceeds available, you get OOM.
        """
        self._experiments[exp_id] = required_gb

    def check_feasibility(self, exp_id: str) -> VRAMForecast:
        """Check whether a registered experiment will fit in available VRAM.

        Parameters
        ----------
        exp_id : str
            Experiment identifier, must have been registered via add_experiment().

        Returns
        -------
        VRAMForecast
            is_feasible=True when required_gb <= available_gb.
            blocking_experiment=None when feasible, exp_id when not feasible
            (the experiment itself is the blocker — it exceeds the budget).

        Raises
        ------
        KeyError
            If exp_id was not registered via add_experiment().

        Spec: REQ-INFRA-054, REQ-INFRA-055, SCENARIO-INFRA-062/063/064
        """
        required_gb = self._experiments[exp_id]
        avail = self.available_gb
        is_feasible = required_gb <= avail
        return VRAMForecast(
            exp_id=exp_id,
            is_feasible=is_feasible,
            required_gb=required_gb,
            available_gb=avail,
            blocking_experiment=None if is_feasible else exp_id,
        )

    def check_all(self) -> list[VRAMForecast]:
        """Check feasibility for all registered experiments, in registration order.

        Returns
        -------
        list[VRAMForecast]
            One VRAMForecast per registered experiment, in the order they were added.
            Experiments that don't fit are flagged with is_feasible=False and
            blocking_experiment set to their own exp_id.

        Why check_all instead of just check_feasibility in a loop:
            The conductor planning loop needs a single artifact that captures the full
            milestone feasibility picture.  check_all() returns it in one call so the
            result can be embedded directly in the experiment artifact without a
            secondary comprehension at the call site.
        """
        return [self.check_feasibility(exp_id) for exp_id in self._experiments]

    def to_yaml(self) -> str:
        """Serialize the ledger to a YAML string for embedding in conductor manifests.

        The YAML format is human-readable and suitable for committing alongside the
        milestone plan.  It captures both the ledger parameters and the full registry
        so a future conductor can reconstruct it without re-running the experiment.

        Returns
        -------
        str
            YAML string with conductor_vram_gb, gpu_total_gb, available_gb,
            and an 'experiments' mapping of exp_id -> required_gb.
        """
        data = {
            "conductor_vram_gb": self.conductor_vram_gb,
            "gpu_total_gb": self.gpu_total_gb,
            "available_gb": self.available_gb,
            "experiments": dict(self._experiments),
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
