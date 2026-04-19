"""GPU1 routing fix — verifies that Qwen3.5-0.8B forward passes run on cuda:1.

**Researcher summary (RETRO-052, Exp 529):**
    Exp 517 confirmed gpu1_compute_pct=0.0 even when transformers was loaded with
    device_map={'': 'cuda:1'}.  The root cause hypothesis: transformers respects
    device_map at VRAM allocation time (weights land on cuda:1) but a
    backend-level dispatch overrides the forward pass back to cuda:0 in some
    versions.

    This module provides three diagnostic primitives used by Exp 529:

    1. ``GPU1RoutingResult`` — typed result capturing device used, utilization
       measured during inference, and an honest verdict.

    2. ``force_cuda1_device_map(model_id)`` — returns the explicit device_map dict
       that FORCES every transformer layer onto cuda:1.  Also records the model_id
       in the returned dict for traceability.  Never returns 'auto' or an integer
       shorthand — 'auto' is the root cause of RETRO-025.

    3. ``verify_model_on_device(model, expected_device_id)`` — inspects the first
       trainable parameter of the model and returns True only when it lives on the
       expected CUDA device index.  This catches the case where transformers silently
       moves a loaded model back to a different device.

**Why triple-layer cuda:1 constraint (belt-and-suspenders):**
    Layer 1: device_map={'': 'cuda:1'} at from_pretrained() tells the
             HuggingFace accelerate dispatch layer which device to use.
    Layer 2: model.to('cuda:1') after load is a PyTorch-level override that
             reassigns ALL buffers and parameters, bypassing accelerate's
             device map registry.
    Layer 3: verify_model_on_device() confirms the model parameters actually
             reside on cuda:1 before inference begins.

    If three independent layers all agree that the model is on cuda:1 and GPU 1
    utilization is STILL 0%, the fault lies in the nvml sampling timing — not
    in the device routing.  That branch sets routing_verified=False and
    honest_verdict='gpu1_still_idle' so the next experiment can focus on the
    nvml sampling window.

Spec: REQ-INFRA-071, REQ-INFRA-072, SCENARIO-INFRA-081, SCENARIO-INFRA-082
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# ---------------------------------------------------------------------------
# GPU1RoutingResult
# ---------------------------------------------------------------------------


@dataclass
class GPU1RoutingResult:
    """Typed result from the GPU1 routing fix verification pass.

    Fields
    ------
    device_used : str
        The device string the model was loaded onto (e.g. 'cuda:1').
        'unknown' if verify_model_on_device could not inspect parameters.
    gpu1_compute_pct_during_inference : float
        Mean compute utilization percentage on GPU 1 measured by
        nvmlDeviceGetUtilizationRates() during the 20 inference passes.
        0.0 in CI / gpu_required mode (no GPU hardware).
    routing_verified : bool
        True iff gpu1_compute_pct_during_inference > 10.0.
        The 10% threshold matches RETRO-052's definition of "active compute".
    honest_verdict : str
        One of:
        - 'gpu1_active'      — routing verified, GPU 1 is computing (RETRO-052 closed)
        - 'gpu1_still_idle'  — live GPU present but GPU 1 still at 0% despite triple fix
        - 'gpu_required'     — no live GPU in this environment; experiment skipped

    Spec: REQ-INFRA-071, SCENARIO-INFRA-081
    """

    device_used: str
    gpu1_compute_pct_during_inference: float
    routing_verified: bool
    honest_verdict: str


# ---------------------------------------------------------------------------
# force_cuda1_device_map
# ---------------------------------------------------------------------------


def force_cuda1_device_map(model_id: str) -> Dict[str, Any]:
    """Return device_map dict that forces ALL layers of *model_id* onto cuda:1.

    Why not device_map='auto' or device_map=1 (integer)?
        'auto' is the root cause of RETRO-025: it distributes layers across
        available GPUs based on free VRAM, which puts some layers on GPU 0
        while allocating VRAM on GPU 1.  The forward pass then runs on GPU 0.

        Integer device maps (e.g. device_map=1) are not supported by all
        accelerate versions and cause a TypeError in transformers >= 4.38.

        The only form guaranteed to pin ALL layers to a single device across
        accelerate versions is {'': 'cuda:1'} — the empty string is the
        accelerate convention for "all unnamed modules go here".

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (e.g. 'Qwen/Qwen2.5-0.5B').  Embedded in the
        returned dict under the '_model_id' key for traceability in logs.

    Returns
    -------
    Dict[str, Any]
        A device_map dict with the empty-string sentinel key pointing to
        'cuda:1', plus '_model_id' for traceability.  Pass this dict to
        ``AutoModelForCausalLM.from_pretrained(..., device_map=result)``.
        The '_model_id' key is ignored by transformers but survives logging.

    Spec: REQ-INFRA-072, SCENARIO-INFRA-082
    """
    return {
        "": "cuda:1",
        "_model_id": model_id,
    }


# ---------------------------------------------------------------------------
# verify_model_on_device
# ---------------------------------------------------------------------------


def verify_model_on_device(model: Any, expected_device_id: int) -> bool:
    """Return True iff the model's first parameter lives on *expected_device_id*.

    Inspects the first trainable parameter (via ``model.parameters()``) to
    determine which CUDA device the model actually occupies.  This is the
    ground-truth check: regardless of what device_map was passed to
    from_pretrained(), if the parameter tensor is on the wrong device then
    the forward pass will run on the wrong device.

    Why the first parameter and not all parameters?
        In a correctly-loaded single-device model every parameter is on the
        same device — checking all is redundant.  In a distributed (multi-GPU
        pipeline-parallel) model the first parameter is conventionally on the
        head device that drives the forward pass.

    Why ``expected_device_id: int`` instead of a device string?
        Integer device IDs are canonical; device strings may vary in format
        ('cuda:1' vs 'cuda1' vs 'gpu:1') across PyTorch versions.  Using
        ``tensor.device.index`` avoids string-format ambiguity.

    Parameters
    ----------
    model : Any
        A PyTorch nn.Module or compatible object with a ``parameters()``
        iterator.  Must not be None.
    expected_device_id : int
        Zero-based CUDA device index (0 = GPU 0, 1 = GPU 1).

    Returns
    -------
    bool
        True if the first parameter's device index matches *expected_device_id*
        AND that device is CUDA (not CPU, MPS, etc.).
        False in all other cases: wrong device index, CPU-only model, no
        parameters, or any AttributeError from inspecting the device.

    Spec: REQ-INFRA-072, SCENARIO-INFRA-082
    """
    try:
        first_param = next(model.parameters())
        dev = first_param.device
        # device.type is 'cuda', 'cpu', 'mps', etc.
        # device.index is None for CPU, an int for CUDA.
        if dev.type != "cuda":
            return False
        return dev.index == expected_device_id
    except (StopIteration, AttributeError):
        # model has no parameters or does not expose .device
        return False
