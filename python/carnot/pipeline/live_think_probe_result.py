"""LiveThinkProbeResult — extends ThinkProbeV2Result with live-GPU provenance fields.

**Why this class exists (RETRO-036, Exp 465):**
    Exp 455 (ThinkProbeV2) reported RETRO-029 CLOSED in the conductor log but the
    result JSON file was absent at retrospective time — path mismatch between conductor
    spec and script output.  Exp 465 re-runs ThinkProbeV2 on live GPU and uses
    DeliverableGuard to guarantee the file is written.

    LiveThinkProbeResult adds three provenance fields to ThinkProbeV2Result:
        inference_mode — always 'live_gpu' for Exp 465 runs
        model_id       — HuggingFace model ID of the inference model
        gpu_used       — torch device string (e.g. 'cuda:0', 'cuda:0 (ROCm)')

    Having a distinct subclass lets the test suite assert on the live-GPU contract
    separately from the ThinkProbeV2 unit tests without modifying existing tests.

Spec: REQ-PROBE-008, REQ-PROBE-009,
      SCENARIO-PROBE-013, SCENARIO-PROBE-014
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.think_probe_v2 import ThinkProbeV2Result


@dataclass
class LiveThinkProbeResult(ThinkProbeV2Result):
    """ThinkProbeV2Result annotated with live-GPU provenance.

    Fields (in addition to those inherited from ThinkProbeV2Result)
    ---------------------------------------------------------------
    inference_mode : str
        Always ``'live_gpu'`` for Exp 465 live runs.  The field lets
        downstream tools distinguish a genuine GPU run from a
        deferred/simulated artifact (``'deferred_to_gpu'``).

    model_id : str
        HuggingFace model ID used for inference, e.g.
        ``'google/gemma-4-E4B-it'``.  Recorded so the artifact is
        self-describing — a reader does not need to dig into script
        source to know which model produced the results.

    gpu_used : str
        Torch device string as reported at load time, e.g. ``'cuda:0'``
        or ``'cuda:0 (ROCm)'``.  Recorded for hardware traceability;
        required by the "all headline results must have live GPU
        provenance" policy from CLAUDE.md.

    Spec: REQ-PROBE-009, SCENARIO-PROBE-014
    """

    inference_mode: str = "live_gpu"
    model_id: str = ""
    gpu_used: str = ""
