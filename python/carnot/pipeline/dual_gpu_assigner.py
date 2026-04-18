"""DualGPUAssigner — assign model specs to separate GPUs for parallel execution.

**Why this exists (RETRO-034, milestone .34):**
    GPU 1 (RTX 3090, 24 GB VRAM) was IDLE for the ENTIRE milestone .34.
    DualGPURunner exists in the codebase but was NOT wired into ExperimentTemplate's
    setup_gpu() path.  The result: every dual-model experiment ran sequentially on
    GPU 0 while GPU 1 sat idle — 48 GB of VRAM wasted, every experiment taking 2×
    longer than necessary.

    DualGPUAssigner is the missing glue layer.  It takes a list of model_specs,
    checks eligibility (≥2 models, CARNOT_FORCE_LIVE=1, ≥2 GPUs), and assigns
    each model to its own GPU by injecting ``device_map={'': 'cuda:N'}`` into
    the spec dict.  ExperimentTemplate.setup_gpu() calls it when eligible.

**Why device_map={'': 'cuda:N'} not device_map='auto':**
    device_map='auto' lets the HuggingFace loader allocate layers across all
    visible GPUs for offloading.  With two models, the loader spreads model A
    across GPU0+GPU1 and model B similarly — both models share both GPUs but
    neither GPU gets a clean forward pass on its own silicon.  This is the
    RETRO-025 zombie pattern.  {'': 'cuda:N'} pins every layer of model N to
    GPU N, giving each model a clean dedicated forward pass.

Spec: REQ-INFRA-034, SCENARIO-INFRA-042
"""

from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)


class DualGPUAssigner:
    """Assign model specs to dedicated GPUs for parallel dual-model experiments.

    Parameters
    ----------
    model_specs : list[dict]
        Each spec must have at minimum a ``'name'`` key.  The assigner will
        inject ``'gpu'`` (integer index) and ``'device_map'`` (pinned map) into
        each spec when eligible.
    n_gpus : int
        Number of CUDA GPUs detected at runtime.  Pass 0 in CI/CPU environments.

    Spec: REQ-INFRA-034, SCENARIO-INFRA-042
    """

    def __init__(self, model_specs: list[dict[str, Any]], n_gpus: int) -> None:
        self._specs = model_specs
        self._n_gpus = n_gpus

    def is_dual_gpu_eligible(self) -> bool:
        """Return True when dual-GPU assignment should be applied.

        Conditions (ALL must hold):
        1. len(model_specs) >= 2  — need at least 2 models to distribute
        2. CARNOT_FORCE_LIVE=1    — live GPU mode; CI skips this
        3. n_gpus >= 2            — at least 2 physical GPUs present

        Why gate on CARNOT_FORCE_LIVE: in CI there are no GPUs and
        model_specs may still list ≥2 models for unit-test coverage.
        Without this gate, assign() would inject cuda:N device maps that
        crash immediately on a CPU-only machine.

        Returns
        -------
        bool
        """
        force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        return len(self._specs) >= 2 and force_live and self._n_gpus >= 2

    def assign(self) -> list[dict[str, Any]]:
        """Assign GPU indices and device maps to model specs.

        When eligible (is_dual_gpu_eligible() is True):
            - spec[i]['gpu'] = i
            - spec[i]['device_map'] = {'': f'cuda:{i}'}
            A warning is logged for models beyond index n_gpus-1 — they are
            assigned to the last available GPU (graceful degradation).

        When NOT eligible (CI mode, single GPU, or CARNOT_FORCE_LIVE not set):
            Returns model_specs unchanged.  No GPU indices or device maps are
            injected, so the caller's existing values (if any) are preserved.

        Returns
        -------
        list[dict]
            The same model_specs list (mutated in-place and returned).
        """
        if not self.is_dual_gpu_eligible():
            _log.debug(
                "DualGPUAssigner: not eligible (n_gpus=%d, n_specs=%d, CARNOT_FORCE_LIVE=%s) "
                "— model_specs unchanged",
                self._n_gpus,
                len(self._specs),
                os.environ.get("CARNOT_FORCE_LIVE", "0"),
            )
            return self._specs

        for i, spec in enumerate(self._specs):
            gpu_idx = min(i, self._n_gpus - 1)
            if i >= self._n_gpus:
                _log.warning(
                    "DualGPUAssigner: model %d ('%s') exceeds GPU count %d — "
                    "assigning to GPU %d (last available).  "
                    "Consider reducing model count for true parallelism.",
                    i,
                    spec.get("name", f"model_{i}"),
                    self._n_gpus,
                    gpu_idx,
                )
            spec["gpu"] = gpu_idx
            spec["device_map"] = {"": f"cuda:{gpu_idx}"}

        _log.info(
            "DualGPUAssigner: assigned %d models across %d GPUs (cuda:0..cuda:%d)",
            len(self._specs),
            self._n_gpus,
            self._n_gpus - 1,
        )
        return self._specs
