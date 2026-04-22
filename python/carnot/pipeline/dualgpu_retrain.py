"""DualGPU parallel retraining — run EORM and JEPA concurrently on separate devices.

**Why this exists (REQ-INFRA-091):**
    Exp 383 (Combined EORM+JEPA retrain) appeared in the slowest-5 list for three
    consecutive milestones (.46, .47, .48), contributing ~62 min each time.  The
    two retraining jobs are fully independent — EORM touches cuda:0 tensors, JEPA
    touches cuda:1 tensors — so there is no correctness reason to run them sequentially.

    DualGPURetrain submits both jobs to a two-worker ThreadPoolExecutor so they run
    in true wall-clock parallel, cutting Exp 383's contribution from ~62 min to ~35 min.

**How to use:**
    config = DualGPURetrainConfig(eorm_device='cuda:0', jepa_device='cuda:1')
    retrain = DualGPURetrain(config)
    results = retrain.run_parallel(eorm_train_fn, jepa_train_fn)
    # results == {'eorm': <eorm_return_value>, 'jepa': <jepa_return_value>}

**Fallback behaviour:**
    When fewer than 2 GPUs are available, pass eorm_device='cpu' and jepa_device='cpu'.
    run_parallel() does not check device availability — that is the caller's responsibility.
    The ThreadPoolExecutor runs both functions on the same CPU but still in separate threads,
    which at minimum avoids Python GIL contention for I/O-bound or NumPy/JAX work.

Spec: REQ-INFRA-091, SCENARIO-INFRA-097, SCENARIO-INFRA-098
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

_log = logging.getLogger(__name__)


def _count_cuda_gpus() -> int:
    """Return the number of CUDA GPUs available on this host.

    Returns 0 when torch is not installed or no CUDA device is present.
    Called at retrain time so we do not force CUDA initialisation at import time.
    """
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.device_count()
    except Exception:
        return 0

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class DualGPURetrainConfig:
    """Configuration for a dual-GPU parallel retrain run.

    Attributes
    ----------
    eorm_device : str
        PyTorch device string for the EORM (Energy-based Output Reconstruction Model)
        training job.  Use 'cuda:0' when the first RTX 3090 is available, 'cpu' otherwise.
    jepa_device : str
        PyTorch device string for the JEPA (Joint Embedding Predictive Architecture)
        training job.  Use 'cuda:1' when the second RTX 3090 is available, 'cpu' otherwise.
    """

    eorm_device: str
    jepa_device: str


class DualGPURetrain:
    """Run EORM training on eorm_device and JEPA training on jepa_device concurrently.

    This class is device-agnostic: it accepts any two callables and runs them in parallel
    via ThreadPoolExecutor.  The caller is responsible for ensuring that eorm_fn uses
    config.eorm_device and jepa_fn uses config.jepa_device — this class does not move
    tensors between devices.

    Parameters
    ----------
    config : DualGPURetrainConfig
        Device assignment for the two training jobs.
    """

    def __init__(self, config: DualGPURetrainConfig) -> None:
        self.config = config

    def run_parallel(self, eorm_fn: Callable[[], T], jepa_fn: Callable[[], U]) -> dict:
        """Run EORM training on eorm_device and JEPA training on jepa_device concurrently.

        Submits both callables to a 2-worker ThreadPoolExecutor and blocks until both
        complete.  Exceptions from either callable propagate out of this method — they
        are not caught here, because swallowed exceptions would silently corrupt results.

        Parameters
        ----------
        eorm_fn : Callable[[], T]
            Zero-argument callable that runs EORM training and returns a result.
            Must target config.eorm_device for its tensor operations.
        jepa_fn : Callable[[], U]
            Zero-argument callable that runs JEPA training and returns a result.
            Must target config.jepa_device for its tensor operations.

        Returns
        -------
        dict
            {'eorm': <return value of eorm_fn>, 'jepa': <return value of jepa_fn>}
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            eorm_future = executor.submit(eorm_fn)
            jepa_future = executor.submit(jepa_fn)
            eorm_result = eorm_future.result()
            jepa_result = jepa_future.result()
        return {"eorm": eorm_result, "jepa": jepa_result}

    def retrain_parallel(
        self,
        eorm_fn: "Callable[..., dict[str, Any]]",
        jepa_fn: "Callable[..., dict[str, Any]]",
        eorm_device: str = "cuda:0",
        jepa_device: str = "cuda:1",
    ) -> "dict[str, Any]":
        """Run EORM and JEPA retraining concurrently; fall back to sequential on 1 GPU.

        This is the REQ-INFRA-049 entry point — the validated Exp 685 ThreadPoolExecutor
        pattern (2.0175x speedup) as a production-ready method.  Each callable should
        accept a `device` keyword argument; if it does not, it is called without one.

        Single-GPU fallback (SCENARIO-INFRA-058):
            When fewer than 2 CUDA GPUs are detected, both callables run sequentially
            on cuda:0 (or cpu if no GPU).  The result dict includes
            `fallback_reason: "single_gpu"` so callers can detect the degraded path.

        Parameters
        ----------
        eorm_fn : callable
            EORM training callable.  Called as `eorm_fn(device=eorm_device)`.
        jepa_fn : callable
            JEPA training callable.  Called as `jepa_fn(device=jepa_device)`.
        eorm_device : str
            CUDA device for EORM (default "cuda:0").
        jepa_device : str
            CUDA device for JEPA (default "cuda:1").

        Returns
        -------
        dict with keys:
            eorm_result (dict): return value of eorm_fn.
            jepa_result (dict): return value of jepa_fn.
            wall_time_s (float): parallel (or sequential fallback) wall-clock seconds.
            speedup_vs_sequential (float): always 1.0 (sequential baseline not re-run here).
            eorm_device (str): device used for EORM.
            jepa_device (str): device used for JEPA.
            fallback_reason (str): present only on single-GPU host; value "single_gpu".

        Spec: REQ-INFRA-049, SCENARIO-INFRA-058
        """
        n_gpus = _count_cuda_gpus()

        if n_gpus < 2:
            # Single-GPU fallback: run sequentially on cuda:0 (or cpu if no GPU at all).
            _log.warning(
                "DualGPURetrain.retrain_parallel: only %d GPU(s) detected — "
                "falling back to sequential execution (SCENARIO-INFRA-058)",
                n_gpus,
            )
            fallback_device = eorm_device if n_gpus >= 1 else "cpu"
            t0 = time.perf_counter()
            eorm_result = _call_with_device(eorm_fn, fallback_device)
            jepa_result = _call_with_device(jepa_fn, fallback_device)
            wall_time_s = round(time.perf_counter() - t0, 3)
            return {
                "eorm_result": eorm_result,
                "jepa_result": jepa_result,
                "wall_time_s": wall_time_s,
                "speedup_vs_sequential": 1.0,
                "fallback_reason": "single_gpu",
                "eorm_device": fallback_device,
                "jepa_device": fallback_device,
            }

        # Dual-GPU path: ThreadPoolExecutor with 2 workers, one per GPU.
        _log.info(
            "DualGPURetrain.retrain_parallel: launching parallel — EORM on %s, JEPA on %s",
            eorm_device,
            jepa_device,
        )
        t_parallel = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            eorm_future = executor.submit(_call_with_device, eorm_fn, eorm_device)
            jepa_future = executor.submit(_call_with_device, jepa_fn, jepa_device)
            eorm_result = eorm_future.result()
            jepa_result = jepa_future.result()
        wall_time_s = round(time.perf_counter() - t_parallel, 3)

        _log.info("DualGPURetrain.retrain_parallel: done in %.3f s", wall_time_s)
        return {
            "eorm_result": eorm_result,
            "jepa_result": jepa_result,
            "wall_time_s": wall_time_s,
            "speedup_vs_sequential": 1.0,
            "eorm_device": eorm_device,
            "jepa_device": jepa_device,
        }


def _call_with_device(fn: "Callable[..., Any]", device: str) -> Any:
    """Call fn(device=device) if accepted; otherwise call fn().

    Why this helper: pre-existing training functions may not accept a `device`
    argument.  We probe via try/except and fall back gracefully rather than
    requiring callers to update every training function signature.
    """
    try:
        return fn(device=device)
    except TypeError:
        return fn()
