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
from dataclasses import dataclass
from typing import Callable, TypeVar

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
