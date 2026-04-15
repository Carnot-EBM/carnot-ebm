"""Live GPU diagnostic: detect and report the exact failure layer when GPU inference is unavailable.

**Researcher summary:**
    Experiments 340, 341, 346, and 347 all ran in *simulated* mode despite
    ``CARNOT_FORCE_LIVE=1``.  Both RTX 3090s were idle throughout two consecutive
    milestones.  The root cause was a silent fallback: when model pre-warm fails,
    ``setup_gpu()`` returned ``all_healthy=False`` and the calling script
    checked that flag—but then quietly continued with synthetic answers and wrote
    artifacts labelled ``inference_mode="simulated"``.  There was no crash, no
    loud error, nothing to tell the researcher that live GPU inference never ran.

    This module fixes that by:
    1. Providing ``diagnose_live_gpu()`` — a CI-safe, layered checker that pinpoints
       exactly WHICH layer failed (CUDA driver visibility, PyTorch CUDA bindings,
       environment variable, model download/load).
    2. Returning a ``LiveGPUDiagnostic`` dataclass so callers get structured, not
       string-parsed, failure info.
    3. Being imported by ``ExperimentTemplate.setup_gpu()`` so that when
       ``CARNOT_FORCE_LIVE=1`` and any model fails, the experiment script raises
       ``RuntimeError`` rather than silently degrading.

**Why each layer matters:**
    - ``cuda_visible``: The GPU driver must be visible to the OS.  ``nvidia-smi``
      returning non-zero (or not found) means CUDA is not accessible at all —
      no point checking torch.
    - ``torch_available``: Even with a driver, PyTorch may have been installed
      without CUDA support (e.g. CPU-only wheel).  ``torch.cuda.is_available()``
      catches this gap.
    - ``carnot_force_live_set``: Confirms the environment was configured for live
      mode — surfaced for traceability, not as a gate.
    - ``model_loadable``: Even with working CUDA, a model may not be cached locally
      and HuggingFace download may fail silently (network, auth, disk space).  We
      attempt a tokenizer load (lightweight, no GPU memory) with a timeout.

Spec: REQ-INFRA-014, SCENARIO-INFRA-014, SCENARIO-INFRA-015
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LiveGPUDiagnostic dataclass
# ---------------------------------------------------------------------------


@dataclass
class LiveGPUDiagnostic:
    """Structured result from ``diagnose_live_gpu()``.

    Every field corresponds to one diagnostic layer.  ``is_live_capable``
    is the summary: ``True`` iff ALL layers passed.  ``failure_reason`` names
    the first layer that failed (empty string on full success).

    Fields
    ------
    cuda_visible : bool
        ``True`` iff ``nvidia-smi`` returned exit code 0, indicating the CUDA
        driver is loaded and at least one GPU is visible to the OS.
    torch_available : bool
        ``True`` iff ``torch.cuda.is_available()`` returned ``True``.  Requires
        CUDA-enabled PyTorch wheel to be installed.
    model_loadable : bool
        ``True`` iff every requested model ID could be loaded (tokenizer check,
        30 s timeout).  ``True`` vacuously when no model IDs were provided.
    carnot_force_live_set : bool
        ``True`` iff ``CARNOT_FORCE_LIVE=1`` is in the environment at call time.
        Informational — not a gate for ``is_live_capable``.
    failure_reason : str
        Human-readable description of the first failed layer.  Empty string
        when ``is_live_capable`` is ``True``.
    is_live_capable : bool
        Summary flag: ``True`` iff cuda_visible AND torch_available AND
        model_loadable all passed.
    """

    cuda_visible: bool
    torch_available: bool
    model_loadable: bool
    carnot_force_live_set: bool
    failure_reason: str
    is_live_capable: bool


# ---------------------------------------------------------------------------
# Individual layer checks
# ---------------------------------------------------------------------------


def check_cuda_visible() -> bool:
    """Return ``True`` iff ``nvidia-smi`` exits 0, indicating driver + GPU are visible.

    **Why subprocess?**  We deliberately avoid importing torch here.  If torch is
    not installed at all we still want an accurate answer about the GPU driver
    layer, so we call ``nvidia-smi`` directly.

    Never raises: any exception (FileNotFoundError, timeout, OS error) returns
    ``False`` with a logged warning.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError:
        _log.warning("check_cuda_visible: nvidia-smi not found — no GPU driver visible")
        return False
    except subprocess.TimeoutExpired:
        _log.warning("check_cuda_visible: nvidia-smi timed out")
        return False
    except Exception as exc:
        _log.warning("check_cuda_visible: unexpected error: %s", exc)
        return False


def check_torch_cuda() -> bool:
    """Return ``True`` iff ``torch.cuda.is_available()`` returns ``True``.

    **Why runtime import?**  torch is optional — Carnot's CI environment
    (JAX_PLATFORMS=cpu) does not install CUDA torch.  Importing at module level
    would crash CI.  We import lazily here and catch ImportError explicitly.

    Never raises.
    """
    try:
        import torch  # noqa: PLC0415 — intentional lazy import

        return bool(torch.cuda.is_available())
    except (ImportError, ModuleNotFoundError):
        _log.warning("check_torch_cuda: torch not installed")
        return False
    except Exception as exc:
        _log.warning("check_torch_cuda: unexpected error: %s", exc)
        return False


def check_carnot_force_live() -> bool:
    """Return ``True`` iff ``CARNOT_FORCE_LIVE=1`` is set in the environment.

    This is a pure env-var read — it never raises.
    """
    return os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"


def _load_tokenizer(model_id: str, timeout_s: float) -> tuple[bool, str]:
    """Attempt to load ``AutoTokenizer`` for *model_id* with a thread timeout.

    Returns ``(True, "")`` on success, or ``(False, error_message)`` on any
    failure including timeout, import error, or network failure.

    This function is separated so the test suite can patch it directly without
    needing to mock deep into transformers internals.
    """

    def _attempt() -> tuple[bool, str]:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            AutoTokenizer.from_pretrained(model_id, local_files_only=False)
            return (True, "")
        except Exception as exc:  # noqa: BLE001
            return (False, f"{type(exc).__name__}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_attempt)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return (False, f"timeout after {timeout_s}s")


def check_model_loadable(
    model_id: str,
    *,
    timeout_s: float = 30.0,
) -> tuple[bool, str]:
    """Return ``(loadable, error_msg)`` for *model_id*.

    Uses ``_load_tokenizer()`` (patchable in tests) with the given timeout.
    Never raises.
    """
    try:
        return _load_tokenizer(model_id, timeout_s)
    except Exception as exc:  # noqa: BLE001
        return (False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# diagnose_live_gpu
# ---------------------------------------------------------------------------


def diagnose_live_gpu(
    model_ids: list[str] | None = None,
) -> LiveGPUDiagnostic:
    """Run all diagnostic layers and return a ``LiveGPUDiagnostic``.

    **Layer order (fail-fast):**
    1. ``check_cuda_visible()`` — CUDA driver / nvidia-smi
    2. ``check_torch_cuda()`` — PyTorch CUDA bindings
    3. ``check_model_loadable(id)`` for each model in *model_ids* — tokenizer load

    ``check_carnot_force_live()`` is recorded independently (informational).

    **CI-safe guarantee:** This function NEVER raises.  If an internal check
    itself raises unexpectedly, the exception is caught, logged, and treated as
    a failure with a descriptive ``failure_reason``.

    Parameters
    ----------
    model_ids : list[str] | None
        Model IDs to check (e.g. ``["Qwen/Qwen3.5-0.8B"]``).  Pass ``[]`` or
        ``None`` to skip the model-load check (model_loadable will be ``True``).

    Returns
    -------
    LiveGPUDiagnostic
        Fully populated diagnostic.  ``is_live_capable=True`` iff all layers
        passed.
    """
    if model_ids is None:
        model_ids = []

    # Layer states — populated as we go.
    cuda_visible = False
    torch_available = False
    model_loadable = True  # vacuously true when no models requested
    carnot_force_live_set = check_carnot_force_live()
    failure_reason = ""
    is_live_capable = False

    try:
        # --- Layer 1: CUDA driver visibility ---
        cuda_visible = check_cuda_visible()
        if not cuda_visible:
            failure_reason = "cuda_visible: nvidia-smi returned non-zero or is absent"
            return LiveGPUDiagnostic(
                cuda_visible=cuda_visible,
                torch_available=torch_available,
                model_loadable=model_loadable,
                carnot_force_live_set=carnot_force_live_set,
                failure_reason=failure_reason,
                is_live_capable=False,
            )

        # --- Layer 2: PyTorch CUDA bindings ---
        torch_available = check_torch_cuda()
        if not torch_available:
            failure_reason = (
                "torch_cuda: torch.cuda.is_available() returned False "
                "(CPU-only wheel or driver mismatch)"
            )
            return LiveGPUDiagnostic(
                cuda_visible=cuda_visible,
                torch_available=torch_available,
                model_loadable=model_loadable,
                carnot_force_live_set=carnot_force_live_set,
                failure_reason=failure_reason,
                is_live_capable=False,
            )

        # --- Layer 3: Model loadability (tokenizer check) ---
        for model_id in model_ids:
            loadable, err_msg = check_model_loadable(model_id)
            if not loadable:
                model_loadable = False
                failure_reason = (
                    f"model_loadable: {model_id} failed to load — {err_msg}"
                )
                return LiveGPUDiagnostic(
                    cuda_visible=cuda_visible,
                    torch_available=torch_available,
                    model_loadable=model_loadable,
                    carnot_force_live_set=carnot_force_live_set,
                    failure_reason=failure_reason,
                    is_live_capable=False,
                )

        # All layers passed.
        is_live_capable = True
        return LiveGPUDiagnostic(
            cuda_visible=cuda_visible,
            torch_available=torch_available,
            model_loadable=model_loadable,
            carnot_force_live_set=carnot_force_live_set,
            failure_reason="",
            is_live_capable=True,
        )

    except Exception as exc:  # noqa: BLE001 — CI-safe: must never raise
        _log.error("diagnose_live_gpu: unexpected exception: %s", exc, exc_info=True)
        fr = failure_reason or f"unexpected exception in diagnostic: {type(exc).__name__}: {exc}"
        return LiveGPUDiagnostic(
            cuda_visible=cuda_visible,
            torch_available=torch_available,
            model_loadable=model_loadable,
            carnot_force_live_set=carnot_force_live_set,
            failure_reason=fr,
            is_live_capable=False,
        )
