"""Robust HuggingFace model loader for Carnot inference pipeline.

**Researcher summary:**
    Centralises all model-loading logic so every experiment and pipeline
    uses the same retry / memory-check / chat-template code path. Eliminates
    the "works in interactive Python, fails in conductor subprocess" failure
    mode that caused benchmark fallbacks to simulated outputs.

**Detailed explanation for engineers:**
    The root cause of conductor subprocess failures is a combination of:
    1. float16 weights on CPU trigger illegal-instruction crashes on older
       kernels that lack AVX2 half-precision support.
    2. OOM failures mid-load leave partial tensors in memory; a second attempt
       then finds even less free RAM and fails faster.
    3. Qwen3 chat-template calls ``enable_thinking`` kwarg which older
       tokenizer versions do not accept — the exception propagates and the
       caller marks the load as failed even though the model itself loaded fine.

    This module fixes all three:
    - Always defaults to float32 on CPU (safe on all hardware).
    - Checks psutil.virtual_memory() before attempting a load and refuses
      to start if less than MIN_FREE_RAM_GB is available.
    - On OOM: calls gc.collect() + torch.cuda.empty_cache(), waits a second,
      then retries — up to max_retries times.
    - Wraps the enable_thinking kwarg in a try/except so the generate path
      degrades gracefully without surfacing as a load failure.

    Environment variables:
    - CARNOT_FORCE_LIVE=1: raise ModelLoadError instead of returning a
      simulated-fallback sentinel. Use this in benchmark scripts that must
      not silently produce invalid results.
    - CARNOT_SKIP_LLM=1: skip all loading and return None immediately
      (useful in CI where no models are cached).
    - CARNOT_FORCE_CPU=1 (default 1): force CPU regardless of GPU availability,
      because ROCm hangs during generation on the current research machine.

    Public API (also exported from carnot.inference):
    - load_model(model_name, device, dtype, max_retries) → (model, tokenizer)
    - generate(model, tokenizer, prompt, max_new_tokens) → str
    - ModelLoadError (re-exported from carnot.pipeline.errors)

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-003
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — loaded at module level so tests can patch them
# without needing create=True. The names are set to None when not installed,
# and load_model checks for None before use.
# ---------------------------------------------------------------------------

try:
    import torch as _torch_module
    import torch
except ImportError:  # pragma: no cover
    _torch_module = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

# Derived flags — always assigned (no branch coverage gap).
_TORCH_AVAILABLE: bool = torch is not None
_TRANSFORMERS_AVAILABLE: bool = AutoModelForCausalLM is not None

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum free RAM (in bytes) required before attempting to load a model.
# Qwen3.5-0.8B in float32 needs ~3 GB; we require at least 2 GB free.
_MIN_FREE_RAM_BYTES: int = 2 * 1024 ** 3  # 2 GiB

# How long to wait between retry attempts when an OOM is detected.
_RETRY_WAIT_SECONDS: float = 1.0


# ---------------------------------------------------------------------------
# Public exception (re-used from pipeline errors for consistency)
# ---------------------------------------------------------------------------

from carnot.pipeline.errors import ModelLoadError  # noqa: E402  (after constants)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _available_ram_bytes() -> int:
    """Return available virtual memory in bytes using psutil.

    **Detailed explanation for engineers:**
        psutil.virtual_memory().available is the amount the OS can give to a
        new process right now (includes reclaimable cache). This is the right
        metric — not 'free' (which excludes cache) and not 'total - used'
        (which overstates pressure).

        If psutil is not installed, returns a large sentinel value so the
        check is effectively skipped. This avoids a hard dependency on psutil
        for environments where it is not available; a warning is logged once.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        logger.warning(
            "psutil not installed — skipping pre-load memory check. "
            "Install with: pip install psutil"
        )
        return 2 ** 63  # effectively unlimited


def _check_memory(model_name: str) -> None:
    """Raise ModelLoadError if available RAM is below the minimum threshold.

    **Detailed explanation for engineers:**
        Called once before the first load attempt and once before each retry
        (RAM may have been freed by gc.collect() between attempts).

    Args:
        model_name: The model being loaded — used in the error message.

    Raises:
        ModelLoadError: If available RAM < _MIN_FREE_RAM_BYTES.
    """
    available = _available_ram_bytes()
    if available < _MIN_FREE_RAM_BYTES:
        available_gb = available / 1024 ** 3
        required_gb = _MIN_FREE_RAM_BYTES / 1024 ** 3
        raise ModelLoadError(
            f"Insufficient memory to load '{model_name}': "
            f"{available_gb:.1f} GiB available, {required_gb:.1f} GiB required.",
            details={
                "model_name": model_name,
                "available_bytes": available,
                "required_bytes": _MIN_FREE_RAM_BYTES,
            },
        )


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    device: str = "cpu",
    dtype: Any = None,
    max_retries: int = 3,
) -> tuple[Any, Any]:
    """Load a HuggingFace causal-LM and its tokenizer robustly.

    **Researcher summary:**
        Single entry point for all model loading in Carnot experiments.
        Checks memory, picks a safe dtype, retries on OOM, and raises a
        structured ModelLoadError so callers can handle failure uniformly.

    **Detailed explanation for engineers:**
        Load sequence:
        1. Check CARNOT_SKIP_LLM — if set, return (None, None) immediately.
        2. Check available RAM via psutil; raise ModelLoadError if < 2 GiB.
        3. Determine device: respect the ``device`` argument unless
           CARNOT_FORCE_CPU=1 (default), which always forces CPU to avoid
           ROCm hangs on the research machine.
        4. Determine dtype: if caller passes None, use torch.float32 on CPU
           and torch.float16 on CUDA. float16 on CPU triggers AVX2 crashes
           on some kernels, so float32 is the safe default.
        5. Attempt to load tokenizer + model. On OOM (RuntimeError containing
           "out of memory" or "CUDA out of memory"):
           - call gc.collect() + torch.cuda.empty_cache()
           - wait _RETRY_WAIT_SECONDS
           - re-check memory
           - retry up to max_retries times total
        6. Set model.eval() (disables dropout; no gradients needed at inference).
        7. Check CARNOT_FORCE_LIVE — if set and load failed, raise instead of
           returning (None, None).

        Returns:
            (model, tokenizer) on success. Both are live HuggingFace objects.
            Returns (None, None) on failure when CARNOT_FORCE_LIVE is not set,
            so callers can fall back to simulated outputs.

    Args:
        model_name: HuggingFace model ID or local path (e.g., "Qwen/Qwen3.5-0.8B").
        device: "cpu" or "cuda". Overridden to "cpu" when CARNOT_FORCE_CPU=1.
        dtype: torch dtype for the model weights, or None to auto-select.
            Defaults to torch.float32 on CPU, torch.float16 on CUDA.
        max_retries: Total number of load attempts before giving up (default 3).

    Returns:
        Tuple of (model, tokenizer) on success, or (None, None) on failure
        when CARNOT_FORCE_LIVE is not set.

    Raises:
        ModelLoadError: On failure when CARNOT_FORCE_LIVE=1, or when
            torch/transformers are not installed, or when available RAM
            is below the minimum threshold.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-003
    """
    # --- Early exits ---
    if os.environ.get("CARNOT_SKIP_LLM"):
        logger.info("CARNOT_SKIP_LLM set — skipping model load for '%s'.", model_name)
        return None, None

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "") == "1"

    # --- Import check ---
    # Check the actual module-level sentinels rather than the boolean flags so
    # that tests can patch AutoModelForCausalLM / AutoTokenizer to mocks and
    # exercise the load path without needing to also patch _TRANSFORMERS_AVAILABLE.
    # (The bool flags can be False when a transitive import inside transformers
    # fails at module-load time even though `import transformers` itself works.)
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        err = ModelLoadError(
            f"torch/transformers not installed. Cannot load '{model_name}'. "
            f"Install with: pip install torch transformers",
            details={"model_name": model_name},
        )
        if force_live:
            raise err
        logger.warning("torch/transformers not available; cannot load model.")
        return None, None

    # --- Device resolution ---
    force_cpu = os.environ.get("CARNOT_FORCE_CPU", "1") == "1"
    if force_cpu:
        effective_device = "cpu"
    else:
        effective_device = device if device == "cpu" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # --- dtype resolution ---
    if dtype is None:
        effective_dtype = torch.float32 if effective_device == "cpu" else torch.float16
    else:
        effective_dtype = dtype

    # --- Retry loop ---
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        logger.info(
            "Loading '%s' on %s (dtype=%s, attempt %d/%d)...",
            model_name, effective_device, effective_dtype, attempt, max_retries,
        )

        try:
            _check_memory(model_name)
        except ModelLoadError:
            if force_live:
                raise
            logger.warning(
                "Memory check failed for '%s' (attempt %d/%d).",
                model_name, attempt, max_retries,
            )
            return None, None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=effective_dtype,
            )
            if effective_device == "cuda":
                model = model.cuda()
            model.eval()
            logger.info("Loaded '%s' successfully on %s.", model_name, effective_device)
            return model, tokenizer

        except RuntimeError as exc:
            exc_str = str(exc).lower()
            is_oom = "out of memory" in exc_str or "cuda out of memory" in exc_str
            if is_oom and attempt < max_retries:
                logger.warning(
                    "OOM loading '%s' (attempt %d/%d). Freeing memory and retrying...",
                    model_name, attempt, max_retries,
                )
                gc.collect()
                try:
                    if torch is not None:
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(_RETRY_WAIT_SECONDS)
                last_exc = exc
                continue
            last_exc = exc

        except Exception as exc:
            last_exc = exc

        # Non-OOM failure or final OOM attempt — stop retrying.
        break

    # All attempts failed.
    err = ModelLoadError(
        f"Failed to load model '{model_name}' after {max_retries} attempt(s): "
        f"{last_exc}",
        details={
            "model_name": model_name,
            "device": effective_device,
            "dtype": str(effective_dtype),
            "attempts": max_retries,
            "last_error": str(last_exc),
        },
    )
    if force_live:
        raise err from last_exc
    logger.warning("Model load failed, returning (None, None): %s", last_exc)
    return None, None


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate text from a loaded HuggingFace model.

    **Researcher summary:**
        Handles Qwen3 chat-template quirks (enable_thinking kwarg) and strips
        ``<think>...</think>`` reasoning tokens from the output automatically.
        Works for any causal-LM tokenizer with or without a chat template.

    **Detailed explanation for engineers:**
        Generation steps:
        1. Wrap the prompt in a ``[{"role": "user", "content": prompt}]``
           message list and apply the tokenizer's chat template.
           - First try with ``enable_thinking=False`` (Qwen3 / newer tokenizers).
           - If that raises TypeError (older tokenizer, no such kwarg), retry
             without the kwarg.
           - If apply_chat_template itself fails entirely, fall back to using
             the raw prompt string.
        2. Tokenize, move tensors to the model's device, run model.generate()
           with greedy decoding (do_sample=False) for reproducibility.
        3. Decode only the newly generated tokens (slice off the input).
        4. Strip ``<think>...</think>`` blocks — Qwen3 emits these as chain-of-
           thought reasoning that should not be in the final answer.

        Greedy decoding is intentional: deterministic outputs are required for
        benchmark reproducibility. Sampling can be added via caller-level
        wrapping if needed.

    Args:
        model: A loaded HuggingFace AutoModelForCausalLM in eval mode.
        tokenizer: The matching AutoTokenizer.
        prompt: The user-facing prompt string.
        max_new_tokens: Maximum tokens to generate (default 256).

    Returns:
        The generated text, with thinking tokens stripped, leading/trailing
        whitespace removed.

    Raises:
        RuntimeError: If model or tokenizer is None (not loaded).

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """
    if model is None or tokenizer is None:
        raise RuntimeError(
            "generate() called with model=None or tokenizer=None. "
            "Call load_model() first and check it succeeded."
        )

    # Detect device from model parameters.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # --- Apply chat template ---
    messages = [{"role": "user", "content": prompt}]
    text: str
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Tokenizer does not support enable_thinking (pre-Qwen3 or non-Qwen).
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # No chat template at all — use raw prompt.
            text = prompt
    except Exception:
        # Any other failure (e.g., jinja template error) — use raw prompt.
        text = prompt

    # --- Tokenize and generate ---
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the echoed input).
    response = tokenizer.decode(
        outputs[0, input_length:],
        skip_special_tokens=True,
    )

    # Strip Qwen3 chain-of-thought thinking tokens.
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response.strip()
