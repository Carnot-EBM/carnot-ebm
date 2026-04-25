"""GGUF model cache resolver for llama.cpp-backed inference.

**Researcher summary:**
    Resolves a HuggingFace GGUF model ID (e.g. ``unsloth/Qwen3.6-35B-A3B-GGUF``)
    to a concrete local ``.gguf`` file path that llama-cpp-python can open.
    When the file is absent and ``can_download=True``, it pulls it from
    HuggingFace Hub via ``huggingface_hub.hf_hub_download``.

**Detailed explanation for engineers:**
    llama.cpp (and its Python wrapper llama-cpp-python) requires a local file path
    to load a model; it cannot pull from HuggingFace Hub at inference time.
    This resolver bridges the gap between the HF model ID used in experiment
    MODEL_SPECS and the on-disk path that llama-cpp-python needs.

    Naming convention used by unsloth GGUF releases on HuggingFace Hub:
        ``<org>/<name>-GGUF`` → saved locally as ``<cache_dir>/<org>_<name>-<quant>.gguf``

    The slash separator in the HF org/name pair is converted to ``_`` to form
    a valid flat filename inside the cache directory.

    Why a dedicated module?  Because every SOTA code-repair experiment that uses
    llama.cpp repeated ad-hoc path-guessing logic (or imported from inconsistent
    locations), which caused ImportError chains across 8 consecutive milestones
    (RETRO-GGUF-CACHE-IMPORT).  A single authoritative resolver stops the rot.

Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from carnot.pipeline.errors import CarnotError


class GGUFModelNotFoundError(CarnotError):
    """Raised when a GGUF model file is not found in the cache directory.

    **Detailed explanation for engineers:**
        This error means the model has not been downloaded yet, or the
        cache_dir is wrong.  The ``details`` dict always contains
        ``expected_path`` so the caller (or user) can see exactly which
        file is missing without having to re-run the resolver.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """


@dataclass
class GGUFCacheConfig:
    """Configuration for GGUFCacheResolver.

    **Detailed explanation for engineers:**
        ``cache_dir`` is relative to the current working directory (usually
        the project root) unless an absolute path is given.  Experiments
        that run from a different cwd should pass an absolute path.

        ``default_quantization`` is the quantization suffix appended when
        the caller does not specify one.  Q4_K_M is the standard
        "best quality-per-size tradeoff" GGUF quantization and the most
        commonly cached format for the SOTA models used in this project.

    Spec: REQ-PIPELINE-030
    """

    cache_dir: str = "models/"
    default_quantization: str = "Q4_K_M"
    timeout_s: int = field(default=30)


class GGUFCacheResolver:
    """Resolves a GGUF model path from a HuggingFace model ID.

    **Researcher summary:**
        Maps ``unsloth/Qwen3.6-35B-A3B-GGUF`` → ``models/unsloth_Qwen3.6-35B-A3B-Q4_K_M.gguf``
        and checks that the file exists before returning the path.  If
        ``can_download=True`` (the default) and the file is missing, it pulls
        the file from HuggingFace Hub before returning the path.

    **Detailed explanation for engineers:**
        llama.cpp requires a local file path to load a model.  HuggingFace
        GGUF models follow a naming convention where the org and model name
        are separated by a slash.  This resolver converts that to a flat
        filename by replacing the slash with ``_`` and appending the
        quantization suffix and ``.gguf`` extension.

        Conversion rule:
            ``<org>/<name>``  →  ``<cache_dir>/<org>_<name>-<quantization>.gguf``

        If ``model_id`` has no ``/`` (i.e. just a model name with no org),
        the result is ``<cache_dir>/<name>-<quantization>.gguf`` — no leading
        underscore.

        ``can_download=True`` means: if the resolved path is absent, call
        ``self.download()`` which uses ``huggingface_hub.hf_hub_download``
        to pull only the specific ``.gguf`` file (not the whole repo).
        Set ``can_download=False`` in offline environments or tests.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """

    can_download: bool = True

    def __init__(self, config: GGUFCacheConfig | None = None) -> None:
        self.config = config or GGUFCacheConfig()

    def _build_path(self, model_id: str, quantization: str) -> str:
        """Build the expected .gguf file path for a model ID and quantization.

        The slash between org and model name is replaced with ``_`` to form
        a flat filename safe for any filesystem.
        """
        quant = quantization or self.config.default_quantization
        flat_name = model_id.replace("/", "_")
        filename = f"{flat_name}-{quant}.gguf"
        return os.path.join(self.config.cache_dir, filename)

    def download(self, hf_repo: str, filename: str, cache_dir: str) -> Path:
        """Pull a single GGUF file from HuggingFace Hub into ``cache_dir``.

        **Detailed explanation for engineers:**
            Uses ``huggingface_hub.hf_hub_download`` to fetch exactly one
            file (the ``.gguf`` quantisation shard) rather than cloning the
            whole repository.  The file is written into ``cache_dir`` and
            the absolute path is returned so that callers can pass it
            straight to llama-cpp-python without a second lookup.

            ``local_dir`` is passed rather than ``cache_dir`` so that the
            file lands at a predictable flat path instead of inside
            huggingface_hub's deeply-nested blob cache.  This matches the
            flat-file convention used by ``_build_path``.

        Args:
            hf_repo: HuggingFace repo ID, e.g. ``unsloth/Qwen3.6-35B-A3B-GGUF``.
            filename: Exact filename inside the repo, e.g.
                ``unsloth_Qwen3.6-35B-A3B-Q4_K_M.gguf``.
            cache_dir: Local directory to save the file.

        Returns:
            ``Path`` to the downloaded file.

        Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
        """
        from huggingface_hub import hf_hub_download  # lazy import — optional dep

        os.makedirs(cache_dir, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            local_dir=cache_dir,
        )
        return Path(local_path)

    def resolve(self, model_id: str, quantization: str | None = None) -> str:
        """Resolve model_id to a local GGUF file path, downloading if needed.

        **Detailed explanation for engineers:**
            Converts the HuggingFace-style ``org/name`` model ID to a flat
            filename and checks that the file exists in ``config.cache_dir``.
            If the file is absent and ``can_download=True``, calls
            ``self.download()`` to fetch it from HuggingFace Hub before
            returning the path.  Raises ``GGUFModelNotFoundError`` only when
            ``can_download=False`` or the download itself fails.

        Args:
            model_id: HuggingFace-style ``org/model-name`` or just
                ``model-name`` for models with no org prefix.
            quantization: Quantization suffix, e.g. ``Q4_K_M``.
                Uses ``config.default_quantization`` if ``None``.

        Returns:
            Absolute (or cache_dir-relative) path to the ``.gguf`` file.

        Raises:
            GGUFModelNotFoundError: If the file does not exist and either
                ``can_download=False`` or the download attempt fails.

        Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
        """
        quant = quantization or self.config.default_quantization
        path = self._build_path(model_id, quant)
        if not os.path.exists(path):
            if not self.can_download:
                raise GGUFModelNotFoundError(
                    f"GGUF model not found: {path!r}. "
                    f"can_download=False — set can_download=True or place the file at the expected path.",
                    details={"expected_path": path, "model_id": model_id},
                )
            flat_name = model_id.replace("/", "_")
            filename = f"{flat_name}-{quant}.gguf"
            try:
                downloaded = self.download(model_id, filename, self.config.cache_dir)
                path = str(downloaded)
            except Exception as exc:
                raise GGUFModelNotFoundError(
                    f"Auto-download failed for {model_id!r} (filename={filename!r}): {exc}",
                    details={"expected_path": path, "model_id": model_id, "filename": filename},
                ) from exc
        return path

    def is_cached(self, model_id: str, quantization: str | None = None) -> bool:
        """Return True if the resolved path exists on disk, False otherwise.

        **Detailed explanation for engineers:**
            A non-raising alternative to ``resolve()``.  Useful for
            conditional branching (e.g. skip experiment if model absent)
            without try/except boilerplate.

        Spec: REQ-PIPELINE-030
        """
        path = self._build_path(model_id, quantization or self.config.default_quantization)
        return os.path.exists(path)


def resolve_gguf_path(
    model_id: str,
    quantization: str = "Q4_K_M",
    cache_dir: str = "models/",
) -> str:
    """Convenience wrapper: resolve model_id to a local GGUF file path.

    **Detailed explanation for engineers:**
        Creates a one-shot ``GGUFCacheResolver`` with a ``GGUFCacheConfig``
        built from the supplied arguments and calls ``resolve()``.  Useful
        for callers that do not need to hold a resolver instance.

    Args:
        model_id: HuggingFace-style ``org/model-name``.
        quantization: Quantization suffix (default ``Q4_K_M``).
        cache_dir: Directory containing cached ``.gguf`` files (default ``models/``).

    Returns:
        Path to the ``.gguf`` file.

    Raises:
        GGUFModelNotFoundError: If the file is not present.

    Spec: REQ-PIPELINE-030, SCENARIO-PIPELINE-040
    """
    config = GGUFCacheConfig(cache_dir=cache_dir, default_quantization=quantization)
    return GGUFCacheResolver(config).resolve(model_id, quantization)
