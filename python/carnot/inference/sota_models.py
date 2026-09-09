"""Central registry for the current local GGUF mandate and old comparators.

**Researcher summary:**
    New headline experiments use ``unsloth/Qwen3.8-27B-GGUF``. The models in
    ``LEGACY_COMPARATOR_GGUF_MODELS`` remain available for named comparisons.
    Small Qwen3.5 and Gemma E4B models remain CPU smoke fixtures only.

**Detailed explanation for engineers:**
    ``current_model()`` and ``cached_current_model()`` are the unambiguous
    single-model path. ``cached_sota_pair()`` still supports explicit old
    comparator indices for historical experiments. Its default includes the
    current Qwen3.8 model first.

    The separate lists prevent an old comparator from looking mandated. The
    compatibility path avoids changing explicit historical comparisons.

Spec: REQ-INFER-SOTA-7157 and SCENARIO-INFER-SOTA-7157-*.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class SotaModelSpec(TypedDict):
    """One frontier-tier local GGUF model approved for Carnot headline runs.

    Fields match the shape used by existing experiment ``MODEL_SPECS`` entries
    (``name`` / ``hf_id``) plus extra metadata that experiments can use to
    pick a quantisation and GPU placement.
    """

    name: str
    hf_id: str
    role: Literal["moe", "dense"]
    active_params_b: float
    total_params_b: float
    quantization: str
    min_vram_gb: int
    mandate_status: Literal["current_headline", "legacy_comparator"]


# The 2026-09-09 directive names one current headline model.
SOTA_GGUF_MODELS: list[SotaModelSpec] = [
    {
        "name": "Qwen3.8-27B",
        "hf_id": "unsloth/Qwen3.8-27B-GGUF",
        "role": "dense",
        "active_params_b": 27.0,
        "total_params_b": 27.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 18,
        "mandate_status": "current_headline",
    },
]

# Keep this order stable. Explicit model_indices calls use these old indices.
LEGACY_COMPARATOR_GGUF_MODELS: list[SotaModelSpec] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "moe",
        "active_params_b": 3.0,
        "total_params_b": 35.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 24,
        "mandate_status": "legacy_comparator",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "moe",
        "active_params_b": 4.0,
        "total_params_b": 26.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 16,
        "mandate_status": "legacy_comparator",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense",
        "active_params_b": 31.0,
        "total_params_b": 31.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 24,
        "mandate_status": "legacy_comparator",
    },
]


def current_model() -> SotaModelSpec:
    """Return the one model mandated for current headline experiments."""

    return SOTA_GGUF_MODELS[0]


def flagship_moe() -> SotaModelSpec:
    """Return the old Qwen3.6 MoE comparator for compatible callers.

    The historical name remains callable. The record labels the model as a
    comparator so new code does not mistake it for the current mandate.
    """

    return LEGACY_COMPARATOR_GGUF_MODELS[0]


def flagship_dense() -> SotaModelSpec:
    """Return the old Gemma4-31B dense comparator for compatible callers.

    The historical name remains callable. The record labels the model as a
    comparator so new code does not mistake it for the current mandate.
    """

    return LEGACY_COMPARATOR_GGUF_MODELS[2]


def resolve_cached_gguf(
    hf_id: str,
    preferred_quant: str = "Q4_K_M",
    cache_root: str | None = None,
) -> str | None:
    """Resolve an HF GGUF hub-id to a concrete ``.gguf`` file path on disk.

    Why this exists
    ---------------
    This helper keeps model selection local. Callers pass a GGUF hub ID and get
    a filesystem path that
    ``llama_cpp.Llama(model_path=...)`` or
    ``Gemma4QuantizedLoader(model_path=...)`` can consume directly.

    unsloth repos ship many quantisation variants (Q2_K through BF16, plus UD-IQ*
    and UD-Q*_XL dynamic variants).  ``preferred_quant="Q4_K_M"`` is the default
    because it fits a 24 GB GPU with headroom and is the recommended quant in
    unsloth's own model cards.  If neither the base Q4_K_M nor the UD variant is
    present, the helper falls back in this order: UD-Q4_K_M → Q4_K_M → UD-Q5_K_M
    → Q5_K_M → UD-Q8_XL → Q8_0 → first ``.gguf`` in the snapshot.

    Parameters
    ----------
    hf_id : str
        HuggingFace hub id ending in ``-GGUF`` (e.g. ``unsloth/Qwen3.6-35B-A3B-GGUF``).
    preferred_quant : str
        Quantisation to prefer.  Default ``Q4_K_M``.
    cache_root : str, optional
        HF hub cache root.  Defaults to ``~/.cache/huggingface/hub``.

    Search order
    ------------
    1. ``cache_root`` (HF hub layout: ``models--<org>--<name>/snapshots/<hash>/``).
       This is where ``hf download <repo> <file>`` without ``--local-dir``
       lands.
    2. ``<project_root>/models/<last_segment_of_hf_id_without_GGUF_suffix>/``
       (flat layout).  This is where ``hf download <repo> <file>
       --local-dir models/<x>`` lands.  Matches the convention we use for
       gpt-oss-safeguard, Qwen, Gemma etc. — keeps the weights in-tree with
       the project so a fresh checkout on a new machine can see them via a
       single rsync of the repo root.
    3. ``<project_root>/models/<first_segment_after_slash_lower>/`` as a fallback
       (e.g. ``models/qwen3.6-35b-a3b-gguf/``) for cases where the directory
       name preserves the ``-GGUF`` suffix.

    Returns
    -------
    str | None
        Absolute path to a ``.gguf`` file, or ``None`` if the model is not
        cached. Headline callers must block. A separate CPU smoke task can
        choose an explicitly labeled small fixture.
    """
    from pathlib import Path

    # Preference cascade shared by both hub-cache and project-local lookups.
    preference_order = [
        preferred_quant,
        f"UD-{preferred_quant}",
        "UD-Q4_K_M",
        "Q4_K_M",
        "UD-Q5_K_M",
        "Q5_K_M",
        "UD-Q8_XL",
        "Q8_0",
    ]

    def _is_language_model_gguf(path: Path) -> bool:
        name = path.name.lower()
        if name.startswith(("mmproj", "mtp-")) or "mmproj" in name:
            return False
        return all(parent.name.lower() != "mtp" for parent in path.parents)

    def _pick(ggufs: list[Path]) -> str | None:
        candidates = [g for g in ggufs if _is_language_model_gguf(g)]
        if not candidates:
            return None
        # Case-insensitive substring match so both
        # "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" and "Qwen3.6-35B-A3B-Q4_K_M.gguf"
        # score equivalently against preferred_quant="Q4_K_M".
        for token in preference_order:
            for g in candidates:
                if token.lower() in g.name.lower():
                    return str(g)
        # Nothing matched — return the first file so callers at least get *something*.
        return str(candidates[0])

    # ---- Search 1: HF hub cache (~/.cache/huggingface/hub or override) ----
    root = Path(cache_root) if cache_root else Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = root / f"models--{hf_id.replace('/', '--')}"
    if model_dir.is_dir():
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                ggufs: list[Path] = []
                for snap in sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True):
                    ggufs.extend(sorted(snap.rglob("*.gguf")))
                hit = _pick(ggufs)
                if hit is not None:
                    return hit

    # ---- Search 2 & 3: project-local models/ directory ----
    # Walk up from this file to find the project root (the dir containing models/).
    # __file__ = <project>/python/carnot/inference/sota_models.py, so four parents up.
    project_root = Path(__file__).resolve().parents[3]
    models_root = project_root / "models"
    if models_root.is_dir():
        # Candidate subdirectory names, in priority order.
        basename = hf_id.split("/", 1)[-1]  # e.g. "gpt-oss-safeguard-20b-GGUF"
        stripped = basename[:-5] if basename.endswith("-GGUF") else basename
        candidates = [
            models_root / stripped,  # models/gpt-oss-safeguard-20b/
            models_root / basename,  # models/gpt-oss-safeguard-20b-GGUF/
            models_root / stripped.lower(),
            models_root / basename.lower(),
        ]
        for candidate in candidates:
            if candidate.is_dir():
                hit = _pick(sorted(candidate.glob("*.gguf")))
                if hit is not None:
                    return hit

    return None


def cached_current_model(
    gpu_index: int = 0,
    preferred_quant: str = "Q4_K_M",
) -> dict | None:
    """Resolve only the current headline model from the local cache.

    ``None`` is a hard local cache miss. The helper never downloads weights
    and never substitutes a comparator or a small smoke fixture.
    """

    model = current_model()
    model_path = resolve_cached_gguf(model["hf_id"], preferred_quant)
    if model_path is None:
        return None
    return {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "gpu": gpu_index,
        "model_path": model_path,
        "selection_role": model["mandate_status"],
    }


def cached_sota_pair(
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
    model_indices: tuple[int, int] | None = None,
) -> list[dict] | None:
    """Return a current-plus-comparator pair or an explicit legacy pair.

    This is the drop-in replacement for ``default_pair()`` when you want
    real SOTA inference rather than hub-IDs.  Each entry has:
      ``{name, hf_id, gpu, model_path}``  — note the extra ``model_path`` key.

    The default selection requires cached Qwen3.8 and one cached comparator.
    Explicit ``model_indices`` retain the old three-entry index order for
    historical comparison code. ``None`` means the requested local pair is not
    complete. Callers must block or use an explicitly labeled smoke path.

    Use:
        from carnot.inference.sota_models import cached_sota_pair
        specs = cached_sota_pair()
        if specs is None:
            write_blocked_artifact("current model or comparator missing")
    """
    cached_models: list[tuple[SotaModelSpec, str]] = []

    if model_indices is not None:
        for i in model_indices:
            model = LEGACY_COMPARATOR_GGUF_MODELS[i]
            model_path = resolve_cached_gguf(model["hf_id"], preferred_quant)
            if model_path is not None:
                cached_models.append((model, model_path))
    else:
        for model in [current_model(), *LEGACY_COMPARATOR_GGUF_MODELS]:
            model_path = resolve_cached_gguf(model["hf_id"], preferred_quant)
            if model_path is not None:
                cached_models.append((model, model_path))

    if len(cached_models) < 2 or (
        model_indices is None and cached_models[0][0]["hf_id"] != current_model()["hf_id"]
    ):
        return None

    return [
        {
            "name": model["name"],
            "hf_id": model["hf_id"],
            "gpu": gpu,
            "model_path": model_path,
            "selection_role": model["mandate_status"],
        }
        for gpu, (model, model_path) in zip(gpu_indices, cached_models[:2], strict=True)
    ]


def gguf_tokenizer_loadable(model_path: str | None) -> tuple[bool, str]:
    """Preflight that a GGUF's EMBEDDED tokenizer loads — the CORRECT check.

    SOTA ``unsloth/*-GGUF`` repos ship NO HuggingFace tokenizer files; the
    tokenizer lives inside the ``.gguf`` and is read by llama.cpp. Experiments
    that need a loadability / ``tokenizer_status`` preflight MUST use this
    helper, NOT ``transformers.AutoTokenizer.from_pretrained(hf_id)`` — the
    latter raises ``ValueError: Couldn't instantiate the backend tokenizer``
    on a GGUF-only repo (no sentencepiece/tiktoken source files to convert).
    That AutoTokenizer misuse — not any model defect — is what blocked
    Qwen3.6-35B in milestones .310/.311 (exp3327/exp3352). See CLAUDE.md
    "SOTA Local Models" → "GGUF tokenizer rule".

    Loads vocab-only (no weights → seconds, low memory) and round-trips a
    probe string through the embedded tokenizer. Returns ``(ok, detail)``.
    """
    import os

    if not model_path or not os.path.exists(model_path):
        return False, f"model_path missing or not on disk: {model_path!r}"
    try:
        from llama_cpp import Llama
    except Exception as e:  # pragma: no cover - environment-dependent
        return False, f"llama_cpp unavailable: {e}"
    try:
        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        toks = llm.tokenize(b"What is 2+2?")
        if not toks:
            return False, "embedded tokenizer returned no tokens"
        return True, f"embedded GGUF tokenizer OK ({len(toks)} tokens on probe)"
    except Exception as e:
        return False, f"llama.cpp GGUF tokenizer load failed: {e}"


def default_pair(gpu_indices: tuple[int, int] = (0, 1)) -> list[dict]:
    """Return the current model plus the first named comparator.

    New single-model work should use ``current_model()``. This pair helper
    remains for code that needs a comparator-shaped two-GPU specification.

    Args:
        gpu_indices: Tuple of (first_gpu, second_gpu) logical IDs.  Defaults
            to ``(0, 1)`` which matches the DualGPURunner convention.
    """
    headline = current_model()
    comparator = LEGACY_COMPARATOR_GGUF_MODELS[0]
    return [
        {
            "name": headline["name"],
            "hf_id": headline["hf_id"],
            "gpu": gpu_indices[0],
            "selection_role": headline["mandate_status"],
        },
        {
            "name": comparator["name"],
            "hf_id": comparator["hf_id"],
            "gpu": gpu_indices[1],
            "selection_role": comparator["mandate_status"],
        },
    ]
