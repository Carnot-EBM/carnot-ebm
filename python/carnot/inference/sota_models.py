"""Centralised registry of state-of-the-art local GGUF models mandated for
headline Carnot research experiments as of April 2026.

**Researcher summary:**
    Any new experiment that exercises an LLM for a headline metric MUST pick
    its ``MODEL_SPECS`` entries from this module.  Legacy models like
    ``Qwen/Qwen3.5-0.8B`` or ``google/gemma-4-E4B-it`` remain available but
    should only be used for smoke-tests or cheap reproduction runs — not for
    results that will appear in the README, landing page, or technical report.

**Detailed explanation for engineers:**
    The module exposes a flat list ``SOTA_GGUF_MODELS`` plus three helpers
    (``flagship_moe``, ``flagship_dense``, ``default_pair``) so that an
    experiment can either pick one model explicitly or ask for a sensible
    default pair.  The records include both parameter counts (for budgeting)
    and quantisation hints (``Q4_K_M`` fits 24 GB GPUs comfortably; ``Q5_K_M``
    needs 32 GB for the 31B dense).  The ``role`` field lets experiments
    distinguish the MoE (fast, routed) path from the dense (predictable) path
    when both need to coexist in a study.

    Why centralise this: each experiment previously inlined its own
    ``MODEL_SPECS`` list, which made it easy to drift back to tiny legacy
    models.  The user explicitly directed on 2026-04-18 that research results
    must be on frontier models.  A shared registry makes the switch mechanical
    and auditable — grep ``SOTA_GGUF_MODELS`` to see every experiment on the
    mandated set.

Spec: REQ-INFER-SOTA-001 (registry exists), REQ-INFER-SOTA-002 (helpers
return the expected models), REQ-INFER-SOTA-003 (all entries are loadable
via the llama.cpp GGUF path).
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


# The three models mandated by the user on 2026-04-18.  Ordering matters:
# flagship MoE first, middle MoE second, flagship dense third.  Helpers below
# use list indices — keep this order stable.
SOTA_GGUF_MODELS: list[SotaModelSpec] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "moe",
        "active_params_b": 3.0,
        "total_params_b": 35.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 24,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "moe",
        "active_params_b": 4.0,
        "total_params_b": 26.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 16,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense",
        "active_params_b": 31.0,
        "total_params_b": 31.0,
        "quantization": "Q4_K_M",
        "min_vram_gb": 24,
    },
]


def flagship_moe() -> SotaModelSpec:
    """Return the flagship MoE model — Qwen3.6-35B-A3B.

    Prefer this for experiments where capability-per-compute matters most
    (verify-repair, reasoning, CoT).  MoE routing keeps active parameter
    count low (~3 B) so inference is fast on a single 24 GB GPU.
    """
    return SOTA_GGUF_MODELS[0]


def flagship_dense() -> SotaModelSpec:
    """Return the flagship dense model — Gemma4-31B-it.

    Prefer this when the experiment needs predictable activation patterns
    (e.g. token-level adversarial probes where MoE routing would introduce
    noise) or when the experiment runs with greedy decoding where MoE's
    routing overhead isn't amortised.
    """
    return SOTA_GGUF_MODELS[2]


def resolve_cached_gguf(
    hf_id: str,
    preferred_quant: str = "Q4_K_M",
    cache_root: str | None = None,
) -> str | None:
    """Resolve an HF GGUF hub-id to a concrete ``.gguf`` file path on disk.

    Why this exists
    ---------------
    Every live-data-collection script in this repo hardcodes transformers-pipeline
    model names (``google/gemma-4-E4B-it``, ``Qwen/Qwen3.5-0.8B``) because
    transformers can't load GGUFs directly.  That kept the pipeline on undersized
    models even after the user cached the mandated SOTA GGUFs locally
    (``unsloth/Qwen3.6-35B-A3B-GGUF`` etc.).  This helper closes the gap:
    callers pass in one of those hub IDs and get back a filesystem path that
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
        Absolute path to a ``.gguf`` file, or ``None`` if the model isn't
        cached at all.  Callers should treat ``None`` as "fall back to the
        legacy transformers path".
    """
    import os
    from pathlib import Path

    # Preference cascade shared by both hub-cache and project-local lookups.
    preference_order = [
        f"UD-{preferred_quant}",
        preferred_quant,
        "UD-Q4_K_M",
        "Q4_K_M",
        "UD-Q5_K_M",
        "Q5_K_M",
        "UD-Q8_XL",
        "Q8_0",
    ]

    def _pick(ggufs: list[Path]) -> str | None:
        if not ggufs:
            return None
        # Case-insensitive substring match so both
        # "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" and "Qwen3.6-35B-A3B-Q4_K_M.gguf"
        # score equivalently against preferred_quant="Q4_K_M".
        for token in preference_order:
            for g in ggufs:
                if token.lower() in g.name.lower():
                    return str(g)
        # Nothing matched — return the first file so callers at least get *something*.
        return str(ggufs[0])

    # ---- Search 1: HF hub cache (~/.cache/huggingface/hub or override) ----
    root = Path(cache_root) if cache_root else Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = root / f"models--{hf_id.replace('/', '--')}"
    if model_dir.is_dir():
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                snap = max(snapshots, key=lambda p: p.stat().st_mtime)
                hit = _pick(sorted(snap.glob("*.gguf")))
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
            models_root / stripped,         # models/gpt-oss-safeguard-20b/
            models_root / basename,         # models/gpt-oss-safeguard-20b-GGUF/
            models_root / stripped.lower(),
            models_root / basename.lower(),
        ]
        for candidate in candidates:
            if candidate.is_dir():
                hit = _pick(sorted(candidate.glob("*.gguf")))
                if hit is not None:
                    return hit

    return None


def cached_sota_pair(
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
) -> list[dict] | None:
    """Return a two-model ``MODEL_SPECS`` list using cached SOTA GGUFs.

    This is the drop-in replacement for ``default_pair()`` when you want
    real SOTA inference rather than hub-IDs.  Each entry has:
      ``{name, hf_id, gpu, model_path}``  — note the extra ``model_path`` key.

    Returns ``None`` if either required model is NOT cached; callers should
    fall back to the legacy ``(google/gemma-4-E4B-it, Qwen/Qwen3.5-0.8B)``
    transformers pair in that case.  This keeps CI / cold-machine runs from
    silently hanging on missing weights.

    Use:
        from carnot.inference.sota_models import cached_sota_pair
        specs = cached_sota_pair() or LEGACY_FALLBACK_SPECS
        if specs[0].get("model_path"):
            loader = Gemma4QuantizedLoader(model_path=specs[0]["model_path"])
        else:
            pipe = transformers.pipeline("text-generation", specs[0]["hf_id"])
    """
    flagship = SOTA_GGUF_MODELS[0]       # Qwen3.6-35B-A3B
    middle = SOTA_GGUF_MODELS[1]          # Gemma4-26B-A4B-it
    p_flagship = resolve_cached_gguf(flagship["hf_id"], preferred_quant)
    p_middle = resolve_cached_gguf(middle["hf_id"], preferred_quant)
    if p_flagship is None or p_middle is None:
        return None
    return [
        {
            "name": flagship["name"],
            "hf_id": flagship["hf_id"],
            "gpu": gpu_indices[0],
            "model_path": p_flagship,
        },
        {
            "name": middle["name"],
            "hf_id": middle["hf_id"],
            "gpu": gpu_indices[1],
            "model_path": p_middle,
        },
    ]


def default_pair(gpu_indices: tuple[int, int] = (0, 1)) -> list[dict]:
    """Return a sensible two-model ``MODEL_SPECS`` list for headline runs.

    Flagship MoE on the first GPU index, middle MoE on the second.  The
    output shape matches what existing experiment scripts pass to
    ``ExperimentTemplate.setup_gpu`` — ``{name, hf_id, gpu}`` — so callers
    can use it as a drop-in replacement for their inline list.

    Why this pairing: putting two different families side-by-side exposes
    model-specific biases (Qwen's reasoning style vs. Gemma's instruction-
    following style) and also happens to balance VRAM (24 GB + 16 GB) across
    the two-GPU workstation Carnot's research rig uses.

    Args:
        gpu_indices: Tuple of (first_gpu, second_gpu) logical IDs.  Defaults
            to ``(0, 1)`` which matches the DualGPURunner convention.
    """
    flagship = SOTA_GGUF_MODELS[0]
    middle = SOTA_GGUF_MODELS[1]
    return [
        {"name": flagship["name"], "hf_id": flagship["hf_id"], "gpu": gpu_indices[0]},
        {"name": middle["name"], "hf_id": middle["hf_id"], "gpu": gpu_indices[1]},
    ]
