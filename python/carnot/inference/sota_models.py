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

    Returns
    -------
    str | None
        Absolute path to a ``.gguf`` file, or ``None`` if the model isn't
        cached at all.  Callers should treat ``None`` as "fall back to the
        legacy transformers path".
    """
    import os
    from pathlib import Path

    root = Path(cache_root) if cache_root else Path.home() / ".cache" / "huggingface" / "hub"
    # HF hub cache layout: models--<org>--<name>/snapshots/<commit-hash>/<files>
    model_dir = root / f"models--{hf_id.replace('/', '--')}"
    if not model_dir.is_dir():
        return None
    snapshots = list((model_dir / "snapshots").iterdir()) if (model_dir / "snapshots").is_dir() else []
    if not snapshots:
        return None
    # There should be at most one snapshot for a GGUF repo we just pulled.  If
    # multiple snapshots exist (re-pulled at different revisions), take the most
    # recently modified one.
    snap = max(snapshots, key=lambda p: p.stat().st_mtime)

    ggufs = sorted(snap.glob("*.gguf"))
    if not ggufs:
        return None

    # Preference cascade.  Case-insensitive substring match so both
    # "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" and "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    # score equivalently against preferred_quant="Q4_K_M".
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
    for token in preference_order:
        for g in ggufs:
            if token.lower() in g.name.lower():
                return str(g)
    # Nothing matched — return the first file so callers at least get *something*.
    return str(ggufs[0])


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
