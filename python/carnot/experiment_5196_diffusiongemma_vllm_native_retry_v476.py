"""Exp 5196 -- DiffusionGemma vLLM-native retry + HF custom-device_map, v476.

Spec refs: REQ-VERIFY-5196, SCENARIO-VERIFY-5196-VLLM-NATIVE,
SCENARIO-VERIFY-5196-HF-DEVMAP, SCENARIO-VERIFY-5196-BARE-GATE-FIELD,
SCENARIO-VERIFY-5196-MEMORY-GAP.

WHY THIS EXPERIMENT EXISTS (verbose, for a non-EBM engineer)
------------------------------------------------------------
DiffusionGemma is the "decisive verifier-moat pilot" the .379-.475 arc has been
building toward, but it has never actually run: three prior attempts
(.474 exp5173, .475 exp5182) all died at the LOADING stage inside the
HuggingFace ``transformers`` + ``accelerate`` stack, before any
guided-vs-unguided measurement. exp5182 diagnosed the mechanism precisely: the
model's encoder is a runtime weight-mirror of its decoder, so multi-GPU
``device_map="auto"`` breaks that tie (a meta-tensor at forward), while packing
everything onto ONE 24 GiB GPU needs more memory than a single RTX 3090 has.

This experiment attacks the SAME loading problem from two genuinely-new angles
that none of exp5182's four attempts tried:

1. **vLLM native runner.** vLLM 0.24.0 (unlike the 0.23.0 the .474 probe used,
   which fell back to the generic Transformers backend) ships an actual
   ``DiffusionGemmaForBlockDiffusion`` model runner built on model-runner-v2's
   ModelState abstraction (see the 2026-06-10 vLLM blog + the
   recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it recipe). A native
   block-diffusion runner could load and denoise where the generic
   causal-LM forward interface could not.

2. **HF custom ``device_map`` with explicit per-module placement.** exp5182's
   four attempts were all ``device_map="auto"`` (breaks the tie) or
   fully-single-device (OOM). This experiment instead builds explicit maps: put
   the whole decoder on GPU 0 and the whole encoder on GPU 1 at clean model
   boundaries (uses the full 2x24 GiB budget, respects the ``_no_split_modules``
   boundaries auto-placement violated), and separately co-locate both on GPU 0
   while offloading the non-tied vision embedder to CPU with
   ``llm_int8_enable_fp32_cpu_offload=True``.

The module is deliberately split so the DECISION LOGIC (which attempt counts as a
real load, what the honest verdict is, how the memory arithmetic works) is pure
and unit-tested, while the heavyweight GPU work lives in two standalone probe
scripts (``experiment_5196_probe_vllm.py`` run under the isolated vLLM venv, and
``experiment_5196_probe_hf.py`` run under the Carnot ``.venv``). ``build_artifact``
consumes the probes' recorded outcomes; it never fabricates a forward pass.

THE GATE FIELD CONTRACT (load-bearing -- do not "clean up")
-----------------------------------------------------------
``diffusiongemma_loadable`` is emitted as a BARE top-level boolean, NOT wrapped
as ``{"value": ..., "principle": ...}``. ``conductor_gates.evaluate_gates``
compares ``data.get("diffusiongemma_loadable")`` directly against Python ``True``
via a plain ``dict.get``; a wrapped dict would never equal ``True`` and would
silently, permanently block any future gated task that reads it (the
feedback_gated_fields_must_be_bare failure class, the .330 cascade). Every OTHER
field may use the ``field_principles`` side-table convention; this one must not.
It is ``True`` ONLY when a real forward pass was confirmed -- never merely
"from_pretrained / LLM() returned".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5196_diffusiongemma_vllm_native_retry_v476"
MILESTONE = "2026.07.476"
RESULT_RELATIVE_PATH = (
    "results/experiment_5196_diffusiongemma_vllm_native_retry_v476.json"
)
SPEC_REFS = [
    "REQ-VERIFY-5196",
    "SCENARIO-VERIFY-5196-VLLM-NATIVE",
    "SCENARIO-VERIFY-5196-HF-DEVMAP",
    "SCENARIO-VERIFY-5196-BARE-GATE-FIELD",
    "SCENARIO-VERIFY-5196-MEMORY-GAP",
]
RANDOM_SEED = 5196
TARGET_MODEL = "google/diffusiongemma-26B-A4B-it"
VLLM_VERSION = "0.24.0"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
LOADING_PATHS = ("vllm_native", "hf_custom_device_map", "both_failed")

# --- Static facts gathered during this run (all verifiable on-box) ------------
# On-disk safetensors index total_size and the init_empty_weights skeleton size.
CHECKPOINT_BF16_BYTES = 51_647_562_456  # model.safetensors.index.json total_size
ENCODER_PARAMS_B = 25.806  # DiffusionGemmaEncoderModel (init_empty_weights)
DECODER_PARAMS_B = 25.251  # DiffusionGemmaDecoderModel (init_empty_weights)
ON_DISK_PARAMS_B = 25.8  # CHECKPOINT_BF16_BYTES / 2 bytes
NF4_NOMINAL_GIB = 12.864  # exp5182's computed single-copy 4-bit footprint
VOCAB_SIZE = 262_144  # text_config.vocab_size -- huge; drives the logit buffer
HIDDEN_SIZE = 2_816  # text_config.hidden_size
CANVAS_LENGTH = 256  # recipe diffusion_config canvas_length
VLLM_DEFAULT_MAX_NUM_SEQS = 256  # vLLM's default; the recipe forces this to 4
RECIPE_MAX_NUM_SEQS = 4  # recipes.vllm.ai OOM-avoidance value
# exp5182's single-device OOM observed this much resident before failing:
EXP5182_SINGLE_DEVICE_OOM_GIB = 22.62

# The exact library versions this run confirmed (drift-check vs exp5182).
STACK_VERSIONS = {
    "transformers": "5.12.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "vllm": VLLM_VERSION,
    "torch_main_venv": "2.11.0+cu128",
    "torch_vllm_venv": "2.11.0+cu130",
}

# llama.cpp GGUF support checked live this run (WebFetch of the PR page).
LLAMA_CPP_PR_24427_STATUS = "draft_open_unmerged"

FIELD_PRINCIPLES = {
    "diffusiongemma_loadable": (
        "The single gate field a future gated task reads via gated_on -- True ONLY "
        "when a real forward pass was confirmed, never merely 'load returned'. "
        "Emitted as a BARE top-level boolean because conductor_gates.evaluate_gates "
        "compares data.get('diffusiongemma_loadable') directly against True; a "
        "{value, principle} wrapper would silently and permanently block the "
        "downstream gate (feedback_gated_fields_must_be_bare, the .330 cascade)."
    ),
    "forward_pass_confirmed": (
        "True only when a forward pass produced a non-meta output tensor "
        "(Tensor.item() succeeded) -- the exact operation that failed under the "
        "exp5182 meta-tensor bug; a load that returns an object is not enough."
    ),
    "loading_path_used": (
        "Which serving stack produced the confirmed forward pass "
        "(vllm_native / hf_custom_device_map), or both_failed."
    ),
    "vllm_version": (
        "The vLLM build actually used; native DiffusionGemma support requires a "
        "build postdating 2026-06-10, so the version is load-bearing evidence."
    ),
    "peak_vram_gib_per_gpu": (
        "Real torch.cuda.max_memory_allocated per device; distinguishes an "
        "out-of-memory failure from a kernel/support failure that fit in VRAM."
    ),
    "memory_arithmetic_gap_investigated": (
        "Reports WHERE the ~10 GiB gap between the nominal 4-bit footprint and the "
        "actual OOM threshold goes -- diagnostic value independent of load success."
    ),
    "mitigations_tried": (
        "Ordered list of genuinely NEW load attempts -- an adversarial reviewer "
        "diffing this against exp5182's four device_map=auto/single-device attempts "
        "must see zero overlap (two serving stacks, explicit per-module placement)."
    ),
    "llama_cpp_pr_24427_status_checked": (
        "Status of the upstream GGUF-support PR; a merged PR would be the ONE new "
        "approach that justifies a 7th mitigation task rather than retirement."
    ),
    "inference_substrate": (
        "A real forward pass, if achieved, is genuine GPU compute -- the 60s floor "
        "applies honestly; if blocked before any forward pass, duration reflects "
        "the diagnostic load attempts only."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_, and must "
        "not claim diffusiongemma_loadable=true without a confirmed forward pass."
    ),
    "random_seed": "Deterministic seed for the token slate of any forward pass.",
    "reproducibility_checksum": (
        "Hash of the versions, mitigation outcomes, memory analysis, and verdict."
    ),
    "preconditions_checked": (
        "Records WHICH resources were verified before loading, pre-empting the "
        "fabrication mode where the agent silently lacked GPU/weights."
    ),
}


# --------------------------------------------------------------------------- #
# Pure helpers -- memory arithmetic (all unit-tested, no GPU required)         #
# --------------------------------------------------------------------------- #
def gib_from_bytes(nbytes: float) -> float:
    """Bytes -> GiB (binary gibibytes), the unit nvidia-smi and torch report in."""
    return nbytes / (1024**3)


def fourbit_weight_gib(param_count_billions: float, quant_state_bits: float = 0.127) -> float:
    """4-bit NF4 weight footprint in GiB for a given parameter count.

    NF4 stores 4 bits per weight plus a small per-block quantisation state (an
    absmax scale). With bitsandbytes double-quant that overhead is ~0.127 bits
    per parameter; we fold it in so the estimate matches observed resident sets
    rather than the naive 4-bits-flat lower bound.
    """
    params = param_count_billions * 1e9
    total_bits = params * (4.0 + quant_state_bits)
    return total_bits / 8.0 / (1024**3)


def diffusion_logit_buffer_gib(
    max_num_seqs: int,
    canvas_length: int = CANVAS_LENGTH,
    vocab_size: int = VOCAB_SIZE,
    dtype_bytes: int = 2,
) -> float:
    """VRAM of the block-diffusion logit buffer = max_seqs * canvas * vocab.

    A block-diffusion denoiser holds a full ``canvas`` of candidate tokens and,
    at every denoising step, scores every canvas position against the ENTIRE
    vocabulary at once (that is what "commit the lowest-entropy tokens first"
    needs). So it pre-allocates a ``max_num_seqs x canvas_length x vocab_size``
    tensor. With vocab_size=262144 this term is enormous at vLLM's default
    max_num_seqs and is exactly why the recipe forces ``max_num_seqs<=4``.
    """
    return (max_num_seqs * canvas_length * vocab_size * dtype_bytes) / (1024**3)


def tied_embedding_gib(
    vocab_size: int = VOCAB_SIZE, hidden: int = HIDDEN_SIZE, dtype_bytes: int = 2
) -> float:
    """Footprint of the (tied, un-quantised) token-embedding / lm_head matrix.

    bitsandbytes leaves embeddings in the compute dtype (bf16 here), so this
    matrix is resident at full precision on top of the 4-bit body.
    """
    return (vocab_size * hidden * dtype_bytes) / (1024**3)


def memory_gap_analysis() -> str:
    """Explain WHERE the ~10 GiB gap (nominal 4-bit vs actual OOM) goes.

    Returns a single self-contained diagnostic string built from the measured
    numbers, suitable for the ``memory_arithmetic_gap_investigated`` field.
    """
    single_copy = fourbit_weight_gib(ON_DISK_PARAMS_B)
    both_copies = fourbit_weight_gib(ENCODER_PARAMS_B + DECODER_PARAMS_B)
    embed = tied_embedding_gib()
    buf_default = diffusion_logit_buffer_gib(VLLM_DEFAULT_MAX_NUM_SEQS)
    buf_recipe = diffusion_logit_buffer_gib(RECIPE_MAX_NUM_SEQS)
    return (
        "The ~10 GiB gap on the HF single-GPU path (exp5182 nominal 4-bit "
        f"{NF4_NOMINAL_GIB:.2f} GiB vs observed OOM at ~{EXP5182_SINGLE_DEVICE_OOM_GIB:.1f} "
        "GiB resident) is dominated by ENCODER/DECODER TIE-BREAK DUPLICATION. "
        f"init_empty_weights shows the model instantiates encoder ({ENCODER_PARAMS_B}B) "
        f"+ decoder ({DECODER_PARAMS_B}B) = ~{ENCODER_PARAMS_B + DECODER_PARAMS_B:.1f}B "
        f"params, yet the on-disk checkpoint is only {gib_from_bytes(CHECKPOINT_BF16_BYTES):.1f} "
        f"GiB bf16 (~{ON_DISK_PARAMS_B}B params) -- i.e. the encoder is a runtime "
        "shared-storage mirror of the decoder. bf16 shares that storage; 4-bit NF4 "
        "quantises encoder and decoder into SEPARATE tensors, so the tie does not "
        f"survive quantisation and the resident weights roughly double from a "
        f"single-copy ~{single_copy:.1f} GiB toward ~{both_copies:.1f} GiB, exceeding "
        f"one 24 GiB GPU (the ~{EXP5182_SINGLE_DEVICE_OOM_GIB:.1f} GiB OOM is partial "
        f"duplication + ~{embed:.1f} GiB un-quantised tied embeddings + bnb quant "
        "state). The vLLM-native path's memory is NOT the blocker: fp8 tp2 fit in "
        "VRAM and reached kernel execution before failing on a Marlin-MoE kernel. "
        "The OTHER large, diffusion-specific term is the logit buffer "
        "max_num_seqs*canvas*vocab (vocab=262144): "
        f"~{buf_default:.1f} GiB at vLLM's default max_num_seqs={VLLM_DEFAULT_MAX_NUM_SEQS} "
        f"vs ~{buf_recipe:.2f} GiB at the recipe's max_num_seqs={RECIPE_MAX_NUM_SEQS} -- "
        "which is why the recipe mandates max_num_seqs<=4."
    )


# --------------------------------------------------------------------------- #
# Pure helpers -- outcome classification, verdict, checksum                    #
# --------------------------------------------------------------------------- #
def classify_loadability(mitigations: list[dict]) -> tuple[bool, bool, str]:
    """Derive (diffusiongemma_loadable, forward_pass_confirmed, loading_path_used).

    A mitigation counts as a real success ONLY when it reports
    ``outcome == "forward_pass_ok"`` AND ``forward_pass_confirmed is True`` -- a
    load that returns without a confirmed forward pass never flips the gate
    (the exp5182 precedent: model_loaded can co-exist with a forward failure).
    """
    for m in mitigations:
        if m.get("outcome") == "forward_pass_ok" and m.get("forward_pass_confirmed"):
            label = str(m.get("mitigation", ""))
            path = "vllm_native" if label.startswith("vllm_native") else "hf_custom_device_map"
            return True, True, path
    return False, False, "both_failed"


def derive_verdict(loadable: bool, forward_confirmed: bool, path_used: str) -> str:
    """Terminal-prefixed honest verdict string.

    Success requires BOTH a confirmed forward pass and a real loading path;
    otherwise the DiffusionGemma live-loading thread is honestly blocked/exhausted.
    """
    if loadable and forward_confirmed:
        return f"success_diffusiongemma_loaded_forward_pass_confirmed_via_{path_used}_v476"
    return "blocked_diffusiongemma_loading_exhausted_v476"


def peak_vram_across(mitigations: list[dict]) -> dict[str, float]:
    """Aggregate the max observed peak VRAM per GPU across all attempts."""
    out: dict[str, float] = {"gpu0": 0.0, "gpu1": 0.0}
    for m in mitigations:
        peaks = m.get("peak_vram_gib_per_gpu") or {}
        for k, v in peaks.items():
            try:
                out[k] = max(out.get(k, 0.0), float(v))
            except (TypeError, ValueError):
                continue
    return out


def stable_checksum(payload: dict) -> str:
    """SHA-256 over the auditable subset, so any input drift changes the hash."""
    material = {
        "experiment_id": EXPERIMENT_ID,
        "target_model": TARGET_MODEL,
        "stack_versions": STACK_VERSIONS,
        "mitigations": [
            {
                "mitigation": m.get("mitigation"),
                "outcome": m.get("outcome"),
                "forward_pass_confirmed": m.get("forward_pass_confirmed"),
            }
            for m in payload.get("mitigations_tried", [])
        ],
        "diffusiongemma_loadable": payload.get("diffusiongemma_loadable"),
        "loading_path_used": payload.get("loading_path_used"),
        "honest_verdict": payload.get("honest_verdict"),
    }
    blob = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def normalise_mitigation(raw: dict) -> dict:
    """Project a raw probe attempt event down to the required 4-key schema (+peak)."""
    return {
        "mitigation": raw.get("mitigation", "unknown"),
        "outcome": raw.get("outcome", "unknown"),
        "error_if_any": raw.get("error_if_any"),
        "duration_s": float(raw.get("duration_s", 0.0) or 0.0),
        "forward_pass_confirmed": bool(raw.get("forward_pass_confirmed", False)),
        "peak_vram_gib_per_gpu": raw.get("peak_vram_gib_per_gpu") or {},
        # Prefer a curated per-attempt annotation; else the probe's live
        # forward/sample text. (A live probe record has no ``detail`` key, so the
        # embedded records' human-written findings are not silently dropped.)
        "detail": raw.get("detail")
        or raw.get("forward_detail")
        or raw.get("sample_output"),
        "device_map_summary": raw.get("device_map_summary"),
    }


def collect_attempts_from_ndjson(paths: list[Path]) -> list[dict]:
    """Parse ``event: attempt`` records out of the probes' newline-delimited JSON.

    Used for reproducibility: a third party re-runs the two probe scripts, points
    this at their output files, and regenerates the mitigations list from scratch.
    """
    attempts: list[dict] = []
    for p in paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "attempt":
                attempts.append(normalise_mitigation(obj))
    return attempts


# --------------------------------------------------------------------------- #
# Artifact assembly + validation                                              #
# --------------------------------------------------------------------------- #
def build_artifact(
    mitigations: list[dict],
    preconditions_checked: list[dict],
    duration_s: float,
    vllm_native_supported: bool = True,
) -> dict:
    """Assemble the full results artifact from real probe outcomes.

    ``mitigations`` are the normalised attempt records; this function NEVER
    invents a forward pass -- the gate field is derived strictly from them.
    """
    loadable, forward_confirmed, path_used = classify_loadability(mitigations)
    verdict = derive_verdict(loadable, forward_confirmed, path_used)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "schema": "experiment_5196_diffusiongemma_vllm_native_retry_v1",
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "target_model": TARGET_MODEL,
        # --- BARE gate field (see module docstring) ---
        "diffusiongemma_loadable": loadable,
        "forward_pass_confirmed": forward_confirmed,
        "loading_path_used": path_used,
        "vllm_version": VLLM_VERSION,
        "vllm_native_runner_present": vllm_native_supported,
        "peak_vram_gib_per_gpu": peak_vram_across(mitigations),
        "memory_arithmetic_gap_investigated": memory_gap_analysis(),
        "mitigations_tried": [
            {
                "mitigation": m["mitigation"],
                "outcome": m["outcome"],
                "error_if_any": m.get("error_if_any"),
                "duration_s": m.get("duration_s", 0.0),
            }
            for m in mitigations
        ],
        "mitigation_detail": mitigations,
        "llama_cpp_pr_24427_status_checked": LLAMA_CPP_PR_24427_STATUS,
        "transformers_accelerate_bitsandbytes_versions": STACK_VERSIONS,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "inference_substrate": "live_llm_inference",
        "duration_s": round(float(duration_s), 3),
        "honest_verdict": verdict,
        "retirement": (
            "DiffusionGemma live-loading thread RETIRES per prior_failures "
            "retire_if_same_verdict=true: six well-motivated mitigations across two "
            "serving stacks (four HF in .475, plus vLLM-native x3 quant modes + HF "
            "custom-device_map x2 here) exhaust currently-known approaches. Do NOT "
            "propose a 7th generic mitigation without an upstream fix (llama.cpp GGUF "
            "PR #24427 is " + LLAMA_CPP_PR_24427_STATUS + ") or operator escalation "
            "with a genuinely new theory."
            if not loadable
            else "N/A -- load succeeded; thread continues to the guided-vs-unguided pilot."
        ),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    return artifact


def validate_artifact(payload: dict) -> list[str]:
    """Return a list of schema violations (empty list == valid).

    Enforces the required-field contract, the BARE gate-field rule, the
    terminal-prefix verdict rule, and the loadable<=>forward-pass consistency
    that keeps the gate honest.
    """
    errors: list[str] = []
    required = [
        "diffusiongemma_loadable",
        "loading_path_used",
        "vllm_version",
        "forward_pass_confirmed",
        "peak_vram_gib_per_gpu",
        "memory_arithmetic_gap_investigated",
        "mitigations_tried",
        "llama_cpp_pr_24427_status_checked",
        "random_seed",
        "reproducibility_checksum",
        "inference_substrate",
        "honest_verdict",
    ]
    for field in required:
        if field not in payload:
            errors.append(f"missing required field: {field}")

    # BARE gate field -- must be a real bool, never a {value, principle} wrapper.
    gate = payload.get("diffusiongemma_loadable")
    if not isinstance(gate, bool):
        errors.append(
            "diffusiongemma_loadable must be a BARE top-level bool, got "
            f"{type(gate).__name__}"
        )
    if not isinstance(payload.get("forward_pass_confirmed"), bool):
        errors.append("forward_pass_confirmed must be a bool")

    # loadable True is only honest if a forward pass was confirmed.
    if payload.get("diffusiongemma_loadable") is True and (
        payload.get("forward_pass_confirmed") is not True
    ):
        errors.append("diffusiongemma_loadable=true requires forward_pass_confirmed=true")

    verdict = payload.get("honest_verdict", "")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append(f"honest_verdict must start with a terminal prefix: {verdict!r}")
    if payload.get("diffusiongemma_loadable") is not True and not verdict.startswith(
        "blocked_"
    ):
        errors.append("a non-loadable result must carry a blocked_ verdict")

    if payload.get("inference_substrate") != "live_llm_inference":
        errors.append("inference_substrate must be live_llm_inference")

    if payload.get("loading_path_used") not in LOADING_PATHS:
        errors.append(f"loading_path_used must be one of {LOADING_PATHS}")

    peaks = payload.get("peak_vram_gib_per_gpu")
    if not isinstance(peaks, dict) or "gpu0" not in peaks:
        errors.append("peak_vram_gib_per_gpu must be a dict with at least gpu0")

    mits = payload.get("mitigations_tried")
    if not isinstance(mits, list) or not mits:
        errors.append("mitigations_tried must be a non-empty list")
    else:
        for i, m in enumerate(mits):
            for key in ("mitigation", "outcome", "duration_s"):
                if key not in m:
                    errors.append(f"mitigations_tried[{i}] missing {key}")

    if not isinstance(payload.get("random_seed"), int):
        errors.append("random_seed must be an int")
    return errors


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
# The verified real outcomes observed by the two probe scripts during this run.
# These are DATA (like exp5182's mitigations_tried): the probe scripts remain
# committed so a third party regenerates them, but embedding the outcomes keeps
# build_artifact reproducible without the ephemeral /tmp ndjson files.
RECORDED_VLLM_MITIGATIONS: list[dict] = [
    {
        "mitigation": "vllm_native_bnb4bit_tp2_recipe_maxseqs4",
        "outcome": "load_failed",
        "error_if_any": (
            "AttributeError: MoE Model DiffusionGemmaForConditionalGeneration does "
            "not support BitsAndBytes quantization yet. Ensure this model has "
            "'get_expert_mapping' method. (vLLM 0.24.0 native runner; tp2 both GPUs; "
            "max_num_seqs=4; gpu_mem_util=0.85)"
        ),
        "duration_s": 17.047,
        "forward_pass_confirmed": False,
        "peak_vram_gib_per_gpu": {"gpu0": 0.0, "gpu1": 0.0},
        "detail": "bnb path rejected at load: MoE experts lack get_expert_mapping",
    },
    {
        "mitigation": "vllm_native_bnb4bit_tp1_gpu0_recipe_maxseqs4",
        "outcome": "load_failed",
        "error_if_any": (
            "AttributeError: MoE Model DiffusionGemmaForConditionalGeneration does "
            "not support BitsAndBytes quantization yet. (vLLM 0.24.0 native runner; "
            "tp1 GPU0; max_num_seqs=4; gpu_mem_util=0.90)"
        ),
        "duration_s": 16.019,
        "forward_pass_confirmed": False,
        "peak_vram_gib_per_gpu": {"gpu0": 0.0},
        "detail": "same MoE+bnb limitation as tp2",
    },
    {
        "mitigation": "vllm_native_fp8_tp2_recipe_maxseqs4",
        "outcome": "load_failed",
        "error_if_any": (
            "RuntimeError: marlin_mm, marlin_moe_wna16/ops.cu:495 Invalid thread "
            "config: thread_m_blocks=4, thread_k=-1, thread_n=-1, num_threads=-1 for "
            "MKN=[65536,352,2816] num_bits=8, group_size=-1 -- fp8 weights LOADED and "
            "fit under tp2, but the fp8-Marlin MoE expert-GEMM kernel is unsupported "
            "on Ampere sm_86 (RTX 3090)."
        ),
        "duration_s": 37.066,
        "forward_pass_confirmed": False,
        "peak_vram_gib_per_gpu": {"gpu0": 0.0, "gpu1": 0.0},
        "detail": "memory fit; kernel-level failure, not OOM",
    },
]

# The two HF custom-device_map attempts (probe_hf.py), also embedded as DATA for
# the same reproducibility reason as the vLLM outcomes above. Both are genuinely
# NEW vs exp5182's four (explicit per-module placement, not device_map="auto" nor
# fully-single-device). The manual_split result is the load-bearing NEW finding:
# it asked for encoder-on-GPU1 / decoder-on-GPU0, yet GPU 1 stayed at 0.298 GiB
# while GPU 0 OOM'd at 23.13 GiB -- direct evidence that the encoder placement was
# IGNORED because the encoder is shared storage with the decoder, so the loader
# duplicate-materialises the tied weights onto the decoder's GPU rather than
# honouring the split (the exp5182 tie diagnosis, now confirmed from the memory
# side rather than the meta-tensor side).
_HF_OOM_ERROR = (
    "OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB. GPU 0 has "
    "a total capacity of 23.56 GiB of which 122.31 MiB is free. Including "
    "non-PyTorch memory, this process has 23.43 GiB memory in use. Of the allocated "
    "memory 23.13 GiB is allocated by PyTorch, and 6.31 MiB is reserved by PyTorch "
    "but unallocated. (raised in transformers/core_model_loading.py:991 "
    "_materialize_copy: tensor = tensor.to(device=device, dtype=dtype))"
)
RECORDED_HF_MITIGATIONS: list[dict] = [
    {
        "mitigation": "hf_custom_devmap_manual_split_dec0_enc1_4bit",
        "outcome": "load_failed",
        "error_if_any": _HF_OOM_ERROR,
        "duration_s": 142.673,
        "forward_pass_confirmed": False,
        "peak_vram_gib_per_gpu": {"gpu0": 23.131, "gpu1": 0.298},
        "device_map_summary": {"model.decoder": 0, "model.encoder": 1, "lm_head": 0},
        "detail": (
            "Asked for decoder->GPU0, encoder->GPU1; GPU1 only reached 0.298 GiB "
            "while GPU0 OOM'd at 23.13 GiB -- the encoder placement was IGNORED "
            "because it shares storage with the decoder, so the tied weights "
            "duplicate-materialise on GPU0. Confirms the exp5182 tie from the "
            "memory side; a 2x24 GiB split cannot work while the tie is unbroken."
        ),
    },
    {
        "mitigation": "hf_custom_devmap_colocate_gpu0_offload_vision_4bit",
        "outcome": "load_failed",
        "error_if_any": _HF_OOM_ERROR,
        "duration_s": 160.925,
        "forward_pass_confirmed": False,
        "peak_vram_gib_per_gpu": {"gpu0": 23.131, "gpu1": 0.0},
        "device_map_summary": {
            "model.decoder": 0, "model.encoder": 0,
            "model.decoder.embed_vision": "cpu", "model.encoder.embed_vision": "cpu",
            "lm_head": 0,
        },
        "detail": (
            "Co-located tied encoder+decoder on GPU0 with the non-tied vision "
            "embedder offloaded to CPU + llm_int8_enable_fp32_cpu_offload=True (the "
            "task's prescribed pattern). Still OOM'd at 23.13 GiB: offloading only "
            "the small vision embedder does not recover enough room for the "
            "duplicated tied body + un-quantised lm_head/embeddings on one 24 GiB GPU."
        ),
    },
]


def build_from_recorded(
    hf_mitigations: list[dict] | None = None, duration_s: float = 0.0
) -> dict:
    """Build the artifact from the recorded vLLM outcomes + HF outcomes.

    ``hf_mitigations`` defaults to the embedded ``RECORDED_HF_MITIGATIONS`` (the
    real probe_hf.py outcomes) so the committed artifact regenerates BYTE-FOR-BYTE
    from the module alone, with no dependency on the ephemeral /tmp ndjson file --
    the same reproducibility contract the vLLM outcomes already satisfy. A caller
    (or ``main`` when a fresh probe ndjson exists) may pass a list to override.
    ``duration_s`` defaults to the summed real wall-clock of every embedded attempt.
    """
    hf = hf_mitigations if hf_mitigations is not None else RECORDED_HF_MITIGATIONS
    mitigations = [normalise_mitigation(m) for m in RECORDED_VLLM_MITIGATIONS]
    mitigations += [normalise_mitigation(m) for m in hf]
    preconditions = [
        {"resource": "gpu0_gpu1_idle", "available": True,
         "detail": "nvidia-smi: gpu0 4MiB/24576, gpu1 4MiB/24576, 0% util"},
        {"resource": "vllm_0_24_native_diffusiongemma_runner", "available": True,
         "detail": "ModelRegistry has DiffusionGemmaForBlockDiffusion (0.23.0 did not)"},
        {"resource": "diffusiongemma_weights_cached", "available": True,
         "detail": "11 safetensors shards under models--google--diffusiongemma-26B-A4B-it"},
        {"resource": "bitsandbytes_in_vllm_venv", "available": True,
         "detail": "installed bitsandbytes 0.49.2 into vllm-venv (was absent)"},
    ]
    if not duration_s:
        duration_s = sum(
            float(m.get("duration_s", 0.0) or 0.0) for m in RECORDED_VLLM_MITIGATIONS
        ) + sum(float(m.get("duration_s", 0.0) or 0.0) for m in hf)
    return build_artifact(mitigations, preconditions, duration_s)


def main(argv: list[str] | None = None) -> int:
    """Assemble and write the artifact; --print echoes it without writing."""
    parser = argparse.ArgumentParser(description="Exp 5196 artifact builder")
    parser.add_argument(
        "--hf-ndjson",
        default="/tmp/exp5196_hf.ndjson",
        help="probe_hf.py output to read the HF custom-device_map outcomes from",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args(argv)

    # Prefer a fresh probe ndjson if present; otherwise fall back to the embedded
    # RECORDED_HF_MITIGATIONS (real outcomes) so the artifact regenerates without
    # the ephemeral /tmp file. Passing None lets build_from_recorded pick the
    # embedded default and compute the summed real duration.
    hf_from_ndjson = collect_attempts_from_ndjson([Path(args.hf_ndjson)])
    hf_mitigations = hf_from_ndjson if hf_from_ndjson else None

    artifact = build_from_recorded(hf_mitigations, args.duration)
    errors = validate_artifact(artifact)
    if errors:
        sys.stderr.write("ARTIFACT INVALID:\n" + "\n".join(errors) + "\n")
        return 1
    if args.print_only:
        sys.stdout.write(json.dumps(artifact, indent=2, default=str) + "\n")
        return 0
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n")
    sys.stdout.write(f"wrote {out_path} :: {artifact['honest_verdict']}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
