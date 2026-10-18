"""Standalone HF transformers custom-device_map DiffusionGemma load probe (Exp 5196).

Runs in the main Carnot ``.venv`` (transformers 5.12.0 + bitsandbytes 0.49.2).
This is the SECONDARY path after the vLLM-native path is exhausted, and it makes
attempts that are genuinely NEW versus exp5182's four (which were all
device_map="auto" or fully-single-device with NO explicit per-module placement):

- ``manual_split_dec0_enc1``: place the ENTIRE decoder on GPU 0 and the ENTIRE
  encoder on GPU 1, at clean model-boundary granularity. exp5182 diagnosed that
  device_map="auto" broke the encoder<->decoder weight tie by splitting layers
  arbitrarily across GPUs (meta-tensor at forward). A manual split that respects
  the two top-level model boundaries is the controlled test of that hypothesis
  AND uses the full 2x24 GiB budget (each ~12.9 GiB 4-bit half fits its own GPU).
- ``colocate_gpu0_offload_vision``: the task's prescribed pattern -- co-locate the
  tied encoder+decoder on GPU 0 and offload the (non-tied) vision embedder to CPU
  with ``llm_int8_enable_fp32_cpu_offload=True``.

Emits newline-delimited JSON events (flushed per line) so a killed process still
leaves a trail. A load that returns is NOT enough -- a real forward pass whose
output ``.item()`` succeeds (i.e. not a meta tensor) is required to count as
``forward_pass_confirmed`` (the exp5182 precedent).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def _vram_per_gpu() -> dict:
    import torch

    out = {}
    for i in range(torch.cuda.device_count()):
        out[f"gpu{i}"] = round(torch.cuda.max_memory_allocated(i) / (1024**3), 3)
    return out


def _confirm_forward(model) -> tuple[bool, str]:
    """Run one real forward pass and prove the output is not a meta tensor."""
    import torch

    try:
        input_ids = torch.tensor([[2, 651, 6238, 563, 573, 12]], dtype=torch.long)
        # Place inputs on the first parameter's device.
        first_dev = next(model.parameters()).device
        if first_dev.type == "cuda":
            input_ids = input_ids.to(first_dev)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        tensor = getattr(out, "logits", None)
        if tensor is None:
            tensor = getattr(out, "last_hidden_state", None)
        if tensor is None and isinstance(out, (tuple, list)):
            tensor = out[0]
        if tensor is None:
            return False, "forward returned no logits/last_hidden_state tensor"
        # The exp5182 failure mode was exactly Tensor.item() on a meta tensor.
        val = float(tensor.float().flatten()[0].item())
        return True, f"forward ok, first_logit={val:.4f}, shape={tuple(tensor.shape)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:600]


def _load(model_cls, device_map, extra_kwargs) -> object:
    import torch
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        **extra_kwargs.pop("bnb_extra", {}),
    )
    return model_cls.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map=device_map,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    )


def _attempt(label: str, build_map, extra_kwargs) -> dict:
    import torch

    # CUDA is lazily initialised; force it before touching memory stats, else
    # reset_peak_memory_stats raises "Invalid device argument" on a cold context.
    torch.cuda.init()
    for i in range(torch.cuda.device_count()):
        try:
            torch.zeros(1, device=f"cuda:{i}")
            torch.cuda.reset_peak_memory_stats(i)
        except Exception:  # pragma: no cover - defensive cold-context guard
            pass
    started = time.time()
    result = {
        "event": "attempt",
        "mitigation": label,
        "outcome": "unknown",
        "forward_pass_confirmed": False,
        "forward_detail": None,
        "peak_vram_gib_per_gpu": {},
        "error_if_any": None,
        "duration_s": 0.0,
    }
    model = None
    try:
        from transformers import DiffusionGemmaForBlockDiffusion

        device_map = build_map(DiffusionGemmaForBlockDiffusion)
        result["device_map_summary"] = {
            k: v for k, v in list(device_map.items())[:12]
        }
        model = _load(DiffusionGemmaForBlockDiffusion, device_map, dict(extra_kwargs))
        result["outcome"] = "load_ok"
        ok, detail = _confirm_forward(model)
        result["forward_pass_confirmed"] = ok
        result["forward_detail"] = detail
        result["outcome"] = "forward_pass_ok" if ok else "loaded_no_forward"
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = "load_failed"
        result["error_if_any"] = f"{type(exc).__name__}: {exc}"[:900]
        result["traceback_tail"] = "".join(
            traceback.format_exc().splitlines(keepends=True)[-5:]
        )[:900]
    finally:
        try:
            result["peak_vram_gib_per_gpu"] = _vram_per_gpu()
        except Exception:  # pragma: no cover - defensive
            pass
        result["duration_s"] = round(time.time() - started, 3)
        del model
    _emit(result)
    return result


def _map_manual_split(_cls) -> dict:
    """Whole decoder on GPU 0, whole encoder on GPU 1 (clean model boundaries)."""
    return {"model.decoder": 0, "model.encoder": 1, "lm_head": 0}


def _map_colocate_offload(_cls) -> dict:
    """Tied encoder+decoder on GPU 0; nested vision embedder offloaded to CPU."""
    return {
        "model.decoder": 0,
        "model.encoder": 0,
        "model.decoder.embed_vision": "cpu",
        "model.encoder.embed_vision": "cpu",
        "lm_head": 0,
    }


def main() -> int:
    attempt = sys.argv[1] if len(sys.argv) > 1 else "manual_split"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if attempt == "manual_split":
        _attempt("hf_custom_devmap_manual_split_dec0_enc1_4bit", _map_manual_split, {})
    elif attempt == "colocate_offload":
        _attempt(
            "hf_custom_devmap_colocate_gpu0_offload_vision_4bit",
            _map_colocate_offload,
            {"bnb_extra": {"llm_int8_enable_fp32_cpu_offload": True}},
        )
    else:
        _emit({"event": "error", "detail": f"unknown attempt {attempt!r}"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
