from importlib import import_module
from typing import Any

from carnot.inference.dual_gpu import (
    DualGPUExecutionResult,
    DualGPURunner,
    estimate_model_size_billions,
    requires_device_map_auto,
)

_LAZY_EXPORTS = {
    "ModelLoadError": ("carnot.errors", "ModelLoadError"),
    "load_model": ("carnot.inference.model_loader", "load_model"),
    "generate": ("carnot.inference.model_loader", "generate"),
    "KNOWN_MODELS": ("carnot.inference.ebm_loader", "KNOWN_MODELS"),
    "get_model_info": ("carnot.inference.ebm_loader", "get_model_info"),
    "load_ebm": ("carnot.inference.ebm_loader", "load_ebm"),
    "EBMCandidateScore": ("carnot.inference.ebm_rejection", "EBMCandidateScore"),
    "EBMRejectionConfig": ("carnot.inference.ebm_rejection", "EBMRejectionConfig"),
    "EBMRejectionResult": ("carnot.inference.ebm_rejection", "EBMRejectionResult"),
    "ebm_rejection_sample": ("carnot.inference.ebm_rejection", "ebm_rejection_sample"),
    "score_activations_with_ebm": (
        "carnot.inference.ebm_rejection",
        "score_activations_with_ebm",
    ),
    "TokenEnergyAnalysis": ("carnot.inference.arm_ebm_bridge", "TokenEnergyAnalysis"),
    "analyze_token_energy": ("carnot.inference.arm_ebm_bridge", "analyze_token_energy"),
    "compute_sequence_energy": ("carnot.inference.arm_ebm_bridge", "compute_sequence_energy"),
    "extract_token_rewards": ("carnot.inference.arm_ebm_bridge", "extract_token_rewards"),
    "extract_token_rewards_from_logprobs": (
        "carnot.inference.arm_ebm_bridge",
        "extract_token_rewards_from_logprobs",
    ),
    "identify_hallucination_tokens": (
        "carnot.inference.arm_ebm_bridge",
        "identify_hallucination_tokens",
    ),
    "ModelServer": ("carnot.inference.model_server", "ModelServer"),
    "benchmark_cold_load_vs_warm_server": (
        "carnot.inference.model_server",
        "benchmark_cold_load_vs_warm_server",
    ),
    "multi_start_repair": ("carnot.inference.multi_start", "multi_start_repair"),
    "PROGRSCentering": ("carnot.inference.jepa_cpmi_pairs", "PROGRSCentering"),
}

_LAZY_SUBMODULES = {"model_server", "sota_models"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, symbol_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), symbol_name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DualGPUExecutionResult",
    "DualGPURunner",
    "estimate_model_size_billions",
    "requires_device_map_auto",
    *_LAZY_EXPORTS,
    *_LAZY_SUBMODULES,
]
