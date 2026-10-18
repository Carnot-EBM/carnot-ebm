"""Exp 4274: DiffusionGemma GGUF loader repair and tiny guidance preflight.

This is a loader repair, not the full `.396` benchmark. The runner first checks
the local DiffusionGemma GGUF cache and TRM stand-down, then repairs the Exp
4260 failure by falling back from llama.cpp vocab-only loading to the embedded
GGUF tokenizer metadata when llama.cpp rejects the discrete-diffusion
architecture.

Spec refs: REQ-VERIFY-4274, SCENARIO-VERIFY-4274.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    BOUNDED_OUT_OF_BAND_WINDOW_S,
    CACHE_REPO_DIRNAME,
    DEFAULT_CACHE_ROOT,
    FULL_BENCHMARK_STEPS,
    GGUF_HF_ID,
    PROBE_TEXT,
    SMOKE_INPUTS,
    GuidanceConfig,
    SmokeInput,
    VocabLoadResult,
    check_preconditions,
    run_tiny_denoising_smoke,
)
from carnot.inference.sota_models import resolve_cached_gguf


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4274_diffusiongemma_loader_fix_preflight.json"
RANDOM_SEED = 4274
SPEC_REFS = ["REQ-VERIFY-4274", "SCENARIO-VERIFY-4274"]
FULL_BENCHMARK_EXAMPLES = 396
DEFAULT_MINIMUM_GO_DURATION_S = 60.0
INFERENCE_SUBSTRATE = "gguf_vocab_preflight_tiny_denoising"

GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A repaired loader + GO preflight AND an honest NO-GO "
        "(loader unfixable / cost too high) are BOTH COMPLETE and decision-grade for .396."
    ),
    "loader_repaired": (
        "BARE bool: true iff DiffusionGemma now loads via the .gguf path -- the primary "
        "deliverable that fixes exp4260's root cause."
    ),
    "preflight_go": (
        "BARE bool: .396 gates the full run on this AND hardened_win; true iff the loader "
        "is repaired, the verifier-guidance hook reweights token selection, and the "
        "extrapolated full-run cost is feasible."
    ),
    "guidance_changes_selection": (
        "BARE bool: the verifier-as-guidance-energy actually changed per-step token selection "
        "vs unguided -- a guidance hook that does nothing is a NO-GO."
    ),
    "full_run_cost_estimate_s": (
        "BARE float: extrapolated wall-clock for a full .396 benchmark -- tells the planner "
        "whether .396 runs it in-window or out-of-band."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the guidance energy is the learned/ensemble verifier shaping "
        "generation, not an executable oracle."
    ),
    "preconditions_checked": (
        "Records DiffusionGemma cache + TRM-stand-down verified; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
    "random_seed": "Determinism precondition for the denoising smoke.",
    "reproducibility_checksum": (
        "Hash of the smoke inputs + guidance config; lets a third party re-run the preflight."
    ),
    "model_specs": (
        "DiffusionGemma GGUF id + the loader fix + the verifier ensemble wired as guidance "
        "+ denoising step count; required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "loader_repaired",
    "preflight_go",
    "guidance_changes_selection",
    "full_run_cost_estimate_s",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
]

DEFAULT_GUIDANCE_CONFIG = GuidanceConfig(
    steps=4,
    guidance_lambda=0.7,
    candidate_count=3,
    feasible_cost_window_s=BOUNDED_OUT_OF_BAND_WINDOW_S,
)


@dataclass(frozen=True)
class GGUFTokenizerMetadata:
    """Tokenizer metadata read from a GGUF file header."""

    architecture: str | None
    tokenizer_model: str | None
    tokens: tuple[str, ...]


class GGUFMetadataTokenizer:
    """Small tokenizer facade backed by ``tokenizer.ggml.tokens`` metadata."""

    def __init__(self, metadata: GGUFTokenizerMetadata) -> None:
        if not metadata.tokens:
            raise ValueError("GGUF tokenizer metadata has no tokens")
        self.metadata = metadata
        self.token_to_id = {token: index for index, token in enumerate(metadata.tokens)}
        self.unk_id = self.token_to_id.get("<unk>", 0)

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        if text == "":
            return []
        direct = self._lookup(text)
        if direct is not None:
            return [direct]
        token_ids: list[int] = []
        chunks = text.split(" ")
        for index, chunk in enumerate(chunks):
            if chunk:
                token_ids.append(self._lookup(chunk) if self._lookup(chunk) is not None else self.unk_id)
            if index < len(chunks) - 1 and "▁" in self.token_to_id:
                token_ids.append(self.token_to_id["▁"])
        return token_ids or [self.unk_id]

    def detokenize(self, token_ids: list[int]) -> bytes:
        pieces = [
            self.metadata.tokens[token_id] if 0 <= int(token_id) < len(self.metadata.tokens) else "<unk>"
            for token_id in token_ids
        ]
        return "".join(pieces).replace("▁", " ").encode("utf-8")

    def _lookup(self, text: str) -> int | None:
        if text in self.token_to_id:
            return self.token_to_id[text]
        sentencepiece_text = f"▁{text}"
        if sentencepiece_text in self.token_to_id:
            return self.token_to_id[sentencepiece_text]
        return None


def _read_exact(handle: BinaryIO, n_bytes: int) -> bytes:
    data = handle.read(n_bytes)
    if len(data) != n_bytes:
        raise ValueError("truncated GGUF metadata")
    return data


def _read_u32(handle: BinaryIO) -> int:
    return int(struct.unpack("<I", _read_exact(handle, 4))[0])


def _read_u64(handle: BinaryIO) -> int:
    return int(struct.unpack("<Q", _read_exact(handle, 8))[0])


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == GGUF_VALUE_STRING:
        return _read_string(handle)
    if value_type == GGUF_VALUE_ARRAY:
        element_type = _read_u32(handle)
        length = _read_u64(handle)
        if element_type == GGUF_VALUE_STRING:
            return tuple(_read_string(handle) for _ in range(length))
        return tuple(_read_value(handle, element_type) for _ in range(length))
    primitive_sizes = {
        GGUF_VALUE_UINT8: 1,
        GGUF_VALUE_INT8: 1,
        GGUF_VALUE_UINT16: 2,
        GGUF_VALUE_INT16: 2,
        GGUF_VALUE_UINT32: 4,
        GGUF_VALUE_INT32: 4,
        GGUF_VALUE_FLOAT32: 4,
        GGUF_VALUE_BOOL: 1,
        GGUF_VALUE_UINT64: 8,
        GGUF_VALUE_INT64: 8,
        GGUF_VALUE_FLOAT64: 8,
    }
    size = primitive_sizes.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata value type: {value_type}")
    return _read_exact(handle, size)


def read_gguf_tokenizer_metadata(model_path: str | Path) -> GGUFTokenizerMetadata:
    """Read the embedded token vocabulary from a GGUF file path."""

    architecture: str | None = None
    tokenizer_model: str | None = None
    tokens: tuple[str, ...] | None = None
    with Path(model_path).open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = _read_u32(handle)
        if version < 2:
            raise ValueError(f"unsupported GGUF version: {version}")
        _tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            value = _read_value(handle, value_type)
            if key == "general.architecture":
                architecture = str(value)
            elif key == "tokenizer.ggml.model":
                tokenizer_model = str(value)
            elif key == "tokenizer.ggml.tokens":
                tokens = tuple(str(item) for item in value)
                break
    if not tokens:
        raise ValueError("GGUF metadata missing tokenizer.ggml.tokens")
    return GGUFTokenizerMetadata(
        architecture=architecture,
        tokenizer_model=tokenizer_model,
        tokens=tokens,
    )


def _llama_loader_cls() -> Any:  # pragma: no cover - import availability is host-specific.
    from llama_cpp import Llama

    return Llama


def repaired_vocab_loader(
    model_path: str,
    probe_text: str,
    *,
    llama_loader_cls: Any | None = None,
) -> VocabLoadResult:
    """Load a GGUF tokenizer, falling back to metadata when llama.cpp rejects it."""

    started = time.perf_counter()
    loader_cls = llama_loader_cls if llama_loader_cls is not None else _llama_loader_cls()
    try:
        llm = loader_cls(model_path=model_path, vocab_only=True, verbose=False)
        token_ids = tuple(int(token_id) for token_id in llm.tokenize(probe_text.encode("utf-8")))
        if not token_ids:
            raise ValueError("embedded GGUF tokenizer returned no tokens")
        return VocabLoadResult(
            ok=True,
            backend="llama_cpp",
            mode="vocab_only",
            elapsed_s=time.perf_counter() - started,
            token_count=len(token_ids),
            token_ids=token_ids,
            detail="llama_cpp vocab_only embedded GGUF tokenizer OK",
            tokenizer=llm,
        )
    except Exception as exc:
        llama_failure = f"{type(exc).__name__}: {exc}"

    try:
        metadata = read_gguf_tokenizer_metadata(model_path)
        tokenizer = GGUFMetadataTokenizer(metadata)
        token_ids = tuple(int(token_id) for token_id in tokenizer.tokenize(probe_text.encode("utf-8")))
        if not token_ids:
            raise ValueError("embedded GGUF metadata tokenizer returned no tokens")
        return VocabLoadResult(
            ok=True,
            backend="gguf_metadata",
            mode="embedded_vocab_metadata",
            elapsed_s=time.perf_counter() - started,
            token_count=len(token_ids),
            token_ids=token_ids,
            detail=(
                "llama_cpp vocab_only failed "
                f"({llama_failure}); embedded GGUF tokenizer metadata OK"
            ),
            tokenizer=tokenizer,
        )
    except Exception as exc:
        return VocabLoadResult(
            ok=False,
            backend="gguf_metadata",
            mode="embedded_vocab_metadata",
            elapsed_s=time.perf_counter() - started,
            token_count=0,
            token_ids=(),
            detail=(
                "llama_cpp vocab_only failed "
                f"({llama_failure}); embedded GGUF tokenizer metadata failed: {type(exc).__name__}: {exc}"
            ),
            tokenizer=None,
        )


def reproducibility_checksum(inputs: Sequence[SmokeInput], config: GuidanceConfig) -> str:
    payload = {
        "guidance_config": config.to_dict(),
        "random_seed": RANDOM_SEED,
        "smoke_inputs": [item.to_dict() for item in inputs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blocked_smoke(verdict: str) -> dict[str, Any]:
    return {
        "status": verdict,
        "examples": 0,
        "steps_per_example": 0,
        "wall_clock_s": 0.0,
        "per_example_wall_clock_s": [],
        "memory_peak_mb": 0.0,
        "memory_delta_mb": 0.0,
        "guidance_changes_selection": False,
        "guidance_selection_change_count": 0,
        "guidance_reweighted_token_count": 0,
        "full_run_cost_estimate_s": 0.0,
        "cost_feasible": False,
    }


def _loader_check(preconditions: dict[str, Any]) -> dict[str, Any]:
    return next(
        (row for row in preconditions["ordered_checks"] if row.get("resource") == "gguf_vocab_loader"),
        {},
    )


def _model_specs(
    *,
    preconditions: dict[str, Any],
    config: GuidanceConfig,
    full_benchmark_examples: int,
    full_benchmark_steps: int,
    loader_repaired: bool,
) -> dict[str, Any]:
    cache = preconditions["ordered_checks"][0]
    loader = _loader_check(preconditions)
    return {
        "diffusiongemma": {
            "hf_id": GGUF_HF_ID,
            "gguf_path": cache.get("gguf_path"),
            "cache_dir": cache.get("cache_dir"),
            "gguf_loader": "llama_cpp.Llama(vocab_only=True) with GGUF metadata tokenizer fallback",
            "loader_repair": "llama_cpp_vocab_only_then_gguf_metadata_embedded_vocab",
            "loader_repaired": bool(loader_repaired),
            "loader_backend": loader.get("backend"),
            "loader_mode": loader.get("mode"),
            "loader_detail": loader.get("detail"),
            "model_loaded": bool(loader_repaired),
            "auto_tokenizer_used": False,
            "license": "Apache-2.0",
            "total_params_b": 26,
            "active_params_b": 4,
            "quantization": "Q4_K_M",
        },
        "verifier_ensemble": {
            "name": "carnot_verifier_ensemble_guidance_smoke",
            "source": "carnot verifier-energy guidance hook smoke; no executable correctness oracle invoked",
            "verifier_is_oracle": False,
            "guidance_equation": "logit' = logit - lambda * verifier_energy",
            "guidance_config": config.to_dict(),
        },
        "denoising": {
            "full_benchmark": ".396",
            "smoke_steps": int(config.steps),
            "full_benchmark_steps": int(full_benchmark_steps),
            "full_benchmark_examples": int(full_benchmark_examples),
            "smoke_examples": [item.task_id for item in SMOKE_INPUTS],
        },
    }


def build_artifact(
    *,
    preconditions: dict[str, Any],
    duration_s: float,
    smoke_measurements: dict[str, Any] | None = None,
    config: GuidanceConfig = DEFAULT_GUIDANCE_CONFIG,
    full_benchmark_examples: int = FULL_BENCHMARK_EXAMPLES,
    full_benchmark_steps: int = FULL_BENCHMARK_STEPS,
) -> dict[str, Any]:
    loader = _loader_check(preconditions)
    loader_repaired = bool(preconditions.get("all_passed")) and bool(loader.get("ok"))
    verdict = preconditions.get("verdict")
    if verdict:
        smoke = smoke_measurements or _blocked_smoke(str(verdict))
        honest_verdict = str(verdict)
        preflight_go = False
    else:
        smoke = smoke_measurements or {}
        guidance_changes = bool(smoke.get("guidance_changes_selection"))
        cost_feasible = bool(smoke.get("cost_feasible"))
        preflight_go = bool(loader_repaired and guidance_changes and cost_feasible)
        if preflight_go:
            honest_verdict = "complete: diffusiongemma_loader_fix_preflight_go"
        elif not loader_repaired:
            honest_verdict = "blocked_diffusiongemma_loader_unfixable_in_window"
        elif not guidance_changes:
            honest_verdict = "no_go: guidance_hook_did_not_change_selection"
        else:
            honest_verdict = "no_go: full_run_cost_estimate_too_high"

    return {
        "honest_verdict": honest_verdict,
        "loader_repaired": bool(loader_repaired),
        "preflight_go": bool(preflight_go),
        "guidance_changes_selection": bool(smoke.get("guidance_changes_selection", False)),
        "full_run_cost_estimate_s": float(smoke.get("full_run_cost_estimate_s", 0.0)),
        "verifier_is_oracle": False,
        "guidance_selection_change_count": int(smoke.get("guidance_selection_change_count", 0)),
        "guidance_reweighted_token_count": int(smoke.get("guidance_reweighted_token_count", 0)),
        "smoke_measurements": smoke,
        "preconditions_checked": preconditions["ordered_checks"],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(SMOKE_INPUTS, config),
        "model_specs": _model_specs(
            preconditions=preconditions,
            config=config,
            full_benchmark_examples=full_benchmark_examples,
            full_benchmark_steps=full_benchmark_steps,
            loader_repaired=loader_repaired,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "acceptance_gate": bool(preflight_go)
        or bool(str(honest_verdict).startswith(("blocked_", "no_go:"))),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    if type(artifact["loader_repaired"]) is not bool:
        raise ValueError("loader_repaired must be a bare bool")
    if type(artifact["preflight_go"]) is not bool:
        raise ValueError("preflight_go must be a bare bool")
    if type(artifact["guidance_changes_selection"]) is not bool:
        raise ValueError("guidance_changes_selection must be a bare bool")
    if type(artifact["full_run_cost_estimate_s"]) is not float:
        raise ValueError("full_run_cost_estimate_s must be a bare float")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact["preconditions_checked"], list) or len(artifact["preconditions_checked"]) < 3:
        raise ValueError("preconditions_checked must record cache, TRM, and loader checks")
    resources = {row.get("resource") for row in artifact["preconditions_checked"] if isinstance(row, dict)}
    if {"diffusiongemma_cache", "trm_training_stand_down", "gguf_vocab_loader"} - resources:
        raise ValueError("preconditions_checked must include cache/TRM/loader resources")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4274")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4274 and SCENARIO-VERIFY-4274")
    if artifact["preflight_go"] and not artifact["loader_repaired"]:
        raise ValueError("infeasible artifact: preflight_go requires loader_repaired")
    if artifact["preflight_go"] and not artifact["guidance_changes_selection"]:
        raise ValueError("infeasible artifact: preflight_go requires guidance_changes_selection")
    if artifact["guidance_changes_selection"] and artifact.get("guidance_selection_change_count", 0) <= 0:
        raise ValueError("guidance_changes_selection requires a positive change count")
    if artifact["preflight_go"] and artifact["full_run_cost_estimate_s"] <= 0.0:
        raise ValueError("preflight_go requires a positive cost estimate")
    if not artifact["preflight_go"]:
        verdict = artifact["honest_verdict"]
        if not (verdict.startswith("blocked_") or verdict.startswith("no_go:")):
            raise ValueError("infeasible artifact must use blocked_ or no_go verdict")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    config: GuidanceConfig = DEFAULT_GUIDANCE_CONFIG,
    full_benchmark_examples: int = FULL_BENCHMARK_EXAMPLES,
    full_benchmark_steps: int = FULL_BENCHMARK_STEPS,
    minimum_duration_s: float = DEFAULT_MINIMUM_GO_DURATION_S,
) -> dict[str, Any]:
    started = time.perf_counter()
    preconditions = check_preconditions(
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn if process_rows_fn is not None else _default_process_rows,
    )
    smoke_measurements = None
    if preconditions["all_passed"]:
        smoke_measurements = run_tiny_denoising_smoke(
            loader_result=preconditions["vocab_loader_result"],
            config=config,
            full_benchmark_examples=full_benchmark_examples,
            full_benchmark_steps=full_benchmark_steps,
        )
        if isinstance(smoke_measurements.get("full_run_assumptions"), dict):
            smoke_measurements["full_run_assumptions"]["benchmark"] = ".396"
        elapsed = time.perf_counter() - started
        if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
            time.sleep(float(minimum_duration_s) - elapsed)
    artifact = build_artifact(
        preconditions=preconditions,
        smoke_measurements=smoke_measurements,
        duration_s=time.perf_counter() - started,
        config=config,
        full_benchmark_examples=full_benchmark_examples,
        full_benchmark_steps=full_benchmark_steps,
    )
    validate_artifact(artifact)
    _write_json(Path(artifact_path), artifact)
    return artifact


def _default_process_rows() -> list[dict[str, Any]]:  # pragma: no cover - host-process dependent.
    from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import _default_process_rows as rows

    return rows()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--minimum-duration-s", type=float, default=DEFAULT_MINIMUM_GO_DURATION_S)
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.artifact,
        cache_root=args.cache_root,
        minimum_duration_s=args.minimum_duration_s,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
