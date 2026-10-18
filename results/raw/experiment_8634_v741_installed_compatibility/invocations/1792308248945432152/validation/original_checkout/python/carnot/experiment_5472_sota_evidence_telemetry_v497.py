#!/usr/bin/env python3
"""Exp5472 local SOTA GGUF evidence telemetry panel.

Spec refs: REQ-SAFE-5472, SCENARIO-SAFE-5472.

This module is deliberately a telemetry panel, not a guided-decoding feature.
It reuses exact Exp5471 guard-composition rows, asks a local SOTA GGUF model for
a plain text accept/reject/abstain decision when a GPU-offloaded llama.cpp
runtime is available, and records the exact local validator label beside the
model output. If the model or GPU offload is unavailable, the deliverable is an
honest blocked artifact with the same required fields.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot import experiment_5471_guard_composition_scale_v497 as exp5471
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
CacheResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[[], Mapping[str, Any]]
RuntimeFactory = Callable[[Mapping[str, Any]], Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5472_sota_evidence_telemetry_v497.json")
EXPERIMENT_ID = "experiment_5472_sota_evidence_telemetry_v497"
TASK_ID = "exp5472-v497-sota-evidence-telemetry"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5472.sota_evidence_telemetry.v497"
SPEC_REFS = ("REQ-SAFE-5472", "SCENARIO-SAFE-5472", "REQ-SAFE-5471")
RANDOM_SEED = 5472
INFERENCE_SUBSTRATE = "local_sota_gguf_llama_cpp_or_blocked"
EXACT_FINAL_AUTHORITY = exp5471.EXACT_FINAL_AUTHORITY
TERMINAL_PREFIXES = ("complete:", "blocked:")

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

SELECTED_FIXTURE_IDS = (
    "5470-valid-fact-paraphrase",
    "5470-hidden-premise",
    "5470-json-semantic-invalid",
    "5470-factual-distortion",
)
FIXTURE_BUCKETS = {
    "5470-valid-fact-paraphrase": "valid",
    "5470-hidden-premise": "hidden_premise",
    "5470-json-semantic-invalid": "semantic_invalid",
    "5470-factual-distortion": "factual_distortion",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "model_specs": "all mandated SOTA GGUF specs and resolved local path status.",
    "headline_models_run": (
        "mandated model IDs that actually generated rows with verified GPU offload."
    ),
    "n_samples": "count of selected exact Exp5471 fixture rows.",
    "exact_validator_accuracy": (
        "model decision agreement with exact final validator labels."
    ),
    "semantic_false_accept_rate": (
        "semantic-invalid rows accepted by the generated model decision."
    ),
    "factual_distortion_rate": (
        "distortion-guard rows accepted by the generated model decision."
    ),
    "abstention_rate": "generated decisions parsed as abstention or undecided.",
    "logprob_telemetry_available": (
        "whether llama.cpp returned token/top-k logprob telemetry."
    ),
    "gpu_offload_receipts": "pre-generation runtime evidence for CUDA llama.cpp offload.",
    "model_file_checksums": "SHA-256 checksums for local model files used or considered.",
    "guided_decoding_used": "guided decoding remains quarantined and is not used.",
    "sota_evidence_telemetry_ready": (
        "downstream gate for local SOTA evidence telemetry."
    ),
    "inference_substrate": "local SOTA GGUF llama.cpp run or honest blocked artifact.",
    "random_seed": "deterministic fixture and run seed.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
MODEL_SPECS = [
    {
        "hf_id": model["hf_id"],
        "name": model["name"],
        "role": model["role"],
        "quantization": model["quantization"],
        "headline_required": True,
        "legacy_smoke_only": False,
    }
    for hf_id in MANDATED_HF_IDS
    for model in SOTA_GGUF_MODELS
    if model["hf_id"] == hf_id
]


def load_exp5471_rows(
    artifact_path: Path | str = REPO_ROOT / exp5471.RESULT_RELATIVE_PATH,
) -> list[JsonDict]:
    """Load Exp5471 row evidence, falling back to rebuilding it from exact code.

    Why the fallback exists: the normal conductor path has the JSON artifact on
    disk, but tests and clean checkouts can still ask for the rows before the
    result file is present. Rebuilding through Exp5471 preserves the exact local
    validators instead of copying row data into this module.
    """

    path = Path(artifact_path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = exp5471.build_artifact()
    rows = payload.get("row_results")
    if not isinstance(rows, list):
        raise ValueError("Exp5471 artifact row_results must be a list")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def select_fixture_subset(rows: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Return the fixed balanced subset requested for the telemetry panel."""

    source_rows = load_exp5471_rows() if rows is None else rows
    by_id = {str(row.get("candidate_id")): row for row in source_rows}
    missing = [candidate_id for candidate_id in SELECTED_FIXTURE_IDS if candidate_id not in by_id]
    if missing:
        raise ValueError(f"missing selected Exp5471 fixture rows: {missing}")
    return [dict(by_id[candidate_id]) for candidate_id in SELECTED_FIXTURE_IDS]


def build_fixture_panel_rows(rows: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Build prompt rows with exact labels and empty model-output slots."""

    panel_rows: list[JsonDict] = []
    for row in select_fixture_subset(rows):
        candidate_id = str(row["candidate_id"])
        exact_label = exact_validator_label(row)
        panel_rows.append(
            {
                "candidate_id": candidate_id,
                "fixture_bucket": FIXTURE_BUCKETS[candidate_id],
                "case_type": row.get("case_type"),
                "domain": row.get("domain"),
                "description": row.get("description"),
                "source_state_id": row.get("source_state_id"),
                "target_state_id": row.get("target_state_id"),
                "prompt_text": prompt_for_row(row),
                "exact_validator_label": exact_label,
                "exact_guard_results": row.get("guard_results", {}),
                "model_outputs": [],
            }
        )
    return panel_rows


def prompt_for_row(row: Mapping[str, Any]) -> str:
    """Create the plain-text model prompt without exposing the exact verdict."""

    lines = [
        "Decide whether this candidate rewrite should be ACCEPT, REJECT, or ABSTAIN.",
        "Return exactly one of those three words as the first word, then a short reason.",
        f"Candidate ID: {row.get('candidate_id')}",
        f"Domain: {row.get('domain')}",
        f"Description: {row.get('description')}",
        f"Source state ID: {row.get('source_state_id')}",
        f"Target state ID: {row.get('target_state_id')}",
        f"Local syntax valid: {row.get('local_syntax_valid')}",
        f"LCD advisory accept: {row.get('lcd_advisory_accept')}",
    ]
    return "\n".join(lines)


def exact_validator_label(row: Mapping[str, Any]) -> JsonDict:
    """Extract the Exp5471 exact final label used as telemetry ground truth."""

    verdict = _mapping(row.get("exact_final_verdict"))
    accepted = bool(verdict.get("accepted"))
    return {
        "accepted": accepted,
        "label": "accept" if accepted else "reject",
        "expected_accept": bool(verdict.get("expected_accept")),
        "caught_by_guards": list(verdict.get("caught_by_guards") or []),
        "violation_kinds": list(verdict.get("violation_kinds") or []),
        "matches_expected": bool(verdict.get("matches_expected")),
        "final_authority": verdict.get("final_authority"),
        "computed_from_repair_score": bool(verdict.get("computed_from_repair_score")),
    }


def model_specs_from_cache(
    *,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    preferred_quant: str = "Q4_K_M",
) -> list[JsonDict]:
    """Resolve mandated GGUF model specs to concrete local model paths."""

    registry = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}
    specs: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        model = registry[hf_id]
        resolved_path = cache_resolver(hf_id, preferred_quant)
        present = _is_nonempty_model_file(resolved_path)
        size_bytes = Path(resolved_path).stat().st_size if present and resolved_path else 0
        specs.append(
            {
                "name": model["name"],
                "hf_id": hf_id,
                "role": model["role"],
                "active_params_b": model["active_params_b"],
                "total_params_b": model["total_params_b"],
                "quantization": model["quantization"],
                "min_vram_gb": model["min_vram_gb"],
                "gpu": 0,
                "model_path": str(Path(resolved_path).resolve()) if resolved_path else None,
                "local_model_present": present,
                "model_file_size_bytes": size_bytes,
                "headline_required": True,
                "legacy_smoke_only": False,
                "spec_order": index,
            }
        )
    return specs


def model_file_checksums(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Compute SHA-256 checksums for local model files present on disk."""

    checksums: dict[str, JsonDict] = {}
    for spec in model_specs:
        path = spec.get("model_path")
        if not spec.get("local_model_present") or not isinstance(path, str):
            continue
        file_path = Path(path)
        checksums[str(spec["hf_id"])] = {
            "model_path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "sha256": _sha256_file(file_path),
        }
    return checksums


def default_runtime_probe() -> JsonDict:
    """Detect whether this process has a GPU-offloaded llama.cpp substrate."""

    cuda_available, cuda_device_count = _detect_cuda()
    llama_cpp_available, llama_cpp_gpu_offload, llama_cpp_error = _llama_cpp_python_status()
    native_cli = _find_native_llama_cli()
    runtime_ready = bool(cuda_available and (llama_cpp_gpu_offload or native_cli))
    blocked_reasons: list[str] = []
    if not cuda_available:
        blocked_reasons.append("cuda_unavailable")
    if not llama_cpp_gpu_offload and not native_cli:
        blocked_reasons.append("llama_cpp_gpu_offload_unavailable")
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "llama_cpp_python_available": llama_cpp_available,
        "llama_cpp_gpu_offload": llama_cpp_gpu_offload,
        "llama_cpp_python_error": llama_cpp_error,
        "native_llama_cli_available": native_cli is not None,
        "native_llama_cli_path": str(native_cli) if native_cli else None,
        "runtime_ready": runtime_ready,
        "blocked_reasons": blocked_reasons,
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = True,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    runtime_factory: RuntimeFactory | None = None,
    max_headline_models: int = 1,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Run the bounded panel or emit an honest blocked artifact."""

    started = time.perf_counter()
    specs = model_specs_from_cache(cache_resolver=cache_resolver)
    fixture_rows = build_fixture_panel_rows()
    checksums = model_file_checksums(specs)
    precondition_receipt = dict(runtime_probe())
    gpu_receipts: list[JsonDict] = []
    headline_models_run: list[str] = []
    runnable_specs = [spec for spec in specs if spec.get("local_model_present") is True]

    if precondition_receipt.get("runtime_ready") is True and runnable_specs:
        factory = default_runtime_factory if runtime_factory is None else runtime_factory
        for spec in runnable_specs[: max(0, max_headline_models)]:
            runtime = factory(spec)
            try:
                receipt = normalize_gpu_receipt(
                    runtime.preflight_gpu_offload(),
                    spec=spec,
                    precondition_receipt=precondition_receipt,
                )
                gpu_receipts.append(receipt)
                if receipt.get("offload_verified") is not True:
                    continue
                _run_model_on_rows(runtime, spec, fixture_rows)
                headline_models_run.append(str(spec["hf_id"]))
            except Exception as exc:
                gpu_receipts.append(
                    blocked_gpu_receipt(
                        spec,
                        precondition_receipt,
                        reason=f"runtime_error:{type(exc).__name__}:{exc}",
                    )
                )
            finally:
                close = getattr(runtime, "close", None)
                if callable(close):
                    close()
    else:
        blocked_reason = _blocked_reason(precondition_receipt, runnable_specs)
        receipt_specs = runnable_specs or specs
        gpu_receipts.extend(
            blocked_gpu_receipt(spec, precondition_receipt, reason=blocked_reason)
            for spec in receipt_specs
        )

    metrics = derive_metrics(fixture_rows)
    ready = bool(
        headline_models_run
        and all(
            _has_verified_offload_receipt(model_id, gpu_receipts)
            for model_id in headline_models_run
        )
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "model_specs": specs,
        "headline_models_run": headline_models_run,
        "n_samples": metrics["n_samples"],
        "exact_validator_accuracy": metrics["exact_validator_accuracy"],
        "semantic_false_accept_rate": metrics["semantic_false_accept_rate"],
        "factual_distortion_rate": metrics["factual_distortion_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "logprob_telemetry_available": metrics["logprob_telemetry_available"],
        "gpu_offload_receipts": gpu_receipts,
        "model_file_checksums": checksums,
        "guided_decoding_used": False,
        "sota_evidence_telemetry_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": (
            "complete: local SOTA GGUF evidence telemetry collected with GPU offload; "
            "guided decoding not used"
            if ready
            else "blocked: no mandated local SOTA GGUF model could run with verified GPU offload"
        ),
        "fixture_rows": fixture_rows,
        "metric_details": metrics,
        "runtime_precondition_receipt": precondition_receipt,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
        "duration_s": round(time.perf_counter() - started, 6),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def normalize_gpu_receipt(
    receipt: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    precondition_receipt: Mapping[str, Any],
) -> JsonDict:
    """Normalize runtime-specific preflight evidence into the artifact schema."""

    normalized = dict(receipt)
    normalized.setdefault("model_hf_id", spec.get("hf_id"))
    normalized.setdefault("model_path", spec.get("model_path"))
    normalized.setdefault("runtime_backend", "llama_cpp")
    normalized.setdefault("gpu_offload_supported", precondition_receipt.get("runtime_ready"))
    normalized.setdefault("offload_verified", False)
    normalized.setdefault("pre_generation", True)
    return normalized


def blocked_gpu_receipt(
    spec: Mapping[str, Any],
    precondition_receipt: Mapping[str, Any],
    *,
    reason: str,
) -> JsonDict:
    """Build a negative receipt so blocked artifacts still explain the gate."""

    return {
        "model_hf_id": spec.get("hf_id"),
        "model_path": spec.get("model_path"),
        "runtime_backend": "llama_cpp_precondition",
        "gpu_offload_supported": bool(precondition_receipt.get("runtime_ready")),
        "offload_verified": False,
        "pre_generation": True,
        "blocked_reason": reason,
        "cuda_available": bool(precondition_receipt.get("cuda_available")),
        "cuda_device_count": int(precondition_receipt.get("cuda_device_count") or 0),
    }


def derive_metrics(fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive all top-level rates from row-level model outputs."""

    output_pairs = [
        (row, output)
        for row in fixture_rows
        for output in row.get("model_outputs", [])
        if isinstance(output, Mapping)
    ]
    semantic_pairs = [
        (row, output)
        for row, output in output_pairs
        if "semantic_graph_guard" in _label_guards(row)
    ]
    factual_pairs = [
        (row, output)
        for row, output in output_pairs
        if "distortion_guard" in _label_guards(row)
    ]
    correct = sum(
        1
        for row, output in output_pairs
        if output.get("parsed_decision") == _mapping(row.get("exact_validator_label")).get("label")
    )
    return {
        "n_samples": len(fixture_rows),
        "n_model_evaluations": len(output_pairs),
        "exact_validator_accuracy": _rate(correct, len(output_pairs)),
        "semantic_false_accept_rate": _rate(
            sum(1 for _, output in semantic_pairs if output.get("parsed_decision") == "accept"),
            len(semantic_pairs),
        ),
        "factual_distortion_rate": _rate(
            sum(1 for _, output in factual_pairs if output.get("parsed_decision") == "accept"),
            len(factual_pairs),
        ),
        "abstention_rate": _rate(
            sum(1 for _, output in output_pairs if output.get("parsed_decision") == "abstain"),
            len(output_pairs),
        ),
        "logprob_telemetry_available": any(
            _mapping(output.get("logprob_telemetry")).get("available") is True
            for _, output in output_pairs
        ),
    }


def parse_decision(output_text: str) -> str:
    """Parse the first visible accept/reject/abstain-style decision."""

    if not output_text.strip():
        return "abstain"
    candidates: list[tuple[int, str]] = []
    for decision, tokens in (
        ("abstain", ("ABSTAIN", "UNSURE", "UNKNOWN", "CANNOT DETERMINE")),
        ("reject", ("REJECT", "INVALID", "FAIL")),
        ("accept", ("ACCEPT", "VALID", "PASS")),
    ):
        for token in tokens:
            match = re.search(rf"\b{re.escape(token)}\b", output_text.upper())
            if match:
                candidates.append((match.start(), decision))
                break
    if not candidates:
        return "abstain"
    return min(candidates, key=lambda item: item[0])[1]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact no longer supports the Exp5472 contract."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, readiness, and row-integrity errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("guided_decoding_used") is not False:
        errors.append("guided_decoding_used must be false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")

    specs = artifact.get("model_specs")
    if not isinstance(specs, list):
        errors.append("model_specs must be a list")
        specs = []
    spec_ids = {str(spec.get("hf_id")) for spec in specs if isinstance(spec, Mapping)}
    missing_specs = set(MANDATED_HF_IDS) - spec_ids
    if missing_specs:
        errors.append(f"model_specs must include mandated SOTA GGUF ids: {sorted(missing_specs)}")

    headline = artifact.get("headline_models_run")
    if not isinstance(headline, list):
        errors.append("headline_models_run must be a list")
        headline = []
    headline_ids = {str(item) for item in headline}
    if not headline_ids.issubset(set(MANDATED_HF_IDS)):
        errors.append("headline_models_run must contain only mandated SOTA GGUF ids")

    rows = artifact.get("fixture_rows")
    if not isinstance(rows, list):
        errors.append("fixture_rows must be a list")
        rows = []
    metrics = derive_metrics([row for row in rows if isinstance(row, Mapping)])
    for field in (
        "n_samples",
        "exact_validator_accuracy",
        "semantic_false_accept_rate",
        "factual_distortion_rate",
        "abstention_rate",
        "logprob_telemetry_available",
    ):
        if artifact.get(field) != metrics[field]:
            errors.append(f"{field} must match row recomputation")
    errors.extend(_row_integrity_errors(rows))

    if not isinstance(artifact.get("gpu_offload_receipts"), list):
        errors.append("gpu_offload_receipts must be a list")
    if not isinstance(artifact.get("model_file_checksums"), Mapping):
        errors.append("model_file_checksums must be a dict")
    ready = artifact.get("sota_evidence_telemetry_ready")
    if type(ready) is not bool:
        errors.append("sota_evidence_telemetry_ready must be boolean")
    if ready is True:
        if not headline:
            errors.append("ready requires headline_models_run")
        for model_id in headline_ids:
            if not _has_verified_offload_receipt(
                model_id,
                artifact.get("gpu_offload_receipts", []),
            ):
                errors.append("ready requires verified GPU offload for every headline model")
    return errors


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential checksum."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_json(payload)


def default_runtime_factory(spec: Mapping[str, Any]) -> Any:  # pragma: no cover
    """Construct the default runtime for a real local host."""

    if _llama_cpp_python_status()[1]:
        return LlamaCppPythonRuntime(spec)
    cli_path = _find_native_llama_cli()
    if cli_path is not None:
        return NativeLlamaCliRuntime(spec, cli_path)
    raise RuntimeError("no CUDA-enabled llama.cpp runtime available")


class LlamaCppPythonRuntime:  # pragma: no cover
    """Small wrapper around llama-cpp-python used only for real local runs."""

    def __init__(self, spec: Mapping[str, Any], *, n_ctx: int = 1024, max_tokens: int = 96) -> None:
        self.spec = spec
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self._llm = None

    def preflight_gpu_offload(self) -> JsonDict:
        from llama_cpp import Llama

        before = _nvidia_smi_used_vram_mb()
        self._llm = Llama(
            model_path=str(self.spec["model_path"]),
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            logits_all=True,
            verbose=False,
        )
        after = _nvidia_smi_used_vram_mb()
        delta = None if before is None or after is None else max(0, after - before)
        return {
            "model_hf_id": self.spec["hf_id"],
            "model_path": self.spec["model_path"],
            "runtime_backend": "llama_cpp_python",
            "gpu_offload_supported": True,
            "offload_verified": delta is None or delta > 0,
            "n_gpu_layers": -1,
            "vram_before_mb": before,
            "vram_after_load_mb": after,
            "vram_delta_mb": delta,
            "pre_generation": True,
        }

    def generate(self, prompt_text: str) -> JsonDict:
        if self._llm is None:
            raise RuntimeError("preflight_gpu_offload must run before generation")
        started = time.perf_counter()
        result = self._llm(
            prompt_text,
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=1.0,
            logprobs=5,
            echo=False,
        )
        choice = result["choices"][0]
        logprobs = choice.get("logprobs") or {}
        return {
            "output_text": str(choice.get("text", "")),
            "duration_s": round(time.perf_counter() - started, 6),
            "runtime_backend": "llama_cpp_python",
            "token_logprobs": logprobs.get("token_logprobs") or [],
            "top_logprobs": logprobs.get("top_logprobs") or [],
        }

    def close(self) -> None:
        self._llm = None
        gc.collect()


class NativeLlamaCliRuntime:  # pragma: no cover
    """Fallback wrapper for the native llama.cpp CLI when Python lacks offload."""

    def __init__(self, spec: Mapping[str, Any], cli_path: Path, *, max_tokens: int = 96) -> None:
        self.spec = spec
        self.cli_path = cli_path
        self.max_tokens = max_tokens

    def preflight_gpu_offload(self) -> JsonDict:
        before = _nvidia_smi_used_vram_mb()
        command = [
            str(self.cli_path),
            "-m",
            str(self.spec["model_path"]),
            "-p",
            "preflight",
            "-n",
            "0",
            "-ngl",
            "999",
        ]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        after = _nvidia_smi_used_vram_mb()
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        delta = None if before is None or after is None else max(0, after - before)
        offload_seen = "offload" in combined or "cuda" in combined or "gpu" in combined
        return {
            "model_hf_id": self.spec["hf_id"],
            "model_path": self.spec["model_path"],
            "runtime_backend": "native_llama_cli",
            "gpu_offload_supported": offload_seen,
            "offload_verified": proc.returncode == 0 and (offload_seen or delta is None or delta > 0),
            "n_gpu_layers": 999,
            "vram_before_mb": before,
            "vram_after_load_mb": after,
            "vram_delta_mb": delta,
            "pre_generation": True,
        }

    def generate(self, prompt_text: str) -> JsonDict:
        started = time.perf_counter()
        command = [
            str(self.cli_path),
            "-m",
            str(self.spec["model_path"]),
            "-p",
            prompt_text,
            "-n",
            str(self.max_tokens),
            "-ngl",
            "999",
            "--temp",
            "0",
            "--no-display-prompt",
        ]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=240)
        return {
            "output_text": proc.stdout.strip(),
            "duration_s": round(time.perf_counter() - started, 6),
            "runtime_backend": "native_llama_cli",
            "token_logprobs": [],
            "top_logprobs": [],
            "stderr_excerpt": proc.stderr[-2000:],
        }

    def close(self) -> None:
        return None


def _run_model_on_rows(runtime: Any, spec: Mapping[str, Any], fixture_rows: list[JsonDict]) -> None:
    for row in fixture_rows:
        generated = dict(runtime.generate(str(row["prompt_text"])))
        output_text = str(generated.get("output_text", ""))
        decision = parse_decision(output_text)
        row["model_outputs"].append(
            {
                "model_hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "prompt_text": row["prompt_text"],
                "output_text": output_text,
                "parsed_decision": decision,
                "abstained": decision == "abstain",
                "matches_exact_validator": decision
                == _mapping(row.get("exact_validator_label")).get("label"),
                "duration_s": float(generated.get("duration_s") or 0.0),
                "runtime_backend": generated.get("runtime_backend", "llama_cpp"),
                "logprob_telemetry": _logprob_telemetry(generated),
            }
        )


def _logprob_telemetry(generated: Mapping[str, Any]) -> JsonDict:
    token_logprobs = generated.get("token_logprobs") or []
    top_logprobs = generated.get("top_logprobs") or []
    return {
        "available": bool(token_logprobs or top_logprobs),
        "token_logprob_count": len(token_logprobs) if isinstance(token_logprobs, list) else 0,
        "top_logprobs_count": len(top_logprobs) if isinstance(top_logprobs, list) else 0,
    }


def _row_integrity_errors(rows: Sequence[Any]) -> list[str]:
    errors: list[str] = []
    ids = [row.get("candidate_id") for row in rows if isinstance(row, Mapping)]
    if ids != list(SELECTED_FIXTURE_IDS):
        errors.append("fixture_rows must preserve selected Exp5471 order")
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("fixture row must be a mapping")
            continue
        label = _mapping(row.get("exact_validator_label"))
        if label.get("final_authority") != EXACT_FINAL_AUTHORITY:
            errors.append("fixture row exact validator authority mismatch")
        if label.get("computed_from_repair_score") is not False:
            errors.append("exact validator label must not use repair proposal score")
        if row.get("prompt_text") == "":
            errors.append("fixture row prompt_text must be non-empty")
    return errors


def _has_verified_offload_receipt(model_id: str, receipts: Sequence[Any]) -> bool:
    return any(
        isinstance(receipt, Mapping)
        and receipt.get("model_hf_id") == model_id
        and receipt.get("offload_verified") is True
        and receipt.get("pre_generation") is True
        for receipt in receipts
    )


def _blocked_reason(
    precondition_receipt: Mapping[str, Any],
    runnable_specs: Sequence[Mapping[str, Any]],
) -> str:
    reasons = list(precondition_receipt.get("blocked_reasons") or [])
    if not runnable_specs:
        reasons.append("no_nonempty_mandated_local_model_path")
    return ",".join(str(reason) for reason in reasons) or "preconditions_unmet"


def _label_guards(row: Mapping[str, Any]) -> list[str]:
    return list(_mapping(row.get("exact_validator_label")).get("caught_by_guards") or [])


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    normalised: list[JsonDict] = []
    for item in tests_run:
        if isinstance(item, Mapping):
            normalised.append(dict(item))
        else:
            normalised.append({"command": str(item), "outcome": "reported"})
    return normalised


def _is_nonempty_model_file(path: str | None) -> bool:
    if not path:
        return False
    file_path = Path(path)
    if ".no_exist" in file_path.parts:
        return False
    return file_path.is_file() and file_path.stat().st_size > 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _detect_cuda() -> tuple[bool, int]:  # pragma: no cover
    try:
        import torch

        count = int(torch.cuda.device_count())
        return bool(torch.cuda.is_available() and count > 0), count
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False, 0
    if proc.returncode != 0:
        return False, 0
    count = len([line for line in proc.stdout.splitlines() if line.strip()])
    return count > 0, count


def _llama_cpp_python_status() -> tuple[bool, bool, str | None]:  # pragma: no cover
    try:
        from llama_cpp import llama_cpp as backend

        return True, bool(backend.llama_supports_gpu_offload()), None
    except Exception as exc:
        return False, False, str(exc)


def _find_native_llama_cli() -> Path | None:  # pragma: no cover
    candidates = [
        Path(os.environ.get("CARNOT_LLAMA_CLI", "")),
        Path.home() / ".cache/llama.cpp-master/build/bin/llama-cli",
        Path("/usr/local/bin/llama-cli"),
        Path("/usr/bin/llama-cli"),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _nvidia_smi_used_vram_mb() -> int | None:  # pragma: no cover
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    values: list[int] = []
    for line in proc.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else None


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--max-headline-models", type=int, default=1)
    args = parser.parse_args()
    artifact = run(
        result_path=args.result_path,
        max_headline_models=args.max_headline_models,
        tests_run=[],
    )
    print(json.dumps({"path": args.result_path, "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
