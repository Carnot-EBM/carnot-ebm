"""Exp5606 clean local-SOTA solve-versus-verify evidence panel.

Spec refs: REQ-VERIFY-5606, SCENARIO-VERIFY-5606.

This module is the evidence-preserving remeasurement after Exp5605.  It keeps
the expensive live llama.cpp path behind explicit preflight gates and stores
every prompt/response in a lossless Exp5605-style envelope before any aggregate
can be promoted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as corpus5566
from carnot import experiment_5567_local_sota_solve_verify_asymmetry as exp5567
from carnot import experiment_5605_raw_response_evidence_envelope as exp5605
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5606_clean_sota_solve_verify_evidence_panel.json")
EVIDENCE_ENVELOPE_RELATIVE_PATH = Path(
    "results/experiment_5606_clean_sota_solve_verify_evidence_panel.responses.jsonl"
)
CORPUS_RELATIVE_PATH = corpus5566.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5606.clean_sota_solve_verify_evidence_panel.v506"
EXPERIMENT = 5606
EXPERIMENT_ID = "exp5606-clean-sota-solve-verify-evidence-panel"
MILESTONE = "2026.07.506"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5606
MIN_INDEPENDENT_INSTANCES = 30
BOOTSTRAP_ITERATIONS = 256
PARSER_FAILURE_CEILING = 0.05
TRUNCATION_CEILING = 0.05
EFFECT_CLAIM_FLOOR = 0.01
INFERENCE_SUBSTRATE = "local_gguf_llamacpp_cuda"
ENVELOPE_SCHEMA_VERSION = exp5605.ENVELOPE_SCHEMA_VERSION
PARSER_NAME = "carnot.exp5606.batch_json_parser"
PARSER_VERSION = SCHEMA
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_HEADLINE_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
ARMS = ("discrete_verdict", "criteria_decomposition", "repeated_verdict_3x")
SPEC_REFS = ("REQ-VERIFY-5606", "SCENARIO-VERIFY-5606", "REQ-VERIFY-5605", "REQ-VERIFY-5566")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "model_specs",
    "gpu_offload_authenticated",
    "instances_evaluated_by_model",
    "evidence_envelope_path",
    "raw_response_replay_passed",
    "per_model_parser_failure_rate",
    "maximum_parser_failure_rate",
    "per_model_truncation_rate",
    "solve_accuracy_by_model",
    "verify_accuracy_by_model_and_arm",
    "exact_oracle_agreement",
    "paired_effects_and_intervals",
    "solve_verify_asymmetry_supported",
    "panel_complete",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Every headline and gate field names the evidence boundary it protects.",
    "model_specs": "Headline identity and cache path are auditable for all three mandated GGUF families.",
    "gpu_offload_authenticated": "Local CUDA evidence is real and CPU fallback cannot unlock headline rows.",
    "instances_evaluated_by_model": "Denominators stay separate for each model and arm.",
    "evidence_envelope_path": "Every aggregate traces to raw prompt and response rows.",
    "raw_response_replay_passed": "Exact outputs remain recoverable from the ledger.",
    "per_model_parser_failure_rate": "One family cannot mask another.",
    "maximum_parser_failure_rate": "Downstream gates use the worst family.",
    "per_model_truncation_rate": "Output-budget failures stay visible.",
    "solve_accuracy_by_model": "Direct generation is separate from verification.",
    "verify_accuracy_by_model_and_arm": "Verification modes remain disaggregated.",
    "exact_oracle_agreement": "Exact validators are the authority.",
    "paired_effects_and_intervals": "Uncertainty bounds claims.",
    "solve_verify_asymmetry_supported": "Headline support requires clean paired evidence.",
    "panel_complete": "Exact extension needs a valid residual ledger.",
    "inference_substrate": "Provenance is explicit.",
    "honest_verdict": "Repeat collapse is terminal retirement evidence.",
}

encode_lossless_payload = exp5605.encode_lossless_payload
decode_lossless_payload = exp5605.decode_lossless_payload
encode_prompt = exp5605.encode_prompt
decode_prompt = exp5605.decode_prompt


class EnvelopeReplayError(ValueError):
    """Raised when a stored Exp5606 response ledger row cannot replay exactly."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 hex digest for byte evidence."""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 hex digest for text evidence."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Return a SHA-256 hex digest for JSON-compatible content."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local GGUF file in chunks so the model identity is auditable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_family(hf_id: str) -> str:
    """Return the headline family label used for per-model parser gates."""

    if hf_id == QWEN_ID:
        return "qwen3.6-35b-a3b"
    if hf_id == GEMMA31_ID:
        return "gemma-4-31b-it"
    if hf_id == GEMMA26_ID:
        return "gemma-4-26b-a4b-it"
    return exp5567.model_family(hf_id)


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize mandated model specs and attach cache path plus file hash evidence."""

    registry = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
    by_id = {str(row.get("hf_id")): row for row in model_specs}
    normalized: list[JsonDict] = []
    for hf_id in MANDATED_HEADLINE_IDS:
        row = by_id.get(hf_id)
        if row is None:
            continue
        registry_row = registry.get(hf_id, {})
        path = str(row.get("model_path", "") or "")
        path_obj = Path(path).expanduser() if path else Path()
        present = bool(path and path_obj.is_file())
        normalized.append(
            {
                "name": str(
                    row.get("name") or registry_row.get("name") or hf_id.rsplit("/", 1)[-1]
                ),
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": str(row.get("role") or registry_row.get("role") or ""),
                "gpu": int(row.get("gpu", len(normalized)) or 0),
                "model_path": path,
                "cache_path": path,
                "local_path_hash": exp5605.local_path_hash(path) if path else "",
                "model_sha256": sha256_file(path_obj) if present else "",
                "local_model_present": present,
                "headline_eligible": row.get("headline_eligible") is not False,
                "active_params_b": row.get("active_params_b", registry_row.get("active_params_b")),
                "total_params_b": row.get("total_params_b", registry_row.get("total_params_b")),
                "quantization": str(
                    row.get("quantization") or registry_row.get("quantization") or ""
                ),
            }
        )
    return normalized


def resolve_all_headline_model_specs() -> list[JsonDict]:  # pragma: no cover
    """Resolve all three mandated local GGUF paths without downloading."""

    specs: list[JsonDict] = []
    registry = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
    for index, hf_id in enumerate(MANDATED_HEADLINE_IDS):
        registry_row = registry[hf_id]
        path = resolve_cached_gguf(hf_id, str(registry_row.get("quantization") or "Q4_K_M"))
        specs.append(
            {
                "name": registry_row["name"],
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": registry_row["role"],
                "gpu": index % 2,
                "model_path": path or "",
                "headline_eligible": True,
                "active_params_b": registry_row["active_params_b"],
                "total_params_b": registry_row["total_params_b"],
                "quantization": registry_row["quantization"],
            }
        )
    return normalize_model_specs(specs)


def sample_independent_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    n: int = MIN_INDEPENDENT_INSTANCES,
) -> list[JsonDict]:
    """Sample balanced Exp5566 valid/near-miss pairs using instance ID as unit."""

    return exp5567.sample_independent_pairs(rows, n=n)


def load_corpus_rows(repo_root: Path = REPO_ROOT) -> list[JsonDict]:  # pragma: no cover
    """Load the checked-in Exp5566 corpus rows without regenerating labels."""

    return exp5567.load_corpus_rows(repo_root)


def build_response_envelope_rows(
    *,
    raw_calls: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    device_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    """Convert raw model calls into an append-only Exp5605-style response ledger."""

    specs = {str(row.get("hf_id")): dict(row) for row in normalize_model_specs(model_specs)}
    rows: list[JsonDict] = []
    previous_hash = ""
    for sequence_index, call in enumerate(raw_calls):
        hf_id = str(call.get("model_hf_id") or "")
        spec = specs.get(hf_id, {})
        prompt = str(call.get("prompt") or "")
        raw = str(call.get("raw_response") or "").encode("utf-8")
        row: JsonDict = {
            "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
            "sequence_index": sequence_index,
            "call_id": str(call.get("call_id") or call.get("task_id") or sequence_index),
            "task_id": str(call.get("task_id") or ""),
            "phase": str(call.get("phase") or ""),
            "arm": str(call.get("arm") or ""),
            "model_family": str(spec.get("family") or model_family(hf_id)),
            "model_hf_id": hf_id,
            "model_path": str(spec.get("model_path", "")),
            "model_local_path_hash": str(spec.get("local_path_hash", "")),
            "model_file_sha256": str(spec.get("model_sha256", "")),
            "prompt_payload": encode_prompt(prompt),
            "prompt_hash": sha256_bytes(prompt.encode("utf-8")),
            "raw_response_payload": encode_lossless_payload(raw),
            "payload_hash": sha256_bytes(raw),
            "llama_cpp_version": str(
                call.get("llama_cpp_version") or _receipt_version(device_receipt, hf_id)
            ),
            "llama_cpp_arguments": dict(call.get("llama_cpp_arguments") or {}),
            "device_offload_receipt": _receipt_for_model(device_receipt, hf_id),
            "sampling_parameters": dict(call.get("sampling_parameters") or {}),
            "seed": int(call.get("seed", RANDOM_SEED) or RANDOM_SEED),
            "stop_reason": str(call.get("stop_reason") or ""),
            "token_counts": dict(call.get("token_counts") or {}),
            "truncation_flag": bool(call.get("truncation_flag") is True),
            "parser_name": PARSER_NAME,
            "parser_version": PARSER_VERSION,
            "parsed_object": call.get("parsed_object"),
            "exact_validator_outcome": dict(
                call.get("exact_validator_outcome")
                or {
                    "validator": "aggregate_exp5566_exact_oracle",
                    "accepted": None,
                    "note": "batched output parsed into aggregate records",
                }
            ),
            "timestamp_utc": str(call.get("timestamp_utc") or _utc_timestamp(sequence_index)),
            "previous_row_hash": previous_hash,
            "row_hash": "",
        }
        row["row_hash"] = row_hash(row)
        previous_hash = row["row_hash"]
        rows.append(row)
    return rows


def rechain_response_envelope_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return rows with previous-row and row hashes rebuilt after fixture edits."""

    rechained: list[JsonDict] = []
    previous = ""
    for row in rows:
        copy_row = dict(row)
        copy_row["previous_row_hash"] = previous
        copy_row["row_hash"] = ""
        copy_row["row_hash"] = row_hash(copy_row)
        previous = copy_row["row_hash"]
        rechained.append(copy_row)
    return rechained


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one envelope row while excluding its self-referential row hash."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def replay_response_envelope_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Verify prompt/response payload hashes and append-only row-chain hashes."""

    previous_hash = ""
    per_model_rows: Counter[str] = Counter()
    per_model_truncated: Counter[str] = Counter()
    for row in rows:
        if row.get("previous_row_hash") != previous_hash:
            raise EnvelopeReplayError("previous_row_hash")
        if row.get("row_hash") != row_hash(row):
            raise EnvelopeReplayError("row_hash")
        try:
            prompt = decode_prompt(_mapping(row.get("prompt_payload"), "prompt_payload"))
            raw = decode_lossless_payload(
                _mapping(row.get("raw_response_payload"), "raw_response_payload")
            )
        except Exception as exc:  # noqa: BLE001
            raise EnvelopeReplayError("payload_decode") from exc
        if row.get("prompt_hash") != sha256_bytes(prompt):
            raise EnvelopeReplayError("prompt_hash")
        if row.get("payload_hash") != sha256_bytes(raw):
            raise EnvelopeReplayError("payload_hash")
        model_id = str(row.get("model_hf_id") or "")
        per_model_rows[model_id] += 1
        if row.get("truncation_flag") is True:
            per_model_truncated[model_id] += 1
        previous_hash = str(row["row_hash"])
    truncation = {
        model_id: _rate(per_model_truncated[model_id], per_model_rows[model_id])
        for model_id in sorted(per_model_rows)
    }
    return {
        "row_count": len(rows),
        "raw_response_replay_passed": bool(rows),
        "per_model_truncation_rate": truncation,
    }


def write_response_envelope_rows(
    rows: Sequence[Mapping[str, Any]],
    path: Path | str,
) -> None:
    """Write one append-only response-envelope row per JSONL line."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def replay_response_envelope_path(path: Path | str) -> JsonDict:
    """Replay a JSONL response-envelope ledger from disk."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return replay_response_envelope_rows(rows)


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    device_receipt: Mapping[str, Any],
    sampled_pairs: Sequence[Mapping[str, Any]],
    panel_result: Mapping[str, Any] | None,
    evidence_rows: Sequence[Mapping[str, Any]],
    evidence_envelope_path: str = EVIDENCE_ENVELOPE_RELATIVE_PATH.as_posix(),
    tests_run: Sequence[Mapping[str, Any]] = (),
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    """Build a complete or blocked terminal Exp5606 artifact."""

    specs = normalize_model_specs(model_specs)
    model_ids = [str(row["hf_id"]) for row in specs]
    panel = dict(panel_result or {})
    solve_records = [dict(row) for row in panel.get("solve_records", [])]
    verifier_records = [dict(row) for row in panel.get("verifier_records", [])]
    rows = [dict(row) for row in evidence_rows]
    replay = _safe_replay(rows)
    truncation_rate = _per_model_rate_with_defaults(
        replay.get("per_model_truncation_rate", {}),
        model_ids,
    )
    parser_rate = _parser_failure_rate_by_model(solve_records, verifier_records, model_ids)
    maximum_parser_failure_rate = max(parser_rate.values(), default=0.0)
    maximum_truncation_rate = max(truncation_rate.values(), default=0.0)
    instances = _instances_evaluated_by_model(solve_records, verifier_records, model_ids)
    solve_accuracy = exp5567.compute_solve_accuracy(solve_records, model_ids) if model_ids else {}
    verify_accuracy = (
        exp5567.compute_verifier_metrics(verifier_records, model_ids, ARMS) if model_ids else {}
    )
    effects = _paired_effects(
        solve_records,
        verifier_records,
        solve_accuracy,
        verify_accuracy,
        model_ids,
        iterations=bootstrap_iterations,
    )
    model_ok = _model_specs_ready(specs)
    gpu_ok = gpu_offload_authenticated(device_receipt, specs)
    denominators_ok = _full_denominators(instances)
    replay_ok = replay.get("raw_response_replay_passed") is True
    parser_ok = maximum_parser_failure_rate <= PARSER_FAILURE_CEILING
    truncation_ok = maximum_truncation_rate <= TRUNCATION_CEILING
    clean_paired_evidence = bool(
        model_ok and gpu_ok and denominators_ok and replay_ok and parser_ok and truncation_ok
    )
    asymmetry_supported = _asymmetry_supported(effects) if clean_paired_evidence else False
    panel_complete = clean_paired_evidence
    blocked_reason = _blocked_reason(
        model_ok=model_ok,
        gpu_ok=gpu_ok,
        denominators_ok=denominators_ok,
        replay_ok=replay_ok,
        parser_ok=parser_ok,
        truncation_ok=truncation_ok,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "model_specs": specs,
        "MODEL_SPECS": specs,
        "model_cache_paths": {str(row["hf_id"]): str(row["model_path"]) for row in specs},
        "gpu_offload_authenticated": gpu_ok,
        "device_receipt": dict(device_receipt),
        "instances_evaluated_by_model": instances,
        "evidence_envelope_path": evidence_envelope_path,
        "response_envelope_rows_written": len(rows),
        "raw_response_replay_passed": replay_ok,
        "per_model_parser_failure_rate": parser_rate,
        "maximum_parser_failure_rate": maximum_parser_failure_rate,
        "per_model_truncation_rate": truncation_rate,
        "maximum_truncation_rate": maximum_truncation_rate,
        "solve_accuracy_by_model": solve_accuracy,
        "verify_accuracy_by_model_and_arm": verify_accuracy,
        "exact_oracle_agreement": _exact_oracle_agreement(solve_records, verifier_records),
        "paired_effects_and_intervals": effects,
        "family_heterogeneity": _family_heterogeneity(effects, model_ids),
        "solve_verify_asymmetry_supported": asymmetry_supported,
        "panel_complete": panel_complete,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(panel_complete, asymmetry_supported, blocked_reason),
        "arms": list(ARMS),
        "corpus_path": CORPUS_RELATIVE_PATH.as_posix(),
        "n_independent_instances_preregistered": MIN_INDEPENDENT_INSTANCES,
        "effect_claim_floor": EFFECT_CLAIM_FLOOR,
        "sub_percent_claims_suppressed": True,
        "fixed_seed_and_budget": {
            "seed": RANDOM_SEED,
            "arms": list(ARMS),
            "min_independent_instances_per_model": MIN_INDEPENDENT_INSTANCES,
        },
        "raw_response_hash": dict(panel.get("raw_response_hash", {})),
        "duration_s": round(float(panel.get("inference_duration_s", 0.0) or 0.0), 6),
        "legacy_smoke_models_used": [],
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail closed on unsupported promotion."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("legacy_smoke_models_used") == [], "legacy_smoke_models_used")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("sub_percent_claims_suppressed") is True, "sub_percent_claims_suppressed")
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )
    if artifact.get("panel_complete") is True:
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
        _require(_model_specs_ready(artifact.get("model_specs", [])), "model_specs")
        _require(artifact.get("gpu_offload_authenticated") is True, "gpu_offload_authenticated")
        _require(artifact.get("raw_response_replay_passed") is True, "raw_response_replay_passed")
        _require(
            float(artifact.get("maximum_parser_failure_rate", 1.0)) <= PARSER_FAILURE_CEILING,
            "maximum_parser_failure_rate",
        )
        _require(
            max((artifact.get("per_model_truncation_rate") or {"": 1.0}).values())
            <= TRUNCATION_CEILING,
            "per_model_truncation_rate",
        )
        _require(
            _full_denominators(artifact.get("instances_evaluated_by_model", {})), "panel_complete"
        )
    else:
        _require(str(artifact.get("honest_verdict", "")).startswith("blocked_"), "honest_verdict")
        _require(
            artifact.get("solve_verify_asymmetry_supported") is False,
            "solve_verify_asymmetry_supported",
        )


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    evidence_envelope_path: Path | str = REPO_ROOT / EVIDENCE_ENVELOPE_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    device_receipt: Mapping[str, Any] | None = None,
    sampled_pairs: Sequence[Mapping[str, Any]] | None = None,
    panel_result: Mapping[str, Any] | None = None,
    evidence_rows: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    """Run Exp5606 live, or write an injected test artifact and response ledger."""

    if (
        model_specs is None
        or device_receipt is None
        or sampled_pairs is None
        or panel_result is None
        or evidence_rows is None
    ):
        return _run_live(
            result_path=result_path,
            evidence_envelope_path=evidence_envelope_path,
            tests_run=tests_run,
            bootstrap_iterations=bootstrap_iterations,
        )  # pragma: no cover

    rows = [dict(row) for row in evidence_rows]
    write_response_envelope_rows(rows, evidence_envelope_path)
    artifact = build_artifact(
        model_specs=model_specs,
        device_receipt=device_receipt,
        sampled_pairs=sampled_pairs,
        panel_result=panel_result,
        evidence_rows=rows,
        evidence_envelope_path=str(Path(evidence_envelope_path)),
        tests_run=tests_run,
        bootstrap_iterations=bootstrap_iterations,
    )
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking the self-referential checksum field."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def gpu_offload_authenticated(
    device_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true only when every headline model has CUDA, offload, PID, and GPU-use evidence."""

    if device_receipt.get("gpu_offload_authenticated") is not True:
        return False
    receipts = {
        str(row.get("model_hf_id")): row
        for row in device_receipt.get("model_receipts", [])
        if isinstance(row, Mapping)
    }
    for spec in model_specs:
        receipt = receipts.get(str(spec.get("hf_id")))
        if receipt is None:
            return False
        if receipt.get("gpu_offload_authenticated") is not True:
            return False
        if receipt.get("worker_ok") is not True:
            return False
        if receipt.get("llama_cpp_supports_gpu_offload") is not True:
            return False
        if receipt.get("torch_cuda_available") is not True:
            return False
        if int(receipt.get("torch_device_count", 0) or 0) <= 0:
            return False
        if int(receipt.get("offloaded_layer_count_from_backend_log", 0) or 0) <= 0:
            return False
        if int(receipt.get("pid", 0) or 0) <= 0:
            return False
        if "port" not in receipt:
            return False
        memory = float(receipt.get("pid_gpu_memory_mb_peak", 0.0) or 0.0)
        util = float(receipt.get("gpu_utilization_pct_peak", 0.0) or 0.0)
        if memory <= 0.0 and util <= 0.0:
            return False
    return _model_specs_ready(model_specs)


def honest_verdict(panel_complete: bool, asymmetry_supported: bool, blocked_reason: str) -> str:
    """Return a terminal verdict without promoting sub-percent differences."""

    if panel_complete and asymmetry_supported:
        return "complete: clean authenticated solve-versus-verify asymmetry supported; no sub-percent claims"
    if panel_complete:
        return "complete: clean authenticated solve-versus-verify panel without supported asymmetry; no sub-percent claims"
    return blocked_reason or "blocked_no_live_panel"


def _safe_replay(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    try:
        return replay_response_envelope_rows(rows)
    except EnvelopeReplayError as exc:
        return {
            "row_count": len(rows),
            "raw_response_replay_passed": False,
            "per_model_truncation_rate": {},
            "replay_error": str(exc),
        }


def _model_specs_ready(model_specs: Any) -> bool:
    if not isinstance(model_specs, Sequence) or isinstance(model_specs, (str, bytes)):
        return False
    ids = [str(row.get("hf_id", "")) for row in model_specs if isinstance(row, Mapping)]
    if ids != list(MANDATED_HEADLINE_IDS):
        return False
    return all(
        isinstance(row, Mapping)
        and row.get("local_model_present") is True
        and str(row.get("model_path", "")).endswith(".gguf")
        and bool(row.get("model_sha256"))
        for row in model_specs
    )


def _instances_evaluated_by_model(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for model_id in model_ids:
        solve_units = {
            str(row.get("instance_id"))
            for row in solve_records
            if row.get("model_hf_id") == model_id
        }
        by_arm: dict[str, JsonDict] = {}
        for arm in ARMS:
            rows = [
                row
                for row in verifier_records
                if row.get("model_hf_id") == model_id and row.get("arm") == arm
            ]
            by_arm[arm] = {
                "instances": len({str(row.get("instance_id")) for row in rows}),
                "candidate_labels": len(rows),
            }
        out[model_id] = {
            "solve_instances": len(solve_units),
            "verify_by_arm": by_arm,
            "full_denominator": len(solve_units) >= MIN_INDEPENDENT_INSTANCES
            and all(
                row["instances"] >= MIN_INDEPENDENT_INSTANCES
                and row["candidate_labels"] >= 2 * MIN_INDEPENDENT_INSTANCES
                for row in by_arm.values()
            ),
        }
    return out


def _full_denominators(instances: Any) -> bool:
    if not isinstance(instances, Mapping):
        return False
    expected = set(MANDATED_HEADLINE_IDS)
    if set(instances) != expected:
        return False
    return all(
        isinstance(row, Mapping) and row.get("full_denominator") is True
        for row in instances.values()
    )


def _parser_failure_rate_by_model(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for model_id in model_ids:
        rows = [
            row for row in (*solve_records, *verifier_records) if row.get("model_hf_id") == model_id
        ]
        failures = sum(1 for row in rows if row.get("parser_ok") is not True)
        out[model_id] = _rate(failures, len(rows))
    return out


def _per_model_rate_with_defaults(
    raw: Mapping[str, Any], model_ids: Sequence[str]
) -> dict[str, float]:
    return {model_id: round(float(raw.get(model_id, 0.0) or 0.0), 6) for model_id in model_ids}


def _paired_effects(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
    solve_accuracy: Mapping[str, Mapping[str, Any]],
    verify_accuracy: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_ids: Sequence[str],
    *,
    iterations: int,
) -> dict[str, dict[str, JsonDict]]:
    if not solve_records or not verifier_records:
        return {
            model_id: {arm: _empty_effect(iterations) for arm in ARMS} for model_id in model_ids
        }
    confidence = exp5567.compute_confidence_intervals(
        solve_records,
        verifier_records,
        model_ids,
        ARMS,
        iterations=iterations,
        seed=RANDOM_SEED,
    )
    asymmetry = exp5567.compute_solve_verify_asymmetry(solve_accuracy, verify_accuracy)
    out: dict[str, dict[str, JsonDict]] = {}
    for model_id in model_ids:
        out[model_id] = {}
        for arm in ARMS:
            interval = dict(
                confidence.get(model_id, {}).get(f"asymmetry_{arm}", _empty_effect(iterations))
            )
            effect = float(
                asymmetry.get(model_id, {})
                .get(arm, {})
                .get("solve_minus_verify_balanced_accuracy", 0.0)
            )
            interval["effect"] = round(effect, 6)
            interval["solve_accuracy"] = float(
                solve_accuracy.get(model_id, {}).get("accuracy", 0.0)
            )
            interval["verify_balanced_accuracy"] = float(
                verify_accuracy.get(model_id, {}).get(arm, {}).get("balanced_accuracy", 0.0)
            )
            interval["non_sub_percent_claim_floor"] = EFFECT_CLAIM_FLOOR
            out[model_id][arm] = interval
    return out


def _empty_effect(iterations: int) -> JsonDict:
    return {
        "low": 0.0,
        "mid": 0.0,
        "high": 0.0,
        "n_bootstrap": iterations,
        "paired_unit": "instance_id",
        "effect": 0.0,
    }


def _asymmetry_supported(effects: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> bool:
    for by_arm in effects.values():
        for effect in by_arm.values():
            point = abs(float(effect.get("effect", 0.0) or 0.0))
            low = float(effect.get("low", 0.0) or 0.0)
            high = float(effect.get("high", 0.0) or 0.0)
            if point >= EFFECT_CLAIM_FLOOR and (high < 0.0 or low > 0.0):
                return True
    return False


def _family_heterogeneity(
    effects: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_ids: Sequence[str],
) -> JsonDict:
    ranges: JsonDict = {}
    for arm in ARMS:
        values = [
            float(effects.get(model_id, {}).get(arm, {}).get("effect", 0.0) or 0.0)
            for model_id in model_ids
        ]
        ranges[arm] = {
            "min_effect": round(min(values), 6) if values else 0.0,
            "max_effect": round(max(values), 6) if values else 0.0,
            "range": round(max(values) - min(values), 6) if values else 0.0,
        }
    return {"models": list(model_ids), "effect_range_by_arm": ranges}


def _exact_oracle_agreement(
    solve_records: Sequence[Mapping[str, Any]],
    verifier_records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    solve_scored = sum(1 for row in solve_records if "exact_accepted" in row)
    verifier_scored = sum(
        1
        for row in verifier_records
        if exp5567.normalize_label(row.get("true_label")) in {"valid", "invalid"}
    )
    return {
        "exact_validator_is_authority": True,
        "llm_judge_used": False,
        "validator_backend": "exp5566_exact_asp_fsm_validators",
        "solve_rows_scored_by_exact_oracle": solve_scored,
        "verify_candidate_labels_from_exact_oracle": verifier_scored,
        "oracle_label_coverage": _rate(verifier_scored, len(verifier_records)),
    }


def _blocked_reason(
    *,
    model_ok: bool,
    gpu_ok: bool,
    denominators_ok: bool,
    replay_ok: bool,
    parser_ok: bool,
    truncation_ok: bool,
) -> str:
    if not model_ok:
        return "blocked_missing_headline_gguf"
    if not gpu_ok:
        return "blocked_no_cuda_offload_authenticated_cpu_fallback_rejected"
    if not replay_ok:
        return "blocked_raw_response_replay_failed"
    if not denominators_ok:
        return "blocked_incomplete_panel_denominators"
    if not parser_ok or not truncation_ok:
        return "blocked_parser_or_truncation_ceiling_failed_terminal_retirement_evidence"
    return ""


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvelopeReplayError(field)
    return value


def _receipt_for_model(device_receipt: Mapping[str, Any], model_id: str) -> JsonDict:
    for row in device_receipt.get("model_receipts", []):
        if isinstance(row, Mapping) and row.get("model_hf_id") == model_id:
            return dict(row)
    return {"model_hf_id": model_id, "receipt_missing": True}


def _receipt_version(device_receipt: Mapping[str, Any], model_id: str) -> str:
    receipt = _receipt_for_model(device_receipt, model_id)
    return str(receipt.get("llama_cpp_version") or device_receipt.get("llama_cpp_version") or "")


def _utc_timestamp(sequence_index: int) -> str:
    return f"2026-07-14T00:00:{sequence_index:02d}Z"


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def _run_live(  # pragma: no cover
    *,
    result_path: Path | str,
    evidence_envelope_path: Path | str,
    tests_run: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int,
) -> JsonDict:
    started = time.perf_counter()
    specs = resolve_all_headline_model_specs()
    corpus_rows = load_corpus_rows(REPO_ROOT)
    pairs = sample_independent_pairs(corpus_rows)
    device = probe_cuda_device_receipt()
    model_ok = _model_specs_ready(specs)
    preflight_ok = bool(
        model_ok
        and len(pairs) >= MIN_INDEPENDENT_INSTANCES
        and device.get("cuda_preflight_ok") is True
    )
    if preflight_ok:
        panel, rows, device = run_live_local_sota_panel(
            model_specs=specs, pairs=pairs, device_receipt=device
        )
        panel["inference_duration_s"] = round(time.perf_counter() - started, 6)
    else:
        panel = {
            "solve_records": [],
            "verifier_records": [],
            "raw_response_hash": {},
            "inference_duration_s": 0.0,
        }
        rows = []
    write_response_envelope_rows(rows, evidence_envelope_path)
    artifact = build_artifact(
        model_specs=specs,
        device_receipt=device,
        sampled_pairs=pairs,
        panel_result=panel,
        evidence_rows=rows,
        evidence_envelope_path=str(Path(evidence_envelope_path)),
        tests_run=tests_run,
        bootstrap_iterations=bootstrap_iterations,
    )
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def probe_cuda_device_receipt() -> JsonDict:  # pragma: no cover
    receipt: JsonDict = {
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "devices": [],
        "llama_cpp_supports_gpu_offload": False,
        "gpu_offload_authenticated": False,
        "cuda_preflight_ok": False,
    }
    try:
        import torch  # noqa: PLC0415

        receipt["torch_cuda_available"] = bool(torch.cuda.is_available())
        receipt["torch_device_count"] = int(torch.cuda.device_count())
        receipt["devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # noqa: BLE001
        receipt["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import llama_cpp  # noqa: PLC0415
        from llama_cpp import llama_cpp as low  # noqa: PLC0415

        receipt["llama_cpp_version"] = str(getattr(llama_cpp, "__version__", "unknown"))
        receipt["llama_cpp_supports_gpu_offload"] = bool(low.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001
        receipt["llama_cpp_error"] = f"{type(exc).__name__}: {exc}"
    receipt["cuda_preflight_ok"] = bool(
        receipt["torch_cuda_available"]
        and int(receipt["torch_device_count"]) > 0
        and receipt["llama_cpp_supports_gpu_offload"]
    )
    return receipt


WORKER_CODE = r"""
import argparse
import json
import os
import subprocess
import threading
import time


def _extract_text(raw):
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first.get("text") or "")
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
    return ""


def _usage(raw, prompt, response):
    if isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
        return raw["usage"]
    return {
        "prompt_tokens": len(str(prompt).split()),
        "completion_tokens": len(str(response).split()),
        "total_tokens": len(str(prompt).split()) + len(str(response).split()),
        "source": "whitespace_estimate",
    }


def _query_pid_gpu_memory(pid):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
    except Exception:
        return 0.0
    peak = 0.0
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                peak = max(peak, float(parts[1]))
            except ValueError:
                pass
    return peak


def _query_gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
    except Exception:
        return 0.0
    values = []
    for line in out.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            pass
    return max(values) if values else 0.0


parser = argparse.ArgumentParser()
parser.add_argument("--workload", required=True)
args = parser.parse_args()
payload = json.loads(open(args.workload, "r", encoding="utf-8").read())
started = time.perf_counter()
pid = os.getpid()
stop_monitor = False
gpu_memory_peak = 0.0
gpu_util_peak = 0.0


def _monitor():
    global gpu_memory_peak, gpu_util_peak
    while not stop_monitor:
        gpu_memory_peak = max(gpu_memory_peak, _query_pid_gpu_memory(pid))
        gpu_util_peak = max(gpu_util_peak, _query_gpu_util())
        time.sleep(0.25)


monitor = threading.Thread(target=_monitor, daemon=True)
monitor.start()
llm = None
responses = []
try:
    import torch
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp import llama_cpp as low

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append({"index": index, "name": torch.cuda.get_device_name(index)})
    llm = Llama(
        model_path=payload["model_path"],
        n_ctx=int(payload.get("n_ctx", 8192)),
        n_batch=int(payload.get("n_batch", 256)),
        n_gpu_layers=int(payload.get("n_gpu_layers", -1)),
        seed=int(payload.get("seed", 5606)),
        verbose=True,
    )
    for task in payload["tasks"]:
        task_started = time.perf_counter()
        try:
            raw = llm(
                task["prompt"],
                max_tokens=int(task["max_tokens"]),
                temperature=float(task.get("temperature", 0.0)),
                top_p=1.0,
                repeat_penalty=1.0,
                seed=int(task.get("seed", payload.get("seed", 5606))),
            )
            text = _extract_text(raw)
            finish = ""
            if isinstance(raw, dict) and raw.get("choices"):
                finish = str(raw["choices"][0].get("finish_reason") or "")
            responses.append(
                {
                    "task_id": task["task_id"],
                    "phase": task.get("phase", ""),
                    "arm": task.get("arm", ""),
                    "ok": bool(text),
                    "text": text,
                    "duration_s": round(time.perf_counter() - task_started, 6),
                    "usage": _usage(raw, task["prompt"], text),
                    "finish_reason": finish,
                    "error": "",
                }
            )
        except Exception as exc:
            responses.append(
                {
                    "task_id": task["task_id"],
                    "phase": task.get("phase", ""),
                    "arm": task.get("arm", ""),
                    "ok": False,
                    "text": "",
                    "duration_s": round(time.perf_counter() - task_started, 6),
                    "usage": {},
                    "finish_reason": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    stop_monitor = True
    monitor.join(timeout=1)
    print(
        json.dumps(
            {
                "ok": True,
                "pid": pid,
                "port": None,
                "runtime_mode": "llama_cpp_python_in_process_no_http_port",
                "model_hf_id": payload["model_hf_id"],
                "model_path": payload["model_path"],
                "llama_cpp_version": getattr(llama_cpp, "__version__", None),
                "llama_cpp_supports_gpu_offload": bool(low.llama_supports_gpu_offload()),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_device_count": int(torch.cuda.device_count()),
                "devices": devices,
                "pid_gpu_memory_mb_peak": gpu_memory_peak,
                "gpu_utilization_pct_peak": gpu_util_peak,
                "load_and_inference_duration_s": round(time.perf_counter() - started, 6),
                "responses": responses,
            },
            sort_keys=True,
        )
    )
except Exception as exc:
    stop_monitor = True
    monitor.join(timeout=1)
    print(
        json.dumps(
            {
                "ok": False,
                "pid": pid,
                "port": None,
                "runtime_mode": "llama_cpp_python_in_process_no_http_port",
                "model_hf_id": payload.get("model_hf_id", ""),
                "model_path": payload.get("model_path", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "pid_gpu_memory_mb_peak": gpu_memory_peak,
                "gpu_utilization_pct_peak": gpu_util_peak,
                "load_and_inference_duration_s": round(time.perf_counter() - started, 6),
                "responses": responses,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1)
finally:
    close = getattr(llm, "close", None)
    if callable(close):
        close()
"""


def run_live_local_sota_panel(  # pragma: no cover
    *,
    model_specs: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    device_receipt: Mapping[str, Any],
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    all_solve: list[JsonDict] = []
    all_verify: list[JsonDict] = []
    raw_hashes: dict[str, str] = {}
    raw_calls: list[JsonDict] = []
    receipts: list[JsonDict] = []
    started = time.perf_counter()
    for spec in model_specs:
        workload = _workload_for_model(pairs)
        worker = _run_model_workload_subprocess(spec, workload)
        receipts.append(worker["model_receipt"])
        response_by_id = {
            str(row.get("task_id")): dict(row)
            for row in worker.get("responses", [])
            if isinstance(row, Mapping)
        }
        solve, verify, hashes, calls = _records_and_calls_from_worker(
            spec=spec,
            pairs=pairs,
            workload=workload,
            response_by_id=response_by_id,
            model_receipt=worker["model_receipt"],
        )
        all_solve.extend(solve)
        all_verify.extend(verify)
        raw_hashes.update(hashes)
        raw_calls.extend(calls)
        gc.collect()
    merged_device = dict(device_receipt)
    merged_device["model_receipts"] = receipts
    merged_device["gpu_offload_authenticated"] = bool(receipts) and all(
        receipt.get("gpu_offload_authenticated") is True for receipt in receipts
    )
    rows = build_response_envelope_rows(
        raw_calls=raw_calls,
        model_specs=model_specs,
        device_receipt=merged_device,
    )
    return (
        {
            "solve_records": all_solve,
            "verifier_records": all_verify,
            "raw_response_hash": raw_hashes,
            "inference_duration_s": round(time.perf_counter() - started, 6),
        },
        rows,
        merged_device,
    )


def _workload_for_model(pairs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:  # pragma: no cover
    tasks: list[JsonDict] = []
    for batch_index, batch in enumerate(_chunk_pairs(pairs, 10)):
        batch_id = f"batch{batch_index:02d}"
        tasks.append(
            {
                "task_id": f"solve_batch::{batch_id}",
                "phase": "solve",
                "arm": "",
                "prompt": exp5567.build_solve_batch_prompt(batch),
                "max_tokens": 2048,
                "temperature": 0.0,
                "seed": RANDOM_SEED + batch_index,
            }
        )
        for arm in ARMS:
            repeats = 3 if arm == "repeated_verdict_3x" else 1
            for repeat in range(repeats):
                tasks.append(
                    {
                        "task_id": f"verify_batch::{batch_id}::{arm}::{repeat}",
                        "phase": "verify",
                        "arm": arm,
                        "prompt": exp5567.build_verifier_batch_prompt(
                            batch, arm=arm, repeat=repeat
                        ),
                        "max_tokens": 1536 if arm == "criteria_decomposition" else 1024,
                        "temperature": 0.2 if arm == "repeated_verdict_3x" else 0.0,
                        "seed": RANDOM_SEED + batch_index * 100 + repeat,
                    }
                )
    return tasks


def _run_model_workload_subprocess(  # pragma: no cover
    spec: Mapping[str, Any],
    workload: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload = {
        "model_hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "seed": RANDOM_SEED,
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "n_batch": 256,
        "tasks": list(workload),
    }
    timeout_s = int(os.environ.get("CARNOT_5606_WORKER_TIMEOUT_S", "7200"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        workload_path = handle.name
    env = dict(os.environ)
    if "gpu" in spec:
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    command = [selected_python(), "-c", WORKER_CODE, "--workload", workload_path]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        payload_out: JsonDict = {
            "ok": False,
            "pid": 0,
            "port": None,
            "runtime_mode": "llama_cpp_python_in_process_no_http_port",
            "model_hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "error": f"TimeoutExpired:{timeout_s}",
            "responses": [],
        }
        stderr = str(exc.stderr or "")
        return {
            "responses": [],
            "model_receipt": _model_receipt_from_worker(spec, payload_out, stderr, None),
        }
    finally:
        Path(workload_path).unlink(missing_ok=True)
    payload_out = _first_json_line(completed.stdout)
    receipt = _model_receipt_from_worker(
        spec,
        payload_out,
        completed.stderr,
        completed.returncode,
    )
    responses = (
        payload_out.get("responses", []) if isinstance(payload_out.get("responses"), list) else []
    )
    return {"responses": responses, "model_receipt": receipt}


def _model_receipt_from_worker(  # pragma: no cover
    spec: Mapping[str, Any],
    payload_out: Mapping[str, Any],
    stderr: str,
    returncode: int | None,
) -> JsonDict:
    backend_text = stderr + "\n" + str(payload_out.get("backend_log_tail", ""))
    receipt: JsonDict = {
        "model_hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "model_sha256": spec.get("model_sha256", ""),
        "returncode": returncode,
        "pid": int(payload_out.get("pid", 0) or 0),
        "port": payload_out.get("port"),
        "runtime_mode": str(
            payload_out.get("runtime_mode") or "llama_cpp_python_in_process_no_http_port"
        ),
        "worker_ok": returncode == 0 and payload_out.get("ok") is True,
        "stderr_tail": _tail(backend_text),
        "offloaded_layer_count_from_backend_log": _parse_offloaded_layers(backend_text) or 0,
        "llama_cpp_version": str(payload_out.get("llama_cpp_version") or ""),
        "llama_cpp_supports_gpu_offload": payload_out.get("llama_cpp_supports_gpu_offload") is True,
        "torch_cuda_available": payload_out.get("torch_cuda_available") is True,
        "torch_device_count": int(payload_out.get("torch_device_count", 0) or 0),
        "devices": payload_out.get("devices", []),
        "pid_gpu_memory_mb_peak": float(payload_out.get("pid_gpu_memory_mb_peak", 0.0) or 0.0),
        "gpu_utilization_pct_peak": float(payload_out.get("gpu_utilization_pct_peak", 0.0) or 0.0),
        "duration_s": float(payload_out.get("load_and_inference_duration_s", 0.0) or 0.0),
    }
    receipt["gpu_offload_authenticated"] = bool(
        receipt["worker_ok"]
        and receipt["llama_cpp_supports_gpu_offload"]
        and receipt["torch_cuda_available"]
        and receipt["torch_device_count"] > 0
        and receipt["offloaded_layer_count_from_backend_log"] > 0
        and receipt["pid"] > 0
        and (receipt["pid_gpu_memory_mb_peak"] > 0 or receipt["gpu_utilization_pct_peak"] > 0)
    )
    return receipt


def _records_and_calls_from_worker(  # pragma: no cover
    *,
    spec: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    workload: Sequence[Mapping[str, Any]],
    response_by_id: Mapping[str, Mapping[str, Any]],
    model_receipt: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], dict[str, str], list[JsonDict]]:
    hf_id = str(spec["hf_id"])
    solve: list[JsonDict] = []
    verify: list[JsonDict] = []
    hashes: dict[str, str] = {}
    batches = _chunk_pairs(pairs, 10)
    for batch_index, batch in enumerate(batches):
        batch_id = f"batch{batch_index:02d}"
        solve_task = f"solve_batch::{batch_id}"
        solve_response = response_by_id.get(solve_task, {})
        solve_text = str(solve_response.get("text", ""))
        solve_hash = exp5567.sha256_text(solve_text)
        hashes[f"{hf_id}:{solve_task}"] = solve_hash
        solve_payload, solve_error = exp5567.extract_json_object(solve_text)
        solves_by_instance = exp5567._items_by_key(solve_payload, "solves", "instance_id")
        for pair in batch:
            instance_id = str(pair["instance_id"])
            item = solves_by_instance.get(instance_id)
            item_text = json.dumps(item, sort_keys=True) if item else ""
            solve_score = (
                exp5567.parse_and_score_solve_response(item_text, pair)
                if item
                else {
                    "parser_ok": False,
                    "exact_accepted": False,
                    "response_hash": solve_hash,
                    "error_type": "solve_batch_missing_item"
                    if solve_payload is not None
                    else f"solve_{solve_error}",
                }
            )
            solve.append(
                {
                    "model_hf_id": hf_id,
                    "instance_id": instance_id,
                    "family": pair["family"],
                    "parser_ok": solve_score["parser_ok"],
                    "exact_accepted": solve_score["exact_accepted"],
                    "latency_s": exp5567._apportioned_float(
                        solve_response, "duration_s", len(batch)
                    ),
                    "prompt_tokens": exp5567._apportioned_usage(
                        solve_response, "prompt_tokens", len(batch)
                    ),
                    "completion_tokens": exp5567._apportioned_usage(
                        solve_response, "completion_tokens", len(batch)
                    ),
                    "response_hash": str(solve_score.get("response_hash", solve_hash)),
                    "error_type": solve_score.get("error_type", ""),
                }
            )
        for pair in batch:
            instance_id = str(pair["instance_id"])
            for arm in ARMS:
                repeats = 3 if arm == "repeated_verdict_3x" else 1
                repeat_maps: list[dict[str, Mapping[str, Any]]] = []
                repeat_errors: list[str] = []
                repeat_hashes: list[str] = []
                latency = 0.0
                prompt_tokens = 0
                completion_tokens = 0
                for repeat in range(repeats):
                    task_id = f"verify_batch::{batch_id}::{arm}::{repeat}"
                    response = response_by_id.get(task_id, {})
                    text = str(response.get("text", ""))
                    response_hash = exp5567.sha256_text(text)
                    hashes[f"{hf_id}:{task_id}"] = response_hash
                    payload, error = exp5567.extract_json_object(text)
                    repeat_maps.append(exp5567._items_by_key(payload, "labels", "candidate_id"))
                    repeat_errors.append(f"verifier_{error}" if error else "")
                    repeat_hashes.append(response_hash)
                    denominator = max(1, len(batch) * 2)
                    latency += exp5567._apportioned_float(response, "duration_s", denominator)
                    prompt_tokens += exp5567._apportioned_usage(
                        response, "prompt_tokens", denominator
                    )
                    completion_tokens += exp5567._apportioned_usage(
                        response, "completion_tokens", denominator
                    )
                for candidate_key in ("valid_row", "invalid_row"):
                    row = dict(pair[candidate_key])
                    labels: list[str | None] = []
                    errors: list[str] = []
                    response_hashes: list[str] = []
                    for repeat_index, label_map in enumerate(repeat_maps):
                        item = label_map.get(str(row["row_id"]))
                        if item is None:
                            labels.append(None)
                            errors.append(
                                repeat_errors[repeat_index] or "verifier_batch_missing_item"
                            )
                            response_hashes.append(repeat_hashes[repeat_index])
                            continue
                        item_text = json.dumps(item, sort_keys=True)
                        label, error = exp5567.parse_verifier_response(item_text, arm)
                        labels.append(label)
                        errors.append(error)
                        response_hashes.append(exp5567.sha256_text(item_text))
                    predicted = (
                        exp5567._majority_label(labels)
                        if arm == "repeated_verdict_3x"
                        else labels[0]
                    )
                    parser_ok = predicted is not None
                    verify.append(
                        {
                            "model_hf_id": hf_id,
                            "instance_id": instance_id,
                            "candidate_id": row["row_id"],
                            "family": pair["family"],
                            "arm": arm,
                            "true_label": row["label"],
                            "predicted_label": predicted,
                            "parser_ok": parser_ok,
                            "latency_s": latency,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "response_hashes": response_hashes,
                            "repeat_labels": [label for label in labels if label is not None],
                            "error_type": ""
                            if parser_ok
                            else next((err for err in errors if err), "verifier_missing_label"),
                        }
                    )
    calls: list[JsonDict] = []
    workload_by_id = {str(row["task_id"]): row for row in workload}
    for task_id, task in workload_by_id.items():
        response = response_by_id.get(task_id, {})
        text = str(response.get("text", ""))
        finish_reason = str(response.get("finish_reason") or "")
        calls.append(
            {
                "call_id": f"{spec['hf_id']}:{task_id}",
                "task_id": task_id,
                "phase": task.get("phase", ""),
                "arm": task.get("arm", ""),
                "model_hf_id": spec["hf_id"],
                "prompt": task["prompt"],
                "raw_response": text,
                "seed": task.get("seed", RANDOM_SEED),
                "stop_reason": finish_reason
                or ("error" if response.get("error") else "stop_sequence"),
                "truncation_flag": finish_reason in {"length", "max_tokens"},
                "sampling_parameters": {
                    "temperature": task.get("temperature", 0.0),
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,
                },
                "llama_cpp_arguments": {"n_gpu_layers": -1, "n_ctx": 8192, "n_batch": 256},
                "llama_cpp_version": model_receipt.get("llama_cpp_version", ""),
                "token_counts": response.get("usage", {}),
            }
        )
    return solve, verify, hashes, calls


def _chunk_pairs(  # pragma: no cover
    pairs: Sequence[Mapping[str, Any]],
    size: int,
) -> list[list[Mapping[str, Any]]]:
    return [list(pairs[index : index + size]) for index in range(0, len(pairs), size)]


def selected_python() -> str:  # pragma: no cover
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _first_json_line(text: str) -> JsonDict:  # pragma: no cover
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _parse_offloaded_layers(text: str) -> int | None:  # pragma: no cover
    patterns = (
        r"offloaded\s+(\d+)\s*/\s*\d+\s+layers?\s+to\s+GPU",
        r"offloading\s+(\d+)\s+repeating\s+layers?\s+to\s+GPU",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _tail(text: str, *, limit: int = 4000) -> str:  # pragma: no cover
    return text[-limit:]


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "panel_complete": artifact["panel_complete"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
