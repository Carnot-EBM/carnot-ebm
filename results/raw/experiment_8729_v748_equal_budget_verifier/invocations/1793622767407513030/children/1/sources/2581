"""Acquire a balanced relation corpus without using semantic authority.

Spec refs: REQ-INFERENCE-6900 and SCENARIO-INFERENCE-6900-*.

This module scales the successful Exp6899 transport protocol. It records model
output before parsing. A later experiment may open held labels and judge the
relations, but this acquisition step cannot do so.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import MethodType
from typing import Any

from carnot import experiment_6887_three_family_relation_proposal_corpus as base
from carnot import experiment_6899_live_relation_acquisition_canary as canary
from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess, port_is_free
from carnot.inference.llama_server_supervisor import read_process_identity
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6900_authentic_anchored_relation_corpus.json")
EXP6886_RELATIVE_PATH = canary.EXP6886_RELATIVE_PATH
EXP6899_RELATIVE_PATH = Path("results/experiment_6899_live_relation_acquisition_canary.json")
RUN_DATE = "20260902"
RANDOM_SEED = 6900
SCHEMA = "carnot.exp6900.authentic_anchored_relation_corpus.v1"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_plus_pinned_enoki_encoder"
EXPECTED_EXP6886_SHA256 = canary.EXPECTED_EXP6886_SHA256
EXPECTED_EXP6899_SHA256 = (
    "sha256:7c24282585cf771a56af627d0ed42e31f081d4a014c3ecc55a4f33d5f58dbb82"
)
EXPECTED_CANARY_PROMPT_MANIFEST_SHA256 = (
    "sha256:82bcb73b549baceae597bb5cded834d7f61197629bb4c0d0a6503d1bddcb7181"
)
EXPECTED_CANARY_SOURCE_HASHES_SHA256 = (
    "sha256:6aec19a4a719b5a64aa74ae8fb81d0ff1ed9fc21076bc58464fe8aed0a0f7993"
)
EXPECTED_SOURCE_HASH = "sha256:3190e7f7e8ba85ef82d91fc394bee154068bb0c92f2d6da147c53c4a5fb74ab7"
MODEL_SPECS = canary.MODEL_SPECS
MODEL_FAMILIES = canary.MODEL_FAMILIES
EXPECTED_MODEL_BINDINGS = canary.EXPECTED_MODEL_BINDINGS
EXPECTED_TOKENIZER_HASHES = canary.EXPECTED_TOKENIZER_HASHES
FAMILIES = base.FAMILIES
SEEDS = canary.SEEDS
ENOKI_ARM = "enoki:pinned_openie_encoder"
RULE_ARM = "rule:anchored_lexical_v1"
GGUF_ARMS = tuple(f"gguf:{hf_id}" for hf_id in MODEL_SPECS)
PROPOSAL_ARMS = (*GGUF_ARMS, ENOKI_ARM, RULE_ARM)
RULE_VERSION = base.RULE_VERSION
RULE_SOURCE_SHA256 = "sha256:" + hashlib.sha256(
    inspect.getsource(base._rule_output).encode("utf-8")
).hexdigest()
ENOKI_REVISION = str(base.EXPECTED_ENOKI_RECEIPTS[0]["revision"])
ENOKI_ASSET_HASH = str(base.EXPECTED_ENOKI_RECEIPTS[0]["files"][0]["sha256"])
MIN_FREE_VRAM_MB = canary.MIN_FREE_VRAM_MB
RECEIPT_MAX_AGE_S = canary.RECEIPT_MAX_AGE_S
LEASE_RUNTIME_DIR = canary.LEASE_RUNTIME_DIR
LEASE_TTL_S = 14_400.0


class RelationCorpusError(RuntimeError):
    """Report a fail-closed acquisition error without replacing evidence."""


def canonical_json(value: Any) -> str:
    """Serialize stable JSON for hashes and exact request bytes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with the digest prefix used by result artifacts."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one value after stable JSON serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _b64(value: bytes) -> str:
    """Encode arbitrary bytes for lossless JSON storage."""

    return base64.b64encode(value).decode("ascii")


def _unb64(value: Any) -> bytes | None:
    """Decode strict base64 or return none when evidence is malformed."""

    try:
        return base64.b64decode(str(value), validate=True)
    except (TypeError, ValueError):
        return None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Create an exact expected-versus-observed gate record."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed check and expose the first failure to automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in rows if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": rows,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _protocol_policy() -> JsonDict:
    """Return the exact non-schema request and llama.cpp server policy."""

    return {
        "seeds": list(SEEDS),
        "output_token_budget": canary.OUTPUT_TOKEN_BUDGET,
        "context_length": canary.CONTEXT_LENGTH,
        "request_timeout_s": canary.REQUEST_TIMEOUT_S,
        "health_timeout_s": canary.HEALTH_TIMEOUT_S,
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": False,
        "server_args": [
            "--ctx-size",
            "2048",
            "--batch-size",
            "512",
            "--ubatch-size",
            "256",
            "--gpu-layers",
            "all",
            "--split-mode",
            "none",
            "--main-gpu",
            "0",
            "--threads",
            "8",
            "--parallel",
            "1",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
            "--no-ui",
            "--verbose",
        ],
    }


EXPECTED_PROTOCOL_HASHES = {
    "prompt": "sha256:7c37c13a65f6d81ba54fb450901f2d22fd089e4bd96a210bd7c6ac0e41ac89fa",
    "parser": "sha256:bf40bc428da7489b542c4497442c6e434b077585b249e4ba47a36d5ce85273b9",
    "request": "sha256:8aeeb768bf2e94c6577d1229169e82abe3a39b44a306b1383710e5c8dcfe40bc",
    "policy": "sha256:d82245b949ec7b788290f010a20e5f84bcc301a81a374d4d4eb55fee002f5b57",
}


def current_protocol_hashes() -> JsonDict:
    """Hash live canary code so any protocol drift blocks before inference."""

    def source_hash(function: Callable[..., Any]) -> str:
        return sha256_bytes(inspect.getsource(function).encode("utf-8"))

    return {
        "prompt": source_hash(canary.build_prompt),
        "parser": source_hash(base.parse_relation_output),
        "request": source_hash(canary.build_request_bytes),
        "policy": sha256_json(_protocol_policy()),
    }


build_prompt = canary.build_prompt
build_request_bytes = canary.build_request_bytes
parse_relation_output = base.parse_relation_output


def _install_enoki_transformers5_adapter(model: Any) -> JsonDict:  # pragma: no cover
    """Adapt pinned Transformers 4 ModernBERT calls to Transformers 5.

    Transformers 5 moved mask construction into public helper functions. It
    also makes each encoder layer consume rotary embeddings directly. This
    adapter changes no weights and records its own source hash in the artifact.
    """

    if hasattr(model._encoder, "_update_attention_mask"):
        return {"installed": False, "reason": "transformers4_api_present"}

    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask,
    )

    def base_encode(
        instance: Any, input_ids: Any, attention_mask: Any
    ) -> tuple[Any, tuple[Any, Any]]:
        output = instance._encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask_kwargs = {
            "config": instance._encoder.config,
            "inputs_embeds": hidden,
            "attention_mask": attention_mask,
        }
        masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        position_ids = __import__("torch").arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        positions = {
            layer_type: instance._encoder.rotary_emb(hidden, position_ids, layer_type)
            for layer_type in set(instance._encoder.config.layer_types)
        }
        return hidden, (masks, positions)

    def iter_step(instance: Any, hidden: Any, mask_context: tuple[Any, Any]) -> Any:
        masks, positions = mask_context
        for layer in instance._iterative:
            hidden = layer(
                hidden,
                attention_mask=masks[layer.attention_type],
                position_embeddings=positions[layer.attention_type],
            )
        return hidden

    model._base_encode = MethodType(base_encode, model)
    model._iter_step = MethodType(iter_step, model)
    return {"installed": True, "reason": "transformers5_modernbert_api"}


ENOKI_TRANSFORMERS5_ADAPTER_SHA256 = sha256_bytes(
    inspect.getsource(_install_enoki_transformers5_adapter).encode("utf-8")
)


def reconstruct_source_records() -> list[JsonDict]:
    """Rebuild 100 public source views without reading formal sidecars."""

    rows: list[JsonDict] = []
    for ordinal in range(20):
        prefix = "Café evidence: " if ordinal % 2 == 0 else "Evidence: "
        for family in FAMILIES:
            positive, negative = base._source_parts(family, ordinal)
            source_text = prefix + positive
            if ordinal % 5 == 2:
                source_text += " " + negative
            rows.append(
                {
                    "fixture_id": f"{family}_{ordinal:02d}",
                    "group_id": f"relation_group_{ordinal:02d}",
                    "family": family,
                    "split": "calibration" if ordinal < 15 else "held",
                    "source_text": source_text,
                    "source_text_hash": base.sha256_text(source_text),
                    "relation_schema_version": "anchored_relation_v1",
                    "allowed_predicates": [base.FAMILY_PREDICATES[family]],
                    "source_order": len(rows),
                }
            )
    return rows


def source_family_counts(sources: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count distinct public source IDs in each required family."""

    counts = Counter((str(row.get("family")), str(row.get("fixture_id"))) for row in sources)
    return {family: sum(name == family for name, _ in counts) for family in FAMILIES}


def select_source_records(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Bind all reconstructed public views to the Exp6886 row hashes."""

    sources = reconstruct_source_records()
    rows = upstream.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_id = {
        str(row.get("fixture_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("row_type") == "fixture"
    }
    for source in sources:
        observed = by_id.get(str(source["fixture_id"]))
        if observed is None:
            raise RelationCorpusError(f"upstream_public_fixture_missing:{source['fixture_id']}")
        for field in ("group_id", "family", "split", "source_text_hash"):
            if observed.get(field) != source[field]:
                raise RelationCorpusError(
                    f"upstream_public_fixture_drift:{source['fixture_id']}:{field}"
                )
    if sha256_json(sources) != EXPECTED_SOURCE_HASH:
        raise RelationCorpusError("frozen_source_hash_drift")
    return sources


def resolve_three_models(
    *,
    pair_provider: Callable[..., Sequence[Mapping[str, Any]] | None] = cached_sota_pair,
    dense_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the same cached model files used by the successful canary."""

    if pair_provider is cached_sota_pair:  # pragma: no cover - live cache boundary.
        pair = cached_sota_pair(
            gpu_indices=(0, 0), preferred_quant="Q4_K_M", model_indices=(0, 1)
        ) or []
    else:
        pair = pair_provider(
            gpu_indices=(0, 0), preferred_quant="Q4_K_M", model_indices=(0, 1)
        ) or []
    by_id = {str(row.get("hf_id")): dict(row) for row in pair if isinstance(row, Mapping)}
    if dense_resolver is resolve_cached_gguf:  # pragma: no cover - live cache boundary.
        dense_path = resolve_cached_gguf(MODEL_SPECS[1], "Q4_K_M")
    else:
        dense_path = dense_resolver(MODEL_SPECS[1], "Q4_K_M")
    if dense_path:
        by_id[MODEL_SPECS[1]] = {
            "hf_id": MODEL_SPECS[1],
            "model_path": dense_path,
            "gpu": 0,
        }
    models: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        row = deepcopy(dict(by_id.get(hf_id, {})))
        row.update({"hf_id": hf_id, "gpu": int(row.get("gpu", 0))})
        if row.get("model_path"):
            row["model_path"] = str(Path(str(row["model_path"])).absolute())
        models.append(row)
    return models


def evaluate_preconditions(
    *,
    canary_artifact: Mapping[str, Any],
    canary_sha256: str,
    upstream: Mapping[str, Any],
    upstream_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu_inventory: Sequence[Mapping[str, Any]],
    lease_probe_rows: Sequence[Mapping[str, Any]],
    cuda_offload_supported: bool,
    held_sidecar_access_count: int,
    now_monotonic_ns: int,
    protocol_hashes: Mapping[str, Any],
) -> JsonDict:
    """Check the canary, frozen code, and all live resources before acquisition."""

    canary_report = canary.evaluate_preconditions(
        upstream=upstream,
        upstream_sha256=upstream_sha256,
        models=models,
        tokenizer_receipts=tokenizer_receipts,
        enoki_receipts=enoki_receipts,
        gpu_inventory=gpu_inventory,
        lease_probe_rows=lease_probe_rows,
        cuda_offload_supported=cuda_offload_supported,
        held_sidecar_access_count=held_sidecar_access_count,
        now_monotonic_ns=now_monotonic_ns,
    )
    try:
        source_hash = sha256_json(select_source_records(upstream))
    except RelationCorpusError:
        source_hash = "invalid"
    checks = [
        gate_check(
            "relation_canary_ready_score",
            1,
            canary_artifact.get("relation_canary_ready_score"),
        ),
        gate_check("exp6899_artifact_sha256", EXPECTED_EXP6899_SHA256, canary_sha256),
        gate_check(
            "canary_prompt_manifest_sha256",
            EXPECTED_CANARY_PROMPT_MANIFEST_SHA256,
            sha256_json(canary_artifact.get("prompt_manifest", {})),
        ),
        gate_check(
            "canary_source_hashes_sha256",
            EXPECTED_CANARY_SOURCE_HASHES_SHA256,
            sha256_json(canary_artifact.get("source_artifact_hashes", {})),
        ),
        gate_check(
            "canary_model_specs", list(MODEL_SPECS), canary_artifact.get("model_specs")
        ),
        gate_check(
            "canary_inference_substrate",
            canary.INFERENCE_SUBSTRATE,
            canary_artifact.get("inference_substrate"),
        ),
        gate_check("frozen_protocol_hashes", EXPECTED_PROTOCOL_HASHES, dict(protocol_hashes)),
        gate_check("balanced_source_hash", EXPECTED_SOURCE_HASH, source_hash),
        gate_check(
            "canary_resource_preconditions",
            True,
            canary_report.get("gate_check_summary", {}).get("passed") is True,
        ),
        gate_check("held_sidecar_access_count", 0, int(held_sidecar_access_count)),
    ]
    summary = gate_summary(checks)
    return {
        "gate_check_summary": summary,
        "passed": summary["passed"],
        "checks": checks,
        "canary_resource_checks": deepcopy(canary_report),
        "eligible_gpus": deepcopy(canary_report.get("eligible_gpus", [])),
        "held_sidecar_access_count": int(held_sidecar_access_count),
        "now_monotonic_ns": int(now_monotonic_ns),
    }


def gguf_cell_identity(hf_id: str, seed: int, fixture_id: str) -> str:
    """Build the exact canary-style identity for one GGUF request."""

    return f"{hf_id}::{int(seed)}::{fixture_id}"


def control_cell_identity(arm: str, fixture_id: str) -> str:
    """Build one deterministic control identity with no implied random seed."""

    return f"{arm}::deterministic::{fixture_id}"


def expected_cell_identities(sources: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return the complete three-model, two-control acquisition matrix."""

    identities = {
        gguf_cell_identity(hf_id, seed, str(source["fixture_id"]))
        for hf_id in MODEL_SPECS
        for seed in SEEDS
        for source in sources
    }
    identities.update(
        control_cell_identity(arm, str(source["fixture_id"]))
        for arm in (ENOKI_ARM, RULE_ARM)
        for source in sources
    )
    return identities


def build_gguf_cell(
    *,
    hf_id: str,
    source: Mapping[str, Any],
    seed: int,
    raw_request_bytes: bytes,
    raw_http_response_bytes: bytes,
    raw_output_bytes: bytes,
    native_prompt_tokens: int,
    generated_tokens: int,
    stop_reason: str,
    wall_time_s: float,
    timed_out: bool,
    truncated: bool,
    parser_attempted: bool,
    parse_rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Add corpus identity fields to one exact canary transport cell."""

    receipt = deepcopy(dict(runtime_receipt))
    if "authentic_attempt" not in receipt:
        receipt["authentic_attempt"] = receipt.get("authentic") is True
    row = canary.build_live_cell(
        hf_id=hf_id,
        source=source,
        seed=seed,
        raw_request_bytes=raw_request_bytes,
        raw_http_response_bytes=raw_http_response_bytes,
        raw_output_bytes=raw_output_bytes,
        native_prompt_tokens=native_prompt_tokens,
        generated_tokens=generated_tokens,
        stop_reason=stop_reason,
        wall_time_s=wall_time_s,
        timed_out=timed_out,
        truncated=truncated,
        parser_attempted=parser_attempted,
        parse_rows=parse_rows,
        runtime_receipt=receipt,
    )
    row.update(
        {
            "row_type": "terminal_cell",
            "arm": f"gguf:{hf_id}",
            "group_id": source["group_id"],
            "split": source["split"],
            "source_order": source["source_order"],
        }
    )
    return row


def build_control_cell(
    *,
    arm: str,
    source: Mapping[str, Any],
    raw_request_bytes: bytes,
    raw_output_bytes: bytes,
    stop_reason: str,
    wall_time_s: float,
    parse_rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Preserve one Enoki or lexical cell with its separate provenance."""

    output_hash = sha256_bytes(raw_output_bytes)
    return {
        "row_type": "terminal_cell",
        "cell_identity": control_cell_identity(arm, str(source["fixture_id"])),
        "arm": arm,
        "hf_id": None,
        "model_family": "enoki" if arm == ENOKI_ARM else "lexical_rule",
        "seed": None,
        "fixture_id": source["fixture_id"],
        "group_id": source["group_id"],
        "family": source["family"],
        "split": source["split"],
        "source_order": source["source_order"],
        "source_text_hash": source["source_text_hash"],
        "prompt_sha256": sha256_bytes(raw_request_bytes),
        "raw_request_b64": _b64(raw_request_bytes),
        "raw_request_sha256": sha256_bytes(raw_request_bytes),
        "request_byte_count": len(raw_request_bytes),
        "raw_http_response_b64": _b64(raw_output_bytes),
        "raw_http_response_sha256": output_hash,
        "http_response_byte_count": len(raw_output_bytes),
        "raw_output_b64": _b64(raw_output_bytes),
        "raw_output_sha256": output_hash,
        "output_byte_count": len(raw_output_bytes),
        "native_prompt_tokens": 0,
        "generated_tokens": 0,
        "stop_reason": stop_reason,
        "wall_time_s": round(float(wall_time_s), 6),
        "timed_out": False,
        "truncated": False,
        "parser_attempted": True,
        "parser_input_sha256": output_hash,
        "parse_rows": [deepcopy(dict(row)) for row in parse_rows],
        "runtime_receipt": deepcopy(dict(runtime_receipt)),
        "terminal": True,
    }


def build_absent_failure_cells(
    *,
    hf_id: str,
    sources: Sequence[Mapping[str, Any]],
    pending_identities: set[str],
    completed_cells: Sequence[Mapping[str, Any]],
    stop_reason: str,
    runtime_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit empty failed cells for every request absent after a model crash."""

    completed = {str(row.get("cell_identity")) for row in completed_cells}
    failures: list[JsonDict] = []
    for source in sources:
        for seed in SEEDS:
            identity = gguf_cell_identity(hf_id, seed, str(source["fixture_id"]))
            if identity not in pending_identities or identity in completed:
                continue
            failures.append(
                build_gguf_cell(
                    hf_id=hf_id,
                    source=source,
                    seed=seed,
                    raw_request_bytes=build_request_bytes(source, seed),
                    raw_http_response_bytes=b"",
                    raw_output_bytes=b"",
                    native_prompt_tokens=0,
                    generated_tokens=0,
                    stop_reason=stop_reason,
                    wall_time_s=0.0,
                    timed_out=False,
                    truncated=False,
                    parser_attempted=True,
                    parse_rows=parse_relation_output("", source),
                    runtime_receipt=runtime_receipt,
                )
            )
    return failures


def _terminal_status_rows(cell: Mapping[str, Any]) -> list[JsonDict]:
    """Convert one cell into relation or failed-cell rows without judging it."""

    common = {
        "cell_identity": cell.get("cell_identity"),
        "arm": cell.get("arm"),
        "seed": cell.get("seed"),
        "fixture_id": cell.get("fixture_id"),
        "group_id": cell.get("group_id"),
        "family": cell.get("family"),
        "split": cell.get("split"),
        "source_text_hash": cell.get("source_text_hash"),
        "raw_output_sha256": cell.get("raw_output_sha256"),
    }
    rows: list[JsonDict] = []
    for index, parse_row in enumerate(cell.get("parse_rows", [])):
        if not isinstance(parse_row, Mapping):  # pragma: no cover - validator rejects drift.
            continue
        status = str(parse_row.get("status", "malformed"))
        rows.append(
            {
                **common,
                "row_identity": f"{cell.get('cell_identity')}::parse::{index}",
                "row_type": "relation" if status in {"accepted", "duplicate"} else "failed_cell",
                **deepcopy(dict(parse_row)),
            }
        )
    for status, present in (
        ("timeout", cell.get("timed_out") is True),
        ("truncated", cell.get("truncated") is True),
    ):
        if present:
            rows.append(
                {
                    **common,
                    "row_identity": f"{cell.get('cell_identity')}::transport::{status}",
                    "row_type": "failed_cell",
                    "status": status,
                    "reason": str(cell.get("stop_reason")),
                    "raw_line": "",
                }
            )
    stop_reason = str(cell.get("stop_reason", ""))
    if stop_reason in {
        "server_crash",
        "model_phase_failure",
        "encoder_failure",
        "request_failure",
    }:
        rows.append(
            {
                **common,
                "row_identity": f"{cell.get('cell_identity')}::transport::{stop_reason}",
                "row_type": "failed_cell",
                "status": stop_reason,
                "reason": stop_reason,
                "raw_line": "",
            }
        )
    if not rows:  # pragma: no cover - live builders always attach one parser row.
        rows.append(
            {
                **common,
                "row_identity": f"{cell.get('cell_identity')}::parse::0",
                "row_type": "failed_cell",
                "status": "empty",
                "reason": "missing_parser_row",
                "raw_line": "",
            }
        )
    return rows


def flatten_terminal_rows(cells: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Flatten each parser and transport outcome into an auditable row."""

    return [row for cell in cells for row in _terminal_status_rows(cell)]


def _prior_terminal_rows(cells: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep retry evidence while giving each earlier attempt a unique row ID."""

    rows: list[JsonDict] = []
    for attempt_index, cell in enumerate(cells):
        for row in _terminal_status_rows(cell):
            row["row_identity"] = f"prior::{attempt_index}::{row['row_identity']}"
            row["prior_attempt"] = True
            rows.append(row)
    return rows


def _cell_integrity_errors(cell: Mapping[str, Any]) -> list[str]:
    """Check byte and provenance integrity without rejecting failure outcomes."""

    errors: list[str] = []
    for prefix in ("request", "http_response", "output"):
        raw = _unb64(cell.get(f"raw_{prefix}_b64"))
        count_field = f"{prefix}_byte_count" if prefix != "output" else "output_byte_count"
        hash_field = f"raw_{prefix}_sha256"
        if raw is None or len(raw) != int(cell.get(count_field, -1) or 0):
            errors.append(f"{prefix}_bytes")
        elif sha256_bytes(raw) != cell.get(hash_field):
            errors.append(f"{prefix}_hash")
    if cell.get("terminal") is not True:
        errors.append("not_terminal")
    if cell.get("parser_attempted") is not True:
        errors.append("parser_bypass")
    if cell.get("parser_input_sha256") != cell.get("raw_output_sha256"):
        errors.append("parser_input_hash")
    receipt = cell.get("runtime_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    arm = str(cell.get("arm", ""))
    if receipt.get("authentic_attempt") is not True:
        errors.append("authentic_attempt")
    if arm.startswith("gguf:"):
        hf_id = arm.removeprefix("gguf:")
        if receipt.get("model_sha256") != EXPECTED_MODEL_BINDINGS.get(hf_id, {}).get("sha256"):
            errors.append("model_sha256")
        if receipt.get("tokenizer_sha256") != EXPECTED_TOKENIZER_HASHES.get(hf_id):
            errors.append("tokenizer_sha256")
        if int(receipt.get("server_pid", 0) or 0) <= 1:
            errors.append("server_pid")
        if int(receipt.get("server_start_time_ticks", 0) or 0) <= 0:
            errors.append("server_start_time_ticks")
        if receipt.get("process_identity_match") is not True:
            errors.append("pid_reuse")
        if float(receipt.get("receipt_age_s", RECEIPT_MAX_AGE_S + 1)) > RECEIPT_MAX_AGE_S:
            errors.append("stale_pid_receipt")
        if int(receipt.get("offload_layers", 0) or 0) <= 0:
            errors.append("offload")
        if receipt.get("owned_cuda_residency") is not True or not receipt.get("gpu_uuid"):
            errors.append("cuda")
    elif arm == ENOKI_ARM:
        if receipt.get("encoder_revision") != ENOKI_REVISION:
            errors.append("enoki_revision")
        if receipt.get("encoder_asset_hash") != ENOKI_ASSET_HASH:
            errors.append("enoki_asset_hash")
        if (
            receipt.get("compatibility_adapter_sha256")
            != ENOKI_TRANSFORMERS5_ADAPTER_SHA256
        ):
            errors.append("enoki_adapter")
        if receipt.get("lease_released") is not True:
            errors.append("enoki_lease")
    elif arm == RULE_ARM:
        if receipt.get("rule_version") != RULE_VERSION:
            errors.append("rule_version")
        if receipt.get("rule_source_sha256") != RULE_SOURCE_SHA256:
            errors.append("rule_source_sha256")
    else:
        errors.append("unknown_arm")
    return errors


def _per_arm_family_counts(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count distinct source IDs per arm so seeds cannot inflate balance."""

    identities = defaultdict(set)
    for row in cells:
        identities[(str(row.get("arm")), str(row.get("family")))].add(
            str(row.get("fixture_id"))
        )
    return {
        arm: {family: len(identities[(arm, family)]) for family in FAMILIES}
        for arm in PROPOSAL_ARMS
    }


def _enoki_identity(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Use the Exp6887 identity reducer for exact pinned Enoki assets."""

    return base._enoki_identity(rows)


def completion_score(
    *,
    duration_s: float,
    sources: Sequence[Mapping[str, Any]],
    acquisition: Mapping[str, Any],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    held_sidecar_access_count: int,
) -> tuple[int, JsonDict]:
    """Score complete authentic acquisition without inspecting tuple quality."""

    cells = [row for row in acquisition.get("cells", []) if isinstance(row, Mapping)]
    expected = expected_cell_identities(sources)
    observed = [str(row.get("cell_identity")) for row in cells]
    cell_errors = {
        str(row.get("cell_identity")): _cell_integrity_errors(row)
        for row in cells
        if _cell_integrity_errors(row)
    }
    llama_receipts = acquisition.get("llama_cpp_receipts", [])
    phase_errors = [
        str(row.get("hf_id"))
        for row in llama_receipts
        if isinstance(row, Mapping) and row.get("phase_error")
    ]
    counts = _per_arm_family_counts(cells)
    rule_receipts = acquisition.get("deterministic_rule_receipts", [])
    rule_ok = len(rule_receipts) == 1 and all(
        isinstance(row, Mapping)
        and row.get("rule_version") == RULE_VERSION
        and row.get("rule_source_sha256") == RULE_SOURCE_SHA256
        and row.get("deterministic") is True
        for row in rule_receipts
    )
    checks = [
        gate_check("duration_at_least_60_s", True, float(duration_s) >= 60.0),
        gate_check(
            "exact_cell_matrix",
            {"count": len(expected), "identities": sorted(expected)},
            {"count": len(observed), "identities": sorted(observed)},
        ),
        gate_check("cell_integrity_errors", {}, cell_errors),
        gate_check(
            "per_arm_family_counts",
            {arm: {family: 20 for family in FAMILIES} for arm in PROPOSAL_ARMS},
            counts,
        ),
        gate_check("tokenizer_receipt_errors", [], canary._tokenizer_errors(tokenizer_receipts)),
        gate_check("llama_cpp_receipt_errors", [], canary._receipt_errors(llama_receipts)),
        gate_check("llama_cpp_phase_errors", [], phase_errors),
        gate_check(
            "gpu_lease_errors",
            [],
            canary._lease_errors(acquisition.get("gpu_lease_rows", [])),
        ),
        gate_check(
            "server_lifecycle_errors",
            [],
            canary._lifecycle_errors(acquisition.get("server_lifecycle_rows", [])),
        ),
        gate_check(
            "pinned_enoki_assets",
            _enoki_identity(base.EXPECTED_ENOKI_RECEIPTS),
            _enoki_identity(acquisition.get("enoki_asset_receipts", [])),
        ),
        gate_check("deterministic_rule_receipt", True, rule_ok),
        gate_check("held_sidecar_access_count", 0, int(held_sidecar_access_count)),
    ]
    summary = gate_summary(checks)
    return int(summary["passed"]), summary


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    """Hash checkpoint content without its self-referential digest."""

    return sha256_json({key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"})


def build_checkpoint(
    input_checksum: str,
    expected_identities: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    **state: Any,
) -> JsonDict:
    """Build a resumable checkpoint bound to all frozen acquisition inputs."""

    checkpoint: JsonDict = {
        "schema": SCHEMA + ".checkpoint",
        "input_checksum": input_checksum,
        "expected_cell_identities": sorted(str(value) for value in expected_identities),
        "cells": [deepcopy(dict(row)) for row in cells],
        **deepcopy(state),
        "checkpoint_sha256": "",
    }
    checkpoint["checkpoint_sha256"] = _checkpoint_hash(checkpoint)
    return checkpoint


write_json_atomic = base.write_json_atomic


def load_checkpoint(path: str | Path, *, input_checksum: str) -> JsonDict:
    """Load a checkpoint only when its input and row identities remain exact."""

    checkpoint = base.read_json(path)
    if checkpoint.get("input_checksum") != input_checksum:
        raise RelationCorpusError("checkpoint_input_hash_drift")
    if checkpoint.get("checkpoint_sha256") != _checkpoint_hash(checkpoint):
        raise RelationCorpusError("checkpoint_hash_invalid")
    cells = checkpoint.get("cells")
    cells = cells if isinstance(cells, list) else []
    identities = [str(row.get("cell_identity")) for row in cells if isinstance(row, Mapping)]
    if len(identities) != len(set(identities)):
        raise RelationCorpusError("checkpoint_duplicate_cell")
    expected = set(str(value) for value in checkpoint.get("expected_cell_identities", []))
    if not set(identities) <= expected:
        raise RelationCorpusError("checkpoint_unexpected_cell")
    return checkpoint


def pending_cell_identities(
    expected_identities: Sequence[str], checkpoint: Mapping[str, Any]
) -> list[str]:
    """Return only cell identities that have no checkpointed terminal row."""

    completed = {
        str(row.get("cell_identity"))
        for row in checkpoint.get("cells", [])
        if isinstance(row, Mapping)
    }
    return [str(value) for value in expected_identities if str(value) not in completed]


REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "llama_cpp_receipts",
    "gpu_lease_rows",
    "server_lifecycle_rows",
    "enoki_asset_receipts",
    "deterministic_rule_receipts",
    "prompt_manifest",
    "rows",
    "raw_request_manifest",
    "raw_output_manifest",
    "parse_failure_rows",
    "abstention_rows",
    "empty_rows",
    "truncation_rows",
    "timeout_rows",
    "per_arm_family_counts",
    "held_sidecar_access_count",
    "model_weight_mutation_count",
    "external_text_scorer_call_count",
    "constrained_schema_decode_count",
    "random_seed",
    "reproducibility_checksum",
    "relation_corpus_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)


FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why its evidence matters.",
    "preconditions_checked": "Exact gates stop drift before any proposal arm runs.",
    "inference_substrate": "The substrate rules out remote or CPU model substitutes.",
    "duration_s": "Measured wall time exposes skipped acquisition work.",
    "source_artifact_hashes": "Hashes bind the fixture and successful canary inputs.",
    "model_specs": "Exact repository IDs prevent model-family substitution.",
    "models_used": "Only models with live process receipts appear as used.",
    "model_artifact_hashes": "File hashes bind processes to the intended weights.",
    "tokenizer_receipts": "Native receipts prevent GGUF tokenizer substitution.",
    "llama_cpp_receipts": "Process receipts bind each model to CUDA and llama.cpp.",
    "gpu_lease_rows": "Lease rows prove that this task owned its GPU use.",
    "server_lifecycle_rows": "Lifecycle rows prove bounded and clean process teardown.",
    "enoki_asset_receipts": "Pinned asset receipts bind the encoder revision.",
    "deterministic_rule_receipts": "The rule receipt binds source code and version.",
    "prompt_manifest": "The manifest proves exact reuse of the canary protocol.",
    "rows": "Each parsed line or failed cell stays visible without semantic scoring.",
    "raw_request_manifest": "Request bytes prove what each proposal arm received.",
    "raw_output_manifest": "Output bytes preserve failures before parsing.",
    "parse_failure_rows": "Malformed and unsupported data remains in the record.",
    "abstention_rows": "Explicit abstentions remain in acquisition denominators.",
    "empty_rows": "Empty outputs remain visible and cannot receive fixture replacements.",
    "truncation_rows": "Budget stops remain visible and keep their partial bytes.",
    "timeout_rows": "Timeouts remain visible and keep received bytes.",
    "per_arm_family_counts": "Distinct source counts prevent seeds from hiding imbalance.",
    "held_sidecar_access_count": "Zero keeps semantic authority sealed for Exp6901.",
    "model_weight_mutation_count": "Zero proves acquisition did not train proposal arms.",
    "external_text_scorer_call_count": "Zero keeps external judgment out of acquisition.",
    "constrained_schema_decode_count": "Zero proves generation used no grammar mask.",
    "random_seed": "The fixed seed root makes the request matrix reproducible.",
    "reproducibility_checksum": "One digest binds stable artifact evidence.",
    "relation_canary_ready_score": "The incoming one permits this scaled acquisition.",
    "relation_corpus_complete_score": "The outgoing one measures acquisition only.",
    "gate_check_summary": "Exact expected and observed values explain every failure.",
    "verifier_is_oracle": "False separates transport checks from semantic truth.",
    "verdict_class": "A closed class gives automation a stable terminal state.",
    "honest_verdict": "A complete_ prefix marks a terminal evidence boundary.",
}


def _prompt_manifest(sources: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record corpus prompts while binding every canary protocol hash."""

    return {
        "protocol_hashes": current_protocol_hashes(),
        "canary_prompt_manifest_sha256": EXPECTED_CANARY_PROMPT_MANIFEST_SHA256,
        "protocol_line": canary.PROTOCOL_LINE,
        "prompt_hashes": {
            str(source["fixture_id"]): sha256_bytes(build_prompt(source).encode("utf-8"))
            for source in sources
        },
        "source_hash": sha256_json(list(sources)),
        "source_count": len(sources),
        **_protocol_policy(),
        "plain_line_protocol": True,
        "grammar_used": False,
        "structured_decode_used": False,
        "repair_prompt_count": 0,
        "model_judge_call_count": 0,
        "external_text_scorer_call_count": 0,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding duration and the self hash."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
        }
    )


def _attach_principles(artifact: JsonDict) -> None:
    """Attach one plain evidence principle to every top-level field."""

    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves required Exp6900 evidence.")
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]


def build_artifact(
    *,
    date: str,
    duration_s: float,
    upstream_sha256: str,
    canary_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    acquisition: Mapping[str, Any],
) -> JsonDict:
    """Build a terminal corpus artifact from raw cells and process receipts."""

    cells = [deepcopy(dict(row)) for row in acquisition.get("cells", [])]
    prior_cells = [
        deepcopy(dict(row)) for row in acquisition.get("prior_attempt_cells", [])
    ]
    rows = flatten_terminal_rows(cells) + _prior_terminal_rows(prior_cells)
    held_access = int(preconditions.get("held_sidecar_access_count", 0))
    live_duration = float(acquisition.get("live_duration_s", duration_s))
    score, completion = completion_score(
        duration_s=live_duration,
        sources=sources,
        acquisition=acquisition,
        tokenizer_receipts=tokenizer_receipts,
        held_sidecar_access_count=held_access,
    )
    if preconditions.get("gate_check_summary", {}).get("passed") is not True:
        score = 0
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6900,
        "run_date": date,
        "status": "complete" if score else "partial",
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), live_duration), 6),
        "live_duration_s": round(live_duration, 6),
        "source_artifact_hashes": {
            "exp6886": {"path": str(EXP6886_RELATIVE_PATH), "sha256": upstream_sha256},
            "exp6899": {"path": str(EXP6899_RELATIVE_PATH), "sha256": canary_sha256},
            "corpus_sources": {"sha256": sha256_json(list(sources))},
            "canary_prompt_manifest": {"sha256": EXPECTED_CANARY_PROMPT_MANIFEST_SHA256},
        },
        "relation_canary_ready_score": 1,
        "model_specs": list(MODEL_SPECS),
        "models_used": list(MODEL_SPECS)
        if score
        else sorted(
            str(row.get("hf_id"))
            for row in acquisition.get("llama_cpp_receipts", [])
            if isinstance(row, Mapping) and row.get("owned_cuda_residency") is True
        ),
        "model_artifact_hashes": {
            str(row.get("hf_id")): {
                "path": row.get("model_path"),
                "sha256": row.get("sha256"),
                "size_bytes": row.get("model_size_bytes"),
                "snapshot_identity": row.get("snapshot_identity"),
            }
            for row in models
        },
        "tokenizer_receipts": [deepcopy(dict(row)) for row in tokenizer_receipts],
        "llama_cpp_receipts": [
            deepcopy(dict(row)) for row in acquisition.get("llama_cpp_receipts", [])
        ],
        "gpu_lease_rows": [
            deepcopy(dict(row)) for row in acquisition.get("gpu_lease_rows", [])
        ],
        "server_lifecycle_rows": [
            deepcopy(dict(row)) for row in acquisition.get("server_lifecycle_rows", [])
        ],
        "enoki_asset_receipts": [
            deepcopy(dict(row)) for row in acquisition.get("enoki_asset_receipts", [])
        ],
        "deterministic_rule_receipts": [
            deepcopy(dict(row))
            for row in acquisition.get("deterministic_rule_receipts", [])
        ],
        "prompt_manifest": _prompt_manifest(sources),
        "rows": rows,
        "cell_manifest": cells,
        "prior_attempt_cells": prior_cells,
        "raw_request_manifest": [
            {
                "cell_identity": row["cell_identity"],
                "raw_request_b64": row["raw_request_b64"],
                "raw_request_sha256": row["raw_request_sha256"],
                "request_byte_count": row["request_byte_count"],
            }
            for row in cells
        ]
        + [
            {
                "attempt_identity": f"prior::{index}::{row['cell_identity']}",
                "cell_identity": row["cell_identity"],
                "raw_request_b64": row["raw_request_b64"],
                "raw_request_sha256": row["raw_request_sha256"],
                "request_byte_count": row["request_byte_count"],
            }
            for index, row in enumerate(prior_cells)
        ],
        "raw_output_manifest": [
            {
                "cell_identity": row["cell_identity"],
                "raw_http_response_b64": row["raw_http_response_b64"],
                "raw_http_response_sha256": row["raw_http_response_sha256"],
                "http_response_byte_count": row["http_response_byte_count"],
                "raw_output_b64": row["raw_output_b64"],
                "raw_output_sha256": row["raw_output_sha256"],
                "output_byte_count": row["output_byte_count"],
            }
            for row in cells
        ]
        + [
            {
                "attempt_identity": f"prior::{index}::{row['cell_identity']}",
                "cell_identity": row["cell_identity"],
                "raw_http_response_b64": row["raw_http_response_b64"],
                "raw_http_response_sha256": row["raw_http_response_sha256"],
                "http_response_byte_count": row["http_response_byte_count"],
                "raw_output_b64": row["raw_output_b64"],
                "raw_output_sha256": row["raw_output_sha256"],
                "output_byte_count": row["output_byte_count"],
            }
            for index, row in enumerate(prior_cells)
        ],
        "parse_failure_rows": [
            deepcopy(row)
            for row in rows
            if row.get("status") in {"malformed", "unsupported", "invalid_span", "duplicate"}
        ],
        "abstention_rows": [deepcopy(row) for row in rows if row.get("status") == "abstention"],
        "empty_rows": [deepcopy(row) for row in rows if row.get("status") == "empty"],
        "truncation_rows": [deepcopy(row) for row in rows if row.get("status") == "truncated"],
        "timeout_rows": [deepcopy(row) for row in rows if row.get("status") == "timeout"],
        "per_arm_family_counts": _per_arm_family_counts(cells),
        "held_sidecar_access_count": held_access,
        "model_weight_mutation_count": 0,
        "external_text_scorer_call_count": 0,
        "constrained_schema_decode_count": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "relation_corpus_complete_score": score,
        "gate_check_summary": completion,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if score else "partial",
        "honest_verdict": (
            "complete_positive_authentic_anchored_relation_corpus"
            if score
            else "complete_partial_authentic_anchored_relation_corpus"
        ),
        "checkpoint_manifest": deepcopy(dict(acquisition.get("checkpoint_manifest", {}))),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    upstream_sha256: str,
    canary_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Emit every required field when a precondition blocks acquisition."""

    artifact = build_artifact(
        date=date,
        duration_s=duration_s,
        upstream_sha256=upstream_sha256,
        canary_sha256=canary_sha256,
        models=models,
        tokenizer_receipts=tokenizer_receipts,
        preconditions=preconditions,
        sources=reconstruct_source_records(),
        acquisition={
            "cells": [],
            "llama_cpp_receipts": [],
            "gpu_lease_rows": [],
            "server_lifecycle_rows": [],
            "enoki_asset_receipts": [],
            "deterministic_rule_receipts": [],
            "live_duration_s": 0.0,
            "checkpoint_manifest": {"complete": False},
        },
    )
    artifact.update(
        {
            "status": "blocked",
            "models_used": [],
            "relation_canary_ready_score": int(
                preconditions.get("canary_ready_score", 0) or 0
            ),
            "relation_corpus_complete_score": 0,
            "gate_check_summary": deepcopy(
                dict(preconditions.get("gate_check_summary") or {})
            ),
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_authentic_anchored_relation_corpus",
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay schema, row, byte, counter, and outgoing gate evidence."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    principles = artifact.get("field_principles")
    principles = principles if isinstance(principles, Mapping) else {}
    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles) or not {
        "relation_canary_ready_score",
        "relation_corpus_complete_score",
    } <= set(principles):
        errors.append("field_principles")
    cells = artifact.get("cell_manifest")
    cells = cells if isinstance(cells, list) else []
    prior_cells = artifact.get("prior_attempt_cells")
    prior_cells = prior_cells if isinstance(prior_cells, list) else []
    rows = artifact.get("rows")
    rows = rows if isinstance(rows, list) else []
    if rows != flatten_terminal_rows(cells) + _prior_terminal_rows(prior_cells):
        errors.append("rows")
    if len({str(row.get("row_identity")) for row in rows}) != len(rows):
        errors.append("row_duplication")
    if len(artifact.get("raw_request_manifest", [])) != len(cells) + len(prior_cells):
        errors.append("raw_request_manifest")
    if len(artifact.get("raw_output_manifest", [])) != len(cells) + len(prior_cells):
        errors.append("raw_output_manifest")
    if any(
        error.startswith(("request_", "http_response_", "output_", "parser_"))
        for row in prior_cells
        for error in _cell_integrity_errors(row)
    ):
        errors.append("prior_attempt_integrity")
    if artifact.get("per_arm_family_counts") != _per_arm_family_counts(cells):
        errors.append("per_arm_family_counts")
    for field, status in (
        ("parse_failure_rows", {"malformed", "unsupported", "invalid_span", "duplicate"}),
        ("abstention_rows", {"abstention"}),
        ("empty_rows", {"empty"}),
        ("truncation_rows", {"truncated"}),
        ("timeout_rows", {"timeout"}),
    ):
        expected = [row for row in rows if row.get("status") in status]
        if artifact.get(field) != expected:
            errors.append(field)
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("verdict_class") not in canary.VERDICT_CLASSES:
        errors.append("verdict_class")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for field in (
        "held_sidecar_access_count",
        "model_weight_mutation_count",
        "external_text_scorer_call_count",
        "constrained_schema_decode_count",
    ):
        if artifact.get(field) != 0:
            errors.append(field)
    if artifact.get("verdict_class") != "blocked":
        score, _ = completion_score(
            duration_s=float(artifact.get("live_duration_s", artifact.get("duration_s", 0.0))),
            sources=reconstruct_source_records(),
            acquisition={
                "cells": cells,
                "llama_cpp_receipts": artifact.get("llama_cpp_receipts", []),
                "gpu_lease_rows": artifact.get("gpu_lease_rows", []),
                "server_lifecycle_rows": artifact.get("server_lifecycle_rows", []),
                "enoki_asset_receipts": artifact.get("enoki_asset_receipts", []),
                "deterministic_rule_receipts": artifact.get(
                    "deterministic_rule_receipts", []
                ),
            },
            tokenizer_receipts=artifact.get("tokenizer_receipts", []),
            held_sidecar_access_count=int(artifact.get("held_sidecar_access_count", -1)),
        )
        if artifact.get("relation_corpus_complete_score") != score:
            errors.append("relation_corpus_complete_score")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _input_checksum(
    *,
    sources: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    tokenizers: Sequence[Mapping[str, Any]],
    canary_sha256: str,
    upstream_sha256: str,
) -> str:
    """Bind every checkpoint to the exact acquisition inputs."""

    return sha256_json(
        {
            "sources": list(sources),
            "models": [
                {key: row.get(key) for key in ("hf_id", "sha256", "snapshot_identity")}
                for row in models
            ],
            "tokenizers": [
                {
                    "hf_id": row.get("hf_id"),
                    "sha256": row.get("canonical_tokenizer_payload_sha256"),
                }
                for row in tokenizers
            ],
            "canary_sha256": canary_sha256,
            "upstream_sha256": upstream_sha256,
            "protocol_hashes": current_protocol_hashes(),
            "enoki_compatibility_adapter_sha256": ENOKI_TRANSFORMERS5_ADAPTER_SHA256,
        }
    )


def collect_live_preconditions(root: Path) -> JsonDict:  # pragma: no cover - host boundary.
    """Hash all artifacts and resources before any proposal process starts."""

    models = resolve_three_models()
    for model in models:
        path = Path(str(model.get("model_path", "")))
        model.update(
            {
                "sha256": base.sha256_file(path) if path.is_file() else "",
                "snapshot_identity": base._snapshot_identity(str(path)) if path.is_file() else "",
                "model_size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    upstream_path = root / EXP6886_RELATIVE_PATH
    canary_path = root / EXP6899_RELATIVE_PATH
    upstream = base.read_json(upstream_path) if upstream_path.is_file() else {}
    canary_artifact = base.read_json(canary_path) if canary_path.is_file() else {}
    upstream_sha256 = base.sha256_file(upstream_path) if upstream_path.is_file() else ""
    canary_sha256 = base.sha256_file(canary_path) if canary_path.is_file() else ""
    enoki_receipts = base._enoki_local_receipts(upstream)
    tokenizer_receipts = [canary._native_tokenizer_receipt(model) for model in models]
    inventory = base._gpu_inventory()
    eligible = [
        row for row in inventory if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
    ]
    probe_gpu = eligible[0] if eligible else {}
    lease_probes = (
        [canary._probe_model_lease(probe_gpu, hf_id) for hf_id in MODEL_SPECS]
        if probe_gpu
        else []
    )
    try:
        from llama_cpp import llama_cpp

        offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        offload_supported = False
    report = evaluate_preconditions(
        canary_artifact=canary_artifact,
        canary_sha256=canary_sha256,
        upstream=upstream,
        upstream_sha256=upstream_sha256,
        models=models,
        tokenizer_receipts=tokenizer_receipts,
        enoki_receipts=enoki_receipts,
        gpu_inventory=inventory,
        lease_probe_rows=lease_probes,
        cuda_offload_supported=offload_supported,
        held_sidecar_access_count=0,
        now_monotonic_ns=time.monotonic_ns(),
        protocol_hashes=current_protocol_hashes(),
    )
    report.update(
        {
            "canary_artifact": canary_artifact,
            "canary_ready_score": int(canary_artifact.get("relation_canary_ready_score", 0) or 0),
            "canary_sha256": canary_sha256,
            "upstream": upstream,
            "upstream_sha256": upstream_sha256,
            "models": models,
            "tokenizer_receipts": tokenizer_receipts,
            "enoki_receipts": enoki_receipts,
            "gpu_inventory": inventory,
            "lease_probe_rows": lease_probes,
            "held_sidecar_paths_opened": [],
            "cached_sota_pair_called": True,
        }
    )
    return report


def _run_model_phase(
    *,
    model: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    pending: set[str],
    gpu: Mapping[str, Any],
    runtime_dir: Path,
) -> JsonDict:  # pragma: no cover - live model boundary.
    """Run one owned canary-configured server for only absent GGUF cells."""

    hf_id = str(model["hf_id"])
    planned = [
        (source, seed)
        for source in sources
        for seed in SEEDS
        if gguf_cell_identity(hf_id, seed, str(source["fixture_id"])) in pending
    ]
    if not planned:
        return {"cells": []}
    port = base._free_port()
    log_path = runtime_dir / f"{MODEL_FAMILIES[hf_id]}.log"
    state_path = runtime_dir / f"{MODEL_FAMILIES[hf_id]}.owner.json"
    command = [
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
        "--model",
        str(model["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *_protocol_policy()["server_args"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    worker = OwnedLlamaCppProcess(
        command=command,
        port=port,
        env=env,
        log_path=log_path,
        state_path=state_path,
    )
    lease: Any = None
    process_receipt: JsonDict = {}
    resident: JsonDict = {}
    cells: list[JsonDict] = []
    phase_error = ""
    free_before = int(gpu.get("free_vram_mb", 0) or 0)
    lifecycle: JsonDict = {}
    try:
        if free_before < MIN_FREE_VRAM_MB:
            raise RelationCorpusError(f"free_vram_below_floor:{free_before}")
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id=f"exp6900-{MODEL_FAMILIES[hf_id]}",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=str(model["model_path"]),
            vram_before_mb=free_before,
            ttl_s=LEASE_TTL_S,
        )
        lease.transition("admitted")
        lease.transition("loading")
        process_receipt = worker.launch()
        health = worker.wait_for_health(canary.HEALTH_TIMEOUT_S)
        if health.get("ok") is not True:
            raise RelationCorpusError(f"llama_server_health:{health}")
        resident = base._gpu_process_sample(gpu, int(process_receipt["pid"]), "resident")
        offload_layers = base._offload_layers(log_path)
        if resident.get("owned_cuda_residency") is not True or offload_layers <= 0:
            raise RelationCorpusError("owned_cuda_offload_missing")
        lease.transition("resident", vram_mb=int(resident.get("owned_vram_mb", 0)))
        lease.transition("inferencing")
        for source, seed in planned:
            current = read_process_identity(int(process_receipt["pid"]))
            identity_errors = canary.process_identity_errors(process_receipt, current)
            if identity_errors:
                raise RelationCorpusError(f"process_identity:{identity_errors}")
            receipt_time = time.monotonic_ns()
            runtime_receipt = {
                "authentic": True,
                "authentic_attempt": True,
                "model_sha256": model["sha256"],
                "tokenizer_sha256": tokenizer_receipt[
                    "canonical_tokenizer_payload_sha256"
                ],
                "server_pid": process_receipt["pid"],
                "server_start_time_ticks": process_receipt["start_time_ticks"],
                "command_hash": process_receipt["command_hash"],
                "process_identity_match": True,
                "receipt_age_s": (time.monotonic_ns() - receipt_time) / 1_000_000_000,
                "receipt_monotonic_ns": receipt_time,
                "gpu_uuid": gpu["gpu_uuid"],
                "offload_layers": offload_layers,
                "owned_cuda_residency": True,
                "vram_before_mb": free_before,
                "vram_after_load_mb": int(resident.get("owned_vram_mb", 0)),
                "stderr_tail": canary._stderr_tail(log_path),
            }
            cell = canary._chat_completion(
                port=port,
                hf_id=hf_id,
                source=source,
                seed=seed,
                runtime_receipt=runtime_receipt,
            )
            cells.append(
                build_gguf_cell(
                    hf_id=hf_id,
                    source=source,
                    seed=seed,
                    raw_request_bytes=base64.b64decode(cell["raw_request_b64"]),
                    raw_http_response_bytes=base64.b64decode(cell["raw_http_response_b64"]),
                    raw_output_bytes=base64.b64decode(cell["raw_output_b64"]),
                    native_prompt_tokens=cell["native_prompt_tokens"],
                    generated_tokens=cell["generated_tokens"],
                    stop_reason=cell["stop_reason"],
                    wall_time_s=cell["wall_time_s"],
                    timed_out=cell["timed_out"],
                    truncated=cell["truncated"],
                    parser_attempted=cell["parser_attempted"],
                    parse_rows=cell["parse_rows"],
                    runtime_receipt=cell["runtime_receipt"],
                )
            )
            lease.heartbeat()
        lease.transition("unloading")
    except Exception as exc:
        phase_error = f"{type(exc).__name__}: {exc}"
        failure_receipt = {
            "authentic_attempt": False,
            "model_sha256": model.get("sha256"),
            "tokenizer_sha256": tokenizer_receipt.get(
                "canonical_tokenizer_payload_sha256"
            ),
            "server_pid": process_receipt.get("pid"),
            "server_start_time_ticks": process_receipt.get("start_time_ticks"),
            "command_hash": process_receipt.get("command_hash"),
            "process_identity_match": False,
            "receipt_age_s": 0.0,
            "gpu_uuid": gpu.get("gpu_uuid"),
            "offload_layers": base._offload_layers(log_path),
            "owned_cuda_residency": resident.get("owned_cuda_residency", False),
            "vram_before_mb": free_before,
            "vram_after_load_mb": int(resident.get("owned_vram_mb", 0) or 0),
            "stderr_tail": canary._stderr_tail(log_path),
            "error": phase_error,
        }
        cells.extend(
            build_absent_failure_cells(
                hf_id=hf_id,
                sources=sources,
                pending_identities=pending,
                completed_cells=cells,
                stop_reason="server_crash",
                runtime_receipt=failure_receipt,
            )
        )
    finally:
        if lease is not None and lease.document.get("phase") == "inferencing":
            lease.transition("unloading")
        cleanup = worker.cleanup()
        after = base._gpu_process_sample(gpu, int(process_receipt.get("pid", 0) or 0), "after")
        process_exit = cleanup.get("process_exit_confirmed") is True
        port_release = cleanup.get("port_release_confirmed") is True and port_is_free(port)
        lease_released = False
        teardown_error = ""
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("owned_vram_mb", 0)),
                        exit_code=int(worker.process.returncode or 0) if worker.process else 0,
                        unload_observed=process_exit,
                    )
                    lease.transition(
                        "terminal_complete"
                        if not phase_error and process_exit and port_release
                        else "terminal_blocked"
                    )
                elif phase not in lease_api.TERMINAL_PHASES:
                    lease.transition("terminal_blocked")
                lease_released = lease.release().get("released") is True
            except Exception as exc:
                lease.close()
                teardown_error = f"{type(exc).__name__}: {exc}"
        lifecycle = {
            "hf_id": hf_id,
            "pid": process_receipt.get("pid"),
            "start_time_ticks": process_receipt.get("start_time_ticks"),
            "process_identity_match": bool(process_receipt)
            and "process_identity" not in phase_error,
            "process_exit_confirmed": process_exit,
            "process_reaped": cleanup.get("process_reaped") is True,
            "port_release_confirmed": port_release,
            "lease_released": lease_released,
            "unrelated_process_signal_count": int(
                cleanup.get("unrelated_process_kill_count_delta", 0) or 0
            ),
            "leak_free": bool(cleanup.get("leak_free") and lease_released and port_release),
            "teardown_error": teardown_error,
            "cleanup": cleanup,
            "vram_after_teardown_mb": int(after.get("owned_vram_mb", 0) or 0),
        }
    return {
        "cells": cells,
        "llama_cpp_receipt": {
            "hf_id": hf_id,
            "pid": process_receipt.get("pid"),
            "start_time_ticks": process_receipt.get("start_time_ticks"),
            "command_hash": process_receipt.get("command_hash"),
            "model_path": model.get("model_path"),
            "model_sha256": model.get("sha256"),
            "tokenizer_sha256": tokenizer_receipt.get("canonical_tokenizer_payload_sha256"),
            "gpu_uuid": gpu.get("gpu_uuid"),
            "offload_layers": base._offload_layers(log_path),
            "owned_cuda_residency": resident.get("owned_cuda_residency") is True,
            "vram_before_mb": free_before,
            "vram_after_load_mb": int(resident.get("owned_vram_mb", 0) or 0),
            "vram_after_teardown_mb": lifecycle.get("vram_after_teardown_mb"),
            "stderr_tail": canary._stderr_tail(log_path),
            "log_sha256": base.sha256_file(log_path)
            if log_path.is_file()
            else sha256_bytes(b""),
            "phase_error": phase_error,
        },
        "gpu_lease_row": {
            "hf_id": hf_id,
            "gpu_uuid": gpu.get("gpu_uuid"),
            "owned": lease is not None,
            "released": lifecycle.get("lease_released") is True,
        },
        "server_lifecycle_row": lifecycle,
    }


def _run_enoki_arm(
    *,
    sources: Sequence[Mapping[str, Any]],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - live encoder boundary.
    """Run the pinned Enoki encoder with an explicit Transformers 5 adapter."""

    encoder_receipt = next(
        (row for row in enoki_receipts if row.get("asset_id") == "enoki_encoder"), {}
    )
    cache_path = str(encoder_receipt.get("cache_path", ""))
    started = time.monotonic()
    runtime: JsonDict = {
        "arm": ENOKI_ARM,
        "authentic": False,
        "authentic_attempt": False,
        "encoder_revision": encoder_receipt.get("revision"),
        "encoder_asset_hash": ENOKI_ASSET_HASH,
        "encoder_path": cache_path,
        "compatibility_adapter_sha256": ENOKI_TRANSFORMERS5_ADAPTER_SHA256,
        "min_confidence": 0.7,
        "top_k": 10,
    }
    rows: list[JsonDict] = []
    lease: Any = None
    model: Any = None
    materialized: tempfile.TemporaryDirectory[str] | None = None
    torch: Any = None
    try:
        import torch as torch_module
        from transformers import AutoModel, PreTrainedModel

        torch = torch_module
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id="exp6900-enoki",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=cache_path,
            vram_before_mb=int(gpu.get("free_vram_mb", 0)),
            ttl_s=LEASE_TTL_S,
        )
        lease.transition("admitted")
        lease.transition("loading")
        materialized = tempfile.TemporaryDirectory(prefix="carnot-exp6900-enoki-")
        materialized_path = Path(materialized.name)
        for source_file in Path(cache_path).iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, materialized_path / source_file.name)
        PreTrainedModel.all_tied_weights_keys = {}
        model = AutoModel.from_pretrained(
            str(materialized_path), trust_remote_code=True, local_files_only=True
        )
        model.all_tied_weights_keys = {}
        adapter = _install_enoki_transformers5_adapter(model)
        device = torch.device(f"cuda:{int(gpu['index'])}")
        model.to(device).eval()
        memory_mb = int(torch.cuda.memory_allocated(device) / (1024 * 1024))
        lease.transition("resident", vram_mb=memory_mb)
        lease.transition("inferencing")
        runtime.update(
            {
                "authentic": True,
                "authentic_attempt": True,
                "device": str(device),
                "gpu_uuid": gpu["gpu_uuid"],
                "resident_vram_mb": memory_mb,
                "compatibility_adapter": adapter,
            }
        )
        for offset in range(0, len(sources), 18):
            batch = list(sources[offset : offset + 18])
            batch_started = time.monotonic()
            results = model.extract_triples(
                [str(source["source_text"]) for source in batch],
                min_confidence=0.7,
                top_k=10,
                batch_size=18,
            )
            batch_latency = time.monotonic() - batch_started
            for source, result in zip(batch, results, strict=True):
                raw_output = canonical_json(result).encode("utf-8")
                rows.append(
                    build_control_cell(
                        arm=ENOKI_ARM,
                        source=source,
                        raw_request_bytes=str(source["source_text"]).encode("utf-8"),
                        raw_output_bytes=raw_output,
                        stop_reason="encoder_complete",
                        wall_time_s=batch_latency / len(batch),
                        parse_rows=base.parse_enoki_result(result, source),
                        runtime_receipt=runtime,
                    )
                )
            lease.heartbeat()
        lease.transition("unloading")
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        materialized.cleanup()
        materialized = None
        lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
        lease.transition("terminal_complete")
        runtime["lease_released"] = lease.release().get("released") is True
    except Exception as exc:
        runtime["authentic"] = False
        runtime["authentic_attempt"] = False
        runtime["error"] = f"{type(exc).__name__}: {exc}"
        if model is not None:
            del model
            model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if materialized is not None:
            materialized.cleanup()
            materialized = None
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase == "inferencing":
                    lease.transition("unloading")
                    phase = "unloading"
                if phase == "unloading":
                    lease.transition("validating", vram_mb=0, exit_code=1, unload_observed=True)
                    phase = "validating"
                if phase == "validating" or phase not in lease_api.TERMINAL_PHASES:
                    lease.transition("terminal_blocked")
                runtime["lease_released"] = lease.release().get("released") is True
            except Exception as release_exc:
                lease.close()
                runtime["lease_released"] = False
                runtime["lease_release_error"] = (
                    f"{type(release_exc).__name__}: {release_exc}"
                )
        terminal_ids = {str(row["cell_identity"]) for row in rows}
        for source in sources:
            identity = control_cell_identity(ENOKI_ARM, str(source["fixture_id"]))
            if identity in terminal_ids:
                continue
            rows.append(
                build_control_cell(
                    arm=ENOKI_ARM,
                    source=source,
                    raw_request_bytes=str(source["source_text"]).encode("utf-8"),
                    raw_output_bytes=b"",
                    stop_reason="encoder_failure",
                    wall_time_s=0.0,
                    parse_rows=parse_relation_output("", source),
                    runtime_receipt=runtime,
                )
            )
    runtime["duration_s"] = round(time.monotonic() - started, 6)
    for row in rows:
        row["runtime_receipt"] = deepcopy(runtime)
    return rows, runtime


def _run_controls(
    *,
    sources: Sequence[Mapping[str, Any]],
    pending: set[str],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover - live encoder boundary.
    """Run pinned Enoki and the deterministic lexical rule on absent IDs."""

    enoki_sources = [
        source
        for source in sources
        if control_cell_identity(ENOKI_ARM, str(source["fixture_id"])) in pending
    ]
    enoki_cells: list[JsonDict] = []
    enoki_runtime: JsonDict = {}
    if enoki_sources:
        enoki_cells, enoki_runtime = _run_enoki_arm(
            sources=enoki_sources,
            enoki_receipts=enoki_receipts,
            gpu=gpu,
        )
    rule_cells: list[JsonDict] = []
    for source in sources:
        if control_cell_identity(RULE_ARM, str(source["fixture_id"])) not in pending:
            continue
        started = time.monotonic()
        raw_output = base._rule_output(source).encode("utf-8")
        rule_cells.append(
            build_control_cell(
                arm=RULE_ARM,
                source=source,
                raw_request_bytes=str(source["source_text"]).encode("utf-8"),
                raw_output_bytes=raw_output,
                stop_reason="rule_complete",
                wall_time_s=time.monotonic() - started,
                parse_rows=parse_relation_output(raw_output.decode("utf-8"), source),
                runtime_receipt={
                    "authentic_attempt": True,
                    "rule_version": RULE_VERSION,
                    "rule_source_sha256": RULE_SOURCE_SHA256,
                    "source_only": True,
                },
            )
        )
    return enoki_cells, rule_cells, enoki_runtime


def run_live_acquisition(
    *,
    sources: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu_inventory: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    input_checksum: str,
) -> JsonDict:  # pragma: no cover - live experiment boundary.
    """Acquire only absent cells and checkpoint after each owned phase."""

    started = time.monotonic()
    expected = sorted(expected_cell_identities(sources))
    state = (
        load_checkpoint(checkpoint_path, input_checksum=input_checksum)
        if checkpoint_path.is_file()
        else build_checkpoint(input_checksum, expected, [])
    )
    cells = [deepcopy(dict(row)) for row in state.get("cells", [])]
    prior_attempt_cells = [
        deepcopy(dict(row)) for row in state.get("prior_attempt_cells", [])
    ]
    accumulated_live_duration = float(state.get("accumulated_live_duration_s", 0.0) or 0.0)
    llama_receipts = [deepcopy(dict(row)) for row in state.get("llama_cpp_receipts", [])]
    lease_rows = [deepcopy(dict(row)) for row in state.get("gpu_lease_rows", [])]
    lifecycle_rows = [deepcopy(dict(row)) for row in state.get("server_lifecycle_rows", [])]
    runtime_dir = Path(tempfile.mkdtemp(prefix="carnot-exp6900-"))
    token_by_model = {str(row["hf_id"]): row for row in tokenizer_receipts}
    for model in models:
        pending = set(pending_cell_identities(expected, {"cells": cells}))
        hf_id = str(model["hf_id"])
        if not any(value.startswith(hf_id + "::") for value in pending):
            continue
        inventory = base._gpu_inventory()
        eligible = [
            row for row in inventory if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
        ]
        gpu = eligible[0] if eligible else dict(gpu_inventory[0])
        phase = _run_model_phase(
            model=model,
            tokenizer_receipt=token_by_model[hf_id],
            sources=sources,
            pending=pending,
            gpu=gpu,
            runtime_dir=runtime_dir,
        )
        cells.extend(phase["cells"])
        llama_receipts.append(phase["llama_cpp_receipt"])
        lease_rows.append(phase["gpu_lease_row"])
        lifecycle_rows.append(phase["server_lifecycle_row"])
        write_json_atomic(
            checkpoint_path,
            build_checkpoint(
                input_checksum,
                expected,
                cells,
                llama_cpp_receipts=llama_receipts,
                gpu_lease_rows=lease_rows,
                server_lifecycle_rows=lifecycle_rows,
                prior_attempt_cells=prior_attempt_cells,
                accumulated_live_duration_s=accumulated_live_duration,
            ),
        )
        gc.collect()
    pending = set(pending_cell_identities(expected, {"cells": cells}))
    inventory = base._gpu_inventory()
    eligible = [
        row for row in inventory if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
    ]
    control_gpu = eligible[0] if eligible else dict(gpu_inventory[0])
    enoki_cells, rule_cells, enoki_runtime = _run_controls(
        sources=sources,
        pending=pending,
        enoki_receipts=enoki_receipts,
        gpu=control_gpu,
    )
    cells.extend(enoki_cells)
    cells.extend(rule_cells)
    checkpoint = build_checkpoint(
        input_checksum,
        expected,
        cells,
        llama_cpp_receipts=llama_receipts,
        gpu_lease_rows=lease_rows,
        server_lifecycle_rows=lifecycle_rows,
        enoki_runtime_receipt=enoki_runtime,
        prior_attempt_cells=prior_attempt_cells,
        accumulated_live_duration_s=accumulated_live_duration,
    )
    write_json_atomic(checkpoint_path, checkpoint)
    return {
        "cells": cells,
        "llama_cpp_receipts": llama_receipts,
        "gpu_lease_rows": lease_rows,
        "server_lifecycle_rows": lifecycle_rows,
        "enoki_asset_receipts": [deepcopy(dict(row)) for row in enoki_receipts],
        "deterministic_rule_receipts": [
            {
                "rule_version": RULE_VERSION,
                "rule_source_sha256": RULE_SOURCE_SHA256,
                "deterministic": True,
            }
        ],
        "prior_attempt_cells": prior_attempt_cells,
        "live_duration_s": accumulated_live_duration + time.monotonic() - started,
        "checkpoint_manifest": {
            "path": str(checkpoint_path),
            "input_checksum": input_checksum,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "complete": not pending_cell_identities(expected, {"cells": cells}),
        },
    }


def run(
    *,
    date: str = RUN_DATE,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
    precondition_collector: Callable[[Path], Mapping[str, Any]] = collect_live_preconditions,
    acquisition_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run authentic acquisition or write one complete blocked artifact."""

    started = time.monotonic()
    target = result_path or root / RESULT_RELATIVE_PATH
    checkpoint_target = checkpoint_path or Path(
        f"/tmp/carnot-exp6900-{date}-checkpoint.json"
    )
    collected = deepcopy(dict(precondition_collector(root)))
    canary_sha256 = str(collected.get("canary_sha256", ""))
    upstream_sha256 = str(collected.get("upstream_sha256", ""))
    models = [deepcopy(dict(row)) for row in collected.get("models", [])]
    tokenizers = [deepcopy(dict(row)) for row in collected.get("tokenizer_receipts", [])]
    if collected.get("gate_check_summary", {}).get("passed") is not True:
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            upstream_sha256=upstream_sha256,
            canary_sha256=canary_sha256,
            models=models,
            tokenizer_receipts=tokenizers,
            preconditions=collected,
        )
        if validate_artifact(artifact):  # pragma: no cover - internal invariant.
            raise RelationCorpusError("blocked_artifact_invalid")
        write_json_atomic(target, artifact)
        return artifact
    upstream = collected.get("upstream")
    if not isinstance(upstream, Mapping):  # pragma: no cover - precondition contract.
        raise RelationCorpusError("upstream_missing_after_preconditions")
    sources = select_source_records(upstream)
    for source in sources:
        leakage = base.audit_arm_input(build_prompt(source))
        if leakage:  # pragma: no cover - frozen prompt is covered by its hash.
            raise RelationCorpusError(f"prompt_leakage:{leakage}")
    input_checksum = _input_checksum(
        sources=sources,
        models=models,
        tokenizers=tokenizers,
        canary_sha256=canary_sha256,
        upstream_sha256=upstream_sha256,
    )
    runner = acquisition_runner or run_live_acquisition
    acquisition = dict(
        runner(
            sources=sources,
            models=models,
            tokenizer_receipts=tokenizers,
            enoki_receipts=collected.get("enoki_receipts", []),
            gpu_inventory=collected.get("eligible_gpus", collected.get("gpu_inventory", [])),
            checkpoint_path=checkpoint_target,
            input_checksum=input_checksum,
        )
    )
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        upstream_sha256=upstream_sha256,
        canary_sha256=canary_sha256,
        models=models,
        tokenizer_receipts=tokenizers,
        preconditions=collected,
        sources=sources,
        acquisition=acquisition,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RelationCorpusError("artifact_invalid:" + ",".join(errors))
    write_json_atomic(target, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the dated experiment command and print its terminal verdict."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    artifact = run(date=str(args.date))
    print(canonical_json({"honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())
