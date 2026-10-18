"""Exp6837 output-free forced-sequence compatibility scoring.

Spec refs: REQ-CONSTRAINT-6837, SCENARIO-CONSTRAINT-6837-*.

The experiment scores Exp6836 fixed candidates as sequences. It masks the
prompt, records candidate token log-probabilities, and reports margins without
generating answers or treating model energy as truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol
from urllib import request

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6837_three_family_output_free_compatibility.py"
)
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6837_three_family_output_free_compatibility.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6837_three_family_output_free_compatibility.json")
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6837_three_family_output_free_compatibility.checkpoint.json"
)
EXP6836_RELATIVE_PATH = Path("results/experiment_6836_typed_obligation_program_fixture.json")

SCHEMA = "carnot.experiment_6837.three_family_output_free_compatibility.v1"
INFERENCE_SUBSTRATE = "live_local_llama_cpp_cuda_forced_sequence_scoring"
RANDOM_SEED = 6837
RUN_DATE = "20260901"
DISK_FLOOR_MB = 1024
DEFAULT_CONTEXT_LENGTH = 8192
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_BATCH = 512
DEFAULT_N_UBATCH = 128
ROUND_DIGITS = 8

MANDATED_MODEL_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen3_6_35b_a3b",
    "unsloth/gemma-4-31B-it-GGUF": "gemma4_31b_it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma4_26b_a4b_it",
}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "process_receipts",
    "accelerator_samples",
    "random_seed",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "rows",
    "per_model_results",
    "per_atom_results",
    "joint_results",
    "shortcut_control_cells",
    "checkpoint_manifest",
    "method_parity_limits",
    "obligation_compatibility_stream_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Documents why each required field exists.",
    "preconditions_checked": "Names every live gate and the exact observed value.",
    "inference_substrate": "Distinguishes forced scoring from answer generation.",
    "duration_s": "Makes skipped or implausibly short work visible.",
    "model_specs": "Freezes the mandated model identities.",
    "models_used": "Shows which models actually ran.",
    "model_artifact_hashes": "Binds model bytes to scored rows.",
    "tokenizer_receipts": "Proves native GGUF tokenization and lengths.",
    "process_receipts": "Proves task-owned process identity and teardown.",
    "accelerator_samples": "Records CUDA device and lease observations.",
    "random_seed": "Makes batch ordering and canaries reproducible.",
    "source_artifact_hashes": "Binds Exp6836 and implementation inputs.",
    "reproducibility_checksum": "Detects drift across code, source, model, and rows.",
    "rows": "Stores every scored model and candidate-pair margin.",
    "per_model_results": "Keeps model families separate.",
    "per_atom_results": "Reports atom-family margins without pooling models.",
    "joint_results": "Reports joint and impossible-case margins separately.",
    "shortcut_control_cells": "Audits label position, length, prompt, and permutation cues.",
    "checkpoint_manifest": "Proves restart skips complete rows.",
    "method_parity_limits": "States this is not HSRM and not truth proof.",
    "obligation_compatibility_stream_ready_score": "Gates only on receipts and rows.",
    "gate_check_summary": "Names the failed gate in blocked artifacts.",
    "verifier_is_oracle": "Keeps exact labels external to model scores.",
    "verdict_class": "Uses the closed terminal class vocabulary.",
    "honest_verdict": "Gives a terminal complete-prefixed outcome.",
}


class SequenceScoringError(ValueError):
    """Stable failure type for forced-sequence scoring contract violations."""


class ForcedSequenceScorer(Protocol):
    """Minimal scorer boundary shared by fake tests and the live worker."""

    def start(self) -> JsonDict:
        """Start the scoring process and return process identity."""

    def score(
        self,
        prompt_text: str,
        candidate_text: str,
        row_identity: Mapping[str, Any],
    ) -> JsonDict:
        """Return prompt-masked candidate token ids and log-probabilities."""

    def close(self) -> JsonDict:
        """Tear down the scorer and return cleanup receipt."""


ScorerFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], ForcedSequenceScorer]


def canonical_json(value: Any) -> str:
    """Serialize JSON once so hashes and row receipts stay stable."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository's prefixed digest format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository's prefixed digest format."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks so large GGUFs do not need one memory buffer."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and reject arrays or scalar payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SequenceScoringError(f"json_object_required:{path}")
    return dict(payload)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact through an atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(target)


def model_family(hf_id: str) -> str:
    """Return the stable model-family identifier for a mandated HF id."""

    try:
        return MODEL_FAMILIES[hf_id]
    except KeyError as exc:
        raise SequenceScoringError(f"unknown_model_hf_id:{hf_id}") from exc


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one exact precondition or readiness gate row."""

    return {
        "check": str(check),
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _registry_by_hf_id() -> dict[str, JsonDict]:
    return {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}


def _tokenizer_receipt(source: Mapping[str, Any], model_path: str) -> JsonDict:
    provided = source.get("tokenizer_receipt")
    if isinstance(provided, Mapping):
        receipt = dict(provided)
        receipt.setdefault("source", "provided")
        receipt.setdefault("loadable", False)
        receipt.setdefault("detail", "")
        receipt["receipt_hash"] = str(
            receipt.get("receipt_hash") or sha256_text(canonical_json(receipt))
        )
        return receipt
    ok, detail = (
        gguf_tokenizer_loadable(model_path) if model_path else (False, "missing model_path")
    )
    receipt = {
        "source": "embedded_gguf_llama_cpp_vocab_only",
        "loadable": bool(ok),
        "detail": str(detail),
        "model_path_hash": sha256_text(str(Path(model_path).expanduser().resolve())),
    }
    receipt["receipt_hash"] = sha256_text(canonical_json(receipt))
    return receipt


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize the exact three model records and compute local hashes."""

    registry = _registry_by_hf_id()
    by_hf_id = {str(row.get("hf_id")): dict(row) for row in model_specs}
    normalized: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        source = by_hf_id.get(hf_id, {})
        reg = registry.get(hf_id, {})
        model_path = str(source.get("model_path") or source.get("cache_path") or "")
        path = Path(model_path).expanduser() if model_path else Path()
        present = bool(model_path and path.is_file())
        tokenizer = (
            _tokenizer_receipt(source, model_path)
            if present
            else {
                "source": "missing_model_path",
                "loadable": False,
                "detail": f"model_path missing or not on disk: {model_path!r}",
                "receipt_hash": "",
            }
        )
        if not tokenizer.get("receipt_hash"):
            tokenizer["receipt_hash"] = sha256_text(canonical_json(tokenizer))
        normalized.append(
            {
                "name": str(source.get("name") or reg.get("name") or hf_id.rsplit("/", 1)[-1]),
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": str(source.get("role") or reg.get("role") or ""),
                "gpu": int(source.get("gpu", index) or 0),
                "model_path": model_path,
                "cache_path": model_path,
                "local_model_present": present,
                "model_sha256": str(
                    source.get("model_sha256") or (sha256_file(path) if present else "")
                ),
                "tokenizer_receipt": tokenizer,
                "headline_eligible": source.get("headline_eligible") is not False,
                "quantization": str(
                    source.get("quantization") or reg.get("quantization") or "Q4_K_M"
                ),
                "context_length": int(source.get("context_length", DEFAULT_CONTEXT_LENGTH)),
                "cached_sota_pair_called": bool(source.get("cached_sota_pair_called", False)),
                "cached_sota_pair_hf_ids": list(source.get("cached_sota_pair_hf_ids") or []),
            }
        )
    return normalized


def resolve_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache dependent.
    """Call `cached_sota_pair()` and resolve all three mandated GGUFs."""

    cached_pair = cached_sota_pair(gpu_indices=(0, 1)) or []
    cached_hf_ids = [str(row.get("hf_id")) for row in cached_pair if isinstance(row, Mapping)]
    by_hf_id = {str(row.get("hf_id")): dict(row) for row in cached_pair if isinstance(row, Mapping)}
    registry = _registry_by_hf_id()
    rows: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        source = dict(by_hf_id.get(hf_id, {}))
        reg = registry.get(hf_id, {})
        quantization = str(reg.get("quantization") or "Q4_K_M")
        if not source.get("model_path"):
            source["model_path"] = resolve_cached_gguf(hf_id, quantization) or ""
        source.setdefault("name", reg.get("name") or hf_id.rsplit("/", 1)[-1])
        source["hf_id"] = hf_id
        source.setdefault("role", reg.get("role") or "")
        source.setdefault("gpu", index)
        source.setdefault("quantization", quantization)
        source["cached_sota_pair_called"] = True
        source["cached_sota_pair_hf_ids"] = cached_hf_ids
        rows.append(source)
    return normalize_model_specs(rows)


def _fixture_payload(root: Path) -> JsonDict:
    return read_json(root / EXP6836_RELATIVE_PATH)


def load_fixture_rows(root: Path, *, limit: int | None = None) -> list[JsonDict]:
    """Load frozen Exp6836 pair rows, optionally truncated for unit tests."""

    rows = [dict(row) for row in _fixture_payload(root).get("rows", [])]
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise SequenceScoringError("exp6836_rows_missing")
    return rows


def _source_artifact_hashes(root: Path) -> JsonDict:
    paths = {
        "exp6836": EXP6836_RELATIVE_PATH,
        "module": MODULE_RELATIVE_PATH,
        "wrapper": WRAPPER_RELATIVE_PATH,
        "spec": SPEC_RELATIVE_PATH,
    }
    return {
        key: {
            "path": str(path),
            "sha256": sha256_file(root / path) if (root / path).is_file() else "missing",
        }
        for key, path in paths.items()
    }


def _row_identity(model_spec: Mapping[str, Any], fixture_row: Mapping[str, Any]) -> str:
    return f"{model_spec['family']}::{fixture_row['row_id']}"


def _candidate_label(candidate: Mapping[str, Any]) -> str:
    exact = dict(candidate.get("exact_check") or {})
    return "compatible" if exact.get("satisfaction_predicate") is True else "violation"


def _select_candidate_pair(fixture_row: Mapping[str, Any]) -> tuple[JsonDict, JsonDict, int]:
    candidates = [dict(row) for row in fixture_row.get("candidates", [])]
    compatible = [row for row in candidates if _candidate_label(row) == "compatible"]
    violations = [row for row in candidates if _candidate_label(row) == "violation"]
    if len(compatible) != 1 or len(violations) != 1:
        raise SequenceScoringError(
            f"one_compatible_one_violation_required:{fixture_row.get('row_id')}"
        )
    return compatible[0], violations[0], candidates.index(compatible[0])


def _score_candidate(
    *,
    scorer: ForcedSequenceScorer,
    prompt_text: str,
    candidate: Mapping[str, Any],
    row_identity: str,
    label: str,
) -> JsonDict:
    inputs = dict(candidate.get("expected_tokenization_inputs") or {})
    candidate_text = str(inputs.get("candidate_text") or candidate.get("raw_text") or "")
    receipt = scorer.score(
        prompt_text,
        candidate_text,
        {
            "row_identity": row_identity,
            "candidate_id": candidate.get("candidate_id"),
            "label": label,
        },
    )
    token_ids = [int(token) for token in receipt.get("candidate_token_ids", [])]
    token_logprobs = [float(value) for value in receipt.get("token_logprobs", [])]
    if len(token_ids) != len(token_logprobs):
        raise SequenceScoringError(
            f"token_logprob_alignment_failed:{candidate.get('candidate_id')}"
        )
    conditional = round(
        float(receipt.get("conditional_log_likelihood", sum(token_logprobs))), ROUND_DIGITS
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "label": label,
        "raw_text_sha256": candidate.get("raw_text_sha256") or sha256_text(candidate_text),
        "prompt_token_ids": [int(token) for token in receipt.get("prompt_token_ids", [])],
        "candidate_token_ids": token_ids,
        "token_logprobs": [round(value, ROUND_DIGITS) for value in token_logprobs],
        "conditional_log_likelihood": conditional,
        "sequence_energy": round(-conditional, ROUND_DIGITS),
        "raw_receipt": dict(receipt.get("raw_receipt") or {}),
    }


def _obligation_count(candidate: Mapping[str, Any]) -> int:
    diagnostics = list(dict(candidate.get("exact_check") or {}).get("diagnostics") or [])
    return len(
        {
            str(row.get("obligation_id"))
            for row in diagnostics
            if row.get("obligation_id") != "__joint__"
        }
    )


def score_fixture_row(
    *,
    fixture_row: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    scorer: ForcedSequenceScorer,
) -> JsonDict:
    """Score one Exp6836 row and compute a prompt-masked sequence margin."""

    compatible_candidate, violation_candidate, compatible_position = _select_candidate_pair(
        fixture_row
    )
    prompt_text = str(fixture_row.get("prompt_text") or "")
    identity = _row_identity(model_spec, fixture_row)
    compatible = _score_candidate(
        scorer=scorer,
        prompt_text=prompt_text,
        candidate=compatible_candidate,
        row_identity=identity,
        label="compatible",
    )
    violation = _score_candidate(
        scorer=scorer,
        prompt_text=prompt_text,
        candidate=violation_candidate,
        row_identity=identity,
        label="violation",
    )
    if len(compatible["candidate_token_ids"]) != len(violation["candidate_token_ids"]):
        raise SequenceScoringError(f"unequal_candidate_token_length:{identity}")
    comp_score = float(compatible["conditional_log_likelihood"])
    viol_score = float(violation["conditional_log_likelihood"])
    violated_atoms = [
        str(row["atom_id"])
        for row in dict(violation_candidate.get("exact_check") or {}).get("diagnostics", [])
        if row.get("passed") is not True and row.get("atom_id")
    ]
    row: JsonDict = {
        "schema": SCHEMA + ".row",
        "row_identity": identity,
        "source_row_id": fixture_row.get("row_id"),
        "source_row_hash": fixture_row.get("row_hash"),
        "model_hf_id": model_spec.get("hf_id"),
        "model_family": model_spec.get("family"),
        "model_hash": model_spec.get("model_sha256"),
        "tokenizer_hash": dict(model_spec.get("tokenizer_receipt") or {}).get("receipt_hash"),
        "pair_id": fixture_row.get("pair_id"),
        "program_id": fixture_row.get("program_id"),
        "scenario_id": fixture_row.get("scenario_id"),
        "case_kind": fixture_row.get("case_kind"),
        "atom_family": fixture_row.get("case_kind"),
        "obligation_count": _obligation_count(compatible_candidate),
        "compatible_label_position": compatible_position,
        "prompt_length": fixture_row.get("prompt_length"),
        "prompt_sha256": fixture_row.get("prompt_sha256"),
        "candidate_length": len(compatible["candidate_token_ids"]),
        "candidate_text_length": len(
            str(
                dict(compatible_candidate.get("expected_tokenization_inputs") or {}).get(
                    "candidate_text",
                    compatible_candidate.get("raw_text", ""),
                )
            ).encode("utf-8")
        ),
        "permutation": fixture_row.get("label_swap"),
        "surface_form": fixture_row.get("surface_form"),
        "compatible": compatible,
        "violation": violation,
        "compatible_candidate_id": compatible_candidate.get("candidate_id"),
        "violation_candidate_id": violation_candidate.get("candidate_id"),
        "violated_atom_ids": violated_atoms,
        "log_likelihood_margin": round(comp_score - viol_score, ROUND_DIGITS),
        "energy_margin": round(
            float(compatible["sequence_energy"]) - float(violation["sequence_energy"]), ROUND_DIGITS
        ),
        "lower_compatible_energy": compatible["sequence_energy"] < violation["sequence_energy"],
        "no_generation": True,
        "prompt_masked": True,
        "row_hash": "",
    }
    row["row_hash"] = row_hash(row)
    return row


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one scored row while excluding its existing hash field."""

    payload = dict(row)
    payload["row_hash"] = ""
    return sha256_text(canonical_json(payload))


def build_checkpoint_manifest(
    rows: Sequence[Mapping[str, Any]], *, expected_row_count: int
) -> JsonDict:
    """Build the restart manifest that stores immutable completed rows."""

    stored_rows = [dict(row) for row in rows]
    hashes = {str(row["row_identity"]): str(row["row_hash"]) for row in stored_rows}
    return {
        "schema": SCHEMA + ".checkpoint",
        "expected_row_count": int(expected_row_count),
        "complete_row_count": len(stored_rows),
        "row_hashes": hashes,
        "rows": stored_rows,
        "resumed_row_count": 0,
        "missing_row_count": max(0, int(expected_row_count) - len(stored_rows)),
        "checkpoint_hash": sha256_text(
            canonical_json({"expected": expected_row_count, "hashes": hashes})
        ),
    }


def write_checkpoint(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Persist checkpoint rows without writing any tracked state in tests."""

    write_json_atomic(path, manifest)


def _read_checkpoint(path: Path) -> JsonDict:
    if not path.is_file():
        return {
            "schema": SCHEMA + ".checkpoint",
            "expected_row_count": 0,
            "complete_row_count": 0,
            "row_hashes": {},
            "rows": [],
        }
    return read_json(path)


def _verified_checkpoint_rows(path: Path) -> dict[str, JsonDict]:
    checkpoint = _read_checkpoint(path)
    rows = [dict(row) for row in checkpoint.get("rows", [])]
    hashes = dict(checkpoint.get("row_hashes") or {})
    verified: dict[str, JsonDict] = {}
    for row in rows:
        identity = str(row.get("row_identity"))
        if row_hash(row) != row.get("row_hash") or hashes.get(identity) != row.get("row_hash"):
            raise SequenceScoringError(f"checkpoint_row_hash_mismatch:{identity}")
        verified[identity] = row
    return verified


def forced_sequence_config(config: Mapping[str, Any] | None = None) -> JsonDict:
    """Return deterministic settings for live and fake scorers."""

    base = {
        "seed": RANDOM_SEED,
        "n_ctx": DEFAULT_CONTEXT_LENGTH,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "n_batch": DEFAULT_N_BATCH,
        "n_ubatch": DEFAULT_N_UBATCH,
        "sampling": False,
        "generation": False,
        "grammar": False,
        "answer_feedback": False,
        "prompt_masking": True,
    }
    if config:
        base.update(dict(config))
    return base


def run_model_phase(
    *,
    model_spec: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
    scorer_factory: ScorerFactory,
    checkpoint_path: str | Path,
    scorer_config: Mapping[str, Any] | None = None,
    expected_total_row_count: int | None = None,
) -> JsonDict:
    """Run one model phase, reusing valid checkpoint rows for that model."""

    checkpoint = Path(checkpoint_path)
    existing = _verified_checkpoint_rows(checkpoint)
    expected_identities = [_row_identity(model_spec, row) for row in fixture_rows]
    model_existing = {
        identity: existing[identity] for identity in expected_identities if identity in existing
    }
    missing_rows = [
        dict(row) for row in fixture_rows if _row_identity(model_spec, row) not in model_existing
    ]
    scorer = scorer_factory(model_spec, forced_sequence_config(scorer_config))
    process_receipt = scorer.start()
    new_rows: list[JsonDict] = []
    try:
        for fixture_row in missing_rows:
            new_rows.append(
                score_fixture_row(
                    fixture_row=fixture_row,
                    model_spec=model_spec,
                    scorer=scorer,
                )
            )
    finally:
        teardown = scorer.close()
    merged_by_identity = {**model_existing, **{str(row["row_identity"]): row for row in new_rows}}
    model_rows = [
        merged_by_identity[identity]
        for identity in expected_identities
        if identity in merged_by_identity
    ]
    all_rows_by_identity = {**existing, **merged_by_identity}
    expected_count = int(expected_total_row_count or len(fixture_rows))
    manifest = build_checkpoint_manifest(
        list(all_rows_by_identity.values()),
        expected_row_count=expected_count,
    )
    manifest["resumed_row_count"] = len(model_existing)
    manifest["missing_row_count"] = len(missing_rows)
    write_checkpoint(checkpoint, manifest)
    scores = [row["compatible"]["conditional_log_likelihood"] for row in model_rows]
    process_receipt = {
        **dict(process_receipt),
        "hf_id": model_spec.get("hf_id"),
        "family": model_spec.get("family"),
        "first_score": scores[0] if scores else None,
        "final_score": scores[-1] if scores else None,
        "teardown": teardown,
    }
    return {
        "model_hf_id": model_spec.get("hf_id"),
        "family": model_spec.get("family"),
        "rows": model_rows,
        "row_count": len(model_rows),
        "new_row_count": len(new_rows),
        "process_receipt": process_receipt,
        "checkpoint_manifest": manifest,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    check_rows = [dict(row) for row in checks]
    failed = [row for row in check_rows if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": first is None,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else None,
        "observed": first.get("observed") if first else None,
        "checks": check_rows,
    }


def _precondition_blockers(preconditions: Mapping[str, Any]) -> list[str]:
    blockers = [str(item) for item in preconditions.get("blocked_reasons") or []]
    if preconditions.get("preconditions_ready") is not True:
        blockers.append("preconditions_not_ready")
    for row in preconditions.get("checks") or []:
        if isinstance(row, Mapping) and row.get("passed") is not True:
            blockers.append(str(row.get("check")))
    return sorted(set(blockers))


def _model_artifact_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(row["hf_id"]): {
            "family": row.get("family"),
            "path": row.get("model_path"),
            "sha256": row.get("model_sha256"),
        }
        for row in model_specs
    }


def _tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "hf_id": row.get("hf_id"),
            "family": row.get("family"),
            **dict(row.get("tokenizer_receipt") or {}),
        }
        for row in model_specs
    ]


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), ROUND_DIGITS) if values else None


def _margin_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    margins = [float(row["log_likelihood_margin"]) for row in rows]
    return {
        "row_count": len(rows),
        "mean_log_likelihood_margin": _mean(margins),
        "min_log_likelihood_margin": round(min(margins), ROUND_DIGITS) if margins else None,
        "max_log_likelihood_margin": round(max(margins), ROUND_DIGITS) if margins else None,
        "lower_compatible_energy_count": sum(
            1 for row in rows if row.get("lower_compatible_energy") is True
        ),
    }


def _grouped_results(
    rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]
) -> list[JsonDict]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    out: list[JsonDict] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: canonical_json(item[0])):
        label = dict(zip(key_fields, key, strict=True))
        out.append({**label, **_margin_summary(group_rows)})
    return out


def _per_model_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return _grouped_results(rows, ("model_hf_id", "model_family"))


def _per_atom_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    atom_rows = [
        row for row in rows if row.get("case_kind") in {"atom_contradiction", "atom_omission"}
    ]
    return _grouped_results(
        atom_rows, ("model_hf_id", "model_family", "atom_family", "obligation_count")
    )


def _joint_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    joint_rows = [
        row for row in rows if row.get("case_kind") in {"joint_violation", "impossible_set"}
    ]
    return _grouped_results(
        joint_rows, ("model_hf_id", "model_family", "case_kind", "obligation_count")
    )


def _shortcut_control_cells(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return _grouped_results(
        rows,
        (
            "model_hf_id",
            "model_family",
            "atom_family",
            "obligation_count",
            "compatible_label_position",
            "prompt_length",
            "candidate_length",
            "permutation",
        ),
    )


def _readiness_checks(
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_row_count: int,
    process_receipts: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
) -> list[JsonDict]:
    families = {model_family(hf_id) for hf_id in MANDATED_MODEL_HF_IDS}
    return [
        gate_check("row_count_complete", expected_row_count, len(rows)),
        gate_check(
            "row_identities_unique", len(rows), len({row.get("row_identity") for row in rows})
        ),
        gate_check(
            "all_required_models_used",
            list(MANDATED_MODEL_HF_IDS),
            [row.get("hf_id") for row in process_receipts],
        ),
        gate_check(
            "model_families_separate",
            sorted(families),
            sorted({str(row.get("model_family")) for row in rows}),
        ),
        gate_check(
            "processes_owned",
            True,
            all(row.get("owned_by_task") is True for row in process_receipts),
        ),
        gate_check(
            "process_teardown_clean",
            True,
            all(
                dict(row.get("teardown") or {}).get("leak_free") is True for row in process_receipts
            ),
        ),
        gate_check(
            "checkpoint_complete",
            expected_row_count,
            checkpoint_manifest.get("complete_row_count"),
        ),
        gate_check(
            "all_row_hashes_valid",
            True,
            all(row_hash(row) == row.get("row_hash") for row in rows),
        ),
        gate_check(
            "raw_candidate_receipts_present",
            True,
            all(
                dict(row.get(side) or {}).get("raw_receipt")
                for row in rows
                for side in ("compatible", "violation")
            ),
        ),
        gate_check(
            "exact_model_hashes_present",
            True,
            all(str(row.get("model_sha256", "")).startswith("sha256:") for row in model_specs),
        ),
    ]


def _method_parity_limits() -> JsonDict:
    return {
        "fits_probe": False,
        "uses_llm_judge": False,
        "generates_answers": False,
        "hsrm_reproduction": False,
        "step_boundary_hidden_states_available": False,
        "effect_estimate_is_truth_proof": False,
        "truth_authority": "Exp6836 exact labels are external.",
    }


def _reproducibility_checksum(
    *,
    source_hashes: Mapping[str, Any],
    model_hashes: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return sha256_text(
        canonical_json(
            {
                "random_seed": RANDOM_SEED,
                "source_hashes": source_hashes,
                "model_hashes": model_hashes,
                "row_hashes": [row.get("row_hash") for row in rows],
                "inference_substrate": INFERENCE_SUBSTRATE,
            }
        )
    )


def _base_artifact(
    *,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    model_hashes = _model_artifact_hashes(model_specs)
    return {
        "schema": SCHEMA,
        "experiment_id": 6837,
        "run_date": RUN_DATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions_checked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "model_specs": [dict(row) for row in model_specs],
        "models_used": [],
        "model_artifact_hashes": model_hashes,
        "tokenizer_receipts": _tokenizer_receipts(model_specs),
        "process_receipts": [],
        "accelerator_samples": list(preconditions_checked.get("accelerator_samples") or []),
        "random_seed": RANDOM_SEED,
        "source_artifact_hashes": dict(source_hashes),
        "reproducibility_checksum": _reproducibility_checksum(
            source_hashes=source_hashes,
            model_hashes=model_hashes,
            rows=[],
        ),
        "rows": [],
        "per_model_results": [],
        "per_atom_results": [],
        "joint_results": [],
        "shortcut_control_cells": [],
        "checkpoint_manifest": build_checkpoint_manifest([], expected_row_count=0),
        "method_parity_limits": _method_parity_limits(),
        "obligation_compatibility_stream_ready_score": 0,
        "gate_check_summary": _gate_summary(list(preconditions_checked.get("checks") or [])),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_output_free_compatibility",
        "status": "blocked",
    }


def _blocked_artifact(
    *,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    artifact = _base_artifact(
        duration_s=duration_s,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        source_hashes=source_hashes,
    )
    artifact["gate_check_summary"] = _gate_summary(list(preconditions_checked.get("checks") or []))
    return artifact


def _complete_artifact(
    *,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    process_receipts: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    expected_row_count: int,
) -> JsonDict:
    model_hashes = _model_artifact_hashes(model_specs)
    readiness = _readiness_checks(
        rows=rows,
        expected_row_count=expected_row_count,
        process_receipts=process_receipts,
        model_specs=model_specs,
        checkpoint_manifest=checkpoint_manifest,
    )
    gate_summary = _gate_summary(readiness)
    ready = gate_summary["passed"]
    artifact = _base_artifact(
        duration_s=duration_s,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        source_hashes=source_hashes,
    )
    artifact.update(
        {
            "models_used": [row.get("hf_id") for row in process_receipts],
            "model_artifact_hashes": model_hashes,
            "process_receipts": [dict(row) for row in process_receipts],
            "rows": [dict(row) for row in rows],
            "per_model_results": _per_model_results(rows),
            "per_atom_results": _per_atom_results(rows),
            "joint_results": _joint_results(rows),
            "shortcut_control_cells": _shortcut_control_cells(rows),
            "checkpoint_manifest": dict(checkpoint_manifest),
            "obligation_compatibility_stream_ready_score": 1 if ready else 0,
            "gate_check_summary": gate_summary,
            "verdict_class": "positive" if ready else "partial",
            "honest_verdict": "complete_output_free_compatibility_stream_ready"
            if ready
            else "complete_partial_output_free_compatibility_stream_incomplete",
            "status": "complete" if ready else "partial",
        }
    )
    artifact["reproducibility_checksum"] = _reproducibility_checksum(
        source_hashes=source_hashes,
        model_hashes=model_hashes,
        rows=rows,
    )
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    scorer_factory: ScorerFactory | None = None,
    scorer_config: Mapping[str, Any] | None = None,
    fixture_row_limit: int | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6837 or write a blocked artifact when live gates fail."""

    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path) if result_path is not None else root / RESULT_RELATIVE_PATH
    checkpoint = (
        Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_RELATIVE_PATH
    )
    specs = normalize_model_specs(model_specs) if model_specs is not None else resolve_model_specs()
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(
            root=root, model_specs=specs, result_path=result, checkpoint_path=checkpoint
        )
    )
    source_hashes = _source_artifact_hashes(root)
    blockers = _precondition_blockers(preconditions)
    if blockers:
        artifact = _blocked_artifact(
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            preconditions_checked=preconditions,
            source_hashes=source_hashes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact
    rows = load_fixture_rows(root, limit=fixture_row_limit)
    factory = scorer_factory or SubprocessForcedSequenceScorer
    expected_total = len(rows) * len(specs)
    all_rows: list[JsonDict] = []
    process_receipts: list[JsonDict] = []
    checkpoint_manifest = build_checkpoint_manifest([], expected_row_count=expected_total)
    for spec in specs:
        phase = run_model_phase(
            model_spec=spec,
            fixture_rows=rows,
            scorer_factory=factory,
            checkpoint_path=checkpoint,
            scorer_config=scorer_config,
            expected_total_row_count=expected_total,
        )
        all_rows.extend(phase["rows"])
        process_receipts.append(dict(phase["process_receipt"]))
        checkpoint_manifest = dict(phase["checkpoint_manifest"])
    artifact = _complete_artifact(
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions_checked=preconditions,
        source_hashes=source_hashes,
        rows=all_rows,
        process_receipts=process_receipts,
        checkpoint_manifest=checkpoint_manifest,
        expected_row_count=expected_total,
    )
    if write:
        write_json_atomic(result, artifact)
    return artifact


def _utc_now() -> str:  # pragma: no cover - clock dependent.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _port_free(port: int) -> bool:  # pragma: no cover - host dependent.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _allocate_ports(count: int) -> list[int]:  # pragma: no cover - host dependent.
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            ports.append(int(sock.getsockname()[1]))
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return ports


def _run_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:  # pragma: no cover.
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _cuda_available() -> JsonDict:  # pragma: no cover - host dependent.
    torch_ok = False
    torch_detail = ""
    try:
        import torch

        torch_ok = bool(torch.cuda.is_available())
        torch_detail = f"torch_device_count={torch.cuda.device_count()}"
    except Exception as exc:
        torch_detail = f"{type(exc).__name__}: {exc}"
    smi = _run_command(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], timeout_s=10
    )
    return {
        "ok": torch_ok or smi.get("ok") is True,
        "torch_cuda_available": torch_ok,
        "torch_detail": torch_detail,
        "nvidia_smi": smi,
    }


def _accelerator_samples() -> list[JsonDict]:  # pragma: no cover - host dependent.
    smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    samples: list[JsonDict] = []
    for line in str(smi.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            samples.append(
                {
                    "index": parts[0],
                    "gpu_uuid": parts[1],
                    "name": parts[2],
                    "free_vram_mb": parts[3],
                    "total_vram_mb": parts[4],
                    "owned_by_task": False,
                }
            )
    return samples


def _compute_apps() -> list[JsonDict]:  # pragma: no cover - host dependent.
    smi = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    rows: list[JsonDict] = []
    for line in str(smi.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            rows.append({"pid": parts[0], "gpu_uuid": parts[1], "used_memory_mb": parts[2]})
    return rows


def _disk_ok(root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _token_logprob_support() -> JsonDict:  # pragma: no cover - environment dependent.
    try:
        from llama_cpp import Llama

        return {"ok": hasattr(Llama, "eval"), "detail": "llama_cpp.Llama.eval present"}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def _run_live_canaries(
    *,
    root: Path,
    model_specs: Sequence[Mapping[str, Any]],
    ports: Sequence[int],
) -> JsonDict:  # pragma: no cover - live model dependent.
    first_row = load_fixture_rows(root, limit=1)[0]
    receipts = []
    for spec, port in zip(model_specs, ports, strict=True):
        scorer = SubprocessForcedSequenceScorer(spec, {"port": port, "canary": True})
        process = scorer.start()
        try:
            row = score_fixture_row(fixture_row=first_row, model_spec=spec, scorer=scorer)
            receipts.append(
                {"hf_id": spec["hf_id"], "ok": True, "margin": row["log_likelihood_margin"]}
            )
        except Exception as exc:
            receipts.append(
                {"hf_id": spec["hf_id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            process["teardown"] = scorer.close()
    return {"ok": all(row.get("ok") is True for row in receipts), "receipts": receipts}


def collect_preconditions(
    *,
    root: Path,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
) -> JsonDict:  # pragma: no cover - host/resource dependent.
    """Collect required live gates while only observing unrelated processes."""

    exp6836 = _fixture_payload(root) if (root / EXP6836_RELATIVE_PATH).is_file() else {}
    ports = _allocate_ports(len(model_specs))
    port_status = all(_port_free(port) for port in ports)
    cuda = _cuda_available()
    apps = _compute_apps()
    disk = _disk_ok(root)
    token_logprobs = _token_logprob_support()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checks = [
        gate_check(
            "cached_sota_pair_called",
            True,
            all(row.get("cached_sota_pair_called") is True for row in model_specs),
        ),
        gate_check(
            "all_three_exact_gguf_files",
            list(MANDATED_MODEL_HF_IDS),
            [row.get("hf_id") for row in model_specs if row.get("local_model_present") is True],
        ),
        gate_check(
            "model_hashes_present",
            True,
            all(str(row.get("model_sha256", "")).startswith("sha256:") for row in model_specs),
        ),
        gate_check(
            "native_tokenizer_metadata",
            True,
            all(
                dict(row.get("tokenizer_receipt") or {}).get("loadable") is True
                for row in model_specs
            ),
        ),
        gate_check("token_log_probability_support", True, token_logprobs["ok"]),
        gate_check("cuda_available", True, cuda["ok"]),
        gate_check("exclusive_gpu_leases", True, not apps),
        gate_check("free_ports", True, port_status),
        gate_check("sufficient_disk", True, disk["ok"]),
        gate_check(
            "exp6836_typed_obligation_program_ready_score",
            1,
            exp6836.get("typed_obligation_program_ready_score"),
        ),
        gate_check(
            "exp6836_obligation_pair_fixture_ready_score",
            1,
            exp6836.get("obligation_pair_fixture_ready_score"),
        ),
    ]
    canary_allowed = all(row["passed"] for row in checks)
    canary = (
        _run_live_canaries(root=root, model_specs=model_specs, ports=ports)
        if canary_allowed
        else {"ok": False, "skipped": True, "reason": "prior_precondition_failed"}
    )
    checks.append(gate_check("live_canary_per_model", True, canary["ok"]))
    blocked = [str(row["check"]) for row in checks if row["passed"] is not True]
    return {
        "schema": SCHEMA + ".preconditions",
        "date": RUN_DATE,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "checks": checks,
        "blocked_reasons": blocked,
        "preconditions_ready": not blocked,
        "cuda": cuda,
        "accelerator_samples": _accelerator_samples(),
        "unrelated_compute_processes_observed": apps,
        "free_ports": ports,
        "disk": disk,
        "token_log_probability_support": token_logprobs,
        "canary": canary,
        "output_paths": {
            "result_path": str(result_path),
            "checkpoint_path": str(checkpoint_path),
            "result_parent_writable": os.access(result_path.parent, os.W_OK),
            "checkpoint_parent_writable": os.access(checkpoint_path.parent, os.W_OK),
        },
    }


class _WorkerEngine:  # pragma: no cover - live model dependent.
    def __init__(self, model_path: str, config: Mapping[str, Any]) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=int(config.get("n_gpu_layers", DEFAULT_N_GPU_LAYERS)),
            n_ctx=int(config.get("n_ctx", DEFAULT_CONTEXT_LENGTH)),
            n_batch=int(config.get("n_batch", DEFAULT_N_BATCH)),
            n_ubatch=int(config.get("n_ubatch", DEFAULT_N_UBATCH)),
            seed=int(config.get("seed", RANDOM_SEED)),
            logits_all=True,
            verbose=False,
        )

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False))

    @staticmethod
    def _logprob(logits: Any, token_id: int) -> float:
        values = [float(value) for value in logits]
        max_logit = max(values)
        denom = sum(math.exp(value - max_logit) for value in values)
        return values[token_id] - max_logit - math.log(denom)

    def score(
        self, prompt_text: str, candidate_text: str, row_identity: Mapping[str, Any]
    ) -> JsonDict:
        prompt_tokens = self._tokenize(prompt_text, add_bos=True)
        candidate_tokens = self._tokenize(candidate_text, add_bos=False)
        all_tokens = prompt_tokens + candidate_tokens
        self.llm.reset()
        self.llm.eval(all_tokens)
        scores = self.llm.scores
        token_logprobs = []
        for absolute_index, token_id in enumerate(
            all_tokens[len(prompt_tokens) :], start=len(prompt_tokens)
        ):
            token_logprobs.append(self._logprob(scores[absolute_index - 1], int(token_id)))
        return {
            "prompt_token_ids": prompt_tokens,
            "candidate_token_ids": candidate_tokens,
            "token_logprobs": [round(value, ROUND_DIGITS) for value in token_logprobs],
            "conditional_log_likelihood": round(sum(token_logprobs), ROUND_DIGITS),
            "raw_receipt": {
                "row_identity": row_identity.get("row_identity"),
                "candidate_id": row_identity.get("candidate_id"),
                "label": row_identity.get("label"),
                "prompt_token_count": len(prompt_tokens),
                "candidate_token_count": len(candidate_tokens),
                "token_logprob_count": len(token_logprobs),
            },
        }


class SubprocessForcedSequenceScorer:  # pragma: no cover - live model dependent.
    """Task-owned HTTP worker around llama.cpp forced scoring."""

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = forced_sequence_config(config)
        self.port = int(self.config.get("port") or _allocate_ports(1)[0])
        self.process: subprocess.Popen[Any] | None = None
        self.command = [
            sys.executable,
            "-m",
            "carnot.experiment_6837_three_family_output_free_compatibility",
            "--score-worker",
            "--model-path",
            str(self.model_spec["model_path"]),
            "--port",
            str(self.port),
        ]

    def start(self) -> JsonDict:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.model_spec.get("gpu", 0))
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise SequenceScoringError(f"score_worker_exited:{self.process.returncode}")
            try:
                with request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise SequenceScoringError("score_worker_health_timeout")
        return {
            "command": list(self.command),
            "pid": self.process.pid,
            "process_start": _utc_now(),
            "port": self.port,
            "gpu_uuid": self.model_spec.get("gpu_uuid"),
            "visible_devices": [self.model_spec.get("gpu")],
            "model_hash": self.model_spec.get("model_sha256"),
            "tokenizer_hash": dict(self.model_spec.get("tokenizer_receipt") or {}).get(
                "receipt_hash"
            ),
            "owned_by_task": True,
            "token_logprob_support": True,
        }

    def score(
        self, prompt_text: str, candidate_text: str, row_identity: Mapping[str, Any]
    ) -> JsonDict:
        body = json.dumps(
            {
                "prompt_text": prompt_text,
                "candidate_text": candidate_text,
                "row_identity": dict(row_identity),
            }
        ).encode("utf-8")
        req = request.Request(
            f"http://127.0.0.1:{self.port}/score",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    def close(self) -> JsonDict:
        if self.process is None:
            return {"action": "not_started", "bounded": True, "leak_free": True}
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return {"action": "already_exited", "bounded": True, "leak_free": True}
        try:
            self.process.wait(timeout=30)
            return {
                "action": "terminated",
                "bounded": True,
                "leak_free": True,
                "unrelated_process_kill_count_delta": 0,
            }
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
            return {
                "action": "force_killed",
                "bounded": True,
                "leak_free": self.process.poll() is not None,
                "unrelated_process_kill_count_delta": 0,
            }


def _run_worker(model_path: str, port: int) -> int:  # pragma: no cover - live model dependent.
    engine = _WorkerEngine(model_path, forced_sequence_config())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def do_POST(self) -> None:
            if self.path != "/score":
                self.send_response(404)
                self.end_headers()
                return
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            result = engine.score(
                str(payload["prompt_text"]),
                str(payload["candidate_text"]),
                dict(payload["row_identity"]),
            )
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", int(port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        gc.collect()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the requested Exp6837 command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--checkpoint-path", default=str(REPO_ROOT / CHECKPOINT_RELATIVE_PATH))
    parser.add_argument("--score-worker", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.score_worker:  # pragma: no cover - exercised only by live worker subprocesses.
        return _run_worker(args.model_path, args.port)
    if args.date != RUN_DATE:
        raise SequenceScoringError(f"run_date_mismatch:{args.date}")
    artifact = run(
        root=REPO_ROOT,
        result_path=Path(args.result_path),
        checkpoint_path=Path(args.checkpoint_path),
        write=True,
    )
    print(
        json.dumps({"result_path": args.result_path, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
