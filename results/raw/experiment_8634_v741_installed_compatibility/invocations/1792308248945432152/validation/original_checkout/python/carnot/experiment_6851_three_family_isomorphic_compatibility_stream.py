"""Measure fixed-sequence compatibility across three local GGUF families.

Spec refs: REQ-CONSTRAINT-6851 and SCENARIO-CONSTRAINT-6851-*.

Exact labels come from Exp6849. Model scores are measurements only. The live
path uses one owner-scoped llama.cpp process and one GPU lease at a time.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6849_typed_program_isomorphic_authority_audit import (
    ISOMORPHIC_TRANSFORMS,
    _candidate_text,
    _pair_source,
    _short_identity,
    _transformed_pair,
)
from carnot.experiment_6850_three_family_scoring_admission_canary import (
    FIXED_CANDIDATE,
    FIXED_PROMPT,
    _choose_free_ports,
    _cuda_token_scoring,
    _gpu_inventory,
    _gpu_snapshot,
    resolve_model_specs,
)
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6851_three_family_isomorphic_compatibility_stream.py"
)
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6851_three_family_isomorphic_compatibility_stream.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6851_three_family_isomorphic_compatibility_stream.py"
)
PROCESS_MODULE_RELATIVE_PATH = Path("python/carnot/inference/llama_cpp_process.py")
EXP6849_RELATIVE_PATH = Path(
    "results/experiment_6849_typed_program_isomorphic_authority_audit.json"
)
EXP6850_RELATIVE_PATH = Path("results/experiment_6850_three_family_scoring_admission_canary.json")
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6851_three_family_isomorphic_compatibility_stream.json"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6851_three_family_isomorphic_compatibility_stream.checkpoint.json"
)

SCHEMA = "carnot.experiment_6851.three_family_isomorphic_compatibility_stream.v1"
INFERENCE_SUBSTRATE = "live_local_llama_cpp_cuda_forced_sequence_scoring"
RUN_DATE = "20260901"
RANDOM_SEED = 6851
REPEAT_COUNT = 2
BATCH_SIZE = 7
LEASE_TTL_S = 900.0
HEALTH_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 300.0
DEFAULT_CONTEXT_LENGTH = 4096
ROUND_DIGITS = 8
BLOCKED_VERDICT = "complete_blocked_three_family_isomorphic_compatibility_stream"
EXPECTED_EXP6849_SHA256 = "sha256:3ad1e98df8e95d7c31cf6942d5b6d4c01ab68c7259c2cc88346623012eba6d3a"
EXPECTED_EXP6850_SHA256 = "sha256:0093104ffb1d4f3f22e7ab9369d56561d8d01fa84406faa89d8f0117ced6977d"

MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
TRANSFORM_KINDS = ("base", *ISOMORPHIC_TRANSFORMS)
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
    "reproducibility_checksum",
    "rows",
    "token_score_receipts",
    "base_isomorphic_join_manifest",
    "per_model_margin_summary",
    "per_atom_margin_summary",
    "per_transform_margin_summary",
    "control_margin_summary",
    "checkpoint_manifest",
    "teardown_receipts",
    "compatibility_stream_complete_score",
    "positive_margin_models",
    "isomorphic_consistency_rate",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Every top-level field states why it exists.",
    "preconditions_checked": "Frozen gates and live resources fail closed before scoring.",
    "inference_substrate": "The exact substrate separates scoring from answer generation.",
    "duration_s": "Wall time exposes skipped or implausibly short live work.",
    "model_specs": "The three mandated model identities cannot be substituted.",
    "models_used": "Only models with complete owned receipts count as used.",
    "model_artifact_hashes": "Exact model bytes stay bound to every scored row.",
    "tokenizer_receipts": "Native GGUF tokenizer hashes detect token-boundary drift.",
    "process_receipts": "Owner identity prevents cleanup of unrelated processes.",
    "accelerator_samples": "CUDA residency and release make live execution auditable.",
    "random_seed": "The fixed seed makes ordering and canaries reproducible.",
    "reproducibility_checksum": "One digest detects input, model, and row drift.",
    "rows": "Each row stores one model, semantic pair, transform, position, and repeat.",
    "token_score_receipts": "Raw prompt and candidate token scores support every margin.",
    "base_isomorphic_join_manifest": "Semantic identities join changed surface forms.",
    "per_model_margin_summary": "Models remain separate rather than exchangeable samples.",
    "per_atom_margin_summary": "Atom-family effects stay separate from joint effects.",
    "per_joint_margin_summary": "Joint and impossible cases retain their own summaries.",
    "per_transform_margin_summary": "Each isomorphic transform retains a separate effect.",
    "control_margin_summary": "Length, position, prompt, and transform controls remain visible.",
    "checkpoint_manifest": "Verified row hashes permit safe bounded-batch restart.",
    "teardown_receipts": "Exit and port release prevent cross-model contamination.",
    "compatibility_stream_complete_score": "Only receipt and row completeness set readiness.",
    "positive_margin_models": "Positive effects are reported without changing readiness.",
    "isomorphic_consistency_rate": "Per-model sign stability measures surface sensitivity.",
    "gate_check_summary": "The first failed gate retains expected and observed values.",
    "verifier_is_oracle": "Model scores never become exact label authority.",
    "verdict_class": "A closed vocabulary keeps completion and effect distinct.",
    "honest_verdict": "A complete_ prefix gives a terminal row-supported outcome.",
    "schema": "The schema name lets readers validate the artifact shape.",
    "experiment_id": "The experiment number prevents artifact confusion.",
    "run_date": "The fixed execution date binds the requested run.",
    "status": "The status gives a compact machine-readable terminal state.",
    "spec_refs": "Requirement anchors connect code, tests, and artifact.",
    "source_artifact_hashes": "Source hashes bind authority, admission, code, and tests.",
    "batch_canary_receipts": "A fresh unlabeled canary precedes every bounded batch.",
    "lease_receipts": "Kernel-backed owner leases isolate model phases.",
    "method_limits": "The artifact states what the scalar score cannot prove.",
    "result_path": "The declared path detects accidental publication elsewhere.",
}


class CompatibilityStreamError(RuntimeError):
    """A stable error for malformed scoring inputs or receipts."""


class ForcedSequenceScorer(Protocol):
    """The small scoring boundary shared by tests and the live worker."""

    def score(
        self,
        prompt_text: str,
        candidate_text: str,
        identity: Mapping[str, Any],
    ) -> JsonDict:
        """Return prompt tokens and candidate-token log-probabilities."""


def canonical_json(value: Any) -> str:
    """Serialize JSON consistently for hashes and checkpoint comparisons."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:  # pragma: no cover - live files include large GGUFs.
    """Hash a file in chunks so large model bytes need no single buffer."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:  # pragma: no cover - live file boundary.
    """Read one JSON object and reject arrays or scalar values."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CompatibilityStreamError(f"json_object_required:{path}")
    return dict(value)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON file atomically so restart never sees partial bytes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Return one exact gate comparison with no truthy coercion."""

    return {
        "check": str(check),
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [dict(row) for row in checks]
    failed = [row for row in rows if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "checks": rows,
        "passed": first is None,
        "failed_check": first.get("check") if first else None,
        "failed_checks": [row.get("check") for row in failed],
        "expected": first.get("expected") if first else True,
        "observed": first.get("observed") if first else True,
    }


def _model_hash_pairs(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(row.get("hf_id")): {
            "model_sha256": row.get("model_sha256"),
            "tokenizer_sha256": dict(row.get("tokenizer_receipt") or {}).get("tokenizer_sha256"),
        }
        for row in model_specs
    }


def evaluate_preconditions(
    *,
    authority: Mapping[str, Any],
    admission: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    observed_hashes: Mapping[str, Any],
    cuda_scoring: bool,
    free_ports: bool,
    lease_available: bool,
) -> JsonDict:
    """Compare every frozen gate and live admission observation."""

    expected_module_hash = (
        dict(dict(authority.get("source_artifact_hashes") or {}).get("implementation") or {})
        .get("module", {})
        .get("file_sha256")
    )
    expected_models = _model_hash_pairs(admission.get("model_specs") or [])
    observed_models = _model_hash_pairs(model_specs)
    checks = [
        gate_check(
            "typed_program_authority_ready_score",
            1,
            authority.get("typed_program_authority_ready_score"),
        ),
        gate_check(
            "isomorphic_fixture_ready_score",
            1,
            authority.get("isomorphic_fixture_ready_score"),
        ),
        gate_check(
            "three_family_scoring_admission_ready_score",
            1,
            admission.get("three_family_scoring_admission_ready_score"),
        ),
        gate_check(
            "unchanged_exp6849_artifact", EXPECTED_EXP6849_SHA256, observed_hashes.get("exp6849")
        ),
        gate_check(
            "unchanged_exp6850_artifact", EXPECTED_EXP6850_SHA256, observed_hashes.get("exp6850")
        ),
        gate_check(
            "unchanged_exp6849_transform_implementation",
            expected_module_hash,
            observed_hashes.get("exp6849_module"),
        ),
        gate_check(
            "all_three_exact_model_specs",
            list(MODEL_SPECS),
            [row.get("hf_id") for row in model_specs],
        ),
        gate_check(
            "unchanged_model_and_tokenizer_hashes",
            expected_models,
            observed_models,
        ),
        gate_check("live_cuda_token_scoring", True, bool(cuda_scoring)),
        gate_check("free_task_ports", True, bool(free_ports)),
        gate_check("task_owned_lease", True, bool(lease_available)),
    ]
    blocked = [str(row["check"]) for row in checks if row["passed"] is not True]
    return {
        "schema": SCHEMA + ".preconditions",
        "preconditions_ready": not blocked,
        "checks": checks,
        "blocked_reasons": blocked,
        "expected_model_and_tokenizer_hashes": expected_models,
        "observed_model_and_tokenizer_hashes": observed_models,
        "accelerator_samples": [],
    }


def _ordered_candidates(pair: Mapping[str, Any]) -> list[JsonDict]:
    by_id = {str(row["candidate_id"]): dict(row) for row in pair.get("candidates") or []}
    return [by_id[str(candidate_id)] for candidate_id in pair.get("candidate_order") or []]


def materialize_score_inputs(
    authority: Mapping[str, Any], *, repeats: int = REPEAT_COUNT
) -> list[JsonDict]:
    """Rebuild base and transformed surfaces from the hash-bound Exp6849 code."""

    if int(repeats) <= 0:
        raise CompatibilityStreamError("repeat_count_must_be_positive")
    pairs = [dict(row) for row in authority.get("sanitized_candidate_pair_manifest") or []]
    transforms = {
        (str(row.get("pair_id")), str(row.get("transform_kind"))): dict(row)
        for row in authority.get("isomorphic_transform_manifest") or []
    }
    if not pairs:
        raise CompatibilityStreamError("sanitized_candidate_pairs_missing")
    inputs: list[JsonDict] = []
    for pair in pairs:
        base_source = _pair_source(pair)
        obligation_count = len(base_source.get("obligations") or [])
        for transform_kind in TRANSFORM_KINDS:
            if transform_kind == "base":
                transform_id = _short_identity("transform-6849", [pair["pair_id"], "base"])
                prompt_text = str(dict(pair["raw_sequence_inputs"])["prompt_text"])
                surface_candidates = [
                    {
                        **candidate,
                        "candidate_text": str(
                            dict(candidate["raw_sequence_inputs"])["candidate_text"]
                        ),
                    }
                    for candidate in _ordered_candidates(pair)
                ]
            else:
                manifest = transforms.get((str(pair["pair_id"]), transform_kind))
                if manifest is None or manifest.get("labels_preserved") is not True:
                    raise CompatibilityStreamError(
                        f"qualified_transform_missing:{pair.get('pair_id')}:{transform_kind}"
                    )
                transform_id = str(manifest["transform_id"])
                _, transformed, prompt_text = _transformed_pair(pair, transform_kind, transform_id)
                surface_candidates = [
                    {**dict(candidate), "candidate_text": _candidate_text(payload)}
                    for candidate, payload in transformed
                ]
            labels = [candidate.get("exact_label") for candidate in surface_candidates]
            if labels.count(True) != 1 or labels.count(False) != 1:
                raise CompatibilityStreamError(
                    f"one_compatible_one_violation_required:{pair.get('pair_id')}:{transform_kind}"
                )
            label_position = labels.index(True)
            surface_identity = sha256_text(
                canonical_json(
                    {
                        "prompt_text": prompt_text,
                        "candidates": [
                            {
                                "candidate_id": row.get("candidate_id"),
                                "candidate_text": row["candidate_text"],
                                "exact_label": row.get("exact_label"),
                            }
                            for row in surface_candidates
                        ],
                    }
                )
            )
            for repeat in range(int(repeats)):
                identity_payload = {
                    "semantic_pair_identity": pair["semantic_identity"],
                    "transform_id": transform_id,
                    "compatible_label_position": label_position,
                    "repeat": repeat,
                }
                inputs.append(
                    {
                        "schema": SCHEMA + ".score_input",
                        "score_input_identity": sha256_text(canonical_json(identity_payload)),
                        "pair_id": pair["pair_id"],
                        "source_pair_id": pair.get("source_pair_id"),
                        "semantic_pair_identity": pair["semantic_identity"],
                        "transform_id": transform_id,
                        "transform_kind": transform_kind,
                        "surface_identity": surface_identity,
                        "obligation_family": pair.get("case_kind"),
                        "obligation_count": obligation_count,
                        "prompt_text": prompt_text,
                        "prompt_text_sha256": sha256_text(prompt_text),
                        "prompt_text_length": len(prompt_text.encode("utf-8")),
                        "candidates": surface_candidates,
                        "compatible_label_position": label_position,
                        "repeat": repeat,
                    }
                )
    return inputs


def base_isomorphic_join_manifest(score_inputs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Join each surface cell to its base only through semantic identity."""

    first_repeat = [row for row in score_inputs if int(row.get("repeat", -1)) == 0]
    base_by_semantic = {
        str(row["semantic_pair_identity"]): str(row["surface_identity"])
        for row in first_repeat
        if row.get("transform_kind") == "base"
    }
    return [
        {
            "pair_id": row.get("pair_id"),
            "semantic_pair_identity": row.get("semantic_pair_identity"),
            "join_key": row.get("semantic_pair_identity"),
            "base_surface_identity": base_by_semantic.get(str(row.get("semantic_pair_identity"))),
            "surface_identity": row.get("surface_identity"),
            "transform_id": row.get("transform_id"),
            "transform_kind": row.get("transform_kind"),
            "compatible_label_position": row.get("compatible_label_position"),
        }
        for row in first_repeat
    ]


def _candidate_score_receipt(
    *,
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    prompt_text: str,
) -> JsonDict:
    prompt_tokens = [int(token) for token in receipt.get("prompt_token_ids") or []]
    candidate_tokens = [int(token) for token in receipt.get("candidate_token_ids") or []]
    token_logprobs = [float(value) for value in receipt.get("token_logprobs") or []]
    if not prompt_tokens:
        raise CompatibilityStreamError("prompt_tokens_missing")
    if not candidate_tokens:
        raise CompatibilityStreamError("candidate_tokens_missing")
    if len(candidate_tokens) != len(token_logprobs):
        raise CompatibilityStreamError("candidate_token_logprob_alignment")
    if not all(math.isfinite(value) for value in token_logprobs):
        raise CompatibilityStreamError("candidate_token_logprob_non_finite")
    raw_receipt = receipt.get("raw_receipt")
    if not isinstance(raw_receipt, Mapping) or not raw_receipt:
        raise CompatibilityStreamError("raw_token_score_receipt_missing")
    conditional = round(sum(token_logprobs), ROUND_DIGITS)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_semantic_identity": candidate.get("semantic_identity"),
        "exact_label": candidate.get("exact_label"),
        "prompt_text_sha256": sha256_text(prompt_text),
        "candidate_text_sha256": sha256_text(str(candidate.get("candidate_text") or "")),
        "prompt_token_ids": prompt_tokens,
        "candidate_token_ids": candidate_tokens,
        "token_logprobs": [round(value, ROUND_DIGITS) for value in token_logprobs],
        "conditional_log_likelihood": conditional,
        "mean_token_log_likelihood": round(conditional / len(candidate_tokens), ROUND_DIGITS),
        "sequence_energy": round(-conditional, ROUND_DIGITS),
        "raw_receipt": dict(raw_receipt),
    }


def score_input_row(
    *,
    score_input: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    scorer: ForcedSequenceScorer,
) -> JsonDict:
    """Score one semantic pair and compute prompt-masked scalar margins."""

    candidates = [dict(row) for row in score_input.get("candidates") or []]
    scored: list[JsonDict] = []
    for candidate in candidates:
        raw = scorer.score(
            str(score_input["prompt_text"]),
            str(candidate["candidate_text"]),
            {
                "score_input_identity": score_input["score_input_identity"],
                "candidate_id": candidate.get("candidate_id"),
                "exact_label": candidate.get("exact_label"),
            },
        )
        scored.append(
            _candidate_score_receipt(
                receipt=raw,
                candidate=candidate,
                prompt_text=str(score_input["prompt_text"]),
            )
        )
    compatible = [row for row in scored if row.get("exact_label") is True]
    violations = [row for row in scored if row.get("exact_label") is False]
    if len(compatible) != 1 or len(violations) != 1:
        raise CompatibilityStreamError("scored_exact_labels_invalid")
    if compatible[0]["prompt_token_ids"] != violations[0]["prompt_token_ids"]:
        raise CompatibilityStreamError("paired_prompt_token_alignment")
    comp = compatible[0]
    viol = violations[0]
    sum_margin = round(
        float(comp["conditional_log_likelihood"]) - float(viol["conditional_log_likelihood"]),
        ROUND_DIGITS,
    )
    mean_margin = round(
        float(comp["mean_token_log_likelihood"]) - float(viol["mean_token_log_likelihood"]),
        ROUND_DIGITS,
    )
    identity = f"{model_spec.get('hf_id')}::{score_input['score_input_identity']}"
    row: JsonDict = {
        "schema": SCHEMA + ".row",
        "row_identity": identity,
        "model_hf_id": model_spec.get("hf_id"),
        "model_family": model_spec.get("family"),
        "model_hash": model_spec.get("model_sha256"),
        "tokenizer_hash": dict(model_spec.get("tokenizer_receipt") or {}).get("tokenizer_sha256"),
        "pair_id": score_input.get("pair_id"),
        "source_pair_id": score_input.get("source_pair_id"),
        "semantic_pair_identity": score_input.get("semantic_pair_identity"),
        "surface_identity": score_input.get("surface_identity"),
        "transform_id": score_input.get("transform_id"),
        "transform_kind": score_input.get("transform_kind"),
        "obligation_family": score_input.get("obligation_family"),
        "obligation_count": score_input.get("obligation_count"),
        "repeat": score_input.get("repeat"),
        "compatible_label_position": score_input.get("compatible_label_position"),
        "exact_labels": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "exact_label": candidate.get("exact_label"),
            }
            for candidate in candidates
        ],
        "prompt_text_sha256": score_input.get("prompt_text_sha256"),
        "prompt_token_count": len(comp["prompt_token_ids"]),
        "compatible_candidate_length": len(comp["candidate_token_ids"]),
        "violation_candidate_length": len(viol["candidate_token_ids"]),
        "candidate_length_delta": len(comp["candidate_token_ids"])
        - len(viol["candidate_token_ids"]),
        "compatible": comp,
        "violation": viol,
        "sum_log_likelihood_margin": sum_margin,
        "mean_token_log_likelihood_margin": mean_margin,
        "scalar_compatibility_margin": mean_margin,
        "energy_margin": round(-sum_margin, ROUND_DIGITS),
        "lower_compatible_energy": sum_margin > 0,
        "prompt_masked": True,
        "no_sampling": True,
        "no_generation": True,
        "no_grammar": True,
        "no_retry_repair": True,
        "no_answer_feedback": True,
        "row_hash": "",
    }
    row["row_hash"] = row_hash(row)
    return row


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one scored row while excluding its self-referential digest."""

    unsigned = dict(row)
    unsigned["row_hash"] = ""
    return sha256_text(canonical_json(unsigned))


def canary_hash(canary: Mapping[str, Any]) -> str:
    """Hash one canary while excluding its self-referential digest."""

    return sha256_text(
        canonical_json({key: value for key, value in canary.items() if key != "canary_hash"})
    )


def _batch_receipt_errors(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    lease = receipt.get("lease_revalidation")
    lease = lease if isinstance(lease, Mapping) else {}
    if lease.get("owner_verified") is not True:
        errors.append("lease_revalidation")
    canary = receipt.get("canary")
    canary = canary if isinstance(canary, Mapping) else {}
    tokens = canary.get("candidate_token_ids")
    logprobs = canary.get("token_logprobs")
    if not isinstance(tokens, list) or not tokens:
        errors.append("canary_candidate_tokens")
    if (
        not isinstance(logprobs, list)
        or not logprobs
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in logprobs
        )
    ):
        errors.append("canary_token_logprobs")
    if isinstance(tokens, list) and isinstance(logprobs, list) and len(tokens) != len(logprobs):
        errors.append("canary_token_alignment")
    if canary.get("scientific_label", "missing") is not None:
        errors.append("canary_scientific_label")
    if canary.get("supports_margin_claim") is not False:
        errors.append("canary_margin_claim")
    if canary.get("canary_hash") != canary_hash(canary):
        errors.append("canary_hash")
    return errors


def checkpoint_input_checksum(
    *,
    source_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    score_inputs: Sequence[Mapping[str, Any]],
) -> str:
    """Bind restart state to sources, models, tokenizers, and input identities."""

    return sha256_text(
        canonical_json(
            {
                "source_hashes": source_hashes,
                "models": _model_hash_pairs(model_specs),
                "score_input_identities": [row.get("score_input_identity") for row in score_inputs],
                "random_seed": RANDOM_SEED,
                "batch_size": BATCH_SIZE,
            }
        )
    )


def build_checkpoint_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_row_count: int,
    input_checksum: str,
    batch_receipts: Sequence[Mapping[str, Any]] = (),
    process_receipts: Sequence[Mapping[str, Any]] = (),
    teardown_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Store complete rows and lifecycle receipts after a bounded batch."""

    stored = [deepcopy(dict(row)) for row in rows]
    row_hashes = {str(row.get("row_identity")): row.get("row_hash") for row in stored}
    checkpoint: JsonDict = {
        "schema": SCHEMA + ".checkpoint",
        "input_checksum": str(input_checksum),
        "expected_row_count": int(expected_row_count),
        "complete_row_count": len(stored),
        "missing_row_count": max(0, int(expected_row_count) - len(stored)),
        "rows": stored,
        "row_hashes": row_hashes,
        "batch_receipts": [deepcopy(dict(row)) for row in batch_receipts],
        "process_receipts": [deepcopy(dict(row)) for row in process_receipts],
        "teardown_receipts": [deepcopy(dict(row)) for row in teardown_receipts],
        "checkpoint_hash": "",
    }
    checkpoint["checkpoint_hash"] = sha256_text(
        canonical_json(
            {
                "input_checksum": checkpoint["input_checksum"],
                "expected_row_count": checkpoint["expected_row_count"],
                "row_hashes": [row.get("row_hash") for row in stored],
                "batch_receipts": checkpoint["batch_receipts"],
                "process_receipts": checkpoint["process_receipts"],
                "teardown_receipts": checkpoint["teardown_receipts"],
            }
        )
    )
    return checkpoint


def write_checkpoint(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write checkpoint state through the shared atomic JSON boundary."""

    write_json_atomic(path, manifest)


def _read_checkpoint(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CompatibilityStreamError("checkpoint_object_required")
    return dict(value)


def verified_checkpoint_rows(path: str | Path, *, input_checksum: str) -> dict[str, JsonDict]:
    """Verify restart input and row hashes before any identity is skipped."""

    checkpoint = _read_checkpoint(Path(path))
    if not checkpoint:
        return {}
    if checkpoint.get("input_checksum") != input_checksum:
        raise CompatibilityStreamError("checkpoint_input_checksum_mismatch")
    rows = [dict(row) for row in checkpoint.get("rows") or []]
    identities = [str(row.get("row_identity")) for row in rows]
    if len(identities) != len(set(identities)):
        raise CompatibilityStreamError("checkpoint_duplicate_row_identity")
    hashes = dict(checkpoint.get("row_hashes") or {})
    for row in rows:
        identity = str(row.get("row_identity"))
        if row_hash(row) != row.get("row_hash") or hashes.get(identity) != row.get("row_hash"):
            raise CompatibilityStreamError(f"checkpoint_row_hash_mismatch:{identity}")
    expected_hash = build_checkpoint_manifest(
        rows,
        expected_row_count=int(checkpoint.get("expected_row_count", len(rows))),
        input_checksum=input_checksum,
        batch_receipts=checkpoint.get("batch_receipts") or [],
        process_receipts=checkpoint.get("process_receipts") or [],
        teardown_receipts=checkpoint.get("teardown_receipts") or [],
    )["checkpoint_hash"]
    if checkpoint.get("checkpoint_hash") != expected_hash:
        raise CompatibilityStreamError("checkpoint_manifest_hash_mismatch")
    return {identity: row for identity, row in zip(identities, rows, strict=True)}


def _chunks(rows: Sequence[Mapping[str, Any]], size: int) -> list[list[Mapping[str, Any]]]:
    return [list(rows[index : index + size]) for index in range(0, len(rows), size)]


def score_missing_batches(
    *,
    model_spec: Mapping[str, Any],
    score_inputs: Sequence[Mapping[str, Any]],
    scorer: ForcedSequenceScorer,
    checkpoint_path: str | Path,
    input_checksum: str,
    expected_total_row_count: int,
    batch_size: int,
    before_batch: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
) -> JsonDict:
    """Score only missing identities and checkpoint each bounded batch."""

    if int(batch_size) <= 0:
        raise CompatibilityStreamError("batch_size_must_be_positive")
    path = Path(checkpoint_path)
    checkpoint = _read_checkpoint(path)
    existing = verified_checkpoint_rows(path, input_checksum=input_checksum)
    expected_ids = [
        f"{model_spec.get('hf_id')}::{row['score_input_identity']}" for row in score_inputs
    ]
    resumed = {identity: existing[identity] for identity in expected_ids if identity in existing}
    missing = [
        row
        for row, identity in zip(score_inputs, expected_ids, strict=True)
        if identity not in resumed
    ]
    batch_receipts = [dict(row) for row in checkpoint.get("batch_receipts") or []]
    prior_batch_indexes = [
        int(row["batch_index"])
        for row in batch_receipts
        if row.get("model_hf_id") == model_spec.get("hf_id")
    ]
    next_batch_index = max(prior_batch_indexes, default=-1) + 1
    all_rows = dict(existing)
    new_rows: list[JsonDict] = []
    for batch_index, batch in enumerate(_chunks(missing, int(batch_size)), start=next_batch_index):
        receipt = dict(before_batch(model_spec, batch_index))
        errors = _batch_receipt_errors(receipt)
        if errors:
            raise CompatibilityStreamError(f"batch_precondition_failed:{','.join(errors)}")
        batch_receipts.append(receipt)
        for score_input in batch:
            row = score_input_row(score_input=score_input, model_spec=model_spec, scorer=scorer)
            all_rows[str(row["row_identity"])] = row
            new_rows.append(row)
        manifest = build_checkpoint_manifest(
            list(all_rows.values()),
            expected_row_count=expected_total_row_count,
            input_checksum=input_checksum,
            batch_receipts=batch_receipts,
            process_receipts=checkpoint.get("process_receipts") or [],
            teardown_receipts=checkpoint.get("teardown_receipts") or [],
        )
        write_checkpoint(path, manifest)
        checkpoint = manifest
    ordered_rows = [all_rows[identity] for identity in expected_ids if identity in all_rows]
    return {
        "model_hf_id": model_spec.get("hf_id"),
        "rows": ordered_rows,
        "resumed_row_count": len(resumed),
        "new_row_count": len(new_rows),
        "batch_receipts": batch_receipts,
        "checkpoint_manifest": checkpoint
        or build_checkpoint_manifest(
            list(all_rows.values()),
            expected_row_count=expected_total_row_count,
            input_checksum=input_checksum,
        ),
    }


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), ROUND_DIGITS) if values else None


def _margin_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    scalar = [float(row["scalar_compatibility_margin"]) for row in rows]
    summed = [float(row["sum_log_likelihood_margin"]) for row in rows]
    return {
        "row_count": len(rows),
        "mean_scalar_compatibility_margin": _mean(scalar),
        "mean_sum_log_likelihood_margin": _mean(summed),
        "positive_scalar_margin_count": sum(value > 0 for value in scalar),
        "zero_scalar_margin_count": sum(value == 0 for value in scalar),
        "negative_scalar_margin_count": sum(value < 0 for value in scalar),
        "min_scalar_compatibility_margin": round(min(scalar), ROUND_DIGITS) if scalar else None,
        "max_scalar_compatibility_margin": round(max(scalar), ROUND_DIGITS) if scalar else None,
    }


def _grouped_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[JsonDict]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output: list[JsonDict] = []
    for key, group in sorted(grouped.items(), key=lambda item: canonical_json(item[0])):
        output.append({**dict(zip(fields, key, strict=True)), **_margin_summary(group)})
    return output


def _token_score_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    output: list[JsonDict] = []
    for row in rows:
        for label in ("compatible", "violation"):
            output.append(
                {
                    "row_identity": row.get("row_identity"),
                    "model_hf_id": row.get("model_hf_id"),
                    "pair_id": row.get("pair_id"),
                    "semantic_pair_identity": row.get("semantic_pair_identity"),
                    "transform_id": row.get("transform_id"),
                    "transform_kind": row.get("transform_kind"),
                    "repeat": row.get("repeat"),
                    "label": label,
                    **deepcopy(dict(row.get(label) or {})),
                }
            )
    return output


def _positive_margin_models(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    positive: list[str] = []
    for model_id in MODEL_SPECS:
        values = [
            float(row["scalar_compatibility_margin"])
            for row in rows
            if row.get("model_hf_id") == model_id
        ]
        if values and float(_mean(values) or 0.0) > 0:
            positive.append(model_id)
    return positive


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _isomorphic_consistency(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_model: list[JsonDict] = []
    for model_id in MODEL_SPECS:
        model_rows = [row for row in rows if row.get("model_hf_id") == model_id]
        grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in model_rows:
            grouped[(row.get("semantic_pair_identity"), row.get("repeat"))].append(row)
        consistent = 0
        comparisons = 0
        for group in grouped.values():
            base = next((row for row in group if row.get("transform_kind") == "base"), None)
            if base is None:
                continue
            base_sign = _sign(float(base["scalar_compatibility_margin"]))
            for row in group:
                if row.get("transform_kind") == "base":
                    continue
                comparisons += 1
                consistent += _sign(float(row["scalar_compatibility_margin"])) == base_sign
        per_model.append(
            {
                "model_hf_id": model_id,
                "comparison_count": comparisons,
                "consistent_count": consistent,
                "consistency_rate": round(consistent / comparisons, ROUND_DIGITS)
                if comparisons
                else None,
            }
        )
    return {"pooled_models": False, "per_model": per_model}


def _process_receipts_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    required = (
        "pid",
        "start_time_ticks",
        "command_hash",
        "process_group_id",
        "owner_pid",
        "owner_start_time_ticks",
        "ownership_token_digest",
        "port",
        "gpu_uuid",
        "visible_devices",
        "model_hash",
        "tokenizer_hash",
    )
    identities = [
        (row.get("pid"), row.get("start_time_ticks"), row.get("command_hash")) for row in receipts
    ]
    return (
        [row.get("hf_id") for row in receipts] == list(MODEL_SPECS)
        and len(identities) == len(set(identities))
        and all(row.get("owned_by_task") is True for row in receipts)
        and all(all(row.get(field) not in {None, ""} for field in required) for row in receipts)
    )


def _teardowns_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return [row.get("hf_id") for row in receipts] == list(MODEL_SPECS) and all(
        row.get("ownership_verified") is True
        and row.get("process_exit_confirmed") is True
        and row.get("process_reaped") is True
        and row.get("port_release_confirmed") is True
        and row.get("leak_free") is True
        and row.get("unrelated_process_kill_count_delta") == 0
        for row in receipts
    )


def _lease_receipts_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    if [row.get("hf_id") for row in receipts] != list(MODEL_SPECS):
        return False
    for row in receipts:
        owner = row.get("owner")
        release = row.get("release")
        if not isinstance(owner, Mapping) or not isinstance(release, Mapping):
            return False
        if (
            row.get("lease_valid") is not True
            or owner.get("token_opaque") is not True
            or release.get("released") is not True
            or release.get("phase") != "terminal_complete"
            or owner.get("device_uuid") != release.get("device_uuid")
        ):
            return False
    return True


def _batch_receipts_complete(
    receipts: Sequence[Mapping[str, Any]], *, expected_row_count: int
) -> bool:
    rows_per_model = math.ceil(expected_row_count / len(MODEL_SPECS))
    expected_batches_per_model = math.ceil(rows_per_model / BATCH_SIZE)
    expected_identities = {
        (model_id, batch_index)
        for model_id in MODEL_SPECS
        for batch_index in range(expected_batches_per_model)
    }
    observed_identities = {(row.get("model_hf_id"), row.get("batch_index")) for row in receipts}
    return (
        len(receipts) == len(expected_identities)
        and observed_identities == expected_identities
        and not any(_batch_receipt_errors(row) for row in receipts)
    )


def _rows_have_complete_receipts(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if row_hash(row) != row.get("row_hash"):
            return False
        for side in ("compatible", "violation"):
            receipt = row.get(side)
            if not isinstance(receipt, Mapping) or not receipt.get("raw_receipt"):
                return False
            tokens = receipt.get("candidate_token_ids")
            logprobs = receipt.get("token_logprobs")
            if not isinstance(tokens, list) or not isinstance(logprobs, list):
                return False
            if not tokens or len(tokens) != len(logprobs):
                return False
    return True


def _readiness_checks(
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_row_count: int,
    process_receipts: Sequence[Mapping[str, Any]],
    batch_receipts: Sequence[Mapping[str, Any]],
    teardown_receipts: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    lease_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        gate_check("row_count_complete", expected_row_count, len(rows)),
        gate_check(
            "row_identities_unique",
            len(rows),
            len({row.get("row_identity") for row in rows}),
        ),
        gate_check(
            "all_models_have_rows",
            list(MODEL_SPECS),
            list(dict.fromkeys(row.get("model_hf_id") for row in rows)),
        ),
        gate_check("owned_process_receipts", True, _process_receipts_complete(process_receipts)),
        gate_check(
            "fresh_batch_canaries",
            True,
            _batch_receipts_complete(batch_receipts, expected_row_count=expected_row_count),
        ),
        gate_check("raw_token_receipts_complete", True, _rows_have_complete_receipts(rows)),
        gate_check(
            "checkpoint_complete", expected_row_count, checkpoint_manifest.get("complete_row_count")
        ),
        gate_check("clean_owned_teardown", True, _teardowns_complete(teardown_receipts)),
        gate_check("task_owned_lease_receipts", True, _lease_receipts_complete(lease_receipts)),
    ]


def _model_artifact_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(row.get("hf_id")): {
            "path": row.get("model_path"),
            "sha256": row.get("model_sha256"),
            "family": row.get("family"),
        }
        for row in model_specs
    }


def _tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {"hf_id": row.get("hf_id"), **deepcopy(dict(row.get("tokenizer_receipt") or {}))}
        for row in model_specs
    ]


def _method_limits() -> JsonDict:
    return {
        "fits_probe": False,
        "uses_llm_judge": False,
        "generates_answer": False,
        "uses_sampling": False,
        "uses_grammar": False,
        "repairs_output": False,
        "uses_answer_feedback": False,
        "lower_compatible_energy_is_exact_truth": False,
        "exact_label_authority": "Exp6849 exact checker",
        "primary_scalar": "mean_token_log_likelihood_margin",
        "summed_margin_retained_as_length_control": True,
    }


def _reproducibility_checksum(
    *,
    source_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return sha256_text(
        canonical_json(
            {
                "source_hashes": source_hashes,
                "models": _model_hash_pairs(model_specs),
                "row_hashes": [row.get("row_hash") for row in rows],
                "random_seed": RANDOM_SEED,
                "inference_substrate": INFERENCE_SUBSTRATE,
            }
        )
    )


def _attach_field_principles(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves Exp6851 audit evidence.")
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    return artifact


def _base_artifact(
    *,
    duration_s: float,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6851,
        "run_date": RUN_DATE,
        "status": BLOCKED_VERDICT,
        "spec_refs": [
            "REQ-CONSTRAINT-6851",
            "SCENARIO-CONSTRAINT-6851-PRECONDITIONS",
            "SCENARIO-CONSTRAINT-6851-FORCED-SCORING",
            "SCENARIO-CONSTRAINT-6851-TOKEN-ALIGNMENT",
            "SCENARIO-CONSTRAINT-6851-ISOMORPHIC-PAIRING",
            "SCENARIO-CONSTRAINT-6851-LABEL-POSITION",
            "SCENARIO-CONSTRAINT-6851-CHECKPOINT-RESTART",
            "SCENARIO-CONSTRAINT-6851-PROCESS-OWNERSHIP",
            "SCENARIO-CONSTRAINT-6851-RAW-RECEIPTS",
        ],
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [],
        "model_artifact_hashes": _model_artifact_hashes(model_specs),
        "tokenizer_receipts": _tokenizer_receipts(model_specs),
        "process_receipts": [],
        "accelerator_samples": deepcopy(list(preconditions.get("accelerator_samples") or [])),
        "random_seed": RANDOM_SEED,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "reproducibility_checksum": _reproducibility_checksum(
            source_hashes=source_hashes, model_specs=model_specs, rows=[]
        ),
        "rows": [],
        "token_score_receipts": [],
        "base_isomorphic_join_manifest": [],
        "per_model_margin_summary": [],
        "per_atom_margin_summary": [],
        "per_joint_margin_summary": [],
        "per_transform_margin_summary": [],
        "control_margin_summary": [],
        "batch_canary_receipts": [],
        "lease_receipts": [],
        "checkpoint_manifest": build_checkpoint_manifest(
            [], expected_row_count=0, input_checksum="blocked"
        ),
        "teardown_receipts": [],
        "method_limits": _method_limits(),
        "compatibility_stream_complete_score": 0,
        "positive_margin_models": [],
        "isomorphic_consistency_rate": {"pooled_models": False, "per_model": []},
        "gate_check_summary": _gate_summary(preconditions.get("checks") or []),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    return artifact


def build_blocked_artifact(
    *,
    duration_s: float,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build the complete blocked shape without any scientific margin row."""

    return _attach_field_principles(
        _base_artifact(
            duration_s=duration_s,
            preconditions=preconditions,
            model_specs=model_specs,
            source_hashes=source_hashes,
        )
    )


def build_complete_artifact(
    *,
    duration_s: float,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    score_inputs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    process_receipts: Sequence[Mapping[str, Any]],
    batch_receipts: Sequence[Mapping[str, Any]],
    teardown_receipts: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    expected_row_count: int,
    lease_receipts: Sequence[Mapping[str, Any]] = (),
    accelerator_samples: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build a terminal artifact whose readiness ignores margin direction."""

    readiness = _readiness_checks(
        rows=rows,
        expected_row_count=expected_row_count,
        process_receipts=process_receipts,
        batch_receipts=batch_receipts,
        teardown_receipts=teardown_receipts,
        checkpoint_manifest=checkpoint_manifest,
        lease_receipts=lease_receipts,
    )
    gate_summary = _gate_summary(readiness)
    ready = gate_summary["passed"] is True
    positives = _positive_margin_models(rows)
    if not ready:
        verdict_class = "disqualified"
    elif len(positives) == len(MODEL_SPECS):
        verdict_class = "positive"
    else:
        verdict_class = "null"
    verdicts = {
        "positive": "complete_positive_three_family_isomorphic_compatibility_stream",
        "null": "complete_null_three_family_isomorphic_compatibility_stream",
        "partial": "complete_partial_three_family_isomorphic_compatibility_stream",
        "disqualified": "complete_disqualified_three_family_isomorphic_compatibility_stream",
    }
    artifact = _base_artifact(
        duration_s=duration_s,
        preconditions=preconditions,
        model_specs=model_specs,
        source_hashes=source_hashes,
    )
    atom_rows = [
        row
        for row in rows
        if row.get("obligation_family") in {"atom_contradiction", "atom_omission"}
    ]
    joint_rows = [
        row for row in rows if row.get("obligation_family") in {"joint_violation", "impossible_set"}
    ]
    artifact.update(
        {
            "status": "complete" if ready else "disqualified",
            "models_used": list(dict.fromkeys(row.get("model_hf_id") for row in rows)),
            "process_receipts": [deepcopy(dict(row)) for row in process_receipts],
            "accelerator_samples": [
                *deepcopy(list(preconditions.get("accelerator_samples") or [])),
                *deepcopy(list(accelerator_samples)),
            ],
            "reproducibility_checksum": _reproducibility_checksum(
                source_hashes=source_hashes, model_specs=model_specs, rows=rows
            ),
            "rows": [deepcopy(dict(row)) for row in rows],
            "token_score_receipts": _token_score_receipts(rows),
            "base_isomorphic_join_manifest": base_isomorphic_join_manifest(score_inputs),
            "per_model_margin_summary": _grouped_summary(rows, ("model_hf_id", "model_family")),
            "per_atom_margin_summary": _grouped_summary(
                atom_rows,
                (
                    "model_hf_id",
                    "model_family",
                    "obligation_family",
                    "obligation_count",
                    "transform_kind",
                    "compatible_label_position",
                ),
            ),
            "per_joint_margin_summary": _grouped_summary(
                joint_rows,
                (
                    "model_hf_id",
                    "model_family",
                    "obligation_family",
                    "obligation_count",
                    "transform_kind",
                    "compatible_label_position",
                ),
            ),
            "per_transform_margin_summary": _grouped_summary(
                rows,
                (
                    "model_hf_id",
                    "model_family",
                    "transform_kind",
                    "compatible_label_position",
                ),
            ),
            "control_margin_summary": _grouped_summary(
                rows,
                (
                    "model_hf_id",
                    "model_family",
                    "obligation_family",
                    "obligation_count",
                    "transform_kind",
                    "prompt_token_count",
                    "compatible_candidate_length",
                    "violation_candidate_length",
                    "compatible_label_position",
                ),
            ),
            "batch_canary_receipts": [deepcopy(dict(row)) for row in batch_receipts],
            "lease_receipts": [deepcopy(dict(row)) for row in lease_receipts],
            "checkpoint_manifest": deepcopy(dict(checkpoint_manifest)),
            "teardown_receipts": [deepcopy(dict(row)) for row in teardown_receipts],
            "compatibility_stream_complete_score": int(ready),
            "positive_margin_models": positives,
            "isomorphic_consistency_rate": _isomorphic_consistency(rows),
            "gate_check_summary": gate_summary,
            "verdict_class": verdict_class,
            "honest_verdict": verdicts[verdict_class],
        }
    )
    return _attach_field_principles(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return every schema or evidence error without changing the artifact."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    output = [f"missing_field:{field}" for field in errors]
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        output.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        output.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        output.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        output.append("honest_verdict_not_complete_prefixed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        output.append("field_principles_incomplete")
    rows = artifact.get("rows")
    rows = rows if isinstance(rows, list) else []
    if any(row_hash(row) != row.get("row_hash") for row in rows if isinstance(row, Mapping)):
        output.append("row_hash_invalid")
    if artifact.get("verdict_class") == "blocked" and rows:
        output.append("blocked_artifact_has_rows")
    return output


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live file boundary.
    paths = {
        "exp6849": EXP6849_RELATIVE_PATH,
        "exp6850": EXP6850_RELATIVE_PATH,
        "module": MODULE_RELATIVE_PATH,
        "wrapper": WRAPPER_RELATIVE_PATH,
        "test": TEST_RELATIVE_PATH,
        "spec": SPEC_RELATIVE_PATH,
        "process_module": PROCESS_MODULE_RELATIVE_PATH,
        "exp6849_module": Path(
            "python/carnot/experiment_6849_typed_program_isomorphic_authority_audit.py"
        ),
    }
    return {
        key: {"path": str(path), "sha256": sha256_file(root / path)} for key, path in paths.items()
    }


def _port_free(port: int) -> bool:  # pragma: no cover - host state varies.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def _probe_lease(runtime_dir: Path, gpu: Mapping[str, Any]) -> bool:  # pragma: no cover
    lease = None
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=runtime_dir,
            task_id="exp6851-preflight",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model="exp6851-preflight-no-model",
            vram_before_mb=int(gpu.get("free_vram_mb", 0)),
            ttl_s=30.0,
        )
        lease.transition("terminal_blocked")
        return lease.release().get("released") is True
    except Exception:
        if lease is not None:
            lease.close()
        return False


def collect_live_preconditions(
    *, root: Path, runtime_dir: Path
) -> JsonDict:  # pragma: no cover - host, cache, and CUDA dependent.
    """Collect immutable upstream, tokenizer, CUDA, port, GPU, and lease gates."""

    authority = read_json(root / EXP6849_RELATIVE_PATH)
    admission = read_json(root / EXP6850_RELATIVE_PATH)
    model_specs = resolve_model_specs()
    source_hashes = _source_hashes(root)
    ports = _choose_free_ports(len(MODEL_SPECS))
    free_ports = len(ports) == len(MODEL_SPECS) and all(_port_free(port) for port in ports)
    cuda = _cuda_token_scoring()
    inventory = _gpu_inventory()
    required_mb = max(
        int(row.get("model_size_bytes", 0)) // (1024 * 1024) + 1024 for row in model_specs
    )
    eligible = [
        row
        for row in inventory
        if "RTX 3090" in str(row.get("name")) and int(row.get("free_vram_mb", 0)) >= required_mb
    ]
    gpu = max(eligible, key=lambda row: int(row["free_vram_mb"])) if eligible else {}
    lease_available = bool(gpu) and _probe_lease(runtime_dir / "leases", gpu)
    observed_hashes = {
        "exp6849": source_hashes["exp6849"]["sha256"],
        "exp6850": source_hashes["exp6850"]["sha256"],
        "exp6849_module": source_hashes["exp6849_module"]["sha256"],
    }
    checked = evaluate_preconditions(
        authority=authority,
        admission=admission,
        model_specs=model_specs,
        observed_hashes=observed_hashes,
        cuda_scoring=cuda.get("ok") is True,
        free_ports=free_ports,
        lease_available=lease_available,
    )
    checked.update(
        {
            "ports": ports,
            "eligible_gpu": gpu,
            "required_free_vram_mb": required_mb,
            "accelerator_samples": [{**row, "phase": "preflight"} for row in inventory],
            "cuda_token_scoring": cuda,
        }
    )
    for model in model_specs:
        model["gpu"] = gpu.get("index")
        model["gpu_uuid"] = gpu.get("gpu_uuid")
    return {
        "authority": authority,
        "admission": admission,
        "model_specs": model_specs,
        "source_hashes": source_hashes,
        "preconditions": checked,
        "ports": ports,
        "gpu": gpu,
    }


class _ScoringEngine:  # pragma: no cover - live llama.cpp CUDA path.
    """Load one GGUF and expose only conditional fixed-sequence scoring."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=DEFAULT_CONTEXT_LENGTH,
            n_batch=512,
            n_ubatch=128,
            seed=RANDOM_SEED,
            logits_all=True,
            verbose=True,
        )

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return [
            int(token)
            for token in self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False)
        ]

    @staticmethod
    def _logprob(logits: Any, token_id: int) -> float:
        import numpy as np

        values = np.asarray(logits, dtype=np.float64)
        maximum = float(values.max())
        return float(values[int(token_id)] - maximum - np.log(np.exp(values - maximum).sum()))

    def score(self, prompt_text: str, candidate_text: str) -> JsonDict:
        prompt = self._tokenize(prompt_text, add_bos=True)
        candidate = self._tokenize(candidate_text, add_bos=False)
        if not prompt or not candidate:
            raise CompatibilityStreamError("worker_tokens_missing")
        tokens = prompt + candidate
        if len(tokens) > DEFAULT_CONTEXT_LENGTH:
            raise CompatibilityStreamError(f"worker_context_overflow:{len(tokens)}")
        self.llm.reset()
        self.llm.eval(tokens)
        scores = self.llm.scores
        logprobs = [
            self._logprob(scores[index - 1], token_id)
            for index, token_id in enumerate(candidate, start=len(prompt))
        ]
        return {
            "prompt_token_ids": prompt,
            "candidate_token_ids": candidate,
            "token_logprobs": [round(value, ROUND_DIGITS) for value in logprobs],
            "conditional_log_likelihood": round(sum(logprobs), ROUND_DIGITS),
            "raw_receipt": {
                "forced_sequence": True,
                "prompt_masked": True,
                "sampling": False,
                "generation": False,
                "grammar": False,
                "retry_repair": False,
                "answer_feedback": False,
                "prompt_token_count": len(prompt),
                "candidate_token_count": len(candidate),
                "token_logprob_count": len(logprobs),
            },
        }


def _run_worker(model_path: str, port: int) -> int:  # pragma: no cover - live subprocess.
    engine = _ScoringEngine(model_path)

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
            result = engine.score(str(payload["prompt_text"]), str(payload["candidate_text"]))
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


class LiveProcessScorer:  # pragma: no cover - live model process.
    """Use the shared owner-scoped process wrapper for one scoring model."""

    def __init__(
        self,
        *,
        model_spec: Mapping[str, Any],
        port: int,
        runtime_dir: Path,
        gpu: Mapping[str, Any],
    ) -> None:
        self.model_spec = dict(model_spec)
        self.port = int(port)
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6851_three_family_isomorphic_compatibility_stream",
            "--score-worker",
            "--model-path",
            str(model_spec["model_path"]),
            "--port",
            str(port),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        family = str(model_spec.get("family"))
        self.process = OwnedLlamaCppProcess(
            command=command,
            port=port,
            env=env,
            log_path=runtime_dir / f"{family}.log",
            state_path=runtime_dir / f"{family}.owner.json",
        )

    def start(self) -> JsonDict:
        receipt = self.process.launch()
        health = self.process.wait_for_health(HEALTH_TIMEOUT_S)
        if health.get("ok") is not True:
            raise CompatibilityStreamError(f"worker_health_failed:{health.get('reason')}")
        receipt.update(
            {
                "hf_id": self.model_spec.get("hf_id"),
                "model_hash": self.model_spec.get("model_sha256"),
                "tokenizer_hash": dict(self.model_spec.get("tokenizer_receipt") or {}).get(
                    "tokenizer_sha256"
                ),
                "gpu_uuid": self.model_spec.get("gpu_uuid"),
                "visible_devices": str(self.model_spec.get("gpu")),
            }
        )
        return receipt

    def score(
        self,
        prompt_text: str,
        candidate_text: str,
        identity: Mapping[str, Any],
    ) -> JsonDict:
        response = self.process.post_json(
            "/score",
            {
                "prompt_text": prompt_text,
                "candidate_text": candidate_text,
                "identity": dict(identity),
            },
            REQUEST_TIMEOUT_S,
        )
        raw = dict(response.get("raw_receipt") or {})
        raw["identity"] = dict(identity)
        response["raw_receipt"] = raw
        return response

    def close(self) -> JsonDict:
        return self.process.cleanup()


def _finish_lease(
    lease: Any,
    *,
    complete: bool,
    teardown: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[JsonDict, str | None]:  # pragma: no cover - live lease state.
    try:
        phase = str(lease.document.get("phase"))
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            phase = "unloading"
        if phase == "unloading":
            lease.transition(
                "validating",
                vram_mb=int(after.get("owned_vram_mb", 0)),
                exit_code=0 if teardown.get("process_exit_confirmed") is True else 1,
                unload_observed=teardown.get("leak_free") is True,
            )
            phase = "validating"
        target = "terminal_complete" if complete and phase == "validating" else "terminal_blocked"
        if phase in {"preflight", "admitted", "loading", "validating"}:
            lease.transition(target)
        return lease.release(), None
    except Exception as exc:
        lease.close()
        return {}, f"{type(exc).__name__}: {exc}"


def run_live_model_phase(
    *,
    model_spec: Mapping[str, Any],
    score_inputs: Sequence[Mapping[str, Any]],
    port: int,
    gpu: Mapping[str, Any],
    runtime_dir: Path,
    checkpoint_path: Path,
    input_checksum: str,
    expected_total_row_count: int,
) -> JsonDict:  # pragma: no cover - live model, lease, and CUDA path.
    """Run one leased model process with a fresh canary before every batch."""

    before = _gpu_snapshot(gpu, phase="before")
    lease = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "leases",
        task_id=f"exp6851-{model_spec.get('family')}",
        device_uuid=str(gpu["gpu_uuid"]),
        expected_model=str(model_spec["hf_id"]),
        vram_before_mb=int(before.get("free_vram_mb", 0)),
        ttl_s=LEASE_TTL_S,
    )
    owner = lease.owner_receipt()
    scorer = LiveProcessScorer(
        model_spec=model_spec,
        port=port,
        runtime_dir=runtime_dir,
        gpu=gpu,
    )
    process_receipt: JsonDict = {}
    teardown: JsonDict = {}
    resident: JsonDict = {}
    phase_result: JsonDict = {}
    run_error: Exception | None = None
    try:
        lease.transition("admitted")
        lease.transition("loading")
        process_receipt = scorer.start()
        resident = _gpu_snapshot(gpu, phase="resident", owned_pid=int(process_receipt["pid"]))
        if resident.get("owned_cuda_residency") is not True:
            raise CompatibilityStreamError("owned_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("owned_vram_mb", 0)))
        lease.transition("inferencing")

        def before_batch(_model: Mapping[str, Any], batch_index: int) -> JsonDict:
            heartbeat = lease.heartbeat()
            raw = scorer.score(
                FIXED_PROMPT,
                FIXED_CANDIDATE,
                {"canary": True, "batch_index": batch_index, "exact_label": None},
            )
            canary = {
                "prompt_token_ids": raw.get("prompt_token_ids"),
                "candidate_token_ids": raw.get("candidate_token_ids"),
                "token_logprobs": raw.get("token_logprobs"),
                "conditional_log_likelihood": raw.get("conditional_log_likelihood"),
                "scientific_label": None,
                "supports_margin_claim": False,
                "canary_hash": "",
            }
            canary["canary_hash"] = canary_hash(canary)
            return {
                "model_hf_id": model_spec.get("hf_id"),
                "batch_index": batch_index,
                "lease_revalidation": heartbeat,
                "canary": canary,
            }

        phase_result = score_missing_batches(
            model_spec=model_spec,
            score_inputs=score_inputs,
            scorer=scorer,
            checkpoint_path=checkpoint_path,
            input_checksum=input_checksum,
            expected_total_row_count=expected_total_row_count,
            batch_size=BATCH_SIZE,
            before_batch=before_batch,
        )
    except Exception as exc:
        run_error = exc
    finally:
        teardown = scorer.close()
        after = _gpu_snapshot(gpu, phase="after", owned_pid=int(process_receipt.get("pid", 0) or 0))
        release, lease_error = _finish_lease(
            lease,
            complete=run_error is None and teardown.get("leak_free") is True,
            teardown=teardown,
            after=after,
        )
    if run_error is not None:
        raise CompatibilityStreamError(
            f"live_model_phase_failed:{type(run_error).__name__}:{run_error}"
        )
    if lease_error is not None:
        raise CompatibilityStreamError(f"lease_finish_failed:{lease_error}")
    lease_receipt = {
        "hf_id": model_spec.get("hf_id"),
        "owner": owner,
        "phase_history": deepcopy(lease.document.get("phase_history", [])),
        "release": release,
        "lease_valid": release.get("phase") == "terminal_complete",
    }
    return {
        **phase_result,
        "process_receipt": process_receipt,
        "teardown_receipt": {"hf_id": model_spec.get("hf_id"), **teardown},
        "lease_receipt": lease_receipt,
        "accelerator_samples": [before, resident, after],
    }


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
    runtime_dir: Path | None = None,
    write: bool = True,
) -> JsonDict:  # pragma: no cover - live orchestration.
    """Run the complete three-model stream or write one blocked artifact."""

    started = time.perf_counter()
    result = result_path or root / RESULT_RELATIVE_PATH
    checkpoint = checkpoint_path or root / CHECKPOINT_RELATIVE_PATH
    runtime = runtime_dir or Path(tempfile.gettempdir()) / "carnot-exp6851"
    live = collect_live_preconditions(root=root, runtime_dir=runtime)
    preconditions = live["preconditions"]
    model_specs = live["model_specs"]
    source_hashes = live["source_hashes"]
    if preconditions.get("preconditions_ready") is not True:
        artifact = build_blocked_artifact(
            duration_s=time.perf_counter() - started,
            preconditions=preconditions,
            model_specs=model_specs,
            source_hashes=source_hashes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact
    score_inputs = materialize_score_inputs(live["authority"])
    input_checksum = checkpoint_input_checksum(
        source_hashes=source_hashes,
        model_specs=model_specs,
        score_inputs=score_inputs,
    )
    expected_count = len(score_inputs) * len(model_specs)
    processes: list[JsonDict] = []
    teardowns: list[JsonDict] = []
    leases: list[JsonDict] = []
    accelerator_samples: list[JsonDict] = []
    try:
        for model, port in zip(model_specs, live["ports"], strict=True):
            phase = run_live_model_phase(
                model_spec=model,
                score_inputs=score_inputs,
                port=int(port),
                gpu=live["gpu"],
                runtime_dir=runtime,
                checkpoint_path=checkpoint,
                input_checksum=input_checksum,
                expected_total_row_count=expected_count,
            )
            processes.append(dict(phase["process_receipt"]))
            teardowns.append(dict(phase["teardown_receipt"]))
            leases.append(dict(phase["lease_receipt"]))
            accelerator_samples.extend(phase["accelerator_samples"])
            current = _read_checkpoint(checkpoint)
            current = build_checkpoint_manifest(
                current.get("rows") or [],
                expected_row_count=expected_count,
                input_checksum=input_checksum,
                batch_receipts=current.get("batch_receipts") or [],
                process_receipts=processes,
                teardown_receipts=teardowns,
            )
            write_checkpoint(checkpoint, current)
    except Exception as exc:
        failed = deepcopy(dict(preconditions))
        failed_check = gate_check("fresh_owned_model_batch", True, f"{type(exc).__name__}: {exc}")
        failed["checks"] = [*failed.get("checks", []), failed_check]
        failed["preconditions_ready"] = False
        failed["blocked_reasons"] = [*failed.get("blocked_reasons", []), "fresh_owned_model_batch"]
        artifact = build_blocked_artifact(
            duration_s=time.perf_counter() - started,
            preconditions=failed,
            model_specs=model_specs,
            source_hashes=source_hashes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact
    checkpoint_manifest = _read_checkpoint(checkpoint)
    rows_by_id = verified_checkpoint_rows(checkpoint, input_checksum=input_checksum)
    rows = list(rows_by_id.values())
    artifact = build_complete_artifact(
        duration_s=time.perf_counter() - started,
        preconditions=preconditions,
        model_specs=model_specs,
        source_hashes=source_hashes,
        score_inputs=score_inputs,
        rows=rows,
        process_receipts=processes,
        batch_receipts=checkpoint_manifest.get("batch_receipts") or [],
        teardown_receipts=teardowns,
        checkpoint_manifest=checkpoint_manifest,
        expected_row_count=expected_count,
        lease_receipts=leases,
        accelerator_samples=accelerator_samples,
    )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise CompatibilityStreamError(f"artifact_validation_failed:{validation_errors}")
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed 20260901 experiment command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--checkpoint-path", default=str(REPO_ROOT / CHECKPOINT_RELATIVE_PATH))
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--score-worker", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.score_worker:  # pragma: no cover - live subprocess only.
        return _run_worker(args.model_path, args.port)
    if args.date != RUN_DATE:
        return 2
    artifact = run(
        root=REPO_ROOT,
        result_path=Path(args.result_path),
        checkpoint_path=Path(args.checkpoint_path),
        runtime_dir=Path(args.runtime_dir) if args.runtime_dir else None,
        write=True,
    )
    print(
        json.dumps(
            {
                "result_path": args.result_path,
                "honest_verdict": artifact["honest_verdict"],
                "compatibility_stream_complete_score": artifact[
                    "compatibility_stream_complete_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
