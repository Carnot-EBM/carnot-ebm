"""Exp5734 sealed chronological exact proposal stream.

Spec refs: REQ-VERIFY-5734, SCENARIO-VERIFY-5734.

This module consumes the already-qualified Exp5733 finite-choice proposal
channel and uses the same one-token label scoring interface for a chronological
stream.  The model selects only a proposal.  Deterministic exact validators mint
the admitted label and record any selected-proposal conflict separately.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from carnot import experiment_5733_sota_finite_choice_proposal_channel as upstream


JsonDict = dict[str, Any]
ScoreRunner = Callable[[JsonDict, list[JsonDict], list[JsonDict], JsonDict], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5734_sota_exact_proposal_stream.json")
ROW_MANIFEST_RELATIVE_PATH = Path("results/experiment_5734_sota_exact_proposal_stream.rows.jsonl")
UPSTREAM_RELATIVE_PATH = upstream.RESULT_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5734_sota_exact_proposal_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5734_sota_exact_proposal_stream.py")

SCHEMA = "carnot.experiment_5734.sota_exact_proposal_stream.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5734
EXPERIMENT_ID = "experiment_5734_sota_exact_proposal_stream"
MILESTONE = "2026.07.512"
RUN_DATE = "20260720"
INFERENCE_SUBSTRATE = "local_llama_cpp_python_cuda_gguf_exact_proposal_stream"
SPEC_REFS = ("REQ-VERIFY-5734", "SCENARIO-VERIFY-5734")

QWEN_ID = upstream.QWEN_ID
GEMMA31_ID = upstream.GEMMA31_ID
HEADLINE_MODEL_IDS = (QWEN_ID, GEMMA31_ID)
LABELS = upstream.LABELS
upstream_label_token_receipt = upstream.label_token_receipt

REQUIRED_FAMILIES = (
    "finite_state_reachability",
    "finite_domain_arithmetic",
    "sat_csp",
    "hard_soft_preference",
    "contradiction",
    "abstention",
    "shortcut_trap",
)
ROW_COUNT = 96
PREFIX_LENGTH = ROW_COUNT // 2
RANDOM_SEEDS: JsonDict = {
    "panel_seed": 5734001,
    "label_permutation_seed": 5734002,
    "runner_seed": 5734003,
    "split_seed": 5734004,
    "base_seed": 5734,
}

MODEL_SPECS: list[JsonDict] = []
_UPSTREAM_SPECS = {row["hf_id"]: row for row in upstream.MODEL_SPECS}
for _index, _hf_id in enumerate(HEADLINE_MODEL_IDS):
    _base = dict(_UPSTREAM_SPECS[_hf_id])
    MODEL_SPECS.append(
        {
            "name": _base["name"],
            "hf_id": _hf_id,
            "model_repo_id": _hf_id,
            "family": upstream.model_family(_hf_id),
            "role": _base.get("role"),
            "active_params_b": _base.get("active_params_b"),
            "total_params_b": _base.get("total_params_b"),
            "quantization": _base.get("quantization"),
            "min_vram_gb": _base.get("min_vram_gb"),
            "gpu": _index,
            "headline_eligible": True,
            "legacy_smoke_only": False,
        }
    )

PRIMARY_VALIDATOR_VERSION = "exp5734_primary_family_dispatch_validator_v1"
INDEPENDENT_VALIDATOR_VERSION = "exp5734_independent_enumerating_validator_v1"
ENUMERATION_VALIDATOR_VERSION = "exp5734_stratified_domain_enumeration_v1"
EXACT_VALIDATOR_VERSIONS: JsonDict = {
    family: f"exp5734_{family}_exact_validator_v1" for family in REQUIRED_FAMILIES
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "names the artifact schema version for downstream validators.",
    "experiment": "numeric experiment id for conductor and result indexing.",
    "experiment_id": "stable experiment slug for traceability.",
    "milestone": "milestone accountability for this GGUF stream run.",
    "run_date": "absolute run date prevents relative-date ambiguity.",
    "spec_refs": "binds the artifact to REQ-VERIFY-5734 and SCENARIO-VERIFY-5734.",
    "result_path": "records where the terminal artifact is expected to live.",
    "field_principles": "every stream field declares the gate or provenance boundary it protects.",
    "preconditions_checked": "records upstream channel, GGUF, tokenizer, CUDA, and split preconditions before stream scoring.",
    "MODEL_SPECS": "declares exactly the two headline GGUF identities used for stream rows.",
    "resolved_model_receipts": "binds each headline model to the immutable local GGUF receipt inherited from Exp5733.",
    "model_hashes": "keeps model-weight provenance fixed to authenticated Exp5733 hashes.",
    "gguf_filenames": "names the concrete GGUF files used by the local llama.cpp path.",
    "quantizations": "records the observed GGUF quantization for each headline model.",
    "llama_cpp_runtime_receipt": "records the llama.cpp CUDA runtime inherited by the frozen channel.",
    "cuda_device_receipts": "preserves CUDA device provenance for both headline models.",
    "n_gpu_layers_offloaded": "requires positive layer offload before any headline stream credit.",
    "gpu_memory_receipts": "authenticates non-CPU execution through before/peak/after memory evidence.",
    "cuda_offload_authenticated": "per-model CUDA gate copied from the qualified proposal channel.",
    "qualified_channel_hash": "seals the exact Exp5733 artifact consumed by the stream.",
    "preregistered_panel": "freezes chronological rows, domains, labels, prompts, seeds, model hashes, and validator versions before scores.",
    "family_counts": "proves the row panel is balanced across required exact families.",
    "model_family_counts": "proves each family is balanced across the two headline model assignments.",
    "row_manifest_path": "points to the full chronological row, score, proposal, and validator receipt manifest.",
    "candidate_domain_hashes": "seals every complete candidate domain before model scores are interpreted.",
    "label_permutation_hashes": "seals model-specific one-token label permutations.",
    "score_vector_hashes": "seals the full finite-choice score vector for each row.",
    "proposal_ids": "records model proposal identities without promoting them to truth.",
    "exact_validator_versions": "pins primary, independent, and enumeration validator implementations.",
    "conflict_receipts": "records selected-proposal versus exact-oracle conflicts without changing labels.",
    "missing_row_count": "blocks incomplete chronological streams.",
    "non_finite_score_count": "blocks NaN or infinity from proposal selection.",
    "label_collision_count": "blocks ambiguous one-token label receipts.",
    "validator_disagreement_count": "blocks when independent exact validators disagree.",
    "verifier_is_oracle": "bare true records exact validators as the only label authority.",
    "prospective_prefix_hash": "seals the learner-visible chronological prefix before use.",
    "sealed_suffix_hash": "seals the untouched chronological suffix before use.",
    "stream_root_commitment": "binds upstream, row, score, proposal, validator, prefix, and suffix commitments.",
    "headline_model_count": "records the two-model headline denominator.",
    "sota_proposal_stream_ready_score": "strict readiness scalar, not model accuracy.",
    "model_weight_mutation": "bare false proves GGUF weights were not changed.",
    "freeform_generation_used": "bare false keeps the stream on finite-choice proposals.",
    "grammar_runtime_used": "bare false keeps grammar runtimes out of the stream claim.",
    "external_scorer_used": "bare false prevents judges or external scorers from deciding rows.",
    "token_scores_are_semantic_authority": "bare false keeps token scores as proposal signals only.",
    "inference_substrate": "declares local llama.cpp CUDA GGUF exact proposal-stream scoring.",
    "random_seed": "legacy scalar seed for methodology linters that do not unwrap random_seeds.",
    "random_seeds": "records deterministic panel, label, split, and runner seeds.",
    "reproducibility_checksum": "hashes the artifact with the checksum field blanked.",
    "honest_verdict": "terminal state starts complete: or blocked: and names the readiness boundary.",
    "row_count": "records the expected chronological stream length.",
    "model_counts": "proves the row panel is globally balanced across headline models.",
    "missing_score_count": "blocks incomplete finite-choice score vectors.",
    "incomplete_domain_count": "blocks rows whose finite candidate domain is not complete.",
    "receipt_failure_count": "single mechanical blocker count for row, score, tokenizer, validator, and provenance failures.",
    "proposal_conflict_count": "counts wrong model proposals while keeping exact labels authoritative.",
    "upstream_gate_receipts": "records the Exp5733 gates consumed by this stream.",
    "forbidden_runtime_receipts": "records forbidden runtime families as absent.",
    "tests_added_or_reused": "names focused unit, coverage, full-test, spec, adversarial, and clutter commands.",
    "blocked_reasons": "lists mechanical blockers when the stream is not ready.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "resolved_model_receipts",
    "model_hashes",
    "gguf_filenames",
    "quantizations",
    "llama_cpp_runtime_receipt",
    "cuda_device_receipts",
    "n_gpu_layers_offloaded",
    "gpu_memory_receipts",
    "cuda_offload_authenticated",
    "qualified_channel_hash",
    "preregistered_panel",
    "family_counts",
    "model_family_counts",
    "row_manifest_path",
    "candidate_domain_hashes",
    "label_permutation_hashes",
    "score_vector_hashes",
    "proposal_ids",
    "exact_validator_versions",
    "conflict_receipts",
    "missing_row_count",
    "non_finite_score_count",
    "label_collision_count",
    "validator_disagreement_count",
    "verifier_is_oracle",
    "prospective_prefix_hash",
    "sealed_suffix_hash",
    "stream_root_commitment",
    "headline_model_count",
    "sota_proposal_stream_ready_score",
    "model_weight_mutation",
    "freeform_generation_used",
    "grammar_runtime_used",
    "external_scorer_used",
    "token_scores_are_semantic_authority",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)


class UpstreamChannelError(ValueError):
    """Raised when the frozen Exp5733 finite-choice channel is not qualified."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__(",".join(self.reasons))


class ManifestReplayError(ValueError):
    """Raised when the row manifest no longer matches artifact commitments."""


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: str | Path) -> JsonDict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model_family(hf_id: str) -> str:
    return upstream.model_family(hf_id)


def _model_hash_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    specs = {str(row.get("hf_id")): dict(row) for row in artifact.get("MODEL_SPECS", [])}
    receipts = dict(artifact.get("resolved_model_receipts") or {})
    hashes = dict(artifact.get("model_hashes") or {})
    for hf_id in HEADLINE_MODEL_IDS:
        spec = specs.get(hf_id, {})
        receipt = dict(receipts.get(hf_id) or {})
        digest = str(hashes.get(hf_id) or "")
        if not digest.startswith("sha256:") or len(digest) != 71:
            errors.append("upstream_model_hash_missing")
        if spec and str(spec.get("model_hash") or "") != digest:
            errors.append("upstream_model_hash_mismatch")
        if receipt and str(receipt.get("model_hash") or "") != digest:
            errors.append("upstream_model_hash_mismatch")
        path = Path(str(receipt.get("resolved_model_path") or spec.get("resolved_model_path") or ""))
        if not path.is_file():
            errors.append("upstream_resolved_gguf_missing")
        elif int(receipt.get("model_size_bytes") or spec.get("model_size_bytes") or 0) != path.stat().st_size:
            errors.append("upstream_model_size_mismatch")
    return errors


def _tokenizer_errors(artifact: Mapping[str, Any]) -> tuple[list[str], int]:
    errors: list[str] = []
    collision_count = 0
    receipts = dict(artifact.get("label_token_receipts") or {})
    for hf_id in HEADLINE_MODEL_IDS:
        receipt = dict(receipts.get(hf_id) or {})
        collisions = int(receipt.get("label_collision_count") or 0)
        non_single = int(receipt.get("non_single_token_label_count") or 0)
        collision_count += collisions
        if receipt.get("all_single_unique_tokens") is not True or collisions or non_single:
            errors.append("upstream_label_collision")
        if receipt.get("vocab_only") is not True:
            errors.append("upstream_tokenizer_not_vocab_only")
        if receipt.get("transformers_used") is not False:
            errors.append("upstream_transformers_tokenizer_used")
    return errors, collision_count


def _cuda_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    cuda_map = dict(artifact.get("cuda_offload_authenticated") or {})
    layer_map = dict(artifact.get("n_gpu_layers_offloaded") or {})
    memory = dict(artifact.get("gpu_memory_receipts") or {})
    for hf_id in HEADLINE_MODEL_IDS:
        mem = dict(memory.get(hf_id) or {})
        if cuda_map.get(hf_id) is not True:
            errors.append("upstream_cuda_offload_unauthenticated")
        if int(layer_map.get(hf_id) or 0) <= 0:
            errors.append("upstream_no_gpu_layers_offloaded")
        if int(mem.get("peak_mb") or 0) <= int(mem.get("before_mb") or 0):
            errors.append("upstream_no_gpu_memory_delta")
    return errors


def _upstream_errors(artifact: Mapping[str, Any]) -> tuple[list[str], int]:
    errors: list[str] = []
    if artifact.get("proposal_channel_ready_score") != 1.0:
        errors.append("upstream_channel_not_ready")
    if int(artifact.get("qualified_flagship_model_count") or 0) < 2:
        errors.append("upstream_flagship_model_count")
    if artifact.get("cuda_offload_authenticated_score") != 1.0:
        errors.append("upstream_cuda_score")
    if int(artifact.get("receipt_failure_count") or 0) != 0:
        errors.append("upstream_receipt_failure")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("upstream_verifier_not_oracle")
    for field in (
        "freeform_generation_used",
        "grammar_runtime_used",
        "external_scorer_used",
        "token_scores_are_semantic_authority",
        "retired_runtime_used",
    ):
        if artifact.get(field) is not False:
            errors.append(f"upstream_{field}")
    if not set(HEADLINE_MODEL_IDS).issubset(set(artifact.get("qualified_model_ids") or [])):
        errors.append("upstream_headline_model_missing")
    errors.extend(_model_hash_errors(artifact))
    tokenizer_errors, collision_count = _tokenizer_errors(artifact)
    errors.extend(tokenizer_errors)
    errors.extend(_cuda_errors(artifact))
    return sorted(set(errors)), collision_count


def load_and_verify_upstream_channel(path: str | Path = REPO_ROOT / UPSTREAM_RELATIVE_PATH) -> JsonDict:
    """Load Exp5733 and fail unless every required channel gate is clean."""

    artifact = _read_json(path)
    errors, _collision_count = _upstream_errors(artifact)
    if errors:
        raise UpstreamChannelError(errors)
    artifact = dict(artifact)
    artifact["_qualified_channel_hash"] = sha256_file(path)
    artifact["_qualified_channel_path"] = str(Path(path))
    return artifact


def model_specs_from_upstream(upstream_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return exactly the two headline GGUF specs inherited from Exp5733."""

    source_specs = {str(row.get("hf_id")): dict(row) for row in upstream_artifact.get("MODEL_SPECS", [])}
    receipts = dict(upstream_artifact.get("resolved_model_receipts") or {})
    specs: list[JsonDict] = []
    for index, base in enumerate(MODEL_SPECS):
        hf_id = str(base["hf_id"])
        source = source_specs.get(hf_id, {})
        receipt = dict(receipts.get(hf_id) or {})
        resolved = str(source.get("resolved_model_path") or receipt.get("resolved_model_path") or "")
        specs.append(
            {
                **base,
                "sequence_index": index,
                "gpu": int(source.get("gpu", index) or index),
                "resolved_model_path": resolved,
                "model_path": resolved,
                "gguf_filename": str(source.get("gguf_filename") or Path(resolved).name),
                "model_hash": str(source.get("model_hash") or receipt.get("model_hash") or ""),
                "model_size_bytes": int(source.get("model_size_bytes") or receipt.get("model_size_bytes") or 0),
                "quantization": str(source.get("quantization") or base.get("quantization") or ""),
                "local_model_present": bool(source.get("local_model_present", receipt.get("local_model_present", False))),
                "headline_eligible": True,
                "legacy_smoke_only": False,
            }
        )
    return specs


def _sat_assignments(payload: Mapping[str, Any]) -> list[str]:
    assignments = []
    for x in (0, 1):
        for y in (0, 1):
            ok = True
            for hard in payload["hard"]:
                if hard == "x" and x != 1:
                    ok = False
                elif hard == "not_x" and x != 0:
                    ok = False
                elif hard == "y" and y != 1:
                    ok = False
                elif hard == "not_y" and y != 0:
                    ok = False
                elif hard == "x_or_y" and not (x or y):
                    ok = False
                elif hard == "x_xor_y" and x == y:
                    ok = False
            if ok:
                assignments.append(f"SAT_X{x}{y}")
    return assignments


def _base_family_row(family: str, occurrence: int) -> JsonDict:
    if family == "finite_state_reachability":
        states = ("S0", "S1", "S2", "S3")
        transitions = {
            "S0": {"L": "S1", "R": "S2", "H": "S0"},
            "S1": {"L": "S3", "R": "S0", "H": "S1"},
            "S2": {"L": "S0", "R": "S3", "H": "S2"},
            "S3": {"L": "S2", "R": "S1", "H": "S3"},
        }
        paths = (("L", "R", "H"), ("R", "L"), ("H", "L", "L"), ("R", "R", "L"))
        start = states[occurrence % len(states)]
        symbols = paths[occurrence % len(paths)]
        return {
            "prompt": f"Start at {start}. Apply moves {' '.join(symbols)} in order. Which final state is reached?",
            "validator_payload": {
                "kind": "finite_state",
                "start": start,
                "symbols": list(symbols),
                "states": list(states),
                "transitions": transitions,
            },
            "domain_values": ["S0", "S1", "S2", "S3", "NO_PATH", "CONFLICT"],
        }
    if family == "finite_domain_arithmetic":
        ops = ("add_mod", "sub_mod", "mul_mod", "affine_mod")
        op = ops[occurrence % len(ops)]
        a = (2 * occurrence + 1) % 6
        b = (3 * occurrence + 2) % 6
        symbol = {"add_mod": "+", "sub_mod": "-", "mul_mod": "*", "affine_mod": "affine"}[op]
        prompt = (
            f"Compute ({a} {symbol} {b}) mod 6."
            if op != "affine_mod"
            else f"Compute (2*{a} + {b}) mod 6."
        )
        return {
            "prompt": prompt,
            "validator_payload": {"kind": "arithmetic", "op": op, "a": a, "b": b, "modulus": 6},
            "domain_values": [str(value) for value in range(6)],
        }
    if family == "sat_csp":
        specs = (
            ("x", "y"),
            ("not_x", "not_y"),
            ("x", "not_y"),
            ("not_x", "y"),
            ("x", "not_x"),
            ("x_or_y",),
            ("x_xor_y", "x"),
            ("x_xor_y", "not_x"),
        )
        hard = specs[occurrence % len(specs)]
        return {
            "prompt": "Boolean domain x,y in {0,1}. Constraints: " + ", ".join(hard) + ". Which bounded result class holds?",
            "validator_payload": {"kind": "sat_csp", "hard": list(hard)},
            "domain_values": ["SAT_X00", "SAT_X01", "SAT_X10", "SAT_X11", "UNSAT", "MULTIPLE"],
        }
    if family == "hard_soft_preference":
        candidates = []
        blocked = (occurrence + 2) % 6
        for choice in range(6):
            hard_ok = choice != blocked
            score = (choice * 5 + occurrence * 2) % 17
            candidates.append({"name": f"P{choice}", "hard_ok": hard_ok, "score": score})
        return {
            "prompt": "Choose the hard-feasible option with highest soft score: "
            + "; ".join(
                f"{row['name']} hard={'yes' if row['hard_ok'] else 'no'} soft={row['score']}"
                for row in candidates
            ),
            "validator_payload": {"kind": "preference", "candidates": candidates},
            "domain_values": [f"P{value}" for value in range(6)],
        }
    if family == "contradiction":
        variable = "A" if occurrence % 2 == 0 else "B"
        facts = [[variable, True], [variable, occurrence % 3 != 0], ["C", True]]
        return {
            "prompt": "Facts: "
            + "; ".join(f"{name}={'true' if value else 'false'}" for name, value in facts)
            + ". Which consistency class holds?",
            "validator_payload": {"kind": "contradiction", "facts": facts},
            "domain_values": ["CONSISTENT", "CONTRADICTION", "UNKNOWN", "BOTH_TRUE", "BOTH_FALSE", "ABSTAIN_REQUIRED"],
        }
    if family == "abstention":
        return {
            "prompt": f"Case {occurrence}: the hidden value is one of A/B/C/D, but no observation is provided. What should the exact system do?",
            "validator_payload": {"kind": "abstention", "information_state": "underdetermined"},
            "domain_values": ["ABSTAIN_REQUIRED", "INSUFFICIENT", "A", "B", "C", "D"],
        }
    if family == "shortcut_trap":
        numbers = [((occurrence + shift * 2) % 6) for shift in range(4)]
        return {
            "prompt": f"Numbers are {numbers}. Do not choose the maximum; choose the second-largest distinct value mod 6.",
            "validator_payload": {"kind": "shortcut_trap", "numbers": numbers, "rule": "second_largest_distinct"},
            "domain_values": [str(value) for value in range(6)],
        }
    raise ValueError(f"unknown family: {family}")


def exact_answer_by_primary(row: Mapping[str, Any]) -> str:
    """Compute the admitted answer with the primary deterministic validator."""

    payload = row["validator_payload"]
    if "expected_override" in payload:
        return str(payload["expected_override"])
    kind = str(payload["kind"])
    if kind == "finite_state":
        final = str(payload["start"])
        for symbol in payload["symbols"]:
            final = str(payload["transitions"][final][symbol])
        return final
    if kind == "arithmetic":
        a = int(payload["a"])
        b = int(payload["b"])
        op = str(payload["op"])
        if op == "add_mod":
            value = a + b
        elif op == "sub_mod":
            value = a - b
        elif op == "mul_mod":
            value = a * b
        else:
            value = 2 * a + b
        return str(value % int(payload["modulus"]))
    if kind == "sat_csp":
        assignments = _sat_assignments(payload)
        if not assignments:
            return "UNSAT"
        return "MULTIPLE" if len(assignments) > 1 else assignments[0]
    if kind == "preference":
        feasible = [candidate for candidate in payload["candidates"] if candidate["hard_ok"]]
        return str(max(feasible, key=lambda candidate: (int(candidate["score"]), -int(str(candidate["name"])[1:])))["name"])
    if kind == "contradiction":
        seen: dict[str, bool] = {}
        for name, value in payload["facts"]:
            if name in seen and seen[name] != bool(value):
                return "CONTRADICTION"
            seen[name] = bool(value)
        return "CONSISTENT"
    if kind == "abstention":
        return "ABSTAIN_REQUIRED"
    if kind == "shortcut_trap":
        distinct = sorted(set(int(value) % 6 for value in payload["numbers"]))
        return str(distinct[-2] if len(distinct) >= 2 else distinct[0])
    raise ValueError(f"unknown validator kind: {kind}")


def exact_answer_by_independent(row: Mapping[str, Any]) -> str:
    """Compute the admitted answer with an independent enumeration style."""

    payload = row["validator_payload"]
    kind = str(payload["kind"])
    if kind == "arithmetic":
        return next(value for value in row["answer_domain"] if value == exact_answer_by_primary({"validator_payload": {k: v for k, v in payload.items() if k != "expected_override"}}))
    if kind == "sat_csp":
        assignments = tuple(_sat_assignments(payload))
        return "UNSAT" if not assignments else ("MULTIPLE" if len(assignments) > 1 else assignments[0])
    if kind == "preference":
        ranked = sorted(
            (candidate for candidate in payload["candidates"] if candidate["hard_ok"]),
            key=lambda candidate: (int(candidate["score"]), -int(str(candidate["name"])[1:])),
            reverse=True,
        )
        return str(ranked[0]["name"])
    clean = dict(row)
    clean["validator_payload"] = {k: v for k, v in payload.items() if k != "expected_override"}
    return exact_answer_by_primary(clean)


def _candidate_domain(row_id: str, values: Sequence[str], expected: str) -> list[JsonDict]:
    candidates = []
    for index, value in enumerate(values):
        candidates.append(
            {
                "candidate_id": f"{row_id}-cand-{index}",
                "candidate": str(value),
                "candidate_hash": sha256_text(str(value)),
                "is_exact": str(value) == expected,
                "distractor_type": "exact" if str(value) == expected else "adversarial_hard_distractor",
            }
        )
    return candidates


def _label_mapping(
    *,
    row_id: str,
    candidates: Sequence[Mapping[str, Any]],
    exact_label: str,
    seed: int,
) -> list[JsonDict]:
    exact = next(candidate for candidate in candidates if candidate["is_exact"])
    distractors = [dict(candidate) for candidate in candidates if not candidate["is_exact"]]
    labels = [label for label in LABELS if label != exact_label]
    rng = random.Random(seed)
    rng.shuffle(labels)
    rng.shuffle(distractors)
    by_label: dict[str, Mapping[str, Any]] = {exact_label: exact}
    by_label.update({label: candidate for label, candidate in zip(labels, distractors, strict=True)})
    return [
        {
            "label": label,
            "candidate_id": str(by_label[label]["candidate_id"]),
            "candidate": str(by_label[label]["candidate"]),
            "candidate_hash": str(by_label[label]["candidate_hash"]),
            "is_exact": bool(by_label[label]["is_exact"]),
        }
        for label in LABELS
    ]


def preregister_panel(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    row_count: int = ROW_COUNT,
) -> list[JsonDict]:
    """Freeze chronological rows, labels, prompts, seeds, and validators before scores."""

    specs = {str(spec["hf_id"]): dict(spec) for spec in model_specs}
    occurrence_by_family: Counter[str] = Counter()
    rows: list[JsonDict] = []
    for sequence_index in range(row_count):
        family = REQUIRED_FAMILIES[sequence_index % len(REQUIRED_FAMILIES)]
        occurrence = occurrence_by_family[family]
        occurrence_by_family[family] += 1
        model_hf_id = HEADLINE_MODEL_IDS[sequence_index % len(HEADLINE_MODEL_IDS)]
        model_spec = specs[model_hf_id]
        row_id = f"stream-{sequence_index:03d}-{family.replace('_', '-')}-{_model_family(model_hf_id)}"
        base = _base_family_row(family, occurrence)
        expected = exact_answer_by_primary({"validator_payload": base["validator_payload"]})
        candidates = _candidate_domain(row_id, base["domain_values"], expected)
        seed = int(RANDOM_SEEDS["label_permutation_seed"]) + sequence_index * 17
        labels = _label_mapping(
            row_id=row_id,
            candidates=candidates,
            exact_label=LABELS[sequence_index % len(LABELS)],
            seed=seed,
        )
        prompt = upstream.finite_choice_prompt({"prompt": base["prompt"]}, labels)
        admitted_label = next(item["label"] for item in labels if item["is_exact"])
        admitted_candidate_id = next(item["candidate_id"] for item in labels if item["is_exact"])
        permutation_payload = {
            "row_id": row_id,
            "model_hf_id": model_hf_id,
            "seed": seed,
            "label_mapping": labels,
        }
        pre_score_payload = {
            "row_id": row_id,
            "family": family,
            "model_hf_id": model_hf_id,
            "candidate_domain": candidates,
            "label_mapping": labels,
            "prompt": prompt,
            "seed": seed,
            "model_hash": model_spec.get("model_hash", ""),
            "expected_exact_validator_version": EXACT_VALIDATOR_VERSIONS[family],
        }
        rows.append(
            {
                "schema": SCHEMA + ".preregistered_panel_row",
                "sequence_index": sequence_index,
                "row_id": row_id,
                "control_id": row_id,
                "family": family,
                "family_ordinal": occurrence,
                "model_hf_id": model_hf_id,
                "model_family": _model_family(model_hf_id),
                "model_hash": str(model_spec.get("model_hash") or ""),
                "row_seed": int(RANDOM_SEEDS["panel_seed"]) + sequence_index,
                "label_permutation_seed": seed,
                "candidate_domain": candidates,
                "answer_domain": [str(value) for value in base["domain_values"]],
                "candidate_domain_hash": sha256_json(candidates),
                "label_mapping": labels,
                "label_permutation_hash": sha256_json(permutation_payload),
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "control_prompt": base["prompt"],
                "validator_payload": base["validator_payload"],
                "expected_answer": expected,
                "admitted_candidate_id": admitted_candidate_id,
                "admitted_label": admitted_label,
                "expected_exact_validator_version": EXACT_VALIDATOR_VERSIONS[family],
                "enumeration_sampled": occurrence in (0, 1),
                "pre_score_seal_hash": sha256_json(pre_score_payload),
                "spec_refs": list(SPEC_REFS),
            }
        )
    return rows


def family_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count rows by exact family."""

    return dict(Counter(str(row["family"]) for row in rows))


def model_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count rows by headline model id."""

    return dict(Counter(str(row["model_hf_id"]) for row in rows))


def model_family_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return per-family model-assignment counts."""

    counts: dict[str, Counter[str]] = {family: Counter() for family in REQUIRED_FAMILIES}
    for row in rows:
        counts[str(row["family"])][str(row["model_hf_id"])] += 1
    return {family: dict(counter) for family, counter in counts.items()}


def split_lengths(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return prospective-prefix and sealed-suffix lengths."""

    return {
        "prospective_prefix_length": min(PREFIX_LENGTH, len(rows)),
        "sealed_suffix_length": max(0, len(rows) - min(PREFIX_LENGTH, len(rows))),
    }


def candidate_domain_receipt(row: Mapping[str, Any]) -> JsonDict:
    """Prove the finite candidate domain is complete and contains one exact candidate."""

    candidates = list(row.get("candidate_domain") or [])
    labels = list(row.get("label_mapping") or [])
    domain_values = [str(candidate.get("candidate")) for candidate in candidates]
    label_values = [str(label.get("candidate")) for label in labels]
    exact_candidates = [candidate for candidate in candidates if candidate.get("is_exact") is True]
    return {
        "row_id": str(row["row_id"]),
        "family": str(row["family"]),
        "domain_size": len(candidates),
        "candidate_count": len(labels),
        "domain_complete": len(candidates) == len(LABELS)
        and len(set(domain_values)) == len(LABELS)
        and set(domain_values) == set(label_values),
        "exact_candidate_present": len(exact_candidates) == 1,
        "plausible_hard_distractor_count": len([candidate for candidate in candidates if candidate.get("is_exact") is not True]),
        "domain_hash": sha256_json(candidates),
        "label_candidate_hash": sha256_json(label_values),
    }


def primary_validate_selection(row: Mapping[str, Any], selected_candidate: str) -> JsonDict:
    """Validate a selected proposal with the primary exact implementation."""

    expected = exact_answer_by_primary(row)
    return {
        "validator_version": PRIMARY_VALIDATOR_VERSION,
        "family_validator_version": EXACT_VALIDATOR_VERSIONS[str(row["family"])],
        "admitted_candidate": expected,
        "selected_candidate": selected_candidate,
        "selected_is_exact": selected_candidate == expected,
    }


def independent_validate_selection(row: Mapping[str, Any], selected_candidate: str) -> JsonDict:
    """Validate a selected proposal with an independent exact implementation."""

    expected = exact_answer_by_independent(row)
    return {
        "validator_version": INDEPENDENT_VALIDATOR_VERSION,
        "admitted_candidate": expected,
        "selected_candidate": selected_candidate,
        "selected_is_exact": selected_candidate == expected,
    }


def enumeration_double_check(row: Mapping[str, Any], selected_candidate: str) -> JsonDict:
    """Replay the exact candidate by enumerating the finite domain."""

    expected = exact_answer_by_independent(row)
    found = next((candidate["candidate"] for candidate in row["candidate_domain"] if candidate["candidate"] == expected), "")
    return {
        "validator_version": ENUMERATION_VALIDATOR_VERSION,
        "sampled": bool(row.get("enumeration_sampled")),
        "enumerated_domain_size": len(row["candidate_domain"]),
        "enumerated_admitted_candidate": found,
        "selected_candidate": selected_candidate,
        "enumeration_agrees": found == expected,
    }


def validator_disagrees(row: Mapping[str, Any], selected_candidate: str) -> bool:
    """Return true when primary, independent, or enumeration exact validators disagree."""

    primary = primary_validate_selection(row, selected_candidate)
    independent = independent_validate_selection(row, selected_candidate)
    enumeration = enumeration_double_check(row, selected_candidate)
    return bool(
        primary["admitted_candidate"] != independent["admitted_candidate"]
        or enumeration["enumerated_admitted_candidate"] != independent["admitted_candidate"]
        or primary["selected_is_exact"] != independent["selected_is_exact"]
    )


def _runtime_row_map(receipts: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], JsonDict]:
    rows: dict[tuple[str, str], JsonDict] = {}
    for receipt in receipts:
        for row in receipt.get("rows", []):
            rows[(str(row.get("model_hf_id")), str(row.get("control_id")))] = dict(row)
    return rows


def _selected_from_scores(score_vector: Mapping[str, Any]) -> tuple[str, str]:
    if set(score_vector) != set(LABELS):
        return "", "missing_score"
    converted: dict[str, float] = {}
    for label in LABELS:
        value = float(score_vector[label])
        if not math.isfinite(value):
            return "", "non_finite_score"
        converted[label] = value
    return max(converted, key=lambda label: (converted[label], -LABELS.index(label))), ""


def score_vector_hash(row: Mapping[str, Any]) -> str:
    """Hash only the finite label score vector and label token ids."""

    return sha256_json(
        {
            "row_id": row["row_id"],
            "model_hf_id": row["model_hf_id"],
            "score_vector": row["score_vector"],
            "label_token_ids": row["label_token_ids"],
        }
    )


def stream_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a stream row while excluding its own hash field."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _proposal_id(row_id: str, selected_label: str, selected_candidate: str) -> str:
    if not selected_label:
        return ""
    return f"{row_id}::{selected_label}::{sha256_text(selected_candidate)[7:15]}"


def conflict_receipt(row: Mapping[str, Any], selected_candidate: str, selected_label: str) -> JsonDict:
    """Record selected-proposal conflict with the exact oracle."""

    admitted = exact_answer_by_independent(row)
    admitted_label = next(item["label"] for item in row["label_mapping"] if item["candidate"] == admitted)
    return {
        "row_id": str(row["row_id"]),
        "model_hf_id": str(row["model_hf_id"]),
        "selected_label": selected_label,
        "selected_candidate": selected_candidate,
        "admitted_label": admitted_label,
        "admitted_candidate": admitted,
        "proposal_matches_oracle": selected_candidate == admitted,
        "conflict_hash": sha256_json(
            {
                "row_id": row["row_id"],
                "selected": selected_candidate,
                "admitted": admitted,
            }
        ),
    }


def build_stream_rows(
    *,
    panel: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join preregistered panel rows with score vectors and exact-validator receipts."""

    raw_by_key = _runtime_row_map(runtime_receipts)
    rows: list[JsonDict] = []
    previous = ""
    for panel_row in panel:
        raw = raw_by_key.get((str(panel_row["model_hf_id"]), str(panel_row["row_id"])), {})
        score_vector = dict(raw.get("score_vector") or {})
        selected_label, score_error = _selected_from_scores(score_vector)
        selected_candidate = ""
        selected_candidate_id = ""
        if selected_label:
            label_row = next(item for item in panel_row["label_mapping"] if item["label"] == selected_label)
            selected_candidate = str(label_row["candidate"])
            selected_candidate_id = str(label_row["candidate_id"])
        primary = primary_validate_selection(panel_row, selected_candidate)
        independent = independent_validate_selection(panel_row, selected_candidate)
        enumeration = enumeration_double_check(panel_row, selected_candidate)
        conflict = conflict_receipt(panel_row, selected_candidate, selected_label)
        missing_scores = len([label for label in LABELS if label not in score_vector])
        non_finite = sum(
            1
            for value in score_vector.values()
            if isinstance(value, (int, float)) and not math.isfinite(float(value))
        )
        provenance_break = bool(raw and raw.get("prompt_hash") != panel_row["prompt_hash"])
        row: JsonDict = {
            "schema": ROW_SCHEMA,
            "sequence_index": int(panel_row["sequence_index"]),
            "row_id": str(panel_row["row_id"]),
            "family": str(panel_row["family"]),
            "model_hf_id": str(panel_row["model_hf_id"]),
            "model_family": str(panel_row["model_family"]),
            "prompt": str(panel_row["prompt"]),
            "prompt_hash": str(panel_row["prompt_hash"]),
            "row_seed": int(panel_row["row_seed"]),
            "model_hash": str(panel_row["model_hash"]),
            "candidate_domain": list(panel_row["candidate_domain"]),
            "candidate_domain_hash": str(panel_row["candidate_domain_hash"]),
            "label_mapping": list(panel_row["label_mapping"]),
            "label_permutation_hash": str(panel_row["label_permutation_hash"]),
            "score_vector": score_vector,
            "score_vector_hash": "",
            "label_token_ids": dict(raw.get("label_token_ids") or {}),
            "selected_label": selected_label,
            "selected_candidate": selected_candidate,
            "selected_candidate_id": selected_candidate_id,
            "selected_proposal_id": _proposal_id(str(panel_row["row_id"]), selected_label, selected_candidate),
            "admitted_label": str(conflict["admitted_label"]),
            "admitted_candidate": str(conflict["admitted_candidate"]),
            "admitted_candidate_id": str(panel_row["admitted_candidate_id"]),
            "primary_validation": primary,
            "independent_validation": independent,
            "enumeration_double_check": enumeration,
            "conflict_receipt": conflict,
            "validator_disagreement": validator_disagrees(panel_row, selected_candidate),
            "score_error": score_error or str(raw.get("error") or ""),
            "score_complete": score_error == "" and missing_scores == 0,
            "missing_score_count": missing_scores,
            "non_finite_score_count": non_finite,
            "runtime_row_missing": not bool(raw),
            "provenance_break": provenance_break,
            "token_scores_are_semantic_authority": False,
            "timing": dict(raw.get("timing") or {}),
            "previous_row_hash": previous,
            "row_hash": "",
        }
        row["score_vector_hash"] = score_vector_hash(row)
        row["row_hash"] = stream_row_hash(row)
        previous = str(row["row_hash"])
        rows.append(row)
    return rows


def write_row_manifest(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write chronological row evidence as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_row_manifest(path: str | Path) -> list[JsonDict]:
    """Read a chronological row manifest."""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def candidate_domain_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): str(row["candidate_domain_hash"]) for row in rows}


def label_permutation_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): str(row["label_permutation_hash"]) for row in rows}


def score_vector_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): str(row["score_vector_hash"]) for row in rows}


def proposal_ids(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): str(row["selected_proposal_id"]) for row in rows}


def conflict_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): dict(row["conflict_receipt"]) for row in rows}


def _split_payload(rows: Sequence[Mapping[str, Any]], name: str) -> JsonDict:
    return {
        "split": name,
        "row_hashes": [str(row["row_hash"]) for row in rows],
        "family_counts": family_counts(rows),
        "model_counts": model_counts(rows),
        "score_vector_hashes": score_vector_hashes(rows),
        "proposal_ids": proposal_ids(rows),
    }


def _split_hashes(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    prefix = rows[:PREFIX_LENGTH]
    suffix = rows[PREFIX_LENGTH:]
    return sha256_json(_split_payload(prefix, "prospective_prefix")), sha256_json(_split_payload(suffix, "sealed_suffix"))


def _stream_root(
    *,
    qualified_channel_hash: str,
    rows: Sequence[Mapping[str, Any]],
    prospective_prefix_hash: str,
    sealed_suffix_hash: str,
) -> str:
    return sha256_json(
        {
            "qualified_channel_hash": qualified_channel_hash,
            "row_hashes": [str(row["row_hash"]) for row in rows],
            "candidate_domain_hashes": candidate_domain_hashes(rows),
            "label_permutation_hashes": label_permutation_hashes(rows),
            "score_vector_hashes": score_vector_hashes(rows),
            "proposal_ids": proposal_ids(rows),
            "conflict_receipts_hash": sha256_json(conflict_receipts(rows)),
            "prospective_prefix_hash": prospective_prefix_hash,
            "sealed_suffix_hash": sealed_suffix_hash,
        }
    )


def _label_collision_count(upstream_artifact: Mapping[str, Any]) -> int:
    return sum(
        int(dict(dict(upstream_artifact.get("label_token_receipts") or {}).get(hf_id) or {}).get("label_collision_count") or 0)
        for hf_id in HEADLINE_MODEL_IDS
    )


def _upstream_gate_receipts(upstream_artifact: Mapping[str, Any], qualified_channel_hash: str) -> JsonDict:
    return {
        "source_experiment_id": upstream_artifact.get("experiment_id"),
        "qualified_channel_hash": qualified_channel_hash,
        "proposal_channel_ready_score": upstream_artifact.get("proposal_channel_ready_score"),
        "qualified_flagship_model_count": upstream_artifact.get("qualified_flagship_model_count"),
        "cuda_offload_authenticated_score": upstream_artifact.get("cuda_offload_authenticated_score"),
        "receipt_failure_count": upstream_artifact.get("receipt_failure_count"),
        "verifier_is_oracle": upstream_artifact.get("verifier_is_oracle"),
        "tokenizer_receipts_verified": {
            hf_id: dict(dict(upstream_artifact.get("label_token_receipts") or {}).get(hf_id) or {}).get("all_single_unique_tokens") is True
            for hf_id in HEADLINE_MODEL_IDS
        },
        "model_hashes_verified": {
            hf_id: str(dict(upstream_artifact.get("model_hashes") or {}).get(hf_id) or "").startswith("sha256:")
            for hf_id in HEADLINE_MODEL_IDS
        },
    }


def _preconditions_checked(
    *,
    upstream_artifact: Mapping[str, Any],
    qualified_channel_hash: str,
    panel: Sequence[Mapping[str, Any]],
    blocked_reasons: Sequence[str],
) -> JsonDict:
    return {
        "exp5733_artifact_verified": not blocked_reasons,
        "qualified_channel_hash": qualified_channel_hash,
        "upstream_blocked_reasons": list(blocked_reasons),
        "headline_model_ids": list(HEADLINE_MODEL_IDS),
        "model_hashes_checked": {
            hf_id: bool(dict(upstream_artifact.get("model_hashes") or {}).get(hf_id))
            for hf_id in HEADLINE_MODEL_IDS
        },
        "tokenizer_receipts_checked": {
            hf_id: dict(dict(upstream_artifact.get("label_token_receipts") or {}).get(hf_id) or {}).get("all_single_unique_tokens") is True
            for hf_id in HEADLINE_MODEL_IDS
        },
        "cuda_offload_checked": {
            hf_id: dict(upstream_artifact.get("cuda_offload_authenticated") or {}).get(hf_id) is True
            for hf_id in HEADLINE_MODEL_IDS
        },
        "row_order_committed_before_scores": bool(panel),
        "balanced_panel_committed_before_scores": bool(panel) and len(panel) == ROW_COUNT,
        "split_lengths": split_lengths(panel),
    }


def exact_validator_versions() -> JsonDict:
    return {
        "primary": PRIMARY_VALIDATOR_VERSION,
        "independent": INDEPENDENT_VALIDATOR_VERSION,
        "enumeration": ENUMERATION_VALIDATOR_VERSION,
        "by_family": dict(EXACT_VALIDATOR_VERSIONS),
        "validator_authority": "deterministic_exact_oracle",
    }


def _artifact_common(
    *,
    upstream_artifact: Mapping[str, Any],
    qualified_channel_hash: str,
    model_specs: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    row_manifest_path: str,
    blocked_reasons: Sequence[str],
    tests_added_or_reused: Sequence[str],
) -> JsonDict:
    prefix_hash, suffix_hash = _split_hashes(rows)
    root_hash = _stream_root(
        qualified_channel_hash=qualified_channel_hash,
        rows=rows,
        prospective_prefix_hash=prefix_hash,
        sealed_suffix_hash=suffix_hash,
    )
    domain_receipts = [candidate_domain_receipt(row) for row in panel]
    missing_row_count = sum(1 for row in rows if row.get("runtime_row_missing") is True) if rows else ROW_COUNT
    non_finite_count = sum(int(row.get("non_finite_score_count") or 0) for row in rows)
    missing_score_count = sum(int(row.get("missing_score_count") or 0) for row in rows)
    incomplete_domain_count = sum(1 for receipt in domain_receipts if receipt["domain_complete"] is not True)
    validator_disagreement_count = sum(1 for row in rows if row.get("validator_disagreement") is True)
    provenance_break_count = sum(1 for row in rows if row.get("provenance_break") is True)
    label_collision_count = _label_collision_count(upstream_artifact)
    receipt_failure_count = (
        missing_row_count
        + non_finite_count
        + missing_score_count
        + incomplete_domain_count
        + validator_disagreement_count
        + provenance_break_count
        + label_collision_count
        + len(blocked_reasons)
    )
    conflict_map = conflict_receipts(rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _preconditions_checked(
            upstream_artifact=upstream_artifact,
            qualified_channel_hash=qualified_channel_hash,
            panel=panel,
            blocked_reasons=blocked_reasons,
        ),
        "MODEL_SPECS": [dict(spec) for spec in model_specs],
        "resolved_model_receipts": {
            hf_id: dict(dict(upstream_artifact.get("resolved_model_receipts") or {}).get(hf_id) or {})
            for hf_id in HEADLINE_MODEL_IDS
        },
        "model_hashes": {
            hf_id: str(dict(upstream_artifact.get("model_hashes") or {}).get(hf_id) or "")
            for hf_id in HEADLINE_MODEL_IDS
        },
        "gguf_filenames": {
            hf_id: str(dict(upstream_artifact.get("gguf_filenames") or {}).get(hf_id) or "")
            for hf_id in HEADLINE_MODEL_IDS
        },
        "quantizations": {
            hf_id: str(dict(upstream_artifact.get("quantizations") or {}).get(hf_id) or "")
            for hf_id in HEADLINE_MODEL_IDS
        },
        "llama_cpp_runtime_receipt": {
            "llama_cpp_version": upstream_artifact.get("llama_cpp_version", ""),
            "llama_cpp_build_info": dict(upstream_artifact.get("llama_cpp_build_info") or {}),
            "source_experiment_id": upstream_artifact.get("experiment_id"),
        },
        "cuda_device_receipts": {
            hf_id: dict(dict(upstream_artifact.get("cuda_device_receipts") or {}).get(hf_id) or {})
            for hf_id in HEADLINE_MODEL_IDS
        },
        "n_gpu_layers_offloaded": {
            hf_id: int(dict(upstream_artifact.get("n_gpu_layers_offloaded") or {}).get(hf_id) or 0)
            for hf_id in HEADLINE_MODEL_IDS
        },
        "gpu_memory_receipts": {
            hf_id: dict(dict(upstream_artifact.get("gpu_memory_receipts") or {}).get(hf_id) or {})
            for hf_id in HEADLINE_MODEL_IDS
        },
        "cuda_offload_authenticated": {
            hf_id: dict(upstream_artifact.get("cuda_offload_authenticated") or {}).get(hf_id) is True
            for hf_id in HEADLINE_MODEL_IDS
        },
        "qualified_channel_hash": qualified_channel_hash,
        "preregistered_panel": [dict(row) for row in panel],
        "family_counts": family_counts(panel),
        "model_family_counts": model_family_counts(panel),
        "row_manifest_path": row_manifest_path,
        "candidate_domain_hashes": candidate_domain_hashes(rows),
        "label_permutation_hashes": label_permutation_hashes(rows),
        "score_vector_hashes": score_vector_hashes(rows),
        "proposal_ids": proposal_ids(rows),
        "exact_validator_versions": exact_validator_versions(),
        "conflict_receipts": conflict_map,
        "missing_row_count": missing_row_count,
        "non_finite_score_count": non_finite_count,
        "label_collision_count": label_collision_count,
        "validator_disagreement_count": validator_disagreement_count,
        "verifier_is_oracle": True,
        "prospective_prefix_hash": prefix_hash,
        "sealed_suffix_hash": suffix_hash,
        "stream_root_commitment": root_hash,
        "headline_model_count": 2,
        "sota_proposal_stream_ready_score": 0.0,
        "model_weight_mutation": False,
        "freeform_generation_used": False,
        "grammar_runtime_used": False,
        "external_scorer_used": False,
        "token_scores_are_semantic_authority": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "row_count": len(panel) if panel else ROW_COUNT,
        "model_counts": model_counts(panel),
        "missing_score_count": missing_score_count,
        "incomplete_domain_count": incomplete_domain_count,
        "receipt_failure_count": receipt_failure_count,
        "proposal_conflict_count": sum(1 for receipt in conflict_map.values() if receipt.get("proposal_matches_oracle") is False),
        "upstream_gate_receipts": _upstream_gate_receipts(upstream_artifact, qualified_channel_hash),
        "forbidden_runtime_receipts": {
            "freeform_generation_used": False,
            "grammar_runtime_used": False,
            "external_scorer_used": False,
            "retired_runtime_used": False,
            "model_weight_mutation": False,
        },
        "tests_added_or_reused": list(tests_added_or_reused),
        "blocked_reasons": list(blocked_reasons),
    }
    artifact["sota_proposal_stream_ready_score"] = sota_proposal_stream_ready_score(artifact)
    if artifact["sota_proposal_stream_ready_score"] == 1.0:
        artifact["blocked_reasons"] = []
    else:
        artifact["blocked_reasons"] = _blocked_reasons(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(
    *,
    upstream_path: str | Path,
    blocked_reasons: Sequence[str],
    row_manifest_path: str,
    tests_added_or_reused: Sequence[str],
) -> JsonDict:
    raw = _read_json(upstream_path) if Path(upstream_path).is_file() else {}
    qualified_hash = sha256_file(upstream_path) if Path(upstream_path).is_file() else ""
    specs = model_specs_from_upstream(raw) if raw else [dict(spec) for spec in MODEL_SPECS]
    return _artifact_common(
        upstream_artifact=raw,
        qualified_channel_hash=qualified_hash,
        model_specs=specs,
        panel=[],
        rows=[],
        row_manifest_path=row_manifest_path,
        blocked_reasons=list(blocked_reasons),
        tests_added_or_reused=tests_added_or_reused,
    )


def build_artifact(
    *,
    upstream_artifact: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    row_manifest_path: str,
    tests_added_or_reused: Sequence[str] = (),
) -> JsonDict:
    """Build the terminal artifact from preregistered rows and score receipts."""

    return _artifact_common(
        upstream_artifact=upstream_artifact,
        qualified_channel_hash=str(upstream_artifact.get("_qualified_channel_hash") or ""),
        model_specs=model_specs,
        panel=panel,
        rows=rows,
        row_manifest_path=row_manifest_path,
        blocked_reasons=[],
        tests_added_or_reused=tests_added_or_reused,
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sota_proposal_stream_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when every stream gate is clean."""

    ready = bool(
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] == list(HEADLINE_MODEL_IDS)
        and int(artifact.get("headline_model_count") or 0) == 2
        and int(artifact.get("row_count") or 0) == ROW_COUNT
        and len(artifact.get("preregistered_panel") or []) == ROW_COUNT
        and int(artifact.get("missing_row_count") or 0) == 0
        and int(artifact.get("missing_score_count") or 0) == 0
        and int(artifact.get("non_finite_score_count") or 0) == 0
        and int(artifact.get("label_collision_count") or 0) == 0
        and int(artifact.get("validator_disagreement_count") or 0) == 0
        and int(artifact.get("incomplete_domain_count") or 0) == 0
        and int(artifact.get("receipt_failure_count") or 0) == 0
        and artifact.get("verifier_is_oracle") is True
        and artifact.get("model_weight_mutation") is False
        and artifact.get("freeform_generation_used") is False
        and artifact.get("grammar_runtime_used") is False
        and artifact.get("external_scorer_used") is False
        and artifact.get("token_scores_are_semantic_authority") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and bool(artifact.get("qualified_channel_hash"))
        and bool(artifact.get("prospective_prefix_hash"))
        and bool(artifact.get("sealed_suffix_hash"))
        and bool(artifact.get("stream_root_commitment"))
        and all(dict(artifact.get("cuda_offload_authenticated") or {}).get(hf_id) is True for hf_id in HEADLINE_MODEL_IDS)
        and all(int(dict(artifact.get("n_gpu_layers_offloaded") or {}).get(hf_id) or 0) > 0 for hf_id in HEADLINE_MODEL_IDS)
        and all(
            int(dict(dict(artifact.get("gpu_memory_receipts") or {}).get(hf_id) or {}).get("peak_mb") or 0)
            > int(dict(dict(artifact.get("gpu_memory_receipts") or {}).get(hf_id) or {}).get("before_mb") or 0)
            for hf_id in HEADLINE_MODEL_IDS
        )
        and dict(artifact.get("preconditions_checked") or {}).get("exp5733_artifact_verified") is True
        and not artifact.get("blocked_reasons")
    )
    return 1.0 if ready else 0.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(artifact.get("blocked_reasons") or [])
    for field in (
        "missing_row_count",
        "missing_score_count",
        "non_finite_score_count",
        "label_collision_count",
        "validator_disagreement_count",
        "incomplete_domain_count",
        "receipt_failure_count",
    ):
        if int(artifact.get(field) or 0) > 0:
            reasons.append(field)
    for field in (
        "model_weight_mutation",
        "freeform_generation_used",
        "grammar_runtime_used",
        "external_scorer_used",
        "token_scores_are_semantic_authority",
    ):
        if artifact.get(field) is not False:
            reasons.append(field)
    if dict(artifact.get("preconditions_checked") or {}).get("exp5733_artifact_verified") is not True:
        reasons.append("exp5733_artifact_not_verified")
    return sorted(set(reasons)) or ["sota_proposal_stream_gate_not_met"]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict from mechanical stream gates."""

    if float(artifact.get("sota_proposal_stream_ready_score") or 0.0) == 1.0:
        return "complete: sealed_chronological_sota_exact_proposal_stream_ready"
    return "blocked: " + ",".join(_blocked_reasons(artifact))


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on schema drift or unsupported readiness claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError("field_principles")
    for field in artifact:
        if field not in principles:
            raise ValueError("field_principles")
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(HEADLINE_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if int(artifact.get("headline_model_count") or 0) != 2:
        raise ValueError("headline_model_count")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    for forbidden in (
        "model_weight_mutation",
        "freeform_generation_used",
        "grammar_runtime_used",
        "external_scorer_used",
        "token_scores_are_semantic_authority",
    ):
        if artifact.get(forbidden) is not False:
            raise ValueError(forbidden)
    expected_score = sota_proposal_stream_ready_score(artifact)
    if artifact.get("sota_proposal_stream_ready_score") != expected_score:
        raise ValueError("sota_proposal_stream_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0 and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_score == 0.0 and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def verify_row_manifest(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay row hashes, score hashes, split hashes, and the stream root."""

    expected_score_hashes = dict(artifact.get("score_vector_hashes") or {})
    expected_domain_hashes = dict(artifact.get("candidate_domain_hashes") or {})
    expected_label_hashes = dict(artifact.get("label_permutation_hashes") or {})
    expected_proposals = dict(artifact.get("proposal_ids") or {})
    expected_conflicts = dict(artifact.get("conflict_receipts") or {})
    previous = ""
    for row in rows:
        row_id = str(row["row_id"])
        if row.get("previous_row_hash") != previous:
            raise ManifestReplayError("previous_row_hash")
        if expected_score_hashes.get(row_id) != row.get("score_vector_hash"):
            raise ManifestReplayError("score_vector_hash")
        if expected_domain_hashes.get(row_id) != row.get("candidate_domain_hash"):
            raise ManifestReplayError("candidate_domain_hash")
        if expected_label_hashes.get(row_id) != row.get("label_permutation_hash"):
            raise ManifestReplayError("label_permutation_hash")
        if expected_proposals.get(row_id) != row.get("selected_proposal_id"):
            raise ManifestReplayError("proposal_id")
        if expected_conflicts.get(row_id) != row.get("conflict_receipt"):
            raise ManifestReplayError("conflict_receipt")
        if score_vector_hash(row) != row.get("score_vector_hash"):
            raise ManifestReplayError("score_vector_hash")
        if stream_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        previous = str(row["row_hash"])
    prefix_hash, suffix_hash = _split_hashes(rows)
    if artifact.get("prospective_prefix_hash") != prefix_hash:
        raise ManifestReplayError("prospective_prefix_hash")
    if artifact.get("sealed_suffix_hash") != suffix_hash:
        raise ManifestReplayError("sealed_suffix_hash")
    root = _stream_root(
        qualified_channel_hash=str(artifact.get("qualified_channel_hash") or ""),
        rows=rows,
        prospective_prefix_hash=prefix_hash,
        sealed_suffix_hash=suffix_hash,
    )
    if artifact.get("stream_root_commitment") != root:
        raise ManifestReplayError("stream_root_commitment")
    return True


def default_score_runner(
    model_spec: JsonDict,
    controls: list[JsonDict],
    candidate_rows: list[JsonDict],
    random_seeds: JsonDict,
) -> JsonDict:  # pragma: no cover - host-dependent live llama.cpp path.
    """Score one model's assigned rows through the frozen Exp5733 llama.cpp interface."""

    return upstream.default_score_runner(model_spec, controls, candidate_rows, random_seeds)


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_manifest_path: Path | str = REPO_ROOT / ROW_MANIFEST_RELATIVE_PATH,
    upstream_artifact_path: Path | str = REPO_ROOT / UPSTREAM_RELATIVE_PATH,
    score_runner: ScoreRunner | None = None,
    tests_added_or_reused: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Run the SOTA proposal stream or write an honest blocked artifact."""

    runner = score_runner or default_score_runner
    try:
        upstream_artifact = load_and_verify_upstream_channel(upstream_artifact_path)
    except UpstreamChannelError as exc:
        artifact = _blocked_artifact(
            upstream_path=upstream_artifact_path,
            blocked_reasons=exc.reasons,
            row_manifest_path=str(Path(row_manifest_path)),
            tests_added_or_reused=tests_added_or_reused,
        )
        if write:
            write_row_manifest([], row_manifest_path)
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        return artifact

    model_specs = model_specs_from_upstream(upstream_artifact)
    panel = preregister_panel(model_specs=model_specs)
    runtime_receipts: list[JsonDict] = []
    for spec in model_specs:
        assigned = [dict(row) for row in panel if row["model_hf_id"] == spec["hf_id"]]
        receipt = runner(dict(spec), [], assigned, dict(RANDOM_SEEDS))
        receipt.setdefault("model_hf_id", spec["hf_id"])
        runtime_receipts.append(receipt)
    rows = build_stream_rows(panel=panel, runtime_receipts=runtime_receipts)
    artifact = build_artifact(
        upstream_artifact=upstream_artifact,
        model_specs=model_specs,
        panel=panel,
        rows=rows,
        row_manifest_path=str(Path(row_manifest_path)),
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_row_manifest(rows, row_manifest_path)
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    gc.collect()
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5734 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
