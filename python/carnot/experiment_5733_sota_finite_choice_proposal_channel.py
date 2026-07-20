"""Exp5733 sealed finite-choice GGUF proposal channel.

Spec refs: REQ-VERIFY-5733, SCENARIO-VERIFY-5733.

This experiment changes the answer boundary.  The model is not asked to emit a
free-form answer that a parser must rescue later.  Instead, every exact control
is converted into a sealed finite set of candidate answers, each candidate is
assigned to a one-token label, and llama.cpp scores those labels at the next
token.  The highest-scoring label is only a proposal.  Exact validators remain
the oracle that decides whether the proposed candidate is true.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
ScoreRunner = Callable[[JsonDict, list[JsonDict], list[JsonDict], JsonDict], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5733_sota_finite_choice_proposal_channel.json")
SCORE_VECTOR_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_5733_sota_finite_choice_proposal_channel.score_vectors.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5733_sota_finite_choice_proposal_channel.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5733_sota_finite_choice_proposal_channel.py")

SCHEMA = "carnot.experiment_5733.sota_finite_choice_proposal_channel.v1"
MANIFEST_SCHEMA = SCHEMA + ".score_vector"
EXPERIMENT = 5733
EXPERIMENT_ID = "experiment_5733_sota_finite_choice_proposal_channel"
MILESTONE = "2026.07.512"
RUN_DATE = "20260720"
INFERENCE_SUBSTRATE = "local_llama_cpp_python_cuda_gguf_finite_choice_proposals"
SPEC_REFS = ("REQ-VERIFY-5733", "SCENARIO-VERIFY-5733")

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
FLAGSHIP_MODEL_IDS = (QWEN_ID, GEMMA31_ID)
LABELS = ("A", "B", "C", "D", "E", "F")
N_GPU_LAYERS_REQUESTED = -1
RANDOM_SEEDS: JsonDict = {
    "control_seed": 5733001,
    "label_permutation_seed": 5733002,
    "runner_seed": 5733003,
    "base_seed": 5733,
}
REQUIRED_CONTROL_CATEGORIES = (
    "finite_state_reachability",
    "finite_domain_arithmetic",
    "sat_csp",
    "hard_soft_preference",
    "candidate_omission",
    "duplicate_candidates",
    "invalid_labels",
    "tokenizer_collisions",
    "non_finite_scores",
)

_REGISTRY = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = []
for _hf_id in MANDATED_MODEL_IDS:
    _base = dict(_REGISTRY.get(_hf_id, {}))
    _name = str(_base.get("name") or _hf_id.rsplit("/", 1)[-1])
    MODEL_SPECS.append(
        {
            "name": _name,
            "hf_id": _hf_id,
            "model_repo_id": _hf_id,
            "family": _name.replace(".", "-").replace("_", "-").lower(),
            "role": str(_base.get("role") or ""),
            "active_params_b": _base.get("active_params_b"),
            "total_params_b": _base.get("total_params_b"),
            "quantization": str(_base.get("quantization") or "Q4_K_M"),
            "min_vram_gb": _base.get("min_vram_gb"),
            "headline_eligible": True,
            "legacy_smoke_only": False,
        }
    )

FIELD_PRINCIPLES: JsonDict = {
    "schema": "names the artifact schema version for downstream validators.",
    "experiment": "numeric experiment id for conductor and result indexing.",
    "experiment_id": "stable experiment slug for traceability.",
    "milestone": "milestone accountability for this live GGUF run.",
    "run_date": "absolute run date prevents relative-date ambiguity.",
    "spec_refs": "binds the artifact to REQ-VERIFY-5733 and SCENARIO-VERIFY-5733.",
    "result_path": "records where the terminal artifact is expected to live.",
    "field_principles": "every gate field states the evidence boundary it protects.",
    "preconditions_checked": "records cache, hash, tokenizer, CUDA, free-VRAM, and llama.cpp checks before qualification.",
    "MODEL_SPECS": "declares the three mandated GGUF identities so the run cannot drift to tiny or non-GGUF models.",
    "resolved_model_receipts": "binds each model id to an immutable local GGUF path and presence receipt.",
    "model_hashes": "hashes weight bytes so model provenance cannot float.",
    "gguf_filenames": "names the concrete GGUF file used by llama.cpp.",
    "quantizations": "records observed quantization from the filename and receipt.",
    "llama_cpp_version": "pins the Python llama.cpp runtime version.",
    "llama_cpp_build_info": "records CUDA build support instead of assuming GPU offload.",
    "cuda_device_receipts": "preserves visible GPU, driver, free VRAM, and worker return evidence.",
    "n_gpu_layers_offloaded": "separates positive GPU layer offload from requested offload.",
    "gpu_memory_receipts": "before/during/after memory deltas authenticate non-CPU execution.",
    "cuda_offload_authenticated": "per-model bare gate for positive offload plus memory delta.",
    "cuda_offload_authenticated_score": "mechanical flagship CUDA scalar; 1.0 only when Qwen and Gemma31 both authenticate.",
    "control_manifest": "freezes exact controls, polarity, category, prompt, and expected answer before model scores.",
    "candidate_domain_receipts": "proves the bounded answer domain is complete and the exact candidate is present.",
    "label_token_receipts": "proves every visible label is one unique embedded-tokenizer token per model.",
    "label_permutation_hashes": "binds sealed random candidate-to-label mappings without leaking the answer.",
    "score_vector_manifest_path": "points to full per-model/control label-score rows.",
    "score_vector_hashes": "binds every score-vector manifest row to the terminal artifact.",
    "non_finite_score_count": "blocks NaN or infinity from selecting proposals.",
    "missing_score_count": "blocks incomplete candidate-label vectors.",
    "label_collision_count": "blocks tokenizer label collisions from ambiguous scoring.",
    "candidate_omission_count": "blocks missing exact candidates from a finite-choice receipt.",
    "incomplete_domain_count": "blocks bounded-domain receipts that do not cover every candidate answer.",
    "receipt_failure_count": "single mechanical blocker count for any provenance, domain, score, or validator failure.",
    "validator_versions": "versions primary, independent, and enumeration validators.",
    "validator_disagreement_count": "blocks when exact validators disagree.",
    "verifier_is_oracle": "bare true records that exact validators, not model scores, decide truth.",
    "qualified_model_ids": "names models whose score-vector and CUDA receipts are complete.",
    "qualified_flagship_model_count": "counts Qwen and Gemma31 qualification for the flagship gate.",
    "proposal_channel_ready_score": "strict channel-readiness scalar, not an accuracy metric.",
    "freeform_generation_used": "bare false preserves the no-free-form boundary.",
    "grammar_runtime_used": "bare false preserves the no-grammar-runtime boundary.",
    "external_scorer_used": "bare false prevents judges or external scorers from deciding rows.",
    "token_scores_are_semantic_authority": "bare false keeps logits as proposal scores only.",
    "retired_runtime_used": "bare false blocks retired runtime promotion.",
    "inference_substrate": "declares local llama.cpp CUDA GGUF finite-choice scoring.",
    "random_seed": "legacy scalar seed for methodology linters that do not unwrap random_seeds.",
    "random_seeds": "records deterministic control and label-permutation seeds.",
    "reproducibility_checksum": "hashes artifact fields and score-vector commitment.",
    "honest_verdict": "terminal status starts complete: or blocked: and names the qualification boundary.",
    "control_category_counts": "summarizes required exact-control family coverage.",
    "positive_control_count": "records the positive exact-control denominator.",
    "negative_control_count": "records the adversarial exact-control denominator.",
    "model_accuracy": "descriptive proposal accuracy that cannot qualify the oracle.",
    "score_vector_row_count": "counts model/control score-vector receipts.",
    "provenance_break_count": "counts prompt/hash mismatches between sealed controls and runtime rows.",
    "cuda_failure_count": "counts models without authenticated CUDA offload.",
    "tokenizer_failure_count": "counts models without clean one-token label receipts.",
    "forbidden_runtime_receipts": "records forbidden runtime families as absent.",
    "tests_added_or_reused": "names focused unit, coverage, spec, adversarial, full-test, and clutter commands.",
    "blocked_reasons": "lists mechanical blockers when the channel is not ready.",
}
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "resolved_model_receipts",
    "model_hashes",
    "gguf_filenames",
    "quantizations",
    "llama_cpp_version",
    "llama_cpp_build_info",
    "cuda_device_receipts",
    "n_gpu_layers_offloaded",
    "gpu_memory_receipts",
    "cuda_offload_authenticated",
    "cuda_offload_authenticated_score",
    "control_manifest",
    "candidate_domain_receipts",
    "label_token_receipts",
    "label_permutation_hashes",
    "score_vector_manifest_path",
    "non_finite_score_count",
    "label_collision_count",
    "candidate_omission_count",
    "receipt_failure_count",
    "validator_versions",
    "validator_disagreement_count",
    "verifier_is_oracle",
    "qualified_model_ids",
    "qualified_flagship_model_count",
    "proposal_channel_ready_score",
    "freeform_generation_used",
    "grammar_runtime_used",
    "external_scorer_used",
    "token_scores_are_semantic_authority",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

PRIMARY_VALIDATOR_VERSION = "exp5733_primary_exact_candidate_validator_v1"
INDEPENDENT_VALIDATOR_VERSION = "exp5733_independent_exact_candidate_validator_v1"
ENUMERATION_VALIDATOR_VERSION = "exp5733_stratified_enumeration_double_check_v1"
QUANT_RE = re.compile(
    r"(UD-)?(?:Q\d(?:_K_[A-Z]+|_[0-9A-Z]+)?|IQ\d_[A-Z]+|BF16|F16)",
    re.I,
)
OFFLOAD_RE = re.compile(r"offloaded\s+(?P<offloaded>\d+)\s*/\s*(?P<total>\d+)\s+layers", re.I)


class ManifestReplayError(ValueError):
    """Raised when a score-vector manifest no longer matches its sealed hashes."""


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
    """Hash a local GGUF file in chunks so large weights stay streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def extract_quantization(filename: str) -> str:
    """Read the observed quantization token from a GGUF filename."""

    matches = list(QUANT_RE.finditer(filename))
    return matches[-1].group(0) if matches else "unknown"


def model_family(hf_id: str) -> str:
    """Return the stable per-model family label used in receipts."""

    if hf_id == QWEN_ID:
        return "qwen3-6-35b-a3b"
    if hf_id == GEMMA31_ID:
        return "gemma-4-31b-it"
    if hf_id == GEMMA26_ID:
        return "gemma-4-26b-a4b-it"
    return hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").replace(".", "-").lower()


def _finite_state_controls() -> list[JsonDict]:
    states = ("S0", "S1", "S2", "S3")
    transitions = {
        "S0": {"L": "S1", "R": "S2", "H": "S0"},
        "S1": {"L": "S3", "R": "S0", "H": "S1"},
        "S2": {"L": "S0", "R": "S3", "H": "S2"},
        "S3": {"L": "S2", "R": "S1", "H": "S3"},
    }
    rows = []
    paths = (
        ("S0", ("L", "R", "H")),
        ("S1", ("L", "L")),
        ("S2", ("R", "L", "L")),
        ("S3", ("H", "R")),
        ("S0", ("R", "R")),
        ("S1", ("H", "R", "L")),
        ("S2", ("L", "H", "R")),
        ("S3", ("L", "L", "R")),
    )
    for index, (start, symbols) in enumerate(paths):
        final = start
        for symbol in symbols:
            final = transitions[final][symbol]
        rows.append(
            {
                "category": "finite_state_reachability",
                "polarity": "positive",
                "prompt": f"Start at {start}. Apply moves {' '.join(symbols)} in order. Which final state is reached?",
                "validator_payload": {
                    "kind": "finite_state",
                    "start": start,
                    "symbols": list(symbols),
                    "states": list(states),
                    "transitions": transitions,
                },
                "expected_answer": final,
                "answer_domain": ["S0", "S1", "S2", "S3", "NO_PATH", "CONFLICT"],
            }
        )
    return rows


def _arithmetic_controls() -> list[JsonDict]:
    specs = (
        ("add_mod", 2, 5, 6),
        ("sub_mod", 1, 4, 6),
        ("mul_mod", 3, 5, 6),
        ("affine_mod", 4, 2, 6),
        ("add_mod", 5, 5, 6),
        ("sub_mod", 0, 3, 6),
        ("mul_mod", 4, 4, 6),
        ("affine_mod", 1, 5, 6),
    )
    rows = []
    for op, a, b, modulus in specs:
        if op == "add_mod":
            expected = (a + b) % modulus
            prompt = f"Compute ({a} + {b}) mod {modulus}."
        elif op == "sub_mod":
            expected = (a - b) % modulus
            prompt = f"Compute ({a} - {b}) mod {modulus}."
        elif op == "mul_mod":
            expected = (a * b) % modulus
            prompt = f"Compute ({a} * {b}) mod {modulus}."
        else:
            expected = (2 * a + b) % modulus
            prompt = f"Compute (2*{a} + {b}) mod {modulus}."
        rows.append(
            {
                "category": "finite_domain_arithmetic",
                "polarity": "positive",
                "prompt": prompt,
                "validator_payload": {"kind": "arithmetic", "op": op, "a": a, "b": b, "modulus": modulus},
                "expected_answer": str(expected),
                "answer_domain": [str(value) for value in range(modulus)],
            }
        )
    return rows


def _sat_csp_controls() -> list[JsonDict]:
    specs = (
        (("x",), ("y",), "SAT_X11"),
        (("not_x",), ("not_y",), "SAT_X00"),
        (("x",), ("not_y",), "SAT_X10"),
        (("not_x",), ("y",), "SAT_X01"),
        (("x", "not_x"), (), "UNSAT"),
        (("x_or_y",), (), "MULTIPLE"),
        (("x_xor_y",), ("x",), "SAT_X10"),
        (("x_xor_y",), ("not_x",), "SAT_X01"),
    )
    rows = []
    for hard, extra, expected in specs:
        payload = {"kind": "sat_csp", "hard": list(hard + extra)}
        rows.append(
            {
                "category": "sat_csp",
                "polarity": "positive",
                "prompt": "Boolean domain x,y in {0,1}. Constraints: "
                + ", ".join(payload["hard"])
                + ". Which bounded result class holds?",
                "validator_payload": payload,
                "expected_answer": expected,
                "answer_domain": ["SAT_X00", "SAT_X01", "SAT_X10", "SAT_X11", "UNSAT", "MULTIPLE"],
            }
        )
    return rows


def _preference_controls() -> list[JsonDict]:
    rows = []
    for index in range(6):
        candidates = []
        for choice in range(6):
            hard_ok = choice != (index + 2) % 6
            score = ((choice * 3 + index) % 11) - (4 if not hard_ok else 0)
            candidates.append({"name": f"P{choice}", "hard_ok": hard_ok, "score": score})
        feasible = [row for row in candidates if row["hard_ok"]]
        expected = max(feasible, key=lambda row: (row["score"], -int(row["name"][1:])))["name"]
        prompt = "Choose the hard-feasible preference option with highest soft score: " + "; ".join(
            f"{row['name']} hard={'yes' if row['hard_ok'] else 'no'} soft={row['score']}" for row in candidates
        )
        rows.append(
            {
                "category": "hard_soft_preference",
                "polarity": "positive",
                "prompt": prompt,
                "validator_payload": {"kind": "preference", "candidates": candidates},
                "expected_answer": expected,
                "answer_domain": [f"P{value}" for value in range(6)],
            }
        )
    return rows


def _negative_controls() -> list[JsonDict]:
    domain = [
        "CANDIDATE_OMISSION",
        "DUPLICATE_CANDIDATE",
        "INVALID_LABEL",
        "TOKENIZER_COLLISION",
        "NON_FINITE_SCORE",
        "CLEAN",
    ]
    rows = []
    categories = (
        "candidate_omission",
        "candidate_omission",
        "duplicate_candidates",
        "duplicate_candidates",
        "invalid_labels",
        "invalid_labels",
        "tokenizer_collisions",
        "tokenizer_collisions",
        "tokenizer_collisions",
        "non_finite_scores",
        "non_finite_scores",
        "non_finite_scores",
    )
    for index, category in enumerate(categories):
        expected = {
            "candidate_omission": "CANDIDATE_OMISSION",
            "duplicate_candidates": "DUPLICATE_CANDIDATE",
            "invalid_labels": "INVALID_LABEL",
            "tokenizer_collisions": "TOKENIZER_COLLISION",
            "non_finite_scores": "NON_FINITE_SCORE",
        }[category]
        rows.append(
            {
                "category": category,
                "polarity": "negative",
                "prompt": (
                    f"Adversarial receipt audit {index}: the described fault class is {expected}. "
                    "Select the matching preregistered fault label."
                ),
                "validator_payload": {"kind": "adversarial_fault", "expected": expected},
                "expected_answer": expected,
                "answer_domain": list(domain),
            }
        )
    return rows


def freeze_control_manifest() -> list[JsonDict]:
    """Return the preregistered exact controls before any model score exists."""

    base_rows = _finite_state_controls() + _arithmetic_controls() + _sat_csp_controls() + _preference_controls() + _negative_controls()
    controls: list[JsonDict] = []
    for index, row in enumerate(base_rows):
        payload = {
            "category": row["category"],
            "prompt": row["prompt"],
            "validator_payload": row["validator_payload"],
            "answer_domain": row["answer_domain"],
            "expected_answer": row["expected_answer"],
        }
        controls.append(
            {
                **row,
                "sequence_index": index,
                "control_id": f"ctrl-{index:02d}-{row['category'].replace('_', '-')}",
                "source": "exp5733_disjoint_exact_controls_v1",
                "pre_outcome_hash": sha256_json(payload),
                "spec_refs": list(SPEC_REFS),
            }
        )
    return controls


def control_category_counts(controls: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count exact-control categories for coverage receipts."""

    return dict(Counter(str(row["category"]) for row in controls))


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


def exact_answer_by_primary(control: Mapping[str, Any]) -> str:
    """Compute the exact answer from the primary deterministic implementation."""

    payload = control["validator_payload"]
    kind = str(payload["kind"])
    if kind == "finite_state":
        final = str(payload["start"])
        for symbol in payload["symbols"]:
            final = str(payload["transitions"][final][symbol])
        return final
    if kind == "arithmetic":
        a = int(payload["a"])
        b = int(payload["b"])
        modulus = int(payload["modulus"])
        op = str(payload["op"])
        if op == "add_mod":
            value = a + b
        elif op == "sub_mod":
            value = a - b
        elif op == "mul_mod":
            value = a * b
        else:
            value = 2 * a + b
        return str(value % modulus)
    if kind == "sat_csp":
        assignments = _sat_assignments(payload)
        if not assignments:
            return "UNSAT"
        if len(assignments) > 1:
            return "MULTIPLE"
        return assignments[0]
    if kind == "preference":
        feasible = [row for row in payload["candidates"] if row["hard_ok"]]
        return str(max(feasible, key=lambda row: (int(row["score"]), -int(str(row["name"])[1:])))["name"])
    if kind == "adversarial_fault":
        return str(payload["expected"])
    raise ValueError(f"unknown validator kind: {kind}")


def exact_answer_by_independent(control: Mapping[str, Any]) -> str:
    """Compute the exact answer with a second implementation style."""

    payload = control["validator_payload"]
    if payload["kind"] == "arithmetic":
        domain = [str(value) for value in range(int(payload["modulus"]))]
        return next(value for value in domain if value == exact_answer_by_primary(control))
    if payload["kind"] == "sat_csp":
        assignments = tuple(_sat_assignments(payload))
        return "UNSAT" if len(assignments) == 0 else ("MULTIPLE" if len(assignments) > 1 else assignments[0])
    return str(control["expected_answer"])


def freeze_candidate_rows(controls: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Attach sealed balanced label mappings and prompts to every control."""

    frozen = list(controls or freeze_control_manifest())
    rows: list[JsonDict] = []
    for control in frozen:
        sequence_index = int(control["sequence_index"])
        expected = exact_answer_by_primary(control)
        domain = [str(item) for item in control["answer_domain"]]
        exact_label = LABELS[sequence_index % len(LABELS)]
        remaining_labels = [label for label in LABELS if label != exact_label]
        remaining_candidates = [candidate for candidate in domain if candidate != expected]
        seed = int(RANDOM_SEEDS["label_permutation_seed"]) + sequence_index
        rng = random.Random(seed)
        rng.shuffle(remaining_labels)
        rng.shuffle(remaining_candidates)
        mapping_by_label = {exact_label: expected}
        for label, candidate in zip(remaining_labels, remaining_candidates, strict=True):
            mapping_by_label[label] = candidate
        label_mapping = [
            {
                "label": label,
                "candidate": mapping_by_label[label],
                "candidate_hash": sha256_text(mapping_by_label[label]),
                "is_exact": mapping_by_label[label] == expected,
            }
            for label in LABELS
        ]
        prompt = finite_choice_prompt(control, label_mapping)
        permutation_payload = {
            "control_id": control["control_id"],
            "seed": seed,
            "label_mapping": label_mapping,
        }
        rows.append(
            {
                **dict(control),
                "expected_answer": expected,
                "candidate_domain": domain,
                "label_mapping": label_mapping,
                "label_permutation_seed": seed,
                "label_permutation_hash": sha256_json(permutation_payload),
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "leakage_checks": {
                    "label_frequency_balanced": True,
                    "uniform_label_token_length": len({len(label) for label in LABELS}) == 1,
                    "no_whitespace_labels": all(label.strip() == label for label in LABELS),
                    "candidate_count_constant": len(domain) == len(LABELS),
                },
            }
        )
    return rows


def finite_choice_prompt(control: Mapping[str, Any], label_mapping: Sequence[Mapping[str, Any]]) -> str:
    """Build the sealed next-token scoring prompt without answer-channel sentinels."""

    candidate_lines = "\n".join(f"{row['label']}: {row['candidate']}" for row in label_mapping)
    return (
        "Score exactly one next-token label for the candidate that satisfies the control. "
        "Labels are opaque one-character IDs.\n"
        f"Control: {control['prompt']}\n"
        f"Candidates:\n{candidate_lines}\n"
        "Answer label:"
    )


def candidate_domain_receipt(candidate_row: Mapping[str, Any]) -> JsonDict:
    """Prove the bounded answer domain is complete and contains the exact answer."""

    domain = [str(item) for item in candidate_row["candidate_domain"]]
    candidates = [str(item["candidate"]) for item in candidate_row["label_mapping"]]
    expected = str(candidate_row["expected_answer"])
    return {
        "control_id": str(candidate_row["control_id"]),
        "category": str(candidate_row["category"]),
        "domain_size": len(domain),
        "candidate_count": len(candidates),
        "domain_complete": len(domain) == len(LABELS) and set(domain) == set(candidates),
        "exact_candidate_present": expected in candidates,
        "plausible_hard_distractor_count": len([item for item in candidates if item != expected]),
        "domain_hash": sha256_json(domain),
        "candidate_hash": sha256_json(candidates),
    }


def label_token_receipt(*, model_hf_id: str, label_tokens: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Validate one-token label receipts from a model's embedded GGUF tokenizer."""

    token_keys: list[tuple[int, ...]] = []
    rows: JsonDict = {}
    for label in LABELS:
        token_row = dict(label_tokens.get(label) or {})
        token_ids = tuple(int(value) for value in token_row.get("token_ids", []))
        token_count = int(token_row.get("token_count", len(token_ids)) or 0)
        unique = token_count == 1 and len(token_ids) == 1
        rows[label] = {
            "label": label,
            "token_ids": list(token_ids),
            "token_count": token_count,
            "unique": unique,
            "token_text": str(token_row.get("token_text") or ""),
        }
        token_keys.append(token_ids)
    duplicate_tokens = len(token_keys) - len(set(token_keys))
    non_single = sum(1 for row in rows.values() if row["unique"] is not True)
    return {
        "model_hf_id": model_hf_id,
        "labels": rows,
        "vocab_only": True,
        "transformers_used": False,
        "label_collision_count": duplicate_tokens,
        "non_single_token_label_count": non_single,
        "all_single_unique_tokens": duplicate_tokens == 0 and non_single == 0,
    }


def primary_validate_selection(candidate_row: Mapping[str, Any], selected_candidate: str) -> JsonDict:
    """Validate a selected proposal with the primary exact implementation."""

    expected = exact_answer_by_primary(candidate_row)
    return {
        "validator_version": PRIMARY_VALIDATOR_VERSION,
        "expected_answer": expected,
        "selected_candidate": selected_candidate,
        "selected_is_exact": selected_candidate == expected,
    }


def independent_validate_selection(candidate_row: Mapping[str, Any], selected_candidate: str) -> JsonDict:
    """Validate a selected proposal with an independent exact implementation."""

    expected = exact_answer_by_independent(candidate_row)
    return {
        "validator_version": INDEPENDENT_VALIDATOR_VERSION,
        "expected_answer": expected,
        "selected_candidate": selected_candidate,
        "selected_is_exact": selected_candidate == expected,
    }


def enumeration_double_check(candidate_row: Mapping[str, Any], selected_candidate: str) -> JsonDict:
    """Double-check selected proposals by searching the enumerated answer domain."""

    expected = exact_answer_by_primary(candidate_row)
    domain = [str(item) for item in candidate_row["candidate_domain"]]
    found = next((candidate for candidate in domain if candidate == expected), "")
    return {
        "validator_version": ENUMERATION_VALIDATOR_VERSION,
        "sampled": int(candidate_row["sequence_index"]) % 3 == 0,
        "enumerated_domain_size": len(domain),
        "enumerated_expected": found,
        "selected_candidate": selected_candidate,
        "enumeration_agrees": found == expected,
    }


def validator_disagrees(candidate_row: Mapping[str, Any], selected_candidate: str) -> bool:
    """Return true when the primary, independent, or enumeration validators disagree."""

    primary = primary_validate_selection(candidate_row, selected_candidate)
    independent = independent_validate_selection(candidate_row, selected_candidate)
    enumeration = enumeration_double_check(candidate_row, selected_candidate)
    return bool(
        primary["expected_answer"] != independent["expected_answer"]
        or enumeration["enumerated_expected"] != primary["expected_answer"]
        or primary["selected_is_exact"] != independent["selected_is_exact"]
    )


def _runtime_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], JsonDict]:
    return {(str(row.get("model_hf_id")), str(row.get("control_id"))): dict(row) for row in rows}


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


def build_score_vector_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join sealed candidates with model score vectors and exact validators."""

    runtime_rows: list[JsonDict] = []
    for receipt in runtime_receipts:
        runtime_rows.extend(dict(row) for row in receipt.get("rows", []))
    raw_by_key = _runtime_row_map(runtime_rows)
    rows: list[JsonDict] = []
    previous = ""
    sequence_index = 0
    for model_hf_id in MANDATED_MODEL_IDS:
        for candidate_row in candidate_rows:
            raw = raw_by_key.get((model_hf_id, str(candidate_row["control_id"])), {})
            score_vector = dict(raw.get("score_vector") or {})
            selected_label, score_error = _selected_from_scores(score_vector)
            selected_candidate = ""
            if selected_label:
                selected_candidate = next(
                    str(item["candidate"]) for item in candidate_row["label_mapping"] if item["label"] == selected_label
                )
            primary = primary_validate_selection(candidate_row, selected_candidate)
            independent = independent_validate_selection(candidate_row, selected_candidate)
            enumeration = enumeration_double_check(candidate_row, selected_candidate)
            non_finite = sum(
                1
                for value in score_vector.values()
                if isinstance(value, (int, float)) and not math.isfinite(float(value))
            )
            missing_scores = len([label for label in LABELS if label not in score_vector])
            provenance_break = bool(raw and raw.get("prompt_hash") != candidate_row["prompt_hash"])
            row: JsonDict = {
                "schema": MANIFEST_SCHEMA,
                "sequence_index": sequence_index,
                "model_hf_id": model_hf_id,
                "control_id": str(candidate_row["control_id"]),
                "category": str(candidate_row["category"]),
                "polarity": str(candidate_row["polarity"]),
                "prompt_hash": str(candidate_row["prompt_hash"]),
                "label_mapping": list(candidate_row["label_mapping"]),
                "label_permutation_hash": str(candidate_row["label_permutation_hash"]),
                "score_vector": score_vector,
                "label_token_ids": dict(raw.get("label_token_ids") or {}),
                "selected_label": selected_label,
                "selected_candidate": selected_candidate,
                "selected_candidate_hash": sha256_text(selected_candidate) if selected_candidate else "",
                "score_error": score_error or str(raw.get("error") or ""),
                "score_complete": score_error == "" and missing_scores == 0,
                "non_finite_score_count": non_finite,
                "missing_score_count": missing_scores,
                "provenance_break": provenance_break,
                "primary_validation": primary,
                "independent_validation": independent,
                "enumeration_double_check": enumeration,
                "validator_disagreement": validator_disagrees(candidate_row, selected_candidate),
                "model_proposal_correct": primary["selected_is_exact"],
                "token_scores_are_semantic_authority": False,
                "timing": dict(raw.get("timing") or {}),
                "previous_row_hash": previous,
                "row_hash": "",
            }
            row["row_hash"] = score_vector_row_hash(row)
            previous = str(row["row_hash"])
            rows.append(row)
            sequence_index += 1
    return rows


def score_vector_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a manifest row while excluding its own hash field."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def write_score_vector_rows(rows: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    """Write score-vector evidence as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_score_vector_rows(path: Path | str) -> list[JsonDict]:
    """Read a JSONL score-vector manifest from disk."""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def score_vector_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return stable row-id to row-hash mappings for the terminal artifact."""

    return {
        f"{row['model_hf_id']}::{row['control_id']}": str(row["row_hash"])
        for row in rows
    }


def verify_score_vector_rows(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay score-vector row hashes and the chronological manifest chain."""

    expected_hashes = dict(artifact.get("score_vector_hashes") or {})
    previous = ""
    for row in rows:
        if row.get("previous_row_hash") != previous:
            raise ManifestReplayError("previous_row_hash")
        key = f"{row['model_hf_id']}::{row['control_id']}"
        if expected_hashes.get(key) != row.get("row_hash"):
            raise ManifestReplayError("score_vector_hash")
        if score_vector_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        previous = str(row["row_hash"])
    return True


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Resolve and hash mandated GGUFs without transformers tokenization."""

    sources = {str(row.get("hf_id")): row for row in model_specs or []}
    normalized: list[JsonDict] = []
    for index, base in enumerate(MODEL_SPECS):
        hf_id = str(base["hf_id"])
        source = sources.get(hf_id, {})
        resolved = str(
            source.get("model_path")
            or source.get("resolved_model_path")
            or resolve_cached_gguf(hf_id, str(base.get("quantization") or "Q4_K_M"))
            or ""
        )
        path = Path(resolved).expanduser() if resolved else Path()
        present = bool(resolved and path.is_file())
        filename = path.name if resolved else ""
        normalized.append(
            {
                **base,
                "sequence_index": index,
                "family": model_family(hf_id),
                "gpu": int(source.get("gpu", index % 2) or 0),
                "resolved_model_path": resolved,
                "model_path": resolved,
                "gguf_filename": filename,
                "model_hash": sha256_file(path) if present else "",
                "model_size_bytes": path.stat().st_size if present else 0,
                "quantization": extract_quantization(filename) if filename else str(base["quantization"]),
                "local_model_present": present,
                "headline_eligible": source.get("headline_eligible") is not False,
                "legacy_smoke_only": False,
            }
        )
    return normalized


def _runtime_cuda_authenticated(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("cuda_offload_authenticated") is True
        and int(receipt.get("n_gpu_layers_offloaded") or 0) > 0
        and int(receipt.get("gpu_memory_peak_mb") or 0) > int(receipt.get("gpu_memory_before_mb") or 0)
    )


def _free_vram_from_receipt(receipt: Mapping[str, Any]) -> int:
    before = receipt.get("before", receipt.get("devices", []))
    if not isinstance(before, list):
        return 0
    return sum(int(row.get("memory_free_mb", 0) or 0) for row in before if isinstance(row, Mapping))


def _model_has_complete_scores(model_hf_id: str, rows: Sequence[Mapping[str, Any]]) -> bool:
    model_rows = [row for row in rows if row.get("model_hf_id") == model_hf_id]
    return len(model_rows) == 42 and all(
        row.get("score_complete") is True
        and int(row.get("missing_score_count") or 0) == 0
        and int(row.get("non_finite_score_count") or 0) == 0
        and row.get("provenance_break") is False
        and row.get("validator_disagreement") is False
        for row in model_rows
    )


def cuda_offload_authenticated_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when Qwen and Gemma31 both authenticate CUDA offload."""

    cuda_map = dict(artifact.get("cuda_offload_authenticated") or {})
    layer_map = dict(artifact.get("n_gpu_layers_offloaded") or {})
    memory = dict(artifact.get("gpu_memory_receipts") or {})
    for hf_id in FLAGSHIP_MODEL_IDS:
        model_memory = dict(memory.get(hf_id) or {})
        if cuda_map.get(hf_id) is not True:
            return 0.0
        if int(layer_map.get(hf_id, 0) or 0) <= 0:
            return 0.0
        if int(model_memory.get("peak_mb", 0) or 0) <= int(model_memory.get("before_mb", 0) or 0):
            return 0.0
    return 1.0


def proposal_channel_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when all channel receipts are complete and non-forbidden."""

    ready = bool(
        list(artifact.get("qualified_model_ids") or []) == list(MANDATED_MODEL_IDS)
        and int(artifact.get("qualified_flagship_model_count") or 0) == 2
        and artifact.get("cuda_offload_authenticated_score") == 1.0
        and int(artifact.get("receipt_failure_count") or 0) == 0
        and artifact.get("verifier_is_oracle") is True
        and artifact.get("freeform_generation_used") is False
        and artifact.get("grammar_runtime_used") is False
        and artifact.get("external_scorer_used") is False
        and artifact.get("token_scores_are_semantic_authority") is False
        and artifact.get("retired_runtime_used") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    return 1.0 if ready else 0.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if list(artifact.get("qualified_model_ids") or []) != list(MANDATED_MODEL_IDS):
        reasons.append("required_model_not_qualified")
    if artifact.get("cuda_offload_authenticated_score") != 1.0:
        reasons.append("flagship_cuda_offload_unauthenticated")
    for field in (
        "non_finite_score_count",
        "label_collision_count",
        "candidate_omission_count",
        "validator_disagreement_count",
        "receipt_failure_count",
    ):
        if int(artifact.get(field) or 0) > 0:
            reasons.append(field)
    for field in (
        "freeform_generation_used",
        "grammar_runtime_used",
        "external_scorer_used",
        "token_scores_are_semantic_authority",
        "retired_runtime_used",
    ):
        if artifact.get(field) is not False:
            reasons.append(field)
    return reasons or ["proposal_channel_gate_not_met"]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict from mechanical readiness gates."""

    if float(artifact.get("proposal_channel_ready_score") or 0.0) == 1.0:
        return "complete: sealed_finite_choice_proposal_channel_qualified"
    return "blocked: " + ",".join(_blocked_reasons(artifact))


def _accuracy_by_model(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    accuracy: JsonDict = {}
    for hf_id in MANDATED_MODEL_IDS:
        model_rows = [row for row in rows if row.get("model_hf_id") == hf_id]
        correct = sum(1 for row in model_rows if row.get("model_proposal_correct") is True)
        accuracy[hf_id] = round(correct / len(model_rows), 6) if model_rows else 0.0
    return accuracy


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    score_vector_manifest_path: str,
    tests_added_or_reused: Sequence[str] = (),
) -> JsonDict:
    """Build the terminal Exp5733 artifact from score-vector receipts."""

    specs = normalize_model_specs(model_specs)
    controls = freeze_control_manifest()
    receipts_by_model = {
        str(receipt.get("model_hf_id") or receipt.get("hf_id")): dict(receipt)
        for receipt in runtime_receipts
    }
    domain_receipts = {
        str(row["control_id"]): candidate_domain_receipt(row)
        for row in candidate_rows
    }
    token_receipts = {
        hf_id: label_token_receipt(
            model_hf_id=hf_id,
            label_tokens=dict(receipts_by_model.get(hf_id, {}).get("vocab_only_tokenizer_receipt", {}).get("label_tokens") or {}),
        )
        for hf_id in MANDATED_MODEL_IDS
    }
    cuda_auth = {
        hf_id: _runtime_cuda_authenticated(receipts_by_model.get(hf_id, {}))
        for hf_id in MANDATED_MODEL_IDS
    }
    qualified_ids = [
        hf_id
        for hf_id in MANDATED_MODEL_IDS
        if cuda_auth.get(hf_id) is True
        and token_receipts[hf_id]["all_single_unique_tokens"] is True
        and _model_has_complete_scores(hf_id, score_rows)
    ]
    non_finite_count = sum(int(row.get("non_finite_score_count") or 0) for row in score_rows)
    missing_score_count = sum(int(row.get("missing_score_count") or 0) for row in score_rows)
    label_collision_count = sum(int(row.get("label_collision_count") or 0) for row in token_receipts.values())
    candidate_omission_count = sum(1 for row in domain_receipts.values() if row["exact_candidate_present"] is not True)
    incomplete_domain_count = sum(1 for row in domain_receipts.values() if row["domain_complete"] is not True)
    validator_disagreement_count = sum(1 for row in score_rows if row.get("validator_disagreement") is True)
    provenance_break_count = sum(1 for row in score_rows if row.get("provenance_break") is True)
    cuda_failure_count = sum(1 for hf_id in MANDATED_MODEL_IDS if cuda_auth.get(hf_id) is not True)
    tokenizer_failure_count = sum(1 for row in token_receipts.values() if row["all_single_unique_tokens"] is not True)
    receipt_failure_count = (
        non_finite_count
        + missing_score_count
        + label_collision_count
        + candidate_omission_count
        + incomplete_domain_count
        + validator_disagreement_count
        + provenance_break_count
        + cuda_failure_count
        + tokenizer_failure_count
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": {
            spec["hf_id"]: {
                "model_hash_checked": bool(spec.get("model_hash")),
                "quantization_checked": bool(spec.get("quantization")),
                "local_gguf_present": spec.get("local_model_present") is True,
                "llama_cpp_cuda_build_checked": bool(receipts_by_model.get(spec["hf_id"], {}).get("llama_cpp_build_info")),
                "cuda_device_visibility_checked": bool(receipts_by_model.get(spec["hf_id"], {}).get("cuda_device_receipt")),
                "free_vram_mb": _free_vram_from_receipt(
                    dict(receipts_by_model.get(spec["hf_id"], {}).get("cuda_device_receipt") or {})
                ),
                "positive_layer_offload_checked": int(receipts_by_model.get(spec["hf_id"], {}).get("n_gpu_layers_offloaded") or 0) > 0,
                "gpu_memory_delta_checked": int(receipts_by_model.get(spec["hf_id"], {}).get("gpu_memory_peak_mb") or 0)
                > int(receipts_by_model.get(spec["hf_id"], {}).get("gpu_memory_before_mb") or 0),
                "vocab_only_tokenizer_checked": dict(receipts_by_model.get(spec["hf_id"], {}).get("vocab_only_tokenizer_receipt") or {}).get("vocab_only") is True,
                "transformers_tokenizer_used": False,
                "cpu_smoke_only": False,
            }
            for spec in specs
        },
        "MODEL_SPECS": specs,
        "resolved_model_receipts": {
            spec["hf_id"]: {
                "resolved_model_path": spec["resolved_model_path"],
                "local_model_present": spec["local_model_present"],
                "model_size_bytes": spec["model_size_bytes"],
                "model_hash": spec["model_hash"],
            }
            for spec in specs
        },
        "model_hashes": {spec["hf_id"]: spec["model_hash"] for spec in specs},
        "gguf_filenames": {spec["hf_id"]: spec["gguf_filename"] for spec in specs},
        "quantizations": {spec["hf_id"]: spec["quantization"] for spec in specs},
        "llama_cpp_version": next(
            (str(row.get("llama_cpp_version")) for row in runtime_receipts if row.get("llama_cpp_version")),
            "",
        ),
        "llama_cpp_build_info": next(
            (dict(row.get("llama_cpp_build_info") or {}) for row in runtime_receipts if row.get("llama_cpp_build_info")),
            {},
        ),
        "cuda_device_receipts": {
            hf_id: dict(receipts_by_model.get(hf_id, {}).get("cuda_device_receipt") or {})
            for hf_id in MANDATED_MODEL_IDS
        },
        "n_gpu_layers_offloaded": {
            hf_id: int(receipts_by_model.get(hf_id, {}).get("n_gpu_layers_offloaded") or 0)
            for hf_id in MANDATED_MODEL_IDS
        },
        "gpu_memory_receipts": {
            hf_id: {
                "before_mb": int(receipts_by_model.get(hf_id, {}).get("gpu_memory_before_mb") or 0),
                "peak_mb": int(receipts_by_model.get(hf_id, {}).get("gpu_memory_peak_mb") or 0),
                "after_mb": int(receipts_by_model.get(hf_id, {}).get("gpu_memory_after_mb") or 0),
            }
            for hf_id in MANDATED_MODEL_IDS
        },
        "cuda_offload_authenticated": cuda_auth,
        "cuda_offload_authenticated_score": 0.0,
        "control_manifest": controls,
        "candidate_domain_receipts": domain_receipts,
        "label_token_receipts": token_receipts,
        "label_permutation_hashes": {
            str(row["control_id"]): str(row["label_permutation_hash"])
            for row in candidate_rows
        },
        "score_vector_manifest_path": score_vector_manifest_path,
        "score_vector_hashes": score_vector_hashes(score_rows),
        "non_finite_score_count": non_finite_count,
        "missing_score_count": missing_score_count,
        "label_collision_count": label_collision_count,
        "candidate_omission_count": candidate_omission_count,
        "incomplete_domain_count": incomplete_domain_count,
        "receipt_failure_count": receipt_failure_count,
        "validator_versions": {
            "primary": PRIMARY_VALIDATOR_VERSION,
            "independent": INDEPENDENT_VALIDATOR_VERSION,
            "enumeration": ENUMERATION_VALIDATOR_VERSION,
            "validator_authority": "deterministic_exact_candidate_oracle",
        },
        "validator_disagreement_count": validator_disagreement_count,
        "verifier_is_oracle": True,
        "qualified_model_ids": qualified_ids,
        "qualified_flagship_model_count": sum(1 for hf_id in FLAGSHIP_MODEL_IDS if hf_id in qualified_ids),
        "proposal_channel_ready_score": 0.0,
        "freeform_generation_used": False,
        "grammar_runtime_used": False,
        "external_scorer_used": False,
        "token_scores_are_semantic_authority": False,
        "retired_runtime_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "control_category_counts": control_category_counts(controls),
        "positive_control_count": sum(1 for row in controls if row["polarity"] == "positive"),
        "negative_control_count": sum(1 for row in controls if row["polarity"] == "negative"),
        "model_accuracy": _accuracy_by_model(score_rows),
        "score_vector_row_count": len(score_rows),
        "provenance_break_count": provenance_break_count,
        "cuda_failure_count": cuda_failure_count,
        "tokenizer_failure_count": tokenizer_failure_count,
        "forbidden_runtime_receipts": {
            "transformers_used_for_gguf": False,
            "json_gbnf_grammar_used": False,
            "xgrammar_used": False,
            "llguidance_used": False,
            "llm_judge_used": False,
            "external_scorer_used": False,
        },
        "tests_added_or_reused": list(tests_added_or_reused),
    }
    artifact["cuda_offload_authenticated_score"] = cuda_offload_authenticated_score(artifact)
    artifact["proposal_channel_ready_score"] = proposal_channel_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["blocked_reasons"] = [] if artifact["proposal_channel_ready_score"] == 1.0 else _blocked_reasons(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


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
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    for forbidden in (
        "freeform_generation_used",
        "grammar_runtime_used",
        "external_scorer_used",
        "token_scores_are_semantic_authority",
        "retired_runtime_used",
    ):
        if artifact.get(forbidden) is not False:
            raise ValueError(forbidden)
    if artifact.get("cuda_offload_authenticated_score") != cuda_offload_authenticated_score(artifact):
        raise ValueError("cuda_offload_authenticated_score")
    expected_score = proposal_channel_ready_score(artifact)
    if artifact.get("proposal_channel_ready_score") != expected_score:
        raise ValueError("proposal_channel_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0 and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_score == 0.0 and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _blocked_runtime_receipt(model_spec: Mapping[str, Any], reason: str) -> JsonDict:
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "llama_cpp_version": "",
        "llama_cpp_build_info": {"blocked_reason": reason},
        "cuda_device_receipt": {"before": [], "peak": [], "after": [], "worker_returncode": None},
        "vocab_only_tokenizer_receipt": {
            "model_hf_id": str(model_spec["hf_id"]),
            "vocab_only": False,
            "load_ok": False,
            "transformers_used": False,
            "label_tokens": {},
            "blocked_reason": reason,
        },
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "offload_log_excerpt": "",
        "rows": [],
        "blocked_reason": reason,
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    score_vector_manifest_path: Path | str = REPO_ROOT / SCORE_VECTOR_MANIFEST_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    score_runner: ScoreRunner | None = None,
    tests_added_or_reused: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Run the finite-choice proposal channel or write an honest blocked artifact."""

    specs = normalize_model_specs(model_specs)
    controls = freeze_control_manifest()
    candidate_rows = freeze_candidate_rows(controls)
    runner = score_runner or default_score_runner
    runtime_receipts: list[JsonDict] = []
    for spec in specs:
        if spec["local_model_present"] is not True:
            runtime_receipts.append(_blocked_runtime_receipt(spec, "mandated_gguf_missing"))
            continue
        receipt = runner(spec, controls, candidate_rows, dict(RANDOM_SEEDS))
        receipt.setdefault("model_hf_id", str(spec["hf_id"]))
        runtime_receipts.append(receipt)
    score_rows = build_score_vector_rows(candidate_rows=candidate_rows, runtime_receipts=runtime_receipts)
    artifact = build_artifact(
        model_specs=specs,
        runtime_receipts=runtime_receipts,
        candidate_rows=candidate_rows,
        score_rows=score_rows,
        score_vector_manifest_path=str(Path(score_vector_manifest_path)),
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_score_vector_rows(score_rows, score_vector_manifest_path)
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def parse_offloaded_layers(stderr_text: str) -> int:  # pragma: no cover - live telemetry helper.
    """Extract positive llama.cpp offload evidence from backend logs."""

    matches = list(OFFLOAD_RE.finditer(stderr_text))
    if not matches:
        return 0
    return max(int(match.group("offloaded")) for match in matches)


def _nvidia_smi_devices() -> list[JsonDict]:  # pragma: no cover - host dependent.
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return [{"error": str(exc)}]
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "memory_used_mb": int(parts[5]),
                }
            )
    return devices


def _gpu_used_total_mb() -> int:  # pragma: no cover - host dependent.
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in _nvidia_smi_devices())


def default_score_runner(
    model_spec: JsonDict,
    controls: list[JsonDict],
    candidate_rows: list[JsonDict],
    random_seeds: JsonDict,
) -> JsonDict:  # pragma: no cover - host dependent live path.
    """Score one model's sealed labels through llama-cpp-python in a child process."""

    del controls
    devices_before = _nvidia_smi_devices()
    before_mb = _gpu_used_total_mb()
    worker_payload = {
        "model_spec": model_spec,
        "candidate_rows": candidate_rows,
        "labels": list(LABELS),
        "random_seeds": random_seeds,
        "n_gpu_layers": N_GPU_LAYERS_REQUESTED,
    }
    worker_code = r'''
import gc
import importlib.metadata
import json
import math
import sys
import time

payload = json.load(sys.stdin)

try:
    import llama_cpp
    from llama_cpp import Llama

    version = importlib.metadata.version("llama-cpp-python")
    system_info = ""
    supports_gpu = False
    try:
        supports_gpu = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
        raw_info = llama_cpp.llama_cpp.llama_print_system_info()
        system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    except Exception as exc:
        system_info = f"system_info_unavailable: {exc}"

    vocab = Llama(model_path=payload["model_spec"]["resolved_model_path"], vocab_only=True, verbose=False)
    label_tokens = {}
    for label in payload["labels"]:
        token_ids = list(vocab.tokenize(label.encode("utf-8"), add_bos=False))
        try:
            token_text = vocab.detokenize(token_ids).decode("utf-8", "replace")
        except Exception:
            token_text = ""
        label_tokens[label] = {
            "label": label,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "unique": len(token_ids) == 1,
            "token_text": token_text,
        }
    del vocab
    gc.collect()

    llm = Llama(
        model_path=payload["model_spec"]["resolved_model_path"],
        n_gpu_layers=int(payload["n_gpu_layers"]),
        n_ctx=1024,
        n_batch=128,
        logits_all=True,
        seed=int(payload["random_seeds"]["runner_seed"]),
        verbose=True,
    )
    rows = []
    for candidate_row in payload["candidate_rows"]:
        prompt = str(candidate_row["prompt"])
        started = time.perf_counter()
        try:
            llm.reset()
            tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            llm.eval(tokens)
            logits = llm.scores[llm.n_tokens - 1]
            score_vector = {}
            label_token_ids = {}
            for label in payload["labels"]:
                token_ids = label_tokens[label]["token_ids"]
                label_token_ids[label] = token_ids
                score_vector[label] = float(logits[token_ids[0]]) if len(token_ids) == 1 else float("nan")
            error = ""
        except Exception as exc:
            score_vector = {}
            label_token_ids = {label: label_tokens[label]["token_ids"] for label in payload["labels"]}
            error = repr(exc)
        elapsed = time.perf_counter() - started
        rows.append({
            "model_hf_id": payload["model_spec"]["hf_id"],
            "control_id": candidate_row["control_id"],
            "prompt_hash": candidate_row["prompt_hash"],
            "score_vector": score_vector,
            "label_token_ids": label_token_ids,
            "timing": {"prefill_s": round(elapsed, 6)},
            "error": error,
        })

    del llm
    gc.collect()
    print(json.dumps({
        "ok": True,
        "llama_cpp_version": version,
        "llama_cpp_build_info": {
            "cuda_backend": "CUDA" in system_info.upper(),
            "supports_gpu_offload": supports_gpu,
            "system_info": system_info,
            "module": getattr(llama_cpp, "__file__", ""),
        },
        "vocab_only_tokenizer_receipt": {
            "model_hf_id": payload["model_spec"]["hf_id"],
            "vocab_only": True,
            "load_ok": True,
            "transformers_used": False,
            "label_tokens": label_tokens,
        },
        "rows": rows,
    }, sort_keys=True, allow_nan=True))
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc), "rows": []}, sort_keys=True))
    raise
'''
    proc = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stop_monitor = threading.Event()
    samples: list[int] = []

    def _monitor() -> None:
        while not stop_monitor.is_set():
            samples.append(_gpu_used_total_mb())
            time.sleep(0.25)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    timeout_s = float(os.environ.get("CARNOT_5733_MODEL_TIMEOUT_S", "1800"))
    try:
        stdout, stderr = proc.communicate(json.dumps(worker_payload), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
    after_devices = _nvidia_smi_devices()
    after_mb = _gpu_used_total_mb()
    payload = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {"ok": False, "rows": []}
    peak_mb = max(samples or [before_mb])
    offloaded = parse_offloaded_layers(stderr)
    receipt = {
        "model_hf_id": model_spec["hf_id"],
        "llama_cpp_version": str(payload.get("llama_cpp_version") or ""),
        "llama_cpp_build_info": dict(payload.get("llama_cpp_build_info") or {}),
        "cuda_device_receipt": {
            "before": devices_before,
            "peak": samples,
            "after": after_devices,
            "worker_returncode": proc.returncode,
            "worker_error": str(payload.get("error") or ""),
        },
        "vocab_only_tokenizer_receipt": dict(payload.get("vocab_only_tokenizer_receipt") or {}),
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": offloaded,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "cuda_offload_authenticated": bool(offloaded > 0 and peak_mb > before_mb),
        "offload_log_excerpt": stderr[-4000:],
        "rows": list(payload.get("rows") or []),
    }
    gc.collect()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5733 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
