"""Measure self-commitment as an audit-only mapping shortcut control.

The model workers receive frozen prompts and candidates, but no exact labels or
authority records. The controller opens exact labels only after all workers exit.
This adapts self-commitment to structured mapping contexts. It does not reproduce
the GSM8K metric from the cited paper and cannot become a verifier feature.

Spec refs: REQ-INF-6998 and SCENARIO-INF-6998-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import (
    build_vram_release_row,
    embedded_tokenizer_probe,
    gpu_inventory,
    llama_cpp_probe,
    parse_offloaded_layers,
)
from carnot.experiment_6973_lease_aware_gguf_runtime import (
    _choose_free_port,
    _port_is_free,
    owned_process_absent,
    terminate_owned_process,
)
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import (
    capture_process_lineage,
    read_process_identity,
    sha256_file,
    write_json_atomic,
)


JsonDict = dict[str, Any]
EXPERIMENT_ID = "experiment_6998_three_family_commitment_controls"
SCHEMA = "carnot.experiment_6998.three_family_commitment_controls.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 6_998_202_609_05
PREFERRED_QUANT = "Q4_K_M"
INFERENCE_SUBSTRATE = "live_local_llama_cpp_three_family_commitment_probe_cuda"
EXPECTED_PAIR_COUNT = 12
EXPECTED_CANDIDATE_COUNT = 24
EXPECTED_UNIT_COUNT = 216
COMMITMENT_THRESHOLD = 0.8
PREFIX_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
CONDITIONS = ("clean", "true_provenance_hint", "permuted_decoy_hint")
VRAM_RELEASE_TOLERANCE_MB = 512
MODEL_TIMEOUT_S = 14_400.0
LEASE_TTL_S = MODEL_TIMEOUT_S + 600.0
POLL_INTERVAL_S = 0.5

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6998_three_family_commitment_controls.json"
CHECKPOINT_ROOT = REPO_ROOT / "results/checkpoints/experiment_6998_three_family_commitment_controls"
EXP6973_PATH = REPO_ROOT / "results/experiment_6973_lease_aware_gguf_runtime.json"
EXP6984_PATH = REPO_ROOT / "results/experiment_6984_exact_contrast_fixture.json"
EXP6997_PATH = REPO_ROOT / "results/experiment_6997_authority_sidecar_rebuild.json"
MUTATION_SIDECAR_PATH = (
    REPO_ROOT
    / "results/raw/experiment_6997_authority_sidecar_rebuild/mutation_authority_sidecar.jsonl"
)
LABEL_SIDECAR_PATH = (
    REPO_ROOT / "results/raw/experiment_6997_authority_sidecar_rebuild/label_split_sidecar.jsonl"
)
EXPECTED_SOURCE_HASHES = {
    "exp6973": "sha256:abb8790a31bd5745edaa13ad4b2718699c569b6e477981a98094400d65404af8",
    "exp6984": "sha256:15f9a9bb58ca7793966f2fbac548f6879a64417b50e31b0504078cfe6ea46a3f",
    "exp6997": "sha256:661a486b9400b879b117adeb6e2fea270b25c19e4037e4eec65fd19af70cc1be",
    "mutation_sidecar": "sha256:67c39c5ca3661f0f333638294aed20b106a2c2fd6d4279a3ecc3b6a4c3e5ed90",
    "label_sidecar": "sha256:68920d1c05f3d068b1aa9e0517b5a70b57b854f79d6a5af6718ac60a15890f6c",
}
SOURCE_PATHS = {
    "exp6973": EXP6973_PATH,
    "exp6984": EXP6984_PATH,
    "exp6997": EXP6997_PATH,
    "mutation_sidecar": MUTATION_SIDECAR_PATH,
    "label_sidecar": LABEL_SIDECAR_PATH,
}
REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LOAD_CONFIG: JsonDict = {
    "config_id": "commitment_probe_dual_cuda_ctx16384",
    "n_ctx": 16_384,
    "n_gpu_layers": -1,
    "n_batch": 2_048,
    "n_ubatch": 512,
    "main_gpu": 0,
    "split_mode": "layer",
    "tensor_split": [0.5, 0.5],
    "visible_devices": [0, 1],
    "logits_all": True,
    "tokenizer_source": "embedded_gguf",
    "temperature": 0.0,
    "prefix_fractions": list(PREFIX_FRACTIONS),
    "commitment_threshold": COMMITMENT_THRESHOLD,
}
SEMANTIC_TASK = (
    "Decide whether the mapping candidate preserves the feasible solutions and "
    "objective ordering of the source and target formulations. Use only the "
    "candidate content and any stated audit hint.\n"
)
PROMPT_SUFFIX = (
    "\nReturn exactly one final marker. Use FINAL_CHOICE: VALID when the mapping "
    "is valid. Use FINAL_CHOICE: INVALID when it is not valid.\n"
)

FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "exact_label",
        "exact_labels",
        "expected_label",
        "certified_relation",
        "authority_records",
        "authority_result",
        "authority_results",
        "sidecar_path",
        "label_sidecar_path",
        "mutation_authority_sidecar_path",
        "source_group",
        "source_group_id",
        "split",
        "future_window",
        "family_scores",
        "model_scores",
        "other_family_scores",
    }
)
WORKER_INPUT_FIELDS = frozenset(
    {
        "pair_id",
        "candidate_id",
        "candidate_hash",
        "pair_position",
        "condition",
        "candidate_text",
        "prompt_preamble",
        "prompt_suffix",
        "prompt_text",
        "prompt_hash",
        "hint_hash",
        "provenance_source_pair_id",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "live_duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes",
    "pair_manifest_rows",
    "pair_manifest_hash",
    "condition_manifest_rows",
    "prompt_hash_rows",
    "candidate_hash_rows",
    "rows",
    "per_pair_results",
    "per_candidate_condition_model_rows",
    "terminal_choice_rows",
    "prefix_fraction_rows",
    "choice_token_rows",
    "commitment_curve_rows",
    "first_commitment_latency_rows",
    "commitment_range_rows",
    "uncommitted_mass_rows",
    "uncertainty_rows",
    "choice_flip_rows",
    "label_denial_rows",
    "process_isolation_rows",
    "late_label_join_rows",
    "paired_condition_delta_rows",
    "bootstrap_interval_rows",
    "family_stratum_rows",
    "gpu_runtime_rows",
    "lease_rows",
    "checkpoint_rows",
    "teardown_rows",
    "vram_release_rows",
    "expected_unit_count",
    "observed_unit_count",
    "commitment_control_complete_score",
    "shortcut_commitment_detected_score",
    "audit_only_control",
    "learner_feature_allowed",
    "verifier_fit_performed",
    "self_commitment_paper_reproduction_claimed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract auditable.",
    "preconditions_checked": "Measured gates stop unsafe or fabricated model work.",
    "inference_substrate": "The substrate distinguishes live CUDA evidence from simulation.",
    "duration_s": "Total wall time exposes skipped execution.",
    "live_duration_s": "Model time separates live work from setup and audit joins.",
    "source_artifact_hashes": "Pinned hashes bind the control to frozen upstream evidence.",
    "MODEL_SPECS": "Exact declarations prevent silent family substitution.",
    "models_used": "The ordered roster makes family coverage falsifiable.",
    "model_file_hashes": "File hashes bind results to exact cached weights.",
    "pair_manifest_rows": "A label-blind roster prevents pair selection after outcomes open.",
    "pair_manifest_hash": "One digest detects pair, candidate, or order drift.",
    "condition_manifest_rows": "Frozen conditions make clean and hinted contexts replayable.",
    "prompt_hash_rows": "Prompt hashes detect hidden context changes.",
    "candidate_hash_rows": "Candidate hashes detect semantic input replacement.",
    "rows": "Terminal unit rows preserve the full planned denominator.",
    "per_pair_results": "Pair summaries retain dependence between matched candidates.",
    "per_candidate_condition_model_rows": "One row per unit prevents pooled evidence gaps.",
    "terminal_choice_rows": "Raw terminal decisions define each model's own target choice.",
    "prefix_fraction_rows": "Prefix receipts prove the fixed partial-context order.",
    "choice_token_rows": "Token scalars replay choice mass without full vectors.",
    "commitment_curve_rows": "Whole curves prevent one selected prefix from controlling the claim.",
    "first_commitment_latency_rows": "Threshold latency measures when the model commits.",
    "commitment_range_rows": "Range records how much commitment changes across context.",
    "uncommitted_mass_rows": "Uncommitted mass retains weak or absent commitment.",
    "uncertainty_rows": "Entropy exposes indecision that latency alone hides.",
    "choice_flip_rows": "Flip counts expose unstable partial-context decisions.",
    "label_denial_rows": "Denial receipts prove oracle fields did not enter model workers.",
    "process_isolation_rows": "Owned child receipts exclude foreign model processes.",
    "late_label_join_rows": "A separate late join keeps labels outside every model process.",
    "paired_condition_delta_rows": "Within-pair deltas isolate hint effects from item difficulty.",
    "bootstrap_interval_rows": "Intervals make the shortcut gate depend on uncertainty.",
    "family_stratum_rows": "Family strata expose effects driven by one model only.",
    "gpu_runtime_rows": "PID-linked samples prove CUDA use for every family.",
    "lease_rows": "Owner-bound journals prove task authority over both devices.",
    "checkpoint_rows": "Pair-family checkpoints permit exact resume after interruption.",
    "teardown_rows": "Exit receipts prove each family closed before handoff.",
    "vram_release_rows": "Memory recovery prevents cross-family contamination.",
    "expected_unit_count": "The fixed 216 denominator prevents silent scope reduction.",
    "observed_unit_count": "The observed count exposes missing or duplicate units.",
    "commitment_control_complete_score": "Completion measures evidence integrity, not effect size.",
    "shortcut_commitment_detected_score": "Detection requires a true-only negative latency interval.",
    "audit_only_control": "True prevents this shortcut probe from becoming a product feature.",
    "learner_feature_allowed": "False keeps commitment outside the learner allowlist.",
    "verifier_fit_performed": "False prevents an audit from becoming hidden model selection.",
    "self_commitment_paper_reproduction_claimed": "False limits the claim to structured mappings.",
    "random_seed": "A fixed seed makes derangement and bootstrap replayable.",
    "reproducibility_checksum": "A timing-inclusive digest detects artifact drift.",
    "gate_check_summary": "Expected and observed values make blocked gates actionable.",
    "verifier_is_oracle": "False states that the probe is not an exact verifier.",
    "verdict_class": "A closed class keeps automation unambiguous.",
    "honest_verdict": "A class-consistent prefix prevents an overstated result.",
}


class ManifestError(ValueError):
    """Raised when a frozen roster or condition could leak or drift."""


class ChoiceError(ValueError):
    """Raised when a model does not produce one exact terminal choice."""


class CheckpointError(ValueError):
    """Raised when checkpoint identity or content cannot replay."""


class LabelJoinError(ValueError):
    """Raised when label opening would precede complete worker teardown."""


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key and separator rules."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the project digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the project digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value in canonical form."""

    return sha256_text(canonical_json(value))


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the mandated cached pair first, then add the third family."""

    pair = cached_pair_func(gpu_indices=(0, 1)) or []
    pair_paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows: list[JsonDict] = []
    for model_id in REQUIRED_MODEL_IDS:
        path = pair_paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": PREFERRED_QUANT,
                "resolution_method": (
                    "cached_sota_pair(gpu_indices=(0, 1))"
                    if model_id in pair_paths
                    else "resolve_cached_gguf exact family extension"
                ),
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject substitutions, auxiliary files, and single-device placement."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id", ""))
        path = str(row.get("model_path", ""))
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("gpu_indices") != [0, 1]:
            errors.append(f"dual_gpu_indices_missing:{model_id}")
        if row.get("headline_eligible") is not True:
            errors.append(f"headline_eligibility_missing:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def _nested_items(value: Any, path: str = "$") -> list[tuple[str, str, Any]]:
    rows: list[tuple[str, str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            rows.append((str(key), child, item))
            rows.extend(_nested_items(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_nested_items(item, f"{path}[{index}]"))
    return rows


def model_input_errors(value: Any) -> list[str]:
    """Find authority fields at any depth before a model can load."""

    return [
        f"denied_field:{path}"
        for key, path, _item in _nested_items(value)
        if key.casefold() in FORBIDDEN_MODEL_FIELDS
    ]


def freeze_pair_manifest(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = EXPECTED_PAIR_COUNT,
    source_groups: Mapping[str, str] | None = None,
) -> JsonDict:
    """Freeze held-out pairs from label-free candidate rows only."""

    selected = [row for row in candidate_rows if row.get("split") == "held_out"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("contrast_group_id", ""))].append(row)
    errors: list[str] = []
    if len(grouped) != expected_pair_count:
        errors.append(f"pair_count:{len(grouped)}")
    output: list[JsonDict] = []
    seen_candidates: set[str] = set()
    seen_sources: set[str] = set()
    for pair_ordinal, pair_id in enumerate(sorted(grouped)):
        pair = sorted(grouped[pair_id], key=lambda row: int(row.get("pair_position", -1)))
        if len(pair) != 2 or [row.get("pair_position") for row in pair] != [0, 1]:
            errors.append(f"pair_shape:{pair_id}")
            continue
        source_group_id = str(
            (source_groups or {}).get(pair_id) or pair[0].get("source_group_id") or pair_id
        )
        if source_group_id in seen_sources:
            errors.append(f"source_group_overlap:{source_group_id}")
        seen_sources.add(source_group_id)
        for candidate_ordinal, row in enumerate(pair):
            candidate_id = str(row.get("candidate_id", ""))
            candidate_text = str(row.get("serialized_candidate", ""))
            candidate_hash = sha256_text(candidate_text)
            if not candidate_id or candidate_id in seen_candidates:
                errors.append(f"candidate_duplicate_or_empty:{candidate_id}")
            seen_candidates.add(candidate_id)
            if candidate_hash != row.get("serialization_hash"):
                errors.append(f"candidate_hash_mismatch:{candidate_id}")
            output.append(
                {
                    "pair_ordinal": pair_ordinal,
                    "candidate_ordinal": candidate_ordinal,
                    "pair_id": pair_id,
                    "source_group_id": source_group_id,
                    "candidate_id": candidate_id,
                    "candidate_hash": candidate_hash,
                    "candidate_text": candidate_text,
                    "pair_position": int(row.get("pair_position", -1)),
                    "formulation_family": str(row.get("formulation_family", "")),
                    "selected_before_label_open": True,
                }
            )
    if len(output) != expected_pair_count * 2:
        errors.append(f"candidate_count:{len(output)}")
    if errors:
        raise ManifestError(";".join(errors))
    return {"rows": output, "pair_manifest_hash": sha256_json(output)}


def render_provenance_text(mutation: Mapping[str, Any]) -> str:
    """Render mutation provenance without labels, witnesses, or authority status."""

    detail = mutation.get("mutation_detail")
    family = mutation.get("fault_family")
    if not detail and not family:
        return "Audit provenance: this is the unchanged baseline candidate."
    detail = detail if isinstance(detail, Mapping) else {}
    surface = str(detail.get("surface", "unspecified surface"))
    before_hash = sha256_json(detail.get("before"))
    after_hash = sha256_json(detail.get("after"))
    return (
        f"Audit provenance: mutation family {family}; changed {surface}; "
        f"before digest {before_hash}; after digest {after_hash}."
    )


_EXACT_HINT_PATTERN = re.compile(
    r"(?:exact\s+label|certified\s+relation|\bis\s+valid\b|\bis\s+invalid\b|"
    r"\bequivalent\b|\bnon[_ -]?equivalent\b)",
    re.IGNORECASE,
)


def build_condition_manifest(
    pair_rows: Sequence[Mapping[str, Any]],
    provenance_by_candidate: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Freeze clean, true-hint, and position-matched deranged prompts."""

    candidate_ids = {str(row["candidate_id"]) for row in pair_rows}
    if set(provenance_by_candidate) != candidate_ids:
        raise ManifestError("provenance_key_mismatch")
    pair_ids = sorted({str(row["pair_id"]) for row in pair_rows})
    if len(pair_ids) < 2:
        raise ManifestError("derangement_requires_two_pairs")
    derangement = {
        pair_id: pair_ids[(index + 1) % len(pair_ids)] for index, pair_id in enumerate(pair_ids)
    }
    by_pair_position = {(str(row["pair_id"]), int(row["pair_position"])): row for row in pair_rows}
    rows: list[JsonDict] = []
    for candidate in pair_rows:
        candidate_id = str(candidate["candidate_id"])
        pair_id = str(candidate["pair_id"])
        position = int(candidate["pair_position"])
        true_record = provenance_by_candidate[candidate_id]
        donor_pair = derangement[pair_id]
        donor = by_pair_position[(donor_pair, position)]
        decoy_record = provenance_by_candidate[str(donor["candidate_id"])]
        true_text = str(true_record.get("provenance_text", ""))
        decoy_text = str(decoy_record.get("provenance_text", ""))
        if not true_text or not decoy_text:
            raise ManifestError("empty_provenance_text")
        if _EXACT_HINT_PATTERN.search(true_text) or _EXACT_HINT_PATTERN.search(decoy_text):
            raise ManifestError("exact_label_text_in_hint")
        for condition, hint, source_pair in (
            ("clean", "", None),
            ("true_provenance_hint", true_text, pair_id),
            ("permuted_decoy_hint", decoy_text, donor_pair),
        ):
            preamble = SEMANTIC_TASK
            if hint:
                preamble += f"AUDIT_HINT:\n{hint}\n"
            preamble += "MAPPING_CANDIDATE:\n"
            prompt_text = preamble + str(candidate["candidate_text"]) + PROMPT_SUFFIX
            rows.append(
                {
                    "pair_id": pair_id,
                    "candidate_id": candidate_id,
                    "candidate_hash": candidate["candidate_hash"],
                    "pair_position": position,
                    "condition": condition,
                    "candidate_text": candidate["candidate_text"],
                    "prompt_preamble": preamble,
                    "prompt_suffix": PROMPT_SUFFIX,
                    "prompt_text": prompt_text,
                    "prompt_hash": sha256_text(prompt_text),
                    "hint_hash": sha256_text(hint),
                    "provenance_source_pair_id": source_pair,
                }
            )
    payload = {
        "rows": rows,
        "derangement": derangement,
        "conditions": list(CONDITIONS),
        "prefix_fractions": list(PREFIX_FRACTIONS),
    }
    return {**payload, "condition_manifest_hash": sha256_json(payload)}


def scoring_blocks(condition_rows: Sequence[Mapping[str, Any]]) -> list[list[JsonDict]]:
    """Strip controller-only fields and group model work by frozen pair."""

    errors = model_input_errors(condition_rows)
    if errors:
        raise ManifestError(";".join(errors))
    blocks: list[list[JsonDict]] = []
    pair_ids = sorted({str(row.get("pair_id")) for row in condition_rows})
    for pair_id in pair_ids:
        block = [
            {field: deepcopy(row[field]) for field in WORKER_INPUT_FIELDS}
            for row in condition_rows
            if row.get("pair_id") == pair_id
        ]
        blocks.append(block)
    return blocks


def freeze_token_prefixes(model: Any, candidate_text: str) -> list[JsonDict]:
    """Freeze increasing candidate-token prefixes with the embedded tokenizer."""

    tokens = list(model.tokenize(candidate_text.encode("utf-8"), add_bos=False, special=True))
    if not tokens:
        raise ManifestError("candidate_tokenization_empty")
    rows: list[JsonDict] = []
    previous = 0
    for ordinal, fraction in enumerate(PREFIX_FRACTIONS):
        count = min(len(tokens), max(previous + 1, int(math.ceil(len(tokens) * fraction))))
        if count > len(tokens):
            raise ManifestError("candidate_too_short_for_prefix_grid")
        prefix_tokens = tokens[:count]
        prefix_bytes = bytes(model.detokenize(prefix_tokens))
        rows.append(
            {
                "prefix_ordinal": ordinal,
                "fraction": fraction,
                "prefix_token_count": count,
                "candidate_token_count": len(tokens),
                "prefix_token_ids": prefix_tokens,
                "prefix_token_hash": sha256_json(prefix_tokens),
                "prefix_bytes_hex": prefix_bytes.hex(),
                "prefix_text": prefix_bytes.decode("utf-8", errors="strict"),
                "tokenizer_source": "embedded_gguf",
            }
        )
        previous = count
    if rows[-1]["prefix_token_ids"] != tokens:
        raise ManifestError("full_prefix_token_mismatch")
    return rows


_TERMINAL_PATTERN = re.compile(r"FINAL_CHOICE:\s*(VALID|INVALID)")


def extract_terminal_choice(text: str) -> str:
    """Extract one final choice only when its marker ends the completion."""

    matches = _TERMINAL_PATTERN.findall(text)
    terminal = re.search(r"FINAL_CHOICE:\s*(VALID|INVALID)\s*\Z", text)
    if len(matches) != 1 or terminal is None:
        raise ChoiceError("terminal_choice_missing_or_ambiguous")
    return str(terminal.group(1))


def _log_probability_row(logits: np.ndarray, token_id: int) -> JsonDict:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if token_id < 0 or token_id >= values.size:
        raise ChoiceError("choice_token_out_of_range")
    safe = np.where(np.isfinite(values), values, -1.0e30)
    maximum = float(np.max(safe))
    logsumexp = maximum + math.log(float(np.exp(safe - maximum).sum()))
    return {
        "selected_token_logit": float(safe[token_id]),
        "selected_token_log_probability": float(safe[token_id] - logsumexp),
        "full_vocabulary_logsumexp": logsumexp,
        "vocabulary_size": int(values.size),
        "full_logit_vector_hash": sha256_bytes(
            np.asarray(logits, dtype=np.float32).reshape(-1).tobytes(order="C")
        ),
    }


def score_terminal_choices(
    model: Any,
    *,
    context_text: str,
    own_choice: str,
    prefix_ordinal: int,
) -> JsonDict:
    """Teacher-force both choices and normalize their exact sequence masses."""

    if own_choice not in {"VALID", "INVALID"}:
        raise ChoiceError("own_choice_invalid")
    context_tokens = list(model.tokenize(context_text.encode("utf-8"), add_bos=True, special=True))
    if not context_tokens:
        raise ChoiceError("choice_context_empty")
    raw_rows: list[JsonDict] = []
    sequence_log_probabilities: dict[str, float] = {}
    parity = True
    for choice in ("VALID", "INVALID"):
        choice_text = " " + choice
        choice_tokens = list(
            model.tokenize(choice_text.encode("utf-8"), add_bos=False, special=True)
        )
        full_tokens = context_tokens + choice_tokens
        combined = list(
            model.tokenize((context_text + choice_text).encode("utf-8"), add_bos=True, special=True)
        )
        choice_parity = bool(choice_tokens and combined == full_tokens)
        parity = parity and choice_parity
        if not choice_parity:
            raise ChoiceError(f"choice_tokenizer_parity_failed:{choice}")
        model.reset()
        model.eval(full_tokens)
        scores = np.asarray(model.scores)
        start = len(context_tokens) - 1
        if start < 0 or start + len(choice_tokens) > scores.shape[0]:
            raise ChoiceError("choice_logit_alignment_unavailable")
        log_probability = 0.0
        for position, token_id in enumerate(choice_tokens):
            features = _log_probability_row(scores[start + position], int(token_id))
            log_probability += float(features["selected_token_log_probability"])
            token_bytes = bytes(model.detokenize([int(token_id)]))
            raw_rows.append(
                {
                    "prefix_ordinal": prefix_ordinal,
                    "choice_option": choice,
                    "choice_token_position": position,
                    "choice_token_id": int(token_id),
                    "choice_token_bytes_hex": token_bytes.hex(),
                    "choice_token_text": token_bytes.decode("utf-8", errors="replace"),
                    "tokenizer_source": "embedded_gguf",
                    "tokenizer_parity_passed": choice_parity,
                    **features,
                }
            )
        sequence_log_probabilities[choice] = log_probability
    maximum = max(sequence_log_probabilities.values())
    weights = {
        choice: math.exp(value - maximum) for choice, value in sequence_log_probabilities.items()
    }
    denominator = sum(weights.values())
    probabilities = {choice: value / denominator for choice, value in weights.items()}
    entropy = -sum(value * math.log(max(value, 1.0e-300)) for value in probabilities.values())
    return {
        "own_choice_probability": probabilities[own_choice],
        "valid_probability": probabilities["VALID"],
        "invalid_probability": probabilities["INVALID"],
        "predicted_choice": max(probabilities, key=probabilities.get),
        "uncertainty": entropy,
        "choice_sequence_log_probabilities": sequence_log_probabilities,
        "tokenizer_parity_passed": parity,
        "choice_token_rows": raw_rows,
    }


def summarize_curve(curve: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one ordered probability curve to the preregistered metrics."""

    if not curve:
        raise ValueError("commitment_curve_empty")
    fractions = [float(row["fraction"]) for row in curve]
    if fractions != sorted(fractions) or len(fractions) != len(set(fractions)):
        raise ValueError("commitment_curve_order_invalid")
    masses = [float(row["own_choice_probability"]) for row in curve]
    committed = [
        fraction
        for fraction, mass in zip(fractions, masses, strict=True)
        if mass >= COMMITMENT_THRESHOLD
    ]
    predicted = [str(row["predicted_choice"]) for row in curve]
    return {
        "first_commitment_latency": committed[0] if committed else 1.0,
        "commitment_reached": bool(committed),
        "commitment_range": max(masses) - min(masses),
        "mean_uncommitted_mass": float(np.mean([1.0 - mass for mass in masses])),
        "mean_uncertainty": float(np.mean([float(row["uncertainty"]) for row in curve])),
        "choice_flip_count": sum(left != right for left, right in zip(predicted, predicted[1:])),
    }


def completed_unit_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    """Return unique candidate-condition-family keys or fail on duplicates."""

    keys = [
        (str(row.get("candidate_id")), str(row.get("condition")), str(row.get("model_id")))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise CheckpointError("duplicate_unit_key")
    return set(keys)


def write_checkpoint(
    path: Path,
    *,
    manifest_hash: str,
    block_key: str,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write one pair-family block and reject changed resume content."""

    completed_unit_keys(rows)
    document: JsonDict = {
        "schema": "carnot.exp6998.checkpoint.v1",
        "manifest_hash": manifest_hash,
        "blocks": [],
    }
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("manifest_hash") != manifest_hash:
            raise CheckpointError("manifest_hash_mismatch")
    block = {
        "block_key": block_key,
        "rows": [deepcopy(dict(row)) for row in rows],
    }
    block["block_hash"] = sha256_json(block)
    existing = next(
        (item for item in document.get("blocks", []) if item.get("block_key") == block_key),
        None,
    )
    if existing is not None:
        if existing != block:
            raise CheckpointError("checkpoint_block_mismatch")
        return {
            "path": str(path),
            "block_key": block_key,
            "manifest_hash": manifest_hash,
            "checkpoint_hash": sha256_json(document),
            "recovered": True,
            "passed": True,
        }
    document["blocks"].append(block)
    write_json_atomic(path, document)
    return {
        "path": str(path),
        "block_key": block_key,
        "manifest_hash": manifest_hash,
        "checkpoint_hash": sha256_json(document),
        "recovered": False,
        "passed": True,
    }


def load_checkpoint(path: Path, *, manifest_hash: str) -> JsonDict:
    """Load one checkpoint only when its manifest and unit keys replay."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("manifest_hash") != manifest_hash:
        raise CheckpointError("manifest_hash_mismatch")
    all_rows = [row for block in document.get("blocks", []) for row in block.get("rows", [])]
    completed_unit_keys(all_rows)
    if len({block.get("block_key") for block in document.get("blocks", [])}) != len(
        document.get("blocks", [])
    ):
        raise CheckpointError("duplicate_block_key")
    return document


def _teardown_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        len(rows) == len(REQUIRED_MODEL_IDS)
        and {str(row.get("model_id")) for row in rows} == set(REQUIRED_MODEL_IDS)
        and all(
            row.get("passed") is True
            and row.get("process_exit_code") == 0
            and row.get("owned_process_absent") is True
            and row.get("port_release_confirmed") is True
            and row.get("model_close_called") is True
            and row.get("signals_sent") == []
            for row in rows
        )
    )


def completion_errors(
    pair_rows: Sequence[Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
    teardown_rows: Sequence[Mapping[str, Any]],
    vram_release_rows: Sequence[Mapping[str, Any]],
    *,
    label_denial_passed: bool,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
) -> list[str]:
    """Reduce integrity gates without using predictive value."""

    errors: list[str] = []
    if len(pair_rows) != expected_candidate_count:
        errors.append("pair_manifest_candidate_count")
    expected = expected_candidate_count * len(CONDITIONS) * len(REQUIRED_MODEL_IDS)
    if len(unit_rows) != expected:
        errors.append("unit_row_count")
    try:
        keys = completed_unit_keys(unit_rows)
    except CheckpointError:
        keys = set()
        errors.append("duplicate_unit_key")
    expected_keys = {
        (str(candidate.get("candidate_id")), condition, family)
        for candidate in pair_rows
        for condition in CONDITIONS
        for family in REQUIRED_MODEL_IDS
    }
    if keys != expected_keys:
        errors.append("unit_key_cross_product")
    candidates = {str(row.get("candidate_id")): row for row in pair_rows}
    if any(
        row.get("terminal") is not True
        or row.get("choice_extraction_passed") is not True
        or row.get("prefix_order_passed") is not True
        or row.get("tokenizer_parity_passed") is not True
        or not str(row.get("prompt_hash", "")).startswith("sha256:")
        or not str(row.get("curve_hash", "")).startswith("sha256:")
        or str(row.get("candidate_hash"))
        != str(candidates.get(str(row.get("candidate_id")), {}).get("candidate_hash"))
        for row in unit_rows
    ):
        errors.append("terminal_hash_or_tokenizer_incomplete")
    if any(row.get("live_cuda") is not True for row in unit_rows):
        errors.append("cuda_incomplete")
    if not label_denial_passed:
        errors.append("label_denial_incomplete")
    if not _teardown_complete(teardown_rows):
        errors.append("teardown_incomplete")
    if (
        len(vram_release_rows) != len(REQUIRED_MODEL_IDS)
        or {str(row.get("model_id")) for row in vram_release_rows} != set(REQUIRED_MODEL_IDS)
        or any(
            row.get("passed") is not True
            or int(row.get("max_residual_mb", VRAM_RELEASE_TOLERANCE_MB + 1))
            > VRAM_RELEASE_TOLERANCE_MB
            for row in vram_release_rows
        )
    ):
        errors.append("vram_release_incomplete")
    return list(dict.fromkeys(errors))


def join_labels(
    unit_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    teardown_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join exact labels only after every model process has exited."""

    if not _teardown_complete(teardown_rows):
        raise LabelJoinError("family_processes_not_exited")
    joined: list[JsonDict] = []
    for row in unit_rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in labels:
            raise LabelJoinError(f"label_missing:{candidate_id}")
        joined.append(
            {
                "candidate_id": candidate_id,
                "condition": row.get("condition"),
                "model_id": row.get("model_id"),
                "terminal_choice": row.get("terminal_choice"),
                "exact_label": labels[candidate_id],
                "choice_matches_exact_label": (row.get("terminal_choice") == "VALID")
                == (labels[candidate_id] == "equivalent"),
                "joined_after_all_model_processes_exit": True,
            }
        )
    return joined


def shortcut_detected(interval_rows: Sequence[Mapping[str, Any]]) -> int:
    """Require a true-only negative latency interval."""

    rows = {str(row.get("comparison")): row for row in interval_rows}
    true_row = rows.get("true_provenance_hint_minus_clean", {})
    decoy_row = rows.get("permuted_decoy_hint_minus_clean", {})
    return int(
        float(true_row.get("ci_high", math.inf)) < 0.0
        and float(decoy_row.get("ci_high", math.inf)) >= 0.0
    )


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record one precondition with explicit expected and observed values."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Promote the first failed check while preserving the complete list."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    live_duration_s: float,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    model_file_hashes: Mapping[str, Any],
    pair_manifest_rows: Sequence[Mapping[str, Any]] = (),
    pair_manifest_hash: str | None = None,
    condition_manifest_rows: Sequence[Mapping[str, Any]] = (),
    per_pair_results: Sequence[Mapping[str, Any]] = (),
    unit_rows: Sequence[Mapping[str, Any]] = (),
    terminal_choice_rows: Sequence[Mapping[str, Any]] = (),
    prefix_fraction_rows: Sequence[Mapping[str, Any]] = (),
    choice_token_rows: Sequence[Mapping[str, Any]] = (),
    commitment_curve_rows: Sequence[Mapping[str, Any]] = (),
    first_commitment_latency_rows: Sequence[Mapping[str, Any]] = (),
    commitment_range_rows: Sequence[Mapping[str, Any]] = (),
    uncommitted_mass_rows: Sequence[Mapping[str, Any]] = (),
    uncertainty_rows: Sequence[Mapping[str, Any]] = (),
    choice_flip_rows: Sequence[Mapping[str, Any]] = (),
    label_denial_rows: Sequence[Mapping[str, Any]] = (),
    process_isolation_rows: Sequence[Mapping[str, Any]] = (),
    late_label_join_rows: Sequence[Mapping[str, Any]] = (),
    paired_condition_delta_rows: Sequence[Mapping[str, Any]] = (),
    bootstrap_interval_rows: Sequence[Mapping[str, Any]] = (),
    family_stratum_rows: Sequence[Mapping[str, Any]] = (),
    gpu_runtime_rows: Sequence[Mapping[str, Any]] = (),
    lease_rows: Sequence[Mapping[str, Any]] = (),
    checkpoint_rows: Sequence[Mapping[str, Any]] = (),
    teardown_rows: Sequence[Mapping[str, Any]] = (),
    vram_release_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build a complete blocked, partial, null, or shortcut-positive artifact."""

    pairs = _copy_rows(pair_manifest_rows)
    units = _copy_rows(unit_rows)
    denial = _copy_rows(label_denial_rows)
    teardown = _copy_rows(teardown_rows)
    vram = _copy_rows(vram_release_rows)
    intervals = _copy_rows(bootstrap_interval_rows)
    preflight_passed = preconditions.get("all_passed") is True
    denial_passed = bool(denial and all(row.get("passed") is True for row in denial))
    integrity_errors = completion_errors(
        pairs,
        units,
        teardown,
        vram,
        label_denial_passed=denial_passed,
    )
    complete = bool(preflight_passed and not integrity_errors)
    shortcut_score = shortcut_detected(intervals) if complete else 0
    checks = _copy_rows(preconditions.get("checks", []))
    if preflight_passed:
        checks.append(gate_check("commitment_control_completion", [], integrity_errors))
    if not preflight_passed:
        verdict_class = "blocked"
        honest_verdict = "blocked_three_family_commitment_controls"
    elif not complete:
        verdict_class = "partial"
        honest_verdict = "partial_three_family_commitment_controls"
    elif shortcut_score == 1:
        verdict_class = "positive"
        honest_verdict = "complete: shortcut_commitment_detected"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_shortcut_commitment_not_detected"
    conditions = _copy_rows(condition_manifest_rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "live_duration_s": float(live_duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "MODEL_SPECS": _copy_rows(model_specs),
        "models_used": [str(row.get("hf_id")) for row in model_specs if row.get("hf_id")],
        "model_file_hashes": deepcopy(dict(model_file_hashes)),
        "pair_manifest_rows": pairs,
        "pair_manifest_hash": pair_manifest_hash,
        "condition_manifest_rows": conditions,
        "prompt_hash_rows": [
            {
                "candidate_id": row.get("candidate_id"),
                "condition": row.get("condition"),
                "prompt_hash": row.get("prompt_hash"),
            }
            for row in conditions
        ],
        "candidate_hash_rows": [
            {"candidate_id": row.get("candidate_id"), "candidate_hash": row.get("candidate_hash")}
            for row in pairs
        ],
        "rows": deepcopy(units),
        "per_pair_results": _copy_rows(per_pair_results),
        "per_candidate_condition_model_rows": units,
        "terminal_choice_rows": _copy_rows(terminal_choice_rows),
        "prefix_fraction_rows": _copy_rows(prefix_fraction_rows),
        "choice_token_rows": _copy_rows(choice_token_rows),
        "commitment_curve_rows": _copy_rows(commitment_curve_rows),
        "first_commitment_latency_rows": _copy_rows(first_commitment_latency_rows),
        "commitment_range_rows": _copy_rows(commitment_range_rows),
        "uncommitted_mass_rows": _copy_rows(uncommitted_mass_rows),
        "uncertainty_rows": _copy_rows(uncertainty_rows),
        "choice_flip_rows": _copy_rows(choice_flip_rows),
        "label_denial_rows": denial,
        "process_isolation_rows": _copy_rows(process_isolation_rows),
        "late_label_join_rows": _copy_rows(late_label_join_rows),
        "paired_condition_delta_rows": _copy_rows(paired_condition_delta_rows),
        "bootstrap_interval_rows": intervals,
        "family_stratum_rows": _copy_rows(family_stratum_rows),
        "gpu_runtime_rows": _copy_rows(gpu_runtime_rows),
        "lease_rows": _copy_rows(lease_rows),
        "checkpoint_rows": _copy_rows(checkpoint_rows),
        "teardown_rows": teardown,
        "vram_release_rows": vram,
        "expected_unit_count": EXPECTED_UNIT_COUNT,
        "observed_unit_count": len(units),
        "commitment_control_complete_score": int(complete),
        "shortcut_commitment_detected_score": int(shortcut_score),
        "audit_only_control": True,
        "learner_feature_allowed": False,
        "verifier_fit_performed": False,
        "self_commitment_paper_reproduction_claimed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _contains_vectors(value: Any) -> bool:
    forbidden = {"logits", "full_vocabulary_logits", "full_logit_vector", "probabilities"}
    return any(key.casefold() in forbidden for key, _path, _item in _nested_items(value))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, projections, bare values, policy flags, and digest."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append(f"required_fields_missing:{sorted(missing)}")
        return errors
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "expected_unit_count",
        "observed_unit_count",
        "commitment_control_complete_score",
        "shortcut_commitment_detected_score",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"not_bare_int:{field}")
    required_flags = {
        "audit_only_control": True,
        "learner_feature_allowed": False,
        "verifier_fit_performed": False,
        "self_commitment_paper_reproduction_claimed": False,
        "verifier_is_oracle": False,
    }
    for field, expected in required_flags.items():
        if type(artifact.get(field)) is not bool or artifact.get(field) is not expected:
            errors.append(f"policy_flag_mismatch:{field}")
    if artifact.get("expected_unit_count") != EXPECTED_UNIT_COUNT:
        errors.append("expected_unit_count_mismatch")
    units = artifact.get("per_candidate_condition_model_rows", [])
    if artifact.get("rows") != units or artifact.get("observed_unit_count") != len(units):
        errors.append("unit_projection_mismatch")
    if _contains_vectors(artifact.get("choice_token_rows", [])):
        errors.append("full_vocabulary_vector_present")
    score = artifact.get("commitment_control_complete_score")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if verdict_class == "blocked" and not verdict.startswith("blocked_"):
        errors.append("blocked_verdict_prefix_mismatch")
    if verdict_class == "partial" and not verdict.startswith("partial_"):
        errors.append("partial_verdict_prefix_mismatch")
    if verdict_class == "null" and not verdict.startswith("complete_null"):
        errors.append("null_verdict_prefix_mismatch")
    if verdict_class == "positive" and not verdict.startswith("complete:"):
        errors.append("positive_verdict_prefix_mismatch")
    if score == 1 and artifact.get("observed_unit_count") != EXPECTED_UNIT_COUNT:
        errors.append("complete_row_count_mismatch")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def generate_terminal_choice(model: Any, prompt_text: str) -> JsonDict:
    """Ask the model for one grammar-bounded terminal decision."""

    from llama_cpp import LlamaGrammar

    grammar = LlamaGrammar.from_string('root ::= "FINAL_CHOICE: VALID" | "FINAL_CHOICE: INVALID"')
    response = model.create_completion(
        prompt=prompt_text,
        max_tokens=12,
        temperature=0.0,
        top_p=1.0,
        seed=RANDOM_SEED,
        grammar=grammar,
        echo=False,
    )
    text = str(response["choices"][0]["text"])
    choice = extract_terminal_choice(text)
    return {
        "terminal_choice": choice,
        "raw_completion": text,
        "raw_completion_hash": sha256_text(text),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "choice_extraction_passed": True,
    }


def measure_condition(model: Any, row: Mapping[str, Any], *, model_id: str) -> JsonDict:
    """Generate one choice and measure its mass across candidate prefixes."""

    generated = generate_terminal_choice(model, str(row["prompt_text"]))
    prefixes = freeze_token_prefixes(model, str(row["candidate_text"]))
    curve: list[JsonDict] = []
    token_rows: list[JsonDict] = []
    prefix_rows: list[JsonDict] = []
    identity = {
        "pair_id": row["pair_id"],
        "candidate_id": row["candidate_id"],
        "condition": row["condition"],
        "model_id": model_id,
    }
    for prefix in prefixes:
        context_text = (
            str(row["prompt_preamble"])
            + str(prefix["prefix_text"])
            + str(row["prompt_suffix"])
            + "FINAL_CHOICE:"
        )
        scored = score_terminal_choices(
            model,
            context_text=context_text,
            own_choice=str(generated["terminal_choice"]),
            prefix_ordinal=int(prefix["prefix_ordinal"]),
        )
        prefix_rows.append(
            {
                **identity,
                **{key: value for key, value in prefix.items() if key != "prefix_text"},
                "prefix_text_hash": sha256_text(str(prefix["prefix_text"])),
                "decision_context_hash": sha256_text(context_text),
                "full_fraction_matches_candidate": (
                    prefix["fraction"] != 1.0
                    or prefix["prefix_token_count"] == prefix["candidate_token_count"]
                ),
            }
        )
        point = {
            **identity,
            "prefix_ordinal": prefix["prefix_ordinal"],
            "fraction": prefix["fraction"],
            "own_choice_probability": scored["own_choice_probability"],
            "valid_probability": scored["valid_probability"],
            "invalid_probability": scored["invalid_probability"],
            "predicted_choice": scored["predicted_choice"],
            "uncertainty": scored["uncertainty"],
            "tokenizer_parity_passed": scored["tokenizer_parity_passed"],
        }
        curve.append(point)
        token_rows.extend({**identity, **item} for item in scored["choice_token_rows"])
    metrics = summarize_curve(curve)
    curve_hash = sha256_json(curve)
    unit = {
        **identity,
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "terminal": True,
        "terminal_choice": generated["terminal_choice"],
        "choice_extraction_passed": True,
        "prefix_order_passed": [point["fraction"] for point in curve] == list(PREFIX_FRACTIONS),
        "tokenizer_parity_passed": all(point["tokenizer_parity_passed"] is True for point in curve),
        "live_cuda": False,
        "curve_hash": curve_hash,
        **metrics,
    }
    terminal = {**identity, **generated}
    return {
        "unit_row": unit,
        "terminal_choice_row": terminal,
        "prefix_fraction_rows": prefix_rows,
        "choice_token_rows": token_rows,
        "commitment_curve_row": {**identity, "curve_hash": curve_hash, "points": curve},
    }


def metric_projection_rows(unit_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project each preregistered scalar to its own audit table."""

    identity_fields = ("pair_id", "candidate_id", "condition", "model_id")

    def project(metric: str, output: str | None = None) -> list[JsonDict]:
        return [
            {
                **{field: row.get(field) for field in identity_fields},
                (output or metric): row.get(metric),
            }
            for row in unit_rows
        ]

    return {
        "first_commitment_latency_rows": project("first_commitment_latency"),
        "commitment_range_rows": project("commitment_range"),
        "uncommitted_mass_rows": project("mean_uncommitted_mass"),
        "uncertainty_rows": project("mean_uncertainty"),
        "choice_flip_rows": project("choice_flip_count"),
    }


def paired_latency_rows(unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Average candidates within each pair before comparing conditions."""

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in unit_rows:
        grouped[(str(row["pair_id"]), str(row["model_id"]), str(row["condition"]))].append(
            float(row["first_commitment_latency"])
        )
    output: list[JsonDict] = []
    for pair_id in sorted({key[0] for key in grouped}):
        for model_id in REQUIRED_MODEL_IDS:
            clean_values = grouped.get((pair_id, model_id, "clean"), [])
            if not clean_values:
                continue
            clean = float(np.mean(clean_values))
            for condition in CONDITIONS[1:]:
                values = grouped.get((pair_id, model_id, condition), [])
                if not values:
                    continue
                hinted = float(np.mean(values))
                output.append(
                    {
                        "pair_id": pair_id,
                        "model_id": model_id,
                        "comparison": f"{condition}_minus_clean",
                        "clean_latency": clean,
                        "hinted_latency": hinted,
                        "latency_delta": hinted - clean,
                        "candidate_count": len(values),
                    }
                )
    return output


def bootstrap_interval(
    values: Sequence[float], *, seed: int = RANDOM_SEED, draws: int = 2_000
) -> JsonDict:
    """Compute one deterministic percentile interval for paired deltas."""

    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "sample_count": 0}
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "sample_count": int(len(array)),
        "bootstrap_draw_count": draws,
    }


def aggregate_condition_statistics(
    unit_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Build pair deltas, overall intervals, and model-family strata."""

    paired = paired_latency_rows(unit_rows)
    intervals: list[JsonDict] = []
    strata: list[JsonDict] = []
    for comparison_ordinal, comparison in enumerate(
        ("true_provenance_hint_minus_clean", "permuted_decoy_hint_minus_clean")
    ):
        values = [float(row["latency_delta"]) for row in paired if row["comparison"] == comparison]
        intervals.append(
            {
                "comparison": comparison,
                "stratum": "all_families",
                **bootstrap_interval(values, seed=RANDOM_SEED + comparison_ordinal),
            }
        )
        for family_ordinal, model_id in enumerate(REQUIRED_MODEL_IDS):
            family_values = [
                float(row["latency_delta"])
                for row in paired
                if row["comparison"] == comparison and row["model_id"] == model_id
            ]
            strata.append(
                {
                    "comparison": comparison,
                    "model_id": model_id,
                    **bootstrap_interval(
                        family_values,
                        seed=RANDOM_SEED + 100 + 10 * comparison_ordinal + family_ordinal,
                    ),
                }
            )
    return paired, intervals, strata


def per_pair_summaries(
    unit_rows: Sequence[Mapping[str, Any]],
    joined_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep pair-level condition means and post-exit exact-label audit counts."""

    pair_ids = sorted({str(row.get("pair_id")) for row in unit_rows})
    labels_by_candidate = {str(row["candidate_id"]): str(row["exact_label"]) for row in joined_rows}
    output: list[JsonDict] = []
    for pair_id in pair_ids:
        pair = [row for row in unit_rows if row.get("pair_id") == pair_id]
        condition_means = {
            condition: float(
                np.mean(
                    [
                        float(row["first_commitment_latency"])
                        for row in pair
                        if row.get("condition") == condition
                    ]
                )
            )
            for condition in CONDITIONS
        }
        candidate_ids = sorted({str(row["candidate_id"]) for row in pair})
        output.append(
            {
                "pair_id": pair_id,
                "candidate_ids": candidate_ids,
                "unit_count": len(pair),
                "condition_mean_latency": condition_means,
                "exact_labels_opened_after_exit": [
                    labels_by_candidate[candidate] for candidate in candidate_ids
                ]
                if all(candidate in labels_by_candidate for candidate in candidate_ids)
                else [],
                "terminal": len(pair) == len(REQUIRED_MODEL_IDS) * len(CONDITIONS) * 2,
            }
        )
    return output


def runtime_label_denial_rows() -> list[JsonDict]:
    """Mutate each denied key and prove that worker validation rejects it."""

    rows: list[JsonDict] = []
    for field in sorted(FORBIDDEN_MODEL_FIELDS):
        errors = model_input_errors({"candidate_id": "probe", "nested": {field: "denied"}})
        rows.append(
            {
                "mutation_field": field,
                "denied_path": f"$.nested.{field}",
                "errors": errors,
                "passed": any(field in error for error in errors),
            }
        )
    return rows


def _selected_json_fields(path: Path, fields: Sequence[str]) -> JsonDict:  # pragma: no cover
    """Read selected top-level fields without retaining authority sibling arrays."""

    import mmap

    selected: JsonDict = {}
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for field in fields:
            marker = f'\n  "{field}":'.encode()
            marker_at = data.find(marker)
            if marker_at < 0:
                raise ManifestError(f"source_field_missing:{path.name}:{field}")
            start = marker_at + len(marker)
            while data[start] in b" \t\r\n":
                start += 1
            opening = data[start]
            if opening in (ord("["), ord("{")):
                closing = ord("]") if opening == ord("[") else ord("}")
                depth = 0
                quoted = False
                escaped = False
                end = start
                while end < len(data):
                    byte = data[end]
                    if quoted:
                        if escaped:
                            escaped = False
                        elif byte == ord("\\"):
                            escaped = True
                        elif byte == ord('"'):
                            quoted = False
                    elif byte == ord('"'):
                        quoted = True
                    elif byte == opening:
                        depth += 1
                    elif byte == closing:
                        depth -= 1
                        if depth == 0:
                            end += 1
                            break
                    end += 1
            elif opening == ord('"'):
                end = start + 1
                escaped = False
                while end < len(data):
                    byte = data[end]
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        end += 1
                        break
                    end += 1
            else:
                comma = data.find(b",", start)
                newline = data.find(b"\n", start)
                candidates = [value for value in (comma, newline) if value >= 0]
                end = min(candidates) if candidates else len(data)
            selected[field] = json.loads(data[start:end])
    return selected


def _load_label_free_sources() -> tuple[JsonDict, JsonDict, JsonDict]:  # pragma: no cover
    fixture = _selected_json_fields(EXP6984_PATH, ("per_candidate_rows", "split_rows"))
    learner = _selected_json_fields(
        EXP6997_PATH, ("blinded_learner_view_ready_score", "feature_allowlist")
    )
    runtime = _selected_json_fields(EXP6973_PATH, ("lease_aware_runtime_ready_score",))
    return fixture, learner, runtime


def load_provenance(
    pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, JsonDict], dict[str, str]]:  # pragma: no cover
    """Project mutation text and join keys without returning authority records."""

    selected_ids = {str(row["candidate_id"]) for row in pair_rows}
    provenance: dict[str, JsonDict] = {}
    candidate_keys: dict[str, str] = {}
    with MUTATION_SIDECAR_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source = record.get("source", {})
            candidate_id = str(source.get("candidate_id", ""))
            if candidate_id not in selected_ids:
                continue
            mutation = record.get("mutation", {})
            pair = next(row for row in pair_rows if row["candidate_id"] == candidate_id)
            provenance[candidate_id] = {
                "pair_id": pair["pair_id"],
                "pair_position": pair["pair_position"],
                "provenance_text": render_provenance_text(mutation),
            }
            candidate_keys[candidate_id] = str(record.get("candidate_key", ""))
    if set(provenance) != selected_ids or not all(candidate_keys.values()):
        raise ManifestError("provenance_key_mismatch")
    return provenance, candidate_keys


def load_late_labels(candidate_keys: Mapping[str, str]) -> dict[str, str]:  # pragma: no cover
    """Open the dedicated label sidecar only after all workers have exited."""

    by_key: dict[str, str] = {}
    with LABEL_SIDECAR_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_key[str(row.get("candidate_key"))] = str(row.get("exact_label"))
    labels = {
        candidate_id: by_key[key] for candidate_id, key in candidate_keys.items() if key in by_key
    }
    if set(labels) != set(candidate_keys):
        raise LabelJoinError("late_label_catalog_incomplete")
    return labels


def _storage_writable(path: Path) -> bool:  # pragma: no cover
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".exp6998-write-probe-", dir=path)
        os.close(descriptor)
        Path(temporary).unlink()
        return True
    except OSError:
        return False


def _memory_rows(gpu: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    return [
        {
            "index": int(row.get("index", -1)),
            "uuid": str(row.get("uuid", "")),
            "memory_used_mb": int(row.get("memory_used_mb", 0) or 0),
            "memory_free_mb": int(row.get("memory_free_mb", 0) or 0),
        }
        for row in gpu.get("devices", [])
    ]


def _lease_preflight(
    devices: Sequence[Mapping[str, Any]], runtime_dir: Path
) -> JsonDict:  # pragma: no cover
    rows: list[JsonDict] = []
    for device in devices:
        uuid = str(device.get("uuid", ""))
        path = lease_api.journal_path_for(runtime_dir, uuid)
        if not path.exists():
            rows.append({"device_uuid": uuid, "classification": "absent", "free": True})
            continue
        try:
            document = lease_api.read_journal(path)
            owner = document.get("owner", {})
            live = bool(
                document.get("released") is not True
                and isinstance(owner, Mapping)
                and isinstance(owner.get("pid"), int)
                and isinstance(owner.get("pid_start_ticks"), int)
                and lease_api.process_start_matches(owner["pid"], owner["pid_start_ticks"])
            )
            rows.append(
                {
                    "device_uuid": uuid,
                    "classification": "live_foreign" if live else "released_or_stale",
                    "free": not live,
                    "journal_hash": sha256_json(document),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "device_uuid": uuid,
                    "classification": "unreadable",
                    "free": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"rows": rows, "free": len(rows) == 2 and all(row["free"] for row in rows)}


def _source_hashes() -> dict[str, str | None]:  # pragma: no cover
    hashes = {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}
    hashes.update(
        {
            "module": sha256_file(Path(__file__)),
            "wrapper": sha256_file(
                REPO_ROOT
                / "scripts/experiments/experiment_6998_three_family_commitment_controls.py"
            ),
            "tests": sha256_file(
                REPO_ROOT / "tests/python/test_experiment_6998_three_family_commitment_controls.py"
            ),
            "spec": sha256_file(REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md"),
        }
    )
    return hashes


def collect_preconditions(
    *,
    frozen: Mapping[str, Any],
    learner: Mapping[str, Any],
    runtime: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    checkpoint_root: Path,
    lease_runtime_dir: Path,
    manifest_error: str | None = None,
) -> JsonDict:  # pragma: no cover
    """Check all frozen inputs and live resources before model inference."""

    hashes = _source_hashes()
    gpu = gpu_inventory()
    devices = list(gpu.get("devices", []))
    binding = llama_cpp_probe()
    lease = _lease_preflight(devices, lease_runtime_dir)
    tokenizers = [embedded_tokenizer_probe(row) for row in model_specs]
    upstream_hashes = {key: hashes.get(key) for key in EXPECTED_SOURCE_HASHES}
    pair_rows = list(frozen.get("rows", []))
    source_groups = {str(row.get("source_group_id")) for row in pair_rows}
    checks = [
        gate_check(
            "blinded_learner_view_ready_score",
            1,
            learner.get("blinded_learner_view_ready_score"),
        ),
        gate_check(
            "lease_aware_runtime_ready_score", 1, runtime.get("lease_aware_runtime_ready_score")
        ),
        gate_check("exact_source_hashes", EXPECTED_SOURCE_HASHES, upstream_hashes),
        gate_check("pair_manifest_error", None, manifest_error),
        gate_check("frozen_pair_count", EXPECTED_PAIR_COUNT, len(source_groups)),
        gate_check("frozen_candidate_count", EXPECTED_CANDIDATE_COUNT, len(pair_rows)),
        gate_check(
            "learner_allowlist_excludes_commitment",
            False,
            any(
                "commitment" in str(field).casefold()
                for field in learner.get("feature_allowlist", [])
            ),
        ),
        gate_check("exact_model_specs", [], model_spec_errors(model_specs)),
        gate_check(
            "all_three_gguf_files",
            {family: True for family in REQUIRED_MODEL_IDS},
            {
                str(row.get("hf_id")): Path(str(row.get("model_path", ""))).is_file()
                for row in model_specs
            },
        ),
        gate_check(
            "two_cuda_devices",
            2,
            len(devices),
            passed=gpu.get("query_ok") is True and len(devices) == 2,
        ),
        gate_check(
            "cuda_capable_llama_cpp",
            {"importable": True, "gpu_offload": True},
            {"importable": binding.get("importable"), "gpu_offload": binding.get("gpu_offload")},
        ),
        gate_check("free_task_lease", True, lease.get("free")),
        gate_check("writable_checkpoints", True, _storage_writable(checkpoint_root)),
        gate_check(
            "embedded_tokenizers",
            {family: True for family in REQUIRED_MODEL_IDS},
            {str(row.get("model_id")): row.get("passed") for row in tokenizers},
        ),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "gpu_topology": gpu,
        "baseline_gpu_memory_rows": _memory_rows(gpu),
        "llama_cpp": binding,
        "lease_preflight": lease,
        "embedded_tokenizer_rows": tokenizers,
        "pair_manifest_frozen_before_label_open": manifest_error is None,
        "exact_labels_opened": False,
    }


def _wait_for_vram_release(
    baseline_rows: Sequence[Mapping[str, Any]], model_id: str
) -> JsonDict:  # pragma: no cover
    deadline = time.monotonic() + 180.0
    latest = gpu_inventory()
    while True:
        after_rows = _memory_rows(latest)
        row = build_vram_release_row(
            model_id=model_id,
            baseline_rows=baseline_rows,
            after_rows=after_rows,
        )
        row["after_rows"] = after_rows
        row["max_residual_mb"] = max(0, int(row.get("max_increase_mb", 0) or 0))
        if row.get("passed") is True or time.monotonic() >= deadline:
            return row
        time.sleep(1.0)
        latest = gpu_inventory()


def _acquire_leases(
    model: Mapping[str, Any],
    devices: Sequence[Mapping[str, Any]],
    runtime_dir: Path,
) -> tuple[list[Any], list[JsonDict]]:  # pragma: no cover
    leases: list[Any] = []
    rows: list[JsonDict] = []
    try:
        for device in sorted(devices, key=lambda item: int(item.get("index", 0))):
            lease = lease_api.GpuLease.acquire(
                runtime_dir=runtime_dir,
                task_id=EXPERIMENT_ID,
                device_uuid=str(device["uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
                ttl_s=LEASE_TTL_S,
            )
            lease.transition("admitted")
            lease.transition("loading")
            owner = lease.owner_receipt()
            leases.append(lease)
            rows.append(
                {
                    "model_id": model["hf_id"],
                    "device_uuid": device["uuid"],
                    "task_id": EXPERIMENT_ID,
                    "owner_pid": owner.get("pid"),
                    "owner_pid_start_ticks": owner.get("pid_start_ticks"),
                    "owner_verified": owner.get("task_id") == EXPERIMENT_ID,
                    "journal_after_acquisition": deepcopy(lease.document),
                    "journal_after_release": None,
                    "release_receipt": {},
                    "signals_sent": [],
                    "consistent": False,
                }
            )
    except Exception:
        for lease in leases:
            lease.close()
        raise
    return leases, rows


def _release_leases(
    leases: Sequence[Any],
    rows: list[JsonDict],
    *,
    complete: bool,
    vram_release: Mapping[str, Any],
    exit_code: int,
) -> None:  # pragma: no cover
    after_by_uuid = {
        str(row.get("uuid")): int(row.get("memory_used_mb", 0) or 0)
        for row in vram_release.get("after_rows", [])
    }
    for lease, row in zip(leases, rows, strict=True):
        try:
            phase = str(lease.document.get("phase"))
            if phase in {"resident", "inferencing"}:
                lease.transition("unloading")
                phase = "unloading"
            if phase == "unloading" and vram_release.get("passed") is True:
                lease.transition(
                    "validating",
                    vram_mb=after_by_uuid.get(lease.device_uuid, 0),
                    exit_code=exit_code,
                    unload_observed=True,
                )
                lease.transition("terminal_complete" if complete else "terminal_blocked")
            elif phase in {"preflight", "admitted", "loading"}:
                lease.transition("terminal_blocked")
            row["release_receipt"] = lease.release()
            row["journal_after_release"] = deepcopy(lease.document)
            validation_errors = lease_api.validate_journal_document(
                lease.document, check_freshness=False
            )
            row["consistent"] = bool(
                not validation_errors
                and lease.document.get("released") is True
                and lease.document.get("phase") in lease_api.TERMINAL_PHASES
                and row["release_receipt"].get("signals_sent") == []
            )
            row["journal_validation_errors"] = validation_errors
        except lease_api.LeaseError as exc:
            lease.close()
            row["journal_validation_errors"] = [f"{type(exc).__name__}: {exc}"]


def _score_worker(
    *, payload_path: Path, output_path: Path, ready_path: Path, port: int
) -> int:  # pragma: no cover
    """Run one label-denied family worker and checkpoint each pair block."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    model: Any = None
    close_called = False
    started = time.perf_counter()
    try:
        listener.bind(("127.0.0.1", int(port)))
        listener.listen(1)
        write_json_atomic(
            ready_path,
            {
                "pid": os.getpid(),
                "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                "port": port,
            },
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        denied = model_input_errors(payload.get("blocks", []))
        for block in payload.get("blocks", []):
            for row in block:
                unknown = set(row) - WORKER_INPUT_FIELDS
                if unknown:
                    denied.append(f"unknown_worker_fields:{sorted(unknown)}")
        if denied:
            raise ManifestError(";".join(denied))

        from llama_cpp import Llama

        model = Llama(
            model_path=str(payload["model_path"]),
            n_ctx=int(LOAD_CONFIG["n_ctx"]),
            n_gpu_layers=int(LOAD_CONFIG["n_gpu_layers"]),
            n_batch=int(LOAD_CONFIG["n_batch"]),
            n_ubatch=int(LOAD_CONFIG["n_ubatch"]),
            main_gpu=int(LOAD_CONFIG["main_gpu"]),
            split_mode=1,
            tensor_split=list(LOAD_CONFIG["tensor_split"]),
            logits_all=True,
            seed=RANDOM_SEED,
            use_mmap=True,
            use_mlock=False,
            verbose=True,
        )
        worker_root = Path(payload["worker_root"])
        worker_root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, list[JsonDict]] = {
            "unit_rows": [],
            "terminal_choice_rows": [],
            "prefix_fraction_rows": [],
            "choice_token_rows": [],
            "commitment_curve_rows": [],
            "checkpoint_rows": [],
        }
        for block_index, block in enumerate(payload["blocks"]):
            pair_id = str(block[0]["pair_id"])
            block_path = worker_root / f"block_{block_index:02d}.json"
            if block_path.exists():
                result = json.loads(block_path.read_text(encoding="utf-8"))
                if result.get("manifest_hash") != payload["manifest_hash"]:
                    raise CheckpointError("manifest_hash_mismatch")
                if result.get("pair_id") != pair_id:
                    raise CheckpointError("checkpoint_pair_mismatch")
                recovered = True
            else:
                result = {
                    "manifest_hash": payload["manifest_hash"],
                    "pair_id": pair_id,
                    "unit_rows": [],
                    "terminal_choice_rows": [],
                    "prefix_fraction_rows": [],
                    "choice_token_rows": [],
                    "commitment_curve_rows": [],
                }
                for row in block:
                    measured = measure_condition(model, row, model_id=str(payload["model_id"]))
                    result["unit_rows"].append(measured["unit_row"])
                    result["terminal_choice_rows"].append(measured["terminal_choice_row"])
                    result["prefix_fraction_rows"].extend(measured["prefix_fraction_rows"])
                    result["choice_token_rows"].extend(measured["choice_token_rows"])
                    result["commitment_curve_rows"].append(measured["commitment_curve_row"])
                write_json_atomic(block_path, result)
                recovered = False
            for field in (
                "unit_rows",
                "terminal_choice_rows",
                "prefix_fraction_rows",
                "choice_token_rows",
                "commitment_curve_rows",
            ):
                outputs[field].extend(result[field])
            outputs["checkpoint_rows"].append(
                {
                    "model_id": payload["model_id"],
                    "pair_id": pair_id,
                    "block_index": block_index,
                    "manifest_hash": payload["manifest_hash"],
                    "checkpoint_hash": sha256_json(result),
                    "recovered": recovered,
                    "row_count": len(result["unit_rows"]),
                    "passed": True,
                }
            )
        model.close()
        close_called = True
        model = None
        gc.collect()
        write_json_atomic(
            output_path,
            {
                "terminal": True,
                "model_id": payload["model_id"],
                "model_close_called": close_called,
                "live_duration_s": time.perf_counter() - started,
                "worker_input_field_names": sorted(WORKER_INPUT_FIELDS),
                "denied_fields_visible": [],
                **outputs,
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json_atomic(
            output_path,
            {
                "terminal": True,
                "model_close_called": close_called,
                "live_duration_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "unit_rows": [],
                "terminal_choice_rows": [],
                "prefix_fraction_rows": [],
                "choice_token_rows": [],
                "commitment_curve_rows": [],
                "checkpoint_rows": [],
            },
        )
        return 1
    finally:
        if model is not None:
            try:
                model.close()
            except Exception:  # noqa: BLE001
                pass
        listener.close()


def run_family_process(
    *,
    model: Mapping[str, Any],
    blocks: Sequence[Sequence[Mapping[str, Any]]],
    manifest_hash: str,
    work_root: Path,
    devices: Sequence[Mapping[str, Any]],
    lease_runtime_dir: Path,
) -> JsonDict:  # pragma: no cover
    """Run one family in an owned process, then prove teardown and VRAM release."""

    model_id = str(model["hf_id"])
    baseline_rows = _memory_rows(gpu_inventory())
    leases, lease_rows = _acquire_leases(model, devices, lease_runtime_dir)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()
    family_root = work_root / slug
    family_root.mkdir(parents=True, exist_ok=True)
    payload_path = family_root / "payload.json"
    output_path = family_root / "worker_output.json"
    ready_path = family_root / "ready.json"
    stdout_path = family_root / "stdout.log"
    stderr_path = family_root / "stderr.log"
    port = _choose_free_port()
    payload = {
        "model_id": model_id,
        "model_path": model["model_path"],
        "manifest_hash": manifest_hash,
        "blocks": [[deepcopy(dict(row)) for row in block] for block in blocks],
        "worker_root": str(family_root / "blocks"),
    }
    write_json_atomic(payload_path, payload)
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6998_three_family_commitment_controls",
        "--score-worker-payload",
        str(payload_path),
        "--score-worker-output",
        str(output_path),
        "--score-worker-ready",
        str(ready_path),
        "--score-worker-port",
        str(port),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    samples: list[JsonDict] = []
    task_identity = read_process_identity(os.getpid()) or {}
    ownership: JsonDict = {"owned": False, "signals_sent": []}
    resident = False
    timed_out = False
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        start_ticks = lease_api.proc_start_ticks(process.pid)
        deadline = time.monotonic() + MODEL_TIMEOUT_S
        while process.poll() is None and time.monotonic() < deadline:
            sample = gpu_inventory()
            sample["monotonic_ns"] = time.monotonic_ns()
            samples.append(sample)
            if ready_path.exists():
                lineage = capture_process_lineage(process.pid, task_identity)
                ownership = {
                    **lineage,
                    "owned": lineage.get("owned") is True,
                    "signals_sent": ownership.get("signals_sent", []),
                }
            owned_uuids = {
                str(row.get("gpu_uuid"))
                for row in sample.get("processes", [])
                if row.get("pid") == process.pid
            }
            if not resident and owned_uuids == {str(row["uuid"]) for row in devices}:
                vram_by_uuid = {
                    str(device["uuid"]): sum(
                        int(row.get("used_memory_mb", 0) or 0)
                        for row in sample.get("processes", [])
                        if row.get("pid") == process.pid
                        and str(row.get("gpu_uuid")) == str(device["uuid"])
                    )
                    for device in devices
                }
                for lease in leases:
                    lease.transition("resident", vram_mb=vram_by_uuid.get(lease.device_uuid, 0))
                    lease.transition("inferencing")
                resident = True
            time.sleep(POLL_INTERVAL_S)
        if process.poll() is None:
            timed_out = True
            if terminate_owned_process(process.pid, os.getpid(), start_ticks):
                ownership["signals_sent"].append("SIGTERM")
        process.wait(timeout=60)

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    layer_row = parse_offloaded_layers(stderr_text)
    worker = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {
            "terminal": True,
            "model_close_called": False,
            "live_duration_s": 0.0,
            "error": "worker_output_missing",
            "unit_rows": [],
            "terminal_choice_rows": [],
            "prefix_fraction_rows": [],
            "choice_token_rows": [],
            "commitment_curve_rows": [],
            "checkpoint_rows": [],
        }
    )
    release = _wait_for_vram_release(baseline_rows, model_id)
    absent = owned_process_absent(process.pid, start_ticks)
    port_released = _port_is_free(port)
    owned_samples = [
        {"monotonic_ns": sample["monotonic_ns"], **row}
        for sample in samples
        for row in sample.get("processes", [])
        if row.get("pid") == process.pid
    ]
    gpu_uuids = sorted({str(row.get("gpu_uuid")) for row in owned_samples})
    live_cuda = bool(layer_row.get("offloaded", 0) > 0 and len(gpu_uuids) == 2)
    complete = bool(
        process.returncode == 0
        and not timed_out
        and worker.get("model_close_called") is True
        and ownership.get("owned") is True
        and absent
        and port_released
        and release.get("passed") is True
        and live_cuda
        and len(worker.get("unit_rows", [])) == EXPECTED_CANDIDATE_COUNT * len(CONDITIONS)
    )
    _release_leases(
        leases,
        lease_rows,
        complete=complete,
        vram_release=release,
        exit_code=int(process.returncode or 0),
    )
    lease_complete = len(lease_rows) == 2 and all(
        row.get("consistent") is True for row in lease_rows
    )
    for row in worker.get("unit_rows", []):
        row["live_cuda"] = live_cuda
    teardown = {
        "model_id": model_id,
        "process_exit_code": process.returncode,
        "owned_process_absent": absent,
        "port": port,
        "port_release_confirmed": port_released,
        "model_close_called": worker.get("model_close_called"),
        "signals_sent": ownership.get("signals_sent", []),
        "passed": bool(complete and lease_complete and ownership.get("signals_sent") == []),
    }
    isolation = {
        "model_id": model_id,
        "process_owned": ownership.get("owned") is True,
        "worker_input_field_names": worker.get("worker_input_field_names", []),
        "denied_fields_visible": worker.get("denied_fields_visible", []),
        "exact_labels_visible": False,
        "sidecar_paths_visible": False,
        "authority_results_visible": False,
        "process_receipt": ownership,
        "passed": bool(
            complete
            and worker.get("worker_input_field_names") == sorted(WORKER_INPUT_FIELDS)
            and worker.get("denied_fields_visible") == []
        ),
    }
    return {
        "model_id": model_id,
        "complete": bool(complete and lease_complete),
        "live_duration_s": float(worker.get("live_duration_s", 0.0) or 0.0),
        "unit_rows": worker.get("unit_rows", []),
        "terminal_choice_rows": worker.get("terminal_choice_rows", []),
        "prefix_fraction_rows": worker.get("prefix_fraction_rows", []),
        "choice_token_rows": worker.get("choice_token_rows", []),
        "commitment_curve_rows": worker.get("commitment_curve_rows", []),
        "checkpoint_rows": worker.get("checkpoint_rows", []),
        "lease_rows": lease_rows,
        "process_isolation": isolation,
        "gpu_runtime": {
            "model_id": model_id,
            "live_cuda": live_cuda,
            "gpu_uuids_used": gpu_uuids,
            "offloaded_layers": layer_row.get("offloaded"),
            "total_layers": layer_row.get("total"),
            "owned_gpu_samples": owned_samples,
            "passed": live_cuda,
        },
        "teardown": teardown,
        "vram_release": {"model_id": model_id, **release},
        "worker_error": worker.get("error"),
        "backend_stdout_hash": sha256_file(stdout_path),
        "backend_stderr_hash": sha256_file(stderr_path),
    }


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover
    """Freeze inputs, run three owned families, join labels, and publish."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    lease_runtime_dir = Path(
        os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases")
    )
    manifest_error: str | None = None
    frozen: JsonDict = {"rows": [], "pair_manifest_hash": None}
    learner: JsonDict = {}
    runtime: JsonDict = {}
    try:
        fixture, learner, runtime = _load_label_free_sources()
        source_groups = {
            str(row["contrast_group_id"]): str(row["source_group_id"])
            for row in fixture["split_rows"]
            if row.get("split") == "held_out"
        }
        frozen = freeze_pair_manifest(fixture["per_candidate_rows"], source_groups=source_groups)
    except Exception as exc:  # noqa: BLE001
        manifest_error = f"{type(exc).__name__}: {exc}"
    preconditions = collect_preconditions(
        frozen=frozen,
        learner=learner,
        runtime=runtime,
        model_specs=specs,
        checkpoint_root=checkpoint_root,
        lease_runtime_dir=lease_runtime_dir,
        manifest_error=manifest_error,
    )
    source_hashes = _source_hashes()
    model_hashes = {
        str(row.get("hf_id")): sha256_file(str(row.get("model_path", "")))
        if Path(str(row.get("model_path", ""))).is_file()
        else None
        for row in specs
    }
    if preconditions.get("all_passed") is not True:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            live_duration_s=0.0,
            preconditions=preconditions,
            model_specs=specs,
            source_artifact_hashes=source_hashes,
            model_file_hashes=model_hashes,
            pair_manifest_rows=frozen.get("rows", []),
            pair_manifest_hash=frozen.get("pair_manifest_hash"),
        )
        write_json_atomic(result_path, artifact)
        return artifact

    provenance, candidate_keys = load_provenance(frozen["rows"])
    condition_manifest = build_condition_manifest(frozen["rows"], provenance)
    manifest_hash = sha256_json(
        {
            "pair_manifest_hash": frozen["pair_manifest_hash"],
            "condition_manifest_hash": condition_manifest["condition_manifest_hash"],
            "load_config": LOAD_CONFIG,
            "model_ids": list(REQUIRED_MODEL_IDS),
        }
    )
    blocks = scoring_blocks(condition_manifest["rows"])
    denial_rows = runtime_label_denial_rows()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    family_results: list[JsonDict] = []
    devices = list(preconditions["gpu_topology"]["devices"])
    for model in specs:
        family = run_family_process(
            model=model,
            blocks=blocks,
            manifest_hash=manifest_hash,
            work_root=checkpoint_root / "workers",
            devices=devices,
            lease_runtime_dir=lease_runtime_dir,
        )
        family_results.append(family)
        if family.get("complete") is not True:
            break

    units = [deepcopy(dict(row)) for family in family_results for row in family["unit_rows"]]
    terminals = [
        deepcopy(dict(row)) for family in family_results for row in family["terminal_choice_rows"]
    ]
    prefixes = [
        deepcopy(dict(row)) for family in family_results for row in family["prefix_fraction_rows"]
    ]
    choice_tokens = [
        deepcopy(dict(row)) for family in family_results for row in family["choice_token_rows"]
    ]
    curves = [
        deepcopy(dict(row)) for family in family_results for row in family["commitment_curve_rows"]
    ]
    teardown_rows = [deepcopy(dict(family["teardown"])) for family in family_results]
    joined: list[JsonDict] = []
    if _teardown_complete(teardown_rows):
        labels = load_late_labels(candidate_keys)
        joined = join_labels(units, labels, teardown_rows)
        preconditions["exact_labels_opened"] = True
        preconditions["label_opened_after_all_model_processes_exit"] = True
    projections = metric_projection_rows(units)
    paired, intervals, strata = aggregate_condition_statistics(units)
    pair_results = per_pair_summaries(units, joined)
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        live_duration_s=sum(float(family["live_duration_s"]) for family in family_results),
        preconditions=preconditions,
        model_specs=specs,
        source_artifact_hashes=source_hashes,
        model_file_hashes=model_hashes,
        pair_manifest_rows=frozen["rows"],
        pair_manifest_hash=frozen["pair_manifest_hash"],
        condition_manifest_rows=condition_manifest["rows"],
        per_pair_results=pair_results,
        unit_rows=units,
        terminal_choice_rows=terminals,
        prefix_fraction_rows=prefixes,
        choice_token_rows=choice_tokens,
        commitment_curve_rows=curves,
        first_commitment_latency_rows=projections["first_commitment_latency_rows"],
        commitment_range_rows=projections["commitment_range_rows"],
        uncommitted_mass_rows=projections["uncommitted_mass_rows"],
        uncertainty_rows=projections["uncertainty_rows"],
        choice_flip_rows=projections["choice_flip_rows"],
        label_denial_rows=denial_rows,
        process_isolation_rows=[family["process_isolation"] for family in family_results],
        late_label_join_rows=joined,
        paired_condition_delta_rows=paired,
        bootstrap_interval_rows=intervals,
        family_stratum_rows=strata,
        gpu_runtime_rows=[family["gpu_runtime"] for family in family_results],
        lease_rows=[row for family in family_results for row in family["lease_rows"]],
        checkpoint_rows=[row for family in family_results for row in family["checkpoint_rows"]],
        teardown_rows=teardown_rows,
        vram_release_rows=[family["vram_release"] for family in family_results],
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the controller, private worker, or cold artifact validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--score-worker-payload", type=Path)
    parser.add_argument("--score-worker-output", type=Path)
    parser.add_argument("--score-worker-ready", type=Path)
    parser.add_argument("--score-worker-port", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.score_worker_payload is not None:
        if (
            args.score_worker_output is None
            or args.score_worker_ready is None
            or args.score_worker_port is None
        ):
            parser.error("worker mode requires output, ready, and port")
        return _score_worker(
            payload_path=args.score_worker_payload,
            output_path=args.score_worker_output,
            ready_path=args.score_worker_ready,
            port=args.score_worker_port,
        )
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        checkpoint_root=args.checkpoint_root,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "expected_unit_count": artifact["expected_unit_count"],
                "observed_unit_count": artifact["observed_unit_count"],
                "commitment_control_complete_score": artifact["commitment_control_complete_score"],
                "shortcut_commitment_detected_score": artifact[
                    "shortcut_commitment_detected_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
