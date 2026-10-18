"""Freeze Exp6867 tokenizer-aware calibration and held splits.

The module loads only each GGUF vocabulary. It records token IDs and tokenizer
identity without evaluating model weights or requesting token likelihoods.

Spec refs: REQ-INFERENCE-6867 and SCENARIO-INFERENCE-6867-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any
import unicodedata

from carnot import experiment_6866_canonical_tokenizer_binding_requalification as binding
from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    flagship_dense,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
RESULT_PATH = Path("results/experiment_6867_tokenizer_aware_semantic_preregistration_v2.json")
SOURCE_PATHS = {
    "exp6862": Path("results/experiment_6862_dual_side_semantic_contrast_bank.json"),
    "exp6866": Path("results/experiment_6866_canonical_tokenizer_binding_requalification.json"),
}

SCHEMA = "carnot.experiment_6867.tokenizer_aware_semantic_preregistration_v2.v1"
INFERENCE_SUBSTRATE = "native_gguf_tokenization_without_inference"
RUN_DATE = "20260902"
RANDOM_SEED = 6867
MINIMUM_GROUP_COUNT = 20
MAX_WHITESPACE_EXTRA = 48
FROZEN_CONTRAST_BANK_HASH = (
    "sha256:dbc2d463153274432dd02bbd188e5e5477a650bcdf4f1fc101f2e6adc5778638"
)
FROZEN_GROUP_ID_HASH = "sha256:f43316b8996944b3ce60061713837eabd44c042eefff8aab5db1fbaf81b19b51"
BLOCKED_VERDICT = "complete_blocked_tokenizer_aware_semantic_preregistration_v2"
READY_VERDICT = "complete_positive_tokenizer_aware_semantic_preregistration_v2_ready_no_scores"
LOW_SAMPLE_VERDICT = (
    "complete_null_tokenizer_aware_semantic_preregistration_v2_sample_floor_not_met"
)

MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    MODEL_SPECS[0]: "qwen_moe",
    MODEL_SPECS[1]: "gemma_dense",
    MODEL_SPECS[2]: "gemma_moe",
}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
RAW_PROMPT_TEMPLATE = (
    "Exact semantic obligation program.\n"
    "PROGRAM_JSON_BEGIN\n{program_json}\nPROGRAM_JSON_END\n"
    "Return only the candidate sequence."
)
CANDIDATE_SEQUENCE_TEMPLATE = "CANDIDATE_JSON_BEGIN\n{candidate_json}\nCANDIDATE_JSON_END"
NUISANCE_CONTROLS = (
    "identifier",
    "order",
    "normalization",
    "label_position",
    "token_count",
    "character_length",
    "surface_form",
)
SPEC_REFS = (
    "REQ-INFERENCE-6867",
    "SCENARIO-INFERENCE-6867-BANK-IDENTITY",
    "SCENARIO-INFERENCE-6867-TOKENIZER-IDENTITY",
    "SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS",
    "SCENARIO-INFERENCE-6867-GROUP-SPLIT",
    "SCENARIO-INFERENCE-6867-HELD-SEAL",
    "SCENARIO-INFERENCE-6867-SAMPLE-FLOOR",
    "SCENARIO-INFERENCE-6867-SCORE-FREEZE",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "frozen_contrast_bank_hash",
    "rows",
    "accepted_cell_manifest",
    "rejected_cell_manifest",
    "nuisance_match_manifest",
    "calibration_group_manifest",
    "sealed_held_group_manifest",
    "split_overlap_count",
    "held_label_access_count",
    "token_likelihood_call_count",
    "generated_answer_count",
    "preregistered_statistic_manifest",
    "sample_size_power_rows",
    "random_seed",
    "reproducibility_checksum",
    "semantic_contrast_preregistration_v2_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
SCORE_FIELDS = {
    "candidate_log_likelihoods",
    "candidate_log_likelihood",
    "token_log_likelihoods",
    "token_log_likelihood",
    "token_logprobs",
    "token_logprob",
}

FIELD_PRINCIPLES = {
    "schema": "The version prevents silent consumer drift.",
    "experiment_id": "The fixed identifier binds this evidence to Exp6867.",
    "run_date": "The supplied date identifies this execution.",
    "status": "The status records one terminal artifact.",
    "result_path": "The stable path gives Exp6868 one exact input.",
    "spec_refs": "The references connect tests and evidence to OpenSpec.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "The checks stop mutated sources and substituted tokenizers.",
    "inference_substrate": "The value separates tokenization from model inference.",
    "duration_s": "Measured wall time records the real tokenizer work.",
    "model_specs": "The list fixes all three required GGUF repositories.",
    "models_used": "The list records every native tokenizer that ran.",
    "model_artifact_hashes": "File identities prevent model substitution.",
    "tokenizer_receipts": "Canonical payload hashes freeze tokenizer semantics.",
    "frozen_contrast_bank_hash": "The exact hash prevents contrast-bank regeneration.",
    "rows": "Each row binds one model and one semantic contrast cell.",
    "accepted_cell_manifest": "The manifest fixes future nuisance-eligible score cells.",
    "rejected_cell_manifest": "The manifest preserves every exclusion and reason.",
    "nuisance_match_manifest": "The manifest freezes all surface controls.",
    "calibration_group_manifest": "The manifest exposes labels only for calibration.",
    "sealed_held_group_manifest": "Commitments freeze held labels without plaintext.",
    "split_overlap_count": "Zero proves calibration and held groups are disjoint.",
    "held_label_access_count": "Zero proves calibration did not load held labels.",
    "token_likelihood_call_count": "Zero proves that no score informed the design.",
    "generated_answer_count": "Zero proves that no answer generation occurred.",
    "preregistered_statistic_manifest": "The analysis rules predate all scores.",
    "sample_size_power_rows": "Rows enforce 20 groups per model and split.",
    "random_seed": "One fixed seed freezes group assignment.",
    "reproducibility_checksum": "The checksum seals all stable artifact content.",
    "semantic_contrast_preregistration_v2_ready_score": "Readiness measures design only.",
    "gate_check_summary": "The summary names exact expected and observed failures.",
    "verifier_is_oracle": "False states that tokenization is not a truth oracle.",
    "verdict_class": "The closed class states the evidence boundary.",
    "honest_verdict": "The complete prefix gives one terminal disposition.",
    "source_artifact_hashes": "Source hashes bind both prerequisite artifacts.",
    "split_manifest": "The manifest records one group-level split checksum.",
    "scientific_effect_claimed": "False prevents design readiness from becoming a result.",
}


class HeldLabelAccessError(RuntimeError):
    """Raised before calibration can read one held-group label."""


class CalibrationLabelLoader:
    """Expose calibration labels and deny every sealed held identity."""

    def __init__(
        self,
        calibration_labels: Mapping[str, tuple[bool, bool]],
        held_group_ids: Iterable[str],
    ) -> None:
        self._calibration_labels = dict(calibration_labels)
        self._held_group_ids = set(held_group_ids)
        self.held_label_access_count = 0
        self.held_label_access_attempt_count = 0

    def load(self, semantic_group_identity: str) -> tuple[bool, bool]:
        """Return calibration labels or reject a held identity before access."""

        if semantic_group_identity in self._held_group_ids:
            self.held_label_access_attempt_count += 1
            raise HeldLabelAccessError(f"held label access denied: {semantic_group_identity}")
        return self._calibration_labels[semantic_group_identity]


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for sequences and content identities."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return one prefixed SHA-256 digest for UTF-8 text."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_text(canonical_json(value))


def sequence_identity(
    model_hf_id: str,
    semantic_group_identity: str,
    prompt_sequence_sha256: str,
    candidate_sequence_sha256: Mapping[str, Any],
) -> str:
    """Bind one tokenizable sequence set to its model and semantic group."""

    return sha256_json(
        {
            "model_hf_id": model_hf_id,
            "semantic_group_identity": semantic_group_identity,
            "prompt_sequence_sha256": prompt_sequence_sha256,
            "candidate_sequence_sha256": dict(candidate_sequence_sha256),
        }
    )


def score_identity(model_hf_id: str, semantic_group_identity: str, split_checksum: str) -> str:
    """Name the future score cell before any score exists."""

    return sha256_json(
        {
            "schema": "carnot.exp6867.score_identity.v1",
            "model_hf_id": model_hf_id,
            "semantic_group_identity": semantic_group_identity,
            "split_checksum": split_checksum,
        }
    )


def gate_check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return one exact gate record."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure while preserving every check."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_check": failed[0].get("check") if failed else None,
        "expected": failed[0].get("expected") if failed else "all checks pass",
        "observed": failed[0].get("observed") if failed else "all checks pass",
        "failed_checks": failed,
        "checks": [dict(row) for row in checks],
    }


def bank_identity_errors(
    raw_bytes: bytes,
    groups: Iterable[Mapping[str, Any]],
    *,
    frozen_hash: str = FROZEN_CONTRAST_BANK_HASH,
    frozen_group_ids: Iterable[str] | None = None,
) -> list[str]:
    """Detect any changed bank byte, count, collision, or group identity."""

    rows = list(groups)
    group_ids = [str(row.get("semantic_identity")) for row in rows]
    errors: list[str] = []
    if sha256_bytes(raw_bytes) != frozen_hash:
        errors.append("frozen_contrast_bank_hash_mismatch")
    if len(group_ids) != 100:
        errors.append("semantic_group_count_mismatch")
    if len(set(group_ids)) != len(group_ids):
        errors.append("semantic_group_identity_collision")
    expected_hash = (
        sha256_json(sorted(str(value) for value in frozen_group_ids))
        if frozen_group_ids is not None
        else FROZEN_GROUP_ID_HASH
    )
    if sha256_json(sorted(group_ids)) != expected_hash:
        errors.append("semantic_group_identity_mismatch")
    return errors


def tokenizer_binding_errors(current: Mapping[str, Any], frozen: Mapping[str, Any]) -> list[str]:
    """Name every model-file or canonical-tokenizer substitution."""

    field_reasons = (
        ("path", "model_path_substitution"),
        ("sha256", "model_hash_drift"),
        ("size_bytes", "model_size_drift"),
        ("quantization", "quantization_drift"),
        ("snapshot_identity", "snapshot_identity_drift"),
        (
            "canonical_tokenizer_payload_sha256",
            "canonical_tokenizer_payload_drift",
        ),
    )
    return [reason for field, reason in field_reasons if current.get(field) != frozen.get(field)]


def find_scored_split_identities(
    documents: Iterable[Any], target_identities: Iterable[str]
) -> list[str]:
    """Find target score identities that already carry token likelihoods."""

    targets = set(target_identities)
    found: set[str] = set()

    def visit(value: Any, inherited_identity: str | None = None) -> None:
        if isinstance(value, Mapping):
            identity = str(value.get("score_identity") or inherited_identity or "")
            if identity in targets and any(
                field in value and value.get(field) not in (None, [], {}) for field in SCORE_FIELDS
            ):
                found.add(identity)
            for item in value.values():
                visit(item, identity or inherited_identity)
        elif isinstance(value, list):
            for item in value:
                visit(item, inherited_identity)

    for document in documents:
        visit(document)
    return sorted(found)


def cell_nuisance_errors(cell: Mapping[str, Any]) -> list[str]:
    """Apply every frozen identifier, order, length, and surface control."""

    errors: list[str] = []
    candidate_ids = cell.get("candidate_ids")
    candidate_ids = candidate_ids if isinstance(candidate_ids, list) else []
    labels = cell.get("candidate_labels_by_slot")
    labels = labels if isinstance(labels, list) else []
    token_ids = cell.get("candidate_token_ids")
    token_ids = token_ids if isinstance(token_ids, Mapping) else {}
    if len(candidate_ids) != 2 or len(labels) != 2 or set(token_ids) != {"slot_0", "slot_1"}:
        errors.append("candidate_count_mismatch")
    if len(candidate_ids) == 2 and len(set(candidate_ids)) != 2:
        errors.append("candidate_identifier_collision")
    if labels != [True, False]:
        errors.append("candidate_label_order_mismatch")
    token_counts = cell.get("candidate_token_counts")
    token_counts = token_counts if isinstance(token_counts, Mapping) else {}
    if token_counts.get("slot_0") != token_counts.get("slot_1"):
        errors.append("unequal_candidate_token_counts")
    character_counts = cell.get("candidate_character_counts")
    character_counts = character_counts if isinstance(character_counts, Mapping) else {}
    if character_counts.get("slot_0") != character_counts.get("slot_1"):
        errors.append("unequal_candidate_character_counts")
    prompt_counts = cell.get("paired_prompt_token_counts")
    prompt_counts = prompt_counts if isinstance(prompt_counts, Mapping) else {}
    if prompt_counts.get("base") != prompt_counts.get("label_swap"):
        errors.append("paired_prompt_count_mismatch")
    prompt_ids = cell.get("paired_prompt_token_ids")
    prompt_ids = prompt_ids if isinstance(prompt_ids, Mapping) else {}
    if prompt_ids.get("base") != prompt_ids.get("label_swap"):
        errors.append("paired_prompt_token_ids_mismatch")
    if cell.get("label_position_contract") != {"base": [0, 1], "swapped": [1, 0]}:
        errors.append("label_position_contract_failed")
    if cell.get("presentation_order") != {"base": [0, 1], "label_swap": [1, 0]}:
        errors.append("presentation_order_failed")
    if cell.get("normalization") != "NFC":
        errors.append("normalization_control_failed")
    surface = cell.get("surface_control")
    expected_surface = {
        "raw_prompt_template_sha256": sha256_text(RAW_PROMPT_TEMPLATE),
        "candidate_sequence_template_sha256": sha256_text(CANDIDATE_SEQUENCE_TEMPLATE),
    }
    if surface != expected_surface:
        errors.append("surface_template_control_failed")
    candidate_hashes = cell.get("candidate_sequence_sha256")
    candidate_hashes = candidate_hashes if isinstance(candidate_hashes, Mapping) else {}
    expected_identity = sequence_identity(
        str(cell.get("model_hf_id")),
        str(cell.get("semantic_group_identity")),
        str(cell.get("prompt_sequence_sha256")),
        candidate_hashes,
    )
    if cell.get("sequence_identity") != expected_identity:
        errors.append("sequence_identity_mismatch")
    return errors


def semantic_family_errors(groups: Iterable[Mapping[str, Any]]) -> list[str]:
    """Reject one semantic identity assigned to more than one family."""

    families: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        families[str(group.get("semantic_identity"))].add(str(group.get("family")))
    return [
        f"semantic_family_overlap:{group_id}"
        for group_id, values in sorted(families.items())
        if len(values) != 1
    ]


def freeze_group_splits(groups: Iterable[Mapping[str, Any]], *, random_seed: int) -> JsonDict:
    """Split whole semantic groups within each family using one fixed seed."""

    by_family: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        by_family[str(group.get("family"))].add(str(group.get("semantic_identity")))
    calibration: list[str] = []
    held: list[str] = []
    family_counts: JsonDict = {}
    for family, group_ids in sorted(by_family.items()):
        ordered = sorted(
            group_ids,
            key=lambda group_id: sha256_json([random_seed, family, group_id]),
        )
        midpoint = len(ordered) // 2
        calibration.extend(ordered[:midpoint])
        held.extend(ordered[midpoint:])
        family_counts[family] = {
            "calibration": midpoint,
            "held": len(ordered) - midpoint,
        }
    calibration = sorted(calibration)
    held = sorted(held)
    checksum = sha256_json(
        {
            "random_seed": random_seed,
            "assignment_unit": "semantic_group_identity",
            "calibration_group_ids": calibration,
            "held_group_ids": held,
        }
    )
    return {
        "random_seed": int(random_seed),
        "assignment_unit": "semantic_group_identity",
        "calibration_group_ids": calibration,
        "held_group_ids": held,
        "family_counts": family_counts,
        "split_checksum": checksum,
    }


def split_errors(
    calibration_group_ids: Iterable[str],
    held_group_ids: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Detect direct overlap and row-level group leakage."""

    calibration = set(calibration_group_ids)
    held = set(held_group_ids)
    errors = [f"split_overlap:{group_id}" for group_id in sorted(calibration & held)]
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        observed[str(row.get("semantic_group_identity"))].add(str(row.get("split")))
    errors.extend(
        f"group_split_leakage:{group_id}"
        for group_id, values in sorted(observed.items())
        if len(values) != 1
    )
    return errors


def seal_held_groups(groups: Iterable[Mapping[str, Any]], *, secret_salt: bytes) -> list[JsonDict]:
    """Commit held labels with a private salt and omit all label plaintext."""

    sealed: list[JsonDict] = []
    for group in groups:
        candidates = group.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        label_payload = [
            [str(row.get("candidate_id")), bool(row.get("expected_label"))]
            for row in candidates
            if isinstance(row, Mapping)
        ]
        commitment = sha256_bytes(secret_salt + b"\0" + canonical_json(label_payload).encode())
        sealed.append(
            {
                "semantic_group_identity": str(group.get("semantic_identity")),
                "semantic_family": str(group.get("family")),
                "candidate_ids": [row[0] for row in label_payload],
                "label_commitment": commitment,
                "label_schema": "salted_two_boolean_labels_no_plaintext_v2",
            }
        )
    return sealed


def preregistered_statistics() -> JsonDict:
    """Return every score-free statistic, interval, and retirement rule."""

    return {
        "frozen_before_token_likelihood": True,
        "within_group_paired_contrast": {
            "estimand": (
                "mean_log_likelihood_per_token(valid) minus mean_log_likelihood_per_token(invalid)"
            ),
            "pairing_unit": "model_hf_id_by_semantic_group_identity",
        },
        "nuisance_difference_in_differences": {
            control: f"paired_contrast minus matched_{control}_contrast"
            for control in NUISANCE_CONTROLS
        },
        "bootstrap_interval": {
            "method": "BCa_cluster_bootstrap",
            "confidence_level": 0.95,
            "resamples": 10000,
            "cluster_unit": "semantic_group_identity",
            "random_seed": RANDOM_SEED,
        },
        "missing_cell_rule": {
            "imputation": "none",
            "analysis_set": "complete nuisance-eligible cells only",
            "floor_action": "retire claim when any required model has fewer than 20 held groups",
        },
        "per_family_effect": {
            "unit": "Exp6862 semantic family",
            "aggregation": "equal-weight mean of within-group paired contrasts",
        },
        "family_replication_rule": (
            "All five semantic families must have the same effect sign in each model. "
            "Qwen MoE, Gemma dense, and Gemma MoE must also share that sign."
        ),
        "pooled_aggregation": {
            "model_weighting": "equal model weight",
            "group_weighting": "equal semantic-group weight within model",
            "family_weighting": "equal semantic-family weight",
        },
        "multiple_comparison_rule": "Holm correction across seven nuisance contrasts per model",
        "retirement_thresholds": [
            "Retire when both Qwen and pooled Gemma upper 95% primary bounds are at or below zero.",
            "Retire after two preregistered runs fail the semantic-family replication rule.",
            "Disqualify a run when a frozen bank, split, model, tokenizer, or held seal drifts.",
        ],
    }


def readiness_score(
    sample_size_power_rows: Iterable[Mapping[str, Any]], *, all_other_checks_pass: bool
) -> int:
    """Return design readiness only when every model and split meets its floor."""

    rows = list(sample_size_power_rows)
    floors_pass = bool(rows) and all(
        int(row.get("accepted_group_count", -1)) >= int(row.get("minimum_group_count", 0))
        and row.get("floor_passed") is True
        for row in rows
    )
    return int(all_other_checks_pass and floors_pass)


def _base_artifact(run_date: str, duration_s: float) -> JsonDict:
    """Create the full blocked-safe schema before native tokenization."""

    return {
        "schema": SCHEMA,
        "experiment_id": 6867,
        "run_date": run_date,
        "status": "complete",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": {},
        "preconditions_checked": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "model_specs": list(MODEL_SPECS),
        "models_used": [],
        "model_artifact_hashes": {},
        "tokenizer_receipts": [],
        "frozen_contrast_bank_hash": FROZEN_CONTRAST_BANK_HASH,
        "rows": [],
        "accepted_cell_manifest": [],
        "rejected_cell_manifest": [],
        "nuisance_match_manifest": {},
        "calibration_group_manifest": [],
        "sealed_held_group_manifest": [],
        "split_overlap_count": 0,
        "held_label_access_count": 0,
        "token_likelihood_call_count": 0,
        "generated_answer_count": 0,
        "preregistered_statistic_manifest": preregistered_statistics(),
        "sample_size_power_rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "semantic_contrast_preregistration_v2_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
        "source_artifact_hashes": {},
        "split_manifest": {},
        "scientific_effect_claimed": False,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable content while excluding wall time and the hash itself."""

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
    }
    return sha256_json(unsigned)


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    """Seal stable content and attach one principle for every top-level field."""

    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves preregistration evidence.")
        for key in artifact
    }
    return artifact


def blocked_artifact(
    *, run_date: str, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Return a terminal blocked artifact with one exact failed gate."""

    artifact = _base_artifact(run_date, duration_s)
    summary = gate_summary(checks)
    artifact["preconditions_checked"] = summary
    artifact["gate_check_summary"] = summary
    return _finish_artifact(artifact)


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    """Return true when nested evidence contains a forbidden label key."""

    if isinstance(value, Mapping):
        return bool(set(value) & forbidden) or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the exact score-free, sealed, consumer-facing contract."""

    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    principles = artifact.get("field_principles")
    principle_keys = set(principles) if isinstance(principles, Mapping) else set()
    if principle_keys != set(artifact):
        errors.append("field_principle_keys")
    scalar_contract = (
        ("inference_substrate", INFERENCE_SUBSTRATE),
        ("token_likelihood_call_count", 0),
        ("generated_answer_count", 0),
        ("split_overlap_count", 0),
        ("held_label_access_count", 0),
        ("verifier_is_oracle", False),
    )
    for field, expected in scalar_contract:
        if artifact.get(field) != expected:
            errors.append(field)
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("frozen_contrast_bank_hash") != FROZEN_CONTRAST_BANK_HASH:
        errors.append("frozen_contrast_bank_hash")
    if artifact.get("semantic_contrast_preregistration_v2_ready_score") not in {0, 1}:
        errors.append("semantic_contrast_preregistration_v2_ready_score")
    if _contains_key(
        artifact.get("sealed_held_group_manifest", []),
        {"expected_label", "candidate_labels_by_slot"},
    ):
        errors.append("sealed_held_group_manifest")
    held_rows = [
        row
        for row in artifact.get("rows", [])
        if isinstance(row, Mapping) and row.get("split") == "held"
    ]
    if _contains_key(held_rows, {"expected_label", "candidate_labels_by_slot"}):
        errors.append("held_rows_label_seal")
    if artifact.get("semantic_contrast_preregistration_v2_ready_score") == 1:
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or summary.get("passed") is not True:
            errors.append("ready_gate_summary")
        if artifact.get("models_used") != list(MODEL_SPECS):
            errors.append("models_used")
    return errors


def _whitespace_variants(length: int) -> list[str]:
    """Return bounded semantic-free suffixes with one exact character length."""

    if length == 0:
        return [""]
    patterns = (" ", "\n", "\t", " \n", "\n ", " \t", "\t ", "\n\t")
    variants: list[str] = []
    for pattern in patterns:
        value = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
        if value not in variants:
            variants.append(value)
    return variants


def _native_tokens(tokenizer: Any, text: str, *, add_bos: bool) -> list[int]:
    """Call only native GGUF tokenization with the frozen special setting."""

    return [
        int(token)
        for token in tokenizer.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False)
    ]


def _equalized_sequences(
    tokenizer: Any, first: str, second: str
) -> tuple[list[str], list[list[int]], JsonDict] | None:
    """Match token and character counts with bounded trailing whitespace."""

    base_length = max(len(first), len(second))
    for extra in range(MAX_WHITESPACE_EXTRA + 1):
        target = base_length + extra
        options: list[dict[int, tuple[str, list[int]]]] = []
        for source in (first, second):
            by_count: dict[int, tuple[str, list[int]]] = {}
            for suffix in _whitespace_variants(target - len(source)):
                sequence = source + suffix
                tokens = _native_tokens(tokenizer, sequence, add_bos=False)
                by_count.setdefault(len(tokens), (sequence, tokens))
            options.append(by_count)
        common = sorted(set(options[0]) & set(options[1]))
        if common:
            count = common[0]
            first_sequence, first_tokens = options[0][count]
            second_sequence, second_tokens = options[1][count]
            return (
                [first_sequence, second_sequence],
                [first_tokens, second_tokens],
                {
                    "method": "bounded_trailing_whitespace_exact_match",
                    "matched": True,
                    "target_character_count": target,
                    "candidate_token_count": count,
                    "extra_character_budget": extra,
                },
            )
    return None


def render_group_cell(
    *,
    tokenizer: Any,
    tokenizer_receipt: Mapping[str, Any],
    model_hf_id: str,
    group: Mapping[str, Any],
) -> JsonDict:
    """Tokenize one contrast and retain every nuisance-control receipt."""

    semantic_group_identity = str(group.get("semantic_identity"))
    prompt = unicodedata.normalize(
        "NFC",
        RAW_PROMPT_TEMPLATE.format(program_json=canonical_json(group.get("program"))),
    )
    candidates_value = group.get("candidates")
    candidates = (
        [dict(row) for row in candidates_value if isinstance(row, Mapping)]
        if isinstance(candidates_value, list)
        else []
    )
    originals = [
        unicodedata.normalize(
            "NFC",
            CANDIDATE_SEQUENCE_TEMPLATE.format(
                candidate_json=canonical_json(candidate.get("content"))
            ),
        )
        for candidate in candidates[:2]
    ]
    prompt_tokens = _native_tokens(tokenizer, prompt, add_bos=True)
    matched = (
        _equalized_sequences(tokenizer, originals[0], originals[1]) if len(originals) == 2 else None
    )
    if matched is None:
        sequences = originals
        candidate_tokens = [
            _native_tokens(tokenizer, sequence, add_bos=False) for sequence in sequences
        ]
        padding_receipt: JsonDict = {
            "method": "bounded_trailing_whitespace_exact_match",
            "matched": False,
            "maximum_extra_character_budget": MAX_WHITESPACE_EXTRA,
        }
    else:
        sequences, candidate_tokens, padding_receipt = matched
    candidate_hashes = {
        f"slot_{index}": sha256_text(sequence) for index, sequence in enumerate(sequences)
    }
    prompt_hash = sha256_text(prompt)
    row: JsonDict = {
        "model_hf_id": model_hf_id,
        "model_family": MODEL_FAMILIES[model_hf_id],
        "semantic_group_identity": semantic_group_identity,
        "semantic_family": str(group.get("family")),
        "source_group_id": group.get("group_id"),
        "candidate_ids": [str(row.get("candidate_id")) for row in candidates[:2]],
        "candidate_semantic_identities": [
            str(row.get("semantic_identity")) for row in candidates[:2]
        ],
        "candidate_labels_by_slot": [bool(row.get("expected_label")) for row in candidates[:2]],
        "paired_prompt_token_ids": {
            "base": prompt_tokens,
            "label_swap": list(prompt_tokens),
        },
        "paired_prompt_token_counts": {
            "base": len(prompt_tokens),
            "label_swap": len(prompt_tokens),
        },
        "candidate_token_ids": {
            f"slot_{index}": tokens for index, tokens in enumerate(candidate_tokens)
        },
        "candidate_token_counts": {
            f"slot_{index}": len(tokens) for index, tokens in enumerate(candidate_tokens)
        },
        "candidate_character_counts": {
            f"slot_{index}": len(sequence) for index, sequence in enumerate(sequences)
        },
        "original_candidate_character_counts": {
            f"slot_{index}": len(sequence) for index, sequence in enumerate(originals)
        },
        "prompt_sequence_sha256": prompt_hash,
        "candidate_sequence_sha256": candidate_hashes,
        "original_candidate_sequence_sha256": {
            f"slot_{index}": sha256_text(sequence) for index, sequence in enumerate(originals)
        },
        "sequence_identity": sequence_identity(
            model_hf_id,
            semantic_group_identity,
            prompt_hash,
            candidate_hashes,
        ),
        "canonical_tokenizer_payload_hash": tokenizer_receipt.get(
            "canonical_tokenizer_payload_sha256"
        ),
        "special_token_settings": deepcopy(tokenizer_receipt.get("special_token_settings", {})),
        "tokenize_settings": {
            "prompt": {"add_bos": True, "special": False},
            "candidate": {"add_bos": False, "special": False},
        },
        "label_position_contract": {"base": [0, 1], "swapped": [1, 0]},
        "presentation_order": {"base": [0, 1], "label_swap": [1, 0]},
        "normalization": "NFC",
        "surface_control": {
            "raw_prompt_template_sha256": sha256_text(RAW_PROMPT_TEMPLATE),
            "candidate_sequence_template_sha256": sha256_text(CANDIDATE_SEQUENCE_TEMPLATE),
        },
        "padding_receipt": padding_receipt,
        "token_likelihood_call_count": 0,
        "generated_answer_count": 0,
    }
    reasons = cell_nuisance_errors(row)
    row["accepted"] = not reasons
    row["rejection_reasons"] = reasons
    return row


def _sample_size_rows(
    accepted_rows: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any]
) -> list[JsonDict]:
    """Count accepted semantic groups for every required model and split."""

    split_ids = {
        "calibration": set(split_manifest.get("calibration_group_ids", [])),
        "held": set(split_manifest.get("held_group_ids", [])),
    }
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in accepted_rows:
        by_model[str(row.get("model_hf_id"))].add(str(row.get("semantic_group_identity")))
    rows: list[JsonDict] = []
    for model_hf_id in MODEL_SPECS:
        for split, group_ids in split_ids.items():
            count = len(by_model[model_hf_id] & group_ids)
            rows.append(
                {
                    "model_hf_id": model_hf_id,
                    "model_family": MODEL_FAMILIES[model_hf_id],
                    "split": split,
                    "accepted_group_count": count,
                    "minimum_group_count": MINIMUM_GROUP_COUNT,
                    "floor_passed": count >= MINIMUM_GROUP_COUNT,
                    "planned_standardized_paired_effect": 0.65,
                    "two_sided_alpha": 0.05,
                    "planned_power": 0.80,
                    "power_status": "preregistered_design_target_not_observed_result",
                }
            )
    return rows


def _redact_held_rows(
    rows: Sequence[Mapping[str, Any]], commitments: Mapping[str, str]
) -> list[JsonDict]:
    """Remove plaintext held labels while retaining their frozen commitment."""

    public_rows: list[JsonDict] = []
    for source in rows:
        row = deepcopy(dict(source))
        row["split_cell_identity"] = row.pop("score_identity")
        if row.get("split") == "held":
            row.pop("candidate_labels_by_slot", None)
            row["candidate_label_commitment"] = commitments[str(row["semantic_group_identity"])]
        public_rows.append(row)
    return public_rows


def _read_sources(root: Path) -> tuple[dict[str, JsonDict], dict[str, bytes], list[JsonDict]]:
    """Read both immutable source artifacts and preserve typed failures."""

    documents: dict[str, JsonDict] = {}
    raw_values: dict[str, bytes] = {}
    failures: list[JsonDict] = []
    for source_id, relative_path in SOURCE_PATHS.items():
        path = root / relative_path
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("top-level JSON is not an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(
                gate_check(
                    f"source.{source_id}.readable",
                    "readable JSON object",
                    f"{type(exc).__name__}:{exc}",
                    False,
                )
            )
            continue
        documents[source_id] = value
        raw_values[source_id] = raw
    return documents, raw_values, failures


def _frozen_bindings(exp6866: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Combine Exp6866 file identities with canonical tokenizer hashes."""

    hashes_value = exp6866.get("model_artifact_hashes")
    hashes = hashes_value if isinstance(hashes_value, Mapping) else {}
    payload_rows = exp6866.get("canonical_payload_hash_rows")
    payload_rows = payload_rows if isinstance(payload_rows, list) else []
    payload_by_id = {str(row.get("hf_id")): row for row in payload_rows if isinstance(row, Mapping)}
    frozen: dict[str, JsonDict] = {}
    for model_hf_id in MODEL_SPECS:
        source = hashes.get(model_hf_id)
        source = source if isinstance(source, Mapping) else {}
        payload = payload_by_id.get(model_hf_id, {})
        frozen[model_hf_id] = {
            "path": source.get("path"),
            "sha256": source.get("sha256"),
            "size_bytes": source.get("size_bytes"),
            "quantization": source.get("quantization"),
            "snapshot_identity": source.get("snapshot_identity"),
            "canonical_tokenizer_payload_sha256": payload.get("live_canonical_payload_sha256"),
        }
    return frozen


def resolve_model_files(exp6866: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    """Call the canonical pair helper, extend dense, and hash exact files."""

    pair = cached_sota_pair() or []
    resolved = {
        str(row.get("hf_id")): str(row.get("model_path") or "")
        for row in pair
        if isinstance(row, Mapping)
    }
    dense = flagship_dense()
    dense_path = resolve_cached_gguf(dense["hf_id"], dense["quantization"])
    if dense_path:
        resolved[dense["hf_id"]] = dense_path
    registry = {str(row["hf_id"]): row for row in SOTA_GGUF_MODELS}
    frozen = _frozen_bindings(exp6866)
    rows: list[JsonDict] = []
    for model_hf_id in MODEL_SPECS:
        path_text = resolved.get(model_hf_id) or str(
            resolve_cached_gguf(model_hf_id, registry[model_hf_id]["quantization"]) or ""
        )
        path = Path(path_text) if path_text else Path()
        present = bool(path_text and path.is_file())
        rows.append(
            {
                "hf_id": model_hf_id,
                "path": path_text,
                "sha256": binding.sha256_path(path) if present else "",
                "size_bytes": path.stat().st_size if present else None,
                "quantization": registry[model_hf_id]["quantization"],
                "snapshot_identity": binding.snapshot_identity(path_text),
                "canonical_tokenizer_payload_sha256": frozen[model_hf_id].get(
                    "canonical_tokenizer_payload_sha256"
                ),
                "cached_sota_pair_called": True,
                "local_model_present": present,
            }
        )
    return rows


def _scan_prior_score_documents(
    root: Path, target_identities: Iterable[str]
) -> list[str]:  # pragma: no cover - repository corpus boundary.
    """Read only artifacts that declare future score identities and score fields."""

    targets = set(target_identities)
    documents: list[Any] = []
    score_markers = tuple(field.encode("utf-8") for field in SCORE_FIELDS)
    for path in sorted((root / "results").glob("experiment_*.json")):
        if path == root / RESULT_PATH:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b'"score_identity"' not in raw or not any(marker in raw for marker in score_markers):
            continue
        try:
            documents.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return find_scored_split_identities(documents, targets)


def _tokenizer_receipt(
    tokenizer: Any,
    *,
    model_hf_id: str,
    exp6866: Mapping[str, Any],
) -> JsonDict:
    """Recompute the Exp6866 canonical payload from the native tokenizer."""

    manifest = exp6866.get("semantic_probe_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Exp6866 semantic probe manifest missing")
    live_rows = exp6866.get("live_receipt_rows")
    live_rows = live_rows if isinstance(live_rows, list) else []
    source_receipts = {str(row.get("hf_id")): row for row in live_rows if isinstance(row, Mapping)}
    probe_outputs = binding.run_probe_matrix(tokenizer, manifest)
    reduced = binding.reduce_tokenizer_receipt(
        tokenizer.evidence,
        probe_outputs,
        source_receipt=source_receipts.get(model_hf_id, {}),
        source_kind="exp6867_live_native_gguf",
    )
    evidence = tokenizer.evidence
    return {
        "hf_id": model_hf_id,
        "source": "native_embedded_gguf_llama_cpp_vocab_only",
        "canonical_payload_schema_version": reduced["canonical_payload_schema_version"],
        "canonical_tokenizer_payload_sha256": reduced["canonical_payload_sha256"],
        "special_token_settings": {
            "special_token_ids": deepcopy(evidence.get("special_token_ids", {})),
            "add_bos_metadata_default": evidence.get("add_bos_metadata_default"),
            "prompt": {"add_bos": True, "special": False},
            "candidate": {"add_bos": False, "special": False},
        },
        "chat_template_identity": sha256_text(str(evidence.get("chat_template") or "")),
        "vocabulary_size": evidence.get("vocabulary_size"),
        "token_pieces_sha256": evidence.get("token_pieces_sha256"),
        "probe_output_count": len(probe_outputs),
        "token_likelihood_call_count": 0,
        "generated_answer_count": 0,
    }


def build_artifact(
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    *,
    model_resolver: Callable[[Mapping[str, Any]], list[JsonDict]] = resolve_model_files,
    tokenizer_factory: Callable[[str], Any] = binding.NativeGgufTokenizer,
    score_scanner: Callable[[Path, Iterable[str]], list[str]] = _scan_prior_score_documents,
) -> JsonDict:  # pragma: no cover - requested native GGUF end-to-end path.
    """Run preconditions, native tokenization, group splitting, and sealing."""

    started = time.monotonic()
    artifact = _base_artifact(run_date, 0.0)
    documents, raw_values, source_failures = _read_sources(root)
    artifact["source_artifact_hashes"] = {
        source_id: {
            "path": SOURCE_PATHS[source_id].as_posix(),
            "sha256": sha256_bytes(raw),
        }
        for source_id, raw in raw_values.items()
    }
    checks: list[JsonDict] = [
        gate_check(
            "required_source_readability",
            "all required artifacts readable",
            source_failures,
            not source_failures,
        )
    ]
    if source_failures:
        artifact["preconditions_checked"] = gate_summary(checks)
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        return _finish_artifact(artifact)

    exp6862 = documents["exp6862"]
    exp6866 = documents["exp6866"]
    groups_value = exp6862.get("semantic_contrast_group_manifest")
    groups = (
        [dict(row) for row in groups_value if isinstance(row, Mapping)]
        if isinstance(groups_value, list)
        else []
    )
    bank_errors = bank_identity_errors(raw_values["exp6862"], groups)
    family_errors = semantic_family_errors(groups)
    split_manifest = freeze_group_splits(groups, random_seed=RANDOM_SEED)
    artifact["split_manifest"] = split_manifest
    future_score_ids = [
        score_identity(
            model_hf_id, str(group.get("semantic_identity")), split_manifest["split_checksum"]
        )
        for model_hf_id in MODEL_SPECS
        for group in groups
    ]
    scored_identities = score_scanner(root, future_score_ids)
    checks.extend(
        [
            gate_check(
                "canonical_tokenizer_binding_ready_score",
                1,
                exp6866.get("canonical_tokenizer_binding_ready_score"),
                exp6866.get("canonical_tokenizer_binding_ready_score") == 1,
            ),
            gate_check(
                "frozen_exp6862_bank_identity",
                [],
                bank_errors,
                not bank_errors,
            ),
            gate_check(
                "semantic_family_disjointness",
                [],
                family_errors,
                not family_errors,
            ),
            gate_check(
                "prior_token_scores_for_new_split_identities",
                [],
                scored_identities,
                not scored_identities,
            ),
        ]
    )
    if any(row["passed"] is not True for row in checks):
        artifact["preconditions_checked"] = gate_summary(checks)
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        return _finish_artifact(artifact)

    frozen_bindings = _frozen_bindings(exp6866)
    resolved_rows = model_resolver(exp6866)
    resolved_by_id = {str(row.get("hf_id")): row for row in resolved_rows}
    checks.append(
        gate_check(
            "all_three_local_model_paths",
            list(MODEL_SPECS),
            sorted(
                model_hf_id
                for model_hf_id, row in resolved_by_id.items()
                if row.get("local_model_present") is True
            ),
            set(resolved_by_id) == set(MODEL_SPECS)
            and all(
                resolved_by_id[model_hf_id].get("local_model_present") is True
                for model_hf_id in MODEL_SPECS
            ),
        )
    )
    for model_hf_id in MODEL_SPECS:
        current = resolved_by_id.get(model_hf_id, {})
        frozen = frozen_bindings[model_hf_id]
        file_errors = [
            reason
            for reason in tokenizer_binding_errors(current, frozen)
            if reason != "canonical_tokenizer_payload_drift"
        ]
        checks.append(
            gate_check(f"model.{model_hf_id}.file_binding", [], file_errors, not file_errors)
        )
    artifact["model_artifact_hashes"] = {
        model_hf_id: {
            key: resolved_by_id.get(model_hf_id, {}).get(key)
            for key in ("path", "sha256", "size_bytes", "quantization", "snapshot_identity")
        }
        for model_hf_id in MODEL_SPECS
    }
    if any(row["passed"] is not True for row in checks):
        artifact["preconditions_checked"] = gate_summary(checks)
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        return _finish_artifact(artifact)

    rows: list[JsonDict] = []
    receipts: list[JsonDict] = []
    for model_hf_id in MODEL_SPECS:
        tokenizer: Any | None = None
        try:
            tokenizer = tokenizer_factory(str(resolved_by_id[model_hf_id]["path"]))
            receipt = _tokenizer_receipt(
                tokenizer,
                model_hf_id=model_hf_id,
                exp6866=exp6866,
            )
            current_binding = {
                **resolved_by_id[model_hf_id],
                "canonical_tokenizer_payload_sha256": receipt["canonical_tokenizer_payload_sha256"],
            }
            errors = tokenizer_binding_errors(current_binding, frozen_bindings[model_hf_id])
            checks.append(
                gate_check(
                    f"model.{model_hf_id}.canonical_tokenizer_binding",
                    [],
                    errors,
                    not errors,
                )
            )
            receipts.append(receipt)
            if errors:
                continue
            for group in groups:
                rows.append(
                    render_group_cell(
                        tokenizer=tokenizer,
                        tokenizer_receipt=receipt,
                        model_hf_id=model_hf_id,
                        group=group,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - terminal blocked evidence must survive.
            checks.append(
                gate_check(
                    f"model.{model_hf_id}.native_tokenizer",
                    "native vocabulary tokenization completes",
                    f"{type(exc).__name__}:{exc}",
                    False,
                )
            )
        finally:
            if tokenizer is not None:
                tokenizer.close()
    artifact["tokenizer_receipts"] = receipts
    artifact["preconditions_checked"] = gate_summary(checks)
    if any(row["passed"] is not True for row in checks):
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        return _finish_artifact(artifact)

    split_by_group = {
        **{group_id: "calibration" for group_id in split_manifest["calibration_group_ids"]},
        **{group_id: "held" for group_id in split_manifest["held_group_ids"]},
    }
    for row in rows:
        group_id = str(row["semantic_group_identity"])
        row["split"] = split_by_group.get(group_id, "unassigned")
        row["score_identity"] = score_identity(
            str(row["model_hf_id"]), group_id, split_manifest["split_checksum"]
        )
    accepted_rows = [row for row in rows if row.get("accepted") is True]
    rejected_rows = [row for row in rows if row.get("accepted") is not True]
    groups_by_id = {str(group.get("semantic_identity")): group for group in groups}
    accepted_models_by_group: dict[str, set[str]] = defaultdict(set)
    for row in accepted_rows:
        accepted_models_by_group[str(row["semantic_group_identity"])].add(str(row["model_hf_id"]))

    calibration_labels: dict[str, tuple[bool, bool]] = {}
    calibration_manifest: list[JsonDict] = []
    for group_id in split_manifest["calibration_group_ids"]:
        group = groups_by_id[group_id]
        candidates = [row for row in group.get("candidates", []) if isinstance(row, Mapping)]
        labels = tuple(bool(row.get("expected_label")) for row in candidates[:2])
        calibration_labels[group_id] = labels  # type: ignore[assignment]
        calibration_manifest.append(
            {
                "semantic_group_identity": group_id,
                "semantic_family": group.get("family"),
                "candidate_ids": [row.get("candidate_id") for row in candidates[:2]],
                "candidate_labels_by_slot": list(labels),
                "eligible_models": sorted(accepted_models_by_group[group_id]),
                "manifest_hash": sha256_json(
                    {
                        "semantic_group_identity": group_id,
                        "candidate_labels_by_slot": labels,
                        "eligible_models": sorted(accepted_models_by_group[group_id]),
                    }
                ),
            }
        )
    held_groups = [groups_by_id[group_id] for group_id in split_manifest["held_group_ids"]]
    private_salt = hashlib.sha256(
        (
            FROZEN_CONTRAST_BANK_HASH
            + split_manifest["split_checksum"]
            + str(RANDOM_SEED)
            + "exp6867-held-label-seal-v2"
        ).encode("utf-8")
    ).digest()
    held_manifest = seal_held_groups(held_groups, secret_salt=private_salt)
    for row in held_manifest:
        group_id = str(row["semantic_group_identity"])
        row["eligible_models"] = sorted(accepted_models_by_group[group_id])
        row["manifest_hash"] = sha256_json(row)
    commitments = {
        str(row["semantic_group_identity"]): str(row["label_commitment"]) for row in held_manifest
    }
    public_rows = _redact_held_rows(rows, commitments)
    public_by_sequence = {str(row["sequence_identity"]): row for row in public_rows}

    loader = CalibrationLabelLoader(
        calibration_labels,
        split_manifest["held_group_ids"],
    )
    for group_id in split_manifest["calibration_group_ids"]:
        loader.load(group_id)
    split_failures = split_errors(
        split_manifest["calibration_group_ids"],
        split_manifest["held_group_ids"],
        public_rows,
    )
    sample_rows = _sample_size_rows(accepted_rows, split_manifest)
    rejected_reason_counts = Counter(
        reason for row in rejected_rows for reason in row.get("rejection_reasons", [])
    )
    seal_passed = (
        len(held_manifest) == len(split_manifest["held_group_ids"])
        and not _contains_key(
            held_manifest,
            {"expected_label", "candidate_labels_by_slot"},
        )
        and not _contains_key(
            [row for row in public_rows if row.get("split") == "held"],
            {"expected_label", "candidate_labels_by_slot"},
        )
    )
    all_other_checks_pass = (
        not split_failures
        and not family_errors
        and seal_passed
        and loader.held_label_access_count == 0
        and len(receipts) == len(MODEL_SPECS)
        and all(not cell_nuisance_errors(row) for row in accepted_rows)
    )
    ready = readiness_score(sample_rows, all_other_checks_pass=all_other_checks_pass)
    checks.extend(
        [
            gate_check(
                "nuisance_eligible_cells_present", True, bool(accepted_rows), bool(accepted_rows)
            ),
            gate_check("split_overlap_count", 0, len(split_failures), not split_failures),
            gate_check("held_label_seal", True, seal_passed, seal_passed),
            gate_check(
                "held_label_access_count",
                0,
                loader.held_label_access_count,
                loader.held_label_access_count == 0,
            ),
            gate_check(
                "sample_size_floors",
                [],
                [row for row in sample_rows if row["floor_passed"] is not True],
                all(row["floor_passed"] is True for row in sample_rows),
            ),
        ]
    )
    artifact.update(
        models_used=list(MODEL_SPECS),
        rows=public_rows,
        accepted_cell_manifest=[
            {
                "score_identity": row["score_identity"],
                "sequence_identity": row["sequence_identity"],
                "model_hf_id": row["model_hf_id"],
                "semantic_group_identity": row["semantic_group_identity"],
                "semantic_family": row["semantic_family"],
                "split": row["split"],
                "cell_hash": sha256_json(public_by_sequence[str(row["sequence_identity"])]),
            }
            for row in accepted_rows
        ],
        rejected_cell_manifest=[
            {
                "score_identity": row["score_identity"],
                "sequence_identity": row["sequence_identity"],
                "model_hf_id": row["model_hf_id"],
                "semantic_group_identity": row["semantic_group_identity"],
                "semantic_family": row["semantic_family"],
                "split": row["split"],
                "reasons": row["rejection_reasons"],
                "cell_hash": sha256_json(public_by_sequence[str(row["sequence_identity"])]),
            }
            for row in rejected_rows
        ],
        nuisance_match_manifest={
            "controls": list(NUISANCE_CONTROLS),
            "candidate_count_rule": "exactly_two_candidates",
            "candidate_label_rule": "base_true_false_and_swap_false_true",
            "candidate_length_rule": "equal_native_token_and_character_counts",
            "prompt_rule": "base_and_label_swap_use_identical_native_prompt_tokens",
            "normalization": "NFC",
            "surface_padding_rule": "bounded_trailing_whitespace_exact_match",
            "maximum_extra_character_budget": MAX_WHITESPACE_EXTRA,
            "accepted_cell_count": len(accepted_rows),
            "rejected_cell_count": len(rejected_rows),
            "rejected_reason_counts": dict(sorted(rejected_reason_counts.items())),
            "manifest_hash": sha256_json(
                [
                    [row["sequence_identity"], row["accepted"], row["rejection_reasons"]]
                    for row in rows
                ]
            ),
        },
        calibration_group_manifest=calibration_manifest,
        sealed_held_group_manifest=held_manifest,
        split_overlap_count=len(
            [reason for reason in split_failures if reason.startswith("split_overlap:")]
        ),
        held_label_access_count=loader.held_label_access_count,
        sample_size_power_rows=sample_rows,
        semantic_contrast_preregistration_v2_ready_score=ready,
        gate_check_summary=gate_summary(checks),
        verdict_class="positive" if ready else "null",
        honest_verdict=READY_VERDICT if ready else LOW_SAMPLE_VERDICT,
        duration_s=round(time.monotonic() - started, 6),
    )
    return _finish_artifact(artifact)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete artifact atomically to the requested path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI path.
    """Run Exp6867 and write its terminal result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"invalid Exp6867 artifact: {errors}")
    write_json_atomic(args.output, artifact)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "ready": artifact["semantic_contrast_preregistration_v2_ready_score"],
                "accepted_cells": len(artifact["accepted_cell_manifest"]),
                "rejected_cells": len(artifact["rejected_cell_manifest"]),
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
