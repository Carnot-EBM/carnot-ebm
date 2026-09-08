"""Compare bounded generation arms under final exact admission.

Spec refs: REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-*.

The model can choose which proposal to inspect next. It cannot authorize a
proposal. The frozen instance checker is the only admission authority, and a
failed final proposal becomes an abstention instead of a repaired answer.
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
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import gpu_inventory, llama_cpp_probe
from carnot.experiment_7080_v620_three_family_entrance_bank import _lease_probe
from carnot.experiment_7129_v626_sota_constraint_bank import (
    _identity_row,
    _memory_telemetry,
    _response_fields,
    _terminalize_leases,
    _wait_for_vram_release,
    parse_model_text,
    verify_direct_answer,
)
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair
from carnot.task_runtime_receipts import sha256_file, write_json_atomic


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_7130_v626_verifier_committed_routing"
SCHEMA = "carnot.experiment_7130.v626_verifier_committed_routing.v1"
RUN_DATE = "20260908"
RANDOM_SEED = 7_130_202_609_08
RESULT_PATH = REPO_ROOT / "results/experiment_7130_v626_verifier_committed_routing.json"
UPSTREAM_PATH = REPO_ROOT / "results/experiment_7129_v626_sota_constraint_bank.json"
UPSTREAM_BANK_HASH = "sha256:e33ec32270a3ca7a794d63c45c814251e549f1e94a4cbcfb0122465ae9c7516d"
RAW_DIR = REPO_ROOT / "results/raw/experiment_7130_v626_verifier_committed_routing"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
INFERENCE_SUBSTRATE = "model_bounded_generation: exact commitment and uncertainty routing"
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
PREFERRED_QUANT = "Q4_K_M"
MODEL_TIMEOUT_S = 7_200.0
UNCERTAINTY_RETRY_THRESHOLD = 0.10

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SOTA_REGISTRY_IDS = tuple(row["hf_id"] for row in SOTA_GGUF_MODELS)
CONSTRAINT_FAMILIES = ("sat_logic", "graph_coloring", "bounded_scheduling")
ARM_NAMES = ("single_shot", "self_review", "exact_commitment", "uncertainty_router")
ARM_PLANS: list[JsonDict] = [
    {
        "arm": "single_shot",
        "max_generation_tokens": 192,
        "stages": [{"stage": "proposal", "max_tokens": 192, "temperature": 0.0}],
    },
    {
        "arm": "self_review",
        "max_generation_tokens": 192,
        "stages": [{"stage": "reviewed_proposal", "max_tokens": 192, "temperature": 0.0}],
    },
    {
        "arm": "exact_commitment",
        "max_generation_tokens": 192,
        "stages": [{"stage": "committed_proposal", "max_tokens": 192, "temperature": 0.0}],
    },
    {
        "arm": "uncertainty_router",
        "max_generation_tokens": 192,
        "stages": [
            {"stage": "sample_a", "max_tokens": 48, "temperature": 0.2},
            {"stage": "sample_b", "max_tokens": 48, "temperature": 0.8},
            {"stage": "retry", "max_tokens": 96, "temperature": 0.2},
        ],
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "MODEL_SPECS",
    "models_used",
    "model_repository_rows",
    "model_path_rows",
    "model_hash_rows",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "gpu_telemetry_rows",
    "token_rows",
    "duration_s",
    "source_artifact_hashes",
    "upstream_bank_hash",
    "raw_trace_manifest",
    "rows",
    "arm_rows",
    "exact_penalty_rows",
    "uncertainty_rows",
    "routing_rows",
    "retry_rows",
    "abstention_rows",
    "accepted_action_rows",
    "rejected_promotion_rows",
    "relabel_sensitivity_rows",
    "paraphrase_consistency_rows",
    "model_identity_confound_rows",
    "useful_retry_rate",
    "accepted_error_rate",
    "exact_rejected_actions_promoted",
    "verifier_committed_routing_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for each required field exposes silent evidence loss.",
    "preconditions_checked": "Observed gates stop model work when the frozen inputs or host are unsafe.",
    "run_date": "The fixed execution date distinguishes this run from later replays.",
    "MODEL_SPECS": "Ordered declarations prevent model or quantization substitution.",
    "models_used": "The invoked roster exposes a missing model family.",
    "model_repository_rows": "Repository rows retain the public model identity.",
    "model_path_rows": "Resolved local paths bind each invocation to one cached file.",
    "model_hash_rows": "File hashes detect silent weight replacement.",
    "inference_substrate": "The declaration states that bounded model generation supplied the proposals.",
    "inference_substrate_class": "The closed class separates live generation from a preflight block.",
    "execution_venue": "The host venue prevents unattributed remote execution.",
    "gpu_telemetry_rows": "Device readings show CUDA use and release around each model.",
    "token_rows": "Per-unit token counts expose truncation and unequal use.",
    "duration_s": "Measured wall time exposes interruption and implausible execution.",
    "source_artifact_hashes": "Source hashes bind the result to reviewed code, tests, and context.",
    "upstream_bank_hash": "The byte hash binds every arm to the frozen exact-labeled bank.",
    "raw_trace_manifest": "Content-addressed shards retain model output before parsing or labels.",
    "rows": "One row per arm and cell prevents aggregate-only claims.",
    "arm_rows": "Separate model and family metrics prevent pooled reversals.",
    "exact_penalty_rows": "Independent failure counts show why exact admission accepted or rejected.",
    "uncertainty_rows": "Repeated-output features expose routing uncertainty without authority labels.",
    "routing_rows": "Requested and final actions make every route replayable.",
    "retry_rows": "Retry receipts show bounded feedback and same-model use.",
    "abstention_rows": "Explicit abstentions preserve their generation cost.",
    "accepted_action_rows": "Accepted rows prove that executed proposals passed exact admission.",
    "rejected_promotion_rows": "Denied proposals prove that learned priority never overrides rejection.",
    "relabel_sensitivity_rows": "Paired symbol changes measure surface sensitivity.",
    "paraphrase_consistency_rows": "Paired wording changes measure semantic surface consistency.",
    "model_identity_confound_rows": "Explicit identities prevent one model from hiding another model's reversal.",
    "useful_retry_rate": "Successful recovery after a rejected proposal measures retry value.",
    "accepted_error_rate": "Accepted exact failures directly measure unsafe execution.",
    "exact_rejected_actions_promoted": "The zero count is the non-negotiable authority safety gate.",
    "verifier_committed_routing_complete_score": "One means every gate, row, budget, and admission replay completed.",
    "random_seed": "A fixed controller seed makes requests and routing repeatable.",
    "reproducibility_checksum": "A canonical digest detects later artifact mutation.",
    "gate_check_summary": "The first failed check retains its expected and observed values.",
    "verifier_is_oracle": "False keeps exact authority labels outside the uncertainty score.",
    "verdict_class": "A closed terminal class separates safety completion from metric uplift.",
    "honest_verdict": "A class-matching prefix states the outcome without hiding null uplift.",
}

FORBIDDEN_FEATURE_MARKERS = (
    "exact",
    "penalty",
    "solver",
    "witness",
    "objective",
    "solution",
    "answer_id",
    "correct",
    "label",
)
ALLOWED_UNCERTAINTY_FEATURES = frozenset(
    {"parsed_answer_hashes", "parse_successes", "completion_token_counts"}
)


def canonical_json(value: Any) -> str:
    """Serialize evidence with one stable byte spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash one UTF-8 string with the repository prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so JSON formatting cannot change the bank identity."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except the digest that contains the result."""

    return sha256_text(
        canonical_json(
            {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
        )
    )


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Retain an exact expected-observed gate decision."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose all checks and promote the first failure for automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def resolve_model_specs(
    *, cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair
) -> list[JsonDict]:
    """Resolve all three mandated files through explicit cached-pair calls."""

    first = (
        cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(0, 2))
        or []
    )
    second = (
        cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(1, 0))
        or []
    )
    resolved = {
        str(row.get("hf_id")): str(row.get("model_path") or "") for row in [*first, *second]
    }
    return [
        {
            "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
            "hf_id": model_id,
            "model_path": resolved.get(model_id, ""),
            "gpu_indices": [0, 1],
            "quantization": PREFERRED_QUANT,
            "chat_template_source": "embedded_gguf",
            "resolution_method": "cached_sota_pair",
            "remote_allowed": False,
        }
        for model_id in REQUIRED_MODEL_IDS
    ]


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject any roster, path, quantization, or template substitution."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_roster_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id"))
        path = str(row.get("model_path") or "")
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_language_gguf:{model_id}")
        if row.get("quantization") != PREFERRED_QUANT:
            errors.append(f"model_quantization_mismatch:{model_id}")
        if row.get("chat_template_source") != "embedded_gguf":
            errors.append(f"chat_template_source_mismatch:{model_id}")
        if row.get("resolution_method") != "cached_sota_pair":
            errors.append(f"model_resolution_method_mismatch:{model_id}")
        if row.get("gpu_indices") != [0, 1] or row.get("remote_allowed") is not False:
            errors.append(f"model_execution_policy_mismatch:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def upstream_gate_checks(bank: Mapping[str, Any], path: Path) -> list[JsonDict]:
    """Recheck the bare producer field and the frozen artifact bytes."""

    producer = bank.get("sota_constraint_bank_ready_score")
    observed_hash = sha256_bytes(path.read_bytes()) if path.is_file() else None
    return [
        gate_row(
            "sota_constraint_bank_ready_score",
            1,
            producer,
            type(producer) is int and producer == 1,
        ),
        gate_row(
            "upstream_bank_hash",
            UPSTREAM_BANK_HASH,
            observed_hash,
            observed_hash == UPSTREAM_BANK_HASH,
        ),
        gate_row(
            "upstream_completed_cell_count",
            108,
            bank.get("completed_cell_count"),
            bank.get("completed_cell_count") == 108,
        ),
        gate_row(
            "upstream_model_roster",
            list(REQUIRED_MODEL_IDS),
            bank.get("models_used"),
            bank.get("models_used") == list(REQUIRED_MODEL_IDS),
        ),
    ]


def _feature_paths(value: Any, prefix: str = "") -> list[str]:
    """List nested feature paths so forbidden data cannot hide in a container."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(_feature_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_feature_paths(item, f"{prefix}[{index}]"))
    return paths


def _oracle_feature_errors(features: Mapping[str, Any]) -> list[str]:
    """Reject unknown and authority-derived uncertainty features."""

    errors = []
    for path in _feature_paths(features):
        normalized = path.lower()
        if any(marker in normalized for marker in FORBIDDEN_FEATURE_MARKERS):
            errors.append(f"oracle_feature_forbidden:{path}")
    for name in features:
        if name not in ALLOWED_UNCERTAINTY_FEATURES:
            errors.append(f"uncertainty_feature_not_allowlisted:{name}")
    return errors


def estimate_uncertainty(model_id: str, instance_id: str, features: Mapping[str, Any]) -> JsonDict:
    """Score disagreement from output-only features, never exact outcomes."""

    errors = _oracle_feature_errors(features)
    if errors:
        raise ValueError(errors[0])
    hashes = [str(value) for value in features.get("parsed_answer_hashes", [])]
    parses = [bool(value) for value in features.get("parse_successes", [])]
    counts = [max(0, int(value)) for value in features.get("completion_token_counts", [])]
    sample_count = max(len(hashes), len(parses), len(counts))
    if sample_count < 2 or not (len(hashes) == len(parses) == len(counts)):
        raise ValueError("uncertainty_requires_aligned_repeated_outputs")
    answer_disagreement = (len(set(hashes)) - 1) / (sample_count - 1)
    parse_disagreement = float(len(set(parses)) > 1)
    token_spread = (max(counts) - min(counts)) / max(1, max(counts))
    uncertainty = min(
        1.0, 0.60 * answer_disagreement + 0.25 * parse_disagreement + 0.15 * token_spread
    )
    return {
        "model_id": model_id,
        "instance_id": instance_id,
        "learner_feature_names": sorted(features),
        "sample_count": sample_count,
        "answer_disagreement": answer_disagreement,
        "parse_disagreement": parse_disagreement,
        "token_count_spread": token_spread,
        "uncertainty": uncertainty,
        "verifier_is_oracle": False,
    }


def uncertainty_diagnostics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report whether repeated-output uncertainty collapsed to one value."""

    values = [float(row["uncertainty"]) for row in rows]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "collapsed": len(set(values)) <= 1,
    }


def arm_budget_errors(plans: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require every declared stage total to equal the common arm cap."""

    errors: list[str] = []
    common = 192
    if [row.get("arm") for row in plans] != list(ARM_NAMES):
        errors.append("arm_roster_mismatch")
    for row in plans:
        total = sum(int(stage.get("max_tokens", 0) or 0) for stage in row.get("stages", []))
        declared = int(row.get("max_generation_tokens", 0) or 0)
        if total != common or declared != common:
            errors.append(f"arm_budget_mismatch:{row.get('arm')}:{total}:{common}")
    return errors


def expected_unit_keys(bank: Mapping[str, Any]) -> list[str]:
    """Enumerate the frozen model-instance-arm product."""

    cells = sorted(
        {(str(row["model_id"]), str(row["instance_id"])) for row in bank.get("rows", [])}
    )
    return [f"{model}|{instance}|{arm}" for model, instance in cells for arm in ARM_NAMES]


def unit_row_errors(rows: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> list[str]:
    """Reject duplicate, missing, or substituted unit keys."""

    keys = [str(row.get("unit_key")) for row in rows]
    if len(keys) != len(set(keys)):
        return ["duplicate_unit_key"]
    if set(keys) != set(expected):
        return ["unit_row_keys_mismatch"]
    return []


def expected_metric_keys() -> list[tuple[str, str, str]]:
    """Enumerate every arm, model, and constraint-family metric cell."""

    return [
        (arm, model, family)
        for arm in ARM_NAMES
        for model in REQUIRED_MODEL_IDS
        for family in CONSTRAINT_FAMILIES
    ]


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate only when the denominator is nonzero."""

    return numerator / denominator if denominator else None


def aggregate_arm_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep all model and family cells separate during reduction."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["arm"]), str(row["model_id"]), str(row["family"]))].append(row)
    metrics = []
    for key in expected_metric_keys():
        group = groups.get(key, [])
        count = len(group)
        retries = sum(bool(row.get("retry_attempted")) for row in group)
        accepted = [row for row in group if row.get("executed") is True]
        completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in group)
        prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in group)
        metrics.append(
            {
                "arm": key[0],
                "model_id": key[1],
                "family": key[2],
                "cell_count": count,
                "parse_rate": _rate(sum(bool(row.get("parse_success")) for row in group), count),
                "exact_success_rate": _rate(
                    sum(bool(row.get("exact_success")) for row in group), count
                ),
                "exact_violation_count": sum(
                    int(row.get("exact_violation_count", 0) or 0) for row in group
                ),
                "useful_retry_rate": _rate(
                    sum(bool(row.get("useful_retry")) for row in group), retries
                ),
                "harmful_retry_rate": _rate(
                    sum(bool(row.get("harmful_retry")) for row in group), retries
                ),
                "abstention_rate": _rate(sum(bool(row.get("abstained")) for row in group), count),
                "accepted_error_rate": _rate(
                    sum(not bool(row.get("exact_success")) for row in accepted), len(accepted)
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "wall_time_s": sum(float(row.get("duration_s", 0.0) or 0.0) for row in group),
                "pooled_models": False,
                "pooled_families": False,
            }
        )
    return metrics


def model_pooling_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require the exact disaggregated metric-key product."""

    observed = {
        (str(row.get("arm")), str(row.get("model_id")), str(row.get("family"))) for row in rows
    }
    return (
        []
        if observed == set(expected_metric_keys()) and len(rows) == len(observed)
        else ["model_family_metric_keys_mismatch"]
    )


def bounded_verifier_receipt(
    *, model_id: str, instance_id: str, failed_constraint_classes: Sequence[str], exact_penalty: int
) -> JsonDict:
    """Expose only failure classes and counts to a same-model retry."""

    classes = sorted(set(str(value) for value in failed_constraint_classes))
    receipt = {
        "model_id": model_id,
        "instance_id": instance_id,
        "same_model_required": True,
        "failed_constraint_count": int(exact_penalty),
        "failed_constraint_classes": classes,
    }
    return {**receipt, "receipt_hash": sha256_text(canonical_json(receipt))}


def assert_retry_same_model(receipt: Mapping[str, Any], retry_model_id: str) -> None:
    """Stop a retry if it changes the proposal's model family."""

    if receipt.get("model_id") != retry_model_id:
        raise ValueError("retry_model_family_mismatch")


def admit_action(
    *, unit_key: str, exact_penalty: int, requested_action: str, learned_priority_score: float
) -> JsonDict:
    """Make a positive exact penalty final, regardless of learned priority."""

    if int(exact_penalty) > 0:
        return {
            "unit_key": unit_key,
            "requested_action": requested_action,
            "final_action": "abstain",
            "exact_penalty": int(exact_penalty),
            "learned_priority_score": float(learned_priority_score),
            "executed": False,
            "promoted": False,
            "rejection_reason": "exact_rejection_is_final",
        }
    return {
        "unit_key": unit_key,
        "requested_action": requested_action,
        "final_action": "accept",
        "exact_penalty": 0,
        "learned_priority_score": float(learned_priority_score),
        "executed": True,
        "promoted": True,
        "rejection_reason": None,
    }


def verdict_for_complete_run(
    *, promoted: int, accepted_errors: int, collapsed: bool
) -> tuple[str, str]:
    """Separate authority safety from a possibly null scientific result."""

    if promoted or accepted_errors:
        return "disqualified", "disqualified_exact_admission_safety_failure"
    if collapsed:
        return "null", "null_complete_uncertainty_collapsed"
    return "positive", "positive_verifier_committed_routing_complete_metric_uplift_may_be_null"


def _failed_constraint_classes(
    family: str, parsed: Mapping[str, Any], parse_success: bool, exact: Mapping[str, Any]
) -> list[str]:
    """Name failed classes without exposing a witness or correct answer."""

    if not parse_success:
        return ["parse"]
    classes = []
    if int(exact.get("constraint_violation_count", 0) or 0) > 0:
        classes.append(
            {"sat_logic": "clause", "graph_coloring": "coloring", "bounded_scheduling": "schedule"}[
                family
            ]
        )
    if exact.get("objective_matches") is False:
        classes.append("objective")
    if parsed.get("status") not in {"SAT", "UNSAT"}:
        classes.append("status")
    return sorted(set(classes or ["constraint"]))


def _score_raw(raw: Mapping[str, Any], receipt: Mapping[str, Any]) -> JsonDict:
    """Parse one persisted output and independently compute exact failures."""

    parsed_row = parse_model_text(str(raw.get("raw_text", "")), str(receipt["family"]))
    exact = (
        verify_direct_answer(receipt, parsed_row["parsed"])
        if parsed_row["parse_success"]
        else {
            "exact_correct": False,
            "constraint_violation_count": 1,
            "objective_observed": None,
            "objective_matches": False,
        }
    )
    return {
        **parsed_row,
        **exact,
        "exact_penalty": int(exact["constraint_violation_count"]),
        "failed_constraint_classes": _failed_constraint_classes(
            str(receipt["family"]),
            dict(parsed_row.get("parsed") or {}),
            bool(parsed_row["parse_success"]),
            exact,
        ),
    }


def _answer_hash(scored: Mapping[str, Any]) -> str:
    """Hash a parsed proposal or its explicit parse-failure marker."""

    payload = scored.get("parsed") if scored.get("parse_success") else {"parse_success": False}
    return sha256_text(canonical_json(payload))


def _surface_rows(rows: Sequence[Mapping[str, Any]], variant: str) -> list[JsonDict]:
    """Pair canonical outcomes with one surface-only variant."""

    by_key = {
        (row["arm"], row["model_id"], row["base_id"], row["variant_kind"]): row for row in rows
    }
    output = []
    for arm in ARM_NAMES:
        for model in REQUIRED_MODEL_IDS:
            base_ids = sorted(
                {row["base_id"] for row in rows if row["arm"] == arm and row["model_id"] == model}
            )
            for base_id in base_ids:
                canonical = by_key.get((arm, model, base_id, "canonical"))
                changed = by_key.get((arm, model, base_id, variant))
                if canonical is None or changed is None:
                    continue
                left = bool(canonical["exact_success"])
                right = bool(changed["exact_success"])
                output.append(
                    {
                        "arm": arm,
                        "model_id": model,
                        "family": canonical["family"],
                        "base_id": base_id,
                        "variant_kind": variant,
                        "canonical_exact_success": left,
                        "variant_exact_success": right,
                        "exact_success_delta": int(right) - int(left),
                        "consistent": left == right,
                    }
                )
    return output


def _model_confound_rows(
    rows: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Retain identity and family ranges instead of one pooled model score."""

    output = []
    for spec in specs:
        model_id = str(spec["hf_id"])
        model_rows = [row for row in rows if row["model_id"] == model_id]
        family_rates = {}
        for family in CONSTRAINT_FAMILIES:
            group = [row for row in model_rows if row["family"] == family]
            family_rates[family] = _rate(
                sum(bool(row["exact_success"]) for row in group), len(group)
            )
        finite = [value for value in family_rates.values() if value is not None]
        output.append(
            {
                "model_id": model_id,
                "model_path": spec.get("model_path"),
                "model_sha256": spec.get("model_sha256"),
                "quantization": spec.get("quantization"),
                "chat_template_source": spec.get("chat_template_source"),
                "family_exact_success_rates": family_rates,
                "family_reversal_range": max(finite) - min(finite) if finite else None,
                "pooled_models": False,
                "pooled_families": False,
            }
        )
    return output


def _reduce_raw(
    bank: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Replay raw calls into exact penalties, routes, and final unit rows."""

    raw_by_key = {str(row["request_key"]): row for row in raw_rows}
    receipts = {str(row["instance_id"]): row for row in bank.get("solver_receipt_rows", [])}
    source_rows = sorted(
        bank.get("rows", []), key=lambda row: (str(row["model_id"]), str(row["instance_id"]))
    )
    rows: list[JsonDict] = []
    penalties: list[JsonDict] = []
    uncertainties: list[JsonDict] = []
    routes: list[JsonDict] = []
    retries: list[JsonDict] = []
    abstentions: list[JsonDict] = []
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for source in source_rows:
        model_id = str(source["model_id"])
        instance_id = str(source["instance_id"])
        receipt = receipts[instance_id]
        for arm in ARM_NAMES:
            unit_key = f"{model_id}|{instance_id}|{arm}"
            plan = next(row for row in ARM_PLANS if row["arm"] == arm)
            stage_names = [stage["stage"] for stage in plan["stages"] if stage["stage"] != "retry"]
            stage_raw = [raw_by_key.get(f"{unit_key}|{stage}") for stage in stage_names]
            if any(raw is None for raw in stage_raw):
                continue
            scored = [_score_raw(raw or {}, receipt) for raw in stage_raw]
            for raw, score, stage in zip(stage_raw, scored, stage_names, strict=True):
                penalties.append(
                    {
                        "unit_key": unit_key,
                        "candidate_stage": stage,
                        "model_id": model_id,
                        "instance_id": instance_id,
                        "family": source["family"],
                        "parse_success": score["parse_success"],
                        "exact_penalty": score["exact_penalty"],
                        "exact_success": score["exact_correct"],
                        "failed_constraint_classes": score["failed_constraint_classes"],
                        "raw_output_hash": raw["raw_output_hash"] if raw else None,
                    }
                )
            uncertainty = 0.0
            retry_attempted = False
            useful_retry = False
            harmful_retry = False
            primary = scored[0]
            final_score = primary
            final_raw = stage_raw[0] or {}
            requested_action = "accept"
            if arm == "uncertainty_router":
                features = {
                    "parsed_answer_hashes": [_answer_hash(score) for score in scored],
                    "parse_successes": [bool(score["parse_success"]) for score in scored],
                    "completion_token_counts": [
                        int(raw.get("completion_tokens", 0) or 0)
                        for raw in stage_raw
                        if raw is not None
                    ],
                }
                uncertainty_row = estimate_uncertainty(model_id, instance_id, features)
                uncertainty_row.update(
                    {
                        "unit_key": unit_key,
                        "sample_output_hashes": [
                            raw["raw_output_hash"] for raw in stage_raw if raw
                        ],
                    }
                )
                uncertainties.append(uncertainty_row)
                uncertainty = float(uncertainty_row["uncertainty"])
                if primary["exact_penalty"] > 0:
                    requested_action = (
                        "retry" if uncertainty >= UNCERTAINTY_RETRY_THRESHOLD else "abstain"
                    )
                    retry_raw = raw_by_key.get(f"{unit_key}|retry")
                    if requested_action == "retry" and retry_raw is not None:
                        retry_attempted = True
                        final_raw = retry_raw
                        final_score = _score_raw(retry_raw, receipt)
                        penalties.append(
                            {
                                "unit_key": unit_key,
                                "candidate_stage": "retry",
                                "model_id": model_id,
                                "instance_id": instance_id,
                                "family": source["family"],
                                "parse_success": final_score["parse_success"],
                                "exact_penalty": final_score["exact_penalty"],
                                "exact_success": final_score["exact_correct"],
                                "failed_constraint_classes": final_score[
                                    "failed_constraint_classes"
                                ],
                                "raw_output_hash": retry_raw["raw_output_hash"],
                            }
                        )
                        useful_retry = (
                            primary["exact_penalty"] > 0 and final_score["exact_penalty"] == 0
                        )
                        harmful_retry = (
                            primary["exact_penalty"] == 0 and final_score["exact_penalty"] > 0
                        )
                        verifier_receipt = bounded_verifier_receipt(
                            model_id=model_id,
                            instance_id=instance_id,
                            failed_constraint_classes=primary["failed_constraint_classes"],
                            exact_penalty=primary["exact_penalty"],
                        )
                        retries.append(
                            {
                                "unit_key": unit_key,
                                "model_id": model_id,
                                "retry_model_id": retry_raw["model_id"],
                                "same_model": retry_raw["model_id"] == model_id,
                                "verifier_receipt": verifier_receipt,
                                "retry_budget_tokens": retry_raw["max_tokens"],
                                "useful_retry": useful_retry,
                                "harmful_retry": harmful_retry,
                                "retry_exact_penalty": final_score["exact_penalty"],
                            }
                        )
                    elif requested_action == "retry":
                        continue
            admission = admit_action(
                unit_key=unit_key,
                exact_penalty=int(final_score["exact_penalty"]),
                requested_action="accept" if retry_attempted else requested_action,
                learned_priority_score=1.0 - uncertainty,
            )
            call_rows = [raw for raw in stage_raw if raw is not None]
            if retry_attempted:
                call_rows.append(final_raw)
            prompt_tokens = sum(int(raw.get("prompt_tokens", 0) or 0) for raw in call_rows)
            completion_tokens = sum(int(raw.get("completion_tokens", 0) or 0) for raw in call_rows)
            wall_time = sum(float(raw.get("duration_s", 0.0) or 0.0) for raw in call_rows)
            row = {
                "unit_key": unit_key,
                "arm": arm,
                "model_id": model_id,
                "instance_id": instance_id,
                "base_id": source["base_id"],
                "family": source["family"],
                "variant_kind": source["variant_kind"],
                "max_generation_tokens": plan["max_generation_tokens"],
                "parse_success": bool(final_score["parse_success"]),
                "exact_success": bool(final_score["exact_correct"]),
                "exact_violation_count": int(final_score["exact_penalty"]),
                "requested_action": admission["requested_action"],
                "final_action": admission["final_action"],
                "executed": admission["executed"],
                "promoted": admission["promoted"],
                "abstained": admission["final_action"] == "abstain",
                "retry_attempted": retry_attempted,
                "useful_retry": useful_retry,
                "harmful_retry": harmful_retry,
                "uncertainty": uncertainty if arm == "uncertainty_router" else None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "duration_s": wall_time,
                "primary_raw_output_hash": stage_raw[0]["raw_output_hash"]
                if stage_raw[0]
                else None,
                "final_raw_output_hash": final_raw.get("raw_output_hash"),
                "terminal_state": "complete",
            }
            rows.append(row)
            route = {
                "unit_key": unit_key,
                "arm": arm,
                "model_id": model_id,
                "instance_id": instance_id,
                "learned_priority_score": 1.0 - uncertainty,
                "uncertainty": uncertainty if arm == "uncertainty_router" else None,
                "requested_action": admission["requested_action"],
                "final_action": admission["final_action"],
                "exact_penalty": admission["exact_penalty"],
                "executed": admission["executed"],
                "promoted": admission["promoted"],
            }
            routes.append(route)
            if admission["executed"]:
                accepted.append(route)
            else:
                rejected_row = {**route, "rejection_reason": admission["rejection_reason"]}
                rejected.append(rejected_row)
                abstentions.append(
                    {
                        "unit_key": unit_key,
                        "arm": arm,
                        "model_id": model_id,
                        "instance_id": instance_id,
                        "exact_penalty": admission["exact_penalty"],
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "token_cost": prompt_tokens + completion_tokens,
                        "reason": admission["rejection_reason"],
                    }
                )
    return {
        "rows": rows,
        "arm_rows": aggregate_arm_rows(rows),
        "exact_penalty_rows": penalties,
        "uncertainty_rows": uncertainties,
        "routing_rows": routes,
        "retry_rows": retries,
        "abstention_rows": abstentions,
        "accepted_action_rows": accepted,
        "rejected_promotion_rows": rejected,
        "relabel_sensitivity_rows": _surface_rows(rows, "relabel"),
        "paraphrase_consistency_rows": _surface_rows(rows, "paraphrase"),
        "model_identity_confound_rows": _model_confound_rows(rows, specs),
    }


def _empty_reduction() -> JsonDict:
    """Return every row surface for an early terminal block."""

    return {
        "rows": [],
        "arm_rows": [],
        "exact_penalty_rows": [],
        "uncertainty_rows": [],
        "routing_rows": [],
        "retry_rows": [],
        "abstention_rows": [],
        "accepted_action_rows": [],
        "rejected_promotion_rows": [],
        "relabel_sensitivity_rows": [],
        "paraphrase_consistency_rows": [],
        "model_identity_confound_rows": [],
    }


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]] = (),
    raw_trace_manifest: Sequence[Mapping[str, Any]] = (),
    gpu_telemetry_rows: Sequence[Mapping[str, Any]] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a schema-complete blocked or measured terminal artifact."""

    specs = [deepcopy(dict(row)) for row in model_specs]
    reduction = _reduce_raw(bank, raw_rows, specs) if bank and raw_rows else _empty_reduction()
    rows = reduction["rows"]
    expected = expected_unit_keys(bank or {})
    row_errors = unit_row_errors(rows, expected) if expected else ["upstream_bank_unavailable"]
    metric_errors = model_pooling_errors(reduction["arm_rows"]) if rows else ["metrics_unavailable"]
    uncertainty = uncertainty_diagnostics(reduction["uncertainty_rows"])
    raw_errors = [
        str(row["request_key"]) for row in raw_rows if row.get("terminal_state") != "complete"
    ]
    promoted = sum(
        bool(row.get("promoted")) and int(row.get("exact_penalty", 0) or 0) > 0
        for row in reduction["rejected_promotion_rows"]
    )
    accepted_errors = sum(
        int(row.get("exact_penalty", 0) or 0) > 0 for row in reduction["accepted_action_rows"]
    )
    completed = (
        preconditions.get("all_passed") is True
        and not row_errors
        and not metric_errors
        and not arm_budget_errors(ARM_PLANS)
        and not raw_errors
        and len(reduction["uncertainty_rows"]) == 108
        and len(raw_trace_manifest) == 3
        and promoted == 0
        and accepted_errors == 0
    )
    if preconditions.get("all_passed") is not True:
        verdict_class, honest_verdict = "blocked", "blocked_verifier_committed_routing_precondition"
    elif not completed:
        verdict_class, honest_verdict = "partial", "partial_verifier_committed_routing_incomplete"
    else:
        verdict_class, honest_verdict = verdict_for_complete_run(
            promoted=promoted,
            accepted_errors=accepted_errors,
            collapsed=bool(uncertainty["collapsed"]),
        )
    retry_count = len(reduction["retry_rows"])
    accepted_count = len(reduction["accepted_action_rows"])
    checks = [deepcopy(dict(row)) for row in preconditions.get("checks", [])]
    if preconditions.get("all_passed") is True:
        observed_completion = {
            "unit_row_errors": row_errors,
            "metric_errors": metric_errors,
            "raw_error_count": len(raw_errors),
            "uncertainty_row_count": len(reduction["uncertainty_rows"]),
            "manifest_count": len(raw_trace_manifest),
            "promoted": promoted,
            "accepted_errors": accepted_errors,
        }
        checks.append(
            gate_row(
                "verifier_committed_routing_completion",
                {
                    "unit_row_errors": [],
                    "metric_errors": [],
                    "raw_error_count": 0,
                    "uncertainty_row_count": 108,
                    "manifest_count": 3,
                    "promoted": 0,
                    "accepted_errors": 0,
                },
                observed_completion,
                completed,
            )
        )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": {**deepcopy(dict(preconditions)), "checks": checks},
        "run_date": run_date,
        "MODEL_SPECS": specs,
        "models_used": [row.get("hf_id") for row in specs if row.get("model_sha256")],
        "model_repository_rows": [
            {"model_id": row.get("hf_id"), "repository": row.get("hf_id")} for row in specs
        ],
        "model_path_rows": [
            {"model_id": row.get("hf_id"), "model_path": row.get("model_path")} for row in specs
        ],
        "model_hash_rows": [
            {"model_id": row.get("hf_id"), "model_sha256": row.get("model_sha256")} for row in specs
        ],
        "inference_substrate": INFERENCE_SUBSTRATE if completed or rows else "blocked_no_run",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS
        if completed or rows
        else "blocked_no_run",
        "execution_venue": "host",
        "gpu_telemetry_rows": [deepcopy(dict(row)) for row in gpu_telemetry_rows],
        "token_rows": [
            {
                "unit_key": row["unit_key"],
                "model_id": row["model_id"],
                "instance_id": row["instance_id"],
                "arm": row["arm"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
            }
            for row in rows
        ],
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes or {})),
        "upstream_bank_hash": UPSTREAM_BANK_HASH if UPSTREAM_PATH.is_file() else None,
        "raw_trace_manifest": [deepcopy(dict(row)) for row in raw_trace_manifest],
        **reduction,
        "arm_plans": deepcopy(ARM_PLANS),
        "uncertainty_diagnostics": uncertainty,
        "useful_retry_rate": _rate(
            sum(bool(row["useful_retry"]) for row in reduction["retry_rows"]), retry_count
        ),
        "harmful_retry_rate": _rate(
            sum(bool(row["harmful_retry"]) for row in reduction["retry_rows"]), retry_count
        ),
        "abstention_rate": _rate(len(reduction["abstention_rows"]), len(rows)),
        "abstention_token_cost": sum(
            int(row["token_cost"]) for row in reduction["abstention_rows"]
        ),
        "accepted_error_rate": _rate(accepted_errors, accepted_count) or 0.0,
        "exact_rejected_actions_promoted": promoted,
        "verifier_committed_routing_complete_score": int(completed),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay required fields, rows, aggregates, authority, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"required_field_missing:{field}")
        if not str(dict(artifact.get("field_principles") or {}).get(field, "")).strip():
            errors.append(f"field_principle_missing:{field}")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    verdict_class = str(artifact.get("verdict_class"))
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(verdict_class):
        errors.append("honest_verdict_class_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    gate = dict(artifact.get("gate_check_summary") or {})
    if verdict_class == "blocked":
        if gate.get("passed") is not False or not gate.get("failed_check"):
            errors.append("blocked_gate_summary_missing_failure")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("verifier_committed_routing_complete_score") != 0:
            errors.append("blocked_complete_score_nonzero")
        return list(dict.fromkeys(errors))
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    errors.extend(model_spec_errors(artifact.get("MODEL_SPECS", [])))
    errors.extend(arm_budget_errors(artifact.get("arm_plans", [])))
    try:
        bank = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
        expected = expected_unit_keys(bank)
        errors.extend(unit_row_errors(artifact.get("rows", []), expected))
    except (OSError, json.JSONDecodeError):
        errors.append("upstream_bank_unavailable")
    rows = list(artifact.get("rows", []))
    if any(row.get("max_generation_tokens") != 192 for row in rows):
        errors.append("row_arm_budget_mismatch")
    expected_metrics = aggregate_arm_rows(rows)
    errors.extend(model_pooling_errors(artifact.get("arm_rows", [])))
    if artifact.get("arm_rows") != expected_metrics:
        errors.append("arm_aggregate_mismatch")
    for row in artifact.get("uncertainty_rows", []):
        names = {str(name) for name in row.get("learner_feature_names", [])}
        if names != ALLOWED_UNCERTAINTY_FEATURES or any(
            any(marker in name.lower() for marker in FORBIDDEN_FEATURE_MARKERS) for name in names
        ):
            errors.append("oracle_feature_forbidden")
    diagnostics = uncertainty_diagnostics(artifact.get("uncertainty_rows", []))
    if artifact.get("uncertainty_diagnostics") != diagnostics:
        errors.append("uncertainty_summary_mismatch")
    rejected = list(artifact.get("rejected_promotion_rows", []))
    if any(
        int(row.get("exact_penalty", 0) or 0) > 0 and (row.get("promoted") or row.get("executed"))
        for row in rejected
    ):
        errors.append("exact_rejected_action_promoted")
    promoted = sum(
        int(row.get("exact_penalty", 0) or 0) > 0 and bool(row.get("promoted")) for row in rejected
    )
    if artifact.get("exact_rejected_actions_promoted") != promoted:
        errors.append("rejected_promotion_count_mismatch")
    accepted = list(artifact.get("accepted_action_rows", []))
    accepted_errors = sum(int(row.get("exact_penalty", 0) or 0) > 0 for row in accepted)
    accepted_error_rate = _rate(accepted_errors, len(accepted)) or 0.0
    if artifact.get("accepted_error_rate") != accepted_error_rate:
        errors.append("accepted_error_rate_mismatch")
    for row in artifact.get("retry_rows", []):
        receipt = dict(row.get("verifier_receipt") or {})
        try:
            assert_retry_same_model(receipt, str(row.get("retry_model_id")))
        except ValueError:
            errors.append("retry_model_family_mismatch")
        if set(receipt) & {
            "correct_answer",
            "witness",
            "objective",
            "solution_set_hash",
            "answer_id",
            "exact_penalty",
        }:
            errors.append("retry_receipt_oracle_leakage")
    expected_relabel = _surface_rows(rows, "relabel")
    expected_paraphrase = _surface_rows(rows, "paraphrase")
    if artifact.get("relabel_sensitivity_rows") != expected_relabel:
        errors.append("relabel_sensitivity_mismatch")
    if artifact.get("paraphrase_consistency_rows") != expected_paraphrase:
        errors.append("paraphrase_consistency_mismatch")
    complete = not errors and len(artifact.get("uncertainty_rows", [])) == 108
    if artifact.get("verifier_committed_routing_complete_score") != int(complete):
        errors.append("complete_score_mismatch")
    expected_class, expected_verdict = verdict_for_complete_run(
        promoted=promoted,
        accepted_errors=accepted_errors,
        collapsed=bool(diagnostics["collapsed"]),
    )
    if complete and (verdict_class, artifact.get("honest_verdict")) != (
        expected_class,
        expected_verdict,
    ):
        errors.append("complete_verdict_mismatch")
    return list(dict.fromkeys(errors))


def _storage_probe(path: Path) -> JsonDict:  # pragma: no cover - host filesystem boundary.
    """Check durable sibling-file creation without changing the target."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7130-write-probe-", dir=path.parent)
        os.write(descriptor, b"routing-storage-probe")
        os.fsync(descriptor)
        os.close(descriptor)
        size = Path(name).stat().st_size
        Path(name).unlink()
        return {"writable": True, "probe_bytes": size, "path": str(path)}
    except OSError as exc:
        return {
            "writable": False,
            "probe_bytes": 0,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_preconditions(  # pragma: no cover - live host, files, and CUDA boundary.
    *,
    bank: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    raw_dir: Path,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    """Check every stable prerequisite before loading full model weights."""

    specs = [deepcopy(dict(row)) for row in model_specs]
    bank_checks = upstream_gate_checks(bank, UPSTREAM_PATH)
    result_storage = _storage_probe(result_path)
    raw_storage = _storage_probe(raw_dir / "write-probe.jsonl")
    model_rows = []
    upstream_hashes = {row["model_id"]: row["sha256"] for row in bank.get("model_hash_rows", [])}
    for spec in specs:
        path = Path(str(spec.get("model_path") or ""))
        actual_hash = sha256_file(path) if path.is_file() else None
        spec["model_sha256"] = actual_hash
        model_rows.append(
            {
                "model_id": spec["hf_id"],
                "path": str(path),
                "exists": path.is_file(),
                "q4_k_m_name_match": "q4_k_m" in path.name.lower(),
                "actual_sha256": actual_hash,
                "upstream_sha256": upstream_hashes.get(spec["hf_id"]),
                "hash_matches_upstream": actual_hash == upstream_hashes.get(spec["hf_id"]),
            }
        )
    gpu = gpu_inventory()
    devices = list(gpu.get("devices", []))
    leases = _lease_probe(devices)
    llama = llama_cpp_probe()
    identity_rows = [_identity_row(spec) for spec in specs if spec.get("model_sha256")]
    identity_errors = [
        row["model_id"] for row in identity_rows if row.get("identity_matches") is not True
    ]
    checks = [
        *bank_checks,
        gate_row(
            "terminal_blocked_artifact_storage",
            {"writable": True},
            result_storage,
            result_storage.get("writable") is True,
        ),
        gate_row(
            "all_three_cached_q4_k_m_models",
            {model_id: True for model_id in REQUIRED_MODEL_IDS},
            {
                row["model_id"]: row["exists"]
                and row["q4_k_m_name_match"]
                and row["hash_matches_upstream"]
                for row in model_rows
            },
            len(model_rows) == 3
            and all(
                row["exists"] and row["q4_k_m_name_match"] and row["hash_matches_upstream"]
                for row in model_rows
            ),
        ),
        gate_row(
            "two_idle_gpu_leases",
            2,
            {
                "device_count": len(devices),
                "available_count": sum(row.get("classification") == "available" for row in leases),
                "lease_rows": leases,
            },
            len(devices) == 2
            and len(leases) == 2
            and all(row.get("classification") == "available" for row in leases),
        ),
        gate_row(
            "cuda_llama_cpp_health",
            {"importable": True, "gpu_offload": True},
            llama,
            llama.get("importable") is True and llama.get("gpu_offload") is True,
        ),
        gate_row(
            "raw_trace_storage",
            {"writable": True},
            raw_storage,
            raw_storage.get("writable") is True,
        ),
        gate_row("embedded_chat_templates", [], identity_errors, not identity_errors),
    ]
    telemetry = _memory_telemetry("preflight", gpu)
    return (
        {
            "all_passed": all(row["passed"] for row in checks),
            "checks": checks,
            "model_file_rows": model_rows,
            "gpu_topology": gpu,
            "gpu_lease_preflight_rows": leases,
            "llama_cpp": llama,
        },
        specs,
        identity_rows,
    )


def _system_prompt(arm: str, stage: str) -> str:
    """Freeze one direct-answer instruction for each registered arm stage."""

    if arm == "self_review":
        return "Solve the instance, review your own proposed assignment for every stated constraint, then return only the final JSON object."
    if arm == "exact_commitment":
        return "Solve the instance and commit to one direct proposal. Return only the JSON object; an external exact checker makes the final decision."
    if arm == "uncertainty_router":
        return f"Produce bounded independent proposal {stage}. Return only the requested direct JSON object and no answer ID."
    return "Return only the requested direct JSON answer. Do not use tools or answer IDs."


def _request_seed(model_id: str, instance_id: str, arm: str, stage: str) -> int:
    """Derive one stable positive llama.cpp seed from the frozen controller."""

    digest = hashlib.sha256(
        f"{RANDOM_SEED}|{model_id}|{instance_id}|{arm}|{stage}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _raw_path_for(model_id: str, raw_dir: Path) -> Path:  # pragma: no cover - path projection.
    """Map one fixed model identity to one durable JSONL shard."""

    return raw_dir / (re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-") + ".jsonl")


def load_raw_rows(path: Path) -> list[JsonDict]:
    """Load raw calls and reject duplicate keys or changed output hashes."""

    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys = [str(row.get("request_key")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_request_key")
    if any(row.get("raw_output_hash") != sha256_text(str(row.get("raw_text", ""))) for row in rows):
        raise ValueError("raw_output_hash_mismatch")
    return rows


def persist_raw_row(
    path: Path, row: Mapping[str, Any]
) -> None:  # pragma: no cover - durable file boundary.
    """Append a call before parsing and keep resume writes idempotent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prior = next(
        (item for item in load_raw_rows(path) if item["request_key"] == row["request_key"]), None
    )
    if prior is not None:
        if prior != dict(row):
            raise ValueError("raw_row_mismatch")
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _invoke(  # pragma: no cover - required live llama.cpp generation boundary.
    llm: Any,
    *,
    model_id: str,
    source: Mapping[str, Any],
    arm: str,
    stage: Mapping[str, Any],
    user_prompt: str,
) -> JsonDict:
    """Invoke one bounded request and return label-free raw evidence."""

    started = time.perf_counter()
    stage_name = str(stage["stage"])
    request_key = f"{model_id}|{source['instance_id']}|{arm}|{stage_name}"
    seed = _request_seed(model_id, str(source["instance_id"]), arm, stage_name)
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _system_prompt(arm, stage_name)},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=int(stage["max_tokens"]),
            temperature=float(stage["temperature"]),
            top_p=1.0,
            seed=seed,
        )
        raw_text, reasoning, prompt_tokens, completion_tokens = _response_fields(response)
        error = None
        terminal = "complete"
    except Exception as exc:  # noqa: BLE001 - failed calls must remain visible.
        response = {}
        raw_text, reasoning, prompt_tokens, completion_tokens = "", "", 0, 0
        error = f"{type(exc).__name__}: {exc}"
        terminal = "failed"
    return {
        "request_key": request_key,
        "unit_key": f"{model_id}|{source['instance_id']}|{arm}",
        "model_id": model_id,
        "instance_id": source["instance_id"],
        "base_id": source["base_id"],
        "family": source["family"],
        "variant_kind": source["variant_kind"],
        "arm": arm,
        "stage": stage_name,
        "prompt": user_prompt,
        "prompt_hash": sha256_text(user_prompt),
        "system_prompt_hash": sha256_text(_system_prompt(arm, stage_name)),
        "seed": seed,
        "max_tokens": int(stage["max_tokens"]),
        "temperature": float(stage["temperature"]),
        "raw_text": raw_text,
        "reasoning_text": reasoning,
        "raw_output_hash": sha256_text(raw_text),
        "raw_response_hash": sha256_text(canonical_json(response)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "duration_s": time.perf_counter() - started,
        "terminal_state": terminal,
        "error": error,
        "raw_persisted_before_parse": True,
        "parsed_at_write_time": False,
        "labeled_at_write_time": False,
    }


def run_model(  # pragma: no cover - required live CUDA and lease boundary.
    *,
    model: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    devices: Sequence[Mapping[str, Any]],
    raw_path: Path,
) -> JsonDict:
    """Run missing calls for one model under two owned GPU leases."""

    from llama_cpp import Llama

    baseline = gpu_inventory()
    leases = []
    telemetry = _memory_telemetry(f"before:{model['hf_id']}", baseline)
    resident = False
    llm: Any = None
    error = None
    after_gpu: JsonDict = {}
    release_receipt: JsonDict = {"passed": False}
    try:
        for device in devices:
            lease = lease_api.GpuLease.acquire(
                runtime_dir=LEASE_RUNTIME_DIR,
                task_id=EXPERIMENT_ID,
                device_uuid=str(device["uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
                ttl_s=MODEL_TIMEOUT_S + 300,
            )
            lease.transition("admitted")
            lease.transition("loading")
            leases.append(lease)
        llm = Llama(
            model_path=str(model["model_path"]),
            n_ctx=2_048,
            n_gpu_layers=-1,
            n_batch=512,
            n_ubatch=512,
            main_gpu=0,
            split_mode=1,
            tensor_split=[0.5, 0.5],
            seed=RANDOM_SEED,
            use_mmap=True,
            verbose=False,
        )
        loaded = gpu_inventory()
        telemetry.extend(_memory_telemetry(f"resident:{model['hf_id']}", loaded))
        expected_devices = {str(row["uuid"]) for row in devices}
        used_devices = {
            str(row.get("gpu_uuid"))
            for row in loaded.get("processes", [])
            if int(row.get("pid", -1)) == os.getpid()
        }
        resident = used_devices == expected_devices
        if not resident:
            raise RuntimeError(f"dual_gpu_residency_missing:{sorted(used_devices)}")
        memory = {str(row["uuid"]): int(row["memory_used_mb"]) for row in loaded["devices"]}
        for lease in leases:
            lease.transition("resident", vram_mb=memory.get(lease.device_uuid, 0))
            lease.transition("inferencing")
        existing = {row["request_key"]: row for row in load_raw_rows(raw_path)}
        for source in source_rows:
            for plan in ARM_PLANS:
                for stage in plan["stages"]:
                    if stage["stage"] == "retry":
                        continue
                    key = f"{model['hf_id']}|{source['instance_id']}|{plan['arm']}|{stage['stage']}"
                    if key not in existing:
                        raw = _invoke(
                            llm,
                            model_id=str(model["hf_id"]),
                            source=source,
                            arm=str(plan["arm"]),
                            stage=stage,
                            user_prompt=str(source["prompt"]),
                        )
                        persist_raw_row(raw_path, raw)
                        existing[key] = raw
            unit_key = f"{model['hf_id']}|{source['instance_id']}|uncertainty_router"
            samples = [existing[f"{unit_key}|sample_a"], existing[f"{unit_key}|sample_b"]]
            scored = [_score_raw(raw, receipts[str(source["instance_id"])]) for raw in samples]
            features = {
                "parsed_answer_hashes": [_answer_hash(row) for row in scored],
                "parse_successes": [bool(row["parse_success"]) for row in scored],
                "completion_token_counts": [int(row["completion_tokens"]) for row in samples],
            }
            uncertainty = estimate_uncertainty(
                str(model["hf_id"]), str(source["instance_id"]), features
            )
            retry_key = f"{unit_key}|retry"
            should_retry = (
                scored[0]["exact_penalty"] > 0
                and uncertainty["uncertainty"] >= UNCERTAINTY_RETRY_THRESHOLD
            )
            if should_retry and retry_key not in existing:
                verifier_receipt = bounded_verifier_receipt(
                    model_id=str(model["hf_id"]),
                    instance_id=str(source["instance_id"]),
                    failed_constraint_classes=scored[0]["failed_constraint_classes"],
                    exact_penalty=scored[0]["exact_penalty"],
                )
                retry_prompt = (
                    str(source["prompt"]).rstrip()
                    + "\nVerifier receipt: "
                    + canonical_json(
                        {
                            "failed_constraint_count": verifier_receipt["failed_constraint_count"],
                            "failed_constraint_classes": verifier_receipt[
                                "failed_constraint_classes"
                            ],
                        }
                    )
                    + "\nTry once more. Return only a new direct JSON proposal."
                )
                retry_stage = next(
                    stage for stage in ARM_PLANS[-1]["stages"] if stage["stage"] == "retry"
                )
                raw = _invoke(
                    llm,
                    model_id=str(model["hf_id"]),
                    source=source,
                    arm="uncertainty_router",
                    stage=retry_stage,
                    user_prompt=retry_prompt,
                )
                persist_raw_row(raw_path, raw)
                existing[retry_key] = raw
    except Exception as exc:  # noqa: BLE001 - model failures must produce terminal evidence.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        llm = None
        gc.collect()
        release_receipt, after_gpu = _wait_for_vram_release(baseline, str(model["hf_id"]))
        telemetry.extend(_memory_telemetry(f"after:{model['hf_id']}", after_gpu))
        complete = error is None and release_receipt.get("passed") is True
        lease_rows = _terminalize_leases(
            leases, resident=resident, complete=complete, release=after_gpu
        )
    return {
        "model_id": model["hf_id"],
        "raw_path": str(raw_path),
        "gpu_telemetry_rows": telemetry,
        "gpu_lease_rows": lease_rows,
        "release_receipt": release_receipt,
        "terminal_state": "complete" if error is None else "partial",
        "error": error,
    }


def _raw_manifest(
    model_specs: Sequence[Mapping[str, Any]], raw_dir: Path
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - live raw-file projection.
    """Return content-addressed shard receipts and all raw calls."""

    manifest = []
    rows = []
    for model in model_specs:
        path = _raw_path_for(str(model["hf_id"]), raw_dir)
        shard = load_raw_rows(path)
        rows.extend(shard)
        if path.is_file():
            manifest.append(
                {
                    "model_id": model["hf_id"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "row_count": len(shard),
                }
            )
    return manifest, rows


def _source_hashes() -> JsonDict:  # pragma: no cover - live repository bytes.
    """Hash implementation, tests, specs, prior artifacts, and requested context."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("results/experiment_6998_three_family_commitment_controls.json"),
        Path("results/experiment_7013_three_family_intervention_surface.json"),
        Path("results/experiment_7129_v626_sota_constraint_bank.json"),
        Path("scripts/experiment_template.py"),
        Path("scripts/adversarial_verify.py"),
        Path("python/carnot/constraint_ir_replay_contract.py"),
        Path("python/carnot/gpu_lease_phase_journal.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/experiment_7130_v626_verifier_committed_routing.py"),
        Path("scripts/experiments/experiment_7130_v626_verifier_committed_routing.py"),
        Path("tests/python/test_experiment_7130_v626_verifier_committed_routing.py"),
        Path("openspec/capabilities/constraint-verification/spec.md"),
        Path("openspec/capabilities/energy-verification/spec.md"),
        Path("openspec/capabilities/verifiable-reasoning/spec.md"),
        Path("openspec/capabilities/research-reporting/spec.md"),
    )
    return {
        str(path): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in paths
    }


def run(  # pragma: no cover - required live command boundary.
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    raw_dir: Path = RAW_DIR,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Gate first, write before GPU setup, invoke models, reduce, and validate."""

    if result_path.is_file():
        try:
            stable = json.loads(result_path.read_text(encoding="utf-8"))
            failed_check = dict(stable.get("gate_check_summary") or {}).get("failed_check")
            placeholder_checks = {"preflight_evaluated", "model_generation_complete"}
            if (
                stable.get("run_date") == run_date
                and failed_check not in placeholder_checks
                and not validate_artifact(stable)
            ):
                return stable
        except (OSError, json.JSONDecodeError):
            pass
    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    try:
        bank = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
        upstream_checks = upstream_gate_checks(bank, UPSTREAM_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        bank = {}
        upstream_checks = [
            gate_row("sota_constraint_bank_ready_score", 1, f"{type(exc).__name__}: {exc}", False)
        ]
    if not all(row["passed"] for row in upstream_checks):
        blocked = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            bank=bank,
            preconditions={"all_passed": False, "checks": upstream_checks},
            source_artifact_hashes=_source_hashes(),
        )
        write_json_atomic(result_path, blocked)
        return blocked
    initialized = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        bank=bank,
        preconditions={
            "all_passed": False,
            "checks": [
                gate_row(
                    "preflight_evaluated", True, "artifact initialized before GPU setup", False
                )
            ],
        },
    )
    write_json_atomic(result_path, initialized)
    preconditions, specs, identity_rows = collect_preconditions(
        bank=bank, model_specs=specs, result_path=result_path, raw_dir=raw_dir
    )
    sources = _source_hashes()
    if preconditions["all_passed"] is not True:
        blocked = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            bank=bank,
            preconditions=preconditions,
            gpu_telemetry_rows=_memory_telemetry("preflight", preconditions["gpu_topology"]),
            source_artifact_hashes=sources,
        )
        blocked["model_identity_confound_rows"] = identity_rows
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        write_json_atomic(result_path, blocked)
        return blocked
    pending = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        bank=bank,
        preconditions={
            "all_passed": False,
            "checks": [gate_row("model_generation_complete", True, False, False)],
        },
        gpu_telemetry_rows=_memory_telemetry("preflight", preconditions["gpu_topology"]),
        source_artifact_hashes=sources,
    )
    write_json_atomic(result_path, pending)
    receipts = {str(row["instance_id"]): row for row in bank["solver_receipt_rows"]}
    phases = []
    devices = preconditions["gpu_topology"]["devices"]
    for spec in specs:
        source_rows = [row for row in bank["rows"] if row["model_id"] == spec["hf_id"]]
        phase = run_model(
            model=spec,
            source_rows=source_rows,
            receipts=receipts,
            devices=devices,
            raw_path=_raw_path_for(str(spec["hf_id"]), raw_dir),
        )
        phases.append(phase)
    manifest, raw_rows = _raw_manifest(specs, raw_dir)
    telemetry = [
        *_memory_telemetry("preflight", preconditions["gpu_topology"]),
        *(row for phase in phases for row in phase["gpu_telemetry_rows"]),
    ]
    phase_errors = [phase for phase in phases if phase["terminal_state"] != "complete"]
    if phase_errors:
        preconditions["checks"].append(gate_row("model_phase_completion", [], phase_errors, False))
        preconditions["all_passed"] = False
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        bank=bank,
        preconditions=preconditions,
        raw_rows=raw_rows,
        raw_trace_manifest=manifest,
        gpu_telemetry_rows=telemetry,
        source_artifact_hashes=sources,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run or independently validate the requested terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(run_date=args.date, result_path=args.result_path, raw_dir=args.raw_dir)
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "row_count": len(artifact["rows"]),
                "verifier_committed_routing_complete_score": artifact[
                    "verifier_committed_routing_complete_score"
                ],
                "exact_rejected_actions_promoted": artifact["exact_rejected_actions_promoted"],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - module command surface.
    raise SystemExit(main())
