"""Audit typed grounding value on frozen Qwen calls and sealed labels.

The audit separates parse success, semantic execution, and end-to-end value.
It keeps abstentions in the denominator and repeats semantic controls in a
fresh process.

Spec refs: REQ-VERIFY-7197 and SCENARIO-VERIFY-7197-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import select
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_7196_v634_qwen_atomic_capture import parse_output
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    TypedRelation,
    execute_relation,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260911"
RANDOM_SEED = 7_197_202_609_11
BOOTSTRAP_SEED = 719_704
BOOTSTRAP_DRAWS = 10_000
AUDITOR_TIMEOUT_S = 300.0
ARMS = (
    "direct_judgment",
    "grammar_validity_only",
    "typed_execution_unknown_abstention",
    "lexical_overlap",
    "shuffled_source_typed_execution",
)
VARIANTS = (
    "original",
    "bijective_entity_rename",
    "relation_or_polarity_flip",
    "evidence_deletion",
)
EVALUATION_FAMILIES = (
    "starts before",
    "ends before",
    "is separated from",
    "occurs before",
)
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
ORACLE_RATIONALE = (
    "Typed execution and scoring use the same complete fixture correctness authority. "
    "A fresh process and separate code do not make that authority oracle-distinct. "
    "Any gain is execution-grounded and circular, not a learned-verifier moat."
)
RESULT_PATH = Path("results/experiment_7197_v634_grounding_value_audit.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7197_v634_grounding_value_audit.json")
AUDIT_REQUEST_PATH = Path(
    "results/checkpoints/experiment_7197_v634_grounding_value_audit_request.json"
)
AUDIT_RESULT_PATH = Path(
    "results/checkpoints/experiment_7197_v634_grounding_value_audit_response.json"
)
SOURCE_PATHS = {
    "exp7195_artifact": Path("results/experiment_7195_v634_typed_grounding.json"),
    "public_view": Path("results/experiment_7195_v634_typed_grounding_public.jsonl"),
    "authority_sidecar": Path("results/experiment_7195_v634_typed_grounding_authority.jsonl"),
    "exp7196_artifact": Path("results/experiment_7196_v634_qwen_atomic_capture.json"),
    "raw_manifest": Path("results/raw/experiment_7196_v634_qwen_atomic_capture/raw_manifest.json"),
    "constraint_spec": Path("openspec/capabilities/constraint-verification/spec.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "executor_module": Path("python/carnot/verify/experiment_7195_source_relation_executor.py"),
    "capture_module": Path("python/carnot/experiment_7196_v634_qwen_atomic_capture.py"),
    "module": Path("python/carnot/experiment_7197_v634_grounding_value_audit.py"),
    "auditor_module": Path(
        "python/carnot/experiment_7197_v634_grounding_value_independent_audit.py"
    ),
    "entrypoint": Path("scripts/experiments/experiment_7197_v634_grounding_value_audit.py"),
    "focused_tests": Path("tests/python/test_experiment_7197_v634_grounding_value_audit.py"),
}
PINNED_INPUT_HASHES = {
    "exp7195_artifact": "sha256:be16bf10a37010e12a3c4851c313c0d834a391ab2040be9201efdce9221054da",
    "public_view": "sha256:ecf5decc1ab53233c3ee011b38508d5e5e10bb3b30ebf6e7c0265397d939150b",
    "authority_sidecar": "sha256:8e664c926b58b8459fc3c2eb27016638d8055511ddf24b8ed05d4a72b908ce21",
    "exp7196_artifact": "sha256:ca16031902d446c66a842047d2ac1066602ddd1ed29deb24c9334b4356f1f6b4",
    "raw_manifest": "sha256:ee35d034a090db9240c13ad3460b9a164db65a6c1d576adfff5f9a702cd4efa2",
}
EXP7195_EXPECTED_FIELDS = {
    "status": "complete",
    "run_date": "20260910",
    "typed_executor_ready_score": 1,
    "verdict_class": "circular_positive",
    "honest_verdict": "complete_circular_positive_typed_executor_ready_no_independent_value_claim",
    "reproducibility_checksum": "sha256:653251a6e721daf041e0f22fbf2edefed819ff61fe9787dfd75b724cbe5b93cc",
    "public_view_path": "results/experiment_7195_v634_typed_grounding_public.jsonl",
    "authority_sidecar_path": "results/experiment_7195_v634_typed_grounding_authority.jsonl",
}
EXP7196_EXPECTED_FIELDS = {
    "status": "complete",
    "run_date": "20260910",
    "atomic_capture_complete_score": 1,
    "verdict_class": "null",
    "honest_verdict": "complete_null_atomic_capture_parse_poor_bank_available_for_independent_audit",
    "reproducibility_checksum": "sha256:481d4f3ec0da3154fd36a5811fe1cbabc1c2628858ecc42aed6990eda3dc59de",
    "inference_mode": "live_gpu",
}

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260911, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when verification uses the same correctness authority; separate implementations alone do not remove circularity.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings, including nulls; blocked_* for external blocks. Never promote infrastructure readiness as scientific benefit.",
    "grounding_audit_complete_score": "Completion and value are different fields.",
    "grounding_value_score": "One requires the preregistered accuracy or efficiency gate.",
    "arm_metrics": "Report full-denominator accuracy alongside coverage.",
    "paired_interval_rows": "Base-case clusters, not variants or seeds, define independent units.",
    "oracle_distinctness_rationale": "Equivalent ground-truth checks remain circular despite separate code.",
    "cold_audit_rows": "Independent recomputation checks each decision and intervention.",
    "accuracy_criterion": "Report the accuracy-branch test without turning alternative branches into contradictory acceptance gates.",
    "efficiency_criterion": "Report complete extraction costs and equal coverage for the efficiency alternative.",
    "acceptance_gate_value": "Pass iff the accuracy OR efficiency criterion passes and the independent semantic audit passes.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
    "scope_answer": "Label the 32-base pilot and avoid a broad model-performance claim.",
    "independent_audit": "Record the fresh process, label isolation, controls, and decision agreement.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Return one compact JSON spelling for stable hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes without changing line endings or encoding."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text without normalization."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash exact source bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding process-local runtime values."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    audit = payload.get("independent_audit")
    if isinstance(audit, dict):
        audit["process_id"] = None
    return sha256_text(canonical_json(payload))


def _progress(phase: int, event: str, detail: str) -> None:  # pragma: no cover
    """Print one flushed event before and after each phase or long call."""

    print(f"PHASE {phase} {event} {detail}", flush=True)


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read a frozen sidecar after its exact byte hash passes."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _call_feature(row: Mapping[str, Any]) -> JsonDict:
    """Reparse one raw call and retain only model-side evidence."""

    raw_output = str(row.get("raw_output", ""))
    if row.get("raw_output_sha256") != sha256_text(raw_output):
        raise ValueError(f"raw_output_hash_mismatch:{row.get('call_id')}")
    parsed = parse_output(str(row.get("call_type")), raw_output)
    return {
        "call_id": row.get("call_id"),
        "parse_status": parsed["parse_status"],
        "parse_error": parsed["parse_error"],
        "parsed": deepcopy(parsed["parsed"]),
        "raw_output_sha256": row.get("raw_output_sha256"),
        "latency_s": float(row.get("latency_s", 0.0)),
        "cache_hit": bool(row.get("cache_hit")),
        "cold_request": bool(row.get("cold_request")),
        "reuse_from_call_id": row.get("reuse_from_call_id"),
    }


def build_candidate_features(
    public_rows: Sequence[Mapping[str, Any]], completion_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reparse three raw calls per unit without any evaluator label input."""

    calls: dict[tuple[str, str], Mapping[str, Any]] = {}
    calls_by_id: dict[str, Mapping[str, Any]] = {}
    for row in completion_rows:
        unit_id = str(row.get("unit_id"))
        call_type = str(row.get("call_type"))
        key = (unit_id, call_type)
        call_id = str(row.get("call_id"))
        if key in calls or call_id in calls_by_id:
            raise ValueError("duplicate_call")
        calls[key] = row
        calls_by_id[call_id] = row
    features: list[JsonDict] = []
    for row_order, public in enumerate(public_rows):
        if set(public) != {"unit_id", "source_text", "claim_text"}:
            raise ValueError(f"public_shape:{row_order}")
        unit_id = str(public["unit_id"])
        required = [(unit_id, call_type) for call_type in ("source", "claim", "direct")]
        if any(key not in calls for key in required):
            raise ValueError(f"missing_call:{unit_id}")
        source_row = calls[(unit_id, "source")]
        source = _call_feature(source_row)
        claim = _call_feature(calls[(unit_id, "claim")])
        direct = _call_feature(calls[(unit_id, "direct")])
        source_cold_latency = source["latency_s"]
        if source["cache_hit"]:
            origin_id = source["reuse_from_call_id"]
            origin = calls_by_id.get(str(origin_id))
            if origin is None or origin.get("call_type") != "source":
                raise ValueError(f"cache_origin_missing:{unit_id}")
            source_cold_latency = float(origin.get("latency_s", 0.0))
        features.append(
            {
                "row_order": row_order,
                "unit_id": unit_id,
                "source_text": str(public["source_text"]),
                "claim_text": str(public["claim_text"]),
                "source_text_sha256": sha256_text(str(public["source_text"])),
                "claim_text_sha256": sha256_text(str(public["claim_text"])),
                "source": source,
                "claim": claim,
                "direct": direct,
                "source_cold_latency_s": source_cold_latency,
                "source_amortized_latency_s": source["latency_s"],
            }
        )
    if len(calls) != len(public_rows) * 3:
        raise ValueError("completion_call_roster")
    return features


def build_shuffle_map(authority_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Freeze same-family and same-variant donors without reading outcomes."""

    groups: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in authority_rows:
        key = (str(row["split"]), str(row["relation_family"]), str(row["variant"]))
        groups[key].append((str(row["base_id"]), str(row["unit_id"])))
    mapping: dict[str, str] = {}
    for values in groups.values():
        ordered = [unit_id for _, unit_id in sorted(values)]
        for index, unit_id in enumerate(ordered):
            mapping[unit_id] = ordered[(index + 1) % len(ordered)]
    if len(mapping) != len(authority_rows):
        raise ValueError("shuffle_map_roster")
    return mapping


def _typed_prediction(
    source_text: str,
    source_call: Mapping[str, Any],
    claim_call: Mapping[str, Any],
) -> JsonDict:
    """Execute separated typed outputs and preserve every unknown reason."""

    parse_errors = [
        f"{name}_parse_failed:{call.get('parse_error')}"
        for name, call in (("source", source_call), ("claim", claim_call))
        if call.get("parse_status") != "valid"
    ]
    if parse_errors:
        return {"prediction": "abstain", "error": ";".join(parse_errors)}
    source = source_call.get("parsed")
    claim = claim_call.get("parsed")
    if not isinstance(source, Mapping) or not isinstance(claim, Mapping):
        return {"prediction": "abstain", "error": "parsed_output_missing"}
    if source.get("missing_fields") or claim.get("missing_fields"):
        return {"prediction": "abstain", "error": "declared_missing_fields"}
    claim_relations = claim.get("relations", [])
    if not isinstance(claim_relations, list) or len(claim_relations) != 1:
        return {"prediction": "abstain", "error": "claim_relation_count"}

    source_bindings = tuple(EntityBinding(**item) for item in source.get("entity_bindings", []))
    claim_surfaces: dict[str, list[str]] = defaultdict(list)
    for item in claim.get("entity_bindings", []):
        claim_surfaces[str(item["entity_id"])].append(str(item["surface"]))
    source_ids: dict[str, list[str]] = defaultdict(list)
    for binding in source_bindings:
        source_ids[binding.surface].append(binding.entity_id)
    relation = claim_relations[0]
    mapped: dict[str, str] = {}
    for field in ("subject_id", "object_id"):
        surfaces = claim_surfaces.get(str(relation[field]), [])
        matches = source_ids.get(surfaces[0], []) if len(surfaces) == 1 else []
        if len(matches) != 1:
            return {"prediction": "abstain", "error": f"entity_mapping:{field}"}
        mapped[field] = matches[0]
    typed_claim = TypedRelation(
        mapped["subject_id"],
        str(relation["operator"]),
        mapped["object_id"],
        str(relation["polarity"]),
        int(relation["source_start"]),
        int(relation["source_end"]),
    )
    source_relations = tuple(TypedRelation(**item) for item in source.get("relations", []))
    result = execute_relation(
        source_text.encode("utf-8"), source_bindings, source_relations, typed_claim
    )
    prediction = {
        "supported": "supported",
        "contradicted": "unsupported",
        "unknown": "abstain",
    }[result.decision]
    return {
        "prediction": prediction,
        "error": ";".join(result.uncertainty_reasons) or None,
    }


def _lexical_score(source_text: str, claim_text: str) -> float:
    """Measure exact public token overlap as a semantics-free control."""

    pattern = re.compile(r"[a-z0-9_-]+")
    source_tokens = set(pattern.findall(source_text.lower()))
    claim_tokens = set(pattern.findall(claim_text.lower()))
    union = source_tokens | claim_tokens
    return len(source_tokens & claim_tokens) / len(union) if union else 0.0


def evaluate_arms(
    features: Sequence[Mapping[str, Any]], shuffle_map: Mapping[str, str]
) -> list[JsonDict]:
    """Freeze five policy outputs without evaluator labels."""

    by_id = {str(row["unit_id"]): row for row in features}
    if set(shuffle_map) != set(by_id) or any(donor not in by_id for donor in shuffle_map.values()):
        raise ValueError("shuffle_map_roster")
    rows: list[JsonDict] = []
    for arm in ARMS:
        for row_order, feature in enumerate(features):
            started = time.perf_counter()
            unit_id = str(feature["unit_id"])
            source = feature["source"]
            claim = feature["claim"]
            direct = feature["direct"]
            donor_id: str | None = None
            score: float | int | None = None
            error: str | None = None
            if arm == "direct_judgment":
                parse_success = direct["parse_status"] == "valid"
                prediction = str(direct["parsed"]["decision"]) if parse_success else "abstain"
                error = None if parse_success else f"direct_parse_failed:{direct['parse_error']}"
                cold_input_latency = float(direct["latency_s"])
                amortized_input_latency = cold_input_latency
            elif arm == "grammar_validity_only":
                parse_success = (
                    source["parse_status"] == "valid" and claim["parse_status"] == "valid"
                )
                prediction = "supported" if parse_success else "abstain"
                score = int(parse_success)
                error = None if parse_success else "separated_extraction_syntax_invalid"
                cold_input_latency = feature["source_cold_latency_s"] + claim["latency_s"]
                amortized_input_latency = feature["source_amortized_latency_s"] + claim["latency_s"]
            elif arm == "typed_execution_unknown_abstention":
                parse_success = (
                    source["parse_status"] == "valid" and claim["parse_status"] == "valid"
                )
                result = _typed_prediction(str(feature["source_text"]), source, claim)
                prediction, error = result["prediction"], result["error"]
                cold_input_latency = feature["source_cold_latency_s"] + claim["latency_s"]
                amortized_input_latency = feature["source_amortized_latency_s"] + claim["latency_s"]
            elif arm == "lexical_overlap":
                parse_success = True
                score = _lexical_score(str(feature["source_text"]), str(feature["claim_text"]))
                prediction = "supported" if score == 1.0 else "unsupported"
                cold_input_latency = 0.0
                amortized_input_latency = 0.0
            else:
                donor_id = str(shuffle_map[unit_id])
                donor = by_id[donor_id]
                parse_success = (
                    donor["source"]["parse_status"] == "valid" and claim["parse_status"] == "valid"
                )
                result = _typed_prediction(str(donor["source_text"]), donor["source"], claim)
                prediction, error = result["prediction"], result["error"]
                cold_input_latency = donor["source_cold_latency_s"] + claim["latency_s"]
                amortized_input_latency = donor["source_amortized_latency_s"] + claim["latency_s"]
            policy_latency = time.perf_counter() - started
            selected_source = by_id[donor_id] if donor_id is not None else feature
            rows.append(
                {
                    "row_order": row_order,
                    "unit_id": unit_id,
                    "arm": arm,
                    "seed": RANDOM_SEED,
                    "prediction": prediction,
                    "parse_status": "valid" if parse_success else "failed",
                    "parse_success": parse_success,
                    "score": score,
                    "error": error,
                    "abstention": prediction == "abstain",
                    "donor_unit_id": donor_id,
                    "raw_output_hashes": {
                        "source": selected_source["source"]["raw_output_sha256"],
                        "claim": claim["raw_output_sha256"],
                        "direct": direct["raw_output_sha256"],
                    },
                    "source_text_sha256": selected_source["source_text_sha256"],
                    "claim_text_sha256": feature["claim_text_sha256"],
                    "policy_latency_s": policy_latency,
                    "cold_latency_s": cold_input_latency + policy_latency,
                    "amortized_latency_s": amortized_input_latency + policy_latency,
                }
            )
    return rows


def prediction_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash policy decisions while excluding labels and timing noise."""

    projected = [
        {
            "unit_id": row["unit_id"],
            "arm": row["arm"],
            "prediction": row["prediction"],
            "parse_status": row["parse_status"],
            "error": row["error"],
            "donor_unit_id": row["donor_unit_id"],
            "raw_output_hashes": row["raw_output_hashes"],
        }
        for row in rows
    ]
    return sha256_text(canonical_json(projected))


def score_evaluation(
    predictions: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Open held-out labels after policies freeze and score all 128 units."""

    evaluation = [row for row in authority_rows if row.get("split") == "evaluation"]
    if len(evaluation) != 128 or {str(row["relation_family"]) for row in evaluation} != set(
        EVALUATION_FAMILIES
    ):
        raise ValueError("evaluation_authority_roster")
    prediction_by_key = {(str(row["arm"]), str(row["unit_id"])): row for row in predictions}
    expected_keys = {(arm, str(row["unit_id"])) for arm in ARMS for row in authority_rows}
    if set(prediction_by_key) != expected_keys:
        raise ValueError("prediction_roster")
    direct_correct: dict[str, bool] = {}
    scored: list[JsonDict] = []
    for arm in ARMS:
        for row_order, truth in enumerate(evaluation):
            unit_id = str(truth["unit_id"])
            row = deepcopy(dict(prediction_by_key[(arm, unit_id)]))
            label = str(truth["support_label"])
            correct = row["prediction"] == label
            if arm == "direct_judgment":
                direct_correct[unit_id] = correct
            row.update(
                {
                    "row_order": row_order,
                    "base_id": str(truth["base_id"]),
                    "relation_family": str(truth["relation_family"]),
                    "variant": str(truth["variant"]),
                    "authority_label": label,
                    "metric": int(correct),
                    "correct": correct,
                    "false_accept": row["prediction"] == "supported" and label == "unsupported",
                    "false_reject": row["prediction"] == "unsupported" and label == "supported",
                    "harmful_flip": False,
                }
            )
            scored.append(row)
    for row in scored:
        if row["arm"] != "direct_judgment":
            row["harmful_flip"] = bool(
                direct_correct[row["unit_id"]] and not row["abstention"] and not row["correct"]
            )
    errors = evaluation_row_errors(scored)
    if errors:
        raise ValueError("evaluation_rows_invalid:" + ",".join(errors))
    return scored


def evaluation_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Detect missing denominators, duplicate units, and changed pair identity."""

    errors: list[str] = []
    if len(rows) != 128 * len(ARMS):
        errors.append("row_count")
    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}
    if any(len(values) != 128 for values in by_arm.values()):
        errors.append("arm_denominator")
    reference = by_arm[ARMS[0]]
    reference_ids = [row.get("unit_id") for row in reference]
    reference_pairs = [
        (
            row.get("unit_id"),
            row.get("base_id"),
            row.get("relation_family"),
            row.get("variant"),
        )
        for row in reference
    ]
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("baseline_unit_roster")
    for arm, values in by_arm.items():
        ids = [row.get("unit_id") for row in values]
        if len(ids) != len(set(ids)) or set(ids) != set(reference_ids):
            errors.append(f"{arm}:unit_roster")
        pairs = [
            (
                row.get("unit_id"),
                row.get("base_id"),
                row.get("relation_family"),
                row.get("variant"),
            )
            for row in values
        ]
        if pairs != reference_pairs:
            errors.append(f"{arm}:pair_roster")
        if [row.get("row_order") for row in values] != list(range(len(values))):
            errors.append(f"{arm}:row_order")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in reference:
        grouped[str(row.get("base_id"))].add(str(row.get("variant")))
    if len(grouped) != 32 or any(value != set(VARIANTS) for value in grouped.values()):
        errors.append("base_variant_roster")
    if reference and {row.get("relation_family") for row in reference} != set(EVALUATION_FAMILIES):
        errors.append("evaluation_families")
    return list(dict.fromkeys(errors))


def summarize_arms(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Report full-denominator quality, coverage, edit behavior, and latency."""

    errors = evaluation_row_errors(rows)
    if errors:
        raise ValueError("cannot_summarize_invalid_rows:" + ",".join(errors))
    summary: dict[str, JsonDict] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        denominator = len(selected)
        covered = [row for row in selected if not row["abstention"]]
        by_base: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in selected:
            by_base[str(row["base_id"])][str(row["variant"])] = row
        rename = 0
        rename_comparable = 0
        edit = 0
        edit_comparable = 0
        for variants in by_base.values():
            original = variants["original"]
            renamed = variants["bijective_entity_rename"]
            if not original["abstention"] and not renamed["abstention"]:
                rename_comparable += 1
                rename += int(original["prediction"] == renamed["prediction"])
            for name in ("relation_or_polarity_flip", "evidence_deletion"):
                changed = variants[name]
                if not original["abstention"] and not changed["abstention"]:
                    edit_comparable += 1
                    edit += int(original["prediction"] != changed["prediction"])
        correct = sum(bool(row["correct"]) for row in selected)
        parse_success = sum(bool(row["parse_success"]) for row in selected)
        summary[arm] = {
            "denominator": denominator,
            "accuracy_count": correct,
            "accuracy": correct / denominator,
            "parse_success_count": parse_success,
            "parse_failure_count": denominator - parse_success,
            "parse_rate": parse_success / denominator,
            "coverage_count": len(covered),
            "coverage": len(covered) / denominator,
            "abstention_count": denominator - len(covered),
            "abstention_rate": (denominator - len(covered)) / denominator,
            "conditional_accuracy": (
                sum(bool(row["correct"]) for row in covered) / len(covered) if covered else None
            ),
            "false_accept_count": sum(bool(row["false_accept"]) for row in selected),
            "false_accept_rate": sum(bool(row["false_accept"]) for row in selected) / denominator,
            "false_reject_count": sum(bool(row["false_reject"]) for row in selected),
            "false_reject_rate": sum(bool(row["false_reject"]) for row in selected) / denominator,
            "harmful_flip_count": sum(bool(row["harmful_flip"]) for row in selected),
            "harmful_flip_rate": sum(bool(row["harmful_flip"]) for row in selected) / denominator,
            "rename_consistency_count": rename,
            "rename_comparable_count": rename_comparable,
            "rename_pair_denominator": 32,
            "rename_consistency": rename / 32,
            "semantic_edit_sensitivity_count": edit,
            "semantic_edit_comparable_count": edit_comparable,
            "semantic_pair_denominator": 64,
            "semantic_edit_sensitivity": edit / 64,
            "cold_latency_s": sum(float(row["cold_latency_s"]) for row in selected),
            "cold_latency_mean_s": sum(float(row["cold_latency_s"]) for row in selected)
            / denominator,
            "amortized_latency_s": sum(float(row["amortized_latency_s"]) for row in selected),
            "amortized_latency_mean_s": sum(float(row["amortized_latency_s"]) for row in selected)
            / denominator,
        }
    return summary


def _interval(values: list[float], estimate: float) -> JsonDict:
    """Return one deterministic percentile interval from stored draws."""

    ordered = sorted(values)
    return {
        "estimate": estimate,
        "ci95_lower": ordered[int((len(ordered) - 1) * 0.025)],
        "ci95_upper": ordered[int((len(ordered) - 1) * 0.975)],
    }


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], seed: int, *, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Resample base cases within family while keeping four variants together."""

    errors = evaluation_row_errors(rows)
    if errors:
        raise ValueError("cannot_bootstrap_invalid_rows:" + ",".join(errors))
    if draws <= 0:
        raise ValueError("bootstrap_draws_positive")
    by_arm_base: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    base_family: dict[str, str] = {}
    for row in rows:
        arm, base_id = str(row["arm"]), str(row["base_id"])
        by_arm_base[arm][base_id].append(row)
        base_family[base_id] = str(row["relation_family"])
    family_bases = {
        family: sorted(base_id for base_id, value in base_family.items() if value == family)
        for family in EVALUATION_FAMILIES
    }
    if any(len(values) != 8 for values in family_bases.values()):
        raise ValueError("bootstrap_family_cluster_count")

    def stats(arm: str, base_id: str) -> tuple[int, int, int]:
        selected = by_arm_base[arm][base_id]
        return (
            sum(bool(row["correct"]) for row in selected),
            sum(bool(row["false_accept"]) for row in selected),
            sum(not row["abstention"] for row in selected),
        )

    fixed = {arm: {base_id: stats(arm, base_id) for base_id in base_family} for arm in ARMS}
    draws_by_arm: dict[str, dict[str, list[float]]] = {
        arm: {metric: [] for metric in ("accuracy", "false_accept_rate", "coverage")}
        for arm in ARMS[1:]
    }
    rng = random.Random(seed)
    for _ in range(draws):
        selected_bases = [
            rng.choice(family_bases[family]) for family in EVALUATION_FAMILIES for _ in range(8)
        ]
        direct = [fixed["direct_judgment"][base_id] for base_id in selected_bases]
        for arm in ARMS[1:]:
            treatment = [fixed[arm][base_id] for base_id in selected_bases]
            for index, metric in enumerate(("accuracy", "false_accept_rate", "coverage")):
                delta = (
                    sum(value[index] for value in treatment) - sum(value[index] for value in direct)
                ) / 128
                draws_by_arm[arm][metric].append(delta)
    metrics = summarize_arms(rows)
    interval_rows: list[JsonDict] = []
    for arm in ARMS[1:]:
        for metric in ("accuracy", "false_accept_rate", "coverage"):
            interval_rows.append(
                {
                    "comparison": (
                        "typed_vs_direct"
                        if arm == "typed_execution_unknown_abstention"
                        else f"{arm}_vs_direct"
                    ),
                    "arm": arm,
                    "baseline_arm": "direct_judgment",
                    "metric": metric,
                    **_interval(
                        draws_by_arm[arm][metric],
                        float(metrics[arm][metric]) - float(metrics["direct_judgment"][metric]),
                    ),
                    "method": "paired_percentile_cluster_bootstrap",
                    "draw_count": draws,
                    "random_seed": seed,
                    "cluster_count": 32,
                    "clusters_per_family": 8,
                    "variants_per_cluster": 4,
                    "stratified_by": "relation_family",
                    "independent_unit": "base_id",
                    "full_denominator": 128,
                }
            )
    return interval_rows


def _interval_lookup(
    rows: Sequence[Mapping[str, Any]], comparison: str, metric: str
) -> Mapping[str, Any]:
    """Select one unique preregistered paired interval."""

    matches = [
        row for row in rows if row.get("comparison") == comparison and row.get("metric") == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"paired_interval_missing:{comparison}:{metric}")
    return matches[0]


def classify_value(
    metrics: Mapping[str, Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    *,
    independent_audit_passed: bool,
) -> JsonDict:
    """Apply accuracy and efficiency as alternative preregistered branches."""

    typed = metrics["typed_execution_unknown_abstention"]
    direct = metrics["direct_judgment"]
    accuracy = _interval_lookup(intervals, "typed_vs_direct", "accuracy")
    false_accept = _interval_lookup(intervals, "typed_vs_direct", "false_accept_rate")
    coverage = _interval_lookup(intervals, "typed_vs_direct", "coverage")
    accuracy_clauses = {
        "paired_accuracy_ci95_lower_strictly_positive": accuracy["ci95_lower"] > 0,
        "false_accept_rate_not_increased": false_accept["estimate"] <= 0,
        "coverage_at_least_0_60": typed["coverage"] >= 0.60,
        "independent_semantic_audit_passed": independent_audit_passed,
    }
    accuracy_criterion = {
        "name": "accuracy_branch",
        "accuracy_delta": deepcopy(dict(accuracy)),
        "false_accept_rate_delta": deepcopy(dict(false_accept)),
        "typed_coverage": typed["coverage"],
        "clauses": accuracy_clauses,
        "passed": all(accuracy_clauses.values()),
    }
    typed_latency = float(typed["amortized_latency_s"])
    direct_latency = float(direct["amortized_latency_s"])
    speedup = direct_latency / typed_latency if typed_latency > 0 else None
    efficiency_clauses = {
        "accuracy_noninferiority_ci95_lower_at_least_minus_0_02": accuracy["ci95_lower"] >= -0.02,
        "equal_coverage": typed["coverage"] == direct["coverage"] and coverage["estimate"] == 0,
        "latency_speedup_at_least_2x_including_extraction": speedup is not None and speedup >= 2.0,
        "independent_semantic_audit_passed": independent_audit_passed,
    }
    efficiency_criterion = {
        "name": "efficiency_branch",
        "noninferiority_margin": 0.02,
        "accuracy_delta": deepcopy(dict(accuracy)),
        "coverage_delta": deepcopy(dict(coverage)),
        "direct_amortized_latency_s": direct_latency,
        "typed_amortized_latency_s_including_extraction": typed_latency,
        "measured_latency_speedup": speedup,
        "clauses": efficiency_clauses,
        "passed": all(efficiency_clauses.values()),
    }
    accepted = bool(
        independent_audit_passed
        and (accuracy_criterion["passed"] or efficiency_criterion["passed"])
    )
    if not independent_audit_passed:
        verdict_class = "disqualified"
        honest = "complete_disqualified_grounding_semantic_audit_failed"
    elif accepted:
        verdict_class = "circular_positive"
        honest = "complete_circular_positive_typed_grounding_pilot_value_gate_passed"
    else:
        verdict_class = "null"
        honest = "complete_null_typed_grounding_value_gate_not_met"
    return {
        "grounding_audit_complete_score": 1,
        "grounding_value_score": int(accepted),
        "acceptance_gate_value": int(accepted),
        "accuracy_criterion": accuracy_criterion,
        "efficiency_criterion": efficiency_criterion,
        "verifier_is_oracle": True,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
    }


def _gate(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str | None,
    field: str,
) -> JsonDict:
    """Record the exact expected and observed sides of one check."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failure into a fixed terminal summary."""

    if failure is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_preconditions_and_recomputations_pass",
            "observed_value": "all_preconditions_and_recomputations_pass",
            "passed": True,
        }
    return {
        "failed_check": failure["check"],
        "upstream": failure["upstream"],
        "field": failure["field"],
        "expected_value": deepcopy(failure["expected_value"]),
        "observed_value": deepcopy(failure["observed_value"]),
        "passed": False,
    }


def _unwrap(value: Any) -> Any:
    """Read a principle-wrapped field without treating its object as true."""

    if isinstance(value, Mapping) and {"principle", "value"} <= set(value):
        return value["value"]
    return value


def _is_quarantined(value: Mapping[str, Any]) -> bool:
    """Reject any explicit structured quarantine flag."""

    return bool(_unwrap(value.get("flagged_adversarial", False))) or bool(
        _unwrap(value.get("quarantined", False))
    )


def _manifest_hits(value: Any, wanted: frozenset[str]) -> set[str]:
    """Find excluded upstream experiment IDs in nested manifest records."""

    hits: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"experiment_id", "experiment_ids"}:
                candidates = child if isinstance(child, list) else [child]
                hits.update(str(item) for item in candidates if str(item) in wanted)
            hits.update(_manifest_hits(child, wanted))
    elif isinstance(value, list):
        for child in value:
            hits.update(_manifest_hits(child, wanted))
    return hits


def _resolve_paths(root: Path, overrides: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Resolve fixed sources while allowing isolated test replacements."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = Path(path)
    return paths


def _preconditions(
    root: Path,
    run_date: str,
    paths: Mapping[str, Path],
    output_paths: Sequence[Path],
    *,
    pinned_hashes: Mapping[str, str] | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Check bytes, gates, quarantine, tools, and destinations before scoring."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}

    def add(row: JsonDict) -> bool:
        _progress(1, "CHECK", str(row["check"]))
        checks.append(row)
        return bool(row["passed"])

    if not add(
        _gate(
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            upstream="experiment_7197",
            field="run_date",
        )
    ):
        return checks, hashes
    for name, default in SOURCE_PATHS.items():
        path = paths[name]
        readable = path.is_file() and os.access(path, os.R_OK)
        if not add(
            _gate(
                f"{name}_path",
                "readable_file",
                str(path) if readable else "missing_or_unreadable",
                readable,
                upstream=default.as_posix(),
                field="path",
            )
        ):
            return checks, hashes
        hashes[name] = sha256_file(path)
    expected_hashes = dict(pinned_hashes or PINNED_INPUT_HASHES)
    for name, expected in expected_hashes.items():
        observed = hashes.get(name)
        if not add(
            _gate(
                f"{name}_hash",
                expected,
                observed,
                observed == expected,
                upstream=SOURCE_PATHS[name].as_posix(),
                field="sha256",
            )
        ):
            return checks, hashes
    spec_text = paths["constraint_spec"].read_text(encoding="utf-8")
    has_requirement = "REQ-VERIFY-7197" in spec_text
    if not add(
        _gate(
            "constraint_spec_requirement",
            "REQ-VERIFY-7197",
            "REQ-VERIFY-7197" if has_requirement else "missing",
            has_requirement,
            upstream=SOURCE_PATHS["constraint_spec"].as_posix(),
            field="requirement",
        )
    ):
        return checks, hashes

    exp7195 = json.loads(paths["exp7195_artifact"].read_text(encoding="utf-8"))
    exp7196 = json.loads(paths["exp7196_artifact"].read_text(encoding="utf-8"))
    exclusion = yaml.safe_load(paths["exclusion_manifest"].read_text(encoding="utf-8"))
    raw_manifest = json.loads(paths["raw_manifest"].read_text(encoding="utf-8"))
    for name, artifact in (("exp7195", exp7195), ("exp7196", exp7196)):
        quarantined = _is_quarantined(artifact)
        if not add(
            _gate(
                f"{name}_structured_quarantine",
                False,
                quarantined,
                not quarantined,
                upstream=SOURCE_PATHS[f"{name}_artifact"].as_posix(),
                field="flagged_adversarial_or_quarantined",
            )
        ):
            return checks, hashes
    hits = sorted(_manifest_hits(exclusion, frozenset({"7195", "7196"})))
    if not add(
        _gate(
            "upstream_manifest_quarantine",
            [],
            hits,
            not hits,
            upstream=SOURCE_PATHS["exclusion_manifest"].as_posix(),
            field="excluded_upstream_ids",
        )
    ):
        return checks, hashes
    for name, artifact, expected in (
        ("exp7195", exp7195, EXP7195_EXPECTED_FIELDS),
        ("exp7196", exp7196, EXP7196_EXPECTED_FIELDS),
    ):
        observed = {field: artifact.get(field) for field in expected}
        if not add(
            _gate(
                f"{name}_terminal_gate_fields",
                expected,
                observed,
                observed == expected,
                upstream=SOURCE_PATHS[f"{name}_artifact"].as_posix(),
                field="terminal_gate_fields",
            )
        ):
            return checks, hashes
    old_values = sorted(
        {
            row.get("old_grounding_value_score")
            for row in exp7195.get("error_decomposition_rows", [])
            if isinstance(row, Mapping)
        }
    )
    known_observed = {
        "exp7182_grounding_value_score": old_values[0] if old_values == [0] else old_values,
        "exp7196_verdict_class": exp7196.get("verdict_class"),
        "promoted_as_value": False,
    }
    known_expected = {
        "exp7182_grounding_value_score": 0,
        "exp7196_verdict_class": "null",
        "promoted_as_value": False,
    }
    if not add(
        _gate(
            "known_failed_value_not_promoted",
            known_expected,
            known_observed,
            known_observed == known_expected,
            upstream="experiments_7182_7195_7196",
            field="scientific_value",
        )
    ):
        return checks, hashes
    raw_observed = {
        "status": raw_manifest.get("status"),
        "forbidden_input_open_count": raw_manifest.get("forbidden_input_open_count"),
        "worker_input_paths": raw_manifest.get("worker_input_paths"),
        "logical_rows": len(raw_manifest.get("raw_rows", [])),
    }
    raw_expected = {
        "status": "complete",
        "forbidden_input_open_count": 0,
        "worker_input_paths": ["results/experiment_7195_v634_typed_grounding_public.jsonl"],
        "logical_rows": 576,
    }
    if not add(
        _gate(
            "raw_call_manifest",
            raw_expected,
            raw_observed,
            raw_observed == raw_expected,
            upstream=SOURCE_PATHS["raw_manifest"].as_posix(),
            field="raw_call_provenance",
        )
    ):
        return checks, hashes
    python_observed = str(Path(sys.executable).resolve())
    python_expected = str((root / ".venv/bin/python").resolve())
    if not add(
        _gate(
            "python_executable",
            python_expected,
            python_observed,
            python_observed == python_expected and os.access(sys.executable, os.X_OK),
            upstream=".venv/bin/python",
            field="executable",
        )
    ):
        return checks, hashes
    for path in output_paths:
        writable = path.parent.is_dir() and os.access(path.parent, os.W_OK)
        if not add(
            _gate(
                f"output_directory:{path.name}",
                "existing_writable_directory",
                str(path.parent) if writable else "missing_or_unwritable",
                writable,
                upstream=str(path),
                field="parent",
            )
        ):
            return checks, hashes
    return checks, hashes


def base_artifact(root: Path, run_date: str) -> JsonDict:
    """Create a schema-complete running shell before fallible work."""

    del root
    return {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_evaluation_rows": 128,
            "completed_evaluation_rows": 0,
            "planned_arm_rows": 640,
            "completed_arm_rows": 0,
            "independent_base_cases": 32,
            "variants_per_base": 4,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": _gate_summary(None),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_grounding_value_audit",
        "grounding_audit_complete_score": 0,
        "grounding_value_score": 0,
        "arm_metrics": {},
        "paired_interval_rows": [],
        "oracle_distinctness_rationale": ORACLE_RATIONALE,
        "cold_audit_rows": [],
        "accuracy_criterion": {},
        "efficiency_criterion": {},
        "acceptance_gate_value": 0,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "scope_answer": "The 32-base pilot has not completed.",
        "independent_audit": {},
    }


def build_audit_request(
    root: Path,
    predictions: Sequence[Mapping[str, Any]],
    shuffle_map: Mapping[str, str],
    *,
    paths: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Build a label-free request for the fresh evaluator process."""

    resolved = _resolve_paths(Path(root).resolve(), paths)
    source_paths = {
        "public_view": resolved["public_view"],
        "capture_artifact": resolved["exp7196_artifact"],
        "authority_sidecar": resolved["authority_sidecar"],
    }
    candidate = [
        {
            "unit_id": row["unit_id"],
            "prediction": row["prediction"],
            "parse_status": row["parse_status"],
        }
        for row in predictions
        if row["arm"] == "typed_execution_unknown_abstention"
    ]
    candidate_checksum = sha256_text(canonical_json(candidate))
    return {
        "schema": "carnot.exp7197.independent_audit_request.v1",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "expected_input_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "candidate_predictions": candidate,
        "candidate_prediction_checksum": candidate_checksum,
        "shuffle_mapping": [
            {"unit_id": unit_id, "donor_unit_id": donor_id}
            for unit_id, donor_id in sorted(shuffle_map.items())
        ],
    }


def _run_auditor(root: Path, request_path: Path, output_path: Path) -> JsonDict:  # pragma: no cover
    """Stream the fresh evaluator with a deadline and external heartbeat."""

    command = [
        str(root / ".venv/bin/python"),
        "-u",
        "-m",
        "carnot.experiment_7197_v634_grounding_value_independent_audit",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(root / "python")
    _progress(4, "START", "fresh evaluator subprocess")
    started = time.monotonic()
    last_heartbeat = started
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    while process.poll() is None:
        ready, _, _ = select.select([process.stdout], [], [], 1.0)
        if ready:
            line = process.stdout.readline()
            if line:
                print(line, end="", flush=True)
        now = time.monotonic()
        if now - last_heartbeat >= 60:
            print(
                f"HEARTBEAT elapsed_s={now - started:.1f} operation=fresh_evaluator",
                flush=True,
            )
            last_heartbeat = now
        if now - started > AUDITOR_TIMEOUT_S:
            process.terminate()
            process.wait(timeout=10)
            raise TimeoutError("independent_auditor_deadline")
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"independent_auditor_exit:{process.returncode}")
    _progress(4, "END", f"fresh evaluator returncode={process.returncode}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _scope_answer(metrics: Mapping[str, Mapping[str, Any]], verdict: Mapping[str, Any]) -> str:
    """State the narrow result without a broad model-performance claim."""

    typed = metrics["typed_execution_unknown_abstention"]
    direct = metrics["direct_judgment"]
    if verdict["grounding_value_score"]:
        opening = "The typed policy passed a preregistered pilot value branch."
    else:
        opening = "The typed policy did not pass either preregistered pilot value branch."
    return (
        f"{opening} Correct/128 was {typed['accuracy_count']}/128 versus "
        f"{direct['accuracy_count']}/128 for direct judgment. Typed coverage was "
        f"{typed['coverage']:.6f}. This 32-base pilot does not establish broad model "
        "performance or a learned-verifier moat."
    )


def build_artifact(
    root: Path,
    run_date: str,
    *,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
    audit_request_path: Path | None = None,
    audit_result_path: Path | None = None,
    source_paths: Mapping[str, Path] | None = None,
    pinned_hashes: Mapping[str, str] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Gate sources, score cached calls, run the child audit, and finish atomically."""

    started = time.monotonic()
    root = Path(root).resolve()
    result = Path(result_path) if result_path is not None else root / RESULT_PATH
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_PATH
    audit_request_path = (
        Path(audit_request_path) if audit_request_path is not None else root / AUDIT_REQUEST_PATH
    )
    audit_result_path = (
        Path(audit_result_path) if audit_result_path is not None else root / AUDIT_RESULT_PATH
    )
    # The supplied date is a checked prerequisite. The artifact itself always
    # records the mandated execution date, including when that check blocks.
    artifact = base_artifact(root, RUN_DATE)
    _progress(0, "START", "running checkpoint before any prerequisite check")
    atomic_write_json(checkpoint, artifact, allow_override=False, sort_keys=True)
    _progress(0, "END", f"checkpoint={checkpoint}")

    _progress(1, "START", "source bytes gates quarantine tools and output paths")
    resolved = _resolve_paths(root, source_paths)
    checks, hashes = _preconditions(
        root,
        run_date,
        resolved,
        (result, checkpoint, audit_request_path, audit_result_path),
        pinned_hashes=pinned_hashes,
    )
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = hashes
    failure = next((row for row in checks if not row["passed"]), None)
    if failure is not None:
        artifact.update(
            {
                "status": "blocked",
                "duration_s": (
                    float(duration_s) if duration_s is not None else time.monotonic() - started
                ),
                "gate_check_summary": _gate_summary(failure),
                "verdict_class": "blocked",
                "honest_verdict": "blocked_grounding_value_audit_external_precondition",
                "scope_answer": "The audit did not run because an external prerequisite failed.",
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        _progress(1, "WRITE_START", f"terminal blocked artifact={result}")
        atomic_write_json(result, artifact, allow_override=False, sort_keys=True)
        _progress(1, "WRITE_END", f"blocked check={failure['check']}")
        return artifact
    _progress(1, "END", f"checks={len(checks)} all_passed=true")

    _progress(2, "START", "raw parsing and label-free policy benchmark")
    public_rows = _read_jsonl(resolved["public_view"])
    capture = json.loads(resolved["exp7196_artifact"].read_text(encoding="utf-8"))
    features = build_candidate_features(public_rows, capture["completion_rows"])
    # This projection uses family and variant identity only. It does not read
    # either outcome field from the evaluator sidecar.
    authority_rows = _read_jsonl(resolved["authority_sidecar"])
    shuffle_map = build_shuffle_map(authority_rows)
    _progress(2, "BENCHMARK_START", "five cached policy arms")
    predictions = evaluate_arms(features, shuffle_map)
    blind_checksum = prediction_checksum(predictions)
    _progress(
        2,
        "BENCHMARK_END",
        f"public_units={len(features)} arm_rows={len(predictions)} checksum={blind_checksum}",
    )

    _progress(3, "START", "held-out scoring metrics and 10000-draw bootstrap")
    rows = score_evaluation(predictions, authority_rows)
    arm_metrics = summarize_arms(rows)
    _progress(3, "BENCHMARK_START", f"paired bootstrap draws={BOOTSTRAP_DRAWS}")
    intervals = paired_cluster_bootstrap(rows, BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
    _progress(3, "BENCHMARK_END", f"interval_rows={len(intervals)}")

    request = build_audit_request(root, predictions, shuffle_map, paths=source_paths)
    _progress(4, "WRITE_START", f"audit request={audit_request_path}")
    atomic_write_json(audit_request_path, request, allow_override=False, sort_keys=True)
    _progress(4, "WRITE_END", "audit request durable")
    audit = _run_auditor(root, audit_request_path, audit_result_path)
    audit_passed = bool(audit.get("independent_semantic_audit_passed"))
    verdict = classify_value(arm_metrics, intervals, independent_audit_passed=audit_passed)
    audit_receipt = {
        key: deepcopy(value) for key, value in audit.items() if key != "cold_audit_rows"
    }
    audit_receipt.update(
        {
            "audit_request_sha256": sha256_file(audit_request_path),
            "audit_result_sha256": sha256_file(audit_result_path),
        }
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": (
                float(duration_s) if duration_s is not None else time.monotonic() - started
            ),
            "rows": rows,
            "sample_size_budget": {
                "planned_evaluation_rows": 128,
                "completed_evaluation_rows": 128,
                "planned_arm_rows": 640,
                "completed_arm_rows": 640,
                "independent_base_cases": 32,
                "variants_per_base": 4,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "exclusions": [],
            },
            "gate_check_summary": _gate_summary(None),
            "arm_metrics": arm_metrics,
            "paired_interval_rows": intervals,
            "cold_audit_rows": audit["cold_audit_rows"],
            "independent_audit": audit_receipt,
            **verdict,
        }
    )
    artifact["scope_answer"] = _scope_answer(arm_metrics, verdict)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(5, "START", "cold validation and final atomic write")
    errors = validate_artifact(artifact, root=root, check_source_hashes=True)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(5, "WRITE_START", f"terminal artifact={result}")
    atomic_write_json(result, artifact, allow_override=False, sort_keys=True)
    _progress(5, "WRITE_END", f"verdict={artifact['verdict_class']}")
    return artifact


def validate_artifact(
    value: Mapping[str, Any] | str | Path,
    *,
    root: Path | None = None,
    check_source_hashes: bool = True,
) -> list[str]:
    """Cold-check schema, rows, metrics, audit, criteria, sources, and verdict."""

    if isinstance(value, (str, Path)):
        try:
            artifact = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
    else:
        artifact = deepcopy(dict(value))
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("artifact_fields")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_invocation")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("status") == "blocked":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict")
        if artifact.get("grounding_audit_complete_score") != 0:
            errors.append("blocked_completion")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary")
        return list(dict.fromkeys(errors))
    if artifact.get("status") != "complete":
        errors.append("terminal_status")
        return list(dict.fromkeys(errors))
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("inference_substrate_class")
    row_errors = evaluation_row_errors(artifact.get("rows", []))
    errors.extend(row_errors)
    if not row_errors:
        metrics = summarize_arms(artifact["rows"])
        if artifact.get("arm_metrics") != metrics:
            errors.append("arm_metrics")
        intervals = paired_cluster_bootstrap(
            artifact["rows"], BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS
        )
        if artifact.get("paired_interval_rows") != intervals:
            errors.append("paired_interval_rows")
        audit_passed = (
            artifact.get("independent_audit", {}).get("independent_semantic_audit_passed") is True
        )
        verdict = classify_value(metrics, intervals, independent_audit_passed=audit_passed)
        for field, expected in verdict.items():
            if artifact.get(field) != expected:
                errors.append("terminal_classification")
                break
        if artifact.get("scope_answer") != _scope_answer(metrics, verdict):
            errors.append("scope_answer")
    audit = artifact.get("independent_audit", {})
    if len(artifact.get("cold_audit_rows", [])) != 128:
        errors.append("cold_audit_rows")
    if audit.get("candidate_module_imported") is not False:
        errors.append("candidate_module_imported")
    if audit.get("labels_opened_after_prediction") is not True:
        errors.append("label_open_order")
    if audit.get("label_mutation_prediction_invariant") is not True:
        errors.append("label_mutation_prediction_invariant")
    if audit.get("decision_disagreement_count") != 0:
        errors.append("audit_decision_disagreement")
    if audit.get("controls_complete") is not True:
        errors.append("audit_controls")
    if artifact.get("oracle_distinctness_rationale") != ORACLE_RATIONALE:
        errors.append("oracle_distinctness_rationale")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("grounding_audit_complete_score") != 1:
        errors.append("grounding_audit_complete_score")
    if any(row.get("passed") is not True for row in artifact.get("preconditions_checked", [])):
        errors.append("preconditions_checked")
    if check_source_hashes:
        repository = Path(root).resolve() if root is not None else find_repo_root()
        paths = _resolve_paths(repository)
        observed = {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file() and os.access(path, os.R_OK)
        }
        if artifact.get("source_artifact_hashes") != observed:
            errors.append("source_artifact_hashes")
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Build the fixed-date audit or cold-validate one existing artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    root = find_repo_root()
    if args.validate is not None:
        _progress(0, "START", f"validate artifact={args.validate}")
        errors = validate_artifact(args.validate, root=root, check_source_hashes=True)
        _progress(0, "END", f"validation_errors={len(errors)}")
        if errors:
            print(json.dumps({"status": "invalid", "errors": errors}, indent=2), flush=True)
            return 1
        print(json.dumps({"status": "valid", "artifact": str(args.validate)}, indent=2), flush=True)
        return 0
    artifact = build_artifact(root, args.date)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "grounding_audit_complete_score": artifact["grounding_audit_complete_score"],
                "grounding_value_score": artifact["grounding_value_score"],
                "result": str(root / RESULT_PATH),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if artifact["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
