"""Audit frozen grounding energy against a separate exact authority.

The candidate path reparses saved Qwen output. It fits only one threshold on
the two calibration families, then evaluates four unseen relation families.
A fresh process implements the formula again and runs causal controls.

Spec refs: REQ-VERIFY-7182 and SCENARIO-VERIFY-7182-*.
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

from carnot.experiment_7180_v633_symbolic_edit_fixture import (
    compute_candidate_energy as frozen_candidate_energy,
)
from carnot.experiment_7181_v633_qwen38_symbolic_traces import parse_structured_output
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260910"
RANDOM_SEED = 7_182_202_609_10
BOOTSTRAP_SEED = 718_204
BOOTSTRAP_DRAWS = 10_000
ARMS = (
    "baseline_direct",
    "energy_from_extracted_tuples",
    "lexical_overlap",
    "syntax_only",
    "shuffled_evidence",
)
ENERGY_TERMS = (
    "tuple_alignment",
    "polarity",
    "quantity_unit_agreement",
    "literal_span_validity",
    "missing_required_fields",
)
ENERGY_WEIGHTS = {
    "tuple_alignment": 3,
    "polarity": 2,
    "quantity_unit_agreement": 2,
    "literal_span_validity": 2,
    "missing_required_fields": 1,
}
CALIBRATION_FAMILIES = ("precedes", "follows")
EVALUATION_FAMILIES = (
    "starts before",
    "ends before",
    "is separated from",
    "occurs before",
)
VARIANTS = (
    "original",
    "bijective_entity_rename",
    "relation_or_polarity_flip",
    "evidence_deletion",
)
RESULT_PATH = Path("results/experiment_7182_v633_grounding_energy_audit.json")
CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7182_v633_grounding_energy_audit_running.json"
)
AUDIT_REQUEST_PATH = Path(
    "results/checkpoints/experiment_7182_v633_grounding_energy_audit_request.json"
)
AUDIT_RESULT_PATH = Path(
    "results/checkpoints/experiment_7182_v633_grounding_energy_audit_response.json"
)
AUDITOR_TIMEOUT_S = 300.0
STUDY_QUESTION = (
    "Does Qwen3.8's frozen structural grounding energy improve held-out decisions "
    "against independent source authority?"
)
INFERENCE_SUBSTRATE = "deterministic_cpu_grounding_energy_replay_and_fresh_process_audit"

SOURCE_PATHS = {
    "trace_artifact": Path("results/experiment_7181_v633_qwen38_symbolic_traces.json"),
    "fixture_artifact": Path("results/experiment_7180_v633_symbolic_edit_fixture.json"),
    "generation_view": Path(
        "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
    ),
    "authority_sidecar": Path("results/experiment_7180_v633_symbolic_edit_fixture_authority.jsonl"),
    "raw_manifest": Path(
        "results/raw/experiment_7181_v633_qwen38_symbolic_traces/raw_manifest.json"
    ),
    "constraint_spec": Path("openspec/capabilities/constraint-verification/spec.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "fixture_module": Path("python/carnot/experiment_7180_v633_symbolic_edit_fixture.py"),
    "trace_module": Path("python/carnot/experiment_7181_v633_qwen38_symbolic_traces.py"),
    "module": Path("python/carnot/experiment_7182_v633_grounding_energy_audit.py"),
    "auditor_module": Path(
        "python/carnot/experiment_7182_v633_grounding_energy_independent_audit.py"
    ),
    "entrypoint": Path("scripts/experiments/experiment_7182_v633_grounding_energy_audit.py"),
    "auditor_entrypoint": Path(
        "scripts/experiments/experiment_7182_v633_grounding_energy_independent_audit.py"
    ),
    "focused_tests": Path("tests/python/test_experiment_7182_v633_grounding_energy_audit.py"),
}
PINNED_HASHES = {
    "trace_artifact": "sha256:0de07860999222d2adf2312f060297114dc02e7d89c16ecb94b73bfeeb15d791",
    "fixture_artifact": "sha256:a76492f304e684d75ea25037261c90bd0367fa54819bb867b38fe2d96a00398a",
    "generation_view": "sha256:6927314707d5075ef7b48e469db14f8a0c67d54caf73de05a9c8c9d082cd628d",
    "authority_sidecar": "sha256:68b3c681d0baf2ccda5b70c6213249b21287507629ae9c2df5379b44fa8df078",
    "raw_manifest": "sha256:183d7738c5a6ca64d8e9f31a28bc8792fe2822a07e4f507cad32cfff8ef1d299",
}
EXP7180_EXPECTED_FIELDS = {
    "status": "complete",
    "run_date": RUN_DATE,
    "inference_substrate": "exact_source_fixture_construction",
    "inference_substrate_class": "cpu_exact_solver_or_simulator",
    "execution_venue": "host",
    "fixture_ready_score": 1,
    "random_seed": 7_180_202_609_10,
    "reproducibility_checksum": (
        "sha256:72d4a6c940dc52825f0eac30cb623df727bd7f45e362dc14d8a611fa5536bec5"
    ),
    "verifier_is_oracle": False,
    "verdict_class": "positive",
    "honest_verdict": "complete_positive_symbolic_edit_fixture_ready_no_live_verifier_result",
}
EXP7181_EXPECTED_FIELDS = {
    "status": "complete",
    "run_date": RUN_DATE,
    "inference_substrate": "native_llama_cpp_qwen38_symbolic_trace_generation",
    "inference_substrate_class": "model_full_generation",
    "execution_venue": "host",
    "trace_capture_complete_score": 1,
    "random_seed": 7_181_202_609_10,
    "reproducibility_checksum": (
        "sha256:e2f6f6859b7500da777ca7240407caa41bd529dc97eaf9fb44af805f2f277195"
    ),
    "verifier_is_oracle": False,
    "verdict_class": "positive",
    "honest_verdict": "complete_positive_transport_capture_no_correctness_claim",
}
FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "authority",
        "authority_label",
        "expected_answer",
        "expected_response",
        "split",
        "support_label",
        "truth",
        "variant",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "grounding_measurement_complete_score",
    "grounding_value_score",
    "paired_metrics",
    "feature_lineage_rows",
    "intervention_rows",
    "independent_audit_rows",
    "study_question",
    "scope_answer",
    "arm_metrics",
    "frozen_threshold_contract",
    "audit_receipt",
)
FIELD_PRINCIPLES = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use no_model_load when the declared work runs; use blocked_no_run only before any qualifying work.",
    "grounding_measurement_complete_score": "Completion gates should preserve honest negative results.",
    "grounding_value_score": "A stricter value gate prevents fixture success from becoming a verifier claim.",
    "paired_metrics": "Cluster intervals preserve dependence among symbolic variants.",
    "feature_lineage_rows": "Every scoring feature must trace to candidate text or evidence.",
    "intervention_rows": "Swaps and deletions test causal source use.",
    "independent_audit_rows": "Independent decisions reveal implementation disagreement.",
    "study_question": "The fixed question keeps the pilot within its declared scope.",
    "scope_answer": "The answer reports the measured outcome without a broad benchmark claim.",
    "arm_metrics": "Full-denominator metrics keep failures and negative outcomes visible.",
    "frozen_threshold_contract": "Calibration receipts prove evaluation labels could not tune the threshold.",
    "audit_receipt": "Process identity and disagreement counts prove the independent replay ran.",
}


def canonical_json(value: Any) -> str:
    """Return one compact JSON spelling for stable hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text and retain the algorithm name."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without changing line endings or spaces."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding runtime-only process values."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    if isinstance(payload.get("audit_receipt"), dict):
        payload["audit_receipt"]["process_id"] = None
    return sha256_text(canonical_json(payload))


def _progress(phase: int, event: str, detail: str) -> None:  # pragma: no cover
    """Print one flushed phase boundary so long tasks remain observable."""

    print(f"PHASE {phase} {event} {detail}", flush=True)


def _prompt_parts(prompt: str) -> tuple[str, str]:
    """Return exact evidence and claim substrings from the frozen prompt."""

    try:
        remainder = prompt.split("Evidence:\n", 1)[1]
        evidence, remainder = remainder.split("\n\nClaim:\n", 1)
        claim = remainder.split("\n\nReturn one JSON object", 1)[0]
    except IndexError as error:
        raise ValueError("prompt_shape_invalid") from error
    return evidence, claim


def _contains_forbidden_key(value: Any) -> bool:
    """Search nested model receipts so hidden authority fields cannot pass."""

    if isinstance(value, Mapping):
        if FORBIDDEN_CANDIDATE_KEYS & set(value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def build_candidate_features(
    trace_rows: Sequence[Mapping[str, Any]], generation_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reparse raw model bytes and never trust producer tuple projections."""

    if len(trace_rows) != 192 or len(generation_rows) != 192:
        raise ValueError("candidate_source_row_count")
    features: list[JsonDict] = []
    for row_order, (trace, generation) in enumerate(zip(trace_rows, generation_rows, strict=True)):
        if _contains_forbidden_key(trace) or _contains_forbidden_key(generation):
            raise ValueError(f"authority_field_exposed:{row_order}")
        if set(generation) != {"unit_id", "text", "response_schema"}:
            raise ValueError(f"generation_shape:{row_order}")
        if trace.get("unit_id") != generation.get("unit_id"):
            raise ValueError(f"candidate_unit_order:{row_order}")
        prompt = str(generation["text"])
        if trace.get("prompt") != prompt:
            raise ValueError(f"candidate_prompt:{row_order}")
        raw_output = str(trace.get("raw_output", ""))
        raw_hash = sha256_text(raw_output)
        if trace.get("raw_output_sha256") != raw_hash:
            raise ValueError(f"raw_output_hash_mismatch:{row_order}")
        source_text, claim_text = _prompt_parts(prompt)
        reparsed = parse_structured_output(raw_output, prompt)
        parsed = reparsed["parsed"]
        terms = (
            frozen_candidate_energy(parsed, source_text) if isinstance(parsed, Mapping) else None
        )
        features.append(
            {
                "row_order": row_order,
                "unit_id": str(trace["unit_id"]),
                "parse_status": reparsed["parse_status"],
                "parse_error": reparsed["parse_error"],
                "direct_decision": (
                    parsed.get("direct_decision") if isinstance(parsed, Mapping) else None
                ),
                "parsed_response": deepcopy(parsed),
                "source_text": source_text,
                "claim_text": claim_text,
                "raw_output_sha256": raw_hash,
                "source_text_sha256": sha256_text(source_text),
                "claim_text_sha256": sha256_text(claim_text),
                "prompt_sha256": sha256_text(prompt),
                "energy_terms": deepcopy(terms),
            }
        )
    return features


def _score_predictions(
    features: Sequence[Mapping[str, Any]], labels: Mapping[str, str], threshold: int
) -> JsonDict:
    """Score one calibration threshold with abstentions in the denominator."""

    accuracy = 0
    false_accepts = 0
    false_rejects = 0
    for feature in features:
        unit_id = str(feature["unit_id"])
        if unit_id not in labels:
            continue
        prediction = (
            "abstain"
            if feature["parse_status"] != "valid"
            else "supported"
            if feature["energy_terms"]["total"] <= threshold
            else "unsupported"
        )
        truth = labels[unit_id]
        accuracy += int(prediction == truth)
        false_accepts += int(prediction == "supported" and truth == "unsupported")
        false_rejects += int(prediction == "unsupported" and truth == "supported")
    denominator = len(labels)
    return {
        "threshold": threshold,
        "denominator": denominator,
        "accuracy_count": accuracy,
        "accuracy": accuracy / denominator,
        "false_accept_count": false_accepts,
        "false_reject_count": false_rejects,
    }


def select_threshold(
    features: Sequence[Mapping[str, Any]], calibration_authority: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Fit only on the two complete calibration families and freeze the result."""

    if (
        len(calibration_authority) != 64
        or {row.get("split") for row in calibration_authority} != {"calibration"}
        or {row.get("relation_family") for row in calibration_authority}
        != set(CALIBRATION_FAMILIES)
    ):
        raise ValueError("calibration_authority_only")
    feature_by_id = {str(row["unit_id"]): row for row in features}
    labels: dict[str, str] = {}
    for row in calibration_authority:
        unit_id = str(row["unit_id"])
        if unit_id not in feature_by_id:
            raise ValueError("calibration_feature_missing")
        labels[unit_id] = str(row["expected_response"]["direct_decision"])
    candidate_thresholds = [-1] + sorted(
        {
            int(feature_by_id[unit_id]["energy_terms"]["total"])
            for unit_id in labels
            if feature_by_id[unit_id]["energy_terms"] is not None
        }
    )
    table = [_score_predictions(features, labels, threshold) for threshold in candidate_thresholds]
    selected = min(
        table,
        key=lambda row: (
            -row["accuracy_count"],
            row["false_accept_count"],
            row["false_reject_count"],
            row["threshold"],
        ),
    )
    return {
        "fit_split": "calibration",
        "fit_relation_families": list(CALIBRATION_FAMILIES),
        "fit_unit_ids": [str(row["unit_id"]) for row in calibration_authority],
        "fit_denominator": 64,
        "energy_weights": deepcopy(ENERGY_WEIGHTS),
        "candidate_thresholds": candidate_thresholds,
        "selection_rule": (
            "maximize_full_denominator_accuracy_then_minimize_false_accepts_"
            "then_false_rejects_then_threshold"
        ),
        "tie_rule": "energy_equal_threshold_is_supported",
        "threshold": selected["threshold"],
        "calibration_results": table,
        "evaluation_truth_accessed": False,
        "frozen_before_evaluation": True,
    }


def _lexical_score(response: Mapping[str, Any]) -> float:
    """Measure exact token-set overlap between generated claim and evidence tuples."""

    claim = response.get("claim_tuple")
    evidence = response.get("evidence_tuple")
    if not isinstance(claim, Mapping) or not isinstance(evidence, Mapping):
        return 0.0
    token_pattern = re.compile(r"[a-z0-9]+")
    claim_tokens = set(token_pattern.findall(" ".join(map(str, claim.values())).lower()))
    evidence_tokens = set(token_pattern.findall(" ".join(map(str, evidence.values())).lower()))
    union = claim_tokens | evidence_tokens
    return len(claim_tokens & evidence_tokens) / len(union) if union else 0.0


def _energy_prediction(total: int, threshold: int) -> str:
    """Apply the preregistered tie rule to one integer energy."""

    return "supported" if total <= threshold else "unsupported"


def _shuffled_score(
    target: Mapping[str, Any], donor: Mapping[str, Any]
) -> tuple[str, JsonDict | None, str | None]:
    """Use the frozen donor evidence without filling failed donor parses."""

    if target["parse_status"] != "valid":
        return "abstain", None, str(target["parse_error"])
    if donor["parse_status"] != "valid":
        return "abstain", None, f"donor_parse_failed:{donor['parse_error']}"
    response = deepcopy(target["parsed_response"])
    donor_response = donor["parsed_response"]
    response["evidence_tuple"] = deepcopy(donor_response["evidence_tuple"])
    response["source_start"] = donor_response["source_start"]
    response["source_end"] = donor_response["source_end"]
    terms = frozen_candidate_energy(response, str(donor["source_text"]))
    return "scored", terms, None


def evaluate_arms(
    features: Sequence[Mapping[str, Any]],
    evaluation_authority: Sequence[Mapping[str, Any]],
    frozen_threshold: Mapping[str, Any],
    score_contract: Mapping[str, Any],
) -> list[JsonDict]:
    """Apply all five arms to one ordered 128-unit held-out roster."""

    if (
        len(evaluation_authority) != 128
        or {row.get("split") for row in evaluation_authority} != {"evaluation"}
        or {row.get("relation_family") for row in evaluation_authority} != set(EVALUATION_FAMILIES)
    ):
        raise ValueError("evaluation_authority_only")
    if frozen_threshold.get("evaluation_truth_accessed") is not False:
        raise ValueError("threshold_not_blind")
    feature_by_id = {str(row["unit_id"]): row for row in features}
    evaluation_ids = [str(row["unit_id"]) for row in evaluation_authority]
    if any(unit_id not in feature_by_id for unit_id in evaluation_ids):
        raise ValueError("evaluation_feature_missing")
    controls = score_contract.get("seeded_shuffle_controls", [])
    if not controls:
        raise ValueError("shuffle_control_missing")
    shuffle_map = {str(row["unit_id"]): str(row["donor_unit_id"]) for row in controls[0]["mapping"]}
    if any(unit_id not in shuffle_map for unit_id in evaluation_ids):
        raise ValueError("shuffle_control_roster")
    threshold = int(frozen_threshold["threshold"])
    rows: list[JsonDict] = []
    baseline_correct: dict[str, bool] = {}
    for arm in ARMS:
        for row_order, truth_row in enumerate(evaluation_authority):
            unit_id = str(truth_row["unit_id"])
            feature = feature_by_id[unit_id]
            truth = str(truth_row["expected_response"]["direct_decision"])
            prediction = "abstain"
            score: int | float | None = None
            energy_terms: JsonDict | None = None
            error = str(feature["parse_error"]) if feature["parse_error"] else None
            donor_id: str | None = None
            lineage_source_hash = feature["source_text_sha256"]
            if feature["parse_status"] == "valid":
                response = feature["parsed_response"]
                if arm == "baseline_direct":
                    prediction = str(response["direct_decision"])
                    score = {"unsupported": 0, "abstain": 0.5, "supported": 1}[prediction]
                elif arm == "energy_from_extracted_tuples":
                    energy_terms = deepcopy(feature["energy_terms"])
                    score = energy_terms["total"]
                    prediction = _energy_prediction(int(score), threshold)
                elif arm == "lexical_overlap":
                    score = _lexical_score(response)
                    prediction = "supported" if score == 1.0 else "unsupported"
                elif arm == "syntax_only":
                    score = 1
                    prediction = "supported"
                else:
                    donor_id = shuffle_map[unit_id]
                    donor = feature_by_id[donor_id]
                    state, energy_terms, error = _shuffled_score(feature, donor)
                    if state == "scored":
                        score = energy_terms["total"]
                        prediction = _energy_prediction(int(score), threshold)
                        lineage_source_hash = donor["source_text_sha256"]
            correct = prediction == truth
            if arm == "baseline_direct":
                baseline_correct[unit_id] = correct
            rows.append(
                {
                    "row_order": row_order,
                    "arm": arm,
                    "unit_id": unit_id,
                    "base_id": str(truth_row["base_id"]),
                    "relation_family": str(truth_row["relation_family"]),
                    "variant": str(truth_row["variant"]),
                    "authority_label": truth,
                    "parse_status": str(feature["parse_status"]),
                    "prediction": prediction,
                    "score": score,
                    "energy_terms": energy_terms,
                    "error": error,
                    "abstention": prediction == "abstain",
                    "correct": correct,
                    "false_accept": prediction == "supported" and truth == "unsupported",
                    "false_reject": prediction == "unsupported" and truth == "supported",
                    "harmful_flip": False,
                    "donor_unit_id": donor_id,
                    "feature_lineage": {
                        "raw_output_sha256": feature["raw_output_sha256"],
                        "source_text_sha256": lineage_source_hash,
                        "authority_used_for_features": False,
                    },
                }
            )
    for row in rows:
        if row["arm"] != "baseline_direct":
            row["harmful_flip"] = bool(baseline_correct[row["unit_id"]] and not row["correct"])
    errors = evaluation_row_errors(rows)
    if errors:
        raise ValueError("evaluation_rows_invalid:" + ",".join(errors))
    return rows


def evaluation_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Detect lost denominators, duplicate units, and changed pair identity."""

    errors: list[str] = []
    if len(rows) != 128 * len(ARMS):
        errors.append("row_count")
    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}
    if any(len(arm_rows) != 128 for arm_rows in by_arm.values()):
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
    for arm, arm_rows in by_arm.items():
        arm_ids = [row.get("unit_id") for row in arm_rows]
        if len(arm_ids) != len(set(arm_ids)) or set(arm_ids) != set(reference_ids):
            errors.append(f"{arm}:unit_roster")
        arm_pairs = [
            (
                row.get("unit_id"),
                row.get("base_id"),
                row.get("relation_family"),
                row.get("variant"),
            )
            for row in arm_rows
        ]
        if arm_pairs != reference_pairs:
            errors.append(f"{arm}:ordered_pair_ids")
        if [row.get("row_order") for row in arm_rows] != list(range(len(arm_rows))):
            errors.append(f"{arm}:row_order")
    if reference:
        if {row.get("relation_family") for row in reference} != set(EVALUATION_FAMILIES):
            errors.append("evaluation_families")
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in reference:
            grouped[str(row.get("base_id"))].add(str(row.get("variant")))
        if len(grouped) != 32 or any(variants != set(VARIANTS) for variants in grouped.values()):
            errors.append("base_variant_roster")
    return list(dict.fromkeys(errors))


def summarize_arms(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Report full-denominator errors and paired metamorphic behavior."""

    errors = evaluation_row_errors(rows)
    if errors:
        raise ValueError("cannot_summarize_invalid_rows:" + ",".join(errors))
    summary: dict[str, JsonDict] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        denominator = len(selected)
        by_base: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in selected:
            by_base[str(row["base_id"])][str(row["variant"])] = row
        rename_count = 0
        semantic_count = 0
        for variants in by_base.values():
            original = variants["original"]
            renamed = variants["bijective_entity_rename"]
            rename_count += int(
                not original["abstention"]
                and not renamed["abstention"]
                and original["prediction"] == renamed["prediction"]
            )
            for variant in ("relation_or_polarity_flip", "evidence_deletion"):
                edited = variants[variant]
                semantic_count += int(
                    not original["abstention"]
                    and not edited["abstention"]
                    and original["prediction"] != edited["prediction"]
                )
        parse_success = sum(row["parse_status"] == "valid" for row in selected)
        coverage_count = sum(not row["abstention"] for row in selected)
        false_accepts = sum(bool(row["false_accept"]) for row in selected)
        false_rejects = sum(bool(row["false_reject"]) for row in selected)
        correct = sum(bool(row["correct"]) for row in selected)
        harmful = sum(bool(row["harmful_flip"]) for row in selected)
        summary[arm] = {
            "denominator": denominator,
            "parse_success_count": parse_success,
            "parse_failure_count": denominator - parse_success,
            "parse_rate": parse_success / denominator,
            "coverage_count": coverage_count,
            "coverage": coverage_count / denominator,
            "false_accept_count": false_accepts,
            "false_accept_rate": false_accepts / denominator,
            "false_reject_count": false_rejects,
            "false_reject_rate": false_rejects / denominator,
            "accuracy_count": correct,
            "accuracy": correct / denominator,
            "harmful_flip_count": harmful,
            "harmful_flip_rate": harmful / denominator,
            "rename_invariance_count": rename_count,
            "rename_pair_denominator": 32,
            "rename_invariance": rename_count / 32,
            "semantic_edit_sensitivity_count": semantic_count,
            "semantic_pair_denominator": 64,
            "semantic_edit_sensitivity": semantic_count / 64,
        }
    return summary


def _interval(values: list[float], estimate: float) -> JsonDict:
    """Return a deterministic percentile interval from stored bootstrap draws."""

    ordered = sorted(values)
    lower = ordered[int((len(ordered) - 1) * 0.025)]
    upper = ordered[int((len(ordered) - 1) * 0.975)]
    return {"estimate": estimate, "ci95_lower": lower, "ci95_upper": upper}


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], seed: int, *, draws: int = BOOTSTRAP_DRAWS
) -> JsonDict:
    """Resample bases within family so all four dependent variants stay together."""

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
        arm = str(row["arm"])
        base_id = str(row["base_id"])
        by_arm_base[arm][base_id].append(row)
        base_family[base_id] = str(row["relation_family"])
    family_bases = {
        family: sorted(base_id for base_id, value in base_family.items() if value == family)
        for family in EVALUATION_FAMILIES
    }
    if any(len(base_ids) != 8 for base_ids in family_bases.values()):
        raise ValueError("bootstrap_family_cluster_count")

    def base_stats(arm: str, base_id: str) -> tuple[int, int, int]:
        selected = by_arm_base[arm][base_id]
        variants = {str(row["variant"]): row for row in selected}
        semantic = sum(
            int(
                not variants["original"]["abstention"]
                and not variants[variant]["abstention"]
                and variants["original"]["prediction"] != variants[variant]["prediction"]
            )
            for variant in ("relation_or_polarity_flip", "evidence_deletion")
        )
        return (
            sum(bool(row["correct"]) for row in selected),
            sum(bool(row["false_accept"]) for row in selected),
            semantic,
        )

    stats = {arm: {base_id: base_stats(arm, base_id) for base_id in base_family} for arm in ARMS}
    rng = random.Random(seed)
    accuracy_draws: list[float] = []
    false_accept_draws: list[float] = []
    syntax_sensitivity_draws: list[float] = []
    shuffle_sensitivity_draws: list[float] = []
    for _ in range(draws):
        selected_bases = [
            rng.choice(family_bases[family]) for family in EVALUATION_FAMILIES for _ in range(8)
        ]
        energy = [stats["energy_from_extracted_tuples"][base_id] for base_id in selected_bases]
        direct = [stats["baseline_direct"][base_id] for base_id in selected_bases]
        syntax = [stats["syntax_only"][base_id] for base_id in selected_bases]
        shuffled = [stats["shuffled_evidence"][base_id] for base_id in selected_bases]
        accuracy_draws.append((sum(row[0] for row in energy) - sum(row[0] for row in direct)) / 128)
        false_accept_draws.append(
            (sum(row[1] for row in energy) - sum(row[1] for row in direct)) / 128
        )
        syntax_sensitivity_draws.append(
            (sum(row[2] for row in energy) - sum(row[2] for row in syntax)) / 64
        )
        shuffle_sensitivity_draws.append(
            (sum(row[2] for row in energy) - sum(row[2] for row in shuffled)) / 64
        )
    metrics = summarize_arms(rows)
    energy_metric = metrics["energy_from_extracted_tuples"]
    direct_metric = metrics["baseline_direct"]
    syntax_metric = metrics["syntax_only"]
    shuffle_metric = metrics["shuffled_evidence"]
    return {
        "method": "paired_percentile_cluster_bootstrap",
        "draw_count": draws,
        "random_seed": seed,
        "cluster_key": "base_id",
        "stratified_by": "relation_family",
        "cluster_count": 32,
        "clusters_per_family": 8,
        "variants_per_cluster": 4,
        "full_denominator": 128,
        "paired_unit_ids": [row["unit_id"] for row in rows if row["arm"] == "baseline_direct"],
        "energy_vs_direct": {
            "accuracy_delta": _interval(
                accuracy_draws, energy_metric["accuracy"] - direct_metric["accuracy"]
            ),
            "false_accept_rate_delta": _interval(
                false_accept_draws,
                energy_metric["false_accept_rate"] - direct_metric["false_accept_rate"],
            ),
        },
        "energy_vs_syntax": {
            "semantic_edit_sensitivity_delta": _interval(
                syntax_sensitivity_draws,
                energy_metric["semantic_edit_sensitivity"]
                - syntax_metric["semantic_edit_sensitivity"],
            )
        },
        "energy_vs_shuffled_evidence": {
            "semantic_edit_sensitivity_delta": _interval(
                shuffle_sensitivity_draws,
                energy_metric["semantic_edit_sensitivity"]
                - shuffle_metric["semantic_edit_sensitivity"],
            )
        },
    }


def classify_verdict(inputs: Mapping[str, Any]) -> JsonDict:
    """Keep completion, value, circularity, and provenance as separate gates."""

    verifier_is_oracle = bool(inputs.get("verifier_is_oracle", False))
    if inputs.get("provenance_corrupt") or int(inputs.get("audit_disagreement_count", 0)) > 0:
        return {
            "grounding_measurement_complete_score": 0,
            "grounding_value_score": 0,
            "verifier_is_oracle": verifier_is_oracle,
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_grounding_energy_provenance_or_audit_disagreement",
        }
    value = bool(
        float(inputs["accuracy_ci95_lower"]) > 0
        and float(inputs["false_accept_ci95_upper"]) <= 0
        and inputs["no_leakage"] is True
        and float(inputs["semantic_sensitivity"]) > float(inputs["syntax_sensitivity"])
        and float(inputs["semantic_sensitivity"]) > float(inputs["shuffle_sensitivity"])
    )
    if value and verifier_is_oracle:
        verdict_class = "circular_positive"
        honest = "complete_circular_positive_grounding_energy_uses_label_authority"
    elif value:
        verdict_class = "positive"
        honest = "complete_positive_grounding_energy_pilot_value_gate_passed_narrow_scope"
    else:
        verdict_class = "null"
        honest = "complete_null_grounding_energy_pilot_value_gate_not_met"
    return {
        "grounding_measurement_complete_score": 1,
        "grounding_value_score": int(value),
        "verifier_is_oracle": verifier_is_oracle,
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
    """Record both sides of one precondition without inferred repair."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failure into the fixed terminal summary shape."""

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


def _base_artifact(run_date: str) -> JsonDict:
    """Create the full running schema before any source inspection."""

    return {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": _gate_summary(None),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_grounding_energy_audit",
        "inference_substrate_class": "blocked_no_run",
        "grounding_measurement_complete_score": 0,
        "grounding_value_score": 0,
        "paired_metrics": {},
        "feature_lineage_rows": [],
        "intervention_rows": [],
        "independent_audit_rows": [],
        "study_question": STUDY_QUESTION,
        "scope_answer": "Measurement has not completed.",
        "arm_metrics": {},
        "frozen_threshold_contract": {},
        "audit_receipt": {},
    }


def _resolve_paths(root: Path, overrides: Mapping[str, Path] | None) -> dict[str, Path]:
    """Resolve the fixed roster while allowing tests to replace one source."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = Path(path)
    return paths


def _preconditions(
    root: Path,
    run_date: str,
    paths: Mapping[str, Path],
    result_path: Path,
    checkpoint_path: Path,
    audit_request_path: Path,
    audit_result_path: Path,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Check exact bytes, terminal gates, tools, and destinations before scoring."""

    checks: list[JsonDict] = [
        _gate("run_date", RUN_DATE, run_date, run_date == RUN_DATE, upstream=None, field="run_date")
    ]
    hashes: dict[str, str] = {}
    for name, default in SOURCE_PATHS.items():
        path = paths[name]
        readable = path.is_file() and os.access(path, os.R_OK)
        checks.append(
            _gate(
                f"{name}_path",
                "readable_file",
                str(path) if readable else "missing_or_unreadable",
                readable,
                upstream=default.as_posix(),
                field="path",
            )
        )
        if not readable:
            return checks, hashes
        hashes[name] = sha256_file(path)
    for name, expected in PINNED_HASHES.items():
        checks.append(
            _gate(
                f"{name}_hash",
                expected,
                hashes[name],
                hashes[name] == expected,
                upstream=SOURCE_PATHS[name].as_posix(),
                field="sha256",
            )
        )
        if hashes[name] != expected:
            return checks, hashes
    spec_text = paths["constraint_spec"].read_text(encoding="utf-8")
    has_requirement = "REQ-VERIFY-7182" in spec_text
    checks.append(
        _gate(
            "constraint_spec_requirement",
            "REQ-VERIFY-7182",
            "REQ-VERIFY-7182" if has_requirement else "missing",
            has_requirement,
            upstream=SOURCE_PATHS["constraint_spec"].as_posix(),
            field="requirement",
        )
    )
    if not has_requirement:
        return checks, hashes
    fixture = json.loads(paths["fixture_artifact"].read_text(encoding="utf-8"))
    fixture_fields = {key: fixture.get(key) for key in EXP7180_EXPECTED_FIELDS}
    checks.append(
        _gate(
            "exp7180_same_milestone_gate_fields",
            EXP7180_EXPECTED_FIELDS,
            fixture_fields,
            fixture_fields == EXP7180_EXPECTED_FIELDS,
            upstream=SOURCE_PATHS["fixture_artifact"].as_posix(),
            field="terminal_gate_fields",
        )
    )
    if fixture_fields != EXP7180_EXPECTED_FIELDS:
        return checks, hashes
    trace = json.loads(paths["trace_artifact"].read_text(encoding="utf-8"))
    trace_fields = {key: trace.get(key) for key in EXP7181_EXPECTED_FIELDS}
    checks.append(
        _gate(
            "exp7181_same_milestone_gate_fields",
            EXP7181_EXPECTED_FIELDS,
            trace_fields,
            trace_fields == EXP7181_EXPECTED_FIELDS,
            upstream=SOURCE_PATHS["trace_artifact"].as_posix(),
            field="terminal_gate_fields",
        )
    )
    if trace_fields != EXP7181_EXPECTED_FIELDS:
        return checks, hashes
    raw_manifest = json.loads(paths["raw_manifest"].read_text(encoding="utf-8"))
    raw_rows = raw_manifest.get("raw_rows", [])
    raw_observed = {
        "row_count": len(raw_rows),
        "trace_row_count": len(trace.get("rows", [])),
        "unit_ids_match": [row.get("unit_id") for row in raw_rows]
        == [row.get("unit_id") for row in trace.get("rows", [])],
    }
    raw_expected = {"row_count": 192, "trace_row_count": 192, "unit_ids_match": True}
    checks.append(
        _gate(
            "raw_manifest_roster",
            raw_expected,
            raw_observed,
            raw_observed == raw_expected,
            upstream=SOURCE_PATHS["raw_manifest"].as_posix(),
            field="rows",
        )
    )
    if raw_observed != raw_expected:
        return checks, hashes
    python_observed = str(Path(sys.executable).resolve())
    python_expected = str((root / ".venv/bin/python").resolve())
    checks.append(
        _gate(
            "python_executable",
            python_expected,
            python_observed,
            python_observed == python_expected and os.access(sys.executable, os.X_OK),
            upstream=".venv/bin/python",
            field="executable",
        )
    )
    if not checks[-1]["passed"]:
        return checks, hashes
    output_paths = {
        "terminal_output_directory": result_path.parent,
        "checkpoint_output_directory": checkpoint_path.parent,
        "audit_request_directory": audit_request_path.parent,
        "audit_result_directory": audit_result_path.parent,
    }
    for check_name, directory in output_paths.items():
        writable = directory.is_dir() and os.access(directory, os.W_OK)
        checks.append(
            _gate(
                check_name,
                "existing_writable_directory",
                str(directory) if writable else "missing_or_unwritable",
                writable,
                upstream=str(directory),
                field="directory",
            )
        )
        if not writable:
            return checks, hashes
    return checks, hashes


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read exact sidecar rows after byte hashes and gates pass."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_auditor(
    root: Path, auditor_path: Path, request_path: Path, output_path: Path
) -> JsonDict:  # pragma: no cover - behavior is checked through the subprocess artifact.
    """Stream the fresh process and enforce its own bounded deadline."""

    command = [
        str(root / ".venv/bin/python"),
        "-u",
        str(auditor_path),
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    _progress(4, "START", "fresh-process independent auditor subprocess")
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
                f"HEARTBEAT elapsed_s={now - started:.1f} completed_units=0 "
                "operation=independent_auditor",
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
    _progress(4, "END", f"fresh-process auditor returncode={process.returncode}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _scope_answer(metrics: Mapping[str, Mapping[str, Any]], verdict: Mapping[str, Any]) -> str:
    """Answer the pilot question with its exact narrow statistical boundary."""

    energy = metrics["energy_from_extracted_tuples"]
    direct = metrics["baseline_direct"]
    delta = energy["accuracy"] - direct["accuracy"]
    if verdict["grounding_value_score"] == 1:
        result = "The frozen grounding energy passed every preregistered pilot value gate."
    else:
        result = "The frozen grounding energy did not pass every preregistered pilot value gate."
    return (
        f"{result} Held-out full-denominator accuracy was {energy['accuracy']:.6f} "
        f"versus {direct['accuracy']:.6f} for direct decisions (delta {delta:.6f}). "
        "This 32-base pilot supports no broad model or benchmark claim."
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
    duration_s: float | None = None,
) -> JsonDict:
    """Gate inputs, measure all arms, run the auditor, and atomically finish."""

    started = time.monotonic()
    _progress(0, "START", "checkpoint shell before any prerequisite check")
    root = Path(root).resolve()
    result = Path(result_path) if result_path is not None else root / RESULT_PATH
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_PATH
    audit_request = (
        Path(audit_request_path) if audit_request_path is not None else root / AUDIT_REQUEST_PATH
    )
    audit_result = (
        Path(audit_result_path) if audit_result_path is not None else root / AUDIT_RESULT_PATH
    )
    artifact = _base_artifact(run_date)
    atomic_write_json(checkpoint, artifact, allow_override=False, sort_keys=True)
    _progress(0, "END", f"running checkpoint={checkpoint}")

    _progress(1, "START", "source bytes same-milestone fields tools and output paths")
    resolved = _resolve_paths(root, source_paths)
    checks, hashes = _preconditions(
        root, run_date, resolved, result, checkpoint, audit_request, audit_result
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
                "honest_verdict": "blocked_grounding_energy_audit_external_precondition",
                "scope_answer": "The audit did not run because an external prerequisite failed.",
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_write_json(result, artifact, allow_override=False, sort_keys=True)
        _progress(1, "END", f"blocked check={failure['check']}")
        return artifact
    _progress(1, "END", f"preconditions={len(checks)} all_passed=true")

    _progress(2, "START", "raw completion parsing and calibration-only threshold fit")
    trace = json.loads(resolved["trace_artifact"].read_text(encoding="utf-8"))
    fixture = json.loads(resolved["fixture_artifact"].read_text(encoding="utf-8"))
    generation_rows = _read_jsonl(resolved["generation_view"])
    authority_rows = _read_jsonl(resolved["authority_sidecar"])
    features = build_candidate_features(trace["rows"], generation_rows)
    calibration = [row for row in authority_rows if row["split"] == "calibration"]
    frozen_threshold = select_threshold(features, calibration)
    _progress(
        2,
        "END",
        f"features={len(features)} threshold={frozen_threshold['threshold']} fit_rows=64",
    )

    _progress(3, "START", "held-out five-arm benchmark and clustered bootstrap")
    evaluation = [row for row in authority_rows if row["split"] == "evaluation"]
    rows = evaluate_arms(features, evaluation, frozen_threshold, fixture["score_contract"])
    arm_metrics = summarize_arms(rows)
    paired_metrics = paired_cluster_bootstrap(rows, BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
    _progress(3, "END", f"arm_rows={len(rows)} bootstrap_draws={BOOTSTRAP_DRAWS}")

    evaluation_ids = {str(row["unit_id"]) for row in evaluation}
    shuffle_mapping = [
        row
        for row in fixture["score_contract"]["seeded_shuffle_controls"][0]["mapping"]
        if str(row["unit_id"]) in evaluation_ids
    ]
    label_mapping = [
        row
        for row in fixture["score_contract"]["label_permutation_control"]["mapping"]
        if str(row["unit_id"]) in evaluation_ids
    ]
    audit_input_hashes = {
        name: hashes[name] for name in ("trace_artifact", "generation_view", "authority_sidecar")
    }
    audit_request_value = {
        "threshold": frozen_threshold["threshold"],
        "weights": deepcopy(ENERGY_WEIGHTS),
        "candidate_rows": [
            {
                "unit_id": row["unit_id"],
                "prediction": row["prediction"],
                "score": row["score"],
                "energy_terms": deepcopy(row["energy_terms"]),
                "parse_status": row["parse_status"],
            }
            for row in rows
            if row["arm"] == "energy_from_extracted_tuples"
        ],
        "expected_input_hashes": audit_input_hashes,
        "random_seed": RANDOM_SEED,
        "source_paths": {
            name: str(resolved[name])
            for name in ("trace_artifact", "generation_view", "authority_sidecar")
        },
        "shuffle_mapping": shuffle_mapping,
        "label_permutation_mapping": label_mapping,
    }
    atomic_write_json(audit_request, audit_request_value, allow_override=False, sort_keys=True)
    audit = _run_auditor(root, resolved["auditor_entrypoint"], audit_request, audit_result)

    audit_receipt = {
        "process_id": audit["process_id"],
        "candidate_module_imported": audit["candidate_module_imported"],
        "formula_implemented_independently": audit["formula_implemented_independently"],
        "threshold_adapted": audit["threshold_adapted"],
        "threshold": audit["threshold"],
        "formula_sha256": audit["formula_sha256"],
        "input_hashes": audit["input_hashes"],
        "disagreement_count": audit["disagreement_count"],
        "disagreements": audit["disagreements"],
        "audit_request_sha256": sha256_file(audit_request),
        "audit_result_sha256": sha256_file(audit_result),
    }
    energy_metric = arm_metrics["energy_from_extracted_tuples"]
    paired = paired_metrics["energy_vs_direct"]
    verdict_inputs = {
        "accuracy_ci95_lower": paired["accuracy_delta"]["ci95_lower"],
        "false_accept_ci95_upper": paired["false_accept_rate_delta"]["ci95_upper"],
        "no_leakage": all(
            row["feature_lineage"]["authority_used_for_features"] is False for row in rows
        )
        and all(
            row["authority_used_for_features"] is False for row in audit["feature_lineage_rows"]
        ),
        "semantic_sensitivity": energy_metric["semantic_edit_sensitivity"],
        "syntax_sensitivity": arm_metrics["syntax_only"]["semantic_edit_sensitivity"],
        "shuffle_sensitivity": arm_metrics["shuffled_evidence"]["semantic_edit_sensitivity"],
        "audit_disagreement_count": audit["disagreement_count"],
        "provenance_corrupt": (
            audit["candidate_module_imported"] is not False
            or audit["threshold_adapted"] is not False
            or audit["input_hashes"] != audit_input_hashes
        ),
        "verifier_is_oracle": False,
    }
    verdict = classify_verdict(verdict_inputs)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate_class": "no_model_load",
            "duration_s": (
                float(duration_s) if duration_s is not None else time.monotonic() - started
            ),
            "rows": rows,
            "gate_check_summary": _gate_summary(None),
            "paired_metrics": paired_metrics,
            "feature_lineage_rows": audit["feature_lineage_rows"],
            "intervention_rows": audit["intervention_rows"],
            "independent_audit_rows": audit["independent_audit_rows"],
            "arm_metrics": arm_metrics,
            "frozen_threshold_contract": frozen_threshold,
            "audit_receipt": audit_receipt,
            **verdict,
        }
    )
    artifact["scope_answer"] = _scope_answer(arm_metrics, verdict)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(5, "START", "terminal validation and atomic result write")
    errors = validate_artifact(artifact, root=root, check_source_hashes=True)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_write_json(result, artifact, allow_override=False, sort_keys=True)
    _progress(5, "END", f"terminal_result={result} verdict={artifact['verdict_class']}")
    return artifact


def validate_artifact(
    value: Mapping[str, Any] | str | Path,
    *,
    root: Path | None = None,
    check_source_hashes: bool = True,
) -> list[str]:
    """Cold-check schema, rows, metrics, audit receipts, sources, and verdict."""

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
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    status = artifact.get("status")
    if status == "blocked":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict")
        if artifact.get("grounding_measurement_complete_score") != 0:
            errors.append("blocked_measurement_score")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary")
        return errors
    if status != "complete":
        errors.append("terminal_status")
        return errors
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class")
    row_errors = evaluation_row_errors(artifact.get("rows", []))
    errors.extend(row_errors)
    if not row_errors:
        metrics = summarize_arms(artifact["rows"])
        if artifact.get("arm_metrics") != metrics:
            errors.append("arm_metrics")
        paired = paired_cluster_bootstrap(artifact["rows"], BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)
        if artifact.get("paired_metrics") != paired:
            errors.append("paired_metrics")
    if len(artifact.get("independent_audit_rows", [])) != 128:
        errors.append("independent_audit_rows")
    if len(artifact.get("feature_lineage_rows", [])) != 256:
        errors.append("feature_lineage_rows")
    intervention_rows = artifact.get("intervention_rows", [])
    if len(intervention_rows) != 128 * (2 + len(ENERGY_TERMS)):
        errors.append("intervention_rows")
    audit_receipt = artifact.get("audit_receipt", {})
    if audit_receipt.get("candidate_module_imported") is not False:
        errors.append("candidate_module_imported")
    if audit_receipt.get("threshold_adapted") is not False:
        errors.append("threshold_adapted")
    independent = artifact.get("independent_audit_rows", [])
    disagreement_rows = [row for row in independent if row.get("disagreement")]
    if audit_receipt.get("disagreement_count") != len(disagreement_rows):
        errors.append("audit_disagreement_count")
    energy_rows = {
        row["unit_id"]: row
        for row in artifact.get("rows", [])
        if row.get("arm") == "energy_from_extracted_tuples"
    }
    for row in independent:
        candidate = energy_rows.get(row.get("unit_id"))
        if candidate is None or (
            row.get("candidate_prediction") != candidate.get("prediction")
            or row.get("candidate_score") != candidate.get("score")
            or row.get("prediction") != candidate.get("prediction")
            or row.get("energy_terms") != candidate.get("energy_terms")
        ):
            errors.append("independent_decision_mismatch")
            break
    if any(
        row.get("authority_used_for_features") is not False
        for row in artifact.get("feature_lineage_rows", [])
    ):
        errors.append("authority_leakage")
    if check_source_hashes:
        repository = Path(root).resolve() if root is not None else find_repo_root()
        paths = _resolve_paths(repository, None)
        observed = {
            name: sha256_file(path)
            for name, path in paths.items()
            if path.is_file() and os.access(path, os.R_OK)
        }
        if artifact.get("source_artifact_hashes") != observed:
            errors.append("source_artifact_hashes")
    if any(not row.get("passed") for row in artifact.get("preconditions_checked", [])):
        errors.append("preconditions_checked")
    if artifact.get("grounding_measurement_complete_score") not in {0, 1}:
        errors.append("grounding_measurement_complete_score")
    if artifact.get("grounding_value_score") not in {0, 1}:
        errors.append("grounding_value_score")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "disqualified",
    }:
        errors.append("verdict_class")
    return list(dict.fromkeys(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Build the fixed-date result or cold-validate an existing artifact."""

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
                "grounding_measurement_complete_score": artifact[
                    "grounding_measurement_complete_score"
                ],
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
