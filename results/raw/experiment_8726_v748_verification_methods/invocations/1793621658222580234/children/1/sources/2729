"""Independently replay the Exp7182 energy and causal controls.

This module uses only the Python standard library. The executable imports this
module in a fresh process, so it cannot reuse the candidate scorer by accident.

Spec refs: REQ-VERIFY-7182 and SCENARIO-VERIFY-7182-AUDIT.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any


JsonDict = dict[str, Any]
CANDIDATE_MODULE = "carnot.experiment_7182_v633_grounding_energy_audit"
RESPONSE_FIELDS = {
    "direct_decision",
    "claim_tuple",
    "evidence_tuple",
    "source_start",
    "source_end",
    "missing_fields",
}
ENERGY_TERMS = (
    "tuple_alignment",
    "polarity",
    "quantity_unit_agreement",
    "literal_span_validity",
    "missing_required_fields",
)


def canonical_json(value: Any) -> str:
    """Use one compact JSON spelling so both processes hash identical bytes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Bind exact text bytes without normalizing model or prompt output."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Bind the files that the independent process opens directly."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _prompt_parts(prompt: str) -> tuple[str, str]:
    """Extract the exact evidence and claim text that the model received."""

    try:
        remainder = prompt.split("Evidence:\n", 1)[1]
        evidence, remainder = remainder.split("\n\nClaim:\n", 1)
        claim = remainder.split("\n\nReturn one JSON object", 1)[0]
    except IndexError as error:
        raise ValueError("prompt_shape_invalid") from error
    return evidence, claim


def _tuple_valid(value: Any) -> bool:
    """Accept only the six-field tuple shape requested from the model."""

    if not isinstance(value, Mapping):
        return False
    required = {"subject", "relation", "object", "polarity", "quantity", "unit"}
    return (
        set(value) == required
        and all(isinstance(value[field], str) for field in ("subject", "relation", "object"))
        and value["polarity"] in {"positive", "negative"}
        and (
            value["quantity"] is None
            or isinstance(value["quantity"], (int, float))
            and not isinstance(value["quantity"], bool)
        )
        and (value["unit"] is None or isinstance(value["unit"], str))
    )


def parse_raw_output(raw_output: str, prompt: str) -> JsonDict:
    """Parse exact output once and refuse to repair malformed model text."""

    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return {"parse_status": "failed", "parse_error": "invalid_json", "parsed": None}
    if not isinstance(value, dict):
        return {"parse_status": "failed", "parse_error": "root_not_object", "parsed": None}
    if set(value) != RESPONSE_FIELDS:
        return {"parse_status": "failed", "parse_error": "field_set_mismatch", "parsed": None}
    if value["direct_decision"] not in {"supported", "unsupported", "abstain"}:
        return {
            "parse_status": "failed",
            "parse_error": "direct_decision_invalid",
            "parsed": None,
        }
    if not _tuple_valid(value["claim_tuple"]):
        return {"parse_status": "failed", "parse_error": "claim_tuple_invalid", "parsed": None}
    if value["evidence_tuple"] is not None and not _tuple_valid(value["evidence_tuple"]):
        return {
            "parse_status": "failed",
            "parse_error": "evidence_tuple_invalid",
            "parsed": None,
        }
    if not isinstance(value["missing_fields"], list) or not all(
        isinstance(field, str) for field in value["missing_fields"]
    ):
        return {"parse_status": "failed", "parse_error": "missing_fields_invalid", "parsed": None}
    evidence, _ = _prompt_parts(prompt)
    start = value["source_start"]
    end = value["source_end"]
    null_span = start is None and end is None
    integer_span = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start <= end <= len(evidence)
    )
    if not (null_span or integer_span):
        return {"parse_status": "failed", "parse_error": "source_span_invalid", "parsed": None}
    return {"parse_status": "valid", "parse_error": None, "parsed": value}


def independent_energy(
    response: Mapping[str, Any], source_text: str, weights: Mapping[str, int]
) -> JsonDict:
    """Implement the five fixed terms locally without importing candidate code."""

    missing = int(any(field not in response for field in RESPONSE_FIELDS))
    claim = response.get("claim_tuple")
    evidence = response.get("evidence_tuple")
    claim_valid = _tuple_valid(claim)
    evidence_valid = evidence is None or _tuple_valid(evidence)
    if not claim_valid or not evidence_valid:
        missing = 1

    alignment = 0
    polarity = 0
    quantity_unit = 0
    if claim_valid and evidence is None:
        alignment = 1
    elif claim_valid and _tuple_valid(evidence):
        alignment = int(
            any(claim[field] != evidence[field] for field in ("subject", "relation", "object"))
        )
        polarity = int(claim["polarity"] != evidence["polarity"])
        quantity_unit = int(
            claim.get("quantity") != evidence.get("quantity")
            or claim.get("unit") != evidence.get("unit")
        )

    start = response.get("source_start")
    end = response.get("source_end")
    literal_span = 0
    if evidence is None:
        literal_span = int(start is not None or end is not None or bool(source_text))
    elif _tuple_valid(evidence):
        valid_bounds = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start == 0
            and end == len(source_text)
        )
        if valid_bounds:
            exact = source_text[start:end]
            literals = [str(evidence[field]) for field in ("subject", "relation", "object")]
            if "quantity" in evidence:
                literals.extend([str(evidence["quantity"]), str(evidence["unit"])])
            literal_span = int(any(literal not in exact for literal in literals))
        else:
            literal_span = 1

    terms: JsonDict = {
        "tuple_alignment": alignment,
        "polarity": polarity,
        "quantity_unit_agreement": quantity_unit,
        "literal_span_validity": literal_span,
        "missing_required_fields": missing,
    }
    terms["total"] = sum(int(weights[name]) * terms[name] for name in ENERGY_TERMS)
    return terms


def validate_request(request: Mapping[str, Any]) -> list[str]:
    """Reject label transport and incomplete frozen contracts before auditing."""

    errors: list[str] = []
    if "evaluation_labels" in request:
        errors.append("request_contains_evaluation_labels")
    required = {"threshold", "weights", "candidate_rows", "expected_input_hashes", "random_seed"}
    if not required <= set(request):
        errors.append("request_fields_missing")
    if set(request.get("weights", {})) != set(ENERGY_TERMS):
        errors.append("weight_terms_mismatch")
    return errors


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read one immutable sidecar without normalizing its bytes."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _features(
    trace_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, int],
) -> list[JsonDict]:
    """Rebuild model-only features from exact raw output and prompt bytes."""

    if len(trace_rows) != len(generation_rows):
        raise ValueError("source_row_count_mismatch")
    built: list[JsonDict] = []
    for trace, generation in zip(trace_rows, generation_rows, strict=True):
        if trace.get("unit_id") != generation.get("unit_id"):
            raise ValueError("source_unit_order_mismatch")
        prompt = str(generation.get("text", ""))
        if trace.get("prompt") != prompt:
            raise ValueError("source_prompt_mismatch")
        source_text, _ = _prompt_parts(prompt)
        raw_output = str(trace.get("raw_output", ""))
        parsed = parse_raw_output(raw_output, prompt)
        response = parsed["parsed"]
        energy = (
            independent_energy(response, source_text, weights)
            if isinstance(response, Mapping)
            else None
        )
        built.append(
            {
                "unit_id": str(trace["unit_id"]),
                "parse_status": parsed["parse_status"],
                "parse_error": parsed["parse_error"],
                "parsed_response": deepcopy(response),
                "source_text": source_text,
                "raw_output_sha256": sha256_text(raw_output),
                "source_text_sha256": sha256_text(source_text),
                "energy_terms": energy,
            }
        )
    return built


def _prediction(feature: Mapping[str, Any], threshold: int) -> str:
    """Map an independently scored row to the frozen decision or abstention."""

    if feature["parse_status"] != "valid":
        return "abstain"
    return "supported" if feature["energy_terms"]["total"] <= threshold else "unsupported"


def _swapped_feature(
    target: Mapping[str, Any], donor: Mapping[str, Any], weights: Mapping[str, int]
) -> tuple[str, JsonDict | None]:
    """Replace only evidence with a same-family donor and keep failures visible."""

    if target["parse_status"] != "valid" or donor["parse_status"] != "valid":
        return "abstain", None
    response = deepcopy(target["parsed_response"])
    donor_response = donor["parsed_response"]
    response["evidence_tuple"] = deepcopy(donor_response["evidence_tuple"])
    response["source_start"] = donor_response["source_start"]
    response["source_end"] = donor_response["source_end"]
    terms = independent_energy(response, str(donor["source_text"]), weights)
    return "scored", terms


def run_audit(request: Mapping[str, Any]) -> JsonDict:
    """Recompute decisions, lineage, swaps, deletions, and permuted labels."""

    errors = validate_request(request)
    if errors:
        raise ValueError("audit_request_invalid:" + ",".join(errors))
    paths = {name: Path(value) for name, value in request["source_paths"].items()}
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if observed_hashes != request["expected_input_hashes"]:
        raise ValueError("audit_input_hash_mismatch")

    trace_value = json.loads(paths["trace_artifact"].read_text(encoding="utf-8"))
    trace_rows = trace_value["rows"]
    generation_rows = _read_jsonl(paths["generation_view"])
    authority_rows = _read_jsonl(paths["authority_sidecar"])
    weights = {key: int(value) for key, value in request["weights"].items()}
    threshold = int(request["threshold"])
    features = _features(trace_rows, generation_rows, weights)
    by_feature = {row["unit_id"]: row for row in features}
    evaluation = [row for row in authority_rows if row["split"] == "evaluation"]
    candidate = {row["unit_id"]: row for row in request["candidate_rows"]}
    if set(candidate) != {row["unit_id"] for row in evaluation}:
        raise ValueError("candidate_evaluation_roster_mismatch")

    independent_rows: list[JsonDict] = []
    lineage_rows: list[JsonDict] = []
    for truth in evaluation:
        unit_id = str(truth["unit_id"])
        feature = by_feature[unit_id]
        prediction = _prediction(feature, threshold)
        candidate_row = candidate[unit_id]
        disagreement = (
            prediction != candidate_row["prediction"]
            or feature["energy_terms"] != candidate_row["energy_terms"]
            or feature["parse_status"] != candidate_row["parse_status"]
        )
        independent_rows.append(
            {
                "unit_id": unit_id,
                "base_id": truth["base_id"],
                "relation_family": truth["relation_family"],
                "variant": truth["variant"],
                "parse_status": feature["parse_status"],
                "parse_error": feature["parse_error"],
                "prediction": prediction,
                "score": (
                    feature["energy_terms"]["total"]
                    if feature["energy_terms"] is not None
                    else None
                ),
                "energy_terms": deepcopy(feature["energy_terms"]),
                "candidate_prediction": candidate_row["prediction"],
                "candidate_score": candidate_row["score"],
                "disagreement": disagreement,
                "raw_output_sha256": feature["raw_output_sha256"],
                "source_text_sha256": feature["source_text_sha256"],
            }
        )
        lineage_rows.append(
            {
                "unit_id": unit_id,
                "condition": "energy_from_extracted_tuples",
                "feature_names": list(ENERGY_TERMS),
                "energy_terms": deepcopy(feature["energy_terms"]),
                "raw_output_sha256": feature["raw_output_sha256"],
                "source_text_sha256": feature["source_text_sha256"],
                "authority_used_for_features": False,
                "donor_unit_id": None,
            }
        )

    shuffle_mapping = {row["unit_id"]: row["donor_unit_id"] for row in request["shuffle_mapping"]}
    if set(shuffle_mapping) != {row["unit_id"] for row in evaluation}:
        raise ValueError("shuffle_mapping_roster_mismatch")
    for truth in evaluation:
        unit_id = str(truth["unit_id"])
        donor_id = str(shuffle_mapping[unit_id])
        target = by_feature[unit_id]
        donor = by_feature[donor_id]
        state, terms = _swapped_feature(target, donor, weights)
        lineage_rows.append(
            {
                "unit_id": unit_id,
                "condition": "shuffled_evidence",
                "feature_names": list(ENERGY_TERMS),
                "energy_terms": deepcopy(terms),
                "raw_output_sha256": target["raw_output_sha256"],
                "source_text_sha256": donor["source_text_sha256"],
                "authority_used_for_features": False,
                "donor_unit_id": donor_id,
                "state": state,
            }
        )

    interventions: list[JsonDict] = []
    family_variant: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in evaluation:
        family_variant[(str(row["relation_family"]), str(row["variant"]))].append(
            str(row["unit_id"])
        )
    swap_donors: dict[str, str] = {}
    for unit_ids in family_variant.values():
        ordered = sorted(unit_ids)
        for index, unit_id in enumerate(ordered):
            swap_donors[unit_id] = ordered[(index + 1) % len(ordered)]

    for truth in evaluation:
        unit_id = str(truth["unit_id"])
        feature = by_feature[unit_id]
        donor_id = swap_donors[unit_id]
        state, terms = _swapped_feature(feature, by_feature[donor_id], weights)
        swap_prediction = (
            "abstain"
            if state == "abstain"
            else "supported"
            if terms["total"] <= threshold
            else "unsupported"
        )
        interventions.append(
            {
                "unit_id": unit_id,
                "intervention": "evidence_swap_within_family",
                "donor_unit_id": donor_id,
                "prediction": swap_prediction,
                "score": terms["total"] if terms is not None else None,
                "energy_terms": deepcopy(terms),
                "threshold": threshold,
            }
        )
        for term in ENERGY_TERMS:
            original = feature["energy_terms"]
            if original is None:
                deleted_total = None
                prediction = "abstain"
            else:
                deleted_total = original["total"] - weights[term] * original[term]
                prediction = "supported" if deleted_total <= threshold else "unsupported"
            interventions.append(
                {
                    "unit_id": unit_id,
                    "intervention": f"delete_term:{term}",
                    "deleted_term": term,
                    "prediction": prediction,
                    "score": deleted_total,
                    "threshold": threshold,
                }
            )

    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    label_mapping = {
        row["unit_id"]: row["donor_unit_id"] for row in request["label_permutation_mapping"]
    }
    if set(label_mapping) != {row["unit_id"] for row in evaluation}:
        raise ValueError("label_mapping_roster_mismatch")
    independent_by_id = {row["unit_id"]: row for row in independent_rows}
    for truth in evaluation:
        unit_id = str(truth["unit_id"])
        donor_id = str(label_mapping[unit_id])
        permuted = authority_by_id[donor_id]["expected_response"]["direct_decision"]
        prediction = independent_by_id[unit_id]["prediction"]
        interventions.append(
            {
                "unit_id": unit_id,
                "intervention": "label_permutation",
                "donor_unit_id": donor_id,
                "prediction": prediction,
                "permuted_authority_label": permuted,
                "correct": prediction == permuted,
                "score": independent_by_id[unit_id]["score"],
                "threshold": threshold,
            }
        )

    disagreements = [row for row in independent_rows if row["disagreement"]]
    return {
        "process_id": os.getpid(),
        "candidate_module_imported": CANDIDATE_MODULE in sys.modules,
        "formula_implemented_independently": True,
        "threshold": threshold,
        "threshold_adapted": False,
        "input_hashes": observed_hashes,
        "formula_sha256": sha256_text(
            canonical_json({"terms": ENERGY_TERMS, "weights": weights, "threshold": threshold})
        ),
        "independent_audit_rows": independent_rows,
        "feature_lineage_rows": lineage_rows,
        "intervention_rows": interventions,
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the audit response only after one complete JSON write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised as a subprocess.
    """Read one frozen request and write the independent response."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print("PHASE AUDIT START independent recomputation", flush=True)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_audit(request)
    _atomic_write_json(args.output, result)
    print(
        "PHASE AUDIT END "
        f"units={len(result['independent_audit_rows'])} "
        f"disagreements={result['disagreement_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
