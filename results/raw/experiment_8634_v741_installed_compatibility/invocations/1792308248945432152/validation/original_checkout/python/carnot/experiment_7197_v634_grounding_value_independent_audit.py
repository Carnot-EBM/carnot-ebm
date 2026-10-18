"""Recompute cached typed-grounding decisions in a fresh evaluator process.

The evaluator opens public model calls before it opens labels. It uses the same
complete fixture authority as the candidate, so separate execution does not
make the verifier oracle-distinct.

Spec refs: REQ-VERIFY-7197 and SCENARIO-VERIFY-7197-AUDIT/CIRCULARITY.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from carnot.experiment_7196_v634_qwen_atomic_capture import parse_output
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    TypedRelation,
    execute_relation,
)


JsonDict = dict[str, Any]
CANDIDATE_MODULE = "carnot.experiment_7197_v634_grounding_value_audit"
REQUEST_FIELDS = {
    "schema",
    "run_date",
    "random_seed",
    "source_paths",
    "expected_input_hashes",
    "candidate_predictions",
    "candidate_prediction_checksum",
    "shuffle_mapping",
}
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "authority_label",
        "evaluation_labels",
        "expected_executor_decision",
        "expected_prediction",
        "support_label",
        "truth",
    }
)


def canonical_json(value: Any) -> str:
    """Use one compact JSON spelling for cross-process hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text without normalizing model bytes."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash each source file exactly as the evaluator opens it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _contains_forbidden_key(value: Any) -> bool:
    """Reject label objects even when a caller nests them."""

    if isinstance(value, Mapping):
        if FORBIDDEN_LABEL_KEYS & set(value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_request(request: Mapping[str, Any]) -> list[str]:
    """Require the frozen request shape and deny transported labels."""

    errors: list[str] = []
    if set(request) != REQUEST_FIELDS:
        errors.append("request_fields")
    if _contains_forbidden_key(request):
        errors.append("forbidden_label_transport")
    if set(request.get("source_paths", {})) != {
        "public_view",
        "capture_artifact",
        "authority_sidecar",
    }:
        errors.append("source_paths")
    if not isinstance(request.get("candidate_predictions"), list):
        errors.append("candidate_predictions")
    return list(dict.fromkeys(errors))


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read immutable JSON lines without changing their byte contract."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _call_features(row: Mapping[str, Any]) -> JsonDict:
    """Reparse one exact raw output instead of trusting producer projections."""

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
    }


def _build_features(
    public_rows: Sequence[Mapping[str, Any]], completion_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Build the evaluator view from public bytes and three raw calls."""

    calls: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in completion_rows:
        key = (str(row.get("unit_id")), str(row.get("call_type")))
        if key in calls:
            raise ValueError("duplicate_call")
        calls[key] = row
    features: list[JsonDict] = []
    for public in public_rows:
        unit_id = str(public["unit_id"])
        required = [(unit_id, call_type) for call_type in ("source", "claim", "direct")]
        if any(key not in calls for key in required):
            raise ValueError(f"missing_call:{unit_id}")
        features.append(
            {
                "unit_id": unit_id,
                "source_text": str(public["source_text"]),
                "claim_text": str(public["claim_text"]),
                "source": _call_features(calls[(unit_id, "source")]),
                "claim": _call_features(calls[(unit_id, "claim")]),
                "direct": _call_features(calls[(unit_id, "direct")]),
            }
        )
    return features


def _typed_prediction(
    source_text: str,
    source_call: Mapping[str, Any],
    claim_call: Mapping[str, Any],
    *,
    intervention: str | None = None,
) -> JsonDict:
    """Map separated typed output into one explicit supported state or abstention."""

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
    claim_bindings = claim.get("entity_bindings", [])
    claim_surfaces: dict[str, list[str]] = {}
    for item in claim_bindings:
        claim_surfaces.setdefault(str(item["entity_id"]), []).append(str(item["surface"]))
    source_ids: dict[str, list[str]] = {}
    for binding in source_bindings:
        source_ids.setdefault(binding.surface, []).append(binding.entity_id)

    relation = deepcopy(claim_relations[0])
    mapped: dict[str, str] = {}
    for field in ("subject_id", "object_id"):
        surfaces = claim_surfaces.get(str(relation[field]), [])
        matches = source_ids.get(surfaces[0], []) if len(surfaces) == 1 else []
        if len(matches) != 1:
            return {"prediction": "abstain", "error": f"entity_mapping:{field}"}
        mapped[field] = matches[0]
    if intervention == "argument_reversal":
        mapped["subject_id"], mapped["object_id"] = mapped["object_id"], mapped["subject_id"]
    typed_claim = TypedRelation(
        mapped["subject_id"],
        str(relation["operator"]),
        mapped["object_id"],
        str(relation["polarity"]),
        int(relation["source_start"]),
        int(relation["source_end"]),
    )
    source_relations = tuple(TypedRelation(**item) for item in source.get("relations", []))
    if intervention == "semantic_deletion":
        source_relations = ()
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


def _prediction_rows(
    features: Sequence[Mapping[str, Any]], shuffle_map: Mapping[str, str]
) -> list[JsonDict]:
    """Compute typed decisions and three interventions without labels."""

    by_id = {str(row["unit_id"]): row for row in features}
    rows: list[JsonDict] = []
    for feature in features:
        unit_id = str(feature["unit_id"])
        primary = _typed_prediction(
            str(feature["source_text"]), feature["source"], feature["claim"]
        )
        reversed_result = _typed_prediction(
            str(feature["source_text"]),
            feature["source"],
            feature["claim"],
            intervention="argument_reversal",
        )
        deleted_result = _typed_prediction(
            str(feature["source_text"]),
            feature["source"],
            feature["claim"],
            intervention="semantic_deletion",
        )
        donor_id = str(shuffle_map[unit_id])
        donor = by_id[donor_id]
        shuffled_result = _typed_prediction(
            str(donor["source_text"]), donor["source"], feature["claim"]
        )
        rows.append(
            {
                "unit_id": unit_id,
                "prediction": primary["prediction"],
                "error": primary["error"],
                "parse_status": (
                    "valid"
                    if feature["source"]["parse_status"] == "valid"
                    and feature["claim"]["parse_status"] == "valid"
                    else "failed"
                ),
                "controls": [
                    {"intervention": "argument_reversal", **reversed_result},
                    {"intervention": "semantic_deletion", **deleted_result},
                    {
                        "intervention": "shuffled_source",
                        "donor_unit_id": donor_id,
                        **shuffled_result,
                    },
                ],
            }
        )
    return rows


def _prediction_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash only policy outputs, not runtime or evaluator labels."""

    projected = [
        {
            "unit_id": row["unit_id"],
            "prediction": row["prediction"],
            "error": row["error"],
            "parse_status": row["parse_status"],
            "controls": row["controls"],
        }
        for row in rows
    ]
    return sha256_text(canonical_json(projected))


def _candidate_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the narrow candidate projection carried across the process boundary."""

    projected = [
        {
            "unit_id": row["unit_id"],
            "prediction": row["prediction"],
            "parse_status": row["parse_status"],
        }
        for row in rows
    ]
    return sha256_text(canonical_json(projected))


def run_audit(request: Mapping[str, Any]) -> JsonDict:
    """Reparse, execute, intervene, then open labels and compare decisions."""

    errors = validate_request(request)
    if errors:
        raise ValueError("audit_request_invalid:" + ",".join(errors))
    paths = {name: Path(value) for name, value in request["source_paths"].items()}
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if observed_hashes != request["expected_input_hashes"]:
        raise ValueError("audit_input_hash_mismatch")

    public_rows = _read_jsonl(paths["public_view"])
    capture = json.loads(paths["capture_artifact"].read_text(encoding="utf-8"))
    features = _build_features(public_rows, capture["completion_rows"])
    shuffle_map = {
        str(row["unit_id"]): str(row["donor_unit_id"]) for row in request["shuffle_mapping"]
    }
    predictions = _prediction_rows(features, shuffle_map)
    prediction_checksum = _prediction_checksum(predictions)
    if (
        _candidate_checksum(request["candidate_predictions"])
        != request["candidate_prediction_checksum"]
    ):
        raise ValueError("candidate_prediction_checksum_mismatch")

    # Labels open only after the policy output is immutable. A label-only
    # mutation cannot enter this function's public feature arguments.
    authority = _read_jsonl(paths["authority_sidecar"])
    changed_authority = deepcopy(authority)
    for row in changed_authority:
        row["support_label"] = "unsupported" if row["support_label"] == "supported" else "supported"
    repeated_checksum = _prediction_checksum(_prediction_rows(features, shuffle_map))
    label_invariant = repeated_checksum == prediction_checksum

    candidate = {str(row["unit_id"]): row for row in request["candidate_predictions"]}
    prediction_by_id = {str(row["unit_id"]): row for row in predictions}
    evaluation = [row for row in authority if row["split"] == "evaluation"]
    audit_rows: list[JsonDict] = []
    for truth in evaluation:
        unit_id = str(truth["unit_id"])
        row = prediction_by_id[unit_id]
        candidate_row = candidate[unit_id]
        disagreement = row["prediction"] != candidate_row.get("prediction") or row[
            "parse_status"
        ] != candidate_row.get("parse_status")
        audit_rows.append(
            {
                "unit_id": unit_id,
                "base_id": truth["base_id"],
                "relation_family": truth["relation_family"],
                "variant": truth["variant"],
                "authority_label": truth["support_label"],
                "prediction": row["prediction"],
                "candidate_prediction": candidate_row.get("prediction"),
                "parse_status": row["parse_status"],
                "error": row["error"],
                "abstention": row["prediction"] == "abstain",
                "correct": row["prediction"] == truth["support_label"],
                "decision_disagreement": disagreement,
                "controls": deepcopy(row["controls"]),
            }
        )
    disagreements = [row for row in audit_rows if row["decision_disagreement"]]
    controls_complete = all(len(row["controls"]) == 3 for row in audit_rows)
    passed = not disagreements and len(audit_rows) == 128 and controls_complete and label_invariant
    return {
        "process_id": os.getpid(),
        "candidate_module_imported": CANDIDATE_MODULE in sys.modules,
        "input_hashes": observed_hashes,
        "labels_opened_after_prediction": True,
        "policy_input_fields": ["public_view", "capture_artifact", "shuffle_mapping"],
        "forbidden_label_transport": False,
        "prediction_checksum_before_labels": prediction_checksum,
        "prediction_checksum_after_label_mutation": repeated_checksum,
        "label_mutation_prediction_invariant": label_invariant,
        "decision_disagreement_count": len(disagreements),
        "controls_complete": controls_complete,
        "verifier_is_oracle": True,
        "oracle_distinct": False,
        "independent_semantic_audit_passed": passed,
        "cold_audit_rows": audit_rows,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the child response only after complete serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Read one frozen request and write the independent response."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print("PHASE AUDIT START raw prediction and semantic controls", flush=True)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = run_audit(request)
    _atomic_write_json(args.output, result)
    print(
        "PHASE AUDIT END "
        f"units={len(result['cold_audit_rows'])} "
        f"disagreements={result['decision_disagreement_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
