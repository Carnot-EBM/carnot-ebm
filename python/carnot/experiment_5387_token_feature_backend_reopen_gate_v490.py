#!/usr/bin/env python3
"""Exp5387: token/internal-feature backend reopen gate.

Spec refs: REQ-VERIFY-5387, SCENARIO-VERIFY-5387.

This module is a gate check, not a signal experiment. It reads existing local
runtime receipts and helper code to decide whether token/internal-feature
energy work can reopen. Generated text, top-logprob rows, token timing, and
helper option strings are useful receipts, but they do not become internal
feature evidence unless a live backend row exposes logits, hidden states,
attention, or intermediate-depth exits with provenance.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5387_token_feature_backend_reopen_gate_v490.json")

EXP5331_RELATIVE_PATH = Path("results/experiment_5331_internal_energy_receipt_harness_v486.json")
EXP5331_SCHEMA_RELATIVE_PATH = Path(
    "results/experiment_5331_internal_energy_receipt_schema_v486.json"
)
EXP5331_TINY_RECEIPT_RELATIVE_PATH = Path(
    "results/experiment_5331_internal_energy_tiny_receipt_v486.json"
)
EXP5353_RELATIVE_PATH = Path("results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json")
EXP5354_RELATIVE_PATH = Path("results/experiment_5354_arithmetic_carry_token_energy_v488.json")
REQUESTED_EXP5372_RELATIVE_PATH = Path(
    "results/experiment_5372_token_internal_feature_precondition_gate_v489.json"
)
CANONICAL_EXP5372_RELATIVE_PATH = Path(
    "results/experiment_5372_token_feature_precondition_gate_v489.json"
)
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

HELPER_RELATIVE_PATHS = (
    Path("python/carnot/experiment_5331_internal_energy_receipt_harness_v486.py"),
    Path("python/carnot/experiment_5353_tokenprob_feature_audit_corrigendum_v488.py"),
)

EXPERIMENT = 5387
EXPERIMENT_ID = "experiment_5387_token_feature_backend_reopen_gate_v490"
MILESTONE = "2026.07.490"
RUN_DATE = "20260708"
SCHEMA = "carnot.experiment_5387.token_feature_backend_reopen_gate.v490"
SPEC_REFS = ("REQ-VERIFY-5387", "SCENARIO-VERIFY-5387")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

TERMINAL_STATUSES = ("complete", "honest_blocked")
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete gate check or honest_blocked if required files are missing.",
    "backend_reopen_allowed": "true only if a backend exposes clean feature receipts.",
    "future_signal_allowed": (
        "true only if logits, hidden states, attention, or depth exits are available with provenance."
    ),
    "logits_available": "backend availability boolean with evidence path.",
    "hidden_states_available": "backend availability boolean with evidence path.",
    "attention_available": "backend availability boolean with evidence path.",
    "intermediate_depth_exits_available": "backend availability boolean with evidence path.",
    "clean_feature_row_provenance": "true only if rows can be traced to live runtime outputs.",
    "forbidden_claims": "list of claims that remain disallowed.",
    "no_live_signal_claim": "must be true.",
    "retired_scope_reopened": "must be false unless a clean backend feature receipt exists.",
    "honest_verdict": "one-line gate result.",
}
REQUIRED_WRAPPED_FIELDS = tuple(REQUIRED_FIELD_PRINCIPLES)
FORBIDDEN_CLAIMS = (
    "text-only energy",
    "incomplete token rows",
    "arithmetic carry signal",
    "external generated-text scoring",
    "DEX-style depth claims without depth exits",
)
FEATURE_FIELDS = (
    "logits_available",
    "hidden_states_available",
    "attention_available",
    "intermediate_depth_exits_available",
)


def value_of(value: Any) -> Any:
    """Unwrap principle-annotated fields while preserving plain JSON values."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrap_field(field: str, value: Any, *, evidence_path: Path | str | None = None) -> JsonDict:
    """Attach the required principle and optional evidence path to a field."""

    wrapped: JsonDict = {"principle": REQUIRED_FIELD_PRINCIPLES[field], "value": value}
    if evidence_path is not None:
        wrapped["evidence_path"] = str(evidence_path)
    return wrapped


def _read_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _bool_value(*values: Any) -> bool:
    return any(value_of(value) is True for value in values)


def _available_surface(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("availability") == "available":
            return True
        return any(bool(value.get(key)) for key in ("top_logits", "rows", "states", "summary"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return False


def _nested_mapping(payload: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    if not payload:
        return {}
    value = value_of(payload.get(field))
    return value if isinstance(value, Mapping) else {}


def _sequence_rows(value: Any) -> list[Any]:
    unwrapped = value_of(value)
    if isinstance(unwrapped, Sequence) and not isinstance(unwrapped, (str, bytes)):
        return list(unwrapped)
    return []


def _feature_evidence(
    *,
    field: str,
    exp5331: Mapping[str, Any],
    exp5331_schema: Mapping[str, Any],
    exp5331_tiny: Mapping[str, Any],
    exp5353: Mapping[str, Any],
) -> JsonDict:
    schema_availability = _nested_mapping(exp5331_schema, "availability")
    feature_audit = _nested_mapping(exp5353, "feature_audit")
    if field == "logits_available":
        available = bool(
            _bool_value(exp5331.get("logits_available"), schema_availability.get("logits_available"))
            or _available_surface(exp5331_tiny.get("logits"))
            or _bool_value(exp5353.get("logits_available"), feature_audit.get("logits_available"))
        )
        row_count = len(_sequence_rows(_nested_mapping(exp5331_tiny, "logits").get("top_logits")))
        return {
            "available": available,
            "evidence_path": str(EXP5331_TINY_RECEIPT_RELATIVE_PATH),
            "evidence_field": "logits",
            "row_count": row_count,
        }
    if field == "hidden_states_available":
        hidden_states = exp5331_tiny.get("hidden_states")
        available = bool(
            _available_surface(hidden_states)
            or _bool_value(exp5353.get("hidden_states_available"), feature_audit.get("hidden_states_available"))
        )
        row_count = len(_sequence_rows(hidden_states))
        return {
            "available": available,
            "evidence_path": str(EXP5331_TINY_RECEIPT_RELATIVE_PATH),
            "evidence_field": "hidden_states",
            "row_count": row_count,
            "hidden_state_proxy_is_not_counted": bool(
                _bool_value(
                    exp5331.get("hidden_state_proxy_available"),
                    schema_availability.get("hidden_state_proxy_available"),
                )
                or _available_surface(exp5331_tiny.get("hidden_state_proxy"))
            ),
        }
    if field == "attention_available":
        available = bool(
            _bool_value(exp5331.get("attention_available"), schema_availability.get("attention_available"))
            or _available_surface(exp5331_tiny.get("attention"))
            or _bool_value(exp5353.get("attention_available"), feature_audit.get("attention_available"))
        )
        row_count = len(_sequence_rows(_nested_mapping(exp5331_tiny, "attention").get("heads")))
        return {
            "available": available,
            "evidence_path": str(EXP5331_TINY_RECEIPT_RELATIVE_PATH),
            "evidence_field": "attention",
            "row_count": row_count,
        }
    depth_receipt = (
        exp5331_tiny.get("intermediate_depth_exits")
        or exp5331_tiny.get("depth_exits")
        or exp5353.get("intermediate_depth_exits")
    )
    return {
        "available": _available_surface(depth_receipt),
        "evidence_path": str(EXP5331_TINY_RECEIPT_RELATIVE_PATH),
        "evidence_field": "intermediate_depth_exits",
        "row_count": len(_sequence_rows(depth_receipt)),
    }


def _helper_inspection(root: Path) -> JsonDict:
    helper_rows: list[JsonDict] = []
    combined = ""
    for relative_path in HELPER_RELATIVE_PATHS:
        text = _read_text_if_exists(root / relative_path)
        helper_rows.append(
            {
                "path": str(relative_path),
                "present": text is not None,
                "mentions_logits": bool(text and "logit" in text.lower()),
                "mentions_hidden": bool(text and "hidden" in text.lower()),
                "mentions_attention": bool(text and "attention" in text.lower()),
                "mentions_depth_exit": bool(
                    text
                    and (
                        "intermediate_depth" in text.lower()
                        or "depth_exit" in text.lower()
                        or "depth exits" in text.lower()
                    )
                ),
            }
        )
        combined += text or ""
    lowered = combined.lower()
    return {
        "helper_rows": helper_rows,
        "logit_audit_code_present": "logit" in lowered,
        "hidden_audit_code_present": "hidden" in lowered,
        "attention_audit_code_present": "attention" in lowered,
        "intermediate_depth_exit_helper_present": (
            "intermediate_depth" in lowered or "depth_exit" in lowered or "depth exits" in lowered
        ),
    }


def _source_bundle(root: Path) -> JsonDict:
    requested_exp5372 = _read_json_if_exists(root / REQUESTED_EXP5372_RELATIVE_PATH)
    canonical_exp5372 = _read_json_if_exists(root / CANONICAL_EXP5372_RELATIVE_PATH)
    payloads = {
        str(EXP5331_RELATIVE_PATH): _read_json_if_exists(root / EXP5331_RELATIVE_PATH),
        str(EXP5331_SCHEMA_RELATIVE_PATH): _read_json_if_exists(root / EXP5331_SCHEMA_RELATIVE_PATH),
        str(EXP5331_TINY_RECEIPT_RELATIVE_PATH): _read_json_if_exists(
            root / EXP5331_TINY_RECEIPT_RELATIVE_PATH
        ),
        str(EXP5353_RELATIVE_PATH): _read_json_if_exists(root / EXP5353_RELATIVE_PATH),
        str(EXP5354_RELATIVE_PATH): _read_json_if_exists(root / EXP5354_RELATIVE_PATH),
        str(CANONICAL_EXP5372_RELATIVE_PATH): canonical_exp5372,
        str(EXCLUSION_MANIFEST_RELATIVE_PATH): _read_text_if_exists(
            root / EXCLUSION_MANIFEST_RELATIVE_PATH
        ),
    }
    return {
        "requested_exp5372": requested_exp5372,
        "canonical_exp5372": canonical_exp5372,
        "payloads": payloads,
        "missing_required_sources": [path for path, payload in payloads.items() if payload is None],
    }


def _retired_scope_reopened(manifest_text: str | None, exp5353: Mapping[str, Any]) -> bool:
    preconditions = _nested_mapping(exp5353, "preconditions_checked")
    retired_check = preconditions.get("retired_scope_check")
    if isinstance(retired_check, Mapping) and retired_check.get("retired_scope_reopened") is True:
        return True
    return bool(manifest_text and "operator_reopen_granted: true" in manifest_text)


def _token_row_summary(exp5353: Mapping[str, Any], exp5354: Mapping[str, Any]) -> JsonDict:
    feature_audit = _nested_mapping(exp5353, "feature_audit")
    exp5353_rows = _sequence_rows(exp5353.get("tokenprob_feature_rows"))
    exp5354_missing = _sequence_rows(exp5354.get("missing_feature_names"))
    return {
        "tokenprob_feature_rows_ready": value_of(exp5353.get("tokenprob_feature_rows_ready")) is True,
        "tokenprob_row_count": int(value_of(exp5353.get("tokenprob_feature_row_count")) or len(exp5353_rows)),
        "tokenprob_rows_source": "backend_completion_probabilities"
        if exp5353_rows or feature_audit.get("per_token_logprob_available") is True
        else "absent",
        "carry_token_energy_signal_ready": value_of(exp5354.get("carry_token_energy_signal_ready"))
        is True,
        "carry_feature_complete_rate": value_of(exp5354.get("feature_complete_rate")),
        "carry_missing_features": exp5354_missing,
    }


def _minimum_next_experiment(has_clean_feature: bool) -> JsonDict:
    if has_clean_feature:
        return {
            "name": "feature_receipt_positive_control",
            "required_before_signal_claim": [
                "freeze backend feature receipt schema and replay verifier",
                "run tiny positive and negative controls using the newly exposed feature",
                "prove prompt/completion split and row provenance for every feature row",
            ],
            "signal_quality_claim_allowed": False,
        }
    return {
        "name": None,
        "required_before_signal_claim": [
            "backend receipt exposing logits, hidden states, attention, or intermediate-depth exits",
            "clean row provenance from live runtime outputs",
            "feature-complete positive and negative controls",
        ],
        "signal_quality_claim_allowed": False,
    }


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _blocked_artifact(
    *,
    missing_sources: Sequence[str],
    requested_exp5372_present: bool,
    canonical_exp5372_present: bool,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    feature_matrix = {
        field: {
            "available": False,
            "evidence_path": str(EXP5331_TINY_RECEIPT_RELATIVE_PATH),
            "evidence_field": field.removesuffix("_available"),
            "row_count": 0,
        }
        for field in FEATURE_FIELDS
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "status": wrap_field("status", "honest_blocked"),
        "backend_reopen_allowed": wrap_field("backend_reopen_allowed", False),
        "future_signal_allowed": wrap_field("future_signal_allowed", False),
        "logits_available": wrap_field(
            "logits_available", False, evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH
        ),
        "hidden_states_available": wrap_field(
            "hidden_states_available", False, evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH
        ),
        "attention_available": wrap_field(
            "attention_available", False, evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH
        ),
        "intermediate_depth_exits_available": wrap_field(
            "intermediate_depth_exits_available",
            False,
            evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH,
        ),
        "clean_feature_row_provenance": wrap_field("clean_feature_row_provenance", False),
        "forbidden_claims": wrap_field("forbidden_claims", list(FORBIDDEN_CLAIMS)),
        "no_live_signal_claim": wrap_field("no_live_signal_claim", True),
        "retired_scope_reopened": wrap_field("retired_scope_reopened", False),
        "honest_verdict": wrap_field(
            "honest_verdict",
            "honest_blocked: required backend feature source evidence missing",
        ),
        "source_evidence": {
            "requested_exp5372_path": str(REQUESTED_EXP5372_RELATIVE_PATH),
            "requested_exp5372_path_present": requested_exp5372_present,
            "canonical_exp5372_path": str(CANONICAL_EXP5372_RELATIVE_PATH),
            "canonical_exp5372_path_present": canonical_exp5372_present,
        },
        "backend_feature_matrix": feature_matrix,
        "minimum_next_experiment": _minimum_next_experiment(False),
        "benchmark_execution": {
            "new_live_signal_benchmark_run": False,
            "reason": "gate check only; source evidence missing",
        },
        "missing_required_sources": list(missing_sources),
        "spec_refs": list(SPEC_REFS),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the Exp5387 gate artifact from existing receipts and helper code."""

    bundle = _source_bundle(root)
    missing_sources = list(bundle["missing_required_sources"])
    requested_exp5372_present = bundle["requested_exp5372"] is not None
    canonical_exp5372_present = bundle["canonical_exp5372"] is not None
    if missing_sources:
        artifact = _blocked_artifact(
            missing_sources=missing_sources,
            requested_exp5372_present=requested_exp5372_present,
            canonical_exp5372_present=canonical_exp5372_present,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    payloads = bundle["payloads"]
    exp5331 = payloads[str(EXP5331_RELATIVE_PATH)]
    exp5331_schema = payloads[str(EXP5331_SCHEMA_RELATIVE_PATH)]
    exp5331_tiny = payloads[str(EXP5331_TINY_RECEIPT_RELATIVE_PATH)]
    exp5353 = payloads[str(EXP5353_RELATIVE_PATH)]
    exp5354 = payloads[str(EXP5354_RELATIVE_PATH)]
    assert isinstance(exp5331, Mapping)
    assert isinstance(exp5331_schema, Mapping)
    assert isinstance(exp5331_tiny, Mapping)
    assert isinstance(exp5353, Mapping)
    assert isinstance(exp5354, Mapping)
    feature_matrix = {
        field: _feature_evidence(
            field=field,
            exp5331=exp5331,
            exp5331_schema=exp5331_schema,
            exp5331_tiny=exp5331_tiny,
            exp5353=exp5353,
        )
        for field in FEATURE_FIELDS
    }
    clean_feature_row_provenance = any(
        row["available"] and int(row.get("row_count") or 0) > 0 for row in feature_matrix.values()
    )
    retired_scope_reopened = _retired_scope_reopened(
        payloads[str(EXCLUSION_MANIFEST_RELATIVE_PATH)], exp5353
    )
    backend_reopen_allowed = clean_feature_row_provenance and not retired_scope_reopened
    future_signal_allowed = backend_reopen_allowed
    helper_inspection = _helper_inspection(root)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "status": wrap_field("status", "complete"),
        "backend_reopen_allowed": wrap_field(
            "backend_reopen_allowed",
            backend_reopen_allowed,
            evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH,
        ),
        "future_signal_allowed": wrap_field(
            "future_signal_allowed",
            future_signal_allowed,
            evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH,
        ),
        "logits_available": wrap_field(
            "logits_available",
            bool(feature_matrix["logits_available"]["available"]),
            evidence_path=feature_matrix["logits_available"]["evidence_path"],
        ),
        "hidden_states_available": wrap_field(
            "hidden_states_available",
            bool(feature_matrix["hidden_states_available"]["available"]),
            evidence_path=feature_matrix["hidden_states_available"]["evidence_path"],
        ),
        "attention_available": wrap_field(
            "attention_available",
            bool(feature_matrix["attention_available"]["available"]),
            evidence_path=feature_matrix["attention_available"]["evidence_path"],
        ),
        "intermediate_depth_exits_available": wrap_field(
            "intermediate_depth_exits_available",
            bool(feature_matrix["intermediate_depth_exits_available"]["available"]),
            evidence_path=feature_matrix["intermediate_depth_exits_available"]["evidence_path"],
        ),
        "clean_feature_row_provenance": wrap_field(
            "clean_feature_row_provenance",
            clean_feature_row_provenance,
            evidence_path=EXP5331_TINY_RECEIPT_RELATIVE_PATH,
        ),
        "forbidden_claims": wrap_field("forbidden_claims", list(FORBIDDEN_CLAIMS)),
        "no_live_signal_claim": wrap_field("no_live_signal_claim", True),
        "retired_scope_reopened": wrap_field(
            "retired_scope_reopened", retired_scope_reopened, evidence_path=EXCLUSION_MANIFEST_RELATIVE_PATH
        ),
        "honest_verdict": wrap_field(
            "honest_verdict",
            "complete: backend gate closed; no logits, hidden states, attention, or depth exits "
            "with clean provenance"
            if not backend_reopen_allowed
            else "complete: clean backend feature receipt exists; only minimum next experiment is allowed",
        ),
        "source_evidence": {
            "requested_exp5372_path": str(REQUESTED_EXP5372_RELATIVE_PATH),
            "requested_exp5372_path_present": requested_exp5372_present,
            "canonical_exp5372_path": str(CANONICAL_EXP5372_RELATIVE_PATH),
            "canonical_exp5372_path_present": canonical_exp5372_present,
            "semantic_sources_loaded": [
                str(EXP5331_RELATIVE_PATH),
                str(EXP5331_SCHEMA_RELATIVE_PATH),
                str(EXP5331_TINY_RECEIPT_RELATIVE_PATH),
                str(EXP5353_RELATIVE_PATH),
                str(EXP5354_RELATIVE_PATH),
                str(CANONICAL_EXP5372_RELATIVE_PATH),
                str(EXCLUSION_MANIFEST_RELATIVE_PATH),
            ],
            "helper_inspection": helper_inspection,
        },
        "backend_feature_matrix": feature_matrix,
        "token_row_summary": _token_row_summary(exp5353, exp5354),
        "minimum_next_experiment": _minimum_next_experiment(clean_feature_row_provenance),
        "benchmark_execution": {
            "new_live_signal_benchmark_run": False,
            "reason": "feature availability gate only",
        },
        "no_signal_quality_claim": True,
        "field_provenance": {
            field: {"principle": principle, "satisfied_by": "wrapped field value and evidence_path"}
            for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
        },
        "spec_refs": list(SPEC_REFS),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5387 artifact and reject unsupported reopen claims."""

    missing = [field for field in REQUIRED_WRAPPED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact[field]
        if not isinstance(wrapped, Mapping) or "principle" not in wrapped or "value" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")
    status = value_of(artifact["status"])
    if status not in TERMINAL_STATUSES:
        raise ValueError("status must be complete or honest_blocked")
    feature_available = any(value_of(artifact[field]) is True for field in FEATURE_FIELDS)
    clean_provenance = value_of(artifact["clean_feature_row_provenance"]) is True
    if value_of(artifact["future_signal_allowed"]) is True and not (feature_available and clean_provenance):
        raise ValueError("future_signal_allowed requires clean feature provenance")
    if value_of(artifact["backend_reopen_allowed"]) is True and not (feature_available and clean_provenance):
        raise ValueError("backend_reopen_allowed requires clean feature receipts")
    if value_of(artifact["no_live_signal_claim"]) is not True:
        raise ValueError("no_live_signal_claim must be true")
    if value_of(artifact["retired_scope_reopened"]) is True:
        raise ValueError("retired_scope_reopened must remain false for this gate")
    forbidden = set(value_of(artifact["forbidden_claims"]) or [])
    if not set(FORBIDDEN_CLAIMS).issubset(forbidden):
        raise ValueError("forbidden_claims must include all still-disallowed claim classes")
    verdict = str(value_of(artifact["honest_verdict"]))
    if not (verdict.startswith("complete:") or verdict.startswith("honest_blocked")):
        raise ValueError("honest_verdict must start with complete: or honest_blocked")
    if not artifact.get("tests_run"):
        raise ValueError("tests_run must be recorded")


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write the validated Exp5387 artifact and return it."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parse_tests_run(raw: str | None) -> list[JsonDict]:  # pragma: no cover - CLI glue.
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("--tests-run-json must decode to a list")
    return [dict(row) for row in parsed]


def main() -> None:  # pragma: no cover - CLI glue.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", default=None)
    args = parser.parse_args()
    run(
        root=args.root,
        result_path=args.result_path,
        tests_run=_parse_tests_run(args.tests_run_json),
    )


if __name__ == "__main__":  # pragma: no cover - CLI glue.
    main()
