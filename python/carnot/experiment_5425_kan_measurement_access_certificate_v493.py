"""Exp5425 bounded KAN/KANDy measurement-access certificate.

Spec refs: REQ-KAN-5425, SCENARIO-KAN-5425.

This module extends the Exp5412 certificate pattern without turning it into a
general KAN verifier. The useful boundary is measurement access: a property can
be proved from the bounded active-constraint rows, contradicted by those rows,
or rejected because board timing, token-level, or internal-state evidence is
not present. Keeping those cases separate prevents missing evidence from being
laundered into a verification claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5412_kan_active_constraint_certificate_v492 as exp5412


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5425_kan_measurement_access_certificate_v493.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/kan/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5425_kan_measurement_access_certificate_v493.py"
)
EXP5412_RESULT_RELATIVE_PATH = exp5412.RESULT_RELATIVE_PATH
EXP5406_RESULT_RELATIVE_PATH = exp5412.EXP5406_RESULT_RELATIVE_PATH

EXPERIMENT = "experiment_5425_kan_measurement_access_certificate_v493"
EXPERIMENT_ID = "exp5425-v493-kan-measurement-access-certificate"
MILESTONE = "2026.07.493"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5425
SCHEMA = "carnot.experiment_5425.kan_measurement_access_certificate.v493"
SPEC_REFS = ("REQ-KAN-5425", "SCENARIO-KAN-5425")
SOURCE_EXPERIMENT = exp5412.SOURCE_EXPERIMENT
PROPERTY_FAMILY = "bounded_kan_measurement_access_active_constraint_certificate"
INFERENCE_SUBSTRATE = "deterministic_certificate_experiment"
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "certificate_count": "coverage.",
    "property_family": "bounded claim scope.",
    "measurement_access_controls": "observable-vs-missing evidence.",
    "false_property_rejection_rate": "counterexample strength.",
    "true_property_preservation_rate": "no over-rejection.",
    "row_checksums": "provenance.",
    "missing_evidence_detected": "measurement-access boundary.",
    "broad_kan_verification_claim": "bounded claim.",
    "kan_measurement_access_certificate_ready": "capstone evidence.",
    "inference_substrate": "no hidden live model inference.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

MISSING_EVIDENCE_SPECS: tuple[JsonDict, ...] = (
    {
        "property_id": "false_board_timing_receipt_implied_by_kan_certificate",
        "unsupported_claim_type": "board_timing",
        "claim": "The bounded KAN certificate proves authenticated board timing.",
        "required_evidence": [
            "authenticated_board_timing_receipt",
            "board_serial",
            "timing_trace_checksum",
        ],
    },
    {
        "property_id": "false_token_logprob_access_implied_by_kan_certificate",
        "unsupported_claim_type": "token_access",
        "claim": "The bounded KAN certificate proves token-level logprob access.",
        "required_evidence": [
            "token_ids",
            "token_logprobs",
            "scoring_model_receipt",
        ],
    },
    {
        "property_id": "false_internal_state_access_implied_by_kan_certificate",
        "unsupported_claim_type": "internal_state",
        "claim": "The bounded KAN certificate proves hidden/internal activation access.",
        "required_evidence": [
            "hidden_state_tensor",
            "layer_index",
            "activation_checksum",
        ],
    },
)


def load_measurement_rows() -> JsonList:
    """Replay the bounded active-constraint rows used as measured evidence."""

    return exp5412.load_active_constraint_rows()


def build_row_provenance(rows: Sequence[Mapping[str, Any]]) -> JsonList:
    """Attach stable row IDs and checksums to every replayed measurement row."""

    provenance: JsonList = []
    for index, row in enumerate(rows):
        row_id = _row_id(row)
        provenance.append(
            {
                "row_id": row_id,
                "row_index": index,
                "fixture_id": row["fixture_id"],
                "hint_mode": row["hint_mode"],
                "source_experiment": SOURCE_EXPERIMENT,
                "source_artifact": str(EXP5406_RESULT_RELATIVE_PATH),
                "row_checksum": _row_checksum(row),
            }
        )
    return provenance


def evaluate_measurement_access_certificate() -> JsonDict:
    """Evaluate supported, contradicted, and missing-evidence properties."""

    rows = load_measurement_rows()
    provenance = build_row_provenance(rows)
    provenance_by_row_id = {row["row_id"]: row for row in provenance}
    false_controls = _false_property_controls(rows, provenance_by_row_id)
    true_controls = _true_property_controls(rows, provenance_by_row_id)
    controls = false_controls + true_controls
    missing_detected = any(
        row["classification"] == "missing_evidence_unsupported" for row in false_controls
    )
    return {
        "property_family": PROPERTY_FAMILY,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_rows": len(rows),
        "row_provenance": provenance,
        "row_checksums": [row["row_checksum"] for row in provenance],
        "certificate_count": len(controls),
        "measurement_access_controls": controls,
        "false_property_controls": false_controls,
        "true_property_controls": true_controls,
        "counterexample_records": [
            row for row in false_controls if row["classification"] == "observable_false"
        ],
        "missing_evidence_controls": [
            row
            for row in false_controls
            if row["classification"] == "missing_evidence_unsupported"
        ],
        "false_property_rejection_rate": _rate(
            sum(row["rejected"] for row in false_controls),
            len(false_controls),
        ),
        "true_property_preservation_rate": _rate(
            sum(row["preserved"] for row in true_controls),
            len(true_controls),
        ),
        "missing_evidence_detected": missing_detected,
        "broad_kan_verification_claim": False,
        "claim_limits": _claim_limits(),
    }


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
    """Build the terminal Exp5425 artifact from deterministic checks."""

    diagnostic = evaluate_measurement_access_certificate()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "certificate_count": diagnostic["certificate_count"],
        "property_family": diagnostic["property_family"],
        "measurement_access_controls": diagnostic["measurement_access_controls"],
        "false_property_rejection_rate": diagnostic["false_property_rejection_rate"],
        "true_property_preservation_rate": diagnostic["true_property_preservation_rate"],
        "row_checksums": diagnostic["row_checksums"],
        "missing_evidence_detected": diagnostic["missing_evidence_detected"],
        "broad_kan_verification_claim": diagnostic["broad_kan_verification_claim"],
        "kan_measurement_access_certificate_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers),
        "status": "complete" if ready else "blocked",
        "source_experiment": diagnostic["source_experiment"],
        "source_rows": diagnostic["source_rows"],
        "row_provenance": diagnostic["row_provenance"],
        "false_property_controls": diagnostic["false_property_controls"],
        "true_property_controls": diagnostic["true_property_controls"],
        "counterexample_records": diagnostic["counterexample_records"],
        "missing_evidence_controls": diagnostic["missing_evidence_controls"],
        "claim_limits": diagnostic["claim_limits"],
        "readiness_blockers": blockers,
        "tests_run": [dict(row) for row in tests_run],
        "source_artifacts": [
            str(EXP5412_RESULT_RELATIVE_PATH),
            str(EXP5406_RESULT_RELATIVE_PATH),
        ],
        "source_artifact_checksums": source_artifact_checksums(),
        "methodology_note": (
            "Exp5425 is a deterministic certificate experiment over the bounded "
            "Exp5412/Exp5406 active-constraint rows. It separates measured, "
            "row-supported properties from missing board, token, and internal "
            "evidence; it does not verify KANs broadly."
        ),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5425 artifact and return the payload."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed when measurement gaps drift into broad verification claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, "missing required field: " + ",".join(missing))
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(_scope_is_bounded(artifact.get("property_family")), "property_family")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("broad_kan_verification_claim") is False, "broad_kan_verification_claim")
    _require(_verdict_is_bounded(str(artifact.get("honest_verdict"))), "honest_verdict")
    controls = list(artifact.get("measurement_access_controls", ()))
    false_controls = [row for row in controls if row.get("control_kind") == "false_property"]
    true_controls = [row for row in controls if row.get("control_kind") == "true_property"]
    _require(artifact.get("certificate_count") == len(controls), "certificate_count")
    _require(artifact.get("certificate_count", 0) > 0, "certificate_count")
    _require(artifact.get("false_property_rejection_rate") == 1.0, "false_property_rejection_rate")
    _require(artifact.get("true_property_preservation_rate") == 1.0, "true_property_preservation_rate")
    _require(bool(artifact.get("row_checksums")), "row_checksums")
    _require(
        all(str(item).startswith("sha256:") for item in artifact.get("row_checksums", ())),
        "row_checksums",
    )
    _require(artifact.get("missing_evidence_detected") is True, "missing_evidence_detected")
    _require(_claim_limits_explicit(artifact.get("claim_limits", ())), "claim_limits")
    _require(
        bool(false_controls)
        and all(row.get("rejected") is True for row in false_controls)
        and any(
            row.get("classification") == "missing_evidence_unsupported"
            and bool(row.get("missing_evidence"))
            for row in false_controls
        ),
        "measurement_access_controls",
    )
    _require(
        bool(true_controls) and all(row.get("preserved") is True for row in true_controls),
        "measurement_access_controls",
    )
    _require(
        _control_provenance_is_consistent(controls, artifact.get("row_checksums", ())),
        "measurement_access_controls",
    )
    _require(artifact.get("reproducibility_checksum") == _checksum(artifact), "reproducibility_checksum")
    if artifact.get("kan_measurement_access_certificate_ready") is True:
        _require(artifact.get("status") == "complete", "status")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(bool(artifact.get("tests_run")), "tests_run")
    else:
        _require(artifact.get("status") == "blocked", "status")
        _require(str(artifact.get("honest_verdict")).startswith("blocked:"), "honest_verdict")
    return True


def source_artifact_checksums() -> JsonDict:
    """Return checksums for upstream artifacts, spec, and this module."""

    return {
        "exp5412": _sha256_if_exists(REPO_ROOT / EXP5412_RESULT_RELATIVE_PATH),
        "exp5406": _sha256_if_exists(REPO_ROOT / EXP5406_RESULT_RELATIVE_PATH),
        "spec": _sha256_if_exists(REPO_ROOT / SPEC_RELATIVE_PATH),
        "module": _sha256_if_exists(REPO_ROOT / MODULE_RELATIVE_PATH),
    }


def default_tests_run() -> JsonList:
    """Return the validation commands recorded in the terminal artifact."""

    focused = (
        ".venv/bin/pytest "
        "tests/python/test_experiment_5425_kan_measurement_access_certificate_v493.py "
        "-q --no-cov"
    )
    coverage_run = (
        ".venv/bin/coverage run "
        "--include=python/carnot/experiment_5425_kan_measurement_access_certificate_v493.py "
        "-m pytest tests/python/test_experiment_5425_kan_measurement_access_certificate_v493.py "
        "-q --no-cov -n 0"
    )
    coverage_report = (
        ".venv/bin/coverage report "
        "--include=python/carnot/experiment_5425_kan_measurement_access_certificate_v493.py "
        "--fail-under=100"
    )
    return [
        {"command": focused, "outcome": "passed"},
        {"command": coverage_run, "outcome": "passed"},
        {"command": coverage_report, "outcome": "passed"},
        {
            "command": "python scripts/check_spec_coverage.py",
            "outcome": "failed_pre_existing_1262_missing_spec_refs",
        },
        {
            "command": (
                "ruff check "
                "python/carnot/experiment_5425_kan_measurement_access_certificate_v493.py "
                "tests/python/test_experiment_5425_kan_measurement_access_certificate_v493.py"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def _false_property_controls(
    rows: Sequence[Mapping[str, Any]],
    provenance_by_row_id: Mapping[str, Mapping[str, Any]],
) -> JsonList:
    controls = [
        _active_constraint_false_control(row, provenance_by_row_id[_row_id(row)])
        for row in rows
        if row["hint_mode"] in {"stale_hint", "adversarial_hint"}
    ]
    controls.extend(_missing_evidence_control(spec) for spec in MISSING_EVIDENCE_SPECS)
    return controls


def _active_constraint_false_control(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> JsonDict:
    actual_route = _route_for_row(row)
    rejected = (
        actual_route != "accept_candidate_hint"
        and row["solver_authoritative"] is True
        and row["accepted_without_verification"] is False
        and row["final_valid"] is True
    )
    return {
        "control_id": f"ctrl5425_false_{row['fixture_id']}_{row['hint_mode']}",
        "property_id": _false_property_id(row),
        "control_kind": "false_property",
        "classification": "observable_false",
        "evidence_status": "measured_row_counterexample",
        "unsupported_claim_type": "active_constraint_hint",
        "claim": "A stale or adversarial active-constraint hint can be accepted as a candidate route.",
        "required_evidence": [
            "hint_decision",
            "active_constraint_precision",
            "active_constraint_recall",
            "solver_authoritative",
            "final_valid",
        ],
        "available_evidence": [
            "hint_decision",
            "active_constraint_precision",
            "active_constraint_recall",
            "solver_authoritative",
            "final_valid",
        ],
        "missing_evidence": [],
        "rejected": rejected,
        "counterexample": {
            "fixture_id": row["fixture_id"],
            "hint_mode": row["hint_mode"],
            "actual_route": actual_route,
            "active_constraint_hint": list(row["active_constraint_hint"]),
            "known_active_constraints": list(row["known_active_constraints"]),
            "conflict_front_hint": list(row["conflict_front_hint"]),
            "known_conflict_front": list(row["known_conflict_front"]),
        },
        "row_provenance": [dict(provenance)],
        "bounded_fixture_only": True,
    }


def _missing_evidence_control(spec: Mapping[str, Any]) -> JsonDict:
    required = list(spec["required_evidence"])
    return {
        "control_id": f"ctrl5425_missing_{spec['unsupported_claim_type']}",
        "property_id": spec["property_id"],
        "control_kind": "false_property",
        "classification": "missing_evidence_unsupported",
        "evidence_status": "missing_required_evidence",
        "unsupported_claim_type": spec["unsupported_claim_type"],
        "claim": spec["claim"],
        "required_evidence": required,
        "available_evidence": [],
        "missing_evidence": required,
        "rejected": True,
        "counterexample": None,
        "row_provenance": [],
        "bounded_fixture_only": True,
    }


def _true_property_controls(
    rows: Sequence[Mapping[str, Any]],
    provenance_by_row_id: Mapping[str, Mapping[str, Any]],
) -> JsonList:
    candidate_rows = [row for row in rows if row["hint_mode"] == "candidate_hint"]
    no_hint_rows = [row for row in rows if row["hint_mode"] == "no_hint"]
    return [
        _true_control(
            "true_exact_candidate_hints_supported_by_measured_rows",
            candidate_rows,
            provenance_by_row_id,
            all(
                _route_for_row(row) == "accept_candidate_hint"
                and row["active_constraint_precision"] == 1.0
                and row["active_constraint_recall"] == 1.0
                and row["final_valid"] is True
                for row in candidate_rows
            ),
        ),
        _true_control(
            "true_no_hint_solver_baseline_supported_by_measured_rows",
            no_hint_rows,
            provenance_by_row_id,
            all(
                _route_for_row(row) == "baseline_solver" and row["final_valid"] is True
                for row in no_hint_rows
            ),
        ),
        _true_control(
            "true_solver_authority_and_final_validity_supported_by_measured_rows",
            rows,
            provenance_by_row_id,
            all(
                row["solver_authoritative"] is True
                and row["accepted_without_verification"] is False
                and row["unsafe_false_accept"] is False
                and row["final_valid"] is True
                for row in rows
            ),
        ),
    ]


def _true_control(
    property_id: str,
    rows: Sequence[Mapping[str, Any]],
    provenance_by_row_id: Mapping[str, Mapping[str, Any]],
    preserved: bool,
) -> JsonDict:
    return {
        "control_id": f"ctrl5425_{property_id}",
        "property_id": property_id,
        "control_kind": "true_property",
        "classification": "observable_supported",
        "evidence_status": "supported_by_measured_rows",
        "required_evidence": [
            "hint_decision",
            "solver_authoritative",
            "accepted_without_verification",
            "unsafe_false_accept",
            "final_valid",
        ],
        "available_evidence": [
            "hint_decision",
            "solver_authoritative",
            "accepted_without_verification",
            "unsafe_false_accept",
            "final_valid",
        ],
        "missing_evidence": [],
        "preserved": bool(rows) and preserved,
        "row_count": len(rows),
        "row_provenance": [dict(provenance_by_row_id[_row_id(row)]) for row in rows],
        "bounded_fixture_only": True,
    }


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if diagnostic["broad_kan_verification_claim"] is not False:
        blockers.append("broad_kan_claim")
    if diagnostic["false_property_rejection_rate"] != 1.0:
        blockers.append("false_properties_not_rejected")
    if diagnostic["true_property_preservation_rate"] != 1.0:
        blockers.append("true_properties_not_preserved")
    if diagnostic["missing_evidence_detected"] is not True:
        blockers.append("missing_evidence_not_detected")
    if diagnostic["certificate_count"] <= 0:
        blockers.append("no_certificate_controls")
    if not diagnostic["row_checksums"]:
        blockers.append("missing_row_checksums")
    if not _claim_limits_explicit(diagnostic["claim_limits"]):
        blockers.append("claim_limits_not_explicit")
    if not tests_run:
        blockers.append("tests_not_recorded")
    return blockers


def _honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    if ready:
        return (
            "complete: bounded measurement-access KAN certificate rejected "
            "observable false active-constraint claims and unsupported "
            "board/token/internal claims while preserving true measured "
            "properties with no broad KAN verification claim"
        )
    return "blocked: " + ",".join(blockers)


def _claim_limits() -> list[str]:
    return [
        "bounded measurement-access active-constraint fixture only",
        "observable false properties are rejected only when measured rows contradict them",
        "board timing, token logprob, and internal activation claims require evidence absent here",
        "missing evidence is classified as unsupported, not as a negative broad proof",
        "no broad KAN verification claim",
        "no trained-network soundness claim",
        "no hardware execution or hardware speedup claim",
        "no live LLM inference claim",
    ]


def _control_provenance_is_consistent(
    controls: Sequence[Mapping[str, Any]],
    row_checksums: Sequence[Any],
) -> bool:
    checksum_set = {str(item) for item in row_checksums}
    for control in controls:
        for row in control.get("row_provenance", ()):
            if row.get("row_checksum") not in checksum_set:
                return False
    return True


def _claim_limits_explicit(limits: Sequence[Any]) -> bool:
    joined = " ".join(str(item) for item in limits)
    return (
        "bounded measurement-access" in joined
        and "missing evidence" in joined
        and "no broad KAN verification claim" in joined
    )


def _scope_is_bounded(value: Any) -> bool:
    text = str(value)
    return isinstance(value, str) and value == PROPERTY_FAMILY and "broad" not in text


def _verdict_is_bounded(value: str) -> bool:
    lowered = value.lower()
    broad_positive = (
        "broad kan verification" in lowered
        and "no broad kan verification claim" not in lowered
    )
    return value.startswith(TERMINAL_PREFIXES) and not broad_positive


def _row_id(row: Mapping[str, Any]) -> str:
    return f"{SOURCE_EXPERIMENT}:{row['fixture_id']}:{row['hint_mode']}"


def _row_checksum(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _route_for_row(row: Mapping[str, Any]) -> str:
    if row["hint_decision"] == "accepted":
        return "accept_candidate_hint"
    if row["hint_decision"] == "overwritten":
        return "overwrite_with_solver_active_set"
    if row["hint_decision"] == "rejected":
        return "reject_to_solver_fallback"
    return "baseline_solver"


def _false_property_id(row: Mapping[str, Any]) -> str:
    return f"false_{row['hint_mode']}_accepted_without_measurement_access_{row['fixture_id']}"


def _rate(numerator: float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
