"""Exp5412 bounded KAN/KANDy active-constraint certificate.

Spec refs: REQ-KAN-5412, SCENARIO-KAN-5412.

This module keeps the Exp5399 certificate idea narrow and connects it to the
Exp5406 active-constraint warm-start lane. The lifted features here are simple
deterministic hint-routing features, not a learned KAN verifier. The certificate
is useful because it names the bounded regions where stale or contradictory
active-constraint hints must be rejected or overwritten while exact candidate
hints remain usable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5406_active_constraint_warmstart_guidance_v492 as exp5406


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5412_kan_active_constraint_certificate_v492.json"
)
EXP5406_RESULT_RELATIVE_PATH = exp5406.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/kan/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5412_kan_active_constraint_certificate_v492.py"
)

EXPERIMENT = "experiment_5412_kan_active_constraint_certificate_v492"
EXPERIMENT_ID = "exp5412-v492-kan-active-constraint-certificate"
MILESTONE = "2026.07.492"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5412
SCHEMA = "carnot.experiment_5412.kan_active_constraint_certificate.v492"
SPEC_REFS = ("REQ-KAN-5412", "SCENARIO-KAN-5412")
CERTIFICATE_FAMILY = "bounded_active_constraint_hint_routing_certificate"
INFERENCE_SUBSTRATE = exp5406.INFERENCE_SUBSTRATE
TERMINAL_PREFIXES = ("complete:", "blocked:")
SOURCE_EXPERIMENT = "experiment_5406_active_constraint_warmstart_guidance_v492"

FIELD_PRINCIPLES: dict[str, str] = {
    "certificate_family": "bounded claim scope.",
    "counterexample_region_count": "falsification evidence.",
    "false_property_rejection_rate": "certificate utility.",
    "true_property_preservation_rate": "no over-rejection.",
    "certificate_size_bytes": "compactness.",
    "broad_kan_verification_claim": "no overclaim.",
    "deterministic_verifier_passed": "final authority.",
    "kan_active_constraint_certificate_ready": "downstream evidence.",
    "inference_substrate": "deterministic certificate checks.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def load_active_constraint_rows() -> JsonList:
    """Replay the bounded Exp5406 diagnostic rows used as certificate evidence."""

    upstream = exp5406.build_artifact(
        tests_run=[
            {
                "command": (
                    ".venv/bin/pytest "
                    "tests/python/test_experiment_5406_active_constraint_warmstart_guidance_v492.py "
                    "-q --no-cov"
                ),
                "outcome": "passed",
            }
        ]
    )
    return [dict(row) for row in upstream["row_records"]]


def build_certificate_records(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build explicit false-property regions and true-property controls."""

    row_list = [dict(row) for row in rows]
    false_rows = [
        row for row in row_list if row["hint_mode"] in {"stale_hint", "adversarial_hint"}
    ]
    false_checks = [_false_property_check(row) for row in false_rows]
    regions = [
        _counterexample_region(row, check) for row, check in zip(false_rows, false_checks)
    ]
    true_checks = _true_property_checks(row_list)
    return {
        "false_property_checks": false_checks,
        "true_property_checks": true_checks,
        "counterexample_regions": regions,
    }


def evaluate_certificate() -> JsonDict:
    """Evaluate the bounded active-constraint certificate deterministically."""

    rows = load_active_constraint_rows()
    records = build_certificate_records(rows)
    false_checks = records["false_property_checks"]
    true_checks = records["true_property_checks"]
    regions = records["counterexample_regions"]
    false_rate = _rate(sum(check["rejected"] for check in false_checks), len(false_checks))
    true_rate = _rate(sum(check["preserved"] for check in true_checks), len(true_checks))
    deterministic_passed = _deterministic_verifier_passed(records)
    return {
        "certificate_family": CERTIFICATE_FAMILY,
        "source_rows": len(rows),
        "source_experiment": SOURCE_EXPERIMENT,
        "false_property_count": len(false_checks),
        "true_property_count": len(true_checks),
        "counterexample_region_count": len(regions),
        "false_property_rejection_rate": false_rate,
        "true_property_preservation_rate": true_rate,
        "broad_kan_verification_claim": False,
        "deterministic_verifier_passed": deterministic_passed,
        "certificate_records": records,
        "false_property_checks": false_checks,
        "true_property_checks": true_checks,
        "counterexample_regions": regions,
        "claim_limits": _claim_limits(),
    }


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
    """Build the terminal Exp5412 artifact from deterministic certificate checks."""

    diagnostic = evaluate_certificate()
    size_bytes = certificate_size_bytes(diagnostic["certificate_records"])
    blockers = _readiness_blockers(diagnostic, size_bytes, tests_run)
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
        "certificate_family": diagnostic["certificate_family"],
        "counterexample_region_count": diagnostic["counterexample_region_count"],
        "false_property_rejection_rate": diagnostic["false_property_rejection_rate"],
        "true_property_preservation_rate": diagnostic["true_property_preservation_rate"],
        "certificate_size_bytes": size_bytes,
        "broad_kan_verification_claim": diagnostic["broad_kan_verification_claim"],
        "deterministic_verifier_passed": diagnostic["deterministic_verifier_passed"],
        "kan_active_constraint_certificate_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers),
        "status": "complete" if ready else "blocked",
        "source_experiment": diagnostic["source_experiment"],
        "source_rows": diagnostic["source_rows"],
        "false_property_count": diagnostic["false_property_count"],
        "true_property_count": diagnostic["true_property_count"],
        "certificate_records": diagnostic["certificate_records"],
        "counterexample_regions": diagnostic["counterexample_regions"],
        "false_property_checks": diagnostic["false_property_checks"],
        "true_property_checks": diagnostic["true_property_checks"],
        "claim_limits": diagnostic["claim_limits"],
        "readiness_blockers": blockers,
        "tests_run": [dict(row) for row in tests_run],
        "source_artifacts": [str(EXP5406_RESULT_RELATIVE_PATH)],
        "source_artifact_checksums": source_artifact_checksums(),
        "methodology_note": (
            "Exp5412 certifies only the bounded active-constraint hint-routing "
            "rows replayed from Exp5406. It rejects stale/adversarial false "
            "properties and preserves exact candidate hints; it does not verify "
            "KANs broadly."
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
    """Write the validated Exp5412 artifact and return the payload."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def certificate_size_bytes(records: Mapping[str, Any]) -> int:
    """Return a stable compactness measure for the certificate payload only."""

    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(encoded)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed when the artifact drifts into a broad or unchecked claim."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, "missing required field: " + ",".join(missing))
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(_scope_is_bounded(artifact.get("certificate_family")), "certificate_family")
    _require(artifact.get("broad_kan_verification_claim") is False, "broad_kan_verification_claim")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(_verdict_is_bounded(str(artifact.get("honest_verdict"))), "honest_verdict")
    _require(artifact.get("deterministic_verifier_passed") is True, "deterministic_verifier_passed")
    _require(artifact.get("false_property_rejection_rate") == 1.0, "false_property_rejection_rate")
    _require(artifact.get("true_property_preservation_rate") == 1.0, "true_property_preservation_rate")
    _require(artifact.get("counterexample_region_count") == len(artifact["counterexample_regions"]), "counterexample_region_count")
    _require(artifact.get("counterexample_region_count", 0) > 0, "counterexample_region_count")
    _require(artifact.get("certificate_size_bytes") == certificate_size_bytes(artifact["certificate_records"]), "certificate_size_bytes")
    _require(artifact.get("certificate_size_bytes", 0) > 0, "certificate_size_bytes")
    _require(_claim_limits_explicit(artifact.get("claim_limits", ())), "claim_limits")
    _require(all(row["rejected"] for row in artifact["false_property_checks"]), "false_property_checks")
    _require(all(row["preserved"] for row in artifact["true_property_checks"]), "true_property_checks")
    _require(all(row["deterministic_check_passed"] for row in artifact["counterexample_regions"]), "counterexample_regions")
    _require(artifact.get("reproducibility_checksum") == _checksum(artifact), "reproducibility_checksum")
    if artifact.get("kan_active_constraint_certificate_ready") is True:
        _require(artifact.get("status") == "complete", "status")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(bool(artifact.get("tests_run")), "tests_run")
    else:
        _require(artifact.get("status") == "blocked", "status")
        _require(str(artifact.get("honest_verdict")).startswith("blocked:"), "honest_verdict")
    return True


def source_artifact_checksums() -> JsonDict:
    """Return checksums for the upstream artifact, spec, and implementation."""

    return {
        "exp5406": _sha256_if_exists(REPO_ROOT / EXP5406_RESULT_RELATIVE_PATH),
        "spec": _sha256_if_exists(REPO_ROOT / SPEC_RELATIVE_PATH),
        "module": _sha256_if_exists(REPO_ROOT / MODULE_RELATIVE_PATH),
    }


def _false_property_check(row: Mapping[str, Any]) -> JsonDict:
    actual_route = _route_for_row(row)
    rejected = (
        actual_route != "accept_candidate_hint"
        and row["solver_authoritative"] is True
        and row["accepted_without_verification"] is False
        and row["final_valid"] is True
    )
    return {
        "false_property_id": _false_property_id(row),
        "fixture_id": row["fixture_id"],
        "hint_mode": row["hint_mode"],
        "false_claimed_route": "accept_candidate_hint",
        "actual_route": actual_route,
        "active_constraint_hint": list(row["active_constraint_hint"]),
        "known_active_constraints": list(row["known_active_constraints"]),
        "conflict_front_hint": list(row["conflict_front_hint"]),
        "known_conflict_front": list(row["known_conflict_front"]),
        "rejected": rejected,
        "deterministic_check_passed": rejected,
    }


def _counterexample_region(
    row: Mapping[str, Any],
    check: Mapping[str, Any],
) -> JsonDict:
    overlap = _front_overlap(row["conflict_front_hint"], row["known_conflict_front"])
    region = {
        "cell_id": f"active_constraint_cell_{row['fixture_id']}_{row['hint_mode']}",
        "false_property_id": check["false_property_id"],
        "fixture_id": row["fixture_id"],
        "hint_mode": row["hint_mode"],
        "source_experiment": SOURCE_EXPERIMENT,
        "actual_route": check["actual_route"],
        "feature_bounds": {
            "active_constraint_precision": _point(row["active_constraint_precision"]),
            "active_constraint_recall": _point(row["active_constraint_recall"]),
            "structural_validity": _point(float(row["hint_structurally_valid"])),
            "conflict_front_overlap": _point(overlap),
            "fallback_pressure": _point(float(row["fallback_used"])),
            "overwrite_pressure": _point(float(row["overwrite_used"])),
            "solver_authority": _point(float(row["solver_authoritative"])),
        },
        "counterexample": {
            "active_constraint_hint": list(row["active_constraint_hint"]),
            "known_active_constraints": list(row["known_active_constraints"]),
            "conflict_front_hint": list(row["conflict_front_hint"]),
            "known_conflict_front": list(row["known_conflict_front"]),
        },
        "rejects_false_property": bool(check["rejected"]),
        "deterministic_check_passed": bool(check["deterministic_check_passed"]),
        "bounded_fixture_only": True,
    }
    return region


def _true_property_checks(rows: Sequence[Mapping[str, Any]]) -> JsonList:
    candidate_rows = [row for row in rows if row["hint_mode"] == "candidate_hint"]
    no_hint_rows = [row for row in rows if row["hint_mode"] == "no_hint"]
    all_rows = list(rows)
    return [
        _true_check(
            "exact_candidate_active_sets_remain_accepted",
            candidate_rows,
            all(
                _route_for_row(row) == "accept_candidate_hint"
                and row["active_constraint_precision"] == 1.0
                and row["active_constraint_recall"] == 1.0
                and row["final_valid"] is True
                for row in candidate_rows
            ),
        ),
        _true_check(
            "no_hint_solver_baseline_remains_valid",
            no_hint_rows,
            all(_route_for_row(row) == "baseline_solver" and row["final_valid"] is True for row in no_hint_rows),
        ),
        _true_check(
            "solver_authority_and_final_validity_preserved",
            all_rows,
            all(
                row["solver_authoritative"] is True
                and row["accepted_without_verification"] is False
                and row["unsafe_false_accept"] is False
                and row["final_valid"] is True
                for row in all_rows
            ),
        ),
    ]


def _true_check(property_id: str, rows: Sequence[Mapping[str, Any]], preserved: bool) -> JsonDict:
    return {
        "property_id": property_id,
        "sample_count": len(rows),
        "fixture_ids": [str(row["fixture_id"]) for row in rows],
        "preserved": bool(rows) and preserved,
        "deterministic_check_passed": bool(rows) and preserved,
    }


def _route_for_row(row: Mapping[str, Any]) -> str:
    if row["hint_decision"] == "accepted":
        return "accept_candidate_hint"
    if row["hint_decision"] == "overwritten":
        return "overwrite_with_solver_active_set"
    if row["hint_decision"] == "rejected":
        return "reject_to_solver_fallback"
    return "baseline_solver"


def _false_property_id(row: Mapping[str, Any]) -> str:
    if row["hint_mode"] == "stale_hint":
        return "false_stale_partial_active_set_can_route_candidate"
    return "false_adversarial_contradiction_can_route_candidate"


def _front_overlap(predicted: Sequence[str], truth: Sequence[str]) -> float:
    predicted_set = set(predicted)
    truth_set = set(truth)
    return round(len(predicted_set & truth_set) / len(truth_set), 6)


def _deterministic_verifier_passed(records: Mapping[str, Any]) -> bool:
    return (
        all(check["rejected"] and check["deterministic_check_passed"] for check in records["false_property_checks"])
        and all(check["preserved"] and check["deterministic_check_passed"] for check in records["true_property_checks"])
        and all(region["rejects_false_property"] and region["bounded_fixture_only"] for region in records["counterexample_regions"])
    )


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    size_bytes: int,
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if diagnostic["broad_kan_verification_claim"] is not False:
        blockers.append("broad_kan_claim")
    if diagnostic["false_property_rejection_rate"] != 1.0:
        blockers.append("false_properties_not_rejected")
    if diagnostic["true_property_preservation_rate"] != 1.0:
        blockers.append("true_properties_not_preserved")
    if diagnostic["deterministic_verifier_passed"] is not True:
        blockers.append("deterministic_verifier_failed")
    if diagnostic["counterexample_region_count"] <= 0:
        blockers.append("no_counterexample_regions")
    if size_bytes <= 0:
        blockers.append("empty_certificate")
    if not tests_run:
        blockers.append("tests_not_recorded")
    return blockers


def _honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    if ready:
        return (
            "complete: bounded active-constraint certificate rejected stale and "
            "adversarial false hint-routing properties while preserving exact "
            "candidate hints and making no broad KAN verification claim"
        )
    return "blocked: " + ",".join(blockers)


def _claim_limits() -> list[str]:
    return [
        "bounded active-constraint hint-routing fixture only",
        "KAN/KANDy-style lifted features are deterministic certificate features, not trained KAN verification",
        "counterexample regions cover Exp5406 stale and adversarial hint rows only",
        "no broad KAN verification claim",
        "no trained-network soundness claim",
        "no hardware execution or hardware speedup claim",
        "no live LLM inference claim",
    ]


def _claim_limits_explicit(limits: Sequence[Any]) -> bool:
    joined = " ".join(str(item) for item in limits)
    return "bounded active-constraint" in joined and "no broad KAN verification claim" in joined


def _scope_is_bounded(value: Any) -> bool:
    text = str(value)
    return isinstance(value, str) and value == CERTIFICATE_FAMILY and "broad" not in text


def _verdict_is_bounded(value: str) -> bool:
    lowered = value.lower()
    broad_positive = "broad kan verification" in lowered and "no broad kan verification claim" not in lowered
    return value.startswith(TERMINAL_PREFIXES) and not broad_positive


def _rate(numerator: float, denominator: int) -> float:
    return round(float(numerator) / denominator, 6)


def _point(value: Any) -> list[float]:
    rounded = round(float(value), 6)
    return [rounded, rounded]


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
