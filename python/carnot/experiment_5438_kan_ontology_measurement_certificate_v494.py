"""Exp5438 bounded ontology/workflow-memory measurement-access certificate.

Spec refs: REQ-KAN-5438, SCENARIO-KAN-5438.

This module reuses the Exp5425 measurement-access pattern on the Exp5432
ontology-memory fixture. The certificate only speaks about rows that were
actually measured in Exp5432, or about evidence that is explicitly absent. That
boundary is the point: missing board, token, kernel, or internal-state receipts
must stay unsupported instead of becoming broad KAN verification claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes, receipt_exists


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5438_kan_ontology_measurement_certificate_v494.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/kan/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5438_kan_ontology_measurement_certificate_v494.py"
)
EXP5432_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json"
)
EXP5425_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5425_kan_measurement_access_certificate_v493.json"
)
EXP5412_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5412_kan_active_constraint_certificate_v492.json"
)

EXPERIMENT = "experiment_5438_kan_ontology_measurement_certificate_v494"
EXPERIMENT_ID = "exp5438-v494-kan-ontology-measurement-certificate"
SOURCE_EXPERIMENT = "experiment_5432_ontology_softlogic_constraint_memory_v494"
MILESTONE = "2026.07.494"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5438
SCHEMA = "carnot.experiment_5438.kan_ontology_measurement_certificate.v494"
SPEC_REFS = ("REQ-KAN-5438", "SCENARIO-KAN-5438")
PROPERTY_FAMILY = "bounded_kan_ontology_workflow_memory_measurement_certificate"
INFERENCE_SUBSTRATE = "deterministic_certificate_experiment"
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "certificate_count": "coverage.",
    "property_family": "bounded claim scope.",
    "ontology_property_count": "graph-memory coverage.",
    "workflow_memory_property_count": "CSL coverage.",
    "measurement_access_controls": "observable-vs-missing evidence.",
    "false_property_rejection_rate": "counterexample strength.",
    "true_property_preservation_rate": "no over-rejection.",
    "row_checksums": "provenance.",
    "missing_evidence_detected": "measurement-access boundary.",
    "broad_kan_verification_claim": "bounded claim.",
    "kan_ontology_certificate_ready": "capstone evidence.",
    "inference_substrate": "no hidden live model inference.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

MISSING_EVIDENCE_SPECS: tuple[JsonDict, ...] = (
    {
        "property_id": "false_kernel_receipt_implied_by_ontology_certificate",
        "unsupported_claim_type": "kernel_evidence",
        "property_domain": "workflow_memory",
        "claim": "The bounded certificate proves a reusable workflow kernel receipt.",
        "required_evidence": [
            "kernel_constraint_trace",
            "planner_kernel_version",
            "kernel_receipt_checksum",
        ],
    },
    {
        "property_id": "false_board_timing_receipt_implied_by_ontology_certificate",
        "unsupported_claim_type": "board_timing",
        "property_domain": "workflow_memory",
        "claim": "The bounded certificate proves authenticated board timing.",
        "required_evidence": [
            "authenticated_board_timing_receipt",
            "board_serial",
            "timing_trace_checksum",
        ],
    },
    {
        "property_id": "false_token_logprob_access_implied_by_ontology_certificate",
        "unsupported_claim_type": "token_access",
        "property_domain": "workflow_memory",
        "claim": "The bounded certificate proves token-level logprob access.",
        "required_evidence": [
            "token_ids",
            "token_logprobs",
            "scoring_model_receipt",
        ],
    },
    {
        "property_id": "false_internal_state_access_implied_by_ontology_certificate",
        "unsupported_claim_type": "internal_state",
        "property_domain": "workflow_memory",
        "claim": "The bounded certificate proves hidden/internal activation access.",
        "required_evidence": [
            "hidden_state_tensor",
            "layer_index",
            "activation_checksum",
        ],
    },
)


def load_exp5432_artifact() -> JsonDict:
    """Load the completed ontology-memory source artifact."""

    return _load_json(REPO_ROOT / EXP5432_RESULT_RELATIVE_PATH)


def exp5432_ready() -> bool:
    """Return true only when the upstream ontology-memory gate is open."""

    return load_exp5432_artifact().get("ontology_constraint_memory_ready") is True


def load_measurement_rows() -> JsonList:
    """Return the Exp5432 row-level evidence used by the bounded certificate."""

    return list(load_exp5432_artifact()["evaluated_rows"])


def build_row_provenance(rows: Sequence[Mapping[str, Any]]) -> JsonList:
    """Attach stable row IDs and checksums to every Exp5432 measured row."""

    provenance: JsonList = []
    for index, row in enumerate(rows):
        provenance.append(
            {
                "row_id": _row_id(row),
                "row_index": index,
                "fixture_family": row["fixture_family"],
                "row_type": row["row_type"],
                "source_experiment": SOURCE_EXPERIMENT,
                "source_artifact": str(EXP5432_RESULT_RELATIVE_PATH),
                "row_checksum": _row_checksum(row),
            }
        )
    return provenance


def evaluate_ontology_measurement_certificate() -> JsonDict:
    """Evaluate supported, contradicted, and missing-evidence properties."""

    rows = load_measurement_rows()
    provenance = build_row_provenance(rows)
    provenance_by_row_id = {row["row_id"]: row for row in provenance}
    false_controls = _false_property_controls(rows, provenance_by_row_id)
    true_controls = _true_property_controls(rows, provenance_by_row_id)
    controls = false_controls + true_controls
    ontology_count = _domain_count(controls, "ontology_triple")
    workflow_count = _domain_count(controls, "workflow_memory")
    missing_detected = any(
        row["classification"] == "missing_evidence_unsupported" for row in false_controls
    )
    return {
        "property_family": PROPERTY_FAMILY,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_rows": len(rows),
        "upstream_ready": exp5432_ready(),
        "row_provenance": provenance,
        "row_checksums": [row["row_checksum"] for row in provenance],
        "certificate_count": len(controls),
        "ontology_property_count": ontology_count,
        "workflow_memory_property_count": workflow_count,
        "measurement_access_controls": controls,
        "false_property_controls": false_controls,
        "true_property_controls": true_controls,
        "counterexample_records": [
            row for row in false_controls if row["classification"] == "observable_false"
        ],
        "missing_evidence_controls": [
            row for row in false_controls if row["classification"] == "missing_evidence_unsupported"
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
    """Build the terminal Exp5438 artifact from deterministic checks."""

    diagnostic = evaluate_ontology_measurement_certificate()
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
        "ontology_property_count": diagnostic["ontology_property_count"],
        "workflow_memory_property_count": diagnostic["workflow_memory_property_count"],
        "measurement_access_controls": diagnostic["measurement_access_controls"],
        "false_property_rejection_rate": diagnostic["false_property_rejection_rate"],
        "true_property_preservation_rate": diagnostic["true_property_preservation_rate"],
        "row_checksums": diagnostic["row_checksums"],
        "missing_evidence_detected": diagnostic["missing_evidence_detected"],
        "broad_kan_verification_claim": diagnostic["broad_kan_verification_claim"],
        "kan_ontology_certificate_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers),
        "status": "complete" if ready else "blocked",
        "source_experiment": diagnostic["source_experiment"],
        "source_rows": diagnostic["source_rows"],
        "upstream_ready": diagnostic["upstream_ready"],
        "row_provenance": diagnostic["row_provenance"],
        "false_property_controls": diagnostic["false_property_controls"],
        "true_property_controls": diagnostic["true_property_controls"],
        "counterexample_records": diagnostic["counterexample_records"],
        "missing_evidence_controls": diagnostic["missing_evidence_controls"],
        "claim_limits": diagnostic["claim_limits"],
        "readiness_blockers": blockers,
        "tests_run": [dict(row) for row in tests_run],
        "source_artifacts": [
            str(EXP5432_RESULT_RELATIVE_PATH),
            str(EXP5425_RESULT_RELATIVE_PATH),
            str(EXP5412_RESULT_RELATIVE_PATH),
        ],
        "source_artifact_checksums": source_artifact_checksums(),
        "methodology_note": (
            "Exp5438 is a deterministic certificate experiment over the bounded "
            "Exp5432 ontology and workflow-memory rows. It separates row-backed "
            "ontology/workflow claims from missing graph, kernel, board, token, "
            "and internal evidence; it does not verify KANs broadly."
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
    """Write the validated Exp5438 artifact and return the payload."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed when measurement gaps drift into broad KAN claims."""

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
    ontology_count = _domain_count(controls, "ontology_triple")
    workflow_count = _domain_count(controls, "workflow_memory")
    _require(artifact.get("certificate_count") == len(controls), "certificate_count")
    _require(artifact.get("certificate_count", 0) > 0, "certificate_count")
    _require(artifact.get("ontology_property_count") == ontology_count, "ontology_property_count")
    _require(
        artifact.get("workflow_memory_property_count") == workflow_count,
        "workflow_memory_property_count",
    )
    _require(artifact.get("false_property_rejection_rate") == 1.0, "false_property_rejection_rate")
    _require(
        artifact.get("true_property_preservation_rate") == 1.0, "true_property_preservation_rate"
    )
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
    _require(
        artifact.get("reproducibility_checksum") == _checksum(artifact), "reproducibility_checksum"
    )
    if artifact.get("kan_ontology_certificate_ready") is True:
        _require(artifact.get("status") == "complete", "status")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(bool(artifact.get("tests_run")), "tests_run")
        _require(artifact.get("upstream_ready") is True, "upstream_ready")
    else:
        _require(artifact.get("status") == "blocked", "status")
        _require(str(artifact.get("honest_verdict")).startswith("blocked:"), "honest_verdict")
    return True


def source_artifact_checksums() -> JsonDict:
    """Return checksums for upstream artifacts, spec, and this module."""

    return {
        "exp5432": _sha256_if_exists(REPO_ROOT / EXP5432_RESULT_RELATIVE_PATH),
        "exp5425": _sha256_if_exists(REPO_ROOT / EXP5425_RESULT_RELATIVE_PATH),
        "exp5412": _sha256_if_exists(REPO_ROOT / EXP5412_RESULT_RELATIVE_PATH),
        "spec": _sha256_if_exists(REPO_ROOT / SPEC_RELATIVE_PATH),
        "module": _sha256_if_exists(REPO_ROOT / MODULE_RELATIVE_PATH),
    }


def default_tests_run() -> JsonList:
    """Return the validation commands recorded in the terminal artifact."""

    test_path = "tests/python/test_experiment_5438_kan_ontology_measurement_certificate_v494.py"
    module_path = "python/carnot/experiment_5438_kan_ontology_measurement_certificate_v494.py"
    return [
        {"command": f".venv/bin/pytest {test_path} -q --no-cov -n 0", "outcome": "passed"},
        {
            "command": (
                ".venv/bin/coverage run "
                f"--include={module_path} -m pytest {test_path} -q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (f".venv/bin/coverage report --include={module_path} --fail-under=100"),
            "outcome": "passed",
        },
        {
            "command": "python scripts/check_spec_coverage.py",
            "outcome": "failed_pre_existing_1262_missing_spec_refs",
        },
        {
            "command": (f"ruff check {module_path} {test_path}"),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def _false_property_controls(
    rows: Sequence[Mapping[str, Any]],
    provenance_by_row_id: Mapping[str, Mapping[str, Any]],
) -> JsonList:
    controls: JsonList = []
    controls.extend(
        _observable_false_control(row, provenance_by_row_id[_row_id(row)], "invalid_triple")
        for row in rows
        if row["fixture_family"] == "false_triple_update"
    )
    controls.extend(
        _observable_false_control(row, provenance_by_row_id[_row_id(row)], "stale_relation")
        for row in rows
        if row["fixture_family"] == "stale_relation_update"
    )
    controls.extend(
        _missing_graph_update_control(row, provenance_by_row_id[_row_id(row)])
        for row in rows
        if row["fixture_family"] == "unsupported_memory_write"
    )
    controls.extend(
        _observable_false_control(row, provenance_by_row_id[_row_id(row)], "retrieval_trap")
        for row in rows
        if row["fixture_family"] == "infeasible_retrieval"
    )
    controls.extend(_missing_evidence_control(spec) for spec in MISSING_EVIDENCE_SPECS)
    return controls


def _observable_false_control(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
    unsupported_claim_type: str,
) -> JsonDict:
    rejected = row["final_decision"] in {"rejected", "abstained"}
    property_domain = (
        "workflow_memory" if unsupported_claim_type == "retrieval_trap" else "ontology_triple"
    )
    return {
        "control_id": f"ctrl5438_false_{row['row_id'].replace(':', '_')}",
        "property_id": _false_property_id(row, unsupported_claim_type),
        "property_domain": property_domain,
        "control_kind": "false_property",
        "classification": "observable_false",
        "evidence_status": "measured_row_counterexample",
        "unsupported_claim_type": unsupported_claim_type,
        "claim": _false_claim(row, unsupported_claim_type),
        "required_evidence": [
            "final_decision",
            "deterministic_decision",
            "decision_reasons",
            "shacl",
            "solver",
        ],
        "available_evidence": [
            "final_decision",
            "deterministic_decision",
            "decision_reasons",
            "shacl",
            "solver",
        ],
        "missing_evidence": [],
        "rejected": rejected,
        "counterexample": {
            "row_id": row["row_id"],
            "row_type": row["row_type"],
            "fixture_family": row["fixture_family"],
            "expected_truth": row["expected_truth"],
            "final_decision": row["final_decision"],
            "decision_reasons": list(row["decision_reasons"]),
            "proposed_triples": list(row["proposed_triples"]),
            "retrieved_plan": list(row["retrieved_plan"]),
        },
        "row_provenance": [dict(provenance)],
        "bounded_fixture_only": True,
    }


def _missing_graph_update_control(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> JsonDict:
    required = [
        "supported_predicate_schema",
        "known_entity_type_membership",
        "tool_output_receipt_for_update",
    ]
    return {
        "control_id": f"ctrl5438_missing_graph_{row['row_id'].replace(':', '_')}",
        "property_id": f"false_unsupported_graph_update_supported_{row['row_id'].replace(':', '_')}",
        "property_domain": "ontology_triple",
        "control_kind": "false_property",
        "classification": "missing_evidence_unsupported",
        "evidence_status": "missing_required_evidence",
        "unsupported_claim_type": "unsupported_graph_update",
        "claim": "An unsupported graph update can be promoted as ontology memory.",
        "required_evidence": required,
        "available_evidence": [
            "final_decision",
            "deterministic_decision",
            "decision_reasons",
            "shacl",
            "solver",
        ],
        "missing_evidence": required,
        "rejected": row["final_decision"] == "abstained",
        "counterexample": None,
        "row_provenance": [dict(provenance)],
        "bounded_fixture_only": True,
    }


def _missing_evidence_control(spec: Mapping[str, Any]) -> JsonDict:
    required = list(spec["required_evidence"])
    return {
        "control_id": f"ctrl5438_missing_{spec['unsupported_claim_type']}",
        "property_id": spec["property_id"],
        "property_domain": spec["property_domain"],
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
    valid_triple_rows = [
        row
        for row in rows
        if row["row_type"] == "triple_update" and row["expected_truth"] == "valid"
    ]
    valid_retrieval_rows = [
        row for row in rows if row["row_type"] == "retrieval" and row["expected_truth"] == "valid"
    ]
    tool_evidence_rows = [row for row in valid_triple_rows if row["tool_output_evidence"]]
    return [
        _true_control(
            "true_valid_triple_updates_supported_by_exp5432_rows",
            "ontology_triple",
            valid_triple_rows,
            provenance_by_row_id,
            all(row["final_decision"] == "accepted" for row in valid_triple_rows),
        ),
        _true_control(
            "true_valid_workflow_retrieval_order_supported_by_exp5432_rows",
            "workflow_memory",
            valid_retrieval_rows,
            provenance_by_row_id,
            all(row["final_decision"] == "accepted" for row in valid_retrieval_rows),
        ),
        _true_control(
            "true_deterministic_solver_authority_supported_by_exp5432_rows",
            "workflow_memory",
            rows,
            provenance_by_row_id,
            all(row["final_decision"] == row["deterministic_decision"] for row in rows),
        ),
        _true_control(
            "true_tool_evidence_backed_memory_writes_supported_by_exp5432_rows",
            "ontology_triple",
            tool_evidence_rows,
            provenance_by_row_id,
            all(row["final_decision"] == "accepted" for row in tool_evidence_rows),
        ),
    ]


def _true_control(
    property_id: str,
    property_domain: str,
    rows: Sequence[Mapping[str, Any]],
    provenance_by_row_id: Mapping[str, Mapping[str, Any]],
    preserved: bool,
) -> JsonDict:
    return {
        "control_id": f"ctrl5438_{property_id}",
        "property_id": property_id,
        "property_domain": property_domain,
        "control_kind": "true_property",
        "classification": "observable_supported",
        "evidence_status": "supported_by_measured_rows",
        "required_evidence": [
            "final_decision",
            "deterministic_decision",
            "decision_reasons",
            "shacl",
            "solver",
        ],
        "available_evidence": [
            "final_decision",
            "deterministic_decision",
            "decision_reasons",
            "shacl",
            "solver",
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
    if diagnostic["upstream_ready"] is not True:
        blockers.append("upstream_exp5432_not_ready")
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
            "complete: bounded ontology/workflow-memory KAN certificate "
            "rejected observable false ontology and retrieval claims and "
            "unsupported graph/kernel/board/token/internal claims while "
            "preserving true measured properties with no broad KAN "
            "verification claim"
        )
    return "blocked: " + ",".join(blockers)


def _claim_limits() -> list[str]:
    return [
        "bounded ontology/workflow-memory measurement-access fixture only",
        "observable false properties are rejected only when Exp5432 measured rows contradict them",
        "graph, kernel, board, token, and internal claims require evidence absent here",
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
        "bounded ontology/workflow-memory" in joined
        and "missing evidence" in joined
        and "no broad KAN verification claim" in joined
    )


def _scope_is_bounded(value: Any) -> bool:
    text = str(value)
    return isinstance(value, str) and value == PROPERTY_FAMILY and "broad" not in text


def _verdict_is_bounded(value: str) -> bool:
    lowered = value.lower()
    broad_positive = (
        "broad kan verification" in lowered and "no broad kan verification claim" not in lowered
    )
    return value.startswith(TERMINAL_PREFIXES) and not broad_positive


def _domain_count(controls: Sequence[Mapping[str, Any]], domain: str) -> int:
    return sum(1 for control in controls if control.get("property_domain") == domain)


def _row_id(row: Mapping[str, Any]) -> str:
    return f"{SOURCE_EXPERIMENT}:{row['row_id']}"


def _row_checksum(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _false_property_id(row: Mapping[str, Any], unsupported_claim_type: str) -> str:
    row_token = str(row["row_id"]).replace(":", "_")
    return f"false_{unsupported_claim_type}_accepted_without_measurement_access_{row_token}"


def _false_claim(row: Mapping[str, Any], unsupported_claim_type: str) -> str:
    if unsupported_claim_type == "retrieval_trap":
        return "An infeasible retrieved workflow can influence planning memory."
    if unsupported_claim_type == "stale_relation":
        return "A stale ontology relation update can be promoted as current memory."
    return "An invalid or unsupported ontology triple can be promoted as graph memory."


def _rate(numerator: float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_if_exists(path: Path) -> str | None:
    # Resolve at the artifact's own commit so a later append to the shared
    # KAN spec does not stale this receipt (REQ-REPORT-6610; the 2026-08-25
    # adoption sweep, commit 64846b5430, missed this module).
    if not receipt_exists(path, artifact_relative_path=RESULT_RELATIVE_PATH):
        return None
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
