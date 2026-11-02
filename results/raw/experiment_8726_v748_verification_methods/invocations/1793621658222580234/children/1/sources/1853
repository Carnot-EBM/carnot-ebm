"""Exp5552 automaton/schema row-completion receipt for hard/soft rows.

Spec refs: REQ-VERIFY-5552, SCENARIO-VERIFY-5552.

This module builds a deterministic table automaton around the Exp5512
hard/soft candidate schema. The states are the set of required model/instance
rows completed so far, and the single terminal state is the table where every
required row has been accepted by the repository parser and exact validators.
It does not invoke an LLM, load a model, or claim SOTA quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5513_sota_hard_soft_structured_panel as panel5513
from carnot import experiment_5539_gram2token_grammar_table_preflight as gate5539
from carnot import experiment_5540_sota_hard_soft_live_panel_v3 as panel5540


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5552_automaton_schema_row_completion_receipt.json")
UPSTREAM_GRAMMAR_PREFLIGHT = gate5539.RESULT_RELATIVE_PATH
UPSTREAM_PANEL_PATH = panel5540.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5552.automaton_schema_row_completion_receipt.v503"
EXPERIMENT = 5552
EXPERIMENT_ID = "exp5552-automaton-schema-row-completion-receipt"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5552
INFERENCE_SUBSTRATE = "deterministic_automaton_no_llm"
AUTOMATON_BACKEND = "deterministic_completed_row_set_dfa_over_exp5512_schema"
SPEC_REFS = ("REQ-VERIFY-5552", "SCENARIO-VERIFY-5552")

REQUIRED_ARTIFACT_FIELDS = (
    "upstream_grammar_preflight",
    "upstream_panel_path",
    "llm_invoked",
    "no_model_specs_required",
    "automaton_backend",
    "schema_hash",
    "required_rows",
    "reachable_state_count",
    "terminal_state_count",
    "dead_end_transition_count",
    "valid_fixture_acceptance_rate",
    "invalid_fixture_rejection_rate",
    "row_completion_support_rate",
    "missing_row_risk",
    "local_mask_bias_diagnostic",
    "automaton_row_completion_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5552_automaton_schema_row_completion_receipt.py",
    "tests/python/test_experiment_5539_gram2token_grammar_table_preflight.py",
    "tests/python/test_experiment_5540_sota_hard_soft_live_panel_v3.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "upstream_grammar_preflight": "Pins grammar reachability evidence to the Exp5539 preflight artifact.",
    "upstream_panel_path": "Pins the prior hard/soft panel whose missing rows motivate this gate.",
    "llm_invoked": "Must remain false because this receipt is deterministic preflight evidence.",
    "no_model_specs_required": "Confirms row-completion analysis needs no model loading or model specs.",
    "automaton_backend": "Names the deterministic completed-row-set DFA used for table completion.",
    "schema_hash": "Pins the receipt to the Exp5512 hard/soft candidate schema.",
    "required_rows": "Fixes every required model/instance row before downstream live inference.",
    "reachable_state_count": "Records how many completed-row states the automaton can reach.",
    "terminal_state_count": "Counts accepting states where all required rows are complete.",
    "dead_end_transition_count": "Counts invalid automaton transitions that cannot complete the table.",
    "valid_fixture_acceptance_rate": "Reuses valid Exp5539 fixture evidence for parser and exact-validator handoff.",
    "invalid_fixture_rejection_rate": "Reuses malformed and semantic rejection evidence before live inference.",
    "row_completion_support_rate": "Measures required rows that are reachable and locally proposal-supported.",
    "missing_row_risk": "Turns row-completion support into a downstream missing-row risk label.",
    "local_mask_bias_diagnostic": "Lists reachable rows missing from the local mask/proposal path.",
    "automaton_row_completion_ready": "Opens only when all required rows are reachable and proposal-supported.",
    "tests_added_or_reused": "Links the receipt to focused row-completion and reused parser tests.",
    "field_principles": "Explains why every headline and gate field exists.",
    "inference_substrate": "Declares deterministic automaton/schema analysis with no LLM.",
    "honest_verdict": "Provides a terminal evidence boundary without a live SOTA quality claim.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes are stable and reviewable."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def required_rows_from_panel(
    panel_artifact: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Return the fixed model/instance row denominator for the candidate table."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    payloads = positive.build_fixture_candidate_payloads(fixture_payload)
    model_ids = _required_model_ids(panel_artifact)
    rows: list[JsonDict] = []
    for model_id in model_ids:
        for payload in payloads:
            instance_id = str(payload["instance_id"])
            classified = positive.classify_candidate_payload(payload, fixture=fixture_payload)
            rows.append(
                {
                    "row_key": row_key(model_id, instance_id),
                    "model_hf_id": model_id,
                    "instance_id": instance_id,
                    "candidate_id": str(payload["candidate_id"]),
                    "expected_status": str(payload["validator_target"]["expected_status"]),
                    "candidate_schema_version": positive.CANDIDATE_SCHEMA_VERSION,
                    "schema_reachable": _accepted(classified),
                    "fixture_payload_hash": positive.sha256_json(payload),
                }
            )
    return rows


def build_completion_automaton(required_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the compact completed-row-set DFA summary for required rows."""

    row_count = len(required_rows)
    reachable_state_count = 1 << row_count
    duplicate_dead_ends = row_count * (1 << (row_count - 1)) if row_count else 0
    schema_dead_ends = reachable_state_count
    malformed_dead_ends = reachable_state_count
    invalid_enum_dead_ends = reachable_state_count
    unknown_row_dead_ends = reachable_state_count
    premature_end_dead_ends = max(0, reachable_state_count - 1)
    templates = [
        {"reason": "duplicate_required_row", "count": duplicate_dead_ends},
        {"reason": "schema_invalid_row", "count": schema_dead_ends},
        {"reason": "malformed_row", "count": malformed_dead_ends},
        {"reason": "invalid_enum_value", "count": invalid_enum_dead_ends},
        {"reason": "unknown_required_row", "count": unknown_row_dead_ends},
        {"reason": "missing_required_row", "count": premature_end_dead_ends},
    ]
    return {
        "reachable_state_count": reachable_state_count,
        "terminal_state_count": 1 if row_count else 0,
        "dead_end_transition_count": sum(int(row["count"]) for row in templates),
        "terminal_states": [
            {
                "completed_row_keys": [str(row["row_key"]) for row in required_rows],
                "accepting": bool(row_count),
            }
        ],
        "dead_end_transition_templates": templates,
    }


def proposal_records_from_text(model_hf_id: str, text: str) -> list[JsonDict]:
    """Extract proposal rows from model-style text through the existing parser."""

    parsed = panel5513.extract_candidate_payloads(text)
    records = [
        {
            "model_hf_id": model_hf_id,
            "parsed_payload": dict(payload),
            "production_mode": "grammar_masking",
        }
        for payload in parsed.get("candidate_payloads", [])
        if isinstance(payload, Mapping)
    ]
    records.extend(
        {
            "model_hf_id": model_hf_id,
            "parse_failure": dict(row),
            "production_mode": "grammar_masking",
        }
        for row in parsed.get("parse_failures", [])
        if isinstance(row, Mapping)
    )
    return records


def proposal_records_from_panel(panel_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Convert prior Exp5540 panel summaries into local proposal-support rows."""

    return [
        {
            "model_hf_id": str(row.get("model_hf_id", "")),
            "classified_row": dict(row),
            "production_mode": str(row.get("production_mode", "grammar_masking")),
        }
        for row in panel_artifact.get("panel_rows", [])
        if isinstance(row, Mapping)
    ]


def evaluate_completion(
    required_rows: Sequence[Mapping[str, Any]],
    proposal_records: Sequence[Mapping[str, Any]],
    *,
    fixture: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run proposal rows through the deterministic table-completion automaton."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    required_by_key = {str(row["row_key"]): dict(row) for row in required_rows}
    accepted_keys: set[str] = set()
    accepted_rows: list[JsonDict] = []
    dead_ends: list[JsonDict] = []
    for record in proposal_records:
        if "parse_failure" in record:
            dead_ends.append(_dead_end("malformed_row", record, detail=record["parse_failure"]))
            continue
        payload = record.get("parsed_payload")
        classified_row = record.get("classified_row")
        if isinstance(payload, Mapping):
            result = _evaluate_payload_record(record, payload, required_by_key, fixture_payload)
        elif isinstance(classified_row, Mapping):
            result = _evaluate_classified_record(record, classified_row, required_by_key)
        else:
            result = {"accepted": False, "dead_end": _dead_end("malformed_row", record)}
        if result["accepted"] is True:
            key = str(result["row_key"])
            if key in accepted_keys:
                dead_ends.append(_dead_end("duplicate_required_row", record, row_key=key))
                continue
            accepted_keys.add(key)
            accepted_rows.append(dict(result["row"]))
        else:
            dead_ends.append(dict(result["dead_end"]))

    missing_rows = [
        row
        for row in required_rows
        if row.get("schema_reachable") is True and str(row["row_key"]) not in accepted_keys
    ]
    dead_ends.extend(
        _dead_end(
            "missing_required_row",
            {"model_hf_id": row["model_hf_id"], "instance_id": row["instance_id"]},
            row_key=str(row["row_key"]),
        )
        for row in missing_rows
    )
    support_rate = _rate(len(accepted_keys), len(required_rows))
    return {
        "accepted_row_count": len(accepted_keys),
        "accepted_row_keys": sorted(accepted_keys),
        "accepted_rows": accepted_rows,
        "unsupported_required_rows": [dict(row) for row in missing_rows],
        "observed_dead_end_transitions": dead_ends,
        "observed_dead_end_transition_count": len(dead_ends),
        "row_completion_support_rate": support_rate,
        "row_completion_terminal": bool(required_rows) and len(accepted_keys) == len(required_rows),
    }


def build_invalid_fixture_payloads(valid_payloads: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create deterministic invalid payloads, including an invalid enum case."""

    invalid_payloads = gate5539.build_invalid_fixture_payloads(valid_payloads)
    if valid_payloads:
        invalid_enum = _json_clone(valid_payloads[0])
        invalid_enum["conclusion"]["status"] = "maybe"
        invalid_payloads.append(invalid_enum)
    return invalid_payloads


def evaluate_fixture_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    fixture: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Classify fixture payloads through the Exp5512 parser/exact handoff."""

    fixture_payload = dict(fixture or positive.load_fixture_artifact()["fixture"])
    rows: list[JsonDict] = []
    for payload in payloads:
        classified = positive.classify_candidate_payload(payload, fixture=fixture_payload)
        accepted = _accepted(classified)
        rows.append(
            {
                "instance_id": str(payload.get("instance_id", classified.get("instance_id", ""))),
                "candidate_id": str(payload.get("candidate_id", classified.get("candidate_id", ""))),
                "accepted": accepted,
                "parse_status": str(classified.get("parse_status", "")),
                "schema_valid": bool(classified.get("schema_valid") is True),
                "parseable": bool(classified.get("parseable") is True),
                "exact_validator_correct": bool(classified.get("exact_validator_correct") is True),
                "schema_errors": [str(error) for error in classified.get("schema_errors", [])],
                "rejection_reason": "accepted" if accepted else _rejection_reason(classified),
            }
        )
    return rows


def acceptance_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of classified rows accepted by parser and validators."""

    return _rate(sum(int(row.get("accepted") is True) for row in rows), len(rows))


def rejection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of classified rows rejected by parser or validators."""

    return _rate(sum(int(row.get("accepted") is not True) for row in rows), len(rows))


def build_artifact(
    *,
    upstream_grammar_path: Path = REPO_ROOT / UPSTREAM_GRAMMAR_PREFLIGHT,
    upstream_panel_path: Path = REPO_ROOT / UPSTREAM_PANEL_PATH,
    upstream_grammar_artifact: Mapping[str, Any] | None = None,
    upstream_panel_artifact: Mapping[str, Any] | None = None,
    proposal_records: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the deterministic Exp5552 row-completion receipt."""

    grammar = dict(upstream_grammar_artifact) if upstream_grammar_artifact is not None else _load_json(upstream_grammar_path)
    panel = dict(upstream_panel_artifact) if upstream_panel_artifact is not None else _load_json(upstream_panel_path)
    fixture = positive.load_fixture_artifact()["fixture"]
    valid_payloads = positive.build_fixture_candidate_payloads(fixture)
    invalid_payloads = build_invalid_fixture_payloads(valid_payloads)
    valid_rows = evaluate_fixture_payloads(valid_payloads, fixture=fixture)
    invalid_rows = evaluate_fixture_payloads(invalid_payloads, fixture=fixture)
    required_rows = required_rows_from_panel(panel, fixture=fixture)
    records = list(proposal_records) if proposal_records is not None else proposal_records_from_panel(panel)
    completion = evaluate_completion(required_rows, records, fixture=fixture)
    automaton = build_completion_automaton(required_rows)
    required_rows_with_support = _required_rows_with_support(
        required_rows,
        completion["accepted_row_keys"],
    )
    valid_rate = acceptance_rate(valid_rows)
    invalid_rate = rejection_rate(invalid_rows)
    diagnostic = local_mask_bias_diagnostic(
        grammar_artifact=grammar,
        required_rows=required_rows_with_support,
        completion=completion,
    )
    blockers = _readiness_blockers(
        grammar_artifact=grammar,
        required_rows=required_rows_with_support,
        valid_fixture_acceptance_rate=valid_rate,
        invalid_fixture_rejection_rate=invalid_rate,
        completion=completion,
        automaton=automaton,
    )
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "upstream_grammar_preflight": UPSTREAM_GRAMMAR_PREFLIGHT.as_posix(),
        "upstream_panel_path": UPSTREAM_PANEL_PATH.as_posix(),
        "llm_invoked": False,
        "no_model_specs_required": True,
        "automaton_backend": AUTOMATON_BACKEND,
        "schema_hash": positive.sha256_json(positive.candidate_schema()),
        "required_rows": required_rows_with_support,
        "reachable_state_count": automaton["reachable_state_count"],
        "terminal_state_count": automaton["terminal_state_count"],
        "dead_end_transition_count": automaton["dead_end_transition_count"],
        "valid_fixture_acceptance_rate": valid_rate,
        "invalid_fixture_rejection_rate": invalid_rate,
        "row_completion_support_rate": completion["row_completion_support_rate"],
        "missing_row_risk": missing_row_risk(completion["row_completion_support_rate"]),
        "local_mask_bias_diagnostic": diagnostic,
        "automaton_row_completion_ready": ready,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready=ready, blockers=blockers),
        "required_row_count": len(required_rows),
        "automaton_terminal_states": automaton["terminal_states"],
        "automaton_dead_end_transition_templates": automaton["dead_end_transition_templates"],
        "observed_dead_end_transitions": completion["observed_dead_end_transitions"],
        "observed_dead_end_transition_count": completion["observed_dead_end_transition_count"],
        "accepted_row_keys": completion["accepted_row_keys"],
        "valid_fixture_rows": valid_rows,
        "invalid_fixture_rows": invalid_rows,
        "upstream_grammar_clean": _grammar_clean(grammar),
        "readiness_blockers": blockers,
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    upstream_grammar_path: Path = REPO_ROOT / UPSTREAM_GRAMMAR_PREFLIGHT,
    upstream_panel_path: Path = REPO_ROOT / UPSTREAM_PANEL_PATH,
    upstream_grammar_artifact: Mapping[str, Any] | None = None,
    upstream_panel_artifact: Mapping[str, Any] | None = None,
    proposal_records: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5552 deliverable JSON."""

    artifact = build_artifact(
        upstream_grammar_path=upstream_grammar_path,
        upstream_panel_path=upstream_panel_path,
        upstream_grammar_artifact=upstream_grammar_artifact,
        upstream_panel_artifact=upstream_panel_artifact,
        proposal_records=proposal_records,
        tests_run=tests_run,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5552 artifact and fail closed on overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("no_model_specs_required") is True, "no_model_specs_required")
    _require("model_specs" not in artifact, "model_specs")
    _require(artifact.get("automaton_backend") == AUTOMATON_BACKEND, "automaton_backend")
    _require(artifact.get("schema_hash") == positive.sha256_json(positive.candidate_schema()), "schema_hash")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(artifact.get("tests_added_or_reused") == list(TESTS_ADDED_OR_REUSED), "tests_added_or_reused")
    _require(isinstance(artifact.get("required_rows"), list), "required_rows")
    _require(isinstance(artifact.get("local_mask_bias_diagnostic"), Mapping), "local_mask_bias_diagnostic")
    for field in (
        "reachable_state_count",
        "terminal_state_count",
        "dead_end_transition_count",
        "required_row_count",
        "observed_dead_end_transition_count",
    ):
        _require(int(artifact.get(field, -1)) >= 0, field)
    for field in (
        "valid_fixture_acceptance_rate",
        "invalid_fixture_rejection_rate",
        "row_completion_support_rate",
    ):
        value = float(artifact.get(field, -1.0))
        _require(0.0 <= value <= 1.0, field)
    if artifact.get("automaton_row_completion_ready") is True:
        _require(artifact.get("readiness_blockers") == [], "automaton_row_completion_ready")
        _require(artifact.get("upstream_grammar_clean") is True, "upstream_grammar_clean")
        _require(artifact.get("valid_fixture_acceptance_rate") == 1.0, "valid_fixture_acceptance_rate")
        _require(artifact.get("invalid_fixture_rejection_rate") == 1.0, "invalid_fixture_rejection_rate")
        _require(artifact.get("row_completion_support_rate") == 1.0, "row_completion_support_rate")
        _require(artifact.get("missing_row_risk") == "low_missing_row_risk", "missing_row_risk")
        _require(int(artifact.get("terminal_state_count", 0)) > 0, "terminal_state_count")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    else:
        _require(bool(artifact.get("readiness_blockers")), "automaton_row_completion_ready")
        _require(str(artifact.get("honest_verdict", "")).startswith("blocked:"), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def local_mask_bias_diagnostic(
    *,
    grammar_artifact: Mapping[str, Any],
    required_rows: Sequence[Mapping[str, Any]],
    completion: Mapping[str, Any],
) -> JsonDict:
    """Summarize reachable rows that the local proposal path did not support."""

    accepted = set(str(key) for key in completion.get("accepted_row_keys", []))
    unsupported = [
        {
            "row_key": str(row["row_key"]),
            "model_hf_id": str(row["model_hf_id"]),
            "instance_id": str(row["instance_id"]),
            "schema_reachable": bool(row.get("schema_reachable") is True),
            "reason": "absent_from_local_proposal_path",
        }
        for row in required_rows
        if row.get("schema_reachable") is True and str(row["row_key"]) not in accepted
    ]
    selected_backend = str(grammar_artifact.get("selected_backend", "none"))
    table_exposed = _selected_backend_table_exposed(grammar_artifact)
    flags: list[str] = []
    if selected_backend == "llama_cpp_gbnf":
        flags.append("generic_json_gbnf_does_not_force_candidate_row_ids")
    if not table_exposed:
        flags.append("selected_backend_token_transition_table_not_exposed")
    if unsupported:
        flags.append("proposal_path_missing_required_rows")
    return {
        "proposal_path_source": "upstream_panel_rows_or_injected_local_records",
        "grammar_backend": selected_backend,
        "grammar_table_exposed": table_exposed,
        "syntactically_reachable_row_count": sum(
            int(row.get("schema_reachable") is True) for row in required_rows
        ),
        "proposal_supported_row_count": len(accepted),
        "unsupported_required_row_count": len(unsupported),
        "reachable_but_proposal_unsupported_rows": unsupported,
        "mask_bias_flags": sorted(set(flags)),
    }


def missing_row_risk(row_completion_support_rate: float) -> str:
    """Return a compact risk label for downstream live row generation."""

    if row_completion_support_rate >= 1.0:
        return "low_missing_row_risk"
    if row_completion_support_rate <= 0.0:
        return "critical_missing_row_risk"
    return "high_missing_row_risk"


def honest_verdict(*, ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that cannot imply live SOTA quality."""

    if ready:
        return "complete: automaton_row_completion_ready_no_llm"
    suffix = "_".join(blockers) if blockers else "row_completion_not_ready"
    return f"blocked: automaton_row_completion_not_ready_{suffix}"


def row_key(model_hf_id: str, instance_id: str) -> str:
    """Return a stable table-row key for one model/instance pair."""

    return f"{model_hf_id}::{instance_id}"


def _evaluate_payload_record(
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    required_by_key: Mapping[str, Mapping[str, Any]],
    fixture: Mapping[str, Any],
) -> JsonDict:
    model_id = str(record.get("model_hf_id", ""))
    instance_id = str(payload.get("instance_id", ""))
    key = row_key(model_id, instance_id)
    if key not in required_by_key:
        return {"accepted": False, "dead_end": _dead_end("unknown_required_row", record, row_key=key)}
    classified = positive.classify_candidate_payload(payload, fixture=fixture)
    if not _accepted(classified):
        return {
            "accepted": False,
            "dead_end": _dead_end(
                _rejection_reason(classified),
                record,
                row_key=key,
                detail={"schema_errors": classified.get("schema_errors", [])},
            ),
        }
    return {
        "accepted": True,
        "row_key": key,
        "row": {
            "row_key": key,
            "model_hf_id": model_id,
            "instance_id": instance_id,
            "candidate_id": str(payload.get("candidate_id", "")),
            "production_mode": str(record.get("production_mode", "grammar_masking")),
        },
    }


def _evaluate_classified_record(
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    required_by_key: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    model_id = str(row.get("model_hf_id", record.get("model_hf_id", "")))
    instance_id = str(row.get("instance_id", ""))
    key = row_key(model_id, instance_id)
    if key not in required_by_key:
        return {"accepted": False, "dead_end": _dead_end("unknown_required_row", record, row_key=key)}
    accepted = bool(
        row.get("schema_valid") is True
        and row.get("parseable") is not False
        and row.get("exact_validator_correct") is True
    )
    if not accepted:
        return {
            "accepted": False,
            "dead_end": _dead_end(_classified_rejection_reason(row), record, row_key=key),
        }
    return {
        "accepted": True,
        "row_key": key,
        "row": {
            "row_key": key,
            "model_hf_id": model_id,
            "instance_id": instance_id,
            "candidate_id": str(row.get("candidate_id", "")),
            "production_mode": str(record.get("production_mode", "grammar_masking")),
        },
    }


def _readiness_blockers(
    *,
    grammar_artifact: Mapping[str, Any],
    required_rows: Sequence[Mapping[str, Any]],
    valid_fixture_acceptance_rate: float,
    invalid_fixture_rejection_rate: float,
    completion: Mapping[str, Any],
    automaton: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not _grammar_clean(grammar_artifact):
        blockers.append("upstream_grammar_preflight_not_clean")
    if not required_rows:
        blockers.append("no_required_rows")
    if any(row.get("schema_reachable") is not True for row in required_rows):
        blockers.append("unreachable_required_rows")
    if int(automaton.get("terminal_state_count", 0)) <= 0:
        blockers.append("no_terminal_completion_state")
    if valid_fixture_acceptance_rate < 1.0:
        blockers.append("valid_fixture_not_fully_accepted")
    if invalid_fixture_rejection_rate < 1.0:
        blockers.append("invalid_fixture_not_fully_rejected")
    if completion.get("row_completion_terminal") is not True:
        blockers.append("proposal_path_missing_required_rows")
    return sorted(set(blockers))


def _required_rows_with_support(
    required_rows: Sequence[Mapping[str, Any]],
    accepted_row_keys: Sequence[str],
) -> list[JsonDict]:
    accepted = set(str(key) for key in accepted_row_keys)
    return [
        {
            **dict(row),
            "proposal_supported": str(row["row_key"]) in accepted,
        }
        for row in required_rows
    ]


def _required_model_ids(panel_artifact: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for value in panel_artifact.get("models_attempted", []):
        if value:
            candidates.append(str(value))
    for row in panel_artifact.get("per_model_reports", []):
        if isinstance(row, Mapping) and row.get("model_hf_id"):
            candidates.append(str(row["model_hf_id"]))
    for row in panel_artifact.get("panel_rows", []):
        if isinstance(row, Mapping) and row.get("model_hf_id"):
            candidates.append(str(row["model_hf_id"]))
    for row in panel_artifact.get("missing_instance_ids", []):
        if isinstance(row, Mapping) and row.get("model_hf_id"):
            candidates.append(str(row["model_hf_id"]))
    deduped = _unique(candidates)
    return deduped or ["deterministic_fixture_table"]


def _selected_backend_table_exposed(grammar_artifact: Mapping[str, Any]) -> bool:
    selected = str(grammar_artifact.get("selected_backend", ""))
    for row in grammar_artifact.get("grammar_backend_candidates", []):
        if isinstance(row, Mapping) and row.get("name") == selected:
            return bool(row.get("table_exposed") is True)
    return False


def _grammar_clean(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("grammar_table_preflight_ready") is True
        and artifact.get("backend_available") is True
        and artifact.get("llm_invoked") is False
        and artifact.get("decoding_speedup_claim") is False
        and artifact.get("schema_hash") == positive.sha256_json(positive.candidate_schema())
        and artifact.get("inference_substrate") == gate5539.INFERENCE_SUBSTRATE
        and artifact.get("research_conductor_modified") is not True
        and "load_error" not in artifact
    )


def _accepted(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("schema_valid") is True
        and row.get("parseable") is True
        and row.get("exact_validator_correct") is True
        and row.get("exact_validator_verdict") != "not_handed_off"
    )


def _rejection_reason(classified: Mapping[str, Any]) -> str:
    errors = [str(error) for error in classified.get("schema_errors", [])]
    if any("expected one of" in error for error in errors):
        return "invalid_enum_value"
    parse_status = str(classified.get("parse_status", ""))
    if parse_status == "schema_invalid":
        return "schema_invalid_row"
    if parse_status:
        return parse_status
    return "exact_validator_rejected_row"


def _classified_rejection_reason(row: Mapping[str, Any]) -> str:
    errors = [str(error) for error in row.get("schema_errors", [])]
    if any("expected one of" in error for error in errors):
        return "invalid_enum_value"
    if row.get("schema_valid") is not True:
        return "schema_invalid_row"
    if row.get("exact_validator_correct") is not True:
        return "exact_validator_rejected_row"
    return "malformed_row"


def _dead_end(
    reason: str,
    record: Mapping[str, Any],
    *,
    row_key: str = "",
    detail: Any | None = None,
) -> JsonDict:
    model_id = str(record.get("model_hf_id", ""))
    instance_id = str(record.get("instance_id", ""))
    payload = record.get("parsed_payload")
    if not instance_id and isinstance(payload, Mapping):
        instance_id = str(payload.get("instance_id", ""))
    return {
        "reason": reason,
        "row_key": row_key or row_key_from_record(model_id, instance_id),
        "model_hf_id": model_id,
        "instance_id": instance_id,
        "detail": detail,
    }


def row_key_from_record(model_hf_id: str, instance_id: str) -> str:
    """Return a row key only when both key parts are available."""

    return row_key(model_hf_id, instance_id) if model_hf_id or instance_id else ""


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except TypeError:
        return deepcopy(value)


def _load_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"load_error": f"{type(exc).__name__}: {exc}"}
    return dict(payload) if isinstance(payload, Mapping) else {"load_error": "json_not_object"}


def _unique(values: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "automaton_row_completion_ready": artifact["automaton_row_completion_ready"],
                "row_completion_support_rate": artifact["row_completion_support_rate"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
