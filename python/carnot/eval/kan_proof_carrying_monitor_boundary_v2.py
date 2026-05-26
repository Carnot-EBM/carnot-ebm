"""Exp 3145 KAN proof-carrying monitor boundary v2.

Spec refs: REQ-KAN-3145, SCENARIO-KAN-3145.

This module attaches the existing tiny KAN PWA/MILP proof evidence from Exp
3131 to a small set of Exp 3126 monitor fixtures. The records are proof-carrying
audit envelopes: they make the fixture link and KAN abstraction evidence
replayable, but they are not a deployed verifier and they do not prove trained
KAN soundness or live generation-path behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from carnot.eval import kan_pwa_milp_verifier_abstraction_audit_v1 as kan3131


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3145_kan_proof_carrying_monitor_boundary_v2"
SCHEMA = "carnot.kan_proof_carrying_monitor_boundary.v2"
MONITOR_RECORD_SCHEMA = "carnot.kan_proof_carrying_monitor.record.v2"
OUTPUT_REL_PATH = Path("results/experiment_3145_kan_proof_carrying_monitor_boundary_v2.json")
EXP3126_REL_PATH = Path("results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json")
EXP3131_REL_PATH = Path("results/experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
KAN_CODE_PATHS = (
    Path("python/carnot/verify/kan_pwa_milp_corrigendum.py"),
    Path("python/carnot/verify/kan_pwa_milp_tiny.py"),
    Path("python/carnot/verify/pwa_kan.py"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "kan_proof_carrying_monitor_v2_ready",
    "kan_code_present",
    "monitor_record_schema",
    "attached_monitor_record_count",
    "local_error_bound_summary",
    "global_error_bound_summary",
    "milp_property_check_count",
    "false_accept_relevance",
    "deployed_verifier_claim",
    "implementation_blockers",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_RECORD_FIELDS = (
    "schema",
    "record_version",
    "record_id",
    "fixture_id",
    "exact_fixture_link",
    "pwa_abstraction_parameters",
    "local_error_bound_summary",
    "global_error_bound_summary",
    "milp_property_result",
    "claim_boundary",
    "record_checksum",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3145_kan_proof_carrying_monitor_boundary_v2.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/kan_proof_carrying_monitor_boundary_v2.py -m pytest -o addopts='' tests/python/test_experiment_3145_kan_proof_carrying_monitor_boundary_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/kan_proof_carrying_monitor_boundary_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("kan_openspec", Path("openspec/capabilities/kan/spec.md"), True),
    ("research_references", Path("research-references.md"), False),
    ("exp3126_fragment_monitor", EXP3126_REL_PATH, True),
    ("exp3131_kan_abstraction_audit", EXP3131_REL_PATH, True),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    (
        "exp3145_module",
        Path("python/carnot/eval/kan_proof_carrying_monitor_boundary_v2.py"),
        False,
    ),
    (
        "exp3145_tests",
        Path("tests/python/test_experiment_3145_kan_proof_carrying_monitor_boundary_v2.py"),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and treat malformed evidence as absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3145 boundary artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    validate_artifact(artifact)
    write_json(out_path, artifact)
    return out_path


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-KAN-3145: attach bounded KAN proof records to monitor fixtures."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    exp3131 = read_json_object(root_path / EXP3131_REL_PATH)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    code_present = kan_code_present(root_path)
    source_rows = source_artifacts(root_path)
    records: list[JsonDict] = []
    local_summary: JsonDict = {}
    global_summary: JsonDict = {}
    if code_present and exp3126 and exp3131:
        proof = kan_proof_payload(exp3131)
        local_summary = proof["local_error_bound_summary"]
        global_summary = proof["global_error_bound_summary"]
        groups = monitor_event_groups_by_fixture(exp3126.get("monitor_events"))
        false_ids = sorted(string_list(exp3136.get("false_accept_row_ids")))
        for fixture_id in selected_fixture_ids(list(groups), false_ids, limit=2):
            records.append(build_monitor_record(fixture_id, groups[fixture_id], proof))
    blockers = implementation_blockers(root_path, code_present, source_rows, records)
    milp_count = unique_milp_property_check_count(records)
    ready = bool(code_present and records and milp_count > 0 and not blockers)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-KAN-3145", "SCENARIO-KAN-3145"],
        "kan_proof_carrying_monitor_v2_ready": ready,
        "kan_code_present": code_present,
        "monitor_record_schema": monitor_record_schema(),
        "attached_monitor_record_count": len(records),
        "monitor_records": records,
        "local_error_bound_summary": local_summary,
        "global_error_bound_summary": global_summary,
        "milp_property_check_count": milp_count,
        "false_accept_relevance": false_accept_relevance(records, exp3136),
        "deployed_verifier_claim": False,
        "implementation_blockers": blockers,
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
        },
        "inference_substrate": inference_substrate(),
        "claim_boundary": claim_boundary(),
        "field_principles": field_principles(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, now_s),
        "honest_verdict": honest_verdict(ready),
    }
    artifact["reproducibility_checksum"] = stable_hash(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )
    validate_artifact(artifact)
    return artifact


def kan_code_present(root: Path) -> bool:
    """Return true only when the local KAN/PWA verifier files exist."""

    return all((root / path).is_file() for path in KAN_CODE_PATHS)


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep JSON object rows from a list, dropping malformed members."""

    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def monitor_event_groups_by_fixture(value: Any) -> dict[str, list[JsonDict]]:
    """Group monitor events by fixture ID in replay order."""

    groups: dict[str, list[JsonDict]] = {}
    for event in mapping_rows(value):
        fixture_id = str(event.get("fixture_id") or "")
        if fixture_id:
            groups.setdefault(fixture_id, []).append(event)
    for fixture_id in groups:
        groups[fixture_id].sort(key=lambda event: int(event.get("event_index") or 0))
    return groups


def selected_fixture_ids(
    available_fixture_ids: Sequence[str],
    false_accept_ids: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    """Prefer known false-accept fixtures, then fall back to sorted monitor IDs."""

    available = set(available_fixture_ids)
    preferred = [fixture_id for fixture_id in sorted(false_accept_ids) if fixture_id in available]
    fallback = [fixture_id for fixture_id in sorted(available) if fixture_id not in preferred]
    return (preferred + fallback)[: max(0, int(limit))]


def kan_proof_payload(exp3131: Mapping[str, Any]) -> JsonDict:
    """Return PWA parameters and proof summaries from Exp 3131 plus the fixture."""

    fixture = kan3131.build_source_fixture()
    pwa_parameters = fixture.as_serializable()
    pwa_parameters["unit_count"] = len(pwa_parameters["units"])
    pwa_parameters["output_segment_count"] = len(pwa_parameters["output_segments"])
    return {
        "pwa_abstraction_parameters": pwa_parameters,
        "local_error_bound_summary": dict(exp3131.get("local_error_bound_summary") or {}),
        "global_error_bound_summary": dict(exp3131.get("global_error_bound_summary") or {}),
        "milp_property_checks": mapping_rows(exp3131.get("milp_property_checks")),
    }


def build_monitor_record(
    fixture_id: str,
    events: Sequence[Mapping[str, Any]],
    proof: Mapping[str, Any],
) -> JsonDict:
    """Build one proof-carrying record for a monitor fixture."""

    property_checks = mapping_rows(proof.get("milp_property_checks"))
    record: JsonDict = {
        "schema": MONITOR_RECORD_SCHEMA,
        "record_version": "v2",
        "record_id": f"kan-proof-monitor-v2:{fixture_id}",
        "fixture_id": fixture_id,
        "exact_fixture_link": exact_fixture_link(events),
        "pwa_abstraction_parameters": dict(proof.get("pwa_abstraction_parameters") or {}),
        "local_error_bound_summary": dict(proof.get("local_error_bound_summary") or {}),
        "global_error_bound_summary": dict(proof.get("global_error_bound_summary") or {}),
        "milp_property_result": property_checks[0] if property_checks else {},
        "claim_boundary": record_claim_boundary(),
        "record_checksum": "",
    }
    record["record_checksum"] = record_checksum(record)
    validate_monitor_record(record)
    return record


def exact_fixture_link(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Extract the exact fixture and monitor decision link from five events."""

    by_type = {str(event.get("event_type") or ""): event for event in events}
    ledger = payload_for(by_type, "constraint_ledger")
    exact = payload_for(by_type, "exact_test_z3_result")
    candidate = payload_for(by_type, "candidate_final_answer")
    drift = payload_for(by_type, "drift_classification")
    event_indices = [int(event.get("event_index") or 0) for event in events]
    first = dict(events[0]) if events else {}
    return {
        "fixture_id": str(first.get("fixture_id") or exact.get("fixture_id") or ""),
        "source_prompt_payload_sha256": first.get("source_prompt_payload_sha256"),
        "monitor_event_indices": event_indices,
        "exact_label": exact.get("exact_label"),
        "expected_action": exact.get("expected_action"),
        "label_source": exact.get("label_source"),
        "solver_label": exact.get("solver_label"),
        "ledger_action": ledger.get("ledger_action"),
        "ledger_source": ledger.get("ledger_source"),
        "ledger_hash": ledger.get("ledger_hash"),
        "live_decision": candidate.get("live_decision"),
        "extracted_answer": candidate.get("extracted_answer"),
        "final_answer_consistent_with_exact": candidate.get("final_answer_consistent_with_exact"),
        "final_answer_consistent_with_ledger": candidate.get("final_answer_consistent_with_ledger"),
        "monitor_failure_mechanism": drift.get("failure_mechanism"),
        "is_monitor_violation": drift.get("is_monitor_violation"),
    }


def payload_for(events_by_type: Mapping[str, Mapping[str, Any]], event_type: str) -> JsonDict:
    """Return one event payload by type."""

    event = events_by_type.get(event_type) or {}
    payload = event.get("payload") if isinstance(event, Mapping) else {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def monitor_record_schema() -> JsonDict:
    """Return the replay contract for attached proof-carrying records."""

    return {
        "schema": MONITOR_RECORD_SCHEMA,
        "required_record_fields": list(REQUIRED_RECORD_FIELDS),
        "exact_fixture_link_fields": [
            "fixture_id",
            "source_prompt_payload_sha256",
            "exact_label",
            "expected_action",
            "label_source",
            "ledger_action",
            "ledger_hash",
            "live_decision",
            "monitor_failure_mechanism",
        ],
        "pwa_abstraction_fields": [
            "units",
            "output_segments",
            "property_domain",
            "property_threshold",
            "bound_procedures",
        ],
        "milp_property_result_fields": [
            "property_verified",
            "certified_upper_bound",
            "property_threshold",
            "milp_backend_available",
            "milp_backend_name",
            "solver_status",
            "counterexample_or_certificate",
        ],
        "checksum_rule": "sha256 over the record with record_checksum removed",
    }


def validate_monitor_record(record: Mapping[str, Any]) -> None:
    """Raise if one monitor record is not replayable or overstates proof scope."""

    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing monitor record fields: {missing}")
    if record.get("schema") != MONITOR_RECORD_SCHEMA:
        raise ValueError("monitor record schema mismatch")
    if record.get("record_checksum") != record_checksum(record):
        raise ValueError("record checksum mismatch")
    link = record.get("exact_fixture_link")
    if not isinstance(link, Mapping) or not link.get("fixture_id"):
        raise ValueError("exact_fixture_link must identify a fixture")
    pwa = record.get("pwa_abstraction_parameters")
    if not isinstance(pwa, Mapping) or not pwa.get("units"):
        raise ValueError("pwa_abstraction_parameters must carry units")
    prop = record.get("milp_property_result")
    if not isinstance(prop, Mapping) or prop.get("property_verified") is not True:
        raise ValueError("milp_property_result must carry a verified property")
    if prop.get("solver_status") != "optimal":
        raise ValueError("milp_property_result must be optimal for this boundary")


def unique_milp_property_check_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Count unique MILP property payloads rather than duplicated attachments."""

    checks = {
        stable_hash(record.get("milp_property_result") or {})
        for record in records
        if isinstance(record.get("milp_property_result"), Mapping)
    }
    return len(checks)


def false_accept_relevance(records: Sequence[Mapping[str, Any]], exp3136: Mapping[str, Any]) -> JsonDict:
    """Report whether these records audit or prevent the `.291` false accepts."""

    known_ids = sorted(string_list(exp3136.get("false_accept_row_ids")))
    attached = [str(record.get("fixture_id") or "") for record in records]
    attached_false = [fixture_id for fixture_id in known_ids if fixture_id in set(attached)]
    false_rows = mapping_rows(exp3136.get("false_accept_rows"))
    primary_by_id = {str(row.get("row_id") or ""): str(row.get("primary_mechanism") or "") for row in false_rows}
    families = sorted(
        {
            str((record.get("exact_fixture_link") or {}).get("monitor_failure_mechanism"))
            for record in records
            if record.get("fixture_id") in attached_false
        }
    )
    primary_mechanisms = sorted({primary_by_id[row_id] for row_id in attached_false if primary_by_id.get(row_id)})
    return {
        "question": "Would the attached KAN proof-carrying records have helped any .291 false-accept family?",
        "known_false_accept_row_ids": known_ids,
        "attached_false_accept_row_ids": attached_false,
        "attached_false_accept_record_count": len(attached_false),
        "covered_false_accept_families": families,
        "covered_primary_mechanisms": primary_mechanisms,
        "would_help_replay_audit": bool(attached_false),
        "would_prevent_live_false_accept": False,
        "deployed_gate_missing": True,
        "reason": (
            "Records preserve exact fixture authority and KAN PWA proof evidence for replay, "
            "but no live accept/reject gate consumes them."
        ),
    }


def implementation_blockers(
    root: Path,
    code_present: bool,
    source_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Name exact missing inputs when the boundary cannot attach records."""

    blockers: list[str] = []
    if not code_present:
        blockers.extend(path.as_posix() for path in KAN_CODE_PATHS if not (root / path).is_file())
    blockers.extend(
        str(row.get("path"))
        for row in source_rows
        if row.get("required") is True and row.get("exists") is not True
    )
    if code_present and not records:
        blockers.append("proof-carrying monitor records could not be attached to monitor fixtures")
    return sorted(dict.fromkeys(blockers))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact omits required fields or overstates integration."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if artifact.get("deployed_verifier_claim") is not False:
        raise ValueError("deployed_verifier_claim must remain false")
    records = mapping_rows(artifact.get("monitor_records"))
    for record in records:
        validate_monitor_record(record)
    if int(artifact.get("attached_monitor_record_count") or 0) != len(records):
        raise ValueError("attached_monitor_record_count mismatch")
    if int(artifact.get("milp_property_check_count") or 0) != unique_milp_property_check_count(records):
        raise ValueError("milp_property_check_count mismatch")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("live LLM inference must remain false")
    if substrate.get("live_model_inference") is not False:
        raise ValueError("live model inference must remain false")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights must not be trained or mutated")
    if substrate.get("hardware_execution") is not False:
        raise ValueError("hardware execution must remain false")
    if substrate.get("deployed_verifier_claim") is not False:
        raise ValueError("deployed verifier claim must remain false")
    if artifact.get("kan_proof_carrying_monitor_v2_ready") is True:
        if artifact.get("kan_code_present") is not True:
            raise ValueError("ready boundary requires KAN code")
        if artifact.get("implementation_blockers"):
            raise ValueError("ready boundary cannot have implementation blockers")
        if not records:
            raise ValueError("ready boundary requires attached monitor records")


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return concrete source artifact paths and checksums."""

    return [
        {
            "id": source_id,
            "path": rel_path.as_posix(),
            "required": required,
            "exists": (root / rel_path).is_file(),
            "sha256": sha256_file(root / rel_path),
        }
        for source_id, rel_path, required in SOURCE_ARTIFACTS
    ]


def inference_substrate() -> JsonDict:
    """Declare that this run is checked-in artifact replay, not live inference."""

    return {
        "mode": "checked_in_artifact_kan_monitor_boundary",
        "executes_models": False,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "hardware_execution": False,
        "solver_only_abstraction_accounting": True,
        "deployed_verifier_claim": False,
    }


def claim_boundary() -> JsonDict:
    """State what the artifact proves and what remains outside scope."""

    return {
        "proves": "two replayable KAN PWA/MILP proof records attached to monitor fixtures",
        "does_not_prove": [
            "trained KAN network soundness",
            "generation-path integration",
            "deployed verifier improvement",
            "hardware execution",
            "live LLM inference",
        ],
    }


def record_claim_boundary() -> JsonDict:
    """Return the claim boundary copied into each monitor record."""

    return {
        "proof_scope": "Exp 3131 two-unit PWA/MILP fixture only",
        "fixture_scope": "Exp 3126 exact monitor fixture link only",
        "deployed_verifier_claim": False,
    }


def field_principles() -> JsonDict:
    """Map required fields to the discipline they enforce."""

    return {
        "kan_proof_carrying_monitor_v2_ready": "KAN follow-up must produce a concrete boundary",
        "kan_code_present": "implementation claims require code",
        "monitor_record_schema": "proof-carrying records must be replayable",
        "attached_monitor_record_count": "attachment scale must be visible",
        "local_error_bound_summary": "unit-level approximation error must be explicit",
        "global_error_bound_summary": "network-level claims need propagated bounds",
        "milp_property_check_count": "verification claims need property checks",
        "false_accept_relevance": "architecture work must address the current blocker or say it does not",
        "deployed_verifier_claim": "bounded monitor records are not deployment",
        "implementation_blockers": "design-only output must say what is missing",
        "tests_run": "verifier/abstraction code must be checked",
        "source_artifacts": "KAN evidence must trace to concrete files",
        "inference_substrate": "solver-only work must declare no live LLM inference",
        "honest_verdict": "terminal verdict must use a success prefix unless honestly blocked",
    }


def honest_verdict(ready: bool) -> str:
    """Return the terminal verdict without implying deployment."""

    if ready:
        return "complete_kan_proof_carrying_monitor_v2_records_attached_no_deployed_claim"
    return "complete_kan_proof_carrying_monitor_v2_design_boundary_no_deployed_claim"


def record_checksum(record: Mapping[str, Any]) -> str:
    """Hash a record while excluding its checksum field."""

    return stable_hash({key: value for key, value in record.items() if key != "record_checksum"})


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-serializable value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def string_list(value: Any) -> list[str]:
    """Return sorted string values from a JSON list."""

    return sorted(str(item) for item in value) if isinstance(value, list) else []


def duration(started_s: float, now_s: float | None) -> float:
    """Return a non-negative rounded duration."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 of a file, or ``None`` when it is absent."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested artifact."""

    output = write_artifact()
    print(json.dumps({"artifact": str(output), "ready": True}))


if __name__ == "__main__":  # pragma: no cover
    main()
