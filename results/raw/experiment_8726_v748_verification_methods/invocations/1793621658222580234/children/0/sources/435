"""Exp 3159 KAN proof-carrying monitor expansion.

Spec refs: REQ-KAN-3159, SCENARIO-KAN-3159.

This module expands the bounded Exp 3145 proof-carrying monitor evidence over
two additional exact clean rows from the Exp 3136 autopsy set. The records are
still replay/audit metadata only: they expose exact-label links, PWA/MILP bound
status, and residual deployment risk, but they do not install or claim a live
accept/reject verifier.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from carnot.eval import kan_proof_carrying_monitor_boundary_v2 as boundary


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3159_kan_proof_carrying_monitor_expansion_v1"
SCHEMA = "carnot.kan_proof_carrying_monitor_expansion.v1"
BOUND_RECORD_SCHEMA = "carnot.kan_proof_carrying_monitor.expanded_bound_record.v1"
OUTPUT_REL_PATH = Path("results/experiment_3159_kan_proof_carrying_monitor_expansion_v1.json")
EXP3126_REL_PATH = boundary.EXP3126_REL_PATH
EXP3131_REL_PATH = boundary.EXP3131_REL_PATH
EXP3136_REL_PATH = boundary.EXP3136_REL_PATH
EXP3145_REL_PATH = boundary.OUTPUT_REL_PATH
NEW_CLEAN_ROW_LIMIT = 2
SUCCESS_PREFIXES = boundary.SUCCESS_PREFIXES
REQUIRED_ARTIFACT_FIELDS = (
    "kan_proof_carrying_monitor_expansion_v1_ready",
    "monitor_record_count",
    "new_monitor_record_count",
    "exact_row_coverage_count",
    "pwa_milp_bound_records",
    "deployed_verifier_claim_allowed",
    "residual_blockers",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_BOUND_RECORD_FIELDS = (
    "schema",
    "record_version",
    "record_id",
    "fixture_id",
    "record_origin",
    "exact_row_set",
    "exact_label_link",
    "domain_bounds",
    "pwa_abstraction_parameters",
    "local_error_bound_summary",
    "global_error_bound_summary",
    "pwa_milp_status",
    "milp_property_result",
    "deployed_verifier_claim_allowed",
    "residual_risk",
    "record_checksum",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3159_kan_proof_carrying_monitor_expansion_v1.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/kan_proof_carrying_monitor_expansion_v1.py -m pytest -o addopts='' tests/python/test_experiment_3159_kan_proof_carrying_monitor_expansion_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/kan_proof_carrying_monitor_expansion_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("kan_openspec", Path("openspec/capabilities/kan/spec.md"), True),
    ("exp3126_fragment_monitor", EXP3126_REL_PATH, True),
    ("exp3131_kan_abstraction_audit", EXP3131_REL_PATH, True),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3145_proof_carrying_boundary", EXP3145_REL_PATH, True),
    (
        "prompt_named_exp3131_artifact_absent",
        Path("results/experiment_3131_kan_pwa_proof_carrying_monitor_pilot_v1.json"),
        False,
    ),
    (
        "prompt_named_exp3145_artifact_absent",
        Path("results/experiment_3145_kan_pwa_monitor_attachment_boundary_v1.json"),
        False,
    ),
    (
        "exp3159_module",
        Path("python/carnot/eval/kan_proof_carrying_monitor_expansion_v1.py"),
        False,
    ),
    (
        "exp3159_tests",
        Path("tests/python/test_experiment_3159_kan_proof_carrying_monitor_expansion_v1.py"),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object through the same fail-closed helper as Exp 3145."""

    return boundary.read_json_object(path)


def monitor_event_groups_by_fixture(value: Any) -> dict[str, list[JsonDict]]:
    """Group monitor events by fixture ID using the Exp 3145 replay helper."""

    return boundary.monitor_event_groups_by_fixture(value)


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3159 expansion artifact."""

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
    """REQ-KAN-3159: expand bounded monitor records over exact clean rows."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    exp3131 = read_json_object(root_path / EXP3131_REL_PATH)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    exp3145 = read_json_object(root_path / EXP3145_REL_PATH)
    groups = monitor_event_groups_by_fixture(exp3126.get("monitor_events"))
    row_sets = exact_row_sets(exp3136)
    prior_ids = existing_monitor_record_ids(exp3145)
    new_ids = selected_new_exact_row_ids(
        row_sets["clean_exact_row_ids"],
        prior_ids=prior_ids,
        available_ids=groups,
        limit=NEW_CLEAN_ROW_LIMIT,
    )
    records = build_bound_records(groups, exp3131, row_sets["false_accept_row_ids"], prior_ids, new_ids)
    new_count = sum(1 for record in records if record["record_origin"] == "new_exp3159_exact_clean_row")
    exact_count = sum(1 for record in records if record.get("exact_label_link", {}).get("exact_label"))
    implementation_blockers = implementation_blockers_for(root_path, records)
    ready = bool(records and new_count > 0 and exact_count == len(records) and not implementation_blockers)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-KAN-3159", "SCENARIO-KAN-3159"],
        "kan_proof_carrying_monitor_expansion_v1_ready": ready,
        "monitor_record_count": len(records),
        "new_monitor_record_count": new_count,
        "exact_row_coverage_count": exact_count,
        "pwa_milp_bound_records": records,
        "prior_monitor_record_count": len(prior_ids),
        "exact_row_sets": row_sets | {"selected_new_clean_row_ids": new_ids},
        "deployed_verifier_claim_allowed": False,
        "residual_blockers": residual_blockers(),
        "implementation_blockers": implementation_blockers,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in source_artifacts(root_path)
            if row["sha256"] is not None
        },
        "inference_substrate": inference_substrate(),
        "claim_boundary": claim_boundary(),
        "field_principles": field_principles(),
        "duration_s": duration(started, now_s),
        "honest_verdict": honest_verdict(ready),
    }
    artifact["reproducibility_checksum"] = boundary.stable_hash(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )
    validate_artifact(artifact)
    return artifact


def existing_monitor_record_ids(exp3145: Mapping[str, Any]) -> set[str]:
    """Return fixture IDs already carrying proof records in Exp 3145."""

    return {
        str(record.get("fixture_id"))
        for record in boundary.mapping_rows(exp3145.get("monitor_records"))
        if record.get("fixture_id")
    }


def exact_row_sets(exp3136: Mapping[str, Any]) -> JsonDict:
    """Load exact false-accept rows and clean exact rows from the autopsy."""

    false_accept_ids = boundary.string_list(exp3136.get("false_accept_row_ids"))
    clean_ids = sorted(
        str(row.get("row_id"))
        for row in boundary.mapping_rows(exp3136.get("verifier_rows"))
        if is_clean_exact_row(row, false_accept_ids)
    )
    return {
        "false_accept_row_ids": false_accept_ids,
        "clean_exact_row_ids": clean_ids,
        "source_live_row_count": int(exp3136.get("source_live_row_count") or 0),
        "source_false_accept_count": int(exp3136.get("source_false_accept_count") or 0),
    }


def is_clean_exact_row(row: Mapping[str, Any], false_accept_ids: Sequence[str]) -> bool:
    """Return true when one exact row has no monitor violation."""

    row_id = str(row.get("row_id") or "")
    if not row_id or row_id in set(false_accept_ids):
        return False
    if str(row.get("failure_mechanism_from_exp3124") or "") != "no_failure":
        return False
    events = boundary.mapping_rows(row.get("monitor_events"))
    link = boundary.exact_fixture_link(events)
    return (
        bool(link.get("exact_label"))
        and link.get("final_answer_consistent_with_exact") is True
        and link.get("final_answer_consistent_with_ledger") is True
        and link.get("is_monitor_violation") is False
    )


def selected_new_exact_row_ids(
    clean_row_ids: Sequence[str],
    *,
    prior_ids: set[str],
    available_ids: Mapping[str, Any],
    limit: int,
) -> list[str]:
    """Select a deterministic small clean-row expansion set."""

    available = set(available_ids)
    return [
        row_id
        for row_id in sorted(clean_row_ids)
        if row_id in available and row_id not in prior_ids
    ][: max(0, int(limit))]


def build_bound_records(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    exp3131: Mapping[str, Any],
    false_accept_ids: Sequence[str],
    prior_ids: set[str],
    new_ids: Sequence[str],
) -> list[JsonDict]:
    """Build carried-forward and new records when all inputs are present."""

    if not groups or not exp3131:
        return []
    proof = boundary.kan_proof_payload(exp3131)
    selected_ids = [row_id for row_id in sorted(false_accept_ids) if row_id in groups]
    selected_ids.extend(row_id for row_id in new_ids if row_id in groups)
    return [
        build_bound_record(
            fixture_id,
            groups[fixture_id],
            proof,
            exact_row_set="false_accept" if fixture_id in set(false_accept_ids) else "clean",
            record_origin=(
                "carried_forward_exp3145"
                if fixture_id in prior_ids
                else "new_exp3159_exact_clean_row"
            ),
        )
        for fixture_id in selected_ids
    ]


def build_bound_record(
    fixture_id: str,
    events: Sequence[Mapping[str, Any]],
    proof: Mapping[str, Any],
    *,
    exact_row_set: str,
    record_origin: str,
) -> JsonDict:
    """Build one expansion record with explicit bounds and residual risk."""

    base = boundary.build_monitor_record(fixture_id, events, proof)
    pwa = dict(base["pwa_abstraction_parameters"])
    local_summary = dict(base["local_error_bound_summary"])
    global_summary = dict(base["global_error_bound_summary"])
    milp_result = dict(base["milp_property_result"])
    record: JsonDict = {
        "schema": BOUND_RECORD_SCHEMA,
        "record_version": "v1",
        "record_id": f"kan-proof-monitor-expansion-v1:{fixture_id}",
        "fixture_id": fixture_id,
        "source_boundary_record_id": base["record_id"],
        "source_boundary_record_checksum": base["record_checksum"],
        "record_origin": record_origin,
        "exact_row_set": exact_row_set,
        "exact_label_link": dict(base["exact_fixture_link"]),
        "domain_bounds": domain_bounds(pwa, local_summary, global_summary, milp_result),
        "pwa_abstraction_parameters": pwa,
        "local_error_bound_summary": local_summary,
        "global_error_bound_summary": global_summary,
        "pwa_milp_status": pwa_milp_status(milp_result),
        "milp_property_result": milp_result,
        "deployed_verifier_claim_allowed": False,
        "residual_risk": residual_risk(exact_row_set),
        "record_checksum": "",
    }
    record["record_checksum"] = record_checksum(record)
    validate_bound_record(record)
    return record


def domain_bounds(
    pwa: Mapping[str, Any],
    local_summary: Mapping[str, Any],
    global_summary: Mapping[str, Any],
    milp_result: Mapping[str, Any],
) -> JsonDict:
    """Collect the proof-carrying domain and error bounds in one place."""

    return {
        "input_domain": list(pwa.get("property_domain") or []),
        "property_threshold": pwa.get("property_threshold"),
        "certified_upper_bound": milp_result.get("certified_upper_bound"),
        "max_local_error_bound": local_summary.get("max_local_error_bound"),
        "global_error_bound": global_summary.get("global_error_bound"),
        "bounds_distinct_by_construction": global_summary.get("bounds_distinct_by_construction"),
    }


def pwa_milp_status(milp_result: Mapping[str, Any]) -> JsonDict:
    """Expose solver status without upgrading it into deployment evidence."""

    return {
        "property_verified": milp_result.get("property_verified"),
        "solver_status": milp_result.get("solver_status"),
        "milp_backend_available": milp_result.get("milp_backend_available"),
        "milp_backend_name": milp_result.get("milp_backend_name"),
        "exact_enumeration_used_only_as_fallback": milp_result.get(
            "exact_enumeration_used_only_as_fallback"
        ),
    }


def residual_risk(exact_row_set: str) -> list[str]:
    """Return residual risks for a false-accept or clean-row proof record."""

    risks = [
        "No deployed accept/reject gate consumes this proof record.",
        "Record is bounded to checked-in exact rows and does not certify live generation.",
        "PWA/MILP evidence covers the tiny Exp 3131 fixture, not trained-network soundness.",
    ]
    if exact_row_set == "clean":
        risks.append("Clean-row coverage increases denominator only; it does not prove false-accept prevention.")
    return risks


def residual_blockers() -> list[str]:
    """Name deployment blockers that remain after bounded record expansion."""

    return [
        "No deployed accept/reject gate consumes these proof records.",
        "No generation-path integration test exercises these records as a live verifier.",
        "No trained KAN network soundness proof or production runtime binding is present.",
    ]


def implementation_blockers_for(root: Path, records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name missing inputs when the expansion cannot build complete records."""

    blockers = [
        row["path"]
        for row in source_artifacts(root)
        if row["required"] is True and row["exists"] is not True
    ]
    if not boundary.kan_code_present(root):
        blockers.extend(path.as_posix() for path in boundary.KAN_CODE_PATHS if not (root / path).is_file())
    if not records:
        blockers.append("bounded KAN/PWA monitor records could not be expanded over exact rows")
    return sorted(dict.fromkeys(blockers))


def validate_bound_record(record: Mapping[str, Any]) -> None:
    """Raise when one PWA/MILP bound record is incomplete or overclaims."""

    missing = [field for field in REQUIRED_BOUND_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing bound record fields: {missing}")
    if record.get("schema") != BOUND_RECORD_SCHEMA:
        raise ValueError("bound record schema mismatch")
    if record.get("record_checksum") != record_checksum(record):
        raise ValueError("record checksum mismatch")
    link = record.get("exact_label_link")
    if not isinstance(link, Mapping) or not link.get("fixture_id") or not link.get("exact_label"):
        raise ValueError("exact_label_link must identify an exact labeled fixture")
    bounds = record.get("domain_bounds")
    if not isinstance(bounds, Mapping) or not bounds.get("input_domain"):
        raise ValueError("domain_bounds must carry the input domain")
    status = record.get("pwa_milp_status")
    if not isinstance(status, Mapping) or status.get("property_verified") is not True:
        raise ValueError("pwa_milp_status must carry a verified PWA/MILP property")
    if status.get("solver_status") != "optimal":
        raise ValueError("pwa_milp_status must be optimal for this bounded expansion")
    if record.get("deployed_verifier_claim_allowed") is not False:
        raise ValueError("deployed verifier claims are not allowed")
    if not record.get("residual_risk"):
        raise ValueError("residual_risk must remain explicit")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact omits fields or permits deployment claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if artifact.get("deployed_verifier_claim_allowed") is not False:
        raise ValueError("deployed verifier claims are not allowed")
    records = boundary.mapping_rows(artifact.get("pwa_milp_bound_records"))
    for record in records:
        validate_bound_record(record)
    if int(artifact.get("monitor_record_count") or 0) != len(records):
        raise ValueError("monitor_record_count mismatch")
    new_count = sum(1 for record in records if record.get("record_origin") == "new_exp3159_exact_clean_row")
    if int(artifact.get("new_monitor_record_count") or 0) != new_count:
        raise ValueError("new_monitor_record_count mismatch")
    exact_count = sum(1 for record in records if record.get("exact_label_link", {}).get("exact_label"))
    if int(artifact.get("exact_row_coverage_count") or 0) != exact_count:
        raise ValueError("exact_row_coverage_count mismatch")
    if not artifact.get("residual_blockers"):
        raise ValueError("residual_blockers must remain explicit")
    validate_inference_substrate(artifact.get("inference_substrate"))
    if artifact.get("kan_proof_carrying_monitor_expansion_v1_ready") is True and new_count <= 0:
        raise ValueError("ready expansion requires new monitor records")


def validate_inference_substrate(value: Any) -> None:
    """Validate the no-live-inference substrate declaration."""

    if not isinstance(value, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if value.get("live_llm_inference") is not False:
        raise ValueError("live LLM inference must remain false")
    if value.get("live_model_inference") is not False:
        raise ValueError("live model inference must remain false")
    if value.get("model_weight_training") is not False or value.get("model_weight_mutation") is not False:
        raise ValueError("model weights must not be trained or mutated")
    if value.get("hardware_execution") is not False:
        raise ValueError("hardware execution must remain false")
    if value.get("deployed_verifier_claim_allowed") is not False:
        raise ValueError("deployed verifier claims are not allowed")


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return concrete file provenance for this expansion."""

    return [
        {
            "id": source_id,
            "path": rel_path.as_posix(),
            "required": required,
            "exists": (root / rel_path).is_file(),
            "sha256": boundary.sha256_file(root / rel_path),
        }
        for source_id, rel_path, required in SOURCE_ARTIFACTS
    ]


def inference_substrate() -> JsonDict:
    """Declare checked-in artifact replay, not live model inference."""

    return {
        "mode": "checked_in_artifact_kan_monitor_expansion",
        "executes_models": False,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "hardware_execution": False,
        "solver_only_abstraction_accounting": True,
        "deployed_verifier_claim_allowed": False,
    }


def claim_boundary() -> JsonDict:
    """State what the expansion proves and what remains outside scope."""

    return {
        "proves": "four replayable exact-row KAN PWA/MILP monitor records, including two new clean rows",
        "does_not_prove": [
            "deployed verifier improvement",
            "trained KAN network soundness",
            "generation-path integration",
            "hardware execution",
            "live LLM inference",
        ],
    }


def field_principles() -> JsonDict:
    """Map required fields to the evidence discipline they enforce."""

    return {
        "kan_proof_carrying_monitor_expansion_v1_ready": "bounded monitor evidence must be complete",
        "monitor_record_count": "denominator must be explicit",
        "new_monitor_record_count": "milestone contribution must be visible",
        "exact_row_coverage_count": "monitor records must tie to exact labels",
        "pwa_milp_bound_records": "proof-carrying data must be inspectable",
        "deployed_verifier_claim_allowed": "bounded monitor evidence is not deployment",
        "residual_blockers": "blocked deployment must be actionable",
        "tests_run": "touched code/self-checks must run",
        "source_artifacts": "monitor expansion must trace to files",
        "inference_substrate": "bounded monitor work must declare no live LLM inference",
        "honest_verdict": "terminal verdict must use a success prefix unless honestly blocked",
    }


def honest_verdict(ready: bool) -> str:
    """Return the terminal verdict without implying deployment."""

    if ready:
        return "complete_kan_proof_carrying_monitor_expansion_v1_records_added_no_deployed_verifier"
    return "complete_kan_proof_carrying_monitor_expansion_v1_blocked_no_deployed_verifier"


def record_checksum(record: Mapping[str, Any]) -> str:
    """Hash a bound record while excluding its checksum field."""

    return boundary.stable_hash({key: value for key, value in record.items() if key != "record_checksum"})


def duration(started_s: float, now_s: float | None) -> float:
    """Return a non-negative rounded duration."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the expansion artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested result artifact."""

    output = write_artifact()
    artifact = read_json_object(output)
    print(
        json.dumps(
            {
                "artifact": output.as_posix(),
                "kan_proof_carrying_monitor_expansion_v1_ready": artifact[
                    "kan_proof_carrying_monitor_expansion_v1_ready"
                ],
                "monitor_record_count": artifact["monitor_record_count"],
                "new_monitor_record_count": artifact["new_monitor_record_count"],
                "deployed_verifier_claim_allowed": artifact["deployed_verifier_claim_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
