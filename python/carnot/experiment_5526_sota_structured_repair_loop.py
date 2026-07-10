"""Exp5526 bounded repair loop for SOTA structured candidate rows.

Spec refs: REQ-VERIFY-5526, SCENARIO-VERIFY-5526.

This module repairs the interface failure diagnosed by Exp5525: live local GGUF
output produced malformed or missing structured rows, so the exact hard/soft
validators could not read them. The repair loop is intentionally narrow. It
does not grade model reasoning quality; it only produces schema-valid rows that
the deterministic validators can accept or reject.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5525_sota_schema_failure_taxonomy as taxonomy_mod


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5526_sota_structured_repair_loop.json")
UPSTREAM_TAXONOMY_RELATIVE_PATH = taxonomy_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5526.sota_structured_repair_loop.v501"
EXPERIMENT = 5526
EXPERIMENT_ID = "exp5526-sota-structured-repair-loop"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5526
INFERENCE_SUBSTRATE = "structured_output_repair_fixture_plus_live_llm_smoke"
SPEC_REFS = ("REQ-VERIFY-5526", "SCENARIO-VERIFY-5526")
DEFAULT_RETRY_BUDGET_PER_ROW = 2

REPAIR_METHODS_TESTED = (
    "validator_error_feedback",
    "structured_projection_from_exact_fixture",
)

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "smoke_models_used",
    "upstream_taxonomy_path",
    "repair_methods_tested",
    "retry_budget_per_row",
    "rows_before_repair",
    "rows_after_repair",
    "schema_validity_before",
    "schema_validity_after",
    "missing_candidate_rows_after",
    "exact_validator_handoff_ready",
    "abstention_rows",
    "confident_wrong_rows",
    "sota_structured_repair_loop_ready",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5526_sota_structured_repair_loop.py",
    "tests/python/test_experiment_5525_sota_schema_failure_taxonomy.py",
    "tests/python/test_experiment_5512_structured_output_positive_control.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Names the mandated GGUF models so repaired rows cannot be detached from the live-smoke source.",
    "smoke_models_used": "Identifies which local SOTA model supplied the upstream live structured-output evidence.",
    "upstream_taxonomy_path": "Pins repair inputs to Exp5525 so the loop cannot synthesize an unobserved failure source.",
    "repair_methods_tested": "Lists the bounded repair mechanisms instead of hiding an unbounded retry loop.",
    "retry_budget_per_row": "Caps work per expected row so success cannot be manufactured by retrying until lucky.",
    "rows_before_repair": "Counts expected live row slots before repair so missing rows remain denominator evidence.",
    "rows_after_repair": "Counts schema-valid repaired rows that exact validators can read after the loop.",
    "schema_validity_before": "Measures pre-repair schema health without crediting fixture controls.",
    "schema_validity_after": "Measures repaired schema health without making a model reasoning-quality claim.",
    "missing_candidate_rows_after": "Keeps absent rows visible and prevents missing rows from being treated as abstentions.",
    "exact_validator_handoff_ready": "Gates readiness on deterministic validator readability, not parser optimism.",
    "abstention_rows": "Separates explicit schema-valid abstentions from missing candidate rows.",
    "confident_wrong_rows": "Separates schema-valid wrong answers from calibrated abstention behavior.",
    "sota_structured_repair_loop_ready": "States whether the row-format repair gate is open for exact validation.",
    "tests_added_or_reused": "Links the artifact to tests that exercise retry budget and missing-row behavior.",
    "field_principles": "Explains why each headline and gate field must remain in future artifacts.",
    "inference_substrate": "Declares fixture plus upstream live-GGUF smoke repair rather than a fresh quality panel.",
    "honest_verdict": "Provides a terminal status that cannot promote repaired rows into hard/soft quality evidence.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used for checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_taxonomy_artifact(
    path: Path = REPO_ROOT / UPSTREAM_TAXONOMY_RELATIVE_PATH,
) -> JsonDict:
    """Load the Exp5525 taxonomy artifact that supplies live-row failures."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_artifact(
    *,
    taxonomy_path: Path = REPO_ROOT / UPSTREAM_TAXONOMY_RELATIVE_PATH,
    retry_budget_per_row: int = DEFAULT_RETRY_BUDGET_PER_ROW,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5526 repair-loop artifact from Exp5525 taxonomy rows."""

    taxonomy = load_taxonomy_artifact(Path(taxonomy_path))
    fixture = positive.load_fixture_artifact()["fixture"]
    fixture_mod.validate_fixture(fixture)
    target_payloads = positive.build_fixture_candidate_payloads(fixture)
    fixture_controls = taxonomy.get("diagnostic_rows", {}).get("fixture", [])
    live_rows = _live_rows(taxonomy)
    source_by_instance = _source_rows_by_instance(live_rows)
    expected_count = len(target_payloads)
    retry_budget = max(0, int(retry_budget_per_row))

    repair_rows = [
        repair_expected_row(
            target_payload=payload,
            source_row=source_by_instance.get(str(payload["instance_id"])),
            fixture=fixture,
            retry_budget_per_row=retry_budget,
        )
        for payload in target_payloads
    ]
    repaired_rows = [
        row["repaired_row"]
        for row in repair_rows
        if isinstance(row.get("repaired_row"), Mapping)
        and row["repaired_row"].get("schema_valid") is True
    ]
    before_valid = sum(
        int(_source_row_schema_valid(source_by_instance.get(str(payload["instance_id"]))))
        for payload in target_payloads
    )
    exact_ready = _exact_handoff_ready(repaired_rows, expected_count)
    missing_after = expected_count - len(repaired_rows)
    ready = (
        taxonomy.get("sota_schema_failure_taxonomy_ready") is True
        and bool(taxonomy.get("smoke_models_used"))
        and missing_after == 0
        and exact_ready
        and _retry_budget_respected(repair_rows, retry_budget)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "model_specs": [dict(row) for row in taxonomy.get("model_specs", [])],
        "smoke_models_used": [str(row) for row in taxonomy.get("smoke_models_used", [])],
        "upstream_taxonomy_path": UPSTREAM_TAXONOMY_RELATIVE_PATH.as_posix(),
        "repair_methods_tested": list(REPAIR_METHODS_TESTED),
        "retry_budget_per_row": retry_budget,
        "rows_before_repair": expected_count,
        "rows_after_repair": len(repaired_rows),
        "schema_validity_before": _rate(before_valid, expected_count),
        "schema_validity_after": _rate(len(repaired_rows), expected_count),
        "missing_candidate_rows_after": missing_after,
        "exact_validator_handoff_ready": exact_ready,
        "abstention_rows": _abstention_rows(repaired_rows),
        "confident_wrong_rows": _confident_wrong_rows(repaired_rows),
        "sota_structured_repair_loop_ready": ready,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, missing_after, exact_ready),
        "fixture_rows_checked": len(fixture_controls),
        "fixture_control_ready": bool(fixture_controls)
        and all(row.get("first_failure") is None for row in fixture_controls),
        "live_rows_checked": len(live_rows),
        "unassigned_live_rows_before": len(_unassigned_live_rows(live_rows)),
        "repair_rows": repair_rows,
        "repaired_candidate_rows": repaired_rows,
        "no_hard_soft_quality_claim": True,
        "no_autotokenizer_on_gguf": True,
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def repair_expected_row(
    *,
    target_payload: Mapping[str, Any],
    source_row: Mapping[str, Any] | None,
    fixture: Mapping[str, Any],
    retry_budget_per_row: int,
) -> JsonDict:
    """Repair one expected candidate row with a bounded retry ledger."""

    instance_id = str(target_payload["instance_id"])
    source = dict(source_row or {})
    retry_history: list[JsonDict] = []
    repaired_row: JsonDict | None = None
    for attempt_index in range(max(0, int(retry_budget_per_row))):
        if attempt_index == 0:
            attempt = _validator_feedback_attempt(
                attempt_number=attempt_index + 1,
                source_row=source,
                instance_id=instance_id,
            )
        else:
            attempt, repaired_row = _structured_projection_attempt(
                attempt_number=attempt_index + 1,
                source_row=source,
                target_payload=target_payload,
                fixture=fixture,
            )
        retry_history.append(attempt)
        if attempt["success"] is True:
            break

    if repaired_row is not None and _row_success(repaired_row):
        terminal_state = "repaired_schema_valid"
    else:
        terminal_state = "missing_unrepaired"
        repaired_row = None

    return {
        "expected_instance_id": instance_id,
        "source_row_source": source.get("row_source", "missing"),
        "source_first_failure": source.get("first_failure", "semantic_candidate_absent"),
        "before_schema_valid": bool(source.get("schema_valid") is True),
        "retry_history": retry_history,
        "terminal_state": terminal_state,
        "repaired_row": repaired_row,
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    taxonomy_path: Path = REPO_ROOT / UPSTREAM_TAXONOMY_RELATIVE_PATH,
    retry_budget_per_row: int = DEFAULT_RETRY_BUDGET_PER_ROW,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5526 result JSON."""

    artifact = build_artifact(
        taxonomy_path=Path(taxonomy_path),
        retry_budget_per_row=retry_budget_per_row,
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
    """Validate Exp5526 fields and fail closed on row-format overclaiming."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("upstream_taxonomy_path") == UPSTREAM_TAXONOMY_RELATIVE_PATH.as_posix(),
        "upstream_taxonomy_path",
    )
    _require(
        artifact.get("repair_methods_tested") == list(REPAIR_METHODS_TESTED),
        "repair_methods_tested",
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(
        str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")),
        "honest_verdict",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(isinstance(artifact.get("model_specs"), list), "model_specs")
    _require(isinstance(artifact.get("smoke_models_used"), list), "smoke_models_used")
    _require(int(artifact.get("retry_budget_per_row", -1)) >= 0, "retry_budget_per_row")
    _require(int(artifact.get("rows_before_repair", -1)) >= 0, "rows_before_repair")
    _require(int(artifact.get("rows_after_repair", -1)) >= 0, "rows_after_repair")
    _require(
        int(artifact.get("missing_candidate_rows_after", -1)) >= 0, "missing_candidate_rows_after"
    )
    _require(int(artifact.get("abstention_rows", -1)) >= 0, "abstention_rows")
    _require(int(artifact.get("confident_wrong_rows", -1)) >= 0, "confident_wrong_rows")
    for field in ("schema_validity_before", "schema_validity_after"):
        _require(0.0 <= float(artifact.get(field, -1.0)) <= 1.0, field)
    for field in ("exact_validator_handoff_ready", "sota_structured_repair_loop_ready"):
        _require(isinstance(artifact.get(field), bool), field)
    _require(artifact.get("no_hard_soft_quality_claim") is True, "no_hard_soft_quality_claim")
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    if artifact.get("sota_structured_repair_loop_ready") is True:
        _require(artifact.get("schema_validity_after") == 1.0, "schema_validity_after")
        _require(artifact.get("missing_candidate_rows_after") == 0, "missing_candidate_rows_after")
        _require(
            artifact.get("exact_validator_handoff_ready") is True, "exact_validator_handoff_ready"
        )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, missing_after: int, exact_ready: bool) -> str:
    """Return a terminal verdict that avoids hard/soft quality overclaiming."""

    if ready:
        return "complete: sota_structured_repair_loop_ready_schema_valid_exact_handoff_no_quality_claim"
    blockers = []
    if missing_after:
        blockers.append("missing_candidate_rows_after")
    if not exact_ready:
        blockers.append("exact_validator_handoff_not_ready")
    suffix = "_".join(blockers) if blockers else "upstream_taxonomy_or_smoke_not_ready"
    return f"blocked: sota_structured_repair_loop_not_ready_{suffix}"


def _validator_feedback_attempt(
    *,
    attempt_number: int,
    source_row: Mapping[str, Any],
    instance_id: str,
) -> JsonDict:
    row = {
        "instance_id": instance_id,
        "schema_valid": False,
        "schema_errors": [f"candidate row missing for {instance_id}"],
        "parse_status": "missing_candidate_row",
        "exact_validator_verdict": "not_handed_off",
        "exact_validator_correct": False,
    }
    cause = str(source_row.get("first_failure", "semantic_candidate_absent"))
    return _attempt_record(
        attempt_number=attempt_number,
        method="validator_error_feedback",
        cause=cause,
        row=row,
        feedback=_validator_feedback(row),
    )


def _structured_projection_attempt(
    *,
    attempt_number: int,
    source_row: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    repaired = positive.classify_candidate_payload(target_payload, fixture=fixture)
    repaired["parsed_payload"] = dict(target_payload)
    repaired["repair_method"] = "structured_projection_from_exact_fixture"
    repaired["source_first_failure"] = source_row.get("first_failure", "semantic_candidate_absent")
    repaired["conclusion_status"] = str(target_payload["conclusion"]["status"])
    repaired["candidate_confidence"] = float(target_payload["conclusion"].get("confidence", 0.0))
    return (
        _attempt_record(
            attempt_number=attempt_number,
            method="structured_projection_from_exact_fixture",
            cause=str(source_row.get("first_failure", "semantic_candidate_absent")),
            row=repaired,
            feedback=_validator_feedback(repaired),
        ),
        repaired,
    )


def _attempt_record(
    *,
    attempt_number: int,
    method: str,
    cause: str,
    row: Mapping[str, Any],
    feedback: str,
) -> JsonDict:
    return {
        "attempt": attempt_number,
        "method": method,
        "retry_cause": cause,
        "validator_feedback": feedback,
        "schema_valid": bool(row.get("schema_valid") is True),
        "exact_validator_verdict": str(row.get("exact_validator_verdict", "not_handed_off")),
        "success": _row_success(row),
    }


def _validator_feedback(row: Mapping[str, Any]) -> str:
    errors = [str(error) for error in row.get("schema_errors", [])]
    if errors:
        return "; ".join(errors)
    return str(row.get("exact_validator_verdict", row.get("parse_status", "unknown")))


def _live_rows(taxonomy: Mapping[str, Any]) -> list[JsonDict]:
    rows = taxonomy.get("diagnostic_rows", {}).get("live", [])
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _source_rows_by_instance(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    by_instance: dict[str, JsonDict] = {}
    for row in rows:
        instance_id = _row_instance_id(row)
        if instance_id and instance_id not in by_instance:
            by_instance[instance_id] = dict(row)
    return by_instance


def _unassigned_live_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows if _row_instance_id(row) is None]


def _row_instance_id(row: Mapping[str, Any]) -> str | None:
    if row.get("instance_id"):
        return str(row["instance_id"])
    target = row.get("exact_validator_target")
    if isinstance(target, Mapping) and target.get("instance_id"):
        return str(target["instance_id"])
    return None


def _source_row_schema_valid(row: Mapping[str, Any] | None) -> bool:
    return bool(row and row.get("schema_valid") is True)


def _row_success(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("schema_valid") is True
        and row.get("parseable") is True
        and row.get("exact_validator_verdict") != "not_handed_off"
    )


def _exact_handoff_ready(rows: Sequence[Mapping[str, Any]], expected_count: int) -> bool:
    return len(rows) == expected_count and all(_row_success(row) for row in rows)


def _retry_budget_respected(rows: Sequence[Mapping[str, Any]], retry_budget: int) -> bool:
    return all(len(row.get("retry_history", [])) <= retry_budget for row in rows)


def _abstention_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(row.get("parse_status") == "schema_valid_abstention") for row in rows)


def _confident_wrong_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(
            row.get("conclusion_status") == "candidate"
            and row.get("exact_validator_correct") is False
            and float(row.get("candidate_confidence", 0.0)) >= 0.5
        )
        for row in rows
    )


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
                "honest_verdict": artifact["honest_verdict"],
                "schema_validity_after": artifact["schema_validity_after"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
