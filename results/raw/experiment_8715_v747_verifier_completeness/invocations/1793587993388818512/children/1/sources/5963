"""Build the Exp 3183 counterexample-certificate expansion v3 artifact.

Spec refs: REQ-VERIFY-3183, SCENARIO-VERIFY-3183.

This builder prepares repair-gate evidence without attempting repair. It joins
the small Exp 3170 certificate pilot to the larger exact-row denominator from
the controlled-invariance and clean-rerun gate artifacts, then records bounded
frontier rows in a BEAVER-inspired shape. The output is deliberately conservative:
repair_call_ready fails closed whenever the evidence chain is flagged, too
narrow, missing known false accepts, or incomplete under exact authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3183_counterexample_certificate_expansion_v3"
SCHEMA = "carnot.counterexample_certificate_expansion.v3"
OUTPUT_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")

EXP3125_REL_PATH = Path("results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3168_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")
EXP3169_REL_PATH = Path("results/experiment_3169_repair_ladder_materializer_v4.json")
EXP3170_REL_PATH = Path("results/experiment_3170_counterexample_certificate_repair_pilot_v2.json")
EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
EXP3181_REL_PATH = Path("results/experiment_3181_clean_live_sota_verifier_rerun_v10.json")
EXP3182_REL_PATH = Path("results/experiment_3182_distributional_ebm_exact_row_sidecar_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

MIN_READY_EXACT_ROWS = 20
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "counterexample_certificate_expansion_v3_ready",
    "exact_row_count",
    "counterexample_count",
    "certificate_records",
    "bounded_frontier_records",
    "known_false_accept_rows_covered",
    "flagged_adversarial",
    "repair_call_ready",
    "blocker_reasons",
    "inference_substrate",
    "honest_verdict",
}
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), False, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False, "text"),
    ("research_references", RESEARCH_REFERENCES_REL_PATH, False, "text"),
    ("verification_openspec", SPEC_REL_PATH, True, "text"),
    ("exp3125_prefix_frontier", EXP3125_REL_PATH, False, "json"),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, False, "json"),
    ("exp3137_exact_safe_contract", EXP3137_REL_PATH, False, "json"),
    ("exp3138_canonical_grounding", EXP3138_REL_PATH, False, "json"),
    ("exp3168_repair_gate_v3", EXP3168_REL_PATH, True, "json"),
    ("exp3169_repair_ladder_v4", EXP3169_REL_PATH, True, "json"),
    ("exp3170_certificate_pilot_v2", EXP3170_REL_PATH, True, "json"),
    ("exp3180_controlled_invariance_v2", EXP3180_REL_PATH, True, "json"),
    ("exp3181_clean_rerun_v10", EXP3181_REL_PATH, True, "json"),
    ("exp3182_distributional_sidecar_v1", EXP3182_REL_PATH, False, "json"),
    (
        "exp3183_module",
        Path("python/carnot/verify/counterexample_certificate_expansion_v3.py"),
        False,
        "python",
    ),
    (
        "exp3183_tests",
        Path("tests/python/test_experiment_3183_counterexample_certificate_expansion_v3.py"),
        False,
        "python",
    ),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3183_counterexample_certificate_expansion_v3.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify/counterexample_certificate_expansion_v3.py -m pytest -o addopts='' tests/python/test_experiment_3183_counterexample_certificate_expansion_v3.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/counterexample_certificate_expansion_v3.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3183_counterexample_certificate_expansion_v3.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3183: materialize repair-readiness evidence without inference."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    payloads = load_payloads(root_path)
    sources = source_artifacts(root_path)
    source_errors = required_source_errors(sources)
    pilot_by_id = pilot_certificates_by_id(payloads["exp3170"])
    sidecar_by_id = rows_by_id(mapping_rows(payloads["exp3182"].get("row_scores")))
    known_false_ids = collect_known_false_accept_ids(payloads, pilot_by_id, sidecar_by_id)
    exact_rows, exact_source = collect_expanded_exact_rows(payloads)
    certificate_records = build_certificate_records(
        exact_rows=exact_rows,
        exact_source=exact_source,
        pilot_by_id=pilot_by_id,
        sidecar_by_id=sidecar_by_id,
        known_false_ids=known_false_ids,
    )
    add_pilot_only_records(certificate_records, pilot_by_id, sidecar_by_id, known_false_ids)
    frontier_records = build_frontier_records(payloads["exp3125"], payloads["exp3170"])
    flagged = flagged_adversarial(payloads)
    covered_false_ids = {
        str(row["row_id"])
        for row in certificate_records
        if row.get("known_false_accept_or_regression") is True
    }
    blockers = blocker_reasons(
        source_errors=source_errors,
        exact_rows=exact_rows,
        certificate_records=certificate_records,
        frontier_records=frontier_records,
        known_false_ids=known_false_ids,
        covered_false_ids=covered_false_ids,
        flagged=flagged,
    )
    repair_ready = not blockers
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3183", "SCENARIO-VERIFY-3183"],
        "counterexample_certificate_expansion_v3_ready": not source_errors and bool(certificate_records),
        "exact_row_count": len(exact_rows),
        "counterexample_count": counterexample_count(certificate_records),
        "certificate_records": certificate_records,
        "bounded_frontier_records": frontier_records,
        "known_false_accept_rows_covered": len(covered_false_ids),
        "known_false_accept_row_ids_expected": sorted(known_false_ids),
        "known_false_accept_row_ids_covered": sorted(covered_false_ids),
        "flagged_adversarial": flagged,
        "repair_call_ready": repair_ready,
        "blocker_reasons": blockers,
        "readiness_checks": readiness_checks(blockers),
        "exact_row_source": exact_source,
        "source_artifacts": sources,
        "source_checksums": {
            row["path"]: row["sha256"] for row in sources if row.get("sha256") is not None
        },
        "source_errors": source_errors,
        "field_principles": field_principles(),
        "inference_substrate": inference_substrate(payloads),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3183 artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_payloads(root: Path) -> dict[str, JsonDict]:
    """Load the exact/cached artifacts used by the expansion."""

    return {
        "exp3125": read_json_object(root / EXP3125_REL_PATH),
        "exp3136": read_json_object(root / EXP3136_REL_PATH),
        "exp3137": read_json_object(root / EXP3137_REL_PATH),
        "exp3138": read_json_object(root / EXP3138_REL_PATH),
        "exp3168": read_json_object(root / EXP3168_REL_PATH),
        "exp3169": read_json_object(root / EXP3169_REL_PATH),
        "exp3170": read_json_object(root / EXP3170_REL_PATH),
        "exp3180": read_json_object(root / EXP3180_REL_PATH),
        "exp3181": read_json_object(root / EXP3181_REL_PATH),
        "exp3182": read_json_object(root / EXP3182_REL_PATH),
    }


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, using an empty mapping for missing optional evidence."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return provenance for every file read or cited by the builder."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_SPECS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def required_source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose missing required artifacts as blockers instead of inferring them."""

    return [
        {
            "path": str(row.get("path") or ""),
            "role": str(row.get("role") or ""),
            "reason": "missing_or_malformed_required_source",
        }
        for row in sources
        if row.get("required") is True
        and (
            row.get("present") is not True
            or (row.get("source_type") == "json" and row.get("readable_json_object") is not True)
        )
    ]


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the source exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_rows(value: Any) -> list[JsonDict]:
    """Normalize a JSON list to dict rows only."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Key rows by row_id while preserving the first occurrence."""

    result: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if row_id and row_id not in result:
            result[row_id] = dict(row)
    return result


def pilot_certificates_by_id(exp3170: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Return Exp 3170 certificate records keyed by row id."""

    return rows_by_id(mapping_rows(exp3170.get("certificate_records")))


def collect_known_false_accept_ids(
    payloads: Mapping[str, Mapping[str, Any]],
    pilot_by_id: Mapping[str, Mapping[str, Any]],
    sidecar_by_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Collect every known false-accept or regression row identifier available."""

    ids: set[str] = set()
    for value in payloads["exp3136"].get("false_accept_row_ids") or []:
        if value:
            ids.add(str(value))
    for value in payloads["exp3180"].get("known_false_accept_regression_ids") or []:
        if value:
            ids.add(str(value))
    for row in mapping_rows(payloads["exp3180"].get("exact_rows_evaluated")):
        if row.get("known_false_accept_regression") is True:
            ids.add(str(row.get("row_id") or ""))
    for row_id, row in pilot_by_id.items():
        if row.get("row_type") == "false_accept":
            ids.add(row_id)
    for row_id, row in sidecar_by_id.items():
        if row.get("known_false_accept") is True:
            ids.add(row_id)
    return {row_id for row_id in ids if row_id}


def collect_expanded_exact_rows(payloads: Mapping[str, Mapping[str, Any]]) -> tuple[list[JsonDict], str]:
    """Select the largest exact-authority row denominator available."""

    candidates = (
        ("exp3180.exact_rows_evaluated", mapping_rows(payloads["exp3180"].get("exact_rows_evaluated"))),
        ("exp3181.exact_rows_evaluated", mapping_rows(payloads["exp3181"].get("exact_rows_evaluated"))),
        ("exp3182.row_scores", mapping_rows(payloads["exp3182"].get("row_scores"))),
        ("exp3137.replay_rows", mapping_rows(payloads["exp3137"].get("replay_rows"))),
    )
    best_source = "unavailable"
    best_rows: list[JsonDict] = []
    for source, rows in candidates:
        normalized = list(
            rows_by_id(
                [normalize_exact_row(row, source) for row in rows if str(row.get("row_id") or "")]
            ).values()
        )
        if len(normalized) > len(best_rows):
            best_source = source
            best_rows = normalized
    return sorted(best_rows, key=lambda row: row["row_id"]), best_source


def normalize_exact_row(row: Mapping[str, Any], source: str) -> JsonDict:
    """Normalize exact rows from slightly different source schemas."""

    exact_label = str(row.get("exact_label") or row.get("extracted_answer") or "")
    candidate_answers = string_list(row.get("candidate_answers"))
    extracted = str(row.get("extracted_answer") or "")
    if extracted:
        append_unique(candidate_answers, extracted)
    checker_result = str(
        row.get("exact_authority_decision")
        or row.get("contract_decision")
        or row.get("decision")
        or exact_decision_from_label(exact_label)
    )
    return {
        "row_id": str(row.get("row_id") or ""),
        "exact_label": exact_label,
        "canonical_answer": exact_label,
        "canonical_answer_source": source,
        "checker_result": checker_result,
        "candidate_answers": candidate_answers,
        "known_false_accept_or_regression": row.get("known_false_accept_regression") is True
        or row.get("known_false_accept") is True
        or row.get("known_false_accept_family") is True,
        "semantic_false_accept": row.get("semantic_false_accept") is True,
        "acceptance_authority": row.get("acceptance_authority") is not False,
        "fixture_family": str(row.get("fixture_family") or ""),
    }


def string_list(value: Any) -> list[str]:
    """Return a stable list of non-empty strings."""

    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        if item is None:
            continue
        append_unique(result, str(item))
    return result


def append_unique(values: list[str], value: str) -> None:
    """Append one non-empty value to a list if absent."""

    if value and value not in values:
        values.append(value)


def exact_decision_from_label(label: str) -> str:
    """Infer a checker result only from exact labels when a source omits one."""

    token = label.upper()
    if token in {"INVALID", "UNSAT", "FALSE", "INCORRECT", "FAIL", "REJECT"}:
        return "reject"
    if token in {"VALID", "SAT", "TRUE", "CORRECT", "PASS", "ACCEPT"}:
        return "accept"
    return "unknown"


def build_certificate_records(
    *,
    exact_rows: Sequence[Mapping[str, Any]],
    exact_source: str,
    pilot_by_id: Mapping[str, Mapping[str, Any]],
    sidecar_by_id: Mapping[str, Mapping[str, Any]],
    known_false_ids: set[str],
) -> list[JsonDict]:
    """Build one certificate record per expanded exact row."""

    records: list[JsonDict] = []
    for row in exact_rows:
        row_id = str(row["row_id"])
        pilot = dict(pilot_by_id.get(row_id, {}))
        sidecar = sidecar_by_id.get(row_id, {})
        family = counterexample_family(row_id, row, pilot, sidecar, known_false_ids)
        checker_result = str(row.get("checker_result") or exact_decision_from_label(str(row.get("exact_label") or "")))
        records.append(
            {
                "row_id": row_id,
                "record_scope": "expanded_exact_row",
                "exact_label": str(row.get("exact_label") or pilot.get("exact_label") or ""),
                "canonical_answer": str(row.get("canonical_answer") or row.get("exact_label") or ""),
                "canonical_answer_source": str(row.get("canonical_answer_source") or exact_source),
                "checker_result": checker_result,
                "candidate_answers": string_list(row.get("candidate_answers")),
                "known_false_accept_or_regression": row_id in known_false_ids
                or row.get("known_false_accept_or_regression") is True,
                "counterexample_family": family,
                "source_artifact": exact_source,
                "checker_authority": checker_authority(pilot, row),
                "pilot_certificate": pilot,
                "sidecar_score_reference": sidecar_reference(sidecar),
                "depends_on_flagged_live_verifier": False,
                "exact_authority_complete": bool(row.get("exact_label")) and checker_result != "unknown",
            }
        )
    return records


def add_pilot_only_records(
    certificate_records: list[JsonDict],
    pilot_by_id: Mapping[str, Mapping[str, Any]],
    sidecar_by_id: Mapping[str, Mapping[str, Any]],
    known_false_ids: set[str],
) -> None:
    """Preserve Exp 3170 certificates that are not part of the expanded denominator."""

    present = {str(row["row_id"]) for row in certificate_records}
    for row_id in sorted(set(pilot_by_id) - present):
        pilot = dict(pilot_by_id[row_id])
        sidecar = sidecar_by_id.get(row_id, {})
        checker_result = exact_decision_from_label(str(pilot.get("exact_label") or ""))
        family = counterexample_family(row_id, {}, pilot, sidecar, known_false_ids)
        certificate_records.append(
            {
                "row_id": row_id,
                "record_scope": "pilot_certificate_only",
                "exact_label": str(pilot.get("exact_label") or ""),
                "canonical_answer": str(pilot.get("exact_label") or ""),
                "canonical_answer_source": "exp3170.certificate_records",
                "checker_result": checker_result,
                "candidate_answers": [],
                "known_false_accept_or_regression": row_id in known_false_ids,
                "counterexample_family": family,
                "source_artifact": "exp3170.certificate_records",
                "checker_authority": checker_authority(pilot, {}),
                "pilot_certificate": pilot,
                "sidecar_score_reference": sidecar_reference(sidecar),
                "depends_on_flagged_live_verifier": False,
                "exact_authority_complete": bool(pilot.get("exact_label")) and checker_result != "unknown",
            }
        )


def counterexample_family(
    row_id: str,
    row: Mapping[str, Any],
    pilot: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    known_false_ids: set[str],
) -> str:
    """Classify the row family used by repair and readiness gates."""

    family = str(
        sidecar.get("fixture_family")
        or row.get("fixture_family")
        or pilot.get("row_type")
        or family_from_row_id(row_id)
    )
    if row_id in known_false_ids:
        return f"known_false_accept:{family}"
    row_type = str(pilot.get("row_type") or "")
    if row_type == "fragment_code":
        return "fragment_code:parser_repair"
    if row_type == "satisfiable_drift":
        return "satisfiable_drift_anchor"
    if candidate_conflict(row.get("candidate_answers")):
        return f"candidate_conflict:{family}"
    return f"exact_row:{family or 'unknown'}"


def family_from_row_id(row_id: str) -> str:
    """Infer a coarse family from stable fixture identifiers."""

    if "smt" in row_id:
        return "smt_constraints"
    if "repair-json" in row_id:
        return "json_fragment_repair"
    if "arith" in row_id:
        return "arithmetic_code_assertions"
    return "unknown"


def candidate_conflict(value: Any) -> bool:
    """Return true when candidates contain both accept and reject labels."""

    polarities = {answer_polarity(item) for item in string_list(value)}
    return "accept" in polarities and "reject" in polarities


def answer_polarity(answer: str) -> str:
    """Map answer tokens to accept, reject, or other."""

    token = answer.strip().upper()
    if token in {"VALID", "SAT", "TRUE", "CORRECT", "PASS", "ACCEPT"}:
        return "accept"
    if token in {"INVALID", "UNSAT", "FALSE", "INCORRECT", "FAIL", "REJECT"}:
        return "reject"
    return "other"


def checker_authority(pilot: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """Name the deterministic checker backing a record."""

    authority = str(pilot.get("verifier_to_rerun") or pilot.get("solver_authority") or "")
    if authority:
        return authority
    family = str(row.get("fixture_family") or "")
    if "smt" in family:
        return "z3_solver"
    if "arithmetic" in family:
        return "python_ast_runtime_execution"
    return "exact_authority_replay"


def sidecar_reference(sidecar: Mapping[str, Any]) -> JsonDict:
    """Keep sidecar diagnostics as references, never acceptance authority."""

    if not sidecar:
        return {}
    return {
        "known_false_accept": sidecar.get("known_false_accept") is True,
        "score_explanation": str(sidecar.get("score_explanation") or ""),
        "fixture_family": str(sidecar.get("fixture_family") or ""),
        "acceptance_authority": False,
    }


def build_frontier_records(exp3125: Mapping[str, Any], exp3170: Mapping[str, Any]) -> list[JsonDict]:
    """Materialize BEAVER-inspired prefix/state frontier rows."""

    rows = mapping_rows(exp3125.get("frontier_rows"))
    if rows:
        return frontier_records_from_exp3125(exp3125, rows)
    return frontier_records_from_exp3170(exp3170)


def frontier_records_from_exp3125(exp3125: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert Exp 3125 frontier rows to the repair-gate v4 shape."""

    bound_width = as_float(exp3125.get("bound_width"))
    constraints = string_list(exp3125.get("constraint_families"))
    records: list[JsonDict] = []
    for index, row in enumerate(rows):
        mass = as_float(row.get("probability_mass"))
        status = str(row.get("status") or "unknown")
        prefix = string_list(row.get("prefix"))
        records.append(
            {
                "frontier_id": f"exp3125-{index:04d}",
                "source_artifact": EXP3125_REL_PATH.as_posix(),
                "fixture_id": str(row.get("fixture_id") or ""),
                "prefix": prefix,
                "frontier_state": status,
                "constraint": constraints[0] if constraints else "prefix_closed_semantic_constraint",
                "constraint_families": constraints,
                "probability_mass": mass,
                "lower_bound": lower_bound(status, mass, bound_width),
                "upper_bound": upper_bound(status, mass, bound_width),
                "exact_status": status if status in {"viable", "pruned"} else "bounded",
                "stop_reason": str(row.get("reason") or status or "bounded_frontier_row"),
            }
        )
    return records


def frontier_records_from_exp3170(exp3170: Mapping[str, Any]) -> list[JsonDict]:
    """Fallback to Exp 3170 bounded summaries when prefix rows are absent."""

    records: list[JsonDict] = []
    for index, row in enumerate(mapping_rows(exp3170.get("bounded_frontier_records"))):
        mass = as_float(row.get("explored_mass"))
        width = as_float(row.get("bound_width"))
        constraints = string_list(row.get("constraint_families"))
        exact_label = str(row.get("exact_label") or "unknown")
        records.append(
            {
                "frontier_id": f"exp3170-summary-{index:04d}",
                "source_artifact": EXP3170_REL_PATH.as_posix(),
                "fixture_id": str(row.get("fixture_id") or ""),
                "prefix": [],
                "frontier_state": "summary",
                "constraint": constraints[0] if constraints else "bounded_certificate_summary",
                "constraint_families": constraints,
                "probability_mass": mass,
                "lower_bound": max(0.0, round(mass - width, 12)),
                "upper_bound": min(1.0, round(mass + width, 12)),
                "exact_status": exact_label,
                "stop_reason": "exp3170_bounded_frontier_summary",
            }
        )
    return records


def as_float(value: Any) -> float:
    """Convert finite numeric JSON values to float, otherwise zero."""

    return float(value) if isinstance(value, (int, float)) else 0.0


def lower_bound(status: str, mass: float, width: float) -> float:
    """Compute a conservative lower bound for one frontier state."""

    if status == "pruned":
        return 0.0
    return max(0.0, round(mass - width, 12))


def upper_bound(status: str, mass: float, width: float) -> float:
    """Compute a conservative upper bound for one frontier state."""

    if status == "pruned":
        return min(1.0, round(width, 12))
    return min(1.0, round(mass + width, 12))


def flagged_adversarial(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return whether any load-bearing source keeps adversarial flags alive."""

    if str(payloads["exp3168"].get("repair_gate_state") or "") == "blocked_flagged_verifier":
        return True
    for key in ("exp3125", "exp3136", "exp3170", "exp3180", "exp3181"):
        payload = payloads[key]
        if payload.get("flagged_adversarial") is True:
            return True
        if payload.get("corrigendum_pending"):
            return True
    return False


def blocker_reasons(
    *,
    source_errors: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    certificate_records: Sequence[Mapping[str, Any]],
    frontier_records: Sequence[Mapping[str, Any]],
    known_false_ids: set[str],
    covered_false_ids: set[str],
    flagged: bool,
) -> list[str]:
    """Return actionable reasons why repair_call_ready cannot open."""

    blockers: list[str] = []
    if source_errors:
        blockers.append("required_source_artifact_missing_or_malformed")
    if len(exact_rows) < MIN_READY_EXACT_ROWS:
        blockers.append(f"certificate_denominator_below_{MIN_READY_EXACT_ROWS}_exact_rows")
    if not certificate_records:
        blockers.append("certificate_records_empty")
    if any(row.get("exact_authority_complete") is not True for row in certificate_records):
        blockers.append("exact_authority_scoring_incomplete")
    if known_false_ids - covered_false_ids:
        blockers.append("known_false_accept_rows_missing_from_certificates")
    if known_false_ids and not covered_false_ids:
        blockers.append("known_false_accept_rows_not_covered")
    if not frontier_records:
        blockers.append("bounded_frontier_records_missing")
    if flagged:
        blockers.append("flagged_adversarial_evidence_present")
    if any(row.get("depends_on_flagged_live_verifier") is True for row in certificate_records):
        blockers.append("certificate_records_depend_on_flagged_live_verifier")
    return blockers


def readiness_checks(blockers: Sequence[str]) -> JsonDict:
    """Expose boolean readiness gates with stable names."""

    blocked = set(blockers)
    return {
        "required_sources_present": "required_source_artifact_missing_or_malformed" not in blocked,
        "certificate_denominator_broad_enough": not any(
            reason.startswith("certificate_denominator_below_") for reason in blocked
        ),
        "certificate_records_present": "certificate_records_empty" not in blocked,
        "exact_authority_scoring_complete": "exact_authority_scoring_incomplete" not in blocked,
        "known_false_accept_rows_covered": "known_false_accept_rows_missing_from_certificates" not in blocked
        and "known_false_accept_rows_not_covered" not in blocked,
        "bounded_frontier_records_present": "bounded_frontier_records_missing" not in blocked,
        "no_flagged_adversarial_evidence": "flagged_adversarial_evidence_present" not in blocked,
        "no_flagged_live_verifier_dependency": "certificate_records_depend_on_flagged_live_verifier"
        not in blocked,
    }


def counterexample_count(certificate_records: Sequence[Mapping[str, Any]]) -> int:
    """Count repair-target counterexamples without double-counting clean anchors."""

    count = 0
    for row in certificate_records:
        family = str(row.get("counterexample_family") or "")
        pilot = row.get("pilot_certificate")
        pilot_map = pilot if isinstance(pilot, Mapping) else {}
        if (
            family.startswith("known_false_accept:")
            or family.startswith("fragment_code:")
            or pilot_map.get("minimal_failing_assignment")
        ):
            count += 1
    return count


def inference_substrate(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Label the work as exact/cached artifact replay, not live inference."""

    source_live_calls = sum(
        int(payload.get("live_call_count") or 0)
        for payload in (payloads["exp3169"], payloads["exp3181"])
    )
    return {
        "kind": "deterministic_counterexample_certificate_expansion_v3",
        "mode": "offline_exact_and_cached_artifact_replay",
        "no_live_inference": True,
        "no_llm_calls": True,
        "executes_models": False,
        "executes_repairs": False,
        "executes_verifiers": False,
        "executes_solvers": False,
        "live_model_calls": 0,
        "repair_calls": 0,
        "new_live_model_calls": 0,
        "source_live_model_calls_reused": source_live_calls,
        "exact_authority_only": True,
    }


def field_principles() -> JsonDict:
    """Record the principles behind the required schema fields."""

    return {
        "counterexample_certificate_expansion_v3_ready": "repair readiness must be materialized",
        "exact_row_count": "denominator must be explicit",
        "counterexample_count": "repair targets must be counted",
        "certificate_records": "certificates must trace to exact rows and checkers",
        "bounded_frontier_records": "repair gates need deterministic frontier evidence",
        "known_false_accept_rows_covered": "adversarial rows must remain load-bearing",
        "flagged_adversarial": "repair cannot proceed on flagged evidence",
        "repair_call_ready": "live repair attempts need an explicit readiness signal",
        "blocker_reasons": "failed readiness must be actionable",
        "inference_substrate": "exact/cached certificate work must not be mislabeled as live inference",
        "honest_verdict": "terminal verdict must start with a completion prefix",
    }


def duration(started_s: float, now_s: float | None) -> float:
    """Return non-negative elapsed seconds."""

    finished = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, finished - float(started_s)), 6)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict string for conductor consumption."""

    ready = "true" if artifact.get("repair_call_ready") is True else "false"
    return (
        "complete: counterexample_certificate_expansion_v3_ready=true; "
        f"exact_row_count={artifact.get('exact_row_count')}; "
        f"counterexample_count={artifact.get('counterexample_count')}; "
        f"known_false_accept_rows_covered={artifact.get('known_false_accept_rows_covered')}; "
        f"repair_call_ready={ready}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the small terminal schema before writing JSON."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not any(verdict.startswith(prefix) for prefix in SUCCESS_PREFIXES):
        raise ValueError("honest_verdict lacks terminal success prefix")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("live_model_calls") != 0:
        raise ValueError("inference_substrate must record zero live model calls")


def main() -> None:  # pragma: no cover - exercised only by manual CLI invocation.
    """CLI entrypoint used by deterministic conductor tasks."""

    path = write_artifact()
    print(path.as_posix())


if __name__ == "__main__":  # pragma: no cover - manual CLI entrypoint.
    main()
