"""Build the Exp 3197 ExVerus inductive certificate expansion artifact.

Spec refs: REQ-VERIFY-3197, SCENARIO-VERIFY-3197.

ExVerus is useful to Carnot as a discipline for what a repair gate must demand:
do not only remember the one failing input. Preserve the counterexample, lift it
into an invariant, and add an exact test that rejects a patch that only handles
the observed failure. This module does that from checked-in artifacts only; it
does not run repair or claim that repair is unlocked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.exverus_inductive_certificate_expansion.v1"
EXPERIMENT_ID = "exp3197"
ARTIFACT = "experiment_3197_exverus_inductive_certificate_expansion_v1"

OUTPUT_REL_PATH = Path(
    "results/experiment_3197_exverus_inductive_certificate_expansion_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3197_exverus_inductive_certificate_expansion_v1.py"
)

EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")
EXP3196_REL_PATH = Path("results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")

REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "invariant_schema",
    "invariant_record_count",
    "exact_guard_count",
    "anti_overfit_test_count",
    "linked_domain_preview_count",
    "repair_call_ready",
    "honest_verdict",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("post_295_research_references", Path("research-references.md"), True, "text"),
    ("verification_openspec", SPEC_REL_PATH, True, "text"),
    ("exp3183_counterexample_certificate_expansion_v3", EXP3183_REL_PATH, True, "json"),
    ("exp3196_gencp_domain_preview_repair_compiler_v1", EXP3196_REL_PATH, True, "json"),
    (
        "exp3197_module",
        Path("python/carnot/verify/exverus_inductive_certificate_expansion_v1.py"),
        False,
        "python",
    ),
    (
        "exp3197_script",
        Path("scripts/experiment_3197_exverus_inductive_certificate_expansion_v1.py"),
        False,
        "python",
    ),
    (
        "exp3197_tests",
        Path("tests/python/test_experiment_3197_exverus_inductive_certificate_expansion_v1.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3197_exverus_inductive_certificate_expansion_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/exverus_inductive_certificate_expansion_v1.py -m pytest -o addopts='' tests/python/test_experiment_3197_exverus_inductive_certificate_expansion_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/exverus_inductive_certificate_expansion_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3197_exverus_inductive_certificate_expansion_v1.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
    invariant_limit: int = 5,
) -> JsonDict:
    """REQ-VERIFY-3197: expand exact certificates into invariant guard records."""

    root_path = Path(root)
    exp3183 = read_json_object(root_path / EXP3183_REL_PATH)
    exp3196 = read_json_object(root_path / EXP3196_REL_PATH)
    sources = source_artifacts(root_path)
    preview_by_id = rows_by_id(mapping_rows(exp3196.get("preview_manifest")))
    selected = select_invariant_source_rows(
        mapping_rows(exp3183.get("certificate_records")),
        max(0, int(invariant_limit)),
    )
    invariant_records = [
        build_invariant_record(row, preview_by_id.get(str(row.get("row_id") or ""), {}))
        for row in selected
    ]
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "source_artifacts": sources,
        "source_checksums": {
            row["path"]: row["sha256"] for row in sources if row.get("sha256") is not None
        },
        "source_errors": source_errors(sources),
        "source_schema_observations": source_schema_observations(exp3183, exp3196),
        "invariant_schema": invariant_schema(),
        "invariant_records": invariant_records,
        "invariant_record_count": len(invariant_records),
        "exact_guard_count": sum(1 for row in invariant_records if row.get("exact_guard")),
        "anti_overfit_test_count": sum(
            1 for row in invariant_records if row.get("anti_overfit_test")
        ),
        "linked_domain_preview_count": sum(
            1 for row in invariant_records if row.get("linked_domain_preview")
        ),
        "repair_call_ready": False,
        "limitations": limitations(),
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
    invariant_limit: int = 5,
) -> Path:
    """Build, validate, and persist the schema-versioned Exp 3197 artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run, invariant_limit=invariant_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and treat absent or non-object evidence as unavailable."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return auditable lineage for every source that shapes the artifact."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        readable = None
        if source_type == "json":
            readable = bool(read_json_object(path))
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_structured_source": readable,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose missing or malformed required sources without inventing replacements."""

    errors: list[JsonDict] = []
    for row in sources:
        structured = row.get("source_type") == "json"
        malformed = structured and row.get("readable_structured_source") is not True
        if row.get("required") is True and row.get("present") is not True:
            errors.append({"path": str(row.get("path")), "reason": "missing_required_source"})
        elif row.get("required") is True and malformed:
            errors.append({"path": str(row.get("path")), "reason": "malformed_required_source"})
    return errors


def sha256_file(path: Path) -> str | None:
    """Hash available source files so the result has stable lineage."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def mapping_rows(value: Any) -> list[JsonDict]:
    """Normalize a JSON array into object rows only."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Key rows by row_id while preserving the first source occurrence."""

    keyed: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if row_id and row_id not in keyed:
            keyed[row_id] = dict(row)
    return keyed


def select_invariant_source_rows(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[JsonDict]:
    """Select a bounded risk-first set of rows that actually carry pilot evidence."""

    eligible = [
        dict(row)
        for row in rows
        if isinstance(row.get("pilot_certificate"), Mapping) and row.get("pilot_certificate")
    ]

    def priority(row: Mapping[str, Any]) -> tuple[int, str]:
        family = str(row.get("counterexample_family") or "")
        if row.get("known_false_accept_or_regression") is True or family.startswith(
            "known_false_accept:"
        ):
            rank = 0
        elif "fragment" in family or "repair" in family:
            rank = 1
        elif "anchor" in family or "drift" in family:
            rank = 2
        else:
            rank = 3
        return (rank, str(row.get("row_id") or ""))

    return sorted(eligible, key=priority)[:limit]


def build_invariant_record(row: Mapping[str, Any], preview: Mapping[str, Any]) -> JsonDict:
    """Turn one exact certificate row into an invariant, guard, and anti-overfit test."""

    row_id = str(row.get("row_id") or "unknown-row")
    pilot = dict(row.get("pilot_certificate") or {})
    observed = observed_counterexample(row, pilot)
    invariant = generalized_invariant(row, pilot)
    guard = exact_guard(row, pilot, preview, invariant)
    return {
        "record_id": f"exverus-invariant:{row_id}",
        "row_id": row_id,
        "source_artifact": str(row.get("source_artifact") or EXP3183_REL_PATH.as_posix()),
        "row_family": str(row.get("counterexample_family") or "unknown"),
        "exact_label": str(row.get("exact_label") or ""),
        "checker_authority": str(row.get("checker_authority") or ""),
        "observed_counterexample": observed,
        "generalized_invariant": invariant,
        "exact_guard": guard,
        "anti_overfit_test": anti_overfit_test(row, pilot, observed, guard),
        "linked_domain_preview": linked_domain_preview(preview),
        "authority_note": "artifact-only invariant guard; exact verifier remains final",
    }


def observed_counterexample(row: Mapping[str, Any], pilot: Mapping[str, Any]) -> JsonDict:
    """Preserve the concrete failing assignment or positive anchor evidence."""

    family = str(row.get("counterexample_family") or "")
    row_type = str(pilot.get("row_type") or "")
    is_anchor = "anchor" in family or "drift" in row_type or row.get("checker_result") == "accept"
    return {
        "kind": "positive_anchor" if is_anchor else "counterexample",
        "candidate_answers": list(row.get("candidate_answers") or []),
        "canonical_answer": str(row.get("canonical_answer") or row.get("exact_label") or ""),
        "exact_label": str(row.get("exact_label") or ""),
        "minimal_failing_assignment": dict(pilot.get("minimal_failing_assignment") or {}),
        "violated_constraint": str(pilot.get("violated_constraint") or ""),
        "certificate_type": str(pilot.get("certificate_type") or ""),
    }


def generalized_invariant(row: Mapping[str, Any], pilot: Mapping[str, Any]) -> JsonDict:
    """Lift the concrete certificate into the invariant a later repair must preserve."""

    statement = str(
        pilot.get("expected_corrected_invariant")
        or f"candidate answer matches exact label {row.get('exact_label')}"
    )
    family = str(row.get("counterexample_family") or "")
    mcs = dict(pilot.get("mcs") or {})
    unsat_core = list(pilot.get("unsat_core") or [])
    return {
        "invariant_id": f"INV-{slug(row.get('row_id'))}",
        "statement": statement,
        "scope": invariant_scope(family),
        "authority": str(pilot.get("solver_authority") or row.get("checker_authority") or ""),
        "minimal_correction_set": mcs,
        "unsat_core": unsat_core,
        "evidence_keys": [
            key
            for key in ("minimal_failing_assignment", "mcs", "unsat_core")
            if pilot.get(key)
        ],
    }


def invariant_scope(family: str) -> str:
    """Name the row family that the invariant generalizes across."""

    if "arithmetic" in family:
        return "all arithmetic assertion rows with claimed and computed values"
    if "smt" in family or "drift" in family:
        return "all SMT rows with solver statuses and token-family labels"
    if "json" in family or "fragment" in family:
        return "all JSON fragment repair rows with parser authority"
    return family or "selected exact certificate rows"


def exact_guard(
    row: Mapping[str, Any],
    pilot: Mapping[str, Any],
    preview: Mapping[str, Any],
    invariant: Mapping[str, Any],
) -> JsonDict:
    """Define the exact predicate a future repair candidate must satisfy."""

    return {
        "guard_id": f"EXACT-GUARD-{slug(row.get('row_id'))}",
        "authority": str(pilot.get("solver_authority") or row.get("checker_authority") or ""),
        "verifier_to_rerun": str(pilot.get("verifier_to_rerun") or row.get("checker_authority") or ""),
        "required_exact_label": str(row.get("exact_label") or ""),
        "canonical_answer": str(row.get("canonical_answer") or row.get("exact_label") or ""),
        "reject_when": "candidate violates generalized invariant or canonical exact label",
        "invariant_id": str(invariant.get("invariant_id") or ""),
        "minimal_correction_set": dict(pilot.get("mcs") or {}),
        "unsat_core": list(pilot.get("unsat_core") or []),
        "preview_candidate_domain": list(preview.get("candidate_domain") or []),
        "preview_rejection_test_ids": list(preview.get("exact_rejection_test_ids") or []),
    }


def anti_overfit_test(
    row: Mapping[str, Any],
    pilot: Mapping[str, Any],
    observed: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> JsonDict:
    """Materialize the repair-gate test that catches single-instance patching."""

    mcs = dict(pilot.get("mcs") or {})
    family = str(row.get("counterexample_family") or "")
    exact_label = str(row.get("exact_label") or "")
    forbidden = [
        str(value) for value in row.get("candidate_answers") or [] if str(value) != exact_label
    ]
    positive_anchor = observed.get("kind") == "positive_anchor"
    return {
        "test_id": f"ANTI-OVERFIT-{slug(row.get('row_id'))}",
        "guard_id": str(guard.get("guard_id") or ""),
        "exact_authority": str(guard.get("authority") or ""),
        "generalization_family": str(mcs.get("kind") or family or "exact_label_guard"),
        "patch_risk": "repair over-reach changes an exact accepted anchor"
        if positive_anchor
        else "repair only patches observed instance while bypassing generalized invariant",
        "heldout_condition": heldout_condition(family, positive_anchor),
        "forbidden_candidate_patterns": forbidden,
        "expected_outcome": "preserve_positive_anchor"
        if positive_anchor
        else "reject_overfit_patch",
    }


def heldout_condition(family: str, positive_anchor: bool) -> str:
    """Describe the exact heldout-style condition used by the anti-overfit test."""

    if positive_anchor:
        return "already accepted exact anchor remains unchanged"
    if "arithmetic" in family:
        return "same equality invariant on changed claimed/computed values"
    if "smt" in family:
        return "same solver-label invariant on equivalent satisfiability evidence"
    if "json" in family or "fragment" in family:
        return "same parser invariant on another bounded JSON fragment"
    return "same invariant on another exact certificate row"


def linked_domain_preview(preview: Mapping[str, Any]) -> JsonDict:
    """Keep the GenCP preview cross-reference small and explicit."""

    if not preview:
        return {}
    return {
        "record_id": str(preview.get("record_id") or ""),
        "row_id": str(preview.get("row_id") or ""),
        "candidate_domain": list(preview.get("candidate_domain") or []),
        "exact_rejection_test_ids": list(preview.get("exact_rejection_test_ids") or []),
        "authority_note": str(preview.get("authority_note") or ""),
    }


def slug(value: Any) -> str:
    """Convert row identifiers into stable uppercase IDs for guards and tests."""

    return "".join(ch if ch.isalnum() else "-" for ch in str(value or "unknown")).upper()


def source_schema_observations(exp3183: Mapping[str, Any], exp3196: Mapping[str, Any]) -> JsonDict:
    """Record which source schemas were inspected while building invariant records."""

    return {
        "exp3183_certificate_record_keys": first_row_keys(exp3183.get("certificate_records")),
        "exp3196_preview_record_keys": first_row_keys(exp3196.get("preview_manifest")),
        "exp3196_exact_rejection_test_count": len(mapping_rows(exp3196.get("exact_rejection_tests"))),
    }


def first_row_keys(value: Any) -> list[str]:
    """Return sorted keys from the first object in a list-shaped source."""

    rows = mapping_rows(value)
    return sorted(rows[0].keys()) if rows else []


def invariant_schema() -> JsonDict:
    """Describe the invariant records consumed by later repair-gate checks."""

    return {
        "type": "object",
        "required": [
            "record_id",
            "row_id",
            "observed_counterexample",
            "generalized_invariant",
            "exact_guard",
            "anti_overfit_test",
            "linked_domain_preview",
        ],
        "properties": {
            "observed_counterexample": "minimal failing assignment or positive anchor",
            "generalized_invariant": "family-level invariant derived from exact evidence",
            "exact_guard": "authority-bound predicate a candidate must satisfy",
            "anti_overfit_test": "test rejecting repairs that only patch the observed row",
            "linked_domain_preview": "optional Exp 3196 GenCP preview cross-reference",
        },
    }


def limitations() -> JsonDict:
    """State the artifact boundary so it cannot be mistaken for repair execution."""

    return {
        "repair_execution_claim_made": False,
        "live_model_or_llm_called": False,
        "bounded_records_only": True,
        "why_repair_stays_blocked": "invariant guards prepare later evaluation but do not open the repair gate",
    }


def inference_substrate() -> JsonDict:
    """Make the no-new-execution boundary machine-readable."""

    return {
        "kind": "artifact_only_inductive_certificate_expansion",
        "llm_called": False,
        "new_repair_calls": 0,
        "new_live_model_calls": 0,
        "new_verifier_scoring_calls": 0,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal truthful verdict required by the result schema."""

    return (
        "complete: exverus_inductive_certificate_expansion_v1_ready=true; "
        f"invariant_record_count={artifact['invariant_record_count']}; "
        f"exact_guard_count={artifact['exact_guard_count']}; "
        f"anti_overfit_test_count={artifact['anti_overfit_test_count']}; "
        f"linked_domain_preview_count={artifact['linked_domain_preview_count']}; "
        "repair_call_ready=false"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema omissions, repair unlocks, or broken guard accounting."""

    missing = REQUIRED_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("repair_call_ready") is not False:
        raise ValueError("repair_call_ready must remain false")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with 'complete:'")
    records = artifact.get("invariant_records", [])
    if not isinstance(records, list):
        raise ValueError("invariant_records must be a list")
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("invariant record rows must be objects")
    if artifact.get("invariant_record_count") != len(records):
        raise ValueError("invariant_record_count must equal invariant_records length")
    if artifact.get("exact_guard_count") != sum(1 for row in records if row.get("exact_guard")):
        raise ValueError("exact_guard_count must match materialized exact guards")
    if artifact.get("anti_overfit_test_count") != sum(
        1 for row in records if row.get("anti_overfit_test")
    ):
        raise ValueError("anti_overfit_test_count must match materialized anti-overfit tests")
    for row in records:
        if not row.get("observed_counterexample") or not row.get("generalized_invariant"):
            raise ValueError("invariant record missing counterexample or invariant")
        if not row.get("exact_guard") or not row.get("anti_overfit_test"):
            raise ValueError("invariant record missing exact guard or anti-overfit test")


def main() -> None:  # pragma: no cover
    """CLI entrypoint for the checked-in experiment script."""

    print(write_artifact().as_posix())


if __name__ == "__main__":  # pragma: no cover
    main()
