"""Build the Exp 3196 GenCP domain preview repair compiler artifact.

Spec refs: REQ-VERIFY-3196, SCENARIO-VERIFY-3196.

GenCP's useful idea for Carnot is not "ask a model again." It is the quieter
step before that: compile the answer and repair space down to a bounded domain
using exact evidence that already exists. This module therefore reads checked-in
certificate, canonical-answer, fixture, and frontier artifacts, then writes a
preview manifest that a later repair prompt can consume only after the repair
gate is separately opened.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.gencp_domain_preview_repair_compiler.v1"
EXPERIMENT_ID = "exp3196"
COMPILER_VERSION = "v1"
ARTIFACT = "experiment_3196_gencp_domain_preview_repair_compiler_v1"

OUTPUT_REL_PATH = Path("results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json")
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3196_gencp_domain_preview_repair_compiler_v1.py"
)

EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")
EXP3195_REL_PATH = Path("results/experiment_3195_adaptive_verification_granularity_policy_v1.json")
EXP3138_REL_PATH = Path("results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json")
EXP3084_REL_PATH = Path("results/experiment_3084_resyn_exact_fixture_bank_generator_v1.json")
FIXTURE_MANIFEST_REL_PATH = Path("results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl")
EXP3018_REL_PATH = Path("results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json")
BEAVER_MANIFEST_REL_PATH = Path(
    "results/beaver_style_validator_frontier_certificate_3018/certificate_manifest.jsonl"
)

PROP_CANONICAL = "PROP-CANONICAL-ANSWER"
PROP_FALSE_ACCEPT = "PROP-FALSE-ACCEPT-CONFLICT"
PROP_MCS = "PROP-MCS-INVARIANT"
PROP_FIXTURE = "PROP-FIXTURE-AUTHORITY"
PROP_FRONTIER = "PROP-FRONTIER-STATUS"
PROP_GATE = "PROP-REPAIR-GATE-BLOCK"

REJECT_CANONICAL = "EXACT-REJECT-CANONICAL-MISMATCH"
REJECT_FALSE_ACCEPT = "EXACT-REJECT-FALSE-ACCEPT-CONFLICT"
REJECT_MCS = "EXACT-REJECT-MCS-INVARIANT"
REJECT_FIXTURE = "EXACT-REJECT-FIXTURE-AUTHORITY-CONTRADICTION"
REJECT_FRONTIER = "EXACT-REJECT-FRONTIER-PRUNED-PREFIX"
REJECT_PROMOTION = "EXACT-REJECT-NONAUTHORITATIVE-PROMOTION"
REJECT_GATE = "EXACT-REJECT-REPAIR-GATE-BLOCKED"

REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "compiler_version",
    "source_artifacts",
    "domain_record_schema",
    "propagation_rules",
    "exact_rejection_tests",
    "preview_domain_count",
    "average_candidate_domain_size",
    "repair_call_ready",
    "promotion_allowed",
    "honest_verdict",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("post_295_research_references", Path("research-references.md"), True, "text"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3183_counterexample_certificate_expansion_v3", EXP3183_REL_PATH, True, "json"),
    ("exp3195_adaptive_verification_granularity_policy_v1", EXP3195_REL_PATH, True, "json"),
    ("exp3138_canonical_answer_grounding_v1", EXP3138_REL_PATH, True, "json"),
    ("exp3084_exact_fixture_bank", EXP3084_REL_PATH, True, "json"),
    ("exp3084_fixture_manifest", FIXTURE_MANIFEST_REL_PATH, True, "jsonl"),
    ("exp3018_frontier_certificate", EXP3018_REL_PATH, False, "json"),
    ("exp3018_frontier_certificate_manifest", BEAVER_MANIFEST_REL_PATH, False, "jsonl"),
    (
        "exp3196_module",
        Path("python/carnot/verify/gencp_domain_preview_repair_compiler_v1.py"),
        False,
        "python",
    ),
    (
        "exp3196_script",
        Path("scripts/experiment_3196_gencp_domain_preview_repair_compiler_v1.py"),
        False,
        "python",
    ),
    (
        "exp3196_tests",
        Path("tests/python/test_experiment_3196_gencp_domain_preview_repair_compiler_v1.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3196_gencp_domain_preview_repair_compiler_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/gencp_domain_preview_repair_compiler_v1.py -m pytest -o addopts='' tests/python/test_experiment_3196_gencp_domain_preview_repair_compiler_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/gencp_domain_preview_repair_compiler_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
    preview_limit: int = 12,
) -> JsonDict:
    """REQ-VERIFY-3196: compile bounded repair domains without model calls."""

    root_path = Path(root)
    exp3183 = read_json_object(root_path / EXP3183_REL_PATH)
    exp3195 = read_json_object(root_path / EXP3195_REL_PATH)
    exp3138 = read_json_object(root_path / EXP3138_REL_PATH)
    exp3084 = read_json_object(root_path / EXP3084_REL_PATH)
    exp3018 = read_json_object(root_path / EXP3018_REL_PATH)
    fixture_rows = read_jsonl_objects(root_path / FIXTURE_MANIFEST_REL_PATH)
    beaver_rows = read_jsonl_objects(root_path / BEAVER_MANIFEST_REL_PATH)
    sources = source_artifacts(root_path)
    records = mapping_rows(exp3183.get("certificate_records"))
    frontier_rows = mapping_rows(exp3183.get("bounded_frontier_records"))
    selected = select_preview_records(records, max(0, int(preview_limit)))
    compiler_context = {
        "canonical_replay_by_id": rows_by_id(mapping_rows(exp3138.get("regression_row_replay"))),
        "fixture_by_id": rows_by_id(fixture_rows, key="fixture_id"),
        "policy_by_id": rows_by_id(mapping_rows(exp3195.get("simulated_policy_rows"))),
        "frontier_summary": frontier_summary(frontier_rows, beaver_rows),
        "repair_call_ready": False,
        "upstream_repair_call_ready": exp3183.get("repair_call_ready") is True,
        "upstream_promotion_allowed": exp3195.get("promotion_allowed") is True,
    }
    preview_manifest = [
        build_domain_record(row, compiler_context) for row in selected
    ]
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "compiler_version": COMPILER_VERSION,
        "run_date": RUN_DATE,
        "source_artifacts": sources,
        "source_checksums": {
            row["path"]: row["sha256"] for row in sources if row.get("sha256") is not None
        },
        "source_errors": source_errors(sources),
        "source_schema_observations": source_schema_observations(
            exp3183, exp3195, exp3138, exp3084, exp3018, fixture_rows, beaver_rows
        ),
        "domain_record_schema": domain_record_schema(),
        "propagation_rules": propagation_rules(),
        "exact_rejection_tests": exact_rejection_tests(),
        "preview_domain_count": len(preview_manifest),
        "average_candidate_domain_size": average_domain_size(preview_manifest),
        "preview_manifest": preview_manifest,
        "candidate_domain_accounting": candidate_domain_accounting(preview_manifest),
        "repair_call_ready": False,
        "promotion_allowed": False,
        "no_llm_repair_rationale": no_llm_repair_rationale(),
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
    preview_limit: int = 12,
) -> Path:
    """Persist the schema-versioned preview compiler artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run, preview_limit=preview_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Load a JSON object, returning an empty mapping for absent evidence."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Load JSONL rows and ignore malformed or non-object lines."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def source_artifacts(root: Path) -> list[JsonDict]:
    """Describe every checked-in source that shapes the preview compiler."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        readable_json = None
        if source_type == "json":
            readable_json = bool(read_json_object(path))
        if source_type == "jsonl":
            readable_json = bool(read_jsonl_objects(path))
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_structured_source": readable_json,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose missing required sources while still allowing preview materialization."""

    errors: list[JsonDict] = []
    for row in sources:
        structured = row.get("source_type") in {"json", "jsonl"}
        malformed = structured and row.get("readable_structured_source") is not True
        if row.get("required") is True and row.get("present") is not True:
            errors.append({"path": str(row.get("path")), "reason": "missing_required_source"})
        elif row.get("required") is True and malformed:
            errors.append({"path": str(row.get("path")), "reason": "malformed_required_source"})
    return errors


def sha256_file(path: Path) -> str | None:
    """Hash source files so the preview artifact has auditable lineage."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def mapping_rows(value: Any) -> list[JsonDict]:
    """Normalize a JSON array to object rows only."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def rows_by_id(rows: Sequence[Mapping[str, Any]], *, key: str = "row_id") -> dict[str, JsonDict]:
    """Key source rows by a stable identifier without inventing missing IDs."""

    keyed: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get(key) or "")
        if row_id and row_id not in keyed:
            keyed[row_id] = dict(row)
    return keyed


def select_preview_records(rows: Sequence[Mapping[str, Any]], limit: int) -> list[JsonDict]:
    """Pick a small, risk-first preview set from existing certificate rows."""

    def priority(row: Mapping[str, Any]) -> tuple[int, str]:
        family = str(row.get("counterexample_family") or "")
        pilot = row.get("pilot_certificate")
        if row.get("known_false_accept_or_regression") is True or family.startswith(
            "known_false_accept:"
        ):
            rank = 0
        elif isinstance(pilot, Mapping) and pilot:
            rank = 1
        elif "repair" in family or "fragment" in family:
            rank = 2
        else:
            rank = 3
        return (rank, str(row.get("row_id") or ""))

    selected = sorted((dict(row) for row in rows), key=priority)
    return selected[:limit]


def build_domain_record(row: Mapping[str, Any], context: Mapping[str, Any]) -> JsonDict:
    """Compile one certificate row into a bounded domain preview record."""

    row_id = str(row.get("row_id") or "unknown-row")
    replay = context["canonical_replay_by_id"].get(row_id, {})
    fixture = context["fixture_by_id"].get(row_id, {})
    policy = context["policy_by_id"].get(row_id, {})
    frontier = context["frontier_summary"]
    initial = initial_candidates(row)
    canonical = canonical_candidate(row, replay)
    fixture_values = fixture_domain_values(row, fixture)
    domain, removed, rule_ids, trace = propagate_domain(
        row=row,
        canonical=canonical,
        initial=initial,
        fixture_values=fixture_values,
        frontier=frontier,
        repair_call_ready=context["repair_call_ready"],
    )
    rejection_ids = rejection_test_ids(row, fixture_values, frontier)
    return {
        "record_id": f"gencp-domain:{row_id}",
        "row_id": row_id,
        "source_artifact": str(row.get("source_artifact") or EXP3183_REL_PATH.as_posix()),
        "row_family": str(row.get("counterexample_family") or "unknown"),
        "canonical_answer": canonical,
        "candidate_domain": domain,
        "removed_candidates": removed,
        "domain_size": len(domain),
        "constraint_evidence": constraint_evidence(row, fixture, policy, replay, frontier),
        "propagation_rule_ids": rule_ids,
        "propagation_trace": trace,
        "exact_rejection_test_ids": rejection_ids,
        "authority_note": "preview only; exact/canonical rejection tests remain final",
    }


def initial_candidates(row: Mapping[str, Any]) -> list[str]:
    """Collect observed candidates from the certificate row before propagation."""

    values: list[str] = []
    answers = row.get("candidate_answers")
    for item in answers if isinstance(answers, list) else []:
        append_unique(values, str(item))
    append_unique(values, str(row.get("canonical_answer") or ""))
    append_unique(values, str(row.get("exact_label") or ""))
    return values or ["unknown"]


def canonical_candidate(row: Mapping[str, Any], replay: Mapping[str, Any]) -> str:
    """Prefer canonical replay evidence, then fall back to the certificate answer."""

    exact_canonical = replay.get("exact_canonical")
    if isinstance(exact_canonical, Mapping) and exact_canonical.get("value"):
        return str(exact_canonical["value"])
    return str(row.get("canonical_answer") or row.get("exact_label") or "unknown")


def fixture_domain_values(row: Mapping[str, Any], fixture: Mapping[str, Any]) -> list[str]:
    """Extract bounded fixture-derived values that are safe for preview prompts."""

    payload = fixture.get("authority_payload")
    if isinstance(payload, Mapping) and payload.get("repair"):
        return [str(payload["repair"])]
    label = fixture.get("exact_label")
    family = str(row.get("counterexample_family") or "")
    if isinstance(label, Mapping) and label.get("solver_status"):
        return [str(label["solver_status"]).upper()]
    if isinstance(label, Mapping) and label.get("assertion_passes") is False and "known_false_accept" in family:
        return [str(row.get("canonical_answer") or row.get("exact_label") or "INVALID")]
    return []


def propagate_domain(
    *,
    row: Mapping[str, Any],
    canonical: str,
    initial: Sequence[str],
    fixture_values: Sequence[str],
    frontier: Mapping[str, Any],
    repair_call_ready: bool,
) -> tuple[list[str], list[str], list[str], list[JsonDict]]:
    """Apply deterministic constraint propagation before any future repair call."""

    domain = stable_unique(initial)
    trace: list[JsonDict] = []
    rule_ids: list[str] = []
    removed: list[str] = []
    family = str(row.get("counterexample_family") or "")
    apply_rule(rule_ids, trace, PROP_CANONICAL, "intersect_with_canonical_answer")
    domain, removed = keep_values(domain, [canonical], removed)
    if row.get("known_false_accept_or_regression") is True or family.startswith("known_false_accept:"):
        apply_rule(rule_ids, trace, PROP_FALSE_ACCEPT, "remove_observed_false_accept_candidates")
        domain, removed = keep_values(domain, [canonical], removed)
    if pilot_has_invariant(row):
        apply_rule(rule_ids, trace, PROP_MCS, "pin_minimal_correction_or_unsat_core")
    if fixture_values:
        apply_rule(rule_ids, trace, PROP_FIXTURE, "intersect_with_exact_fixture_authority")
        domain, removed = keep_values(stable_unique(list(domain) + list(fixture_values)), fixture_values, removed)
    if int(frontier.get("pruned_frontier_count") or 0) > 0:
        apply_rule(rule_ids, trace, PROP_FRONTIER, "forbid_frontier_pruned_prefixes")
    if repair_call_ready is False:
        apply_rule(rule_ids, trace, PROP_GATE, "mark_preview_only_until_repair_gate_opens")
    return domain, stable_unique(removed), rule_ids, trace


def keep_values(
    domain: Sequence[str],
    allowed: Sequence[str],
    removed: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return domain values that survive one exact-authority intersection."""

    allowed_set = {value for value in allowed if value}
    kept = [value for value in domain if value in allowed_set]
    next_removed = list(removed) + [value for value in domain if value not in allowed_set]
    return kept or stable_unique(allowed), next_removed


def pilot_has_invariant(row: Mapping[str, Any]) -> bool:
    """Detect whether a pilot certificate carries repair-shaping invariants."""

    pilot = row.get("pilot_certificate")
    return isinstance(pilot, Mapping) and any(
        pilot.get(key) for key in ("mcs", "unsat_core", "expected_corrected_invariant")
    )


def apply_rule(rule_ids: list[str], trace: list[JsonDict], rule_id: str, action: str) -> None:
    """Append a propagation step with stable identifiers for auditability."""

    append_unique(rule_ids, rule_id)
    trace.append({"rule_id": rule_id, "action": action, "authority": "deterministic_artifact"})


def append_unique(values: list[str], value: str) -> None:
    """Keep deterministic lists compact and stable."""

    if value and value not in values:
        values.append(value)


def stable_unique(values: Sequence[str]) -> list[str]:
    """Deduplicate values without reordering source evidence."""

    result: list[str] = []
    for value in values:
        append_unique(result, str(value))
    return result


def rejection_test_ids(
    row: Mapping[str, Any],
    fixture_values: Sequence[str],
    frontier: Mapping[str, Any],
) -> list[str]:
    """List exact predicates that later repair output must survive."""

    ids = [REJECT_CANONICAL, REJECT_PROMOTION, REJECT_GATE]
    family = str(row.get("counterexample_family") or "")
    if row.get("known_false_accept_or_regression") is True or family.startswith("known_false_accept:"):
        ids.append(REJECT_FALSE_ACCEPT)
    if pilot_has_invariant(row):
        ids.append(REJECT_MCS)
    if fixture_values:
        ids.append(REJECT_FIXTURE)
    if int(frontier.get("pruned_frontier_count") or 0) > 0:
        ids.append(REJECT_FRONTIER)
    return ids


def constraint_evidence(
    row: Mapping[str, Any],
    fixture: Mapping[str, Any],
    policy: Mapping[str, Any],
    replay: Mapping[str, Any],
    frontier: Mapping[str, Any],
) -> JsonDict:
    """Preserve the exact evidence that justified domain pruning."""

    pilot = row.get("pilot_certificate") if isinstance(row.get("pilot_certificate"), Mapping) else {}
    return {
        "checker_authority": str(row.get("checker_authority") or "unknown"),
        "checker_result": str(row.get("checker_result") or "unknown"),
        "minimal_correction_set": dict(pilot.get("mcs") or {}),
        "unsat_core": list(pilot.get("unsat_core") or []),
        "expected_corrected_invariant": str(pilot.get("expected_corrected_invariant") or ""),
        "fixture_label_source": str(fixture.get("label_source") or ""),
        "policy_action": str(policy.get("selected_action") or ""),
        "canonical_blockers": list(replay.get("blocked_by") or []),
        "frontier_summary": dict(frontier),
    }


def frontier_summary(
    frontier_rows: Sequence[Mapping[str, Any]],
    beaver_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize invariant/certificate frontiers as pruning context."""

    statuses = [str(row.get("exact_status") or row.get("certificate_status") or "unknown") for row in frontier_rows]
    beaver_statuses = [str(row.get("certificate_status") or "unknown") for row in beaver_rows]
    return {
        "exp3183_frontier_count": len(frontier_rows),
        "frontier_statuses": sorted(set(statuses)),
        "pruned_frontier_count": sum(1 for status in statuses if status == "pruned"),
        "beaver_certificate_count": len(beaver_rows),
        "beaver_certificate_statuses": sorted(set(beaver_statuses)),
    }


def average_domain_size(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Report bounded-domain accounting without inventing a denominator."""

    if not records:
        return None
    return round(
        sum(int(row.get("domain_size") or 0) for row in records) / len(records),
        6,
    )


def candidate_domain_accounting(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate the preview scope by family and domain size."""

    by_family: dict[str, int] = {}
    for row in records:
        family = str(row.get("row_family") or "unknown")
        by_family[family] = by_family.get(family, 0) + 1
    return {
        "record_count": len(records),
        "average_candidate_domain_size": average_domain_size(records),
        "families": dict(sorted(by_family.items())),
    }


def source_schema_observations(
    exp3183: Mapping[str, Any],
    exp3195: Mapping[str, Any],
    exp3138: Mapping[str, Any],
    exp3084: Mapping[str, Any],
    exp3018: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
    beaver_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Record the row schemas inspected by the compiler."""

    return {
        "exp3183_certificate_record_keys": sorted_keys(exp3183.get("certificate_records")),
        "exp3183_frontier_record_keys": sorted_keys(exp3183.get("bounded_frontier_records")),
        "exp3195_policy_row_keys": sorted_keys(exp3195.get("simulated_policy_rows")),
        "exp3138_regression_replay_keys": sorted_keys(exp3138.get("regression_row_replay")),
        "exp3084_top_level_keys": sorted(exp3084.keys()),
        "fixture_manifest_row_keys": sorted(fixture_rows[0].keys()) if fixture_rows else [],
        "exp3018_top_level_keys": sorted(exp3018.keys()),
        "beaver_manifest_row_keys": sorted(beaver_rows[0].keys()) if beaver_rows else [],
    }


def sorted_keys(value: Any) -> list[str]:
    """Return keys from the first object in a list-shaped source."""

    rows = mapping_rows(value)
    return sorted(rows[0].keys()) if rows else []


def domain_record_schema() -> JsonDict:
    """Describe the machine-readable records emitted in the preview manifest."""

    return {
        "type": "object",
        "required": [
            "record_id",
            "row_id",
            "source_artifact",
            "row_family",
            "canonical_answer",
            "candidate_domain",
            "domain_size",
            "constraint_evidence",
            "propagation_rule_ids",
            "exact_rejection_test_ids",
            "authority_note",
        ],
        "properties": {
            "candidate_domain": "bounded list of deterministic candidate values",
            "constraint_evidence": "canonical, fixture, MCS, unsat-core, and frontier evidence",
            "propagation_trace": "ordered GenCP-style pruning steps applied before repair",
            "authority_note": "must state preview-only authority boundary",
        },
    }


def propagation_rules() -> list[JsonDict]:
    """Return the explicit constraint-propagation table used by the compiler."""

    return [
        {
            "id": PROP_CANONICAL,
            "name": "canonical answer intersection",
            "effect": "retain candidates equivalent to the exact canonical answer",
        },
        {
            "id": PROP_FALSE_ACCEPT,
            "name": "false-accept conflict elimination",
            "effect": "remove observed candidates that created known false accepts",
        },
        {
            "id": PROP_MCS,
            "name": "minimal correction or unsat-core preservation",
            "effect": "carry MCS, invariant, and unsat-core constraints into later rejection tests",
        },
        {
            "id": PROP_FIXTURE,
            "name": "fixture authority compatibility",
            "effect": "intersect repair domains with exact fixture labels or repairs",
        },
        {
            "id": PROP_FRONTIER,
            "name": "frontier status pruning",
            "effect": "forbid prefixes already marked pruned by bounded frontier evidence",
        },
        {
            "id": PROP_GATE,
            "name": "repair gate block",
            "effect": "mark records preview-only until a separate repair gate opens",
        },
    ]


def exact_rejection_tests() -> list[JsonDict]:
    """Define exact predicates that remain final for later repair attempts."""

    return [
        {
            "id": REJECT_CANONICAL,
            "reject_when": "candidate canonical form differs from exact canonical answer",
            "authority_source": "Exp 3138 canonical grounding and Exp 3183 exact labels",
        },
        {
            "id": REJECT_FALSE_ACCEPT,
            "reject_when": "candidate repeats an observed false-accept answer family conflict",
            "authority_source": "Exp 3183 known false-accept certificates",
        },
        {
            "id": REJECT_MCS,
            "reject_when": "candidate violates the recorded MCS, invariant, or unsat core",
            "authority_source": "Exp 3170/3183 pilot certificate fields",
        },
        {
            "id": REJECT_FIXTURE,
            "reject_when": "candidate contradicts exact fixture authority payload or label source",
            "authority_source": "Exp 3084 fixture manifest",
        },
        {
            "id": REJECT_FRONTIER,
            "reject_when": "candidate uses a prefix or state already marked pruned",
            "authority_source": "Exp 3183 and Exp 3018 bounded frontier certificates",
        },
        {
            "id": REJECT_PROMOTION,
            "reject_when": "candidate treats receipts, sidecars, EBM scores, or LLM text as authority",
            "authority_source": "Carnot repair-gate authority boundary",
        },
        {
            "id": REJECT_GATE,
            "reject_when": "candidate is submitted while repair_call_ready is false",
            "authority_source": "Exp 3183/3184/3195 gate state",
        },
    ]


def no_llm_repair_rationale() -> JsonDict:
    """State why this artifact does not unlock or run repair generation."""

    return {
        "why_no_repair_call": "domain preview only; it prepares bounded spaces for later prompts",
        "blocked_gate_respected": True,
        "exact_authority_final": True,
        "llm_or_ebm_promotion_claim": False,
    }


def inference_substrate() -> JsonDict:
    """Make the no-new-execution boundary machine-readable."""

    return {
        "kind": "artifact_only_domain_preview_compiler",
        "llm_called": False,
        "new_repair_calls": 0,
        "new_live_model_calls": 0,
        "new_verifier_scoring_calls": 0,
        "ebm_trained_or_promoted": False,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal truthful verdict while keeping repair blocked."""

    return (
        "complete: gencp_domain_preview_repair_compiler_v1_ready=true; "
        f"preview_domain_count={artifact['preview_domain_count']}; "
        f"average_candidate_domain_size={artifact['average_candidate_domain_size']}; "
        "repair_call_ready=false; promotion_allowed=false"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema omissions, promotion, or empty preview domains."""

    missing = REQUIRED_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("repair_call_ready") is not False:
        raise ValueError("repair_call_ready must remain false")
    if artifact.get("promotion_allowed") is not False:
        raise ValueError("promotion_allowed must remain false")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with 'complete:'")
    for row in artifact.get("preview_manifest", []):
        if not isinstance(row, Mapping):
            raise ValueError("preview_manifest rows must be objects")
        domain = row.get("candidate_domain")
        if not isinstance(domain, list) or not domain:
            raise ValueError("preview records must include a non-empty candidate_domain")
        if int(row.get("domain_size") or 0) != len(domain):
            raise ValueError("domain_size must equal candidate_domain length")
        if not row.get("exact_rejection_test_ids"):
            raise ValueError("preview records must include exact rejection tests")


def main() -> None:  # pragma: no cover
    """CLI entrypoint for the checked-in experiment script."""

    print(write_artifact().as_posix())


if __name__ == "__main__":  # pragma: no cover
    main()
