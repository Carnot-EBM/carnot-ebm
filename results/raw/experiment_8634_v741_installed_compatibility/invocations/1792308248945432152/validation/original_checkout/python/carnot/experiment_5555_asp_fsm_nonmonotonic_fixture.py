"""Exp5555 deterministic ASP/FSM nonmonotonic exact fixture.

Spec refs: REQ-VERIFY-5555, SCENARIO-VERIFY-5555.

The experiment extends the Exp5541 exact finite-state fixture with a tiny
Answer Set Programming style layer. The evaluator is deliberately narrow: it
handles finite propositional facts, normal rules, hard constraints, and
default negation by exact stable-model enumeration. That is enough for these
fixture rows and avoids pretending that a full ASP solver or an LLM was used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod


JsonDict = dict[str, Any]
Rule = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5555_asp_fsm_nonmonotonic_fixture.json")
UPSTREAM_FSM_FIXTURE = fsm_mod.RESULT_RELATIVE_PATH
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")

SCHEMA = "carnot.experiment_5555.asp_fsm_nonmonotonic_fixture.v503"
EXPERIMENT = 5555
EXPERIMENT_ID = "exp5555-asp-fsm-nonmonotonic-fixture"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5555
INFERENCE_SUBSTRATE = "deterministic_asp_fsm_exact_fixture_no_llm"
ASP_SOLVER_BACKEND = "tiny_fixture_local_stable_model_enumerator"
SPEC_REFS = ("REQ-VERIFY-5555", "SCENARIO-VERIFY-5555", "REQ-VERIFY-5541")

REQUIRED_ARTIFACT_FIELDS = (
    "upstream_fsm_fixture",
    "llm_invoked",
    "no_model_specs_required",
    "asp_row_count",
    "default_rule_count",
    "contradiction_row_count",
    "stable_model_count",
    "sat_count",
    "unsat_count",
    "ambiguous_count",
    "exact_asp_validator_ready",
    "exact_fsm_fixture_extended_ready",
    "spec_files_updated_or_confirmed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)

TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5555_asp_fsm_nonmonotonic_fixture.py",
    "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
)

SPEC_FILES_UPDATED_OR_CONFIRMED = ("openspec/capabilities/verification/spec.md",)

FIELD_PRINCIPLES: JsonDict = {
    "upstream_fsm_fixture": "Pins the ASP rows to the exact FSM substrate they extend.",
    "llm_invoked": "Prevents stable-model validation from being mistaken for live model inference.",
    "no_model_specs_required": "Confirms the deterministic validator has no model dependency to disclose.",
    "asp_row_count": "Keeps the nonmonotonic fixture denominator visible.",
    "default_rule_count": "Counts default-negation coverage instead of hiding it in prose.",
    "contradiction_row_count": "Preserves explicit hard-conflict controls as first-class evidence.",
    "stable_model_count": "Records the exact stable-model evidence across all rows.",
    "sat_count": "Counts one-model rows separately from nulls and ambiguity.",
    "unsat_count": "Counts no-stable-model rows without collapsing them into parser failures.",
    "ambiguous_count": "Counts multi-stable-model rows so underdetermination stays visible.",
    "exact_asp_validator_ready": "Opens only when every ASP row matches its expected stable-model class.",
    "exact_fsm_fixture_extended_ready": "Opens only when the upstream FSM fixture is ready and the ASP gate is clean.",
    "spec_files_updated_or_confirmed": "Shows which OpenSpec contract anchors the artifact.",
    "tests_added_or_reused": "Names the focused tests and reused upstream FSM tests.",
    "field_principles": "Explains why every headline and gate field exists.",
    "inference_substrate": "Declares deterministic ASP/FSM exact validation with no LLM.",
    "honest_verdict": "Provides a terminal evidence boundary without a model quality claim.",
}

SOLVER_BOUNDARY = {
    "supported": [
        "finite propositional facts",
        "normal single-head rules",
        "hard constraints with no head",
        "default negation in rule bodies",
        "exact finite stable-model enumeration",
    ],
    "unsupported": [
        "aggregates",
        "optimization statements",
        "arithmetic terms",
        "variables",
        "external predicates",
        "disjunctive heads",
    ],
}

ATOM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SOLVER_STATUSES = ("satisfiable", "unsatisfiable", "ambiguous")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so checksums stay reviewable."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def asp_rule(
    rule_id: str,
    head: str | None,
    *,
    positive: Sequence[str] = (),
    default_negated: Sequence[str] = (),
) -> Rule:
    """Build one limited ASP-style rule row.

    A ``None`` head means a hard constraint. Default-negated atoms use the ASP
    ``not atom`` convention and are interpreted only by the stable-model
    reduct, not by a text scorer.
    """

    return {
        "rule_id": rule_id,
        "head": head,
        "positive": list(positive),
        "default_negated": list(default_negated),
    }


def build_asp_fixture_rows(upstream_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Build five ASP rows that reuse Exp5541 exact FSM facts."""

    facts = fsm_facts(upstream_artifact)
    sat = facts["fsm_sat_accept_error"]
    unsat = facts["fsm_unsat_conflicting_transition"]
    ambiguous = facts["fsm_ambiguous_sparse_branch"]
    return [
        {
            "row_id": "asp_sat_fsm_acceptance_default_guard",
            "description": "Satisfiable row guarded by an accepted trace and default absence of a false accept.",
            "fsm_instance_id": "fsm_sat_accept_error",
            "facts": [
                sat["status_satisfiable"],
                sat["trace_sat_b_accepts_accepted"],
                sat["trace_sat_a_errors_error"],
                sat["trace_sat_empty_rejects_rejected"],
            ],
            "rules": [
                asp_rule(
                    "ASP_SAT_00",
                    "sat_guarded_accept",
                    positive=(sat["status_satisfiable"], sat["trace_sat_b_accepts_accepted"]),
                    default_negated=(sat["trace_sat_a_errors_accepted"],),
                ),
                asp_rule(
                    "ASP_SAT_01",
                    None,
                    positive=("sat_guarded_accept", sat["trace_sat_empty_rejects_accepted"]),
                ),
            ],
            "expected_status": "satisfiable",
            "contradiction_row": False,
        },
        {
            "row_id": "asp_unsat_fsm_forbidden_error",
            "description": "Unsatisfiable row where an upstream FSM contradiction triggers a hard constraint.",
            "fsm_instance_id": "fsm_unsat_conflicting_transition",
            "facts": [unsat["status_unsatisfiable"], unsat["has_contradiction"]],
            "rules": [
                asp_rule(
                    "ASP_UNSAT_00",
                    "unsat_error_seen",
                    positive=(unsat["status_unsatisfiable"], unsat["has_contradiction"]),
                ),
                asp_rule("ASP_UNSAT_01", None, positive=("unsat_error_seen",)),
            ],
            "expected_status": "unsatisfiable",
            "contradiction_row": False,
        },
        {
            "row_id": "asp_ambiguous_fsm_default_repair_choice",
            "description": "Ambiguous row with two default-negation repair choices over the sparse FSM branch.",
            "fsm_instance_id": "fsm_ambiguous_sparse_branch",
            "facts": [
                ambiguous["status_ambiguous"],
                ambiguous["trace_amb_go_go_underconstrained_underdetermined"],
            ],
            "rules": [
                asp_rule(
                    "ASP_AMB_00",
                    "repair_choose_accept",
                    positive=(
                        ambiguous["status_ambiguous"],
                        ambiguous["trace_amb_go_go_underconstrained_underdetermined"],
                    ),
                    default_negated=("repair_choose_reject",),
                ),
                asp_rule(
                    "ASP_AMB_01",
                    "repair_choose_reject",
                    positive=(
                        ambiguous["status_ambiguous"],
                        ambiguous["trace_amb_go_go_underconstrained_underdetermined"],
                    ),
                    default_negated=("repair_choose_accept",),
                ),
                asp_rule(
                    "ASP_AMB_02",
                    None,
                    positive=("repair_choose_accept", "repair_choose_reject"),
                ),
            ],
            "expected_status": "ambiguous",
            "contradiction_row": False,
        },
        {
            "row_id": "asp_default_negation_no_exception",
            "description": "Default-negation row where no exception fact derives the safe default.",
            "fsm_instance_id": "fsm_sat_accept_error",
            "facts": [sat["status_satisfiable"]],
            "rules": [
                asp_rule(
                    "ASP_DEFAULT_00",
                    "default_safe_accept",
                    positive=(sat["status_satisfiable"],),
                    default_negated=("exception_seen",),
                ),
                asp_rule(
                    "ASP_DEFAULT_01", "default_trace_accept", positive=("default_safe_accept",)
                ),
                asp_rule("ASP_DEFAULT_02", None, positive=("exception_seen",)),
            ],
            "expected_status": "satisfiable",
            "contradiction_row": False,
        },
        {
            "row_id": "asp_contradiction_fact_constraint",
            "description": "Contradiction row where an explicit fact is forbidden by a hard constraint.",
            "fsm_instance_id": "fsm_sat_accept_error",
            "facts": ["contradiction_fact"],
            "rules": [asp_rule("ASP_CONTRA_00", None, positive=("contradiction_fact",))],
            "expected_status": "unsatisfiable",
            "contradiction_row": True,
        },
    ]


def fsm_facts(upstream_artifact: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Translate Exp5541 exact reports into finite propositional fact names."""

    facts: dict[str, dict[str, str]] = {}
    for report in upstream_artifact.get("exact_check_reports", []):
        if not isinstance(report, Mapping):
            continue
        instance_id = str(report.get("instance_id", ""))
        prefix = _atomize(instance_id)
        row: dict[str, str] = {}
        status = _atomize(str(report.get("solver_status", "")))
        row[f"status_{status}"] = f"{prefix}_status_{status}"
        if report.get("contradictions"):
            row["has_contradiction"] = f"{prefix}_has_contradiction"
        for trace in report.get("trace_checks", []):
            if not isinstance(trace, Mapping):
                continue
            trace_id = _atomize(str(trace.get("trace_id", "")))
            label = _atomize(str(trace.get("actual_label", "")))
            row[f"trace_{trace_id}_{label}"] = f"{prefix}_trace_{trace_id}_{label}"
            for possible in ("accepted", "rejected", "error", "underdetermined", "contradiction"):
                row.setdefault(
                    f"trace_{trace_id}_{possible}",
                    f"{prefix}_trace_{trace_id}_{possible}",
                )
        facts[instance_id] = row
    return facts


def evaluate_asp_row(row: Mapping[str, Any]) -> JsonDict:
    """Evaluate one limited ASP row by exact stable-model enumeration."""

    facts = [str(atom) for atom in row.get("facts", [])]
    rules = [dict(rule) for rule in row.get("rules", [])]
    _validate_rule_atoms(facts, rules)
    fact_rules = [asp_rule(f"FACT_{atom}", atom) for atom in facts]
    program = [*fact_rules, *rules]
    atoms = sorted(_program_atoms(program))
    stable_models: list[list[str]] = []
    violated_constraint_count = 0
    unsupported_candidate_count = 0
    for candidate in _powerset(atoms):
        result = _candidate_stable_result(program, set(candidate))
        if result["stable"]:
            stable_models.append(sorted(candidate))
        elif result["reason"] == "constraint_violated":
            violated_constraint_count += 1
        elif result["reason"] == "not_least_model":
            unsupported_candidate_count += 1
    solver_status = _solver_status(stable_models)
    expected_status = str(row.get("expected_status", solver_status))
    default_rule_count = sum(1 for rule in rules if rule.get("default_negated"))
    return {
        "row_id": str(row.get("row_id", "")),
        "description": str(row.get("description", "")),
        "fsm_instance_id": str(row.get("fsm_instance_id", "")),
        "expected_status": expected_status,
        "solver_status": solver_status,
        "status_matches_expected": solver_status == expected_status,
        "stable_model_count": len(stable_models),
        "stable_model_samples": stable_models,
        "atom_count": len(atoms),
        "atoms": atoms,
        "rule_count": len(rules),
        "default_rule_count": default_rule_count,
        "contains_default_negation": default_rule_count > 0,
        "constraint_rule_count": sum(1 for rule in rules if rule.get("head") is None),
        "violated_constraint_count": violated_constraint_count,
        "unsupported_candidate_count": unsupported_candidate_count,
        "contradiction_row": bool(row.get("contradiction_row") is True),
    }


def stable_models(program: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Return stable models for a finite propositional normal program."""

    _validate_rule_atoms((), program)
    atoms = sorted(_program_atoms(program))
    models: list[list[str]] = []
    for candidate in _powerset(atoms):
        if _candidate_stable_result(program, set(candidate))["stable"] is True:
            models.append(sorted(candidate))
    return models


def build_artifact(
    *,
    upstream_path: Path = REPO_ROOT / UPSTREAM_FSM_FIXTURE,
    upstream_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5555 deterministic ASP/FSM result artifact."""

    upstream = (
        dict(upstream_artifact) if upstream_artifact is not None else _load_json(upstream_path)
    )
    upstream_ready = upstream.get("exact_fsm_fixture_ready") is True
    rows = build_asp_fixture_rows(upstream) if upstream.get("exact_check_reports") else []
    reports = [evaluate_asp_row(row) for row in rows]
    counts = _status_counts(reports)
    default_rule_count = sum(int(report["default_rule_count"]) for report in reports)
    stable_model_count = sum(int(report["stable_model_count"]) for report in reports)
    contradiction_row_count = sum(int(report["contradiction_row"] is True) for report in reports)
    exact_asp_ready = bool(
        reports
        and all(report["status_matches_expected"] for report in reports)
        and counts["satisfiable"] >= 1
        and counts["unsatisfiable"] >= 1
        and counts["ambiguous"] >= 1
        and default_rule_count >= 1
        and contradiction_row_count >= 1
    )
    extended_ready = bool(upstream_ready and exact_asp_ready)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "upstream_fsm_fixture": UPSTREAM_FSM_FIXTURE.as_posix(),
        "llm_invoked": False,
        "no_model_specs_required": True,
        "asp_row_count": len(rows),
        "default_rule_count": default_rule_count,
        "contradiction_row_count": contradiction_row_count,
        "stable_model_count": stable_model_count,
        "sat_count": counts["satisfiable"],
        "unsat_count": counts["unsatisfiable"],
        "ambiguous_count": counts["ambiguous"],
        "exact_asp_validator_ready": exact_asp_ready,
        "exact_fsm_fixture_extended_ready": extended_ready,
        "spec_files_updated_or_confirmed": list(SPEC_FILES_UPDATED_OR_CONFIRMED),
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(extended_ready, upstream_ready=upstream_ready),
        "asp_solver_backend": ASP_SOLVER_BACKEND,
        "solver_boundary": dict(SOLVER_BOUNDARY),
        "upstream_fsm_fixture_ready": upstream_ready,
        "asp_fixture_rows": rows,
        "stable_model_reports": reports,
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
    upstream_path: Path = REPO_ROOT / UPSTREAM_FSM_FIXTURE,
    upstream_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5555 deliverable JSON."""

    artifact = build_artifact(
        upstream_path=upstream_path,
        upstream_artifact=upstream_artifact,
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
    """Validate the artifact and fail closed on hidden model or readiness claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("upstream_fsm_fixture") == UPSTREAM_FSM_FIXTURE.as_posix(),
        "upstream_fsm_fixture",
    )
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("no_model_specs_required") is True, "no_model_specs_required")
    _require("model_specs" not in artifact, "model_specs")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(
        str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")),
        "honest_verdict",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        artifact.get("tests_added_or_reused") == list(TESTS_ADDED_OR_REUSED),
        "tests_added_or_reused",
    )
    _require(
        artifact.get("spec_files_updated_or_confirmed") == list(SPEC_FILES_UPDATED_OR_CONFIRMED),
        "spec_files_updated_or_confirmed",
    )
    reports = list(artifact.get("stable_model_reports", []))
    _require(int(artifact.get("asp_row_count", -1)) == len(reports), "asp_row_count")
    _require(
        int(artifact.get("default_rule_count", -1))
        == sum(int(report.get("default_rule_count", 0)) for report in reports),
        "default_rule_count",
    )
    _require(
        int(artifact.get("contradiction_row_count", -1))
        == sum(int(report.get("contradiction_row") is True) for report in reports),
        "contradiction_row_count",
    )
    _require(
        int(artifact.get("stable_model_count", -1))
        == sum(int(report.get("stable_model_count", 0)) for report in reports),
        "stable_model_count",
    )
    counts = _status_counts(reports)
    _require(int(artifact.get("sat_count", -1)) == counts["satisfiable"], "sat_count")
    _require(int(artifact.get("unsat_count", -1)) == counts["unsatisfiable"], "unsat_count")
    _require(int(artifact.get("ambiguous_count", -1)) == counts["ambiguous"], "ambiguous_count")
    if artifact.get("exact_asp_validator_ready") is True:
        _require(bool(reports), "exact_asp_validator_ready")
        _require(
            all(report.get("status_matches_expected") is True for report in reports),
            "exact_asp_validator_ready",
        )
        _require(int(artifact.get("sat_count", 0)) >= 1, "sat_count")
        _require(int(artifact.get("unsat_count", 0)) >= 1, "unsat_count")
        _require(int(artifact.get("ambiguous_count", 0)) >= 1, "ambiguous_count")
        _require(int(artifact.get("default_rule_count", 0)) >= 1, "default_rule_count")
        _require(int(artifact.get("contradiction_row_count", 0)) >= 1, "contradiction_row_count")
    if artifact.get("exact_fsm_fixture_extended_ready") is True:
        _require(artifact.get("upstream_fsm_fixture_ready") is True, "upstream_fsm_fixture_ready")
        _require(artifact.get("exact_asp_validator_ready") is True, "exact_asp_validator_ready")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    else:
        _require(str(artifact.get("honest_verdict", "")).startswith("blocked:"), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, *, upstream_ready: bool) -> str:
    """Return a terminal verdict that cannot imply live model quality."""

    if ready:
        return "complete: exact ASP/FSM stable-model fixture ready with no LLM"
    if not upstream_ready:
        return "blocked: upstream exact FSM fixture not ready for ASP extension"
    return "blocked: ASP stable-model validation gate failed"


def _candidate_stable_result(program: Sequence[Mapping[str, Any]], candidate: set[str]) -> JsonDict:
    reduct = [rule for rule in program if not (set(rule.get("default_negated", [])) & candidate)]
    non_constraint_rules = [rule for rule in reduct if rule.get("head") is not None]
    constraint_rules = [rule for rule in reduct if rule.get("head") is None]
    least = _least_model(non_constraint_rules)
    if least != candidate:
        return {"stable": False, "reason": "not_least_model", "least_model": sorted(least)}
    if any(set(rule.get("positive", [])).issubset(candidate) for rule in constraint_rules):
        return {"stable": False, "reason": "constraint_violated", "least_model": sorted(least)}
    return {"stable": True, "reason": "stable", "least_model": sorted(least)}


def _least_model(rules: Sequence[Mapping[str, Any]]) -> set[str]:
    model: set[str] = set()
    changed = True
    while changed:
        changed = False
        for rule in rules:
            head = rule.get("head")
            if head is None:
                continue
            if set(rule.get("positive", [])).issubset(model) and str(head) not in model:
                model.add(str(head))
                changed = True
    return model


def _program_atoms(program: Sequence[Mapping[str, Any]]) -> set[str]:
    atoms: set[str] = set()
    for rule in program:
        head = rule.get("head")
        if head is not None:
            atoms.add(str(head))
        atoms.update(str(atom) for atom in rule.get("positive", []))
        atoms.update(str(atom) for atom in rule.get("default_negated", []))
    return atoms


def _powerset(atoms: Sequence[str]) -> list[tuple[str, ...]]:
    return [subset for size in range(len(atoms) + 1) for subset in combinations(atoms, size)]


def _solver_status(stable_model_rows: Sequence[Sequence[str]]) -> str:
    if not stable_model_rows:
        return "unsatisfiable"
    return "satisfiable" if len(stable_model_rows) == 1 else "ambiguous"


def _status_counts(reports: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for report in reports if report.get("solver_status") == status)
        for status in SOLVER_STATUSES
    }


def _validate_rule_atoms(facts: Sequence[str], rules: Sequence[Mapping[str, Any]]) -> None:
    atoms = set(facts)
    atoms.update(_program_atoms(rules))
    invalid = sorted(atom for atom in atoms if not ATOM_RE.fullmatch(str(atom)))
    if invalid:
        raise ValueError(f"unsupported_atom:{invalid[0]}")


def _atomize(value: str) -> str:
    atom = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    atom = re.sub(r"_+", "_", atom).strip("_")
    if not atom:
        atom = "empty"
    if not atom[0].isalpha():
        atom = f"a_{atom}"
    return atom


def _load_json(path: Path) -> JsonDict:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"load_error": "missing", "path": path.as_posix()}
    except json.JSONDecodeError as exc:
        return {"load_error": "json_decode", "path": path.as_posix(), "detail": str(exc)}
    if not isinstance(decoded, dict):
        return {"load_error": "json_not_object", "path": path.as_posix()}
    return decoded


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "exact_fsm_fixture_extended_ready": artifact["exact_fsm_fixture_extended_ready"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
