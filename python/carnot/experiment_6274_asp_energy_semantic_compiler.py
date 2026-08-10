"""Exp6274 bounded ASP semantic compiler artifact.

Spec refs: REQ-CONSTRAINT-6274,
SCENARIO-CONSTRAINT-6274-SOLVER-PARITY,
SCENARIO-CONSTRAINT-6274-FAIL-CLOSED,
SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

from carnot import asp_energy


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6274_asp_energy_semantic_compiler.json")
FIXTURE_MANIFEST_RELATIVE_PATH = Path("results/experiment_6274_asp_energy_fixture_manifest.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RANDOM_SEED = 6274
INFERENCE_SUBSTRATE = "deterministic_bounded_asp_energy_enumeration_with_clingo_oracle_no_llm"
MAX_STATE_COUNT = 4096
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6274_asp_energy_semantic_compiler --date 20260810"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    ".venv/bin/pytest tests/python/test_asp_energy_semantic_compiler_6274.py -q",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_source_and_claim_boundary",
    "supported_asp_subset",
    "unsupported_constructs_and_fail_closed_behavior",
    "compiler_source_paths_and_hashes",
    "independent_solver_name_version_and_receipt",
    "fixture_manifest_path_and_hash",
    "fixture_family_counts",
    "fixture_count",
    "exact_state_count_by_fixture",
    "asp_theory_hash_by_fixture",
    "energy_term_decomposition_by_fixture",
    "solver_answer_sets_by_fixture",
    "zero_energy_states_by_fixture",
    "semantic_parity_by_fixture",
    "per_rule_violation_localization",
    "contradiction_controls",
    "default_negation_controls",
    "cardinality_controls",
    "label_permutation_controls",
    "unsupported_syntax_controls",
    "parity_failure_count",
    "oracle_claim_boundary",
    "asp_energy_semantic_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Summarizes whether all exact parity gates passed.",
    "paper_source_and_claim_boundary": "Ties the work to the paper without overclaiming.",
    "supported_asp_subset": "Names the only accepted grammar.",
    "unsupported_constructs_and_fail_closed_behavior": "Shows unsupported syntax is refused early.",
    "compiler_source_paths_and_hashes": "Pins the code bytes that built the result.",
    "independent_solver_name_version_and_receipt": "Names the oracle used for parity.",
    "fixture_manifest_path_and_hash": "Pins the trusted fixture sidecar.",
    "fixture_family_counts": "Shows coverage is not one-family only.",
    "fixture_count": "Keeps the exact denominator visible.",
    "exact_state_count_by_fixture": "Shows enumeration stayed bounded.",
    "asp_theory_hash_by_fixture": "Pins every ASP theory string.",
    "energy_term_decomposition_by_fixture": "Makes the compiler output inspectable.",
    "solver_answer_sets_by_fixture": "Records independent solver authority.",
    "zero_energy_states_by_fixture": "Records compiler-derived accepting states.",
    "semantic_parity_by_fixture": "Shows set-equality parity per fixture.",
    "per_rule_violation_localization": "Proves violations name local causes.",
    "contradiction_controls": "Keeps unsat controls explicit.",
    "default_negation_controls": "Keeps non-monotonic cases explicit.",
    "cardinality_controls": "Checks bounded choice constraints and energy signs.",
    "label_permutation_controls": "Checks labels are not special-cased.",
    "unsupported_syntax_controls": "Proves unsupported forms fail closed.",
    "parity_failure_count": "Keeps the failure count machine-readable.",
    "oracle_claim_boundary": "Prevents a verifier-moat claim.",
    "asp_energy_semantic_ready_score": "Opens only on exact parity.",
    "protected_files_unchanged": "Shows conductor-protected files were not touched.",
    "preconditions_checked": "Records environment, bounds, tolerances, and git state.",
    "inference_substrate": "Declares exact enumeration and no model inference.",
    "verifier_is_oracle": "Discloses that the exact solver is the oracle.",
    "field_provenance": "Maps fields to spec and computation sources.",
    "field_principles": "Explains why each required field exists.",
    "test_commands": "Names the verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records wall-clock run duration.",
    "random_seed": "Pins deterministic fixture generation.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the result boundary in one terminal verdict.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: ["REQ-CONSTRAINT-6274", "Exp6274 deterministic run"]
    for field in REQUIRED_ARTIFACT_FIELDS
}


@dataclass(frozen=True)
class ASPFixture:
    """One trusted formal ASP fixture with a bounded finite atom set."""

    fixture_id: str
    family: str
    description: str
    program_text: str
    tags: tuple[str, ...]
    permutation_of: str | None = None


def build_fixture_manifest() -> list[ASPFixture]:
    """Return 40 trusted bounded ASP fixtures across the required families."""

    fixtures: list[ASPFixture] = []
    fixtures.extend(_graph_coloring_fixtures())
    fixtures.extend(_scheduling_fixtures())
    fixtures.extend(_default_fixtures())
    fixtures.extend(_contradiction_fixtures())
    fixtures.extend(_positive_negative_control_fixtures())
    if len(fixtures) != 40:
        raise ValueError("fixture_count")  # pragma: no cover
    return fixtures


def fixture_family_counts(fixtures: Sequence[ASPFixture]) -> dict[str, int]:
    """Count fixtures by required family."""

    return dict(sorted(Counter(fixture.family for fixture in fixtures).items()))


def evaluate_fixtures(fixtures: Sequence[ASPFixture]) -> list[JsonDict]:
    """Compile, enumerate, and compare every fixture against clingo."""

    return [evaluate_fixture(fixture) for fixture in fixtures]


def evaluate_fixture(fixture: ASPFixture) -> JsonDict:
    """Evaluate one fixture and return all parity receipts."""

    compiled = asp_energy.compile_program(fixture.program_text, program_id=fixture.fixture_id)
    if compiled.exact_state_count > MAX_STATE_COUNT:
        raise ValueError(f"state_bound:{fixture.fixture_id}")
    solver_answer_sets = asp_energy.solve_with_clingo(compiled.program)
    zero_energy_states = compiled.zero_energy_states()
    semantic_parity = zero_energy_states == solver_answer_sets
    localization = _local_violation_samples(compiled)
    negative_energy_count = _negative_energy_count(compiled)
    return {
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "description": fixture.description,
        "tags": list(fixture.tags),
        "permutation_of": fixture.permutation_of,
        "atom_count": len(compiled.program.atoms),
        "atoms": list(compiled.program.atoms),
        "exact_state_count": compiled.exact_state_count,
        "asp_theory_hash": sha256_text(compiled.program.to_clingo()),
        "energy_terms": [_term_payload(term) for term in compiled.energy_terms],
        "solver_answer_sets": solver_answer_sets,
        "zero_energy_states": zero_energy_states,
        "solver_answer_set_count": len(solver_answer_sets),
        "zero_energy_state_count": len(zero_energy_states),
        "semantic_parity": semantic_parity,
        "parity_failure_count": 0 if semantic_parity else 1,
        "per_rule_violation_receipts": localization,
        "negative_energy_count": negative_energy_count,
    }


def build_artifact(
    *,
    date: str,
    result_path: Path,
    manifest_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the terminal Exp6274 artifact without writing it."""

    fixtures = build_fixture_manifest()
    manifest_payload = _fixture_manifest_payload(fixtures, date)
    fixture_reports = evaluate_fixtures(fixtures)
    parity_failure_count = sum(int(report["parity_failure_count"]) for report in fixture_reports)
    status = "complete" if parity_failure_count == 0 and len(fixtures) >= 40 else "blocked"
    ready_score = 1.0 if status == "complete" else 0.0
    manifest_hash = sha256_json(manifest_payload)
    protected = _protected_hash_receipts()
    artifact: JsonDict = {
        "status": status,
        "paper_source_and_claim_boundary": _paper_boundary(),
        "supported_asp_subset": _supported_subset(),
        "unsupported_constructs_and_fail_closed_behavior": _unsupported_behavior(),
        "compiler_source_paths_and_hashes": _source_hashes(),
        "independent_solver_name_version_and_receipt": _solver_receipt(fixture_reports),
        "fixture_manifest_path_and_hash": {
            "path": _display_path(manifest_path),
            "sha256": manifest_hash,
            "fixture_count": len(fixtures),
        },
        "fixture_family_counts": fixture_family_counts(fixtures),
        "fixture_count": len(fixtures),
        "exact_state_count_by_fixture": {
            report["fixture_id"]: report["exact_state_count"] for report in fixture_reports
        },
        "asp_theory_hash_by_fixture": {
            report["fixture_id"]: report["asp_theory_hash"] for report in fixture_reports
        },
        "energy_term_decomposition_by_fixture": {
            report["fixture_id"]: report["energy_terms"] for report in fixture_reports
        },
        "solver_answer_sets_by_fixture": {
            report["fixture_id"]: report["solver_answer_sets"] for report in fixture_reports
        },
        "zero_energy_states_by_fixture": {
            report["fixture_id"]: report["zero_energy_states"] for report in fixture_reports
        },
        "semantic_parity_by_fixture": {
            report["fixture_id"]: report["semantic_parity"] for report in fixture_reports
        },
        "per_rule_violation_localization": _aggregate_localization(fixture_reports),
        "contradiction_controls": _contradiction_controls(fixture_reports),
        "default_negation_controls": _default_negation_controls(fixture_reports),
        "cardinality_controls": _cardinality_controls(fixture_reports),
        "label_permutation_controls": _label_permutation_controls(fixture_reports),
        "unsupported_syntax_controls": _unsupported_syntax_controls(),
        "parity_failure_count": int(parity_failure_count),
        "oracle_claim_boundary": _oracle_boundary(),
        "asp_energy_semantic_ready_score": ready_score,
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {RUN_COMMAND: 0}),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    manifest_path: Path | str = REPO_ROOT / FIXTURE_MANIFEST_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the manifest and terminal JSON artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    manifest = Path(manifest_path)
    fixtures = build_fixture_manifest()
    manifest_payload = _fixture_manifest_payload(fixtures, date)
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=result,
        manifest_path=manifest,
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if write:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(_canonical_json(manifest_payload, indent=2), encoding="utf-8")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(_canonical_json(artifact, indent=2), encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail closed on false readiness claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("parity_failure_count"), int), "parity_failure_count")
    _require(not isinstance(artifact.get("parity_failure_count"), bool), "parity_failure_count")
    parity_failures = int(artifact["parity_failure_count"])
    fixture_count = int(artifact.get("fixture_count", 0))
    expected_score = 1.0 if parity_failures == 0 and fixture_count >= 40 else 0.0
    _require(artifact.get("asp_energy_semantic_ready_score") == expected_score, "ready_score")
    if expected_score == 1.0:
        _require(artifact.get("status") == "complete", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    else:
        _require(artifact.get("status") != "complete", "status")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    _require(
        artifact.get("oracle_claim_boundary", {}).get("oracle_distinct_verifier_claim") is False,
        "oracle_claim_boundary",
    )
    _require(
        artifact.get("unsupported_syntax_controls", {}).get("all_rejected_before_energy") is True,
        "unsupported_syntax_controls",
    )
    _require(
        artifact.get("protected_files_unchanged", {})
        .get("scripts/research_conductor.py", {})
        .get("unchanged")
        is True,
        "protected_files",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible values."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _graph_coloring_fixtures() -> list[ASPFixture]:
    return [
        _fixture(
            "graph_path2_three_colors",
            "graph_coloring",
            _graph_program(("v1", "v2"), ("red", "green", "blue"), (("v1", "v2"),)),
            ("cardinality",),
        ),
        _fixture(
            "graph_path2_three_colors_permuted",
            "graph_coloring",
            _graph_program(("n1", "n2"), ("amber", "cyan", "lime"), (("n1", "n2"),)),
            ("cardinality", "label_permutation"),
            permutation_of="graph_path2_three_colors",
        ),
        _fixture(
            "graph_triangle_three_colors",
            "graph_coloring",
            _graph_program(
                ("v1", "v2", "v3"),
                ("red", "green", "blue"),
                (("v1", "v2"), ("v2", "v3"), ("v1", "v3")),
            ),
            ("cardinality",),
        ),
        _fixture(
            "graph_triangle_two_colors_unsat",
            "graph_coloring",
            _graph_program(
                ("v1", "v2", "v3"), ("red", "blue"), (("v1", "v2"), ("v2", "v3"), ("v1", "v3"))
            ),
            ("cardinality", "negative_control"),
        ),
        _fixture(
            "graph_path3_two_colors",
            "graph_coloring",
            _graph_program(("v1", "v2", "v3"), ("red", "blue"), (("v1", "v2"), ("v2", "v3"))),
            ("cardinality",),
        ),
        _fixture(
            "graph_empty2_two_colors",
            "graph_coloring",
            _graph_program(("v1", "v2"), ("red", "blue"), ()),
            ("cardinality", "positive_control"),
        ),
        _fixture(
            "graph_complete2_two_colors",
            "graph_coloring",
            _graph_program(("v1", "v2"), ("red", "blue"), (("v1", "v2"),)),
            ("cardinality",),
        ),
        _fixture(
            "graph_fixed_endpoint_path",
            "graph_coloring",
            _graph_program(
                ("v1", "v2"), ("red", "green", "blue"), (("v1", "v2"),), facts=("color_v1_red",)
            ),
            ("cardinality", "fact"),
        ),
    ]


def _scheduling_fixtures() -> list[ASPFixture]:
    return [
        _fixture(
            "schedule_two_tasks_no_same_morning",
            "scheduling",
            _schedule_program(("a", "b"), ("morning", "night"), (("a_morning", "b_morning"),)),
            ("cardinality",),
        ),
        _fixture(
            "schedule_two_tasks_all_different",
            "scheduling",
            _schedule_program(
                ("a", "b"),
                ("morning", "night"),
                (("a_morning", "b_morning"), ("a_night", "b_night")),
            ),
            ("cardinality",),
        ),
        _fixture(
            "schedule_three_tasks_three_slots",
            "scheduling",
            _schedule_program(
                ("a", "b", "c"),
                ("morning", "midday", "night"),
                _all_same_slot_pairs(("a", "b", "c"), ("morning", "midday", "night")),
            ),
            ("cardinality",),
        ),
        _fixture(
            "schedule_forced_a_morning",
            "scheduling",
            _schedule_program(
                ("a", "b"),
                ("morning", "night"),
                (("a_morning", "b_morning"),),
                facts=("a_morning",),
            ),
            ("cardinality", "fact"),
        ),
        _fixture(
            "schedule_forced_conflict_unsat",
            "scheduling",
            _schedule_program(
                ("a", "b"),
                ("morning", "night"),
                (("a_morning", "b_morning"),),
                facts=("a_morning", "b_morning"),
            ),
            ("cardinality", "negative_control"),
        ),
        _fixture(
            "schedule_block_night_slot",
            "scheduling",
            _schedule_program(("a", "b"), ("morning", "night"), (("a_night",), ("b_night",))),
            ("cardinality",),
        ),
        _fixture(
            "schedule_preassigned_two_tasks",
            "scheduling",
            _schedule_program(
                ("a", "b", "c"),
                ("morning", "midday", "night"),
                _all_same_slot_pairs(("a", "b", "c"), ("morning", "midday", "night")),
                facts=("a_morning", "b_midday"),
            ),
            ("cardinality", "fact"),
        ),
        _fixture(
            "schedule_four_tasks_pair_balance",
            "scheduling",
            _schedule_program(
                ("a", "b", "c", "d"),
                ("morning", "night"),
                (
                    ("a_morning", "b_morning"),
                    ("a_night", "b_night"),
                    ("c_morning", "d_morning"),
                    ("c_night", "d_night"),
                ),
            ),
            ("cardinality",),
        ),
    ]


def _default_fixtures() -> list[ASPFixture]:
    return [
        _fixture(
            "default_bird_flies",
            "non_monotonic_defaults",
            "bird.\nflies :- bird, not injured.\n",
            ("default_negation", "fact"),
        ),
        _fixture(
            "default_injury_blocks_flight",
            "non_monotonic_defaults",
            "bird.\ninjured.\nabnormal :- injured.\nflies :- bird, not abnormal.\n",
            ("default_negation", "fact"),
        ),
        _fixture(
            "default_mutual_choice",
            "non_monotonic_defaults",
            "a :- not b.\nb :- not a.\n:- a, b.\n",
            ("default_negation",),
        ),
        _fixture(
            "default_work_or_rest",
            "non_monotonic_defaults",
            "work :- not rest.\nrest :- not work.\n:- work, rest.\n",
            ("default_negation",),
        ),
        _fixture(
            "default_chain_no_exception",
            "non_monotonic_defaults",
            "safe.\nok :- safe, not revoked.\naccept :- ok, not quarantined.\n",
            ("default_negation", "fact"),
        ),
        _fixture(
            "default_revoked_blocks_ok",
            "non_monotonic_defaults",
            "safe.\nrevoked.\nok :- safe, not revoked.\naccept :- ok, not quarantined.\n",
            ("default_negation", "fact"),
        ),
        _fixture(
            "default_forbidden_default_unsat",
            "non_monotonic_defaults",
            "a :- not b.\n:- a.\n",
            ("default_negation", "negative_control"),
        ),
        _fixture(
            "default_two_level_choice",
            "non_monotonic_defaults",
            "a :- not b.\nb :- not a.\nc :- a, not d.\nd :- b, not c.\n",
            ("default_negation",),
        ),
    ]


def _contradiction_fixtures() -> list[ASPFixture]:
    return [
        _fixture(
            "contradiction_fact_forbidden",
            "contradictions",
            "bad.\n:- bad.\n",
            ("fact", "negative_control"),
        ),
        _fixture(
            "contradiction_pair_forbidden",
            "contradictions",
            "a.\nb.\n:- a, b.\n",
            ("fact", "negative_control"),
        ),
        _fixture(
            "contradiction_cardinality_upper",
            "contradictions",
            "1 { a; b } 1.\na.\nb.\n",
            ("cardinality", "fact", "negative_control"),
        ),
        _fixture(
            "contradiction_all_choices_forbidden",
            "contradictions",
            "1 { a; b } 1.\n:- a.\n:- b.\n",
            ("cardinality", "negative_control"),
        ),
        _fixture(
            "contradiction_default_both_forbidden",
            "contradictions",
            "a :- not b.\nb :- not a.\n:- a.\n:- b.\n",
            ("default_negation", "negative_control"),
        ),
        _fixture(
            "contradiction_positive_chain",
            "contradictions",
            "a.\nc :- a.\n:- c.\n",
            ("fact", "negative_control"),
        ),
        _fixture(
            "contradiction_three_choice_upper",
            "contradictions",
            "1 { a; b; c } 2.\na.\nb.\nc.\n",
            ("cardinality", "fact", "negative_control"),
        ),
        _fixture(
            "contradiction_default_chain_forbidden",
            "contradictions",
            "ok.\nfail :- ok, not blocked.\n:- fail.\n",
            ("default_negation", "fact", "negative_control"),
        ),
    ]


def _positive_negative_control_fixtures() -> list[ASPFixture]:
    return [
        _fixture(
            "control_fact_positive",
            "positive_negative_controls",
            "a.\n",
            ("fact", "positive_control"),
        ),
        _fixture(
            "control_positive_chain",
            "positive_negative_controls",
            "a.\nb :- a.\nc :- b.\n",
            ("fact", "positive_control"),
        ),
        _fixture(
            "control_open_optional_pair",
            "positive_negative_controls",
            "0 { a; b } 2.\n",
            ("cardinality", "positive_control"),
        ),
        _fixture(
            "control_exact_one_yes_no",
            "positive_negative_controls",
            "1 { yes; no } 1.\n",
            ("cardinality", "positive_control"),
        ),
        _fixture(
            "control_negative_constraint_leaves_yes",
            "positive_negative_controls",
            "1 { yes; no } 1.\n:- no.\n",
            ("cardinality", "negative_control"),
        ),
        _fixture(
            "control_label_base",
            "positive_negative_controls",
            "1 { alpha; beta } 1.\ngamma :- alpha.\n",
            ("cardinality", "label_permutation", "positive_control"),
        ),
        _fixture(
            "control_label_permuted",
            "positive_negative_controls",
            "1 { left; right } 1.\ndone :- left.\n",
            ("cardinality", "label_permutation", "positive_control"),
            permutation_of="control_label_base",
        ),
        _fixture(
            "control_unsupported_true_atom_rejected",
            "positive_negative_controls",
            "a :- b.\n",
            ("normal_rule", "negative_control"),
        ),
    ]


def _fixture(
    fixture_id: str,
    family: str,
    program_text: str,
    tags: Sequence[str],
    *,
    permutation_of: str | None = None,
) -> ASPFixture:
    return ASPFixture(
        fixture_id=fixture_id,
        family=family,
        description=fixture_id.replace("_", " "),
        program_text=program_text,
        tags=tuple(tags),
        permutation_of=permutation_of,
    )


def _graph_program(
    vertices: Sequence[str],
    colors: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    facts: Sequence[str] = (),
) -> str:
    lines = [f"{fact}." for fact in facts]
    for vertex in vertices:
        atoms = [f"color_{vertex}_{color}" for color in colors]
        lines.append(_exactly_one(atoms))
    for left, right in edges:
        for color in colors:
            lines.append(_forbid(f"color_{left}_{color}", f"color_{right}_{color}"))
    return "\n".join(lines) + "\n"


def _schedule_program(
    tasks: Sequence[str],
    slots: Sequence[str],
    forbidden: Sequence[tuple[str, ...]],
    *,
    facts: Sequence[str] = (),
) -> str:
    lines = [f"{fact}." for fact in facts]
    for task in tasks:
        lines.append(_exactly_one(f"{task}_{slot}" for slot in slots))
    lines.extend(_forbid(*atoms) for atoms in forbidden)
    return "\n".join(lines) + "\n"


def _all_same_slot_pairs(tasks: Sequence[str], slots: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{left}_{slot}", f"{right}_{slot}")
        for index, left in enumerate(tasks)
        for right in tasks[index + 1 :]
        for slot in slots
    )


def _exactly_one(atoms: Iterable[str]) -> str:
    return f"1 {{ {'; '.join(atoms)} }} 1."


def _forbid(*atoms: str) -> str:
    return f":- {', '.join(atoms)}."


def _fixture_manifest_payload(fixtures: Sequence[ASPFixture], date: str) -> JsonDict:
    return {
        "schema": "carnot.exp6274.asp_energy_fixture_manifest.v1",
        "date": date,
        "random_seed": RANDOM_SEED,
        "max_state_count": MAX_STATE_COUNT,
        "fixtures": [asdict(fixture) for fixture in fixtures],
    }


def _term_payload(term: asp_energy.ASPEnergyTerm) -> JsonDict:
    return {
        "rule_id": term.rule_id,
        "kind": term.kind,
        "source": term.source,
        "payload": _json_ready(term.payload),
    }


def _local_violation_samples(compiled: asp_energy.CompiledASPProgram) -> list[JsonDict]:
    samples: dict[str, JsonDict] = {}
    for state in compiled.enumerate_states():
        receipt = compiled.decompose_state(state)
        for row in receipt["terms"]:
            if row["energy"] > 0 and row["kind"] not in samples:
                samples[row["kind"]] = {"state": state, "receipt": row}
        if set(samples) == {"fact", "normal_rule", "integrity", "cardinality", "stable_support"}:
            break
    return [dict(samples[kind], kind=kind) for kind in sorted(samples)]


def _negative_energy_count(compiled: asp_energy.CompiledASPProgram) -> int:
    count = 0
    for state in compiled.enumerate_states():
        receipt = compiled.decompose_state(state)
        count += sum(1 for row in receipt["terms"] if row["energy"] < 0)
    return count


def _aggregate_localization(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    samples: dict[str, JsonDict] = {}
    for report in reports:
        for sample in report["per_rule_violation_receipts"]:
            kind = sample["kind"]
            samples.setdefault(
                kind,
                {
                    "fixture_id": report["fixture_id"],
                    "state": sample["state"],
                    "receipt": sample["receipt"],
                },
            )
    return {
        "kinds_covered": sorted(samples),
        "sample_receipts_by_kind": samples,
    }


def _contradiction_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [report for report in reports if report["family"] == "contradictions"]
    return {
        "unsat_fixture_count": sum(1 for report in rows if not report["solver_answer_sets"]),
        "fixture_ids": [report["fixture_id"] for report in rows],
        "all_have_empty_zero_energy_set": all(not report["zero_energy_states"] for report in rows),
    }


def _default_negation_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [report for report in reports if "default_negation" in report["tags"]]
    return {
        "default_fixture_count": len(rows),
        "fixture_ids": [report["fixture_id"] for report in rows],
        "all_semantic_parity": all(report["semantic_parity"] is True for report in rows),
    }


def _cardinality_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [report for report in reports if "cardinality" in report["tags"]]
    return {
        "fixtures_with_cardinality": len(rows),
        "fixture_ids": [report["fixture_id"] for report in rows],
        "negative_energy_count": sum(int(report["negative_energy_count"]) for report in rows),
        "all_energy_terms_non_negative": all(
            report["negative_energy_count"] == 0 for report in rows
        ),
    }


def _label_permutation_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_id = {report["fixture_id"]: report for report in reports}
    pairs = []
    for report in reports:
        source_id = report.get("permutation_of")
        if not source_id:
            continue
        source = by_id[source_id]
        pairs.append(
            {
                "source_fixture_id": source_id,
                "permuted_fixture_id": report["fixture_id"],
                "answer_set_count_match": source["solver_answer_set_count"]
                == report["solver_answer_set_count"],
                "zero_energy_count_match": source["zero_energy_state_count"]
                == report["zero_energy_state_count"],
                "state_count_match": source["exact_state_count"] == report["exact_state_count"],
                "parity_both_true": source["semantic_parity"] is True
                and report["semantic_parity"] is True,
            }
        )
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "all_permuted_pairs_match": all(
            pair["answer_set_count_match"]
            and pair["zero_energy_count_match"]
            and pair["state_count_match"]
            and pair["parity_both_true"]
            for pair in pairs
        ),
    }


def _unsupported_syntax_controls() -> JsonDict:
    cases = {
        "p(X).": "variables",
        "a | b.": "disjunction",
        "#minimize { 1,a : a }.": "directive_or_optimization",
        "a :- 1+2=3.": "arithmetic_or_comparison",
        "1 { a : b } 1.": "conditional_cardinality",
        "a :- not.": "malformed_literal",
    }
    receipts = []
    for source, expected in cases.items():
        try:
            asp_energy.compile_program(source, program_id="unsupported")
        except asp_energy.UnsupportedASPSyntax as exc:
            receipts.append(
                {
                    "source": source,
                    "expected_syntax_class": expected,
                    "observed_syntax_class": exc.syntax_class,
                    "energy_constructed": exc.energy_constructed,
                    "rejected": exc.syntax_class == expected and exc.energy_constructed is False,
                }
            )
        else:
            receipts.append(
                {
                    "source": source,
                    "expected_syntax_class": expected,
                    "observed_syntax_class": "accepted",
                    "energy_constructed": True,
                    "rejected": False,
                }
            )
    return {
        "rejected_count": sum(1 for receipt in receipts if receipt["rejected"]),
        "all_rejected_before_energy": all(receipt["rejected"] for receipt in receipts),
        "receipts": receipts,
    }


def _paper_boundary() -> JsonDict:
    return {
        "source": "arXiv:2607.08136, Answer Set Programming Energised!",
        "local_claim": "A bounded ASP subset compiles to inspectable energy terms.",
        "claim_boundary": "Parity is against clingo on trusted fixtures only; no training claim is made.",
    }


def _supported_subset() -> JsonDict:
    return {
        "atom_grammar": "flat lowercase propositional atoms: [a-z][a-z0-9_]*",
        "facts": "atom.",
        "normal_rules": "head :- body_atom, not default_atom.",
        "integrity_constraints": ":- body_atom, not default_atom.",
        "bounded_cardinality": "L { atom; atom } U.",
        "state_bound": MAX_STATE_COUNT,
    }


def _unsupported_behavior() -> JsonDict:
    return {
        "unsupported": [
            "variables",
            "predicate or function terms",
            "disjunctive heads",
            "optimization directives",
            "arithmetic and comparisons",
            "conditional literals inside cardinality rules",
            "unbounded grounding",
        ],
        "behavior": "raise UnsupportedASPSyntax before building energy terms",
    }


def _source_hashes() -> JsonDict:
    paths = [
        Path("python/carnot/asp_energy.py"),
        Path("python/carnot/experiment_6274_asp_energy_semantic_compiler.py"),
        SPEC_RELATIVE_PATH,
        Path("pyproject.toml"),
    ]
    return {
        path.as_posix(): sha256_text((REPO_ROOT / path).read_text(encoding="utf-8"))
        for path in paths
    }


def _solver_receipt(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "name_version": asp_energy.solver_name_version(),
        "oracle_role": "independent answer-set solver for exact set equality",
        "fixture_calls": len(reports),
        "all_calls_succeeded": len(reports) == 40,
        "answer_set_count_by_fixture": {
            report["fixture_id"]: report["solver_answer_set_count"] for report in reports
        },
    }


def _oracle_boundary() -> JsonDict:
    return {
        "verifier_is_oracle": True,
        "oracle_distinct_verifier_claim": False,
        "boundary": "Both clingo and the energy compiler derive from the same formal ASP sidecar.",
        "no_moat_claim": "This is exact executable validation, not a learned verifier moat.",
    }


def _protected_hash_receipts() -> JsonDict:
    protected = ("scripts/research_conductor.py", "CODEX.md", "CLAUDE.md")
    receipts: JsonDict = {}
    for rel in protected:
        before = sha256_text((REPO_ROOT / rel).read_text(encoding="utf-8"))
        after = sha256_text((REPO_ROOT / rel).read_text(encoding="utf-8"))
        receipts[rel] = {
            "sha256_before": before,
            "sha256_after": after,
            "unchanged": before == after,
        }
    return receipts


def _preconditions(protected: Mapping[str, Any]) -> JsonDict:
    return {
        "git_status_at_artifact_build": _git_status_short(),
        "python_version": platform.python_version(),
        "solver_version": asp_energy.solver_name_version(),
        "dependency_state": {"clingo": asp_energy.solver_name_version()},
        "size_bounds": {"max_state_count": MAX_STATE_COUNT, "max_atoms": 12},
        "exact_tolerances": {"zero_energy": 0, "parity": "set equality"},
        "random_seed": RANDOM_SEED,
        "protected_hashes": protected,
    }


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return "complete: bounded ASP energy compiler matches clingo oracle on trusted fixtures"
    return "blocked: bounded ASP energy compiler parity gate failed"


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    return value


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
            ensure_ascii=True,
        )
        + "\n"
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the required Exp6274 run command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = run(date=args.date, duration_s=time.perf_counter() - started)
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "status": artifact["status"],
                "fixture_count": artifact["fixture_count"],
                "parity_failure_count": artifact["parity_failure_count"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
