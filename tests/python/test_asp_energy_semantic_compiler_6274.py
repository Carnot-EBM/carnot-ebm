"""Tests for the Exp6274 bounded ASP energy compiler.

Spec refs: REQ-CONSTRAINT-6274,
SCENARIO-CONSTRAINT-6274-SOLVER-PARITY,
SCENARIO-CONSTRAINT-6274-FAIL-CLOSED,
SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import asp_energy
from carnot import experiment_6274_asp_energy_semantic_compiler as exp6274


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"


def test_req_6274_spec_declares_bounded_asp_contract() -> None:
    """REQ-CONSTRAINT-6274: OpenSpec declares the bounded ASP contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6274") :]

    for marker in (
        "SCENARIO-CONSTRAINT-6274-SOLVER-PARITY",
        "SCENARIO-CONSTRAINT-6274-FAIL-CLOSED",
        "SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS",
        "results/experiment_6274_asp_energy_semantic_compiler.json",
        "verifier_is_oracle=true",
        "oracle-distinct verifier moat",
    ):
        assert marker in section


def test_scenario_6274_solver_parity_for_default_negation_and_cardinality() -> None:
    """SCENARIO-CONSTRAINT-6274-SOLVER-PARITY: zero energy equals answer sets."""

    program = """
    1 { assign_a_morning; assign_a_night } 1.
    1 { assign_b_morning; assign_b_night } 1.
    :- assign_a_morning, assign_b_morning.
    ready :- assign_a_night, not blocked.
    """
    compiled = asp_energy.compile_program(program, program_id="unit_schedule")

    solver_sets = asp_energy.solve_with_clingo(compiled.program)
    zero_energy = compiled.zero_energy_states()
    assert zero_energy == solver_sets
    assert {term.kind for term in compiled.energy_terms} == {
        "cardinality",
        "integrity",
        "normal_rule",
        "stable_support",
    }
    assert compiled.exact_state_count == 2 ** len(compiled.program.atoms)

    incomplete_receipt = compiled.decompose_state(["assign_a_night"])
    nonzero = [row for row in incomplete_receipt["terms"] if row["energy"] > 0]
    assert {row["kind"] for row in nonzero} >= {"cardinality", "stable_support"}
    assert all(row["energy"] >= 0 for row in incomplete_receipt["terms"])


def test_scenario_6274_local_receipts_for_facts_rules_constraints_and_support() -> None:
    """SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS: receipts name local failures."""

    program = """
    bird.
    flies :- bird, not injured.
    :- flies, injured.
    """
    compiled = asp_energy.compile_program(program, program_id="unit_default")
    assert compiled.zero_energy_states() == [["bird", "flies"]]

    receipt = compiled.decompose_state(["bird"])
    failing = {row["rule_id"]: row for row in receipt["terms"] if row["energy"] > 0}
    assert "R001" in failing
    assert failing["R001"]["violation"] == "body_true_head_false"
    assert "STABLE_SUPPORT" in failing
    assert "missing_atoms" in failing["STABLE_SUPPORT"]["violation"]

    fact_receipt = compiled.decompose_state(["flies"])
    fact_failures = [row for row in fact_receipt["terms"] if row["kind"] == "fact"]
    assert fact_failures == [
        {
            "rule_id": "F001",
            "kind": "fact",
            "energy": 1,
            "violation": "missing_fact:bird",
        }
    ]

    contradiction = asp_energy.compile_program("bad.\n:- bad.\n", program_id="unit_unsat")
    assert contradiction.zero_energy_states() == []
    assert asp_energy.solve_with_clingo(contradiction.program) == []


def test_scenario_6274_fail_closed_before_energy_construction() -> None:
    """SCENARIO-CONSTRAINT-6274-FAIL-CLOSED: unsupported syntax is rejected early."""

    cases = {
        "": "empty_program",
        ".": "empty_program",
        "a": "missing_period",
        "p(X).": "variables",
        "p(a).": "function_or_predicate_terms",
        "1bad.": "malformed_atom",
        "a | b.": "disjunction",
        "#minimize { 1,a : a }.": "directive_or_optimization",
        "a :- 1+2=3.": "arithmetic_or_comparison",
        "1 { a : b } 1.": "conditional_cardinality",
        "{ a }.": "malformed_cardinality",
        "0 { } 0.": "empty_cardinality",
        "0 { a; a } 1.": "duplicate_cardinality_atom",
        "2 { a } 1.": "invalid_cardinality_bounds",
        "a :- .": "malformed_body",
        "a :- b,,c.": "malformed_literal",
        "a :- not.": "malformed_literal",
    }

    for source, syntax_class in cases.items():
        with pytest.raises(asp_energy.UnsupportedASPSyntax) as excinfo:
            asp_energy.compile_program(source, program_id="bad")
        assert excinfo.value.syntax_class == syntax_class
        assert excinfo.value.energy_constructed is False


def test_req_6274_fixture_manifest_has_required_families_and_exact_bounds() -> None:
    """REQ-CONSTRAINT-6274: fixtures cover all required families with exact bounds."""

    fixtures = exp6274.build_fixture_manifest()
    family_counts = exp6274.fixture_family_counts(fixtures)

    assert len(fixtures) == 40
    assert family_counts == {
        "graph_coloring": 8,
        "scheduling": 8,
        "non_monotonic_defaults": 8,
        "contradictions": 8,
        "positive_negative_controls": 8,
    }
    assert all(fixture.program_text.strip().endswith(".") for fixture in fixtures)

    reports = exp6274.evaluate_fixtures(fixtures)
    assert len(reports) == 40
    assert all(report["semantic_parity"] is True for report in reports)
    assert all(1 <= report["exact_state_count"] <= 4096 for report in reports)
    assert sum(report["parity_failure_count"] for report in reports) == 0


def test_req_6274_artifact_schema_controls_and_provenance(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6274: artifact fields, controls, and provenance are complete."""

    result_path = tmp_path / exp6274.RESULT_RELATIVE_PATH.name
    manifest_path = tmp_path / exp6274.FIXTURE_MANIFEST_RELATIVE_PATH.name
    artifact = exp6274.run(
        date="20260810",
        result_path=result_path,
        manifest_path=manifest_path,
        duration_s=1.25,
        test_exit_codes={
            ".venv/bin/python -m carnot.experiment_6274_asp_energy_semantic_compiler --date 20260810": 0,
            ".venv/bin/pytest tests/python/test_asp_energy_semantic_compiler_6274.py -q": 0,
        },
        write=True,
    )

    assert result_path.exists()
    assert manifest_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp6274.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6274.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    assert artifact["status"] == "complete"
    assert artifact["fixture_count"] == 40
    assert artifact["parity_failure_count"] == 0
    assert isinstance(artifact["parity_failure_count"], int)
    assert artifact["asp_energy_semantic_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["oracle_claim_boundary"]["oracle_distinct_verifier_claim"] is False
    assert (
        artifact["protected_files_unchanged"]["scripts/research_conductor.py"]["unchanged"] is True
    )

    assert artifact["contradiction_controls"]["unsat_fixture_count"] == 8
    assert artifact["default_negation_controls"]["default_fixture_count"] >= 8
    assert artifact["cardinality_controls"]["fixtures_with_cardinality"] >= 16
    assert artifact["label_permutation_controls"]["all_permuted_pairs_match"] is True
    assert artifact["unsupported_syntax_controls"]["all_rejected_before_energy"] is True
    assert artifact["unsupported_syntax_controls"]["rejected_count"] >= 5
    assert artifact["per_rule_violation_localization"]["kinds_covered"] == [
        "cardinality",
        "fact",
        "integrity",
        "normal_rule",
        "stable_support",
    ]
    assert artifact["reproducibility_checksum"] == exp6274.payload_checksum(artifact)


def test_req_6274_validate_artifact_fails_closed_on_missing_or_bad_fields(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6274: schema validation rejects incomplete or false claims."""

    artifact = exp6274.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        manifest_path=tmp_path / "manifest.json",
        duration_s=0.5,
        write=False,
    )

    missing = dict(artifact)
    missing.pop("supported_asp_subset")
    with pytest.raises(ValueError, match="supported_asp_subset"):
        exp6274.validate_artifact(missing)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        exp6274.validate_artifact(bad_oracle)

    bad_score = dict(artifact)
    bad_score["parity_failure_count"] = 1
    bad_score["asp_energy_semantic_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = exp6274.payload_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        exp6274.validate_artifact(bad_score)

    blocked = dict(artifact)
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = "blocked: parity failed"
    blocked["parity_failure_count"] = 1
    blocked["asp_energy_semantic_ready_score"] = 0.0
    blocked["reproducibility_checksum"] = exp6274.payload_checksum(blocked)
    exp6274.validate_artifact(blocked)


def test_req_6274_branch_controls_cover_bounds_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6274: defensive bounds and CLI receipts stay covered."""

    too_large = exp6274.ASPFixture(
        fixture_id="too_large",
        family="positive_negative_controls",
        description="too large",
        program_text="0 { a0; a1; a2; a3; a4; a5; a6; a7; a8; a9; a10; a11; a12 } 13.\n",
        tags=("cardinality",),
    )
    with pytest.raises(ValueError, match="state_bound"):
        exp6274.evaluate_fixture(too_large)

    rich_program = """
    a.
    1 { b; c } 1.
    d :- a, not e.
    :- b, d.
    """
    compiled = asp_energy.compile_program(rich_program, program_id="all_terms")
    samples = exp6274._local_violation_samples(compiled)
    assert {sample["kind"] for sample in samples} == {
        "cardinality",
        "fact",
        "integrity",
        "normal_rule",
        "stable_support",
    }
    assert exp6274._honest_verdict("blocked").startswith("blocked:")
    assert exp6274._json_ready(["x"]) == ["x"]
    assert exp6274._display_path(tmp_path / "outside.json").endswith("outside.json")

    monkeypatch.setattr(exp6274.asp_energy, "compile_program", lambda *args, **kwargs: object())
    unsupported = exp6274._unsupported_syntax_controls()
    assert unsupported["all_rejected_before_energy"] is False
    assert unsupported["receipts"][0]["observed_syntax_class"] == "accepted"

    fake_artifact = {
        "status": "complete",
        "fixture_count": 40,
        "parity_failure_count": 0,
        "honest_verdict": "complete: fake",
    }
    monkeypatch.setattr(exp6274, "run", lambda **kwargs: fake_artifact)
    assert exp6274.main(["--date", "20260810"]) == 0
    assert json.loads(capsys.readouterr().out)["fixture_count"] == 40
