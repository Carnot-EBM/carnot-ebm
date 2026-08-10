"""Tests for Exp6287 ASP continuous relaxation.

Spec refs: REQ-KONA-6287, SCENARIO-KONA-6287-VERTEX-PARITY,
SCENARIO-KONA-6287-GRADIENT-CHECK, SCENARIO-KONA-6287-CONTROLS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import asp_continuous_relaxation as relax
from carnot import asp_energy
from carnot import experiment_6287_asp_continuous_relaxation as exp6287


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/phase3-kona/spec.md"


def _exact_one_table() -> relax.VertexEnergyTable:
    compiled = asp_energy.compile_program("1 { yes; no } 1.\n", program_id="unit_exact_one")
    return relax.build_energy_table(compiled, fixture_id="unit_exact_one")


def test_req_kona_6287_spec_declares_relaxation_contract() -> None:
    """REQ-KONA-6287: OpenSpec anchors the bounded relaxation bridge."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-KONA-6287") :]

    for marker in (
        "SCENARIO-KONA-6287-VERTEX-PARITY",
        "SCENARIO-KONA-6287-GRADIENT-CHECK",
        "SCENARIO-KONA-6287-CONTROLS",
        "multilinear extension",
        "results/experiment_6287_asp_continuous_relaxation.json",
        "verifier_is_oracle",
        "learned Kona model",
        "diffusion language model",
    ):
        assert marker in section


def test_scenario_kona_6287_vertex_parity_and_probability_validation() -> None:
    """SCENARIO-KONA-6287-VERTEX-PARITY: vertices equal discrete energy."""

    compiled = asp_energy.compile_program(
        """
        bird.
        flies :- bird, not injured.
        :- flies, injured.
        """,
        program_id="unit_default",
    )
    table = relax.build_energy_table(compiled, fixture_id="unit_default")
    parity = relax.verify_vertex_parity(compiled, table)

    assert parity["parity_passed"] is True
    assert parity["checked_vertices"] == table.vertex_count
    assert parity["max_abs_delta"] == 0.0
    assert relax.energy_at(relax.vertex_probability_vector(table, ["bird", "flies"]), table) == 0.0
    assert relax.round_probabilities(table, [1.0, 1.0, 0.49]) == ["bird", "flies"]

    with pytest.raises(ValueError, match="probability_length"):
        relax.energy_at([0.5], table)
    with pytest.raises(ValueError, match="probability_bounds"):
        relax.gradient_at([0.2, 1.2, 0.3], table)


def test_scenario_kona_6287_analytic_gradient_matches_finite_difference() -> None:
    """SCENARIO-KONA-6287-GRADIENT-CHECK: analytic gradient is checked."""

    table = _exact_one_table()

    assert relax.energy_at([0.5, 0.5], table) == pytest.approx(0.5)
    assert relax.gradient_at([0.25, 0.75], table) == pytest.approx([0.5, -0.5])

    check = relax.check_gradient(table, [0.25, 0.75], epsilon=1e-5, tolerance=1e-8)
    assert check["passed"] is True
    assert check["max_abs_error"] < 1e-8
    assert check["analytic"] == pytest.approx(check["finite_difference"])

    stationary = relax.stationary_point_record(
        table,
        [0.5, 0.5],
        gradient_tolerance=1e-12,
        box_tolerance=1e-9,
    )
    assert stationary["stationary"] is True
    assert stationary["fractional"] is True
    assert stationary["energy"] == pytest.approx(0.5)


def test_scenario_kona_6287_refinement_reports_rounding_failure() -> None:
    """SCENARIO-KONA-6287-CONTROLS: refinement and rounding are separate."""

    table = _exact_one_table()
    outcome = relax.refine(table, [0.0, 0.0], steps=4, step_size=0.25)
    rounded = relax.round_probabilities(table, outcome["final_probabilities"])

    assert outcome["steps"] == 4
    assert outcome["energy_evaluations"] >= 5
    assert 0.0 < outcome["final_energy"] < 1.0
    assert rounded == []
    assert table.discrete_energy(rounded) == 1
    assert table.best_discrete_energy == 0


def test_req_kona_6287_rejects_unsupported_size_and_bad_table_inputs() -> None:
    """REQ-KONA-6287: bounded fixtures and malformed inputs fail closed."""

    too_large = asp_energy.compile_program(
        "0 { a0; a1; a2; a3; a4; a5; a6; a7; a8; a9; a10; a11; a12 } 13.\n",
        program_id="too_large",
    )
    with pytest.raises(relax.UnsupportedRelaxationFixture, match="vertex_bound"):
        relax.build_energy_table(too_large, fixture_id="too_large", max_atoms=12, max_vertices=4096)

    table = _exact_one_table()
    with pytest.raises(ValueError, match="unknown_atom"):
        table.mask_for_state(["missing"])
    with pytest.raises(ValueError, match="unknown_atom"):
        relax.vertex_probability_vector(table, ["missing"])
    with pytest.raises(ValueError, match="finite_difference_epsilon"):
        relax.finite_difference_gradient(table, [0.5, 0.5], epsilon=0.0)
    with pytest.raises(ValueError, match="finite_difference_boundary"):
        relax.check_gradient(table, [1e-8, 0.5], epsilon=1e-5, tolerance=1e-8)
    with pytest.raises(ValueError, match="steps"):
        relax.refine(table, [0.5, 0.5], steps=-1, step_size=0.1)
    with pytest.raises(ValueError, match="step_size"):
        relax.refine(table, [0.5, 0.5], steps=1, step_size=0.0)

    compiled = asp_energy.compile_program("1 { yes; no } 1.\n", program_id="bad_parity")
    bad_table = relax.VertexEnergyTable(
        fixture_id="bad_parity",
        atoms=table.atoms,
        energies=tuple(energy + 1 for energy in table.energies),
        vertex_states=table.vertex_states,
    )
    parity = relax.verify_vertex_parity(compiled, bad_table)
    assert parity["parity_passed"] is False
    assert parity["failure_count"] == bad_table.vertex_count


def test_req_kona_6287_artifact_schema_controls_and_provenance(tmp_path: Path) -> None:
    """REQ-KONA-6287: terminal artifact carries controls and claim boundary."""

    result_path = tmp_path / exp6287.RESULT_RELATIVE_PATH.name
    artifact = exp6287.run(
        date="20260810",
        result_path=result_path,
        duration_s=1.5,
        test_exit_codes={
            exp6287.RUN_COMMAND: 0,
            ".venv/bin/pytest tests/python/test_asp_continuous_relaxation_6287.py -q --no-cov": 0,
        },
        write=True,
    )

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp6287.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6287.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    assert artifact["status"] == "complete"
    assert artifact["fixture_count"] == 40
    assert artifact["parity_failure_count"] == 0
    assert type(artifact["parity_failure_count"]) is int
    assert artifact["asp_continuous_relaxation_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "learned" in artifact["relaxation_definition_and_claim_boundary"]["not_claimed"]
    assert artifact["finite_difference_gradient_checks"]["all_passed"] is True
    assert artifact["exact_completion_controls"]["clingo"]["all_calls_succeeded"] is True
    assert artifact["cold_start_controls"]["all_exact_enumerations_completed"] is True
    assert artifact["unsupported_size_and_syntax_controls"]["unsupported_size_rejected"] is True
    assert artifact["unsupported_size_and_syntax_controls"]["malformed_bounds_rejected"] is True
    assert artifact["unsupported_size_and_syntax_controls"]["sign_reversal_detected"] is True
    assert artifact["unsupported_size_and_syntax_controls"]["label_permutation_control"]["passed"] is True
    assert artifact["rounding_failures_by_fixture"]["failure_count"] >= 1
    assert artifact["fractional_stationary_points_by_fixture"]["stationary_point_count"] >= 1
    assert artifact["reproducibility_checksum"] == exp6287.payload_checksum(artifact)


def test_req_kona_6287_validate_artifact_fails_closed(tmp_path: Path) -> None:
    """REQ-KONA-6287: validator rejects false readiness and missing fields."""

    artifact = exp6287.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        duration_s=0.2,
        write=False,
    )

    missing = dict(artifact)
    missing.pop("source_paths_and_hashes")
    with pytest.raises(ValueError, match="source_paths_and_hashes"):
        exp6287.validate_artifact(missing)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = exp6287.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        exp6287.validate_artifact(bad_oracle)

    bad_score = dict(artifact)
    bad_score["parity_failure_count"] = 1
    bad_score["asp_continuous_relaxation_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = exp6287.payload_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        exp6287.validate_artifact(bad_score)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "not_terminal"
    bad_prefix["reproducibility_checksum"] = exp6287.payload_checksum(bad_prefix)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp6287.validate_artifact(bad_prefix)

    blocked = dict(artifact)
    blocked["status"] = "blocked"
    blocked["finite_difference_gradient_checks"] = dict(blocked["finite_difference_gradient_checks"])
    blocked["finite_difference_gradient_checks"]["all_passed"] = False
    blocked["asp_continuous_relaxation_ready_score"] = 0.0
    blocked["honest_verdict"] = exp6287._honest_verdict("blocked")
    blocked["reproducibility_checksum"] = exp6287.payload_checksum(blocked)
    exp6287.validate_artifact(blocked)


def test_req_kona_6287_defensive_control_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KONA-6287: adversarial controls have fail-closed fallbacks."""

    empty_table = relax.VertexEnergyTable(
        fixture_id="empty",
        atoms=(),
        energies=(0,),
        vertex_states=((),),
    )
    assert exp6287._fractional_stationary_points(empty_table) == []

    monkeypatch.setattr(exp6287.relax, "build_energy_table", lambda *args, **kwargs: object())
    assert exp6287._unsupported_size_control()["rejected"] is False

    monkeypatch.setattr(exp6287.asp_energy, "compile_program", lambda *args, **kwargs: object())
    assert exp6287._malformed_bounds_control()["rejected"] is False

    no_reversal_reports = [
        {
            "vertex_parity": {"checked_vertices": 1},
            "best_discrete_energy": 1,
            "refinement_outcomes": {},
        },
        {
            "vertex_parity": {"checked_vertices": 1},
            "best_discrete_energy": 0,
            "refinement_outcomes": {"s": {"attempts": [{"rounded_energy": 0}]}},
        },
    ]
    assert exp6287._sign_reversal_detected(no_reversal_reports) is False

    mismatch = exp6287._label_permutation_control(
        [
            {
                "fixture_id": "base",
                "atom_count": 1,
                "vertex_count": 2,
                "energy_spectrum_hash": "a",
            },
            {
                "fixture_id": "perm",
                "permutation_of": "base",
                "atom_count": 2,
                "vertex_count": 4,
                "energy_spectrum_hash": "b",
            },
        ]
    )
    assert mismatch["passed"] is False


def test_req_kona_6287_cli_writes_requested_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-KONA-6287: CLI writes the terminal JSON artifact."""

    result_path = tmp_path / "experiment_6287.json"

    assert exp6287.main(["--date", "20260810", "--result-path", str(result_path)]) == 0
    emitted = json.loads(capsys.readouterr().out)

    assert emitted["result"] == str(result_path)
    assert emitted["status"] == "complete"
    assert emitted["parity_failure_count"] == 0
    assert result_path.exists()
