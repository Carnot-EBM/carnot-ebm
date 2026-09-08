"""Tests for the corrected bounded multiscale Ising proposal.

Spec refs: REQ-SAMPLER-7133, REQ-SAMPLER-7133-PROPOSAL,
REQ-SAMPLER-7133-CORRECTION, REQ-SAMPLER-7133-SEAL,
REQ-SAMPLER-7133-FINITE-LAW, REQ-SAMPLER-7133-ATTACKS,
SCENARIO-SAMPLER-7133-FROZEN-SEAL, SCENARIO-SAMPLER-7133-MH-PARITY,
SCENARIO-SAMPLER-7133-REPLAY, SCENARIO-SAMPLER-7133-FAIL-CLOSED,
SCENARIO-SAMPLER-7133-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7133_v626_multiscale_sampler_prototype as exp


REPO = Path(__file__).resolve().parents[2]


def _rehash(payload: dict) -> dict:
    payload["reproducibility_checksum"] = exp.artifact_checksum(payload)
    return payload


@pytest.fixture(scope="module")
def ready_artifact() -> dict:
    """Build reusable exact evidence without writing tracked state."""

    return exp.build_artifact(root=REPO, run_date="20260908")


def test_req_sampler_7133_spec_precedes_implementation() -> None:
    """REQ-SAMPLER-7133 fixes the proposal, correction, seal, and boundary."""

    text = (REPO / exp.SAMPLER_SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-SAMPLER-7133", 1)[1]
    for anchor in (
        "REQ-SAMPLER-7133-FIXTURES",
        "REQ-SAMPLER-7133-PROPOSAL",
        "REQ-SAMPLER-7133-CORRECTION",
        "REQ-SAMPLER-7133-SEAL",
        "REQ-SAMPLER-7133-FINITE-LAW",
        "REQ-SAMPLER-7133-ATTACKS",
        "REQ-SAMPLER-7133-ARTIFACT",
        "REQ-SAMPLER-7133-READINESS",
        "REQ-SAMPLER-7133-BOUNDARY",
        "SCENARIO-SAMPLER-7133-FROZEN-SEAL",
        "SCENARIO-SAMPLER-7133-MH-PARITY",
        "SCENARIO-SAMPLER-7133-REPLAY",
        "SCENARIO-SAMPLER-7133-FAIL-CLOSED",
        "SCENARIO-SAMPLER-7133-ARTIFACT",
    ):
        assert anchor in section


def test_req_sampler_7133_fixtures_are_frozen_square_and_frustrated() -> None:
    """REQ-SAMPLER-7133-FIXTURES binds each finite target before evaluation."""

    fixtures = exp.frozen_fixtures()
    assert len(fixtures) >= 2
    assert len({item.fixture_hash for item in fixtures}) == len(fixtures)
    assert exp.frozen_fixtures() == fixtures
    for fixture in fixtures:
        receipt = exp.validate_fixture(fixture)
        assert receipt["passed"] is True
        assert fixture.width == fixture.height == 2
        assert receipt["frustrated_plaquette_count"] >= 1
        assert fixture.temperature > 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"width": 3}, "square"),
        ({"fields": (0.0,)}, "field count"),
        ({"fields": (float("nan"), 0.0, 0.0, 0.0)}, "fields must be finite"),
        ({"temperature": 0.0}, "temperature"),
        ({"edges": ((0, 0, 1.0),)}, "self-loop"),
        ({"edges": ((0, 1, 1.0), (1, 0, -1.0))}, "duplicate"),
        ({"edges": ((0, 9, 1.0),)}, "endpoint"),
        ({"edges": ((0, 1, float("nan")),)}, "finite"),
        ({"edges": ((0, 1, 1.0),)}, "perimeter"),
        ({"edges": ((0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0))}, "frustrated"),
    ],
)
def test_req_sampler_7133_invalid_fixtures_fail_closed(mutation: dict, message: str) -> None:
    """REQ-SAMPLER-7133-FIXTURES rejects malformed or unfrustrated cells."""

    fixture = exp.replace_fixture(exp.frozen_fixtures()[0], **mutation)
    with pytest.raises(ValueError, match=message):
        exp.validate_fixture(fixture)


def test_req_sampler_7133_coarse_map_is_bijective_and_proposal_is_positive() -> None:
    """REQ-SAMPLER-7133-PROPOSAL keeps all target states reachable."""

    for fixture in exp.frozen_fixtures():
        states = exp.enumerate_states(fixture.n_spins)
        proposal = exp.proposal_law(fixture)
        assert len(proposal.probabilities) == len(states)
        assert sum(proposal.probabilities) == pytest.approx(1.0, abs=exp.TOLERANCE)
        assert min(proposal.probabilities) > 0.0
        assert sum(proposal.coarse_probabilities) == pytest.approx(1.0, abs=exp.TOLERANCE)
        for state in states:
            coarse, fine = exp.to_coarse_fine(fixture, state)
            assert exp.from_coarse_fine(fixture, coarse, fine) == state
            assert math.isfinite(exp.log_proposal_probability(fixture, states[0], state))
        with pytest.raises(ValueError, match="state"):
            exp.to_coarse_fine(fixture, (1,))
        with pytest.raises(ValueError, match="coarse"):
            exp.from_coarse_fine(fixture, (1,), (1, -1))
    with pytest.raises(ValueError, match="n_spins"):
        exp.enumerate_states(0)


def test_req_sampler_7133_forward_reverse_probabilities_are_explicit_and_asymmetric() -> None:
    """REQ-SAMPLER-7133-PROPOSAL records both directions even for an independent law."""

    fixture = exp.frozen_fixtures()[0]
    states = exp.enumerate_states(fixture.n_spins)
    witnessed = False
    for source in states:
        for target in states:
            forward = exp.log_proposal_probability(fixture, source, target)
            reverse = exp.log_proposal_probability(fixture, target, source)
            assert math.isfinite(forward)
            assert math.isfinite(reverse)
            witnessed = witnessed or not math.isclose(forward, reverse, abs_tol=1.0e-9)
    assert witnessed is True


def test_req_sampler_7133_energy_and_temperature_match_independent_reference() -> None:
    """REQ-SAMPLER-7133-CORRECTION uses the sealed Ising sign and temperature."""

    for fixture in exp.frozen_fixtures():
        target = exp.target_law(fixture)
        sealed = exp.create_sealed_reference(fixture, sequence=0)
        freeze = exp.freeze_receipt(REPO, exp.frozen_fixtures(), sequence=1)
        opened = exp.open_sealed_reference(sealed, fixture, freeze, sequence=2)
        assert target.states == tuple(tuple(row) for row in opened["states"])
        assert target.probabilities == pytest.approx(opened["probabilities"], abs=exp.TOLERANCE)
        for state, energy in zip(target.states, target.energies, strict=True):
            favorable = sum(
                coupling * state[left] * state[right] for left, right, coupling in fixture.edges
            ) + sum(field * state[index] for index, field in enumerate(fixture.fields))
            assert energy == pytest.approx(-favorable, abs=exp.TOLERANCE)


def test_req_sampler_7133_sealed_reference_rejects_order_hash_and_fixture_drift() -> None:
    """REQ-SAMPLER-7133-SEAL fails before parsing stale or mistimed reference bytes."""

    fixture = exp.frozen_fixtures()[0]
    sealed = exp.create_sealed_reference(fixture, sequence=0)
    freeze = exp.freeze_receipt(REPO, exp.frozen_fixtures(), sequence=1)
    with pytest.raises(ValueError, match="after the proposal freeze"):
        exp.open_sealed_reference(sealed, fixture, freeze, sequence=1)
    bad_hash = exp.SealedLaw(sealed.fixture_id, sealed.payload, "sha256:bad", sealed.sequence)
    with pytest.raises(ValueError, match="hash mismatch"):
        exp.open_sealed_reference(bad_hash, fixture, freeze, sequence=2)
    changed_fixture = exp.replace_fixture(fixture, seed=fixture.seed + 1)
    with pytest.raises(ValueError, match="fixture mismatch"):
        exp.open_sealed_reference(sealed, changed_fixture, freeze, sequence=2)


def test_scenario_sampler_7133_mh_matrix_has_balance_stationarity_and_support() -> None:
    """SCENARIO-SAMPLER-7133-MH-PARITY proves the corrected finite kernel."""

    for fixture in exp.frozen_fixtures():
        target = exp.target_law(fixture)
        proposal = exp.proposal_law(fixture)
        transition, acceptances = exp.transition_matrix(fixture, proposal.probabilities)
        checks = exp.kernel_diagnostics(target.probabilities, proposal.probabilities, transition)
        assert checks["proposal_normalization_error"] <= exp.TOLERANCE
        assert checks["transition_normalization_error_max"] <= exp.TOLERANCE
        assert checks["target_support_min"] > 0.0
        assert checks["proposal_support_min"] > 0.0
        assert checks["transition_support_min"] > 0.0
        assert checks["detailed_balance_error_max"] <= exp.TOLERANCE
        assert checks["stationarity_error_max"] <= exp.TOLERANCE
        assert math.isfinite(checks["finite_total_variation"])
        assert 0.0 <= min(acceptances) <= max(acceptances) <= 1.0
        hole = list(proposal.probabilities)
        hole[0] = 0.0
        hole = [value / sum(hole) for value in hole]
        _hole_transition, hole_acceptances = exp.transition_matrix(fixture, hole)
        assert 0.0 in hole_acceptances
        with pytest.raises(ValueError, match="proposal probabilities"):
            exp.transition_matrix(fixture, proposal.probabilities[:-1])


def test_scenario_sampler_7133_replay_uses_state_independent_rng() -> None:
    """SCENARIO-SAMPLER-7133-REPLAY binds a fresh stream to fixture and seed only."""

    fixture = exp.frozen_fixtures()[0]
    first = exp.sample_trace(fixture, steps=128, seed=fixture.seed)
    replay = exp.sample_trace(fixture, steps=128, seed=fixture.seed)
    other = exp.sample_trace(fixture, steps=128, seed=fixture.seed + 1)
    assert first == replay
    assert first["trace_sha256"] != other["trace_sha256"]
    assert exp.rng_stream_seed(fixture.seed, fixture.fixture_id) == first["stream_seed"]
    assert "source_state" not in first["rng_inputs"]
    with pytest.raises(ValueError, match="steps"):
        exp.sample_trace(fixture, steps=0, seed=fixture.seed)


def test_scenario_sampler_7133_required_mutations_are_detected() -> None:
    """SCENARIO-SAMPLER-7133-FAIL-CLOSED exercises every semantic mutation."""

    rows = exp.run_mutations(exp.frozen_fixtures())
    assert {row["mutation_id"] for row in rows} == exp.REQUIRED_MUTATIONS
    assert all(row["detected"] is True and row["passed"] is True for row in rows)
    by_id = {row["mutation_id"]: row for row in rows}
    assert by_id["omitted_reverse_probability"]["observed_value"] > exp.TOLERANCE
    assert by_id["support_hole"]["observed_value"] == 0.0
    assert by_id["wrong_temperature"]["observed_value"] > exp.TOLERANCE
    assert by_id["energy_sign_reversal"]["observed_value"] > exp.TOLERANCE
    assert by_id["reference_leakage"]["observed_value"] == "sealed_probabilities"
    assert by_id["state_dependent_rng"]["observed_value"] == "stream_changed_with_source_state"
    assert by_id["aggregate_only_rows"]["observed_value"] == 0
    assert by_id["hidden_coupling_mismatch"]["observed_value"] > exp.TOLERANCE


def test_scenario_sampler_7133_seal_order_and_freeze_receipts_are_bound(
    ready_artifact: dict,
) -> None:
    """SCENARIO-SAMPLER-7133-FROZEN-SEAL makes post-outcome tuning visible."""

    seal = ready_artifact["seal_receipt"]
    assert ready_artifact["reference_opened_after_freeze"] is True
    assert seal["seal_sequence"] < seal["freeze_sequence"] < seal["open_sequence"]
    assert seal["proposal_inputs"] == sorted(exp.PROPOSAL_INPUTS)
    assert "sealed_probabilities" not in seal["proposal_inputs"]
    assert seal["code_hash"] == ready_artifact["code_hash"]
    assert seal["fixture_hashes"] == ready_artifact["fixture_hashes"]
    assert all(row["hash_match"] is True for row in seal["reference_rows"])


def test_scenario_sampler_7133_artifact_is_complete_recomputable_and_bounded(
    ready_artifact: dict,
) -> None:
    """SCENARIO-SAMPLER-7133-ARTIFACT retains per-instance finite evidence."""

    assert exp.validate_artifact(ready_artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(ready_artifact)
    assert set(ready_artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(ready_artifact["field_principles"].values())
    assert ready_artifact["multiscale_sampler_ready_score"] == 1
    assert ready_artifact["verdict_class"] == "positive"
    assert ready_artifact["honest_verdict"].startswith("complete:")
    assert ready_artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert ready_artifact["inference_substrate_class"] == exp.INFERENCE_SUBSTRATE_CLASS
    assert ready_artifact["execution_venue"] == "host"
    assert ready_artifact["hardware_execution_claimed"] is False
    assert ready_artifact["wcrg_replication_claimed"] is False
    assert len(ready_artifact["rows"]) == len(exp.frozen_fixtures())
    assert all(row["passed"] is True for row in ready_artifact["rows"])
    assert ready_artifact["gate_check_summary"]["passed"] is True
    assert ready_artifact["duration_s"] > 0.0


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda row: row.pop("code_hash"), "missing_required_fields"),
        (lambda row: row["field_principles"].pop("rows"), "field_principles_invalid"),
        (
            lambda row: row.update(inference_substrate_class="no_model_load"),
            "substrate_class_invalid",
        ),
        (lambda row: row.update(execution_venue="kv260"), "execution_venue_invalid"),
        (lambda row: row.update(hardware_execution_claimed=True), "claim_boundary_invalid"),
        (lambda row: row.update(wcrg_replication_claimed=True), "claim_boundary_invalid"),
        (lambda row: row.update(reference_opened_after_freeze=False), "seal_order_invalid"),
        (lambda row: row["seal_receipt"].update(open_sequence=1), "seal_order_invalid"),
        (lambda row: row.update(rows=[]), "aggregate_only_rows"),
        (lambda row: row["instance_rows"].pop(), "instance_rows_incomplete"),
        (lambda row: row["mutation_rows"][0].update(detected=False), "mutation_rows_invalid"),
        (lambda row: row["normalization_rows"][0].update(passed=False), "invariant_rows_failed"),
        (lambda row: row.update(multiscale_sampler_ready_score=0), "readiness_invalid"),
        (lambda row: row.update(verdict_class="null"), "verdict_invalid"),
        (lambda row: row.update(honest_verdict="wrong"), "verdict_invalid"),
        (lambda row: row["gate_check_summary"].update(passed=False), "gate_summary_invalid"),
        (lambda row: row.update(duration_s=-1.0), "duration_invalid"),
    ],
)
def test_req_sampler_7133_artifact_validator_rejects_mutations(
    ready_artifact: dict, mutator: object, error: str
) -> None:
    """REQ-SAMPLER-7133-ARTIFACT rejects incomplete or over-claimed evidence."""

    changed = deepcopy(ready_artifact)
    mutator(changed)  # type: ignore[operator]
    _rehash(changed)
    assert error in exp.validate_artifact(changed)


def test_req_sampler_7133_checksum_and_seal_tampering_fail(
    ready_artifact: dict,
) -> None:
    """REQ-SAMPLER-7133-SEAL binds both the artifact and opened reference bytes."""

    changed = deepcopy(ready_artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(ready_artifact)
    changed["seal_receipt"]["reference_rows"][0]["opened_payload_sha256"] = "sha256:bad"
    _rehash(changed)
    assert "seal_hash_invalid" in exp.validate_artifact(changed)


def test_req_sampler_7133_blocked_precondition_keeps_exact_diagnostic() -> None:
    """REQ-SAMPLER-7133-READINESS emits terminal no-run evidence on a failed gate."""

    checks = exp.collect_preconditions(REPO)
    checks[0] = {
        **checks[0],
        "available": False,
        "observed_value": 0,
        "expected_value": ">=1073741824",
    }
    artifact = exp.build_artifact(root=REPO, run_date="20260908", preconditions=checks)
    assert exp.validate_artifact(artifact) == []
    assert artifact["multiscale_sampler_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"] == {
        "failed_check": "host_ram",
        "expected_value": ">=1073741824",
        "observed_value": 0,
        "passed": False,
    }
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = {}
    _rehash(changed)
    assert "blocked_terminal_state_invalid" in exp.validate_artifact(changed)


def test_req_sampler_7133_atomic_writer_and_cli_use_redirected_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-SAMPLER-7133-ARTIFACT writes only the caller-selected destination."""

    artifact = exp.build_artifact(root=REPO, run_date="20260908")
    output = tmp_path / "artifact.json"
    receipt = exp.write_json_atomic(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert receipt["atomic_replace"] is True
    assert receipt["sha256"] == exp.sha256_file(output)
    assert not list(tmp_path.glob("*.tmp"))
    assert exp.main(["--date", "20260908", "--output", str(output)]) == 0
    assert "multiscale_sampler_ready_score" in capsys.readouterr().out
    assert exp.main(["--validate", str(output)]) == 0
    assert "validated" in capsys.readouterr().out
    output.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", str(output)]) == 2
    assert "missing_required_fields" in capsys.readouterr().out
    output.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", str(output)]) == 2
    assert "artifact_read_error" in capsys.readouterr().out


def test_req_sampler_7133_writer_cleanup_and_run_validation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SAMPLER-7133-ARTIFACT removes partial bytes and refuses invalid evidence."""

    artifact = exp.build_artifact(root=REPO, run_date="20260908")
    output = tmp_path / "failed.json"
    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        exp.write_json_atomic(output, artifact)
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.undo()
    monkeypatch.setattr(exp, "validate_artifact", lambda _payload: ["forced_invalid"])
    with pytest.raises(ValueError, match="forced_invalid"):
        exp.run_experiment(root=REPO, output=output, run_date="20260908")


def test_req_sampler_7133_adversarial_verifier_accepts_declared_cpu_class(
    ready_artifact: dict, tmp_path: Path
) -> None:
    """REQ-SAMPLER-7133-READINESS keeps the required CPU class verifier-clean."""

    from scripts import adversarial_verify

    output = tmp_path / exp.RESULT_PATH.name
    exp.write_json_atomic(output, ready_artifact)
    report = adversarial_verify.verify_artifact(output)
    assert report["flags"] == []
