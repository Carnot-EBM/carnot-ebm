"""REQ-CONSTRAINT-6836 and REQ-CL-6836 typed obligation program tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6836_typed_obligation_program_fixture as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_bytes() -> dict[str, bytes]:
    """SCENARIO-CONSTRAINT-6836-PRECONDITIONS uses immutable source bytes."""

    return {name: (REPO_ROOT / path).read_bytes() for name, path in exp.SOURCE_PATHS.items()}


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    """REQ-CONSTRAINT-6836 builds the complete fixture without writing results."""

    return exp.build_artifact(duration_s=0.25)


def _candidate_text(
    program: exp.TypedObligationProgram,
    *,
    candidate_id: str = "candidate-test-a",
    selected_action_ids: list[str] | None = None,
    atom_values: dict[str, str] | None = None,
) -> str:
    return exp.render_candidate_text(
        {
            "atom_values": atom_values
            if atom_values is not None
            else program.compatible_atom_values(),
            "candidate_id": candidate_id,
            "padding_control": "",
            "scenario_id": program.scenario_id,
            "selected_action_ids": selected_action_ids
            if selected_action_ids is not None
            else program.legal_action_ids,
            "surface_form": "compact_json",
        }
    )


def test_req_6836_specs_precede_implementation() -> None:
    """REQ-CONSTRAINT-6836 and REQ-CL-6836 declare required fields first."""

    constraint = (REPO_ROOT / exp.CONSTRAINT_SPEC_PATH).read_text(encoding="utf-8")
    learning = (REPO_ROOT / exp.CONTINUOUS_LEARNING_SPEC_PATH).read_text(encoding="utf-8")
    section = constraint[constraint.index("## REQ-CONSTRAINT-6836:") :]
    learning_section = learning[learning.index("## REQ-CL-6836:") :]

    for marker in (
        "SCENARIO-CONSTRAINT-6836-PRECONDITIONS",
        "SCENARIO-CONSTRAINT-6836-COMPILE-PARITY",
        "SCENARIO-CONSTRAINT-6836-ATOM-FAILURES",
        "SCENARIO-CONSTRAINT-6836-CANDIDATE-PAIRS",
        "SCENARIO-CONSTRAINT-6836-CONTROLS",
        "SCENARIO-CONSTRAINT-6836-SERIALIZATION",
    ):
        assert marker in section
    for marker in (
        "SCENARIO-CL-6836-MEMORY-GUARD",
        "SCENARIO-CL-6836-FAIL-CLOSED",
        "SCENARIO-CL-6836-CONTROLS",
    ):
        assert marker in learning_section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6836_compile_parity_views_share_atom_identities() -> None:
    """SCENARIO-CONSTRAINT-6836-COMPILE-PARITY shares one atom ledger."""

    scenario = exp.selected_source_scenarios()["constructive"]
    program = exp.TypedObligationProgram.compile(scenario)
    compatible = program.evaluate_candidate(_candidate_text(program))

    assert program.compiled_view_names == exp.COMPILED_VIEW_NAMES
    assert compatible["energy"] == 0
    assert compatible["satisfaction_predicate"] is True
    assert compatible["memory_admission_guard"] is True
    assert compatible["arc_shadow_action_guard"] is True
    assert compatible["view_atom_identities"]["all_views_equal"] is True
    assert compatible["view_atom_identities"]["energy"] == program.atom_ids
    assert {row["atom_id"] for row in compatible["diagnostics"]} == set(program.atom_ids)
    assert all(row["passed"] for row in compatible["diagnostics"])

    atom_values = program.compatible_atom_values()
    atom_values[program.field_atoms[0]["atom_id"]] = exp.ATOM_VALUE_BLOCK
    violating = program.evaluate_candidate(_candidate_text(program, atom_values=atom_values))
    assert violating["energy"] == 1
    assert violating["satisfaction_predicate"] is False
    assert violating["memory_admission_guard"] is False
    assert violating["arc_shadow_action_guard"] is False
    assert violating["view_atom_identities"]["all_views_equal"] is True


def test_scenario_6836_atom_omission_contradiction_impossible_and_fail_close() -> None:
    """SCENARIO-CONSTRAINT-6836-ATOM-FAILURES rejects unsafe candidates."""

    program = exp.TypedObligationProgram.compile(exp.selected_source_scenarios()["constructive"])
    atom_values = program.compatible_atom_values()
    omitted_id = program.field_atoms[0]["atom_id"]
    atom_values.pop(omitted_id)
    omitted = program.evaluate_candidate(_candidate_text(program, atom_values=atom_values))
    assert omitted["energy"] == 1
    assert exp.diagnostic_causes(omitted) == {"atom_omission"}

    atom_values = program.compatible_atom_values()
    atom_values[omitted_id] = exp.ATOM_VALUE_BLOCK
    contradicted = program.evaluate_candidate(_candidate_text(program, atom_values=atom_values))
    assert contradicted["energy"] == 1
    assert exp.diagnostic_causes(contradicted) == {"atom_contradiction"}

    unknown = program.evaluate_candidate(
        _candidate_text(program, selected_action_ids=[*program.legal_action_ids, "unknown-action"])
    )
    assert unknown["energy"] > 0
    assert unknown["memory_admission_guard"] is False
    assert "unknown_action" in exp.diagnostic_causes(unknown)

    parsed_fail = program.evaluate_candidate("not-json")
    assert parsed_fail["parse_error"] == "invalid_json"
    assert parsed_fail["energy"] == len(program.atom_ids)
    assert parsed_fail["memory_admission_guard"] is False

    impossible = exp.TypedObligationProgram.compile(exp.selected_source_scenarios()["impossible"])
    target = impossible.scenario["obligations"][0]["action"]["action_id"]
    failed = impossible.evaluate_candidate(
        _candidate_text(impossible, selected_action_ids=[target])
    )
    assert failed["energy"] > 0
    assert failed["arc_shadow_action_guard"] is False
    assert "field_violation" in exp.diagnostic_causes(failed)


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (b"\xff", "invalid_utf8"),
        ("[]", "invalid_candidate_fields"),
        ({"atom_values": {}, "candidate_id": "c"}, "invalid_candidate_fields"),
        (
            (
                '{"atom_values":{},"candidate_id":"c","padding_control":"",'
                '"scenario_id":"s","selected_action_ids":[],"surface_form":"x" }'
            ),
            "non_canonical_json",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "c",
                "padding_control": "",
                "scenario_id": "s",
                "selected_action_ids": "bad",
                "surface_form": "x",
            },
            "invalid_action_list",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "c",
                "padding_control": "",
                "scenario_id": "s",
                "selected_action_ids": [""],
                "surface_form": "x",
            },
            "invalid_action_id",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "c",
                "padding_control": "",
                "scenario_id": "s",
                "selected_action_ids": ["a", "a"],
                "surface_form": "x",
            },
            "duplicate_action_id",
        ),
        (
            {
                "atom_values": [],
                "candidate_id": "c",
                "padding_control": "",
                "scenario_id": "s",
                "selected_action_ids": [],
                "surface_form": "x",
            },
            "invalid_atom_values",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "",
                "padding_control": "",
                "scenario_id": "s",
                "selected_action_ids": [],
                "surface_form": "x",
            },
            "invalid_candidate_id",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "c",
                "padding_control": [],
                "scenario_id": "s",
                "selected_action_ids": [],
                "surface_form": "x",
            },
            "invalid_padding_control",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "c",
                "padding_control": "",
                "scenario_id": "",
                "selected_action_ids": [],
                "surface_form": "x",
            },
            "invalid_scenario_id",
        ),
        (
            {
                "atom_values": {},
                "candidate_id": "c",
                "padding_control": "",
                "scenario_id": "s",
                "selected_action_ids": [],
                "surface_form": "",
            },
            "invalid_surface_form",
        ),
    ],
)
def test_req_6836_parser_defensive_errors(payload: Any, expected_error: str) -> None:
    """SCENARIO-CONSTRAINT-6836-ATOM-FAILURES parses fixed sequences strictly."""

    raw = payload if isinstance(payload, (str, bytes)) else exp.render_candidate_text(payload)
    assert exp.parse_candidate_text(raw)["error"] == expected_error


def test_req_6836_atom_identity_drift_and_invalid_values_fail_closed() -> None:
    """SCENARIO-CL-6836-FAIL-CLOSED rejects atom identity drift."""

    program = exp.TypedObligationProgram.compile(exp.selected_source_scenarios()["constructive"])
    payload = json.loads(_candidate_text(program))
    payload["scenario_id"] = "wrong-scenario"
    mismatch = program.evaluate_candidate(exp.render_candidate_text(payload))
    assert mismatch["parse_error"] == "scenario_id_mismatch"
    assert mismatch["energy"] == len(program.atom_ids)

    payload = json.loads(_candidate_text(program))
    payload["atom_values"]["extra-atom"] = exp.ATOM_VALUE_ALLOW
    extra = program.evaluate_candidate(exp.render_candidate_text(payload))
    assert "atom_identity_drift" in exp.diagnostic_causes(extra)

    payload = json.loads(_candidate_text(program))
    payload["atom_values"][program.field_atoms[0]["atom_id"]] = "maybe"
    invalid = program.evaluate_candidate(exp.render_candidate_text(payload))
    assert exp.diagnostic_causes(invalid) == {"invalid_atom_value"}


def test_scenario_6836_candidate_pairs_have_controls_no_scores(fixture: dict[str, Any]) -> None:
    """SCENARIO-CONSTRAINT-6836-CANDIDATE-PAIRS freezes matched candidates."""

    assert fixture["typed_obligation_program_ready_score"] == 1
    assert fixture["obligation_pair_fixture_ready_score"] == 1
    assert fixture["verdict_class"] == "null"
    assert fixture["honest_verdict"].startswith("complete_")
    assert exp.validate_artifact(fixture) == []

    rows = fixture["rows"]
    assert len(rows) == 8
    assert {row["label_swap"] for row in rows} == {"canonical", "swapped"}
    assert {row["surface_form"] for row in rows} == {"compact_json", "flat_json"}
    assert {row["case_kind"] for row in rows} >= {
        "atom_contradiction",
        "atom_omission",
        "joint_violation",
        "impossible_set",
    }
    assert [row["row_order"] for row in rows] == list(range(len(rows)))

    labels = fixture["exact_candidate_labels"]
    for row in rows:
        token_lengths = [candidate["token_lengths"] for candidate in row["candidates"]]
        assert token_lengths[0] == token_lengths[1]
        assert row["pair_token_length_equal"] is True
        assert len({candidate["prompt_length"] for candidate in row["candidates"]}) == 1
        for candidate in row["candidates"]:
            raw_text = candidate["raw_text"]
            assert candidate["candidate_id"] in labels
            assert "generated_answer" not in raw_text
            assert "model_score" not in json.dumps(candidate, sort_keys=True)
            assert candidate["expected_tokenization_inputs"]["candidate_text"] == raw_text
            assert candidate["expected_tokenization_inputs"]["prompt_text"] == row["prompt_text"]


def test_scenario_6836_exact_labels_and_checker_mutations(fixture: dict[str, Any]) -> None:
    """REQ-CONSTRAINT-6836 validates every candidate and mutation."""

    labels = fixture["exact_candidate_labels"]
    checked = {
        candidate["candidate_id"]: candidate["exact_check"]["satisfaction_predicate"]
        for row in fixture["rows"]
        for candidate in row["candidates"]
    }
    assert checked
    assert all(
        labels[candidate_id]["compatible"] is passed for candidate_id, passed in checked.items()
    )
    assert all("model_score" not in labels[candidate_id] for candidate_id in labels)

    mutation_results = fixture["checker_mutation_results"]
    assert {row["case"] for row in mutation_results} == {
        "atom_contradiction",
        "atom_omission",
        "impossible_set",
        "invalid_json",
        "joint_violation",
        "unknown_action",
    }
    assert all(row["expected_rejected"] and row["observed_rejected"] for row in mutation_results)
    parity = fixture["compile_parity_results"]
    assert parity["all_views_share_atom_identities"] is True
    assert parity["energy_zero_matches_satisfaction"] is True
    assert parity["satisfaction_matches_guards"] is True
    assert parity["all_candidates_exactly_checked"] is True


def test_scenario_6836_serialization_permutation_invariance_and_hashes(
    fixture: dict[str, Any],
) -> None:
    """SCENARIO-CONSTRAINT-6836-SERIALIZATION keeps canonical bytes stable."""

    rebuilt = exp.build_artifact(duration_s=999.0)
    assert rebuilt["reproducibility_checksum"] == fixture["reproducibility_checksum"]

    scenario = exp.selected_source_scenarios()["constructive"]
    permuted = deepcopy(scenario)
    permuted["obligations"] = list(reversed(permuted["obligations"]))
    permuted["candidates"] = list(reversed(permuted["candidates"]))
    original_program = exp.TypedObligationProgram.compile(scenario)
    permuted_program = exp.TypedObligationProgram.compile(permuted)
    assert permuted_program.atom_ids == original_program.atom_ids
    assert permuted_program.legal_action_ids == original_program.legal_action_ids

    changed = deepcopy(fixture)
    changed["rows"][0]["candidates"][0]["raw_text"] += "x"
    assert exp.reproducibility_checksum(changed) != fixture["reproducibility_checksum"]


def test_scenario_6836_preconditions_and_blocked_artifacts(
    source_bytes: dict[str, bytes],
    tmp_path: Path,
) -> None:
    """SCENARIO-CONSTRAINT-6836-PRECONDITIONS blocks drift and bad fixture text."""

    assert all(row["passed"] for row in exp.evaluate_preconditions(source_bytes))
    assert exp._json_object(b"not-json") == {}
    assert exp._json_object(b"[]") == {}
    assert exp._json_object(exp._read_source(tmp_path / "missing.json"))["read_error"] == (
        "FileNotFoundError"
    )
    with pytest.raises(exp.ProgramFixtureError, match="missing_check"):
        exp.check_by_name([], "missing")

    with pytest.raises(exp.ProgramFixtureError, match="exp6832_scenarios_missing"):
        exp.selected_source_scenarios(tmp_path)

    empty_root = tmp_path / "empty"
    (empty_root / exp.SOURCE_PATHS["exp6832"].parent).mkdir(parents=True)
    (empty_root / exp.SOURCE_PATHS["exp6832"]).write_text('{"scenarios":[]}', encoding="utf-8")
    with pytest.raises(exp.ProgramFixtureError, match="missing_source_scenario"):
        exp.selected_source_scenarios(empty_root)

    scenario = exp.selected_source_scenarios()["safe_noop"]
    with pytest.raises(exp.ProgramFixtureError, match="nonlegal_action_missing"):
        exp._nonlegal_action(scenario, [row["action_id"] for row in scenario["candidates"]])

    drift = dict(source_bytes)
    exp6835 = json.loads(source_bytes["exp6835"])
    exp6835["v598_evidence_root_ready_score"] = 0
    drift["exp6835"] = exp.canonical_bytes(exp6835)
    failed = exp.evaluate_preconditions(drift)
    assert exp.check_by_name(failed, "exp6835_file_sha256")["passed"] is False
    assert exp.check_by_name(failed, "v598_evidence_root_ready_score")["passed"] is False
    source_blocked = exp.build_artifact(duration_s=0.1, source_bytes=drift)
    assert source_blocked["status"] == "complete_blocked_typed_obligation_program_fixture"
    assert exp.validate_artifact(source_blocked) == []
    generated_key = exp.candidate_fixture_generated_answer_absent([{"generated_answer": "x"}])
    assert generated_key["passed"] is False
    assert generated_key["observed"]["violations"][0]["reason"] == "generated_answer"

    rows = exp.build_candidate_rows()
    rows[0]["candidates"][0]["raw_text"] = exp.render_candidate_text(
        {
            "atom_values": {},
            "candidate_id": "candidate-generated-answer",
            "generated_answer": "not allowed",
            "padding_control": "",
            "scenario_id": rows[0]["scenario_id"],
            "selected_action_ids": [],
            "surface_form": "compact_json",
        }
    )
    blocked = exp.build_artifact(duration_s=0.1, candidate_rows=rows)
    assert blocked["status"] == "complete_blocked_typed_obligation_program_fixture"
    assert blocked["rows"] == []
    assert blocked["verdict_class"] == "blocked"
    assert blocked["typed_obligation_program_ready_score"] == 0
    assert blocked["obligation_pair_fixture_ready_score"] == 0
    assert (
        blocked["gate_check_summary"]["failed_check"] == "candidate_fixture_generated_answer_absent"
    )
    assert exp.validate_artifact(blocked) == []


def test_req_6836_validator_reports_terminal_mutations(fixture: dict[str, Any]) -> None:
    """REQ-CONSTRAINT-6836 validates ready and blocked terminal fields."""

    ready = deepcopy(fixture)
    ready["field_principles"].pop("schema")
    ready["inference_substrate"] = "bad"
    ready["verifier_is_oracle"] = True
    ready["duration_s"] = True
    ready["honest_verdict"] = "bad"
    ready["gate_check_summary"] = {"passed": False, "failed_check": "x"}
    ready["typed_obligation_program_ready_score"] = 0
    ready["obligation_pair_fixture_ready_score"] = 0
    ready["rows"][0]["candidates"][0]["raw_text"] += '"generated_answer"'
    ready["compile_parity_results"]["all_candidates_exactly_checked"] = False
    ready["checker_mutation_results"] = []
    ready["reproducibility_checksum"] = exp.reproducibility_checksum(ready)
    errors = exp.validate_artifact(ready)
    assert "field principles do not cover every top-level field" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "duration_s must be a nonnegative number" in errors
    assert "honest_verdict must start with complete_" in errors
    assert "typed program readiness must be 1" in errors
    assert "pair fixture readiness must be 1" in errors
    assert "ready artifact has failed gate" in errors
    assert "ready artifact pair integrity failed" in errors
    assert "ready artifact compile parity failed" in errors
    assert "checker mutation results incomplete" in errors

    checksum = deepcopy(fixture)
    checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility checksum mismatch" in exp.validate_artifact(checksum)

    blocked = exp.build_artifact(duration_s=0.1, source_bytes={"exp6811": b"bad"})
    blocked["rows"] = [{}]
    blocked["typed_obligation_program_ready_score"] = 1
    blocked["obligation_pair_fixture_ready_score"] = 1
    blocked["gate_check_summary"] = {"passed": False, "failed_check": ""}
    blocked["verdict_class"] = "partial"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked artifact emitted rows" in blocked_errors
    assert "blocked artifact has typed readiness" in blocked_errors
    assert "blocked artifact has pair readiness" in blocked_errors
    assert "blocked artifact lacks failed gate" in blocked_errors
    assert "blocked terminal verdict mismatch" in blocked_errors


def test_req_6836_validation_and_cli_paths(
    tmp_path: Path,
    fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6836 writes and validates the terminal artifact."""

    output = tmp_path / "experiment_6836.json"
    artifact = exp.execute(exp.REPO_ROOT, "20260901", output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    assert exp.main(["--date", "20260901", "--output", str(output), "--validate"]) == 0
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 0
    assert '"artifact"' in capsys.readouterr().out
    assert exp.main(["--date", "2026-09-01", "--output", str(output)]) == 2
    with pytest.raises(exp.ProgramFixtureError, match="invalid_run_date"):
        exp.execute(exp.REPO_ROOT, "20261301", output)

    broken = deepcopy(fixture)
    broken["verdict_class"] = "bad"
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    output.write_bytes(exp.canonical_bytes(broken))
    assert exp.main(["--output", str(output), "--validate"]) == 1
    assert "verdict class is outside the closed set" in capsys.readouterr().err

    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: {})
    with pytest.raises(exp.ProgramFixtureError, match="invalid_artifact"):
        exp.execute(exp.REPO_ROOT, "20260901", output)

    output.write_text("not-json", encoding="utf-8")
    assert exp.main(["--output", str(output), "--validate"]) == 2
    assert "Expecting value" in capsys.readouterr().err
