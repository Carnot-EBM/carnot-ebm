"""Tests for the fresh typed-program authority audit.

Spec refs: REQ-CONSTRAINT-6849,
SCENARIO-CONSTRAINT-6849-PRECONDITIONS,
SCENARIO-CONSTRAINT-6849-DUPLICATE-IDS,
SCENARIO-CONSTRAINT-6849-SEMANTIC-ALIAS,
SCENARIO-CONSTRAINT-6849-ATOM-OMISSION,
SCENARIO-CONSTRAINT-6849-IMPOSSIBLE,
SCENARIO-CONSTRAINT-6849-ROW-COLLISION,
SCENARIO-CONSTRAINT-6849-ISOMORPHIC,
SCENARIO-CONSTRAINT-6849-REDUCER-MUTATION, and
SCENARIO-CONSTRAINT-6849-SANITIZED-FIXTURE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6849_typed_program_isomorphic_authority_audit as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_bytes() -> dict[str, bytes]:
    """REQ-CONSTRAINT-6849: read each immutable authority input once."""

    return exp.read_source_bytes(REPO)


@pytest.fixture(scope="module")
def artifact(source_bytes: dict[str, bytes]) -> dict:
    """REQ-CONSTRAINT-6849: build without writing tracked result state."""

    return exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.25,
        source_bytes=source_bytes,
    )


def _json_source(source_bytes: dict[str, bytes], source_id: str) -> dict:
    return json.loads(source_bytes[source_id])


def _changed_source(source_bytes: dict[str, bytes], source_id: str, payload: object) -> dict:
    changed = dict(source_bytes)
    changed[source_id] = exp.canonical_json(payload)
    return changed


def _first_program_and_candidate(artifact: dict) -> tuple[exp.FreshTypedProgram, dict]:
    pair = artifact["sanitized_candidate_pair_manifest"][0]
    source = exp.parse_prompt_program(pair["raw_sequence_inputs"]["prompt_text"])
    program = exp.FreshTypedProgram(source, atom_namespace=pair["atom_namespace"])
    candidate = json.loads(pair["candidates"][0]["raw_sequence_inputs"]["candidate_text"])
    return program, candidate


def test_req_constraint_6849_spec_declares_fields_and_scenarios() -> None:
    """REQ-CONSTRAINT-6849: OpenSpec owns the implementation contract."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-CONSTRAINT-6849", 1)[1]
    anchors = set(exp.spec_anchors(section))

    assert set(exp.SPEC_REFS) <= anchors
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.RESULT_PATH.as_posix() in section


def test_req_constraint_6849_current_artifact_passes_all_authority_gates(
    artifact: dict,
) -> None:
    """REQ-CONSTRAINT-6849: current raw sources build a complete exact audit."""

    assert exp.validate_artifact(artifact) == []
    assert artifact["authority_audit_complete_score"] == 1
    assert artifact["typed_program_authority_ready_score"] == 1
    assert artifact["isomorphic_fixture_ready_score"] == 1
    assert artifact["gate_check_summary"]["passed"] is True
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(artifact)


@pytest.mark.parametrize(
    ("source_id", "mutate", "failed_check"),
    [
        (
            "exp6848",
            lambda payload: payload.__setitem__("v599_evidence_contract_ready_score", 0),
            "v599_evidence_contract_ready_score",
        ),
        (
            "exp6836",
            lambda payload: payload.__setitem__("rows", []),
            "exp6836_raw_fixture_material_readable",
        ),
        (
            "exp6847",
            lambda payload: payload.__setitem__("rows", []),
            "exp6847_discrepancy_rows_present",
        ),
    ],
)
def test_scenario_constraint_6849_preconditions_block_before_reduction(
    source_bytes: dict[str, bytes],
    source_id: str,
    mutate: object,
    failed_check: str,
) -> None:
    """SCENARIO-CONSTRAINT-6849-PRECONDITIONS emits the exact blocked shape."""

    payload = _json_source(source_bytes, source_id)
    mutate(payload)  # type: ignore[operator]
    blocked = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        source_bytes=_changed_source(source_bytes, source_id, payload),
    )

    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "complete_blocked_typed_program_isomorphic_authority_audit"
    assert blocked["honest_verdict"] == (
        "complete_blocked_typed_program_isomorphic_authority_audit"
    )
    assert blocked["rows"] == []
    assert blocked["authority_audit_complete_score"] == 0
    assert blocked["typed_program_authority_ready_score"] == 0
    assert blocked["isomorphic_fixture_ready_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == failed_check
    assert blocked["gate_check_summary"]["observed"] is not None


def test_scenario_constraint_6849_unreadable_json_is_a_named_block(
    source_bytes: dict[str, bytes],
) -> None:
    """SCENARIO-CONSTRAINT-6849-PRECONDITIONS rejects unreadable bytes."""

    blocked = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        source_bytes={**source_bytes, "exp6836": b"{not-json"},
    )

    assert blocked["gate_check_summary"]["failed_check"] == (
        "exp6836_raw_fixture_material_readable"
    )
    assert blocked["gate_check_summary"]["observed"] == "unreadable"


def test_req_constraint_6849_fresh_reducer_does_not_import_exp6836() -> None:
    """REQ-CONSTRAINT-6849: reducer independence is visible in source and manifest."""

    source = (REPO / exp.MODULE_PATH).read_text(encoding="utf-8")

    assert "from carnot import experiment_6836" not in source
    assert "import carnot.experiment_6836" not in source


def test_scenario_constraint_6849_duplicate_ids_emit_collision_witnesses() -> None:
    """SCENARIO-CONSTRAINT-6849-DUPLICATE-IDS rejects repeated identities."""

    records = [
        {"candidate_id": "same", "semantic_identity": "sem-a"},
        {"candidate_id": "same", "semantic_identity": "sem-b"},
    ]
    witnesses = exp.identity_collision_witnesses(
        records,
        namespace="candidate",
        identifier_field="candidate_id",
        semantic_field="semantic_identity",
    )

    assert witnesses == [
        {
            "collision_kind": "duplicate_identifier",
            "identity": "same",
            "namespace": "candidate",
            "occurrences": 2,
            "semantic_identities": ["sem-a", "sem-b"],
        }
    ]


def test_scenario_constraint_6849_semantic_aliases_are_removed(artifact: dict) -> None:
    """SCENARIO-CONSTRAINT-6849-SEMANTIC-ALIAS keeps one canonical definition."""

    result = artifact["duplicate_removal_results"]

    assert result["source_candidate_occurrence_count"] == 16
    assert result["source_unique_semantic_candidate_count"] == 8
    assert result["removed_alias_count"] == 8
    assert result["sanitized_candidate_count"] == 8
    assert result["labels_preserved"] is True
    assert len(result["removed_aliases"]) == 8
    assert artifact["collision_witnesses"]["source"]
    assert artifact["collision_witnesses"]["sanitized"] == []


def test_scenario_constraint_6849_atom_omission_fails_every_boolean_view(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6849-ATOM-OMISSION rejects a missing ledger atom."""

    program, candidate = _first_program_and_candidate(artifact)
    candidate["atom_values"].pop(next(iter(candidate["atom_values"])))
    evaluation = program.evaluate(candidate)

    assert evaluation["energy"] > 0
    assert evaluation["satisfaction_predicate"] is False
    assert evaluation["memory_admission_guard"] is False
    assert evaluation["arc_shadow_action_guard"] is False
    assert "atom_omission" in {row["cause"] for row in evaluation["diagnostics"]}


def test_scenario_constraint_6849_impossible_program_has_no_compatible_candidate(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6849-IMPOSSIBLE rejects a program without a safe action."""

    pair = artifact["sanitized_candidate_pair_manifest"][-1]
    source = exp.parse_prompt_program(pair["raw_sequence_inputs"]["prompt_text"])
    source["candidate_actions"] = [
        row for row in source["candidate_actions"] if row["kind"] != "fail_closed"
    ]
    program = exp.FreshTypedProgram(source, atom_namespace="missing-safe-action")
    candidate = json.loads(pair["candidates"][0]["raw_sequence_inputs"]["candidate_text"])
    candidate["atom_values"] = {atom_id: "allow" for atom_id in program.atom_ids}
    evaluation = program.evaluate(candidate)

    assert program.impossible is True
    assert program.legal_action_ids == ()
    assert evaluation["satisfaction_predicate"] is False
    assert "impossible_program" in {row["cause"] for row in evaluation["diagnostics"]}


def test_scenario_constraint_6849_row_collisions_are_detected(artifact: dict) -> None:
    """SCENARIO-CONSTRAINT-6849-ROW-COLLISION checks IDs and semantic rows."""

    rows = deepcopy(artifact["rows"][:2])
    rows[1]["row_id"] = rows[0]["row_id"]
    rows[1]["semantic_row_identity"] = rows[0]["semantic_row_identity"]
    witnesses = exp.row_collision_witnesses(rows)

    assert {row["collision_kind"] for row in witnesses} == {
        "duplicate_identifier",
        "semantic_alias",
    }


def test_scenario_constraint_6849_isomorphic_transforms_preserve_labels(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6849-ISOMORPHIC checks every required transform."""

    transforms = artifact["isomorphic_transform_manifest"]
    assert {row["transform_kind"] for row in transforms} == {
        "identifier_permutation",
        "atom_rename",
        "label_swap",
        "row_reordering",
        "surface_paraphrase",
        "duplicate_removal",
    }
    assert len({row["transform_id"] for row in transforms}) == len(transforms)
    assert all(row["labels_preserved"] is True for row in transforms)
    assert all(row["exact_negative_control"] is False for row in transforms)
    assert all(row["all_views_recompiled"] is True for row in transforms)


def test_req_constraint_6849_all_views_have_exact_parity(artifact: dict) -> None:
    """REQ-CONSTRAINT-6849: one typed source controls every compiled view row."""

    rows = artifact["compiled_view_parity_rows"]

    assert rows == artifact["rows"]
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert len({row["semantic_row_identity"] for row in rows}) == len(rows)
    assert {row["view_name"] for row in rows} == set(exp.COMPILED_VIEW_NAMES)
    assert all(row["parity_passed"] is True for row in rows)
    assert all(row["atom_identities_match"] is True for row in rows)
    assert all(row["candidate_id"] and row["transform_id"] and row["view_id"] for row in rows)


def test_scenario_constraint_6849_semantic_mutations_flip_exact_labels(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6849-REDUCER-MUTATION pins exact negative controls."""

    rows = artifact["semantic_mutation_rows"]

    assert len(rows) == 4
    assert len({row["mutation_id"] for row in rows}) == len(rows)
    assert all(row["before_label"] is True for row in rows)
    assert all(row["after_label"] is False for row in rows)
    assert all(row["label_changed"] is True for row in rows)
    assert all(row["mutated_atom_id"] in row["after_failed_atom_ids"] for row in rows)
    assert all(row["all_views_recompiled"] is True for row in rows)


def test_scenario_constraint_6849_sanitized_manifest_is_unique_and_score_free(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6849-SANITIZED-FIXTURE freezes safe Exp6851 input."""

    pairs = artifact["sanitized_candidate_pair_manifest"]
    candidates = [candidate for pair in pairs for candidate in pair["candidates"]]
    identities = artifact["candidate_identity_manifest"]

    assert len(pairs) == 4
    assert len({row["pair_id"] for row in pairs}) == len(pairs)
    assert len(candidates) == 8
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)
    assert len({row["semantic_identity"] for row in candidates}) == len(candidates)
    assert len({row["candidate_id"] for row in identities}) == len(identities) == 8
    assert all(row["raw_sequence_inputs"]["candidate_text"] for row in candidates)
    assert all(pair["raw_sequence_inputs"]["prompt_text"] for pair in pairs)
    assert all(pair["token_equality_claimed"] is False for pair in pairs)
    assert all(pair["tokenizer_receipts"] == [] for pair in pairs)
    assert "model_score" not in json.dumps(pairs, sort_keys=True)


def test_req_constraint_6849_checksum_ignores_only_duration(artifact: dict) -> None:
    """REQ-CONSTRAINT-6849: deterministic content has one stable checksum."""

    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)

    assert changed["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    changed["rows"][0]["parity_passed"] = False
    assert exp.reproducibility_checksum(changed) != artifact["reproducibility_checksum"]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda value: value.pop("rows"), "missing required fields"),
        (
            lambda value: value.__setitem__("inference_substrate", "wrong"),
            "inference_substrate mismatch",
        ),
        (
            lambda value: value.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must be false",
        ),
        (
            lambda value: value.__setitem__("verdict_class", "unknown"),
            "verdict_class must use the closed enum",
        ),
        (
            lambda value: value.__setitem__("verdict_class", "positive"),
            "ready artifact must have a null verdict",
        ),
        (
            lambda value: value.__setitem__("honest_verdict", "pending"),
            "honest_verdict must start with complete_",
        ),
        (
            lambda value: value.__setitem__("authority_audit_complete_score", 2),
            "readiness scores must be zero or one",
        ),
        (
            lambda value: value["rows"][1].__setitem__("row_id", value["rows"][0]["row_id"]),
            "ready artifact has identity collisions",
        ),
        (
            lambda value: value["semantic_mutation_rows"][0].__setitem__("label_changed", False),
            "ready artifact has a failed authority gate",
        ),
    ],
)
def test_req_constraint_6849_validator_rejects_corruption(
    artifact: dict, change: object, error: str
) -> None:
    """REQ-CONSTRAINT-6849: malformed authority claims fail validation."""

    changed = deepcopy(artifact)
    change(changed)  # type: ignore[operator]

    assert any(error in item for item in exp.validate_artifact(changed))


def test_req_constraint_6849_parser_and_cli_fail_closed(tmp_path: Path, artifact: dict) -> None:
    """REQ-CONSTRAINT-6849: parser errors and CLI validation have explicit outcomes."""

    with pytest.raises(exp.AuthorityAuditError, match="raw_prompt_program_missing"):
        exp.parse_prompt_program("not a fixture")
    with pytest.raises(exp.AuthorityAuditError, match="candidate_json_invalid"):
        exp.parse_candidate_text("not-json")
    with pytest.raises(exp.AuthorityAuditError, match="candidate_fields_invalid"):
        exp.parse_candidate_text("{}")

    direct = tmp_path / "direct.json"
    exp.write_json_atomic(direct, artifact)
    assert json.loads(direct.read_text(encoding="utf-8")) == artifact

    output = tmp_path / "cli.json"
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--date", "20260901"]) == 0
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--validate"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["typed_program_authority_ready_score"] == 1
    assert payload["duration_s"] > 0

    payload["verifier_is_oracle"] = True
    exp.write_json_atomic(output, payload)
    assert exp.main(["--repo-root", str(REPO), "--output", str(output), "--validate"]) == 1
    assert exp.main(["--date", "2026-09-01", "--output", str(tmp_path / "bad.json")]) == 2


def test_req_constraint_6849_defensive_source_and_parser_paths(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6849: missing files and malformed prompt forms stay explicit."""

    assert exp.sha256_file(tmp_path / "missing.json") is None
    assert exp.read_source_bytes(tmp_path) == {source_id: b"" for source_id in exp.SOURCE_PATHS}
    checks = exp.evaluate_preconditions({source_id: b"" for source_id in exp.SOURCE_PATHS})
    assert {row["observed"] for row in checks} == {"unreadable"}

    with pytest.raises(exp.AuthorityAuditError, match="raw_prompt_program_missing"):
        exp.parse_prompt_program("HANDOFF_JSON_BEGIN\n{not-json}\nHANDOFF_JSON_END")
    with pytest.raises(exp.AuthorityAuditError, match="raw_prompt_program_missing"):
        exp.parse_prompt_program(
            "OBSERVED_FACTS_JSON=[]x\n"
            "OBLIGATIONS_JSON=[]\n"
            "CANDIDATE_ACTIONS_JSON=[]\n"
            "OUTPUT_JSON_SCHEMA={}"
        )
    with pytest.raises(exp.AuthorityAuditError, match="raw_prompt_program_missing"):
        exp.parse_prompt_program("HANDOFF_JSON_BEGIN\n{}\nHANDOFF_JSON_END")


def test_req_constraint_6849_malformed_programs_resolve_fail_closed(artifact: dict) -> None:
    """REQ-CONSTRAINT-6849: malformed ledgers cannot acquire a compatible action set."""

    source = _first_program_and_candidate(artifact)[0].source

    duplicate = deepcopy(source)
    duplicate["candidate_actions"].append(deepcopy(duplicate["candidate_actions"][0]))
    assert exp.FreshTypedProgram(duplicate, atom_namespace="duplicate").impossible is True

    missing_contract = deepcopy(source)
    missing_contract["obligations"][0]["contract"] = None
    program = exp.FreshTypedProgram(missing_contract, atom_namespace="missing-contract")
    assert program.legal_action_ids == (
        next(
            row["action_id"] for row in source["candidate_actions"] if row["kind"] == "fail_closed"
        ),
    )

    missing_prerequisite = deepcopy(source)
    missing_prerequisite["obligations"][0]["contract"]["prerequisite"] = None
    assert (
        exp.FreshTypedProgram(
            missing_prerequisite, atom_namespace="missing-prerequisite"
        ).impossible
        is False
    )

    inactive = next(
        pair
        for pair in artifact["sanitized_candidate_pair_manifest"]
        if pair["case_kind"] == "atom_omission"
    )
    inactive_source = exp.parse_prompt_program(inactive["raw_sequence_inputs"]["prompt_text"])
    fallback_id = inactive_source["obligations"][0]["contract"]["fallback"]["action_id"]
    inactive_source["candidate_actions"] = [
        row for row in inactive_source["candidate_actions"] if row["action_id"] != fallback_id
    ]
    inactive_program = exp.FreshTypedProgram(
        inactive_source, atom_namespace="missing-inactive-fallback"
    )
    assert inactive_program.legal_action_ids[0].endswith("fail-closed")

    interacting = next(
        pair
        for pair in artifact["sanitized_candidate_pair_manifest"]
        if pair["case_kind"] == "joint_violation"
    )
    interacting_source = exp.parse_prompt_program(interacting["raw_sequence_inputs"]["prompt_text"])
    loser = sorted(
        interacting_source["obligations"],
        key=lambda row: row["contract"]["authority"]["order"],
    )[1]
    loser_fallback = loser["contract"]["fallback"]["action_id"]
    interacting_source["candidate_actions"] = [
        row for row in interacting_source["candidate_actions"] if row["action_id"] != loser_fallback
    ]
    interacting_program = exp.FreshTypedProgram(
        interacting_source, atom_namespace="missing-preempted-fallback"
    )
    assert interacting_program.legal_action_ids[0].endswith("fail-closed")


def test_req_constraint_6849_candidate_shape_and_atom_drift_fail_closed(
    artifact: dict,
) -> None:
    """REQ-CONSTRAINT-6849: invalid values and unknown atoms cannot pass a view."""

    program, candidate = _first_program_and_candidate(artifact)

    wrong_scenario = deepcopy(candidate)
    wrong_scenario["scenario_id"] = "wrong"
    assert {row["cause"] for row in program.evaluate(wrong_scenario)["diagnostics"]} == {
        "candidate_shape_invalid"
    }

    invalid_value = deepcopy(candidate)
    invalid_value["atom_values"][next(iter(invalid_value["atom_values"]))] = "unknown"
    assert "invalid_atom_value" in {
        row["cause"] for row in program.evaluate(invalid_value)["diagnostics"]
    }

    extra_atom = deepcopy(candidate)
    extra_atom["atom_values"]["unknown-atom"] = "allow"
    assert {row["cause"] for row in program.evaluate(extra_atom)["diagnostics"]} == {
        "atom_identity_drift"
    }


def test_req_constraint_6849_validator_covers_blocked_corruption(artifact: dict) -> None:
    """REQ-CONSTRAINT-6849: a blocked verdict cannot retain rows or omit its failed gate."""

    blocked = deepcopy(artifact)
    blocked["status"] = "complete_blocked_typed_program_isomorphic_authority_audit"
    blocked["verdict_class"] = "null"
    blocked["authority_audit_complete_score"] = 0
    blocked["typed_program_authority_ready_score"] = 0
    blocked["isomorphic_fixture_ready_score"] = 0
    blocked["gate_check_summary"]["failed_check"] = None
    blocked["duration_s"] = -1
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)

    errors = exp.validate_artifact(blocked)
    assert "duration_s must be a nonnegative number" in errors
    assert "blocked artifact must have no rows or readiness" in errors
    assert "blocked artifact must use blocked verdict class" in errors
    assert "blocked artifact must name the failed check" in errors


def test_req_constraint_6849_cli_read_and_build_errors_are_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6849: CLI failures return nonzero without publishing a result."""

    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1
    assert exp.main(["--date", "20260230", "--output", str(tmp_path / "bad-date.json")]) == 2

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced invalid artifact"])
    assert (
        exp.main(
            [
                "--repo-root",
                str(REPO),
                "--date",
                "20260901",
                "--output",
                str(tmp_path / "forced.json"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "forced.json").exists()
