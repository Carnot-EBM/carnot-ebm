"""Tests for the Exp6862 dual-side semantic contrast bank.

Spec refs: REQ-CONSTRAINT-6862 and SCENARIO-CONSTRAINT-6862-*.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6862_dual_side_semantic_contrast_bank as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_bytes() -> dict[str, bytes]:
    """REQ-CONSTRAINT-6862: read immutable inputs once for deterministic tests."""

    return exp.read_source_bytes(REPO)


@pytest.fixture(scope="module")
def source_programs(source_bytes: dict[str, bytes]) -> list[dict]:
    """REQ-CONSTRAINT-6862: reduce raw Exp6849 prompt sources without imports."""

    return exp.source_program_records(json.loads(source_bytes["exp6849"]))


@pytest.fixture(scope="module")
def base_group(source_programs: list[dict]) -> dict:
    """REQ-CONSTRAINT-6862: expose one exact group for adversarial tests."""

    return exp.build_contrast_group(source_programs[0], "exact_energy", 0)


@pytest.fixture(scope="module")
def artifact(source_bytes: dict[str, bytes]) -> dict:
    """REQ-CONSTRAINT-6862: build in memory so tests never rewrite results."""

    return exp.build_artifact(
        REPO,
        run_date="20260902",
        duration_s=0.25,
        source_bytes=source_bytes,
    )


def _changed_source(
    source_bytes: dict[str, bytes], source_id: str, payload: object
) -> dict[str, bytes]:
    changed = dict(source_bytes)
    changed[source_id] = exp.canonical_json(payload)
    return changed


def _candidate(group: dict, expected_label: bool) -> dict:
    return next(row for row in group["candidates"] if row["expected_label"] is expected_label)


def test_req_constraint_6862_spec_declares_contract() -> None:
    """REQ-CONSTRAINT-6862: OpenSpec owns all fields and testable scenarios."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-CONSTRAINT-6862", 1)[1]
    anchors = set(exp.spec_anchors(section))

    assert set(exp.SPEC_REFS) <= anchors
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.RESULT_PATH.as_posix() in section


def test_req_constraint_6862_current_sources_build_ready_bank(artifact: dict) -> None:
    """REQ-CONSTRAINT-6862: at least 96 dual-authority groups pass all gates."""

    assert exp.validate_artifact(artifact) == []
    assert artifact["accepted_contrast_group_count"] == 100
    assert artifact["dual_side_semantic_contrast_bank_ready_score"] == 1
    assert artifact["gate_check_summary"]["passed"] is True
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert set(artifact["field_principles"]) == set(artifact)
    assert len(artifact["structure_side_check_rows"]) == 1_400
    assert len(artifact["solution_side_check_rows"]) == 1_400
    assert len(artifact["rows"]) == 2_800
    assert all(
        {"candidate_id", "authority_side", "transform_id"} <= set(row) for row in artifact["rows"]
    )


def test_req_constraint_6862_covers_each_typed_family(artifact: dict) -> None:
    """REQ-CONSTRAINT-6862: all five exact obligation families have 20 groups."""

    counts = {
        row["family"]: row["accepted_group_count"] for row in artifact["typed_family_manifest"]
    }

    assert counts == {family: 20 for family in exp.TYPED_FAMILIES}
    assert all(row["both_authorities_passed"] for row in artifact["typed_family_manifest"])


@pytest.mark.parametrize(
    ("source_id", "mutate", "failed_check"),
    [
        (
            "exp6861",
            lambda payload: payload.__setitem__("v600_evidence_contract_ready_score", 0),
            "v600_evidence_contract_ready_score",
        ),
        (
            "exp6849",
            lambda payload: payload.__setitem__("sanitized_candidate_pair_manifest", []),
            "exp6849_typed_sources_readable",
        ),
        (
            "exp6852",
            lambda payload: payload.__setitem__("shortcut_attack_results", []),
            "exp6852_failure_witnesses_present",
        ),
    ],
)
def test_scenario_constraint_6862_preconditions_block(
    source_bytes: dict[str, bytes],
    source_id: str,
    mutate: object,
    failed_check: str,
) -> None:
    """SCENARIO-CONSTRAINT-6862-PRECONDITIONS preserves a complete block."""

    payload = json.loads(source_bytes[source_id])
    mutate(payload)  # type: ignore[operator]
    blocked = exp.build_artifact(
        REPO,
        run_date="20260902",
        duration_s=0.1,
        source_bytes=_changed_source(source_bytes, source_id, payload),
    )

    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "complete_blocked_dual_side_semantic_contrast_bank"
    assert blocked["honest_verdict"] == ("complete_blocked_dual_side_semantic_contrast_bank")
    assert blocked["accepted_contrast_group_count"] == 0
    assert blocked["dual_side_semantic_contrast_bank_ready_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == failed_check
    assert blocked["gate_check_summary"]["observed"] is not None


def test_scenario_constraint_6862_unreadable_source_is_named_block(
    source_bytes: dict[str, bytes],
) -> None:
    """SCENARIO-CONSTRAINT-6862-PRECONDITIONS rejects invalid source JSON."""

    blocked = exp.build_artifact(
        REPO,
        run_date="20260902",
        duration_s=0.1,
        source_bytes={**source_bytes, "exp6849": b"{not-json"},
    )

    assert blocked["gate_check_summary"]["failed_check"] == ("exp6849_typed_sources_readable")
    assert blocked["gate_check_summary"]["observed"] == "unreadable"


def test_scenario_constraint_6862_identity_collision_is_preserved() -> None:
    """SCENARIO-CONSTRAINT-6862-IDENTITY-COLLISION rejects reused IDs."""

    rows = [
        {"candidate_id": "same", "semantic_identity": "semantic-a"},
        {"candidate_id": "same", "semantic_identity": "semantic-b"},
    ]

    assert exp.identity_collision_witnesses(
        rows, "candidate_id", "semantic_identity", "candidate"
    ) == [
        {
            "collision_kind": "content_identity_collision",
            "identity": "same",
            "namespace": "candidate",
            "semantic_identities": ["semantic-a", "semantic-b"],
            "rejected": True,
        }
    ]


def test_scenario_constraint_6862_semantic_alias_is_preserved() -> None:
    """SCENARIO-CONSTRAINT-6862-SEMANTIC-ALIAS rejects duplicate meaning."""

    rows = [
        {"group_id": "group-a", "semantic_identity": "same"},
        {"group_id": "group-b", "semantic_identity": "same"},
    ]

    assert exp.semantic_alias_witnesses(rows) == [
        {
            "collision_kind": "semantic_alias",
            "group_ids": ["group-a", "group-b"],
            "semantic_identity": "same",
            "rejected": True,
        }
    ]


def test_scenario_constraint_6862_omitted_atom_is_rejected(base_group: dict) -> None:
    """SCENARIO-CONSTRAINT-6862-OMITTED-ATOM names the missing ledger atom."""

    group = deepcopy(base_group)
    valid = _candidate(group, True)
    missing = valid["content"]["atom_assertions"].pop()
    exp.refresh_candidate_record(valid, group["semantic_identity"])
    result = exp.audit_group(group)

    assert result["accepted"] is False
    assert "omitted_atom" in result["rejection_reasons"]
    assert missing["atom_key"] in result["omitted_atom_keys"]


def test_scenario_constraint_6862_vacuous_program_is_rejected(base_group: dict) -> None:
    """SCENARIO-CONSTRAINT-6862-VACUOUS-CONSTRAINT rejects empty atoms."""

    program = deepcopy(base_group["program"])
    contract = program["obligations"][0]["contract"]
    contract["prerequisite"] = {"all_of": [], "none_of": []}
    contract["authority"] = {"issuer": "", "order": 0}
    contract["fallback"] = {"action_id": "", "reason": ""}
    contract["execution_consequence"] = {"add": [], "remove": []}
    contract["priority"] = {"class": "", "weight": 0}

    reasons = exp.validate_program_structure(program, base_group["mutation_map"])

    assert "vacuous_constraint" in reasons


def test_scenario_constraint_6862_checker_disagreement_is_rejected(
    base_group: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6862-CHECKER-DISAGREEMENT preserves both labels."""

    group = deepcopy(base_group)
    valid = _candidate(group, True)
    valid["content"]["selected_action_ids"] = []
    exp.refresh_candidate_record(valid, group["semantic_identity"])
    result = exp.audit_group(group)

    assert result["accepted"] is False
    assert result["authority_disagreement_witnesses"]
    witness = result["authority_disagreement_witnesses"][0]
    assert witness["structure_label"] is True
    assert witness["solution_label"] is False


def test_scenario_constraint_6862_split_mutation_is_rejected(base_group: dict) -> None:
    """SCENARIO-CONSTRAINT-6862-SPLIT-MUTATION rejects two changed atoms."""

    group = deepcopy(base_group)
    invalid = _candidate(group, False)
    assertions = invalid["content"]["atom_assertions"]
    assertions[1]["value"] = {"split_mutation": True}
    exp.refresh_candidate_record(invalid, group["semantic_identity"])
    result = exp.audit_group(group)

    assert result["accepted"] is False
    assert "split_mutation" in result["rejection_reasons"]


def test_req_constraint_6862_checkers_do_not_call_each_other() -> None:
    """REQ-CONSTRAINT-6862: each authority has an independent call graph."""

    structure_tree = ast.parse(inspect.getsource(exp.structure_side_check))
    solution_tree = ast.parse(inspect.getsource(exp.solution_side_check))
    structure_names = {node.id for node in ast.walk(structure_tree) if isinstance(node, ast.Name)}
    solution_names = {node.id for node in ast.walk(solution_tree) if isinstance(node, ast.Name)}

    assert "solution_side_check" not in structure_names
    assert "structure_side_check" not in solution_names


def test_scenario_constraint_6862_nuisance_transforms_preserve_labels(
    base_group: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6862-NUISANCE-INVARIANCE defeats names and order."""

    transformed = exp.nuisance_variants(base_group)

    assert {row["transform_kind"] for row in transformed} == set(exp.NUISANCE_TRANSFORMS)
    for variant in transformed:
        result = exp.audit_group(variant["group"])
        assert result["accepted"] is True
        assert result["labels"] == {False: False, True: True}


def test_scenario_constraint_6862_one_atom_mutation_flips_both_labels(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6862-SEMANTIC-MUTATION requires exact sensitivity."""

    assert len(artifact["semantic_mutation_rows"]) == 100
    assert all(row["changed_atom_count"] == 1 for row in artifact["semantic_mutation_rows"])
    assert all(row["structure_label_changed"] for row in artifact["semantic_mutation_rows"])
    assert all(row["solution_label_changed"] for row in artifact["semantic_mutation_rows"])


def test_scenario_constraint_6862_templates_are_frozen_without_split_or_tokenizer(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6862-FROZEN-TEMPLATES reserves Exp6863 decisions."""

    manifest = artifact["frozen_sequence_template_manifest"]

    assert manifest["calibration_group_assignment"] is None
    assert manifest["held_group_assignment"] is None
    assert manifest["tokenizer_inspected"] is False
    assert manifest["model_scores_present"] is False
    assert "{program_json}" in manifest["raw_prompt_template"]
    assert "{candidate_json}" in manifest["candidate_sequence_template"]
    assert "model_score" not in exp.canonical_json(artifact["rows"]).decode()


def test_req_constraint_6862_all_identities_are_content_derived(artifact: dict) -> None:
    """REQ-CONSTRAINT-6862: manifests expose unique SHA-256 content identities."""

    group_ids = [row["group_id"] for row in artifact["semantic_contrast_group_manifest"]]
    semantic_ids = [
        row["semantic_identity"] for row in artifact["semantic_contrast_group_manifest"]
    ]
    row_ids = [row["row_id"] for row in artifact["rows"]]
    transform_ids = [row["transform_id"] for row in artifact["nuisance_transform_manifest"]]

    assert len(group_ids) == len(set(group_ids))
    assert len(semantic_ids) == len(set(semantic_ids))
    assert len(row_ids) == len(set(row_ids))
    assert len(transform_ids) == len(set(transform_ids))
    assert all(identity.startswith("sha256:") for identity in group_ids + row_ids)


def test_req_constraint_6862_negative_controls_are_preserved(artifact: dict) -> None:
    """REQ-CONSTRAINT-6862: every required rejection mode has a witness."""

    kinds = {row["rejection_kind"] for row in artifact["rejected_group_manifest"]}

    assert {
        "ambiguous_program",
        "checker_disagreement",
        "identity_collision",
        "omitted_atom",
        "semantic_alias",
        "split_mutation",
        "vacuous_constraint",
    } <= kinds
    assert artifact["authority_disagreement_witnesses"]
    assert artifact["identity_collision_witnesses"]
    assert all(row["rejected"] for row in artifact["rejected_group_manifest"])


def test_req_constraint_6862_validator_reports_tampering(artifact: dict) -> None:
    """REQ-CONSTRAINT-6862: artifact validation fails closed on gate tampering."""

    changed = deepcopy(artifact)
    changed["accepted_contrast_group_count"] = 95
    changed["dual_side_semantic_contrast_bank_ready_score"] = 1

    assert "ready_score_without_96_groups" in exp.validate_artifact(changed)


def test_req_constraint_6862_malformed_source_rows_fail_closed(
    source_bytes: dict[str, bytes], tmp_path: Path
) -> None:
    """REQ-CONSTRAINT-6862: malformed prompt and source shapes cannot enter."""

    malformed_pairs = [
        None,
        {"raw_sequence_inputs": [], "candidates": []},
        {"raw_sequence_inputs": {}, "candidates": []},
        {"raw_sequence_inputs": {"prompt_text": "bad"}, "candidates": []},
        {
            "raw_sequence_inputs": {"prompt_text": "HANDOFF_JSON_BEGIN\n{}\nHANDOFF_JSON_END"},
            "candidates": [{"exact_label": True, "selected_action_ids": []}],
        },
        {
            "raw_sequence_inputs": {"prompt_text": "HANDOFF_JSON_BEGIN\n{}\nHANDOFF_JSON_END"},
            "candidates": [
                {"exact_label": True, "selected_action_ids": "bad"},
                {"exact_label": False},
            ],
        },
    ]

    assert exp.source_program_records({"sanitized_candidate_pair_manifest": {}}) == []
    assert exp.source_program_records({"sanitized_candidate_pair_manifest": malformed_pairs}) == []
    with pytest.raises(ValueError, match="missing prompt section"):
        exp.parse_typed_program("OBSERVED_FACTS_JSON=[]")
    with pytest.raises(ValueError, match="not a mapping"):
        exp.parse_typed_program("HANDOFF_JSON_BEGIN\n[]\nHANDOFF_JSON_END")
    checks = exp.evaluate_preconditions({**source_bytes, "exp6852": b"bad"})
    assert checks[-1]["observed"] == "unreadable"
    assert exp.read_source_bytes(tmp_path) == {key: b"" for key in exp.SOURCE_PATHS}


def test_req_constraint_6862_malformed_program_and_structure_paths(
    base_group: dict,
) -> None:
    """REQ-CONSTRAINT-6862: structural defenses name malformed typed input."""

    assert (
        exp.program_atom_rows({"obligations": [None, {"obligation_id": 1, "contract": []}]}) == []
    )
    with pytest.raises(ValueError, match="unknown typed family"):
        exp.build_contrast_group({}, "unknown", 0)
    assert exp.validate_program_structure({}, {}) == ["ambiguous_program"]

    broken = deepcopy(base_group["program"])
    broken["obligations"].extend([None, {"obligation_id": "broken"}])
    broken["candidate_actions"].append(None)
    broken["output_schema"] = []
    mutation = deepcopy(base_group["mutation_map"])
    mutation["changed_atom_count"] = 2
    mutation["after"] = mutation["before"]
    reasons = exp.validate_program_structure(broken, mutation)
    assert {"ambiguous_program", "split_mutation", "vacuous_constraint"} <= set(reasons)

    candidate = deepcopy(_candidate(base_group, True)["content"])
    first = deepcopy(candidate["atom_assertions"][0])
    candidate["atom_assertions"] = [None, first, first, {"atom_key": "extra", "value": 1}]
    structure = exp.structure_side_check(
        base_group["program"], candidate, base_group["mutation_map"]
    )
    assert structure["authority_passed"] is False
    assert structure["duplicate_atom_keys"] == ["<malformed>", first["atom_key"]]
    assert structure["extra_atom_keys"] == ["extra"]


def test_req_constraint_6862_solution_checker_defensive_paths(
    source_programs: list[dict], base_group: dict
) -> None:
    """REQ-CONSTRAINT-6862: solution authority rejects malformed independent input."""

    candidate = deepcopy(_candidate(base_group, True)["content"])
    malformed_program = deepcopy(base_group["program"])
    malformed_program["candidate_actions"].extend(
        [None, deepcopy(malformed_program["candidate_actions"][0])]
    )
    malformed_program["obligations"].append(None)
    candidate["atom_assertions"].extend([None, {"atom_key": "extra", "value": 1}])
    result = exp.solution_side_check(malformed_program, candidate, "unknown")
    assert result["authority_passed"] is False
    assert {"duplicate_action", "malformed_action", "malformed_atom_assertion"} <= set(
        result["diagnostics"]
    )
    assert "malformed_obligation" in result["diagnostics"]
    assert "extra_atom:extra" in result["diagnostics"]

    nonlist = deepcopy(candidate)
    nonlist["atom_assertions"] = None
    assert (
        "malformed_atom_ledger"
        in exp.solution_side_check(base_group["program"], nonlist, "exact_energy")["diagnostics"]
    )

    missing_fallback = deepcopy(source_programs[1]["program"])
    fallback_id = missing_fallback["obligations"][0]["contract"]["fallback"]["action_id"]
    missing_fallback["candidate_actions"] = [
        row for row in missing_fallback["candidate_actions"] if row["action_id"] != fallback_id
    ]
    assert (
        exp.solution_side_check(missing_fallback, candidate, "exact_energy")["observed_label"]
        is False
    )

    conflict_without_fallback = deepcopy(source_programs[2]["program"])
    fallback_id = conflict_without_fallback["obligations"][1]["contract"]["fallback"]["action_id"]
    conflict_without_fallback["candidate_actions"] = [
        row
        for row in conflict_without_fallback["candidate_actions"]
        if row["action_id"] != fallback_id
    ]
    assert (
        exp.solution_side_check(conflict_without_fallback, candidate, "exact_energy")[
            "observed_label"
        ]
        is False
    )


def test_req_constraint_6862_audit_rejects_stale_and_duplicate_candidates(
    base_group: dict,
) -> None:
    """REQ-CONSTRAINT-6862: candidate aliases and stale content IDs fail closed."""

    group = deepcopy(base_group)
    duplicate = deepcopy(group["candidates"][0])
    duplicate["expected_label"] = True
    group["candidates"] = [group["candidates"][0], duplicate]
    group["candidates"][0]["candidate_id"] = "stale"
    group["candidates"][0]["semantic_identity"] = "stale"
    result = exp.audit_group(group)

    assert result["accepted"] is False
    assert {"ambiguous_program", "identity_collision", "semantic_alias"} <= set(
        result["rejection_reasons"]
    )


def test_req_constraint_6862_validator_and_writer_defenses(artifact: dict, tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6862: terminal validation and writes fail closed."""

    changed = deepcopy(artifact)
    changed.pop("duration_s")
    changed["inference_substrate"] = "wrong"
    changed["verifier_is_oracle"] = True
    changed["verdict_class"] = "wrong"
    changed["honest_verdict"] = "wrong"
    changed["field_principles"] = {}
    changed["gate_check_summary"]["passed"] = False
    changed["semantic_contrast_group_manifest"][1]["group_id"] = changed[
        "semantic_contrast_group_manifest"
    ][0]["group_id"]
    changed["semantic_contrast_group_manifest"][1]["semantic_identity"] = changed[
        "semantic_contrast_group_manifest"
    ][0]["semantic_identity"]
    errors = exp.validate_artifact(changed)

    assert any(row.startswith("missing_required_fields:") for row in errors)
    assert {
        "duplicate_group_id",
        "duplicate_group_semantic_identity",
        "field_principles_do_not_cover_top_level",
        "invalid_verdict_class",
        "nonterminal_honest_verdict",
        "ready_score_with_failed_gate",
        "reproducibility_checksum_mismatch",
        "verifier_oracle_must_be_false",
        "wrong_inference_substrate",
    } <= set(errors)
    with pytest.raises(ValueError, match="invalid Exp6862 artifact"):
        exp.write_artifact(changed, tmp_path)

    output = exp.write_artifact(artifact, tmp_path)
    assert json.loads(output.read_text(encoding="utf-8"))["accepted_contrast_group_count"] == 100


def test_req_constraint_6862_main_writes_blocked_artifact_only_to_requested_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CONSTRAINT-6862: the CLI preserves a terminal block on missing inputs."""

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)

    assert exp.main(["--date", "20260902"]) == 0
    output = json.loads((tmp_path / exp.RESULT_PATH).read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert output["honest_verdict"] == "complete_blocked_dual_side_semantic_contrast_bank"
    assert summary["result_path"] == exp.RESULT_PATH.as_posix()


def test_req_constraint_6862_build_rejection_disqualifies_bank(
    source_bytes: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CONSTRAINT-6862: an accepted-transform failure keeps readiness zero."""

    original = exp.nuisance_variants

    def broken_variants(group: dict) -> list[dict]:
        variants = original(group)
        candidate = variants[0]["group"]["candidates"][0]
        candidate["content"]["selected_action_ids"] = []
        exp.refresh_candidate_record(candidate, group["semantic_identity"])
        return variants

    monkeypatch.setattr(exp, "nuisance_variants", broken_variants)
    artifact = exp.build_artifact(duration_s=0.1, source_bytes=source_bytes)

    assert artifact["dual_side_semantic_contrast_bank_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["rejected_group_manifest"]
