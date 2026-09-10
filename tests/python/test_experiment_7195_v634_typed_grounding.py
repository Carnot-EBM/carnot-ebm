"""Focused tests for REQ-VERIFY-7195 and SCENARIO-VERIFY-7195-*.

All writer tests use private temporary paths. They never change the checked-in
research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7195_v634_typed_grounding as exp
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    TypedRelation,
    execute_relation,
)
from carnot.verify import experiment_7195_source_relation_executor as executor


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"


def _binding(source: bytes, entity_id: str, surface: str, occurrence: int = 0) -> EntityBinding:
    """Make one exact byte binding for a named occurrence in a test source."""

    needle = surface.encode("utf-8")
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(needle, start + 1)
    return EntityBinding(entity_id, surface, start, start + len(needle))


def _relation(
    source: bytes,
    subject_id: str,
    operator: str,
    object_id: str,
    polarity: str = "positive",
) -> TypedRelation:
    """Bind one relation to the full source span for concise semantic tests."""

    return TypedRelation(subject_id, operator, object_id, polarity, 0, len(source))


def test_req_verify_7195_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7195 defines every focused semantic and artifact scenario."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7195") :]
    for scenario in (
        "EXECUTION",
        "UNKNOWN",
        "PANEL",
        "BLINDING",
        "CONTRACTS",
        "DIAGNOSIS",
        "PREFLIGHT",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7195-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7195_execution_handles_direction_and_negation() -> None:
    """SCENARIO-VERIFY-7195-EXECUTION uses typed direction and polarity."""

    source = b"entity-a precedes entity-b."
    bindings = (
        _binding(source, "a", "entity-a"),
        _binding(source, "b", "entity-b"),
    )
    evidence = (_relation(source, "a", "precedes", "b"),)

    same = execute_relation(source, bindings, evidence, _relation(source, "a", "precedes", "b"))
    inverse = execute_relation(source, bindings, evidence, _relation(source, "b", "follows", "a"))
    reversed_args = execute_relation(
        source, bindings, evidence, _relation(source, "b", "precedes", "a")
    )
    negated = execute_relation(
        source, bindings, evidence, _relation(source, "a", "precedes", "b", "negative")
    )

    assert same.decision == inverse.decision == "supported"
    assert reversed_args.decision == negated.decision == "contradicted"
    assert all(not result.abstention for result in (same, inverse, reversed_args, negated))
    assert same.uncertainty_reasons == ()


def test_scenario_verify_7195_unknown_preserves_each_uncertainty() -> None:
    """SCENARIO-VERIFY-7195-UNKNOWN never coerces incomplete semantics to false."""

    source = b"Alex precedes Alex."
    ambiguous_bindings = (
        _binding(source, "left", "Alex", 0),
        _binding(source, "right", "Alex", 1),
    )
    claim = _relation(source, "left", "precedes", "right")
    ambiguous = execute_relation(
        source,
        ambiguous_bindings,
        (_relation(source, "left", "precedes", "right"),),
        claim,
    )
    missing = execute_relation(source, ambiguous_bindings[:1], (), claim)
    bad_offset = execute_relation(
        source,
        ambiguous_bindings,
        (TypedRelation("left", "precedes", "right", "positive", 0, len(source) + 1),),
        claim,
    )
    unsupported = execute_relation(
        source,
        ambiguous_bindings,
        (_relation(source, "left", "touches", "right"),),
        _relation(source, "left", "touches", "right"),
    )

    assert {result.decision for result in (ambiguous, missing, bad_offset, unsupported)} == {
        "unknown"
    }
    assert all(result.abstention for result in (ambiguous, missing, bad_offset, unsupported))
    assert "ambiguous_entity_mapping" in ambiguous.uncertainty_reasons
    assert "missing_source_relation" in missing.uncertainty_reasons
    assert "invalid_source_offsets" in bad_offset.uncertainty_reasons
    assert "unsupported_relation_operator" in unsupported.uncertainty_reasons


def test_scenario_verify_7195_unknown_detects_contradictory_evidence() -> None:
    """SCENARIO-VERIFY-7195-UNKNOWN exposes both polarities as a contradiction."""

    source = b"entity-a precedes entity-b and does not precede entity-b."
    bindings = (
        _binding(source, "a", "entity-a"),
        _binding(source, "b", "entity-b", 0),
    )
    evidence = (
        _relation(source, "a", "precedes", "b"),
        _relation(source, "a", "precedes", "b", "negative"),
    )

    result = execute_relation(source, bindings, evidence, evidence[0])

    assert result.decision == "unknown"
    assert result.abstention is True
    assert "contradictory_evidence" in result.uncertainty_reasons
    assert result.matched_relation_indexes == (0, 1)


def test_scenario_verify_7195_unknown_rejects_every_typed_input_defect() -> None:
    """SCENARIO-VERIFY-7195-UNKNOWN retains concrete byte-contract failures."""

    source = b"entity-a precedes entity-b."
    valid = (
        _binding(source, "a", "entity-a"),
        _binding(source, "b", "entity-b"),
    )
    claim = _relation(source, "a", "precedes", "b")
    cases = (
        (
            (EntityBinding("a", "entity-a", 0, len(source) + 1), valid[1]),
            (claim,),
            claim,
            "invalid_source_offsets",
        ),
        (
            (EntityBinding("a", "not-a", 0, 5), valid[1]),
            (claim,),
            claim,
            "entity_surface_mismatch",
        ),
        (
            valid,
            (TypedRelation("a", "precedes", "b", "maybe", 0, len(source)),),
            claim,
            "unsupported_polarity",
        ),
        (
            valid,
            (TypedRelation("a", "precedes", "b", "negative", 0, len(source)),),
            claim,
            "polarity_span_mismatch",
        ),
        (
            valid,
            (TypedRelation("a", "precedes", "b", "positive", 9, 17),),
            claim,
            "entity_outside_relation_span",
        ),
    )
    for bindings, relations, tested_claim, reason in cases:
        result = execute_relation(source, bindings, relations, tested_claim)
        assert result.decision == "unknown"
        assert reason in result.uncertainty_reasons

    unresolved = execute_relation(
        source,
        valid,
        (claim,),
        _relation(source, "a", "starts before", "b"),
    )
    assert unresolved.decision == "unknown"
    assert unresolved.uncertainty_reasons == ("unresolved_evidence",)
    assert executor._assertions(_relation(source, "a", "touches", "b")) == ()


def test_scenario_verify_7195_panel_is_fresh_sealed_and_grouped() -> None:
    """SCENARIO-VERIFY-7195-PANEL makes 48 new paired base groups."""

    upstream = json.loads(
        (REPO / "results/experiment_7158_v630_entity_evidence_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    panel = exp.build_fresh_panel(upstream)
    old_public = {
        row["unit_id"]
        for row in exp.read_jsonl(
            REPO / "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
        )
    }
    old_authority = exp.read_jsonl(
        REPO / "results/experiment_7180_v633_symbolic_edit_fixture_authority.jsonl"
    )
    old_entities = {
        str(tuple_value[field])
        for row in old_authority
        for tuple_value in (
            row["expected_response"]["claim_tuple"],
            row["expected_response"]["evidence_tuple"],
        )
        if isinstance(tuple_value, dict)
        for field in ("subject", "object")
    }
    new_entities = {
        str(row["claim_tuple"][field])
        for row in panel["internal_rows"]
        for field in ("subject", "object")
    }

    assert len(panel["internal_rows"]) == 192
    assert len({row["base_id"] for row in panel["internal_rows"]}) == 48
    assert panel["split_manifest"]["calibration_base_count"] == 16
    assert panel["split_manifest"]["evaluation_base_count"] == 32
    assert set(row["unit_id"] for row in panel["internal_rows"]).isdisjoint(old_public)
    assert new_entities.isdisjoint(old_entities)
    assert any(value.startswith("entity-7195-") for value in new_entities)


def test_scenario_verify_7195_blinding_and_contracts_exclude_private_fields() -> None:
    """SCENARIO-VERIFY-7195-BLINDING keeps all producer inputs public."""

    upstream = json.loads(
        (REPO / "results/experiment_7158_v630_entity_evidence_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    panel = exp.build_fresh_panel(upstream)
    public_rows = exp.build_public_rows(panel["internal_rows"])
    authority_rows = exp.build_authority_rows(panel)
    contract = exp.freeze_generation_contract(public_rows, panel["split_manifest"])

    assert exp.public_view_errors(public_rows, authority_rows) == []
    assert all(set(row) == {"unit_id", "source_text", "claim_text"} for row in public_rows)
    assert set(contract["atomic_prompts"]) == {"source", "claim", "direct"}
    assert contract["atomic_prompts"]["source"]["visible_fields"] == ["source_text"]
    assert contract["atomic_prompts"]["claim"]["visible_fields"] == ["claim_text"]
    assert contract["atomic_prompts"]["direct"]["visible_fields"] == [
        "source_text",
        "claim_text",
    ]
    assert contract["grammar_only_control"]["semantic_constraints"] == []
    changed_authority = deepcopy(authority_rows)
    changed_authority[0]["expected_executor_decision"] = "unknown"
    assert exp.jsonl_bytes(public_rows) == exp.jsonl_bytes(
        exp.build_public_rows(panel["internal_rows"])
    )
    assert changed_authority != authority_rows

    malformed = deepcopy(public_rows)
    malformed[0]["split"] = "evaluation"
    assert "public_view_shape" in exp.public_view_errors(malformed, authority_rows)
    assert "public_authority_leak" in exp.public_view_errors(malformed, authority_rows)
    duplicated = deepcopy(public_rows)
    duplicated[1]["unit_id"] = duplicated[0]["unit_id"]
    assert "public_authority_roster" in exp.public_view_errors(duplicated, authority_rows)
    nonopaque = deepcopy(public_rows)
    nonopaque[0]["unit_id"] = "visible-label"
    assert "public_unit_id_not_opaque" in exp.public_view_errors(nonopaque, authority_rows)


def test_scenario_verify_7195_diagnosis_uses_raw_hashes_and_preserves_null() -> None:
    """SCENARIO-VERIFY-7195-DIAGNOSIS retains the failed V633 mechanism."""

    trace = json.loads(
        (REPO / "results/experiment_7181_v633_qwen38_symbolic_traces.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (REPO / "results/experiment_7182_v633_grounding_energy_audit.json").read_text(
            encoding="utf-8"
        )
    )
    authority = exp.read_jsonl(
        REPO / "results/experiment_7180_v633_symbolic_edit_fixture_authority.jsonl"
    )
    public = exp.read_jsonl(
        REPO / "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
    )

    rows = exp.decompose_old_errors(trace, audit, public, authority)
    exact_raw_row_hashes = {
        exp.sha256_bytes(exp.canonical_json(row).encode("utf-8")) for row in trace["rows"]
    }

    assert [row["category"] for row in rows] == list(exp.ERROR_CATEGORIES)
    assert all(row["observed_count"] > 0 for row in rows)
    assert all(row["raw_row_hashes"] for row in rows)
    assert all(value in exact_raw_row_hashes for row in rows for value in row["raw_row_hashes"])
    assert all(row["old_grounding_value_score"] == 0 for row in rows)
    assert rows[0]["old_parse_failure_denominator"] == 128

    changed_trace = deepcopy(trace)
    authority_by_id = {row["unit_id"]: row for row in authority}
    target = next(
        row
        for row in changed_trace["rows"]
        if authority_by_id[row["unit_id"]]["split"] == "evaluation"
        and authority_by_id[row["unit_id"]]["expected_response"]["evidence_tuple"] is not None
        and row["parse_status"] == "valid"
        and json.loads(row["raw_output"])["evidence_tuple"] is not None
    )
    response = json.loads(target["raw_output"])
    response.update({"evidence_tuple": None, "missing_fields": ["evidence_tuple"]})
    response.update({"source_start": None, "source_end": None})
    target["raw_output"] = exp.canonical_json(response)
    target["raw_output_sha256"] = exp.sha256_bytes(target["raw_output"].encode("utf-8"))
    changed_rows = exp.decompose_old_errors(changed_trace, audit, public, authority)
    assert (
        next(row for row in changed_rows if row["category"] == "unresolved_evidence")[
            "observed_count"
        ]
        > next(row for row in rows if row["category"] == "unresolved_evidence")["observed_count"]
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], dict[str, Path]]:
    """Build one complete artifact through private sidecar and checkpoint paths."""

    directory = tmp_path_factory.mktemp("exp7195")
    paths = {
        "result": directory / "result.json",
        "checkpoint": directory / "checkpoints" / "running.json",
        "public": directory / "public.jsonl",
        "authority": directory / "authority.jsonl",
    }
    paths["checkpoint"].parent.mkdir()
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        output_path=paths["result"],
        checkpoint_path=paths["checkpoint"],
        public_view_path=paths["public"],
        authority_sidecar_path=paths["authority"],
        duration_s=0.25,
    )
    return artifact, paths


def test_scenario_verify_7195_artifact_replays_end_to_end(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7195-ARTIFACT runs public-to-executor-to-label scoring."""

    artifact, paths = built

    assert artifact["typed_executor_ready_score"] == 1
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert len(artifact["rows"]) == 192
    assert len(artifact["semantic_mutation_rows"]) >= 8
    assert (
        exp.validate_artifact(
            artifact,
            root=REPO,
            public_view_path=paths["public"],
            authority_sidecar_path=paths["authority"],
        )
        == []
    )
    assert (
        exp.validate_artifact(
            paths["result"],
            root=REPO,
            public_view_path=paths["public"],
            authority_sidecar_path=paths["authority"],
        )
        == []
    )

    changed = deepcopy(artifact)
    changed["rows"][0]["metric"] = 0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed,
        root=REPO,
        public_view_path=paths["public"],
        authority_sidecar_path=paths["authority"],
    )

    bad_sidecar_hash = deepcopy(artifact)
    bad_sidecar_hash["fixture_manifest"]["sidecar_hashes"]["public_view_sha256"] = "sha256:0"
    assert "sidecar_hashes_mismatch" in exp.validate_artifact(
        bad_sidecar_hash,
        root=REPO,
        public_view_path=paths["public"],
        authority_sidecar_path=paths["authority"],
    )


def test_scenario_verify_7195_preflight_rejects_structured_quarantine(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7195-PREFLIGHT checks quarantine before field gates."""

    quarantined = json.loads(
        (REPO / "results/experiment_7182_v633_grounding_energy_audit.json").read_text(
            encoding="utf-8"
        )
    )
    quarantined["flagged_adversarial"] = True
    upstream_path = tmp_path / "quarantined.json"
    upstream_path.write_text(json.dumps(quarantined), encoding="utf-8")
    source_paths = {"exp7182_artifact": upstream_path}
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoints" / "running.json"
    checkpoint.parent.mkdir()

    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        output_path=output,
        checkpoint_path=checkpoint,
        public_view_path=tmp_path / "public.jsonl",
        authority_sidecar_path=tmp_path / "authority.jsonl",
        source_paths=source_paths,
        duration_s=0.1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["typed_executor_ready_score"] == 0
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == "exp7182_artifact_quarantine"
    assert output.is_file()
    assert exp.validate_artifact(artifact, root=REPO) == []
    assert exp._is_quarantined(
        {"flagged_adversarial": {"principle": "structured flag", "value": True}}
    )


def test_scenario_verify_7195_preflight_reports_each_external_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7195-PREFLIGHT makes every external block reproducible."""

    outputs = (
        tmp_path / "result.json",
        tmp_path / "checkpoint.json",
        tmp_path / "public.jsonl",
        tmp_path / "authority.jsonl",
    )

    def failed_check(
        overrides: dict[str, Path] | None = None,
        *,
        run_date: str = exp.RUN_DATE,
        selected_outputs: tuple[Path, ...] = outputs,
    ) -> str | None:
        checks, failure, _, _ = exp._preconditions(
            REPO,
            run_date,
            exp._resolved_paths(REPO, overrides),
            selected_outputs,
        )
        assert checks
        return None if failure is None else str(failure["check"])

    assert failed_check(run_date="19000101") == "run_date"
    assert failed_check({"exp7158_artifact": tmp_path / "missing.json"}) == "exp7158_artifact_path"

    no_requirement = tmp_path / "spec.md"
    no_requirement.write_text("# no requirement\n", encoding="utf-8")
    assert failed_check({"constraint_spec": no_requirement}) == "constraint_spec_requirement"

    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]\n", encoding="utf-8")
    assert failed_check({"exp7182_artifact": not_object}) == "exp7182_artifact_json"

    excluded = tmp_path / "exclusions.yaml"
    excluded.write_text("exclusions:\n  - experiment_id: '7182'\n", encoding="utf-8")
    assert failed_check({"exclusion_manifest": excluded}) == "upstream_manifest_quarantine"

    upstream = json.loads((REPO / exp.SOURCE_PATHS["exp7182_artifact"]).read_text(encoding="utf-8"))
    wrong_field = deepcopy(upstream)
    wrong_field["status"] = "failed"
    wrong_field_path = tmp_path / "wrong-field.json"
    wrong_field_path.write_text(json.dumps(wrong_field), encoding="utf-8")
    assert (
        failed_check({"exp7182_artifact": wrong_field_path}) == "exp7182_artifact_terminal_fields"
    )

    wrong_checksum = deepcopy(upstream)
    wrong_checksum["arm_metrics"]["baseline_direct"]["accuracy"] = -1.0
    wrong_checksum_path = tmp_path / "wrong-checksum.json"
    wrong_checksum_path.write_text(json.dumps(wrong_checksum), encoding="utf-8")
    assert failed_check({"exp7182_artifact": wrong_checksum_path}) == "exp7182_artifact_checksum"

    short_sidecar = tmp_path / "short.jsonl"
    short_sidecar.write_text("{}\n", encoding="utf-8")
    assert failed_check({"exp7180_public": short_sidecar}) == "exp7180_sidecar_rows"

    monkeypatch.setattr(exp.sys, "executable", str(tmp_path / "missing-python"))
    assert failed_check() == "python_executable"
    monkeypatch.undo()

    missing_parent = tmp_path / "missing-parent" / "result.json"
    assert failed_check(selected_outputs=(missing_parent,)) == "output_directory"


def test_scenario_verify_7195_cold_validator_rejects_tampering(
    built: tuple[dict[str, object], dict[str, Path]], tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-7195-ARTIFACT rejects every terminal evidence seam."""

    artifact, paths = built
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_unreadable"]
    assert exp.validate_artifact([]) == ["artifact_not_object"]
    extra = deepcopy(artifact)
    extra["unexpected"] = True
    assert exp.validate_artifact(extra) == ["artifact_fields_mismatch"]

    complete_mutations = (
        ("field_principles", {}, "field_principles_mismatch"),
        ("run_date", "19000101", "run_date_mismatch"),
        ("inference_substrate", "live_model", "inference_substrate_mismatch"),
        ("execution_venue", "unknown", "execution_venue_mismatch"),
        ("MODEL_SPECS", [{}], "model_invocation_mismatch"),
        ("random_seed", 0, "random_seed_mismatch"),
        ("status", "running", "complete_status_mismatch"),
        ("inference_substrate_class", "blocked_no_run", "complete_substrate_class_mismatch"),
        ("gate_check_summary", {}, "complete_gate_summary_mismatch"),
        ("verdict_class", "null", "complete_verdict_class_mismatch"),
        ("honest_verdict", "complete_null", "complete_honest_verdict_mismatch"),
        ("source_artifact_hashes", {}, "source_artifact_hashes_mismatch"),
        ("error_decomposition_rows", [], "error_decomposition_rows_mismatch"),
        ("sample_size_budget", {}, "sample_size_budget_mismatch"),
        ("semantic_mutation_rows", [], "semantic_mutation_rows_mismatch"),
        ("fixture_manifest", {}, "fixture_manifest_mismatch"),
    )
    for field, value, expected in complete_mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(
            changed,
            root=REPO,
            public_view_path=paths["public"],
            authority_sidecar_path=paths["authority"],
        )

    failed = exp._gate("external", "upstream", "field", True, False, False)
    blocked = exp._blocked_artifact(
        exp._base_artifact(REPO, exp.RUN_DATE, paths["public"], paths["authority"]),
        [failed],
        failed,
        {},
        0.1,
    )
    assert exp.validate_artifact(blocked, root=REPO) == []
    for field, value, expected in (
        ("status", "complete", "blocked_status_mismatch"),
        (
            "inference_substrate_class",
            exp.INFERENCE_SUBSTRATE_CLASS,
            "blocked_substrate_class_mismatch",
        ),
        ("gate_check_summary", {}, "blocked_gate_summary_mismatch"),
        ("typed_executor_ready_score", 1, "blocked_readiness_mismatch"),
    ):
        changed = deepcopy(blocked)
        changed[field] = value
        assert expected in exp.validate_artifact(changed, root=REPO)

    assert "source_artifact_missing" in exp.validate_artifact(artifact, root=tmp_path)
    assert "sidecar_or_upstream_unreadable" in exp.validate_artifact(
        artifact,
        root=REPO,
        public_view_path=tmp_path / "missing-public.jsonl",
        authority_sidecar_path=paths["authority"],
    )

    public_rows = exp.read_jsonl(paths["public"])
    authority_rows = exp.read_jsonl(paths["authority"])
    changed_public = deepcopy(public_rows)
    changed_public[0]["source_text"] += " changed"
    changed_public_path = tmp_path / "changed-public.jsonl"
    changed_public_path.write_bytes(exp.jsonl_bytes(changed_public))
    assert "public_view_rows_mismatch" in exp.validate_artifact(
        artifact,
        root=REPO,
        public_view_path=changed_public_path,
        authority_sidecar_path=paths["authority"],
    )
    changed_authority = deepcopy(authority_rows)
    changed_authority[0]["support_label"] = "unsupported"
    changed_authority_path = tmp_path / "changed-authority.jsonl"
    changed_authority_path.write_bytes(exp.jsonl_bytes(changed_authority))
    assert "authority_sidecar_rows_mismatch" in exp.validate_artifact(
        artifact,
        root=REPO,
        public_view_path=paths["public"],
        authority_sidecar_path=changed_authority_path,
    )

    upstream = json.loads((REPO / exp.SOURCE_PATHS["exp7158_artifact"]).read_text(encoding="utf-8"))
    panel = exp.build_fresh_panel(upstream)
    malformed_public = deepcopy(public_rows)
    malformed_public[0]["split"] = "evaluation"
    assert "public_view_invalid" in exp._validate_core(
        artifact, malformed_public, authority_rows, panel
    )

    nonobject = tmp_path / "nonobject.jsonl"
    nonobject.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL row is not an object"):
        exp.read_jsonl(nonobject)
