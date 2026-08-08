"""Tests for Exp6211 V538 source ingress and ARC causal preregistration.

Spec refs: REQ-INFRA-6211, SCENARIO-INFRA-6211-1,
SCENARIO-INFRA-6211-2, SCENARIO-INFRA-6211-3,
SCENARIO-INFRA-6211-4, SCENARIO-INFRA-6211-5,
SCENARIO-INFRA-6211-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6211_v538_post_marker_source_scope_prereg as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _fake_runner(argv: tuple[str, ...], _root: Path) -> mod.JsonDict:
    return {
        "command": " ".join(argv),
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _references() -> str:
    return (
        "## V538 Planner Refresh (2026-08-07, after milestone 2026.08.537)\n\n"
        "- **ARCANA: A Reflective Multi-Agent Program Synthesis Framework** - "
        "arXiv:2607.09059.\n"
        "- **Cost-Effective Agent Harnesses for Abstract Reasoning and "
        "Generalization on ARC-AGI-1** - arXiv:2607.06764.\n"
        "- **Hyper-SET: Designing Transformers via Hyperspherical Energy "
        "Minimization** - arXiv:2502.11646.\n"
        "<!-- V538-PLANNER-REFRESH-20260807-END -->\n"
    )


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        _references(), encoding="utf-8"
    )
    shutil.copyfile(REPO / "research-roadmap.yaml", root / mod.ROADMAP_RELATIVE_PATH)
    (root / "results").mkdir(parents=True, exist_ok=True)
    for rel_path in mod.PROTECTED_RELATIVE_PATHS:
        path = root / rel_path
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    return root


def _accepted_candidate() -> mod.JsonDict:
    return {
        "stable_id": "arxiv:2608.99999",
        "title": "Post-marker primary ARC control fixture",
        "url": "https://arxiv.org/abs/2608.99999",
        "date": "2026-08-08T03:00:00Z",
        "authority": "arXiv",
        "source_kind": "primary",
        "date_evidence": "submitted 2026-08-08T03:00:00Z",
        "local_reachability": "reachable_primary_fixture",
        "roadmap_task": "exp6214-arc-object-delta-heldout-ab",
        "changed_method_or_gate": "tightens treatment-fire and matched-control gate",
        "retirement_conflict": "none",
        "reason": "Primary fixture is strictly after the marker and changes a V538 gate.",
        "primary_or_first_party": True,
        "dated_reproducible": True,
        "new_applicability": True,
    }


def test_req_infra_6211_spec_declares_v538_contract() -> None:
    """REQ-INFRA-6211: OpenSpec names the V538 source and ARC prereg audit."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6211") :]

    for marker in (
        "REQ-INFRA-6211",
        "SCENARIO-INFRA-6211-1",
        "SCENARIO-INFRA-6211-2",
        "SCENARIO-INFRA-6211-3",
        "SCENARIO-INFRA-6211-4",
        "SCENARIO-INFRA-6211-5",
        "SCENARIO-INFRA-6211-6",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6211_marker_null_and_receipts_preserve_references(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6211-2: null source searches do not rewrite references."""

    root = _make_repo(tmp_path / "repo")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    report = mod.write_prereg(
        root,
        date="20260808",
        command_runner=_fake_runner,
        duration_s=2.0,
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )

    after = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()
    target = artifact_root / mod.RESULT_RELATIVE_PATH.name

    assert after == before
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert report["status"] == "complete"
    assert report["accepted_count"] == 0
    assert report["accepted_findings"] == []
    assert report["honest_verdict"].startswith("complete_null:")
    assert report["references_append_receipt"]["append_count"] == 0
    assert report["references_append_receipt"]["references_byte_identical"] is True
    assert report["planner_marker_and_hash"]["marker_count"] == 1
    assert report["planner_marker_and_hash"]["marker_text"] == mod.PLANNER_MARKER
    assert report["query_window"]["window_start_exclusive"] == mod.MARKER_COMMITTED_AT
    assert report["retired_scope_match_count"] == 0
    assert set(report["source_channel_receipts"]["required_channels_observed"]) == set(
        mod.REQUIRED_SOURCE_CHANNELS
    )
    assert mod.validate_report(report) == []


def test_scenario_infra_6211_candidate_classifier_is_marker_strict() -> None:
    """SCENARIO-INFRA-6211-3: candidate novelty fails closed before append."""

    references_text = _references()
    accepted = _accepted_candidate()
    assert mod.classify_candidate(accepted, references_text)["disposition"] == "accepted"

    same_time = deepcopy(accepted)
    same_time["date"] = mod.MARKER_COMMITTED_AT
    assert (
        mod.classify_candidate(same_time, references_text)["disposition"]
        == "cutoff_confound"
    )

    bare_same_day = deepcopy(accepted)
    bare_same_day["date"] = "2026-08-08"
    assert (
        mod.classify_candidate(bare_same_day, references_text)["disposition"]
        == "cutoff_confound"
    )

    in_refs = deepcopy(accepted)
    in_refs["stable_id"] = "arxiv:2607.09059"
    in_refs["url"] = "https://arxiv.org/abs/2607.09059"
    in_refs["date"] = "2026-08-08T03:00:00Z"
    assert mod.classify_candidate(in_refs, references_text)["disposition"] == "duplicate"

    retired = deepcopy(accepted)
    retired["retirement_conflict"] = "retired ARC solve farming"
    assert mod.classify_candidate(retired, references_text)["disposition"] == "guarded"

    endpoint_failed = deepcopy(accepted)
    endpoint_failed["endpoint_failed"] = True
    assert (
        mod.classify_candidate(endpoint_failed, references_text)["disposition"]
        == "endpoint_failed"
    )

    no_scope_change = deepcopy(accepted)
    no_scope_change["new_applicability"] = False
    assert mod.classify_candidate(no_scope_change, references_text)["disposition"] == "rejected"

    no_primary = deepcopy(accepted)
    no_primary["primary_or_first_party"] = False
    assert mod.classify_candidate(no_primary, references_text)["disposition"] == "rejected"

    malformed_date = deepcopy(accepted)
    malformed_date["date"] = "not-a-timestamp"
    assert (
        mod.classify_candidate(malformed_date, references_text)["disposition"]
        == "cutoff_confound"
    )

    unique, duplicates = mod.deduplicate_candidates(
        [accepted, deepcopy(accepted)], references_text
    )
    assert unique == [accepted]
    assert duplicates[0]["disposition"] == "duplicate"


def test_scenario_infra_6211_active_roadmap_counts_and_contracts() -> None:
    """SCENARIO-INFRA-6211-4: the V538 roadmap and contracts are mechanical."""

    report = mod.build_report(
        REPO,
        date="20260808",
        command_runner=_fake_runner,
        duration_s=2.0,
    )

    assert report["task_count"] == 14
    assert report["phase_counts"] == {
        "arc": 6,
        "capstone": 1,
        "continuous_learning": 1,
        "infrastructure": 2,
        "phase_d": 3,
        "sampling": 1,
    }
    assert report["arc_task_count"] == 6
    assert report["continuous_self_learning_slot_count"] == 1
    assert report["arc_outcome_vocabulary"]["arc_ab_task_count"] == 4
    assert report["matched_control_contract"]["matched_budget_required"] is True
    assert report["no_solve_and_registry_nonmutation_contract"][
        "registry_mutation_allowed"
    ] is False
    assert report["hardware_boundary_result"]["unauthorized_hardware_promotion"] is False
    assert report["roadmap_schema_result"]["passed"] is True
    assert report["exclusion_manifest_lint_result"]["passed"] is True
    assert report["gate_structure_result"]["roadmap_gate_audit_passed"] is True
    assert report["prior_failure_contract_result"]["passed"] is True
    assert report["model_specs_rule_result"]["passed"] is True


def test_scenario_infra_6211_causal_contracts_fail_closed() -> None:
    """SCENARIO-INFRA-6211-5: incomplete ARC prereg contracts are invalid."""

    report = mod.build_report(
        REPO,
        date="20260808",
        command_runner=_fake_runner,
        duration_s=2.0,
    )
    assert mod.validate_report(report) == []

    missing_vocab = deepcopy(report)
    missing_vocab["arc_outcome_vocabulary"]["allowed_terminal_outcomes"].remove(
        "instrument_failure"
    )
    missing_vocab["reproducibility_checksum"] = mod.payload_checksum(missing_vocab)
    assert "arc_outcome_vocabulary" in mod.validate_report(missing_vocab)

    weak_controls = deepcopy(report)
    weak_controls["matched_control_contract"]["aa_noise_floor_required"] = False
    weak_controls["reproducibility_checksum"] = mod.payload_checksum(weak_controls)
    assert "matched_control_contract" in mod.validate_report(weak_controls)

    registry_mutable = deepcopy(report)
    registry_mutable["no_solve_and_registry_nonmutation_contract"][
        "registry_mutation_allowed"
    ] = True
    registry_mutable["reproducibility_checksum"] = mod.payload_checksum(registry_mutable)
    assert "no_solve_and_registry_nonmutation_contract" in mod.validate_report(
        registry_mutable
    )

    wrong_ab_count = deepcopy(report)
    wrong_ab_count["arc_outcome_vocabulary"]["arc_ab_task_count"] = 3
    wrong_ab_count["reproducibility_checksum"] = mod.payload_checksum(wrong_ab_count)
    assert "arc_outcome_vocabulary" in mod.validate_report(wrong_ab_count)

    retired_scope = deepcopy(report)
    retired_scope["retired_scope_match_count"] = 1
    retired_scope["reproducibility_checksum"] = mod.payload_checksum(retired_scope)
    assert "retired_scope_match_count" in mod.validate_report(retired_scope)


def test_scenario_infra_6211_required_fields_principles_and_validation() -> None:
    """SCENARIO-INFRA-6211-6: malformed Exp6211 artifacts fail validation."""

    report = mod.build_report(
        REPO,
        date="20260808",
        command_runner=_fake_runner,
        duration_s=2.0,
    )

    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(report)
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    bad_count = deepcopy(report)
    bad_count["accepted_count"] = 1
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    assert "accepted_count" in mod.validate_report(bad_count)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.payload_checksum(bad_provenance)
    assert "field_provenance:status" in mod.validate_report(bad_provenance)

    missing = deepcopy(report)
    del missing["status"]
    assert "missing:status" in mod.validate_report(missing)

    mutations = (
        ("inference_substrate", "wrong", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("honest_verdict", "wrong", "honest_verdict"),
        ("task_count", 13, "task_count"),
        ("phase_counts", {}, "phase_counts"),
        ("arc_task_count", 5, "arc_task_count"),
        ("continuous_self_learning_slot_count", 0, "continuous_self_learning_slot_count"),
        ("field_principles", [], "field_principles"),
        ("field_provenance", [], "field_provenance"),
    )
    for field, value, error in mutations:
        malformed = deepcopy(report)
        malformed[field] = value
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert error in mod.validate_report(malformed)

    for field in (
        "roadmap_schema_result",
        "exclusion_manifest_lint_result",
        "prior_failure_contract_result",
        "model_specs_rule_result",
        "hardware_boundary_result",
        "protected_files_unchanged",
    ):
        malformed = deepcopy(report)
        key = "all_unchanged" if field == "protected_files_unchanged" else "passed"
        malformed[field][key] = False
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert field in mod.validate_report(malformed)

    bad_gate = deepcopy(report)
    bad_gate["gate_structure_result"]["roadmap_gate_audit_passed"] = False
    bad_gate["reproducibility_checksum"] = mod.payload_checksum(bad_gate)
    assert "gate_structure_result" in mod.validate_report(bad_gate)

    bad_source = deepcopy(report)
    bad_source["source_channel_receipts"]["missing_required_channels"] = ["arxiv_topics"]
    bad_source["reproducibility_checksum"] = mod.payload_checksum(bad_source)
    assert "source_channel_receipts" in mod.validate_report(bad_source)

    bad_exit = deepcopy(report)
    first_command = next(iter(bad_exit["test_exit_codes"]))
    bad_exit["test_exit_codes"][first_command] = 1
    bad_exit["reproducibility_checksum"] = mod.payload_checksum(bad_exit)
    assert "test_exit_codes" in mod.validate_report(bad_exit)


def test_req_infra_6211_helper_error_branches_are_explicit(tmp_path: Path) -> None:
    """REQ-INFRA-6211: helper edge cases return structured failures."""

    assert mod.path_sha256(tmp_path / "missing.txt") is None
    assert mod._read_text(tmp_path / "missing.txt") == ""
    assert mod.honest_verdict("blocked", 0).startswith("blocked:")

    receipt = mod._recorded_command_runner(tuple(mod.TEST_COMMANDS[0].split()), REPO)
    assert receipt["exit_code"] == 0
    assert receipt["classification"] == "passed"

    valid = _accepted_candidate()
    valid["content_hash"] = "sha256:accepted-fixture"
    mod.validate_accepted_candidate(valid)

    missing = deepcopy(valid)
    del missing["url"]
    with pytest.raises(ValueError, match="missing"):
        mod.validate_accepted_candidate(missing)

    mutations = (
        ("content_hash", "not-a-sha", "content hash"),
        ("date", mod.MARKER_COMMITTED_AT, "strictly after"),
        ("primary_or_first_party", False, "primary"),
        ("dated_reproducible", False, "reproducible"),
        ("new_applicability", False, "change"),
        ("retirement_conflict", "retired", "retirement"),
    )
    for field, value, message in mutations:
        row = deepcopy(valid)
        row[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_accepted_candidate(row)

    schema = mod._roadmap_schema_result({"tasks": [{"id": "bad"}]})
    assert schema["passed"] is False
    assert schema["task_count"] == 1

    prior = mod._prior_failure_contract(
        [{"id": "malformed", "prior_failures": [{"experiment_id": "exp1"}]}]
    )
    assert prior["passed"] is False
    assert prior["malformed_task_ids"] == ["malformed"]

    assert mod._legacy_mentions_are_guarded("Qwen3.5-0.8B supplies headline rows") is False
    bad_model = mod._model_specs_rule(
        [
            {
                "id": "bad-model",
                "prompt": "MODEL_SPECS:\n- hf_id: unsupported/model\nRequired deliverable:",
            }
        ]
    )
    assert bad_model["passed"] is False
    assert bad_model["failure_task_ids"] == ["bad-model"]

    bad_prompt = mod._prompt_sections([{"id": "bad-prompt", "prompt": "CONTEXT\n"}])
    assert bad_prompt["passed"] is False
    assert bad_prompt["failures"][0]["task_id"] == "bad-prompt"


def test_req_infra_6211_append_helpers_are_marker_bounded(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6211-2: reference append helpers write only after marker."""

    accepted = mod.classify_candidate(_accepted_candidate(), _references())
    block = mod._execution_delta_block([accepted])
    assert accepted["title"] in block
    assert mod.EXECUTION_DELTA_END_MARKER in block

    source = _references()
    inserted = mod._insert_after_marker(source, block)
    assert source != inserted
    assert inserted.index(mod.PLANNER_END_MARKER) < inserted.index(
        mod.EXECUTION_DELTA_HEADING
    )
    assert mod._insert_after_marker(inserted, block) == inserted
    assert mod._insert_after_marker("no marker here", block).endswith(block)

    root = _make_repo(tmp_path / "append-repo")
    artifact_root = tmp_path / "append-artifacts"
    artifact_root.mkdir()
    report = mod.write_prereg(
        root,
        date="20260808",
        candidates=[_accepted_candidate()],
        command_runner=_fake_runner,
        duration_s=2.0,
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )
    assert report["accepted_count"] == 1
    assert report["references_append_receipt"]["append_count"] == 1
    assert mod.EXECUTION_DELTA_HEADING in (
        root / mod.RESEARCH_REFERENCES_RELATIVE_PATH
    ).read_text(encoding="utf-8")


def test_req_infra_6211_writer_refuses_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6211: invalid reports are not written."""

    root = _make_repo(tmp_path / "repo")
    target_root = tmp_path / "artifacts"
    target_root.mkdir()
    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])

    with pytest.raises(ValueError, match="invalid Exp6211 prereg"):
        mod.write_prereg(
            root,
            date="20260808",
            command_runner=_fake_runner,
            env={ARTIFACT_ROOT_ENV: str(target_root)},
        )
    assert not (target_root / mod.RESULT_RELATIVE_PATH.name).exists()
