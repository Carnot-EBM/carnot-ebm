"""Tests for Exp6198 V537 post-marker source and roadmap-scope audit.

Spec refs: REQ-INFRA-6198, SCENARIO-INFRA-6198-1,
SCENARIO-INFRA-6198-2, SCENARIO-INFRA-6198-3,
SCENARIO-INFRA-6198-4, SCENARIO-INFRA-6198-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6198_v537_post_marker_source_scope_audit as mod
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


def test_req_infra_6198_recorded_full_suite_receipt_is_nonzero() -> None:
    """REQ-INFRA-6198: broad-suite instability is recorded, not hidden."""

    receipt = mod._recorded_command_runner(
        (".venv/bin/pytest", "tests/python", "-q"), REPO
    )

    assert receipt["exit_code"] == 2
    assert receipt["classification"] == "interrupted_after_unrelated_broad_suite_failures"
    assert "359 failed" in receipt["stdout_tail"]


def _references() -> str:
    return (
        "## V537 Planner Refresh (2026-08-07, after milestone 2026.08.536)\n\n"
        "- **WybeCoder: Verified Imperative Code Generation** - arXiv:2603.29088.\n"
        "- **RepoZero: Can LLMs Generate a Code Repository from Scratch?** - arXiv:2605.07122.\n"
        "<!-- V537-PLANNER-REFRESH-20260807-END -->\n"
    )


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(_references(), encoding="utf-8")
    shutil.copyfile(REPO / "research-roadmap.yaml", root / mod.ACTIVE_ROADMAP_RELATIVE_PATH)
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
        "title": "Post-marker primary method fixture",
        "url": "https://arxiv.org/abs/2608.99999",
        "date": "2026-08-07T19:00:00Z",
        "authority": "arXiv",
        "source_kind": "primary",
        "date_evidence": "submitted 2026-08-07T19:00:00Z",
        "local_reachability": "reachable_primary_fixture",
        "roadmap_task": "exp6200-three-family-raw-code-transport-canary",
        "changed_method_or_gate": "tightens the canary transport gate",
        "retirement_conflict": "none",
        "reason": "Primary fixture is strictly after the marker and changes a V537 gate.",
        "primary_or_first_party": True,
        "dated_reproducible": True,
        "new_applicability": True,
    }


def test_req_infra_6198_spec_declares_source_scope_contract() -> None:
    """REQ-INFRA-6198: OpenSpec names the V537 source and roadmap audit."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6198") :]

    for marker in (
        "REQ-INFRA-6198",
        "SCENARIO-INFRA-6198-1",
        "SCENARIO-INFRA-6198-2",
        "SCENARIO-INFRA-6198-3",
        "SCENARIO-INFRA-6198-4",
        "SCENARIO-INFRA-6198-5",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6198_marker_and_zero_delta_preserve_references(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6198-2: source-null reports do not rewrite references."""

    root = _make_repo(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    report = mod.write_audit(
        root,
        date="20260807",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
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
    assert report["roadmap_path_and_hash"]["requested_missing"] is True
    assert report["roadmap_path_and_hash"]["audited_path"] == "research-roadmap.yaml"
    assert mod.validate_report(report) == []


def test_scenario_infra_6198_candidate_classifier_is_strict() -> None:
    """SCENARIO-INFRA-6198-3: accepted rows need strict date and scope safety."""

    accepted = _accepted_candidate()
    assert mod.classify_candidate(accepted)["disposition"] == "accepted"

    same_time = deepcopy(accepted)
    same_time["date"] = mod.MARKER_COMMITTED_AT
    assert mod.classify_candidate(same_time)["disposition"] == "cutoff_confound"

    same_day_without_time = deepcopy(accepted)
    same_day_without_time["date"] = "2026-08-07"
    assert mod.classify_candidate(same_day_without_time)["disposition"] == (
        "cutoff_confound"
    )

    retired = deepcopy(accepted)
    retired["retirement_conflict"] = "retired KAN training"
    assert mod.classify_candidate(retired)["disposition"] == "guarded"

    no_change = deepcopy(accepted)
    no_change["new_applicability"] = False
    assert mod.classify_candidate(no_change)["disposition"] == "rejected"

    self_repo = deepcopy(accepted)
    self_repo["stable_id"] = "github:Carnot-EBM/carnot-ebm"
    self_repo["url"] = "https://github.com/Carnot-EBM/carnot-ebm"
    assert mod.classify_candidate(self_repo)["disposition"] == "duplicate"

    duplicate = deepcopy(accepted)
    duplicate["stable_id"] = accepted["stable_id"]
    unique, duplicates = mod.deduplicate_candidates([accepted, duplicate])
    assert unique == [accepted]
    assert duplicates[0]["disposition"] == "duplicate"

    endpoint_failed = deepcopy(accepted)
    endpoint_failed["endpoint_failed"] = True
    assert mod.classify_candidate(endpoint_failed)["disposition"] == "endpoint_failed"

    malformed_date = deepcopy(accepted)
    malformed_date["date"] = "not-a-timestamp"
    assert mod.classify_candidate(malformed_date)["disposition"] == "cutoff_confound"

    no_primary_evidence = deepcopy(accepted)
    no_primary_evidence["primary_or_first_party"] = False
    assert mod.classify_candidate(no_primary_evidence)["disposition"] == "rejected"


def test_req_infra_6198_accepted_candidate_contract_rejects_bad_rows() -> None:
    """SCENARIO-INFRA-6198-3: accepted rows fail closed before appending."""

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
        ("stable_id", "github:Carnot-EBM/carnot-ebm", "current project"),
    )
    for field, value, message in mutations:
        row = deepcopy(valid)
        row[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_accepted_candidate(row)


def test_scenario_infra_6198_active_roadmap_scope_audit_is_mechanical() -> None:
    """SCENARIO-INFRA-6198-4: the V537 roadmap satisfies allocation rules."""

    report = mod.build_report(
        REPO,
        date="20260807",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
        command_runner=_fake_runner,
        duration_s=2.0,
    )

    assert report["task_count"] == 14
    assert report["infra_slot_count"] >= 2
    assert report["phase_d_slot_count"] == 6
    assert report["arc_slot_count"] == 1
    assert report["continuous_self_learning_slot_count"] >= 1
    assert report["hardware_continuity_result"]["gatemate_task_count"] == 1
    assert report["roadmap_schema_result"]["passed"] is True
    assert report["exclusion_manifest_lint_result"]["passed"] is True
    assert report["gate_structure_result"]["roadmap_gate_audit_passed"] is True
    assert report["prior_failure_contract_result"]["missing_prior_failure_count"] == 0
    assert report["model_specs_rule_result"]["passed"] is True
    assert report["prompt_section_and_ending_result"]["prompt_count"] == 14
    assert report["prompt_section_and_ending_result"]["all_prompts_passed"] is True
    assert report["retired_scope_match_count"] == 0


def test_req_infra_6198_mechanical_audits_fail_closed_on_bad_inputs(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6198-4: malformed roadmap and prompt contracts are visible."""

    root = _make_repo(tmp_path)
    shutil.copyfile(REPO / "research-roadmap.yaml", root / mod.STAGED_ROADMAP_RELATIVE_PATH)
    roadmap_path, requested_missing, note = mod._select_roadmap_path(root)
    assert roadmap_path.name == mod.STAGED_ROADMAP_RELATIVE_PATH.name
    assert requested_missing is False
    assert note == "requested staged roadmap exists"

    assert mod._read_text(root / "missing.txt") == ""
    schema = mod._roadmap_schema_result({"tasks": [{"id": "bad"}]})
    assert schema["passed"] is False
    assert schema["task_count"] == 1

    prior = mod._prior_failure_contract(
        [
            {"id": "missing"},
            {"id": "malformed", "prior_failures": [{"experiment_id": "exp1"}]},
        ]
    )
    assert prior["missing_task_ids"] == ["missing"]
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


def test_scenario_infra_6198_required_fields_principles_and_validation() -> None:
    """SCENARIO-INFRA-6198-5: malformed Exp6198 artifacts fail validation."""

    report = mod.build_report(
        REPO,
        date="20260807",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
        command_runner=_fake_runner,
        duration_s=2.0,
    )

    assert mod.validate_report(report) == []
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

    bad_models = deepcopy(report)
    bad_models["model_specs_rule_result"]["passed"] = False
    bad_models["reproducibility_checksum"] = mod.payload_checksum(bad_models)
    assert "model_specs_rule_result" in mod.validate_report(bad_models)

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
        ("infra_slot_count", 1, "infra_slot_count"),
        ("phase_d_slot_count", 5, "phase_d_slot_count"),
        ("arc_slot_count", 2, "arc_slot_count"),
        ("continuous_self_learning_slot_count", 0, "continuous_self_learning_slot_count"),
        ("field_principles", [], "field_principles"),
        ("field_provenance", [], "field_provenance"),
    )
    for field, value, error in mutations:
        malformed = deepcopy(report)
        malformed[field] = value
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert error in mod.validate_report(malformed)

    bad_gate = deepcopy(report)
    bad_gate["gate_structure_result"]["roadmap_gate_audit_passed"] = False
    bad_gate["reproducibility_checksum"] = mod.payload_checksum(bad_gate)
    assert "gate_structure_result" in mod.validate_report(bad_gate)

    for field in (
        "roadmap_schema_result",
        "exclusion_manifest_lint_result",
        "prior_failure_contract_result",
        "hardware_continuity_result",
        "prompt_section_and_ending_result",
        "protected_files_unchanged",
    ):
        malformed = deepcopy(report)
        key = "all_unchanged" if field == "protected_files_unchanged" else "passed"
        malformed[field][key] = False
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert field in mod.validate_report(malformed)

    assert mod.honest_verdict("blocked", 0).startswith("blocked:")
    assert mod.honest_verdict("complete", 1).startswith("complete_delta:")


def test_req_infra_6198_append_helpers_are_marker_bounded(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6198-2: reference append helpers write only after the marker."""

    accepted = mod.classify_candidate(_accepted_candidate())
    block = mod._execution_delta_block([accepted])
    assert accepted["title"] in block
    assert mod.EXECUTION_DELTA_END_MARKER in block

    source = _references()
    inserted = mod._insert_after_marker(source, block)
    assert source != inserted
    assert inserted.index(mod.PLANNER_END_MARKER) < inserted.index(mod.EXECUTION_DELTA_HEADING)
    assert mod._insert_after_marker(inserted, block) == inserted
    assert mod._insert_after_marker("no marker here", block).endswith(block)

    root = _make_repo(tmp_path / "append-repo")
    artifact_root = tmp_path / "append-artifacts"
    artifact_root.mkdir()
    report = mod.write_audit(
        root,
        date="20260807",
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


def test_req_infra_6198_writer_refuses_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6198: invalid reports are not written."""

    root = _make_repo(tmp_path / "repo")
    target_root = tmp_path / "artifacts"
    target_root.mkdir()
    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])

    with pytest.raises(ValueError, match="invalid Exp6198 audit"):
        mod.write_audit(
            root,
            date="20260807",
            source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
            command_runner=_fake_runner,
            env={ARTIFACT_ROOT_ENV: str(target_root)},
        )
    assert not (target_root / mod.RESULT_RELATIVE_PATH.name).exists()
