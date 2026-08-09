"""Tests for Exp6226 V539 source scope freeze.

Spec refs: REQ-INFRA-6226, SCENARIO-INFRA-6226-1,
SCENARIO-INFRA-6226-2, SCENARIO-INFRA-6226-3,
SCENARIO-INFRA-6226-4, SCENARIO-INFRA-6226-5,
SCENARIO-INFRA-6226-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6226_v539_post_marker_source_scope_freeze as mod
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
        "## V539 Planner Refresh (2026-08-09, after milestone 2026.08.538)\n\n"
        "- **The Calibration Floor: Format Repair Can Masquerade as "
        "Self-Correction at Small-to-Mid Scale** - arXiv:2608.04355.\n"
        "- **Continual Learning in Transition** - arXiv:2608.06216.\n"
        "- **From One to One Billion: Torx, Thermalizers, and Z1** - Extropic.\n"
        "<!-- V539-PLANNER-REFRESH-20260809-END -->\n"
    )


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        _references(), encoding="utf-8"
    )
    shutil.copyfile(REPO / "research-roadmap.yaml", root / mod.ROADMAP_RELATIVE_PATH)
    staged_source = REPO / "research-roadmap-next.yaml"
    if staged_source.exists():
        shutil.copyfile(staged_source, root / mod.STAGED_ROADMAP_RELATIVE_PATH)
    else:
        shutil.copyfile(REPO / "research-roadmap.yaml", root / mod.STAGED_ROADMAP_RELATIVE_PATH)
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
        "stable_id": "arxiv:2608.99998",
        "title": "Post-marker V539 runtime ownership fixture",
        "url": "https://arxiv.org/abs/2608.99998",
        "date": "2026-08-09T23:00:00Z",
        "authority": "arXiv",
        "source_kind": "primary",
        "date_evidence": "submitted 2026-08-09T23:00:00Z",
        "local_reachability": "reachable_primary_fixture",
        "roadmap_task": "exp6227-llama-server-signal-sender-diagnostic",
        "changed_method_or_gate": "tightens runtime ownership gate",
        "retirement_conflict": "none",
        "reason": "Primary fixture is strictly after the marker and changes a V539 gate.",
        "primary_or_first_party": True,
        "dated_reproducible": True,
        "new_applicability": True,
    }


def test_req_infra_6226_spec_declares_v539_scope_freeze() -> None:
    """REQ-INFRA-6226: OpenSpec names the V539 scope-freeze contract."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6226") :]

    for marker in (
        "REQ-INFRA-6226",
        "SCENARIO-INFRA-6226-1",
        "SCENARIO-INFRA-6226-2",
        "SCENARIO-INFRA-6226-3",
        "SCENARIO-INFRA-6226-4",
        "SCENARIO-INFRA-6226-5",
        "SCENARIO-INFRA-6226-6",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6226_bootstrap_null_and_receipts_preserve_references(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6226-2: bootstrap survives a null source search."""

    root = _make_repo(tmp_path / "repo")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    report = mod.write_freeze(
        root,
        date="20260809",
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
    assert report["accepted_count"] is None
    assert report["accepted_findings"] == []
    assert report["honest_verdict"].startswith("complete_null:")
    assert report["references_append_receipt"]["append_count"] == 0
    assert report["references_append_receipt"]["references_byte_identical"] is True
    assert report["planner_marker_and_hash"]["marker_count"] == 1
    assert report["query_window"]["window_start_exclusive"] == mod.MARKER_COMMITTED_AT
    assert report["bootstrap_artifact_write_receipt"]["status"] == "bootstrap_written"
    assert report["bootstrap_artifact_write_receipt"]["survived_to_final"] is True
    assert "research-roadmap-next.yaml" in report["preconditions_checked"]["input_hashes"]
    assert set(report["source_channel_receipts"]["required_channels_observed"]) == set(
        mod.REQUIRED_SOURCE_CHANNELS
    )
    assert mod.validate_report(report) == []


def test_scenario_infra_6226_candidate_classifier_is_marker_strict() -> None:
    """SCENARIO-INFRA-6226-4: candidate novelty fails closed before append."""

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
    bare_same_day["date"] = "2026-08-09"
    assert (
        mod.classify_candidate(bare_same_day, references_text)["disposition"]
        == "cutoff_confound"
    )

    in_refs = deepcopy(accepted)
    in_refs["stable_id"] = "arxiv:2608.04355"
    in_refs["url"] = "https://arxiv.org/abs/2608.04355"
    in_refs["date"] = "2026-08-09T23:00:00Z"
    assert mod.classify_candidate(in_refs, references_text)["disposition"] == "duplicate"

    retired = deepcopy(accepted)
    retired["retirement_conflict"] = "retired finite-id grammar retry lane"
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


def test_scenario_infra_6226_current_roadmap_and_contracts_are_mechanical() -> None:
    """REQ-INFRA-6226: the V539 roadmap and frozen contracts validate."""

    report = mod.build_report(
        REPO,
        date="20260809",
        command_runner=_fake_runner,
        duration_s=2.0,
        bootstrap_receipt=mod.bootstrap_artifact_write_receipt(
            REPO,
            date="20260809",
            env=None,
            write_artifact=False,
        ),
    )

    assert report["roadmap_schema_result"]["passed"] is True
    assert report["exclusion_manifest_lint_result"]["passed"] is True
    assert report["prior_failure_contract_result"]["passed"] is True
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["frozen_runtime_contract"]["bounded_wait_required"] is True
    assert (
        report["frozen_arc_provenance_contract"]["solve_provenance_required"]
        == "live_agent_self_discovery"
    )
    assert report["frozen_code_content_margin_contract"]["parse_content_separated"] is True
    assert report["frozen_csl_contract"]["model_weight_mutation_allowed"] is False
    assert report["frozen_sampler_activation_contract"]["inactive_treatment_is_failure"] is True
    assert report["frozen_hardware_boundary"]["hardware_claim_allowed_without_receipt"] is False


def test_scenario_infra_6226_frozen_contracts_fail_closed() -> None:
    """SCENARIO-INFRA-6226-5: incomplete freeze contracts are invalid."""

    report = mod.build_report(
        REPO,
        date="20260809",
        command_runner=_fake_runner,
        duration_s=2.0,
        bootstrap_receipt=mod.bootstrap_artifact_write_receipt(
            REPO,
            date="20260809",
            env=None,
            write_artifact=False,
        ),
    )
    assert mod.validate_report(report) == []

    mutations = (
        ("frozen_runtime_contract", "task_owned_process_provenance_required", False),
        ("frozen_arc_provenance_contract", "hidden_game_source_access_allowed", True),
        ("frozen_code_content_margin_contract", "parse_content_separated", False),
        ("frozen_csl_contract", "post_outcome_commit_only", False),
        ("frozen_sampler_activation_contract", "treatment_activation_required", False),
        ("frozen_hardware_boundary", "hardware_claim_allowed_without_receipt", True),
    )
    for field, key, value in mutations:
        malformed = deepcopy(report)
        malformed[field][key] = value
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert field in mod.validate_report(malformed)


def test_scenario_infra_6226_required_fields_principles_and_validation() -> None:
    """SCENARIO-INFRA-6226-6: malformed Exp6226 artifacts fail validation."""

    report = mod.build_report(
        REPO,
        date="20260809",
        command_runner=_fake_runner,
        duration_s=2.0,
        bootstrap_receipt=mod.bootstrap_artifact_write_receipt(
            REPO,
            date="20260809",
            env=None,
            write_artifact=False,
        ),
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

    for field, value, error in (
        ("inference_substrate", "wrong", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("honest_verdict", "wrong", "honest_verdict"),
        ("field_principles", [], "field_principles"),
        ("field_provenance", [], "field_provenance"),
    ):
        malformed = deepcopy(report)
        malformed[field] = value
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert error in mod.validate_report(malformed)

    for field in (
        "roadmap_schema_result",
        "exclusion_manifest_lint_result",
        "prior_failure_contract_result",
        "protected_files_unchanged",
        "bootstrap_artifact_write_receipt",
    ):
        malformed = deepcopy(report)
        key = "all_unchanged" if field == "protected_files_unchanged" else "passed"
        if field == "bootstrap_artifact_write_receipt":
            key = "survived_to_final"
        malformed[field][key] = False
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert field in mod.validate_report(malformed)

    bad_source = deepcopy(report)
    bad_source["source_channel_receipts"]["missing_required_channels"] = ["arxiv_topics"]
    bad_source["reproducibility_checksum"] = mod.payload_checksum(bad_source)
    assert "source_channel_receipts" in mod.validate_report(bad_source)

    bad_exit = deepcopy(report)
    first_command = next(iter(bad_exit["test_exit_codes"]))
    bad_exit["test_exit_codes"][first_command] = 1
    bad_exit["reproducibility_checksum"] = mod.payload_checksum(bad_exit)
    assert "test_exit_codes" in mod.validate_report(bad_exit)


def test_req_infra_6226_helper_error_branches_are_explicit(tmp_path: Path) -> None:
    """REQ-INFRA-6226: helper edge cases return structured failures."""

    assert mod.path_sha256(tmp_path / "missing.txt") is None
    assert mod._read_text(tmp_path / "missing.txt") == ""
    assert mod.honest_verdict("blocked", None).startswith("blocked:")

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

    for field, value, message in (
        ("content_hash", "not-a-sha", "content hash"),
        ("date", mod.MARKER_COMMITTED_AT, "strictly after"),
        ("primary_or_first_party", False, "primary"),
        ("dated_reproducible", False, "reproducible"),
        ("new_applicability", False, "change"),
        ("retirement_conflict", "retired", "retirement"),
    ):
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


def test_req_infra_6226_append_helpers_are_marker_bounded(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6226-3: reference append helpers write only after marker."""

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
    report = mod.write_freeze(
        root,
        date="20260809",
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


def test_req_infra_6226_writer_refuses_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6226: invalid reports are not written."""

    root = _make_repo(tmp_path / "repo")
    target_root = tmp_path / "artifacts"
    target_root.mkdir()
    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])

    with pytest.raises(ValueError, match="invalid Exp6226 freeze"):
        mod.write_freeze(
            root,
            date="20260809",
            command_runner=_fake_runner,
            env={ARTIFACT_ROOT_ENV: str(target_root)},
        )
    bootstrap = json.loads(
        (target_root / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")
    )
    assert bootstrap["status"] == "bootstrap"
