"""Tests for Exp6273 V541 post-marker source scope freeze.

Spec refs: REQ-INFRA-6273, SCENARIO-INFRA-6273-1,
SCENARIO-INFRA-6273-2, SCENARIO-INFRA-6273-3,
SCENARIO-INFRA-6273-4, SCENARIO-INFRA-6273-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6273_v541_post_marker_source_scope_freeze as mod
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
        "## V541 Planner Refresh (2026-08-10, after milestone 2026.08.540)\n\n"
        "- **Test-Time Adaptation with Online Personalized Energy-Based Cache** - "
        "arXiv:2608.06467.\n"
        "- **Answer Set Programming Energised!** - arXiv:2607.08136.\n"
        "<!-- V541-PLANNER-REFRESH-20260810-END -->\n"
    )


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(_references(), encoding="utf-8")
    shutil.copyfile(REPO / "research-roadmap.yaml", root / mod.ACTIVE_ROADMAP_RELATIVE_PATH)
    (root / mod.VNEXT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / mod.VNEXT_RELATIVE_PATH, root / mod.VNEXT_RELATIVE_PATH)
    (root / mod.PRIOR_RESULT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / mod.PRIOR_RESULT_RELATIVE_PATH, root / mod.PRIOR_RESULT_RELATIVE_PATH)
    for rel_path in mod.PROTECTED_RELATIVE_PATHS:
        if rel_path == mod.STAGED_ROADMAP_RELATIVE_PATH:
            continue
        path = root / rel_path
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    return root


def _accepted_candidate() -> mod.JsonDict:
    return {
        "stable_id": "arxiv:2608.99997",
        "title": "Post-marker V541 contract fixture",
        "url": "https://arxiv.org/abs/2608.99997",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-10T13:30:00Z",
        "date_evidence": "submitted 2026-08-10T13:30:00Z",
        "scope_effect": "tightens the frozen ASP semantics contract",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "scope_changing": True,
        "watch_only": False,
        "content_hash": "sha256:" + "b" * 64,
    }


def test_req_infra_6273_spec_declares_contract() -> None:
    """REQ-INFRA-6273: OpenSpec records the V541 source freeze contract."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6273") :]

    for token in (
        "REQ-INFRA-6273",
        "SCENARIO-INFRA-6273-1",
        "SCENARIO-INFRA-6273-2",
        "SCENARIO-INFRA-6273-3",
        "SCENARIO-INFRA-6273-4",
        "SCENARIO-INFRA-6273-5",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6273_marker_exclusivity_and_rejections() -> None:
    """SCENARIO-INFRA-6273-1: the marker lower bound is exclusive."""

    accepted = _accepted_candidate()
    assert mod.classify_candidate(accepted, reference_text="")["disposition"] == "accepted"

    at_marker = deepcopy(accepted)
    at_marker["source_timestamp"] = mod.MARKER_COMMITTED_AT_UTC
    assert mod.classify_candidate(at_marker, reference_text="")["disposition"] == (
        "cutoff_confound"
    )

    bare_same_day = deepcopy(accepted)
    bare_same_day["source_timestamp"] = "2026-08-10"
    assert mod.classify_candidate(bare_same_day, reference_text="")["disposition"] == (
        "cutoff_confound"
    )

    unstable = deepcopy(accepted)
    unstable["url"] = "https://github.com/search?q=ASP+EBM"
    assert mod.classify_candidate(unstable, reference_text="")["disposition"] == ("unstable_url")

    duplicate_ref = deepcopy(accepted)
    assert (
        mod.classify_candidate(duplicate_ref, reference_text="already saw arxiv:2608.99997")[
            "disposition"
        ]
        == "duplicate"
    )
    duplicate_title = deepcopy(accepted)
    duplicate_title["stable_id"] = "arxiv:2608.99995"
    duplicate_title["url"] = "https://arxiv.org/abs/2608.99995"
    assert (
        mod.classify_candidate(duplicate_title, reference_text="Post-marker V541 contract fixture")[
            "rejection_reason"
        ]
        == "title already appears in research-references.md"
    )

    watch_only = deepcopy(accepted)
    watch_only["watch_only"] = True
    assert mod.classify_candidate(watch_only, reference_text="")["disposition"] == ("watch_only")

    no_scope = deepcopy(accepted)
    no_scope["scope_changing"] = False
    assert mod.classify_candidate(no_scope, reference_text="")["disposition"] == "rejected"

    endpoint_failed = deepcopy(accepted)
    endpoint_failed["endpoint_failed"] = True
    assert mod.classify_candidate(endpoint_failed, reference_text="")["disposition"] == (
        "endpoint_failed"
    )


def test_scenario_infra_6273_zero_findings_preserve_references(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6273-2: zero-source reports do not rewrite references."""

    root = _make_repo(tmp_path / "repo")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    report = mod.write_freeze(
        root,
        date="20260810",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
        candidates=mod.DEFAULT_DISCOVERED_CANDIDATES,
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
    assert isinstance(report["accepted_count"], int)
    assert report["accepted_findings"] == []
    assert report["honest_verdict"].startswith("complete_null:")
    assert report["references_append_receipt"]["append_count"] == 0
    assert report["references_append_receipt"]["references_byte_identical"] is True
    assert report["planner_marker_and_hash"]["marker_count"] == 1
    assert report["query_window"]["window_start_exclusive"] == mod.MARKER_COMMITTED_AT_UTC
    assert report["roadmap_path_and_hash"]["requested_missing"] is True
    assert mod.validate_report(report) == []


def test_scenario_infra_6273_duplicates_and_stable_urls_fail_closed() -> None:
    """SCENARIO-INFRA-6273-3: duplicate and unstable rows are rejected."""

    accepted = _accepted_candidate()
    duplicate = deepcopy(accepted)
    duplicate["title"] = "Duplicate fixture"
    unstable = deepcopy(accepted)
    unstable["stable_id"] = "github:unstable"
    unstable["url"] = "https://github.com/search?q=Energy-Based+Transfer"

    unique, rejected = mod.deduplicate_candidates(
        [accepted, duplicate, unstable], reference_text=""
    )

    assert [row["stable_id"] for row in unique] == [accepted["stable_id"]]
    assert unique[0]["disposition"] == "accepted"
    assert [row["disposition"] for row in rejected] == ["duplicate", "unstable_url"]

    valid = deepcopy(accepted)
    mod.validate_accepted_candidate(valid)
    for field, value, message in (
        ("url", "ftp://example.com/file", "stable URL"),
        ("content_hash", "sha256:bad", "content hash"),
        ("source_timestamp", mod.MARKER_COMMITTED_AT_UTC, "strictly after"),
        ("reproducible_evidence", False, "reproducible"),
        ("primary_or_first_party", False, "primary"),
        ("scope_changing", False, "scope"),
        ("watch_only", True, "watch-only"),
    ):
        row = deepcopy(valid)
        row[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_accepted_candidate(row)


def test_scenario_infra_6273_frozen_contracts_are_serialized() -> None:
    """SCENARIO-INFRA-6273-4: contracts preserve V541 boundaries."""

    report = mod.build_report(
        REPO,
        date="20260810",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
        candidates=mod.DEFAULT_DISCOVERED_CANDIDATES,
        command_runner=_fake_runner,
        duration_s=2.0,
    )

    assert report["status"] == "complete"
    assert {row["channel"] for row in report["source_channel_receipts"]} == set(
        mod.REQUIRED_SOURCE_CHANNELS
    )
    assert report["frozen_asp_semantics_contract"]["version"] == "v541.6273"
    assert report["frozen_asp_semantics_contract"]["exact_asp_solver_is_oracle"] is True
    assert report["frozen_flagship_benchmark_contract"]["model_weight_mutation_allowed"] is False
    assert report["frozen_certified_cache_contract"]["positive_negative_caches"] is True
    assert report["frozen_chronological_csl_contract"]["event_order"] == (
        "strict chronological replay"
    )
    assert report["frozen_variable_cardinality_sampler_contract"]["fixed_six_state_only"] is False
    assert report["frozen_arc_live_provenance_contract"]["level_solve_claim_allowed"] is False
    assert report["frozen_hardware_boundary"]["current_board_execution_route_supported"] is False
    assert report["frozen_hardware_boundary"]["current_tsu_execution_route_supported"] is False
    assert (
        "No current board or TSU route supports execution"
        in (report["frozen_hardware_boundary"]["claim_boundary"])
    )


def test_scenario_infra_6273_validation_is_machine_checkable() -> None:
    """SCENARIO-INFRA-6273-5: malformed artifacts fail validation."""

    report = mod.build_report(
        REPO,
        date="20260810",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
        candidates=mod.DEFAULT_DISCOVERED_CANDIDATES,
        command_runner=_fake_runner,
        duration_s=2.0,
    )

    assert mod.validate_report(report) == []
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False

    missing = deepcopy(report)
    del missing["status"]
    assert "missing:status" in mod.validate_report(missing)

    bad_count = deepcopy(report)
    bad_count["accepted_count"] = "0"
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    assert "accepted_count_bare_integer" in mod.validate_report(bad_count)

    count_mismatch = deepcopy(report)
    count_mismatch["accepted_count"] = 1
    count_mismatch["reproducibility_checksum"] = mod.payload_checksum(count_mismatch)
    assert "accepted_count" in mod.validate_report(count_mismatch)

    bad_contract = deepcopy(report)
    bad_contract["frozen_asp_semantics_contract"]["version"] = "wrong"
    bad_contract["reproducibility_checksum"] = mod.payload_checksum(bad_contract)
    assert "frozen_asp_semantics_contract" in mod.validate_report(bad_contract)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.payload_checksum(bad_provenance)
    assert "field_provenance:status" in mod.validate_report(bad_provenance)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    for field, key in (
        ("roadmap_schema_result", "passed"),
        ("exclusion_manifest_lint_result", "passed"),
        ("prior_failure_contract_result", "passed"),
        ("protected_files_unchanged", "all_unchanged"),
    ):
        malformed = deepcopy(report)
        malformed[field][key] = False
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert field in mod.validate_report(malformed)

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

    assert mod.honest_verdict("blocked", 0).startswith("blocked:")
    assert mod.honest_verdict("complete", 1).startswith("complete_delta:")


def test_req_infra_6273_append_helpers_are_marker_bounded(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6273-2: reference append helpers write after marker."""

    accepted = mod.classify_candidate(_accepted_candidate(), reference_text="")
    block = mod.execution_delta_block([accepted])
    source = _references()

    inserted = mod.insert_after_marker(source, block)
    assert source != inserted
    assert inserted.index(mod.PLANNER_END_MARKER) < inserted.index(mod.EXECUTION_DELTA_HEADING)
    assert mod.insert_after_marker(inserted, block) == inserted
    assert mod.insert_after_marker("no marker here", block).endswith(block)

    root = _make_repo(tmp_path / "append-repo")
    artifact_root = tmp_path / "append-artifacts"
    artifact_root.mkdir()
    report = mod.write_freeze(
        root,
        date="20260810",
        candidates=[_accepted_candidate()],
        command_runner=_fake_runner,
        duration_s=2.0,
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )

    assert report["accepted_count"] == 1
    assert report["references_append_receipt"]["append_count"] == 1
    assert mod.EXECUTION_DELTA_HEADING in (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )


def test_req_infra_6273_writer_refuses_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6273: invalid reports are not written."""

    root = _make_repo(tmp_path / "repo")
    target_root = tmp_path / "artifacts"
    target_root.mkdir()
    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])

    with pytest.raises(ValueError, match="invalid Exp6273 freeze"):
        mod.write_freeze(
            root,
            date="20260810",
            source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
            command_runner=_fake_runner,
            env={ARTIFACT_ROOT_ENV: str(target_root)},
        )
    assert not (target_root / mod.RESULT_RELATIVE_PATH.name).exists()


def test_req_infra_6273_helper_edges_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6273: helper edge cases stay deterministic and closed."""

    assert mod._read_text(tmp_path / "missing.txt") == ""
    assert mod._load_yaml_mapping(tmp_path / "missing.yaml") == {}
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("a: [\n", encoding="utf-8")
    assert mod._load_yaml_mapping(yaml_path) == {}
    yaml_path.write_text("- 1\n", encoding="utf-8")
    assert mod._load_yaml_mapping(yaml_path) == {}

    staged_root = tmp_path / "staged"
    staged_root.mkdir()
    staged = staged_root / mod.STAGED_ROADMAP_RELATIVE_PATH
    staged.write_text("milestone: test\n", encoding="utf-8")
    selected, missing, note = mod._select_roadmap_path(staged_root)
    assert selected == staged
    assert missing is False
    assert note == "requested staged roadmap exists"

    accepted = _accepted_candidate()
    invalid_time = deepcopy(accepted)
    invalid_time["source_timestamp"] = "not-a-time"
    assert mod.classify_candidate(invalid_time, reference_text="")["disposition"] == (
        "cutoff_confound"
    )
    naive_time = deepcopy(accepted)
    naive_time["source_timestamp"] = "2026-08-10T13:30:00"
    assert mod.classify_candidate(naive_time, reference_text="")["disposition"] == ("accepted")
    no_repro = deepcopy(accepted)
    no_repro["reproducible_evidence"] = False
    assert mod.classify_candidate(no_repro, reference_text="")["rejection_reason"] == (
        "candidate lacks reproducible evidence"
    )
    no_primary = deepcopy(accepted)
    no_primary["primary_or_first_party"] = False
    assert mod.classify_candidate(no_primary, reference_text="")["rejection_reason"] == (
        "candidate is not primary or first-party"
    )

    duplicate_hash = deepcopy(accepted)
    duplicate_hash["stable_id"] = "arxiv:2608.99996"
    duplicate_hash["url"] = "https://arxiv.org/abs/2608.99996"
    unique, rejected = mod.deduplicate_candidates([accepted, duplicate_hash], reference_text="")
    assert len(unique) == 1
    assert rejected[0]["rejection_reason"] == "content hash repeated in this sweep"

    missing_required = deepcopy(accepted)
    del missing_required["url"]
    with pytest.raises(ValueError, match="missing fields"):
        mod.validate_accepted_candidate(missing_required)

    schema = mod._roadmap_schema_result({"tasks": [{"id": "bad"}]})
    assert schema["passed"] is False
    prior = mod._prior_failure_contract(
        [
            {"id": "missing"},
            {"id": "not-mapping", "prior_failures": ["bad"]},
            {"id": "malformed", "prior_failures": [{"experiment_id": "x"}]},
        ]
    )
    assert prior["failure_count"] == 3

    def _raise_gate(_path: Path) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "audit_roadmap", _raise_gate)
    assert mod._gate_audit_result(tmp_path / "roadmap.yaml") == {
        "roadmap_gate_audit_passed": False,
        "error": "boom",
    }

    collision_root = tmp_path / "collision"
    (collision_root / "results").mkdir(parents=True)
    (collision_root / "results/experiment_6273_unexpected.json").write_text("{}", encoding="utf-8")
    collision = mod._collision_result(collision_root)
    assert collision["passed"] is False
    assert collision["collision_count"] == 1

    def _raise_run(*_args: object, **_kwargs: object) -> object:
        raise OSError("no git")

    monkeypatch.setattr(mod.subprocess, "run", _raise_run)
    assert mod._git_status(tmp_path) == []

    report = mod.build_report(
        REPO,
        date="20260810",
        source_receipts=mod.DEFAULT_SOURCE_CHANNEL_RECEIPTS,
        candidates=mod.DEFAULT_DISCOVERED_CANDIDATES,
        command_runner=_fake_runner,
        duration_s=2.0,
    )
    bad_channels = deepcopy(report)
    bad_channels["source_channel_receipts"] = bad_channels["source_channel_receipts"][:-1]
    bad_channels["reproducibility_checksum"] = mod.payload_checksum(bad_channels)
    assert "source_channel_receipts" in mod.validate_report(bad_channels)

    assert mod.check_roadmap_only(REPO)["ok"] is True
