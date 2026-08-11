"""Tests for Exp6311 V544 post-marker source scope freeze.

Spec refs: REQ-INFRA-6311, SCENARIO-INFRA-6311-1,
SCENARIO-INFRA-6311-2, SCENARIO-INFRA-6311-3,
SCENARIO-INFRA-6311-4, SCENARIO-INFRA-6311-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6311_v544_post_marker_source_scope_freeze as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _references() -> str:
    return (
        "## V543 Planner Refresh (2026-08-10, after milestone 2026.08.542)\n\n"
        "- prior paper.\n"
        "<!-- V543-PLANNER-REFRESH-20260810-END -->\n\n"
        "## V544 Planner Refresh (2026-08-11, after milestone 2026.08.543)\n\n"
        "- **Activation Probes Surface Code-Security Signals that the Model's Output "
        "Misses** - arXiv:2608.09643, https://arxiv.org/abs/2608.09643.\n"
        "<!-- V544-PLANNER-REFRESH-20260811-END -->\n"
    )


def _make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel_path in mod.PROTECTED_RELATIVE_PATHS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == mod.RESEARCH_REFERENCES_RELATIVE_PATH:
            path.write_text(_references(), encoding="utf-8")
        elif rel_path == mod.ROADMAP_RELATIVE_PATH:
            shutil.copyfile(REPO / mod.ROADMAP_RELATIVE_PATH, path)
        elif rel_path == mod.VNEXT_RELATIVE_PATH:
            shutil.copyfile(REPO / mod.VNEXT_RELATIVE_PATH, path)
        elif rel_path == mod.EXCLUSION_MANIFEST_RELATIVE_PATH:
            shutil.copyfile(REPO / mod.EXCLUSION_MANIFEST_RELATIVE_PATH, path)
        else:
            path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    return root


def _accepted_candidate() -> mod.JsonDict:
    return {
        "stable_id": "arxiv:2608.99999",
        "title": "Post-marker V544 model-local fixture",
        "url": "https://arxiv.org/abs/2608.99999",
        "source_channel": "arxiv",
        "source_kind": "primary",
        "source_timestamp": "2026-08-11T11:15:27Z",
        "date_evidence": "submitted one second after the V544 marker commit",
        "scope_effect": "tightens the frozen model-local surface contract",
        "reproducible_evidence": True,
        "primary_or_first_party": True,
        "local_executable_consequence": True,
        "watch_only": False,
        "content_hash": "sha256:" + "c" * 64,
    }


def test_req_infra_6311_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6311: OpenSpec records the V544 source freeze contract."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6311") : text.index("REQ-INFRA-6298")]

    for token in (
        "REQ-INFRA-6311",
        "SCENARIO-INFRA-6311-1",
        "SCENARIO-INFRA-6311-2",
        "SCENARIO-INFRA-6311-3",
        "SCENARIO-INFRA-6311-4",
        "SCENARIO-INFRA-6311-5",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6311_marker_line_and_window_are_exclusive() -> None:
    """SCENARIO-INFRA-6311-1: marker parsing and timestamps fail closed."""

    marker = mod.v544_marker_snapshot(REPO)
    assert marker["marker_text"] == mod.PLANNER_MARKER
    assert marker["marker_line"] == 33811
    assert marker["marker_count"] == 1
    assert marker["marker_committed_at_utc"] == mod.MARKER_COMMITTED_AT_UTC

    accepted = _accepted_candidate()
    assert mod.classify_candidate(accepted, reference_text="")["disposition"] == "accepted"

    at_marker = deepcopy(accepted)
    at_marker["source_timestamp"] = mod.MARKER_COMMITTED_AT_UTC
    assert mod.classify_candidate(at_marker, reference_text="")["disposition"] == (
        "cutoff_confound"
    )

    bare_date = deepcopy(accepted)
    bare_date["source_timestamp"] = "2026-08-11"
    assert mod.classify_candidate(bare_date, reference_text="")["disposition"] == (
        "cutoff_confound"
    )

    unstable = deepcopy(accepted)
    unstable["url"] = "https://github.com/search?q=activation+probe"
    assert mod.classify_candidate(unstable, reference_text="")["disposition"] == "excluded"


def test_scenario_infra_6311_dedupe_watch_inaccessible_and_scope_hashes(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6311-3: duplicates and protected hash drift fail closed."""

    root = _make_repo(tmp_path / "repo")
    accepted = _accepted_candidate()
    duplicate_ref = deepcopy(accepted)
    duplicate_ref["stable_id"] = "arxiv:2608.09643"
    duplicate_ref["url"] = "https://arxiv.org/abs/2608.09643"
    duplicate_ref["title"] = (
        "Activation Probes Surface Code-Security Signals that the Model's Output Misses"
    )
    duplicate_ref["content_hash"] = "sha256:" + "d" * 64
    watch_only = deepcopy(accepted)
    watch_only["stable_id"] = "arxiv:2608.99998"
    watch_only["url"] = "https://arxiv.org/abs/2608.99998"
    watch_only["watch_only"] = True
    watch_only["content_hash"] = "sha256:" + "e" * 64
    inaccessible = deepcopy(accepted)
    inaccessible["stable_id"] = "openreview:challenge"
    inaccessible["url"] = "https://openreview.net/forum?id=dcBOEwDXP2"
    inaccessible["inaccessible"] = True
    inaccessible["content_hash"] = "sha256:" + "f" * 64
    no_consequence = deepcopy(accepted)
    no_consequence["stable_id"] = "arxiv:2608.99997"
    no_consequence["url"] = "https://arxiv.org/abs/2608.99997"
    no_consequence["local_executable_consequence"] = False
    no_consequence["content_hash"] = "sha256:" + "a" * 64

    partitions = mod.partition_candidates(
        [duplicate_ref, accepted, watch_only, inaccessible, no_consequence],
        reference_text=(root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8"),
    )

    assert [row["stable_id"] for row in partitions["accepted"]] == [accepted["stable_id"]]
    assert [row["disposition"] for row in partitions["duplicate_findings"]] == ["duplicate"]
    assert [row["disposition"] for row in partitions["watch_only_findings"]] == ["watch_only"]
    assert [row["disposition"] for row in partitions["inaccessible_sources"]] == ["inaccessible"]
    assert partitions["excluded_findings_and_reasons"][0]["rejection_reason"] == (
        "candidate lacks a local executable consequence"
    )

    before = mod.protected_hashes(root)
    (root / "CODEX.md").write_text("changed\n", encoding="utf-8")
    changed = mod.protected_unchanged(root, before)
    assert changed["all_unchanged"] is False
    assert changed["paths"]["CODEX.md"]["unchanged"] is False


def test_scenario_infra_6311_zero_delta_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6311-2: zero-source reports do not rewrite references."""

    root = _make_repo(tmp_path / "repo")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    report = mod.write_freeze(
        root,
        date="20260811",
        source_receipts=mod.DEFAULT_SOURCE_RECEIPTS,
        candidates=mod.DEFAULT_SOURCE_CANDIDATES,
        duration_s=2.0,
        search_completed_utc="2026-08-11T11:40:56Z",
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
    assert report["roadmap_scope_delta"]["delta_kind"] == "zero_source_delta"
    assert report["roadmap_scope_delta"]["references_byte_identical"] is True
    assert report["search_window_start_utc"] == mod.MARKER_COMMITTED_AT_UTC
    assert mod.validate_report(report) == []


def test_scenario_infra_6311_frozen_contracts_preserve_v544_boundaries() -> None:
    """SCENARIO-INFRA-6311-4: contract serializers exclude retired mechanisms."""

    report = mod.build_report(
        REPO,
        date="20260811",
        source_receipts=mod.DEFAULT_SOURCE_RECEIPTS,
        candidates=mod.DEFAULT_SOURCE_CANDIDATES,
        duration_s=2.0,
        search_completed_utc="2026-08-11T11:40:56Z",
    )

    assert report["status"] == "complete"
    assert report["frozen_model_local_surface_contract"]["version"] == mod.CONTRACT_VERSION
    assert report["frozen_model_local_surface_contract"]["shared_activation_bus_allowed"] is False
    assert report["frozen_exact_pair_fixture_contract"]["exact_sidecar_is_release_oracle"] is True
    assert report["frozen_model_local_energy_contract"]["one_head_per_model"] is True
    assert report["frozen_versioned_learning_contract"]["same_domain_only"] is True
    assert report["frozen_protected_validation_contract"]["adaptive_loop_can_read"] is False
    assert report["frozen_arc_shadow_no_solve_contract"]["solve_credit_allowed"] is False
    assert report["frozen_hardware_exclusions"]["hardware_claim_count"] == 0
    assert set(report["frozen_hardware_exclusions"]["excluded_retired_mechanisms"]) >= {
        "shared_activation_bus",
        "licensed_cross_family_transfer",
        "external_generated_text_scorer",
        "extropic_tsu_execution",
        "unchanged_physical_board_probe",
    }


def test_scenario_infra_6311_schema_and_checksum_are_machine_checkable() -> None:
    """SCENARIO-INFRA-6311-5: malformed artifacts fail validation."""

    report = mod.build_report(
        REPO,
        date="20260811",
        source_receipts=mod.DEFAULT_SOURCE_RECEIPTS,
        candidates=mod.DEFAULT_SOURCE_CANDIDATES,
        duration_s=2.0,
        search_completed_utc="2026-08-11T11:40:56Z",
    )

    assert mod.validate_report(report) == []
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["source_queries_by_channel"]) == set(mod.REQUIRED_SOURCE_CHANNELS)
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False

    missing = deepcopy(report)
    del missing["status"]
    assert "missing:status" in mod.validate_report(missing)

    bad_count = deepcopy(report)
    bad_count["accepted_count"] = "0"
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    assert "accepted_count_bare_integer" in mod.validate_report(bad_count)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)

    bad_principles = deepcopy(report)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.payload_checksum(bad_principles)
    assert "field_principles" in mod.validate_report(bad_principles)

    bad_status = deepcopy(report)
    bad_status["honest_verdict"] = "running"
    bad_status["reproducibility_checksum"] = mod.payload_checksum(bad_status)
    assert "honest_verdict" in mod.validate_report(bad_status)

    assert mod.honest_verdict("blocked", 0).startswith("blocked:")
    assert mod.honest_verdict("complete", 1).startswith("complete_delta:")


def test_req_infra_6311_helper_edges_and_append_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6311: helper edge cases stay deterministic and closed."""

    assert mod._read_text(tmp_path / "missing.md") == ""
    assert mod._parse_timestamp("not-a-time") is None
    assert mod._parse_timestamp("2026-08-11T11:15:27").tzinfo is not None
    assert mod._is_stable_url("ftp://example.com/file") is False

    accepted = _accepted_candidate()
    title_dup = deepcopy(accepted)
    assert (
        mod.classify_candidate(title_dup, reference_text=title_dup["title"])["rejection_reason"]
        == "title already appears in research-references.md"
    )
    seen_id = deepcopy(accepted)
    assert (
        mod.classify_candidate(seen_id, reference_text="", seen_ids={seen_id["stable_id"]})[
            "rejection_reason"
        ]
        == "stable id repeated in this sweep"
    )
    seen_hash = deepcopy(accepted)
    assert (
        mod.classify_candidate(
            seen_hash, reference_text="", seen_hashes={seen_hash["content_hash"]}
        )["rejection_reason"]
        == "content hash repeated in this sweep"
    )
    for field, value, reason in (
        ("reproducible_evidence", False, "candidate lacks reproducible evidence"),
        ("primary_or_first_party", False, "candidate is not primary or first-party"),
    ):
        row = deepcopy(accepted)
        row[field] = value
        assert mod.classify_candidate(row, reference_text="")["rejection_reason"] == reason

    for field, value, message in (
        ("url", None, "missing fields"),
        ("url", "ftp://example.com/file", "stable URL"),
        ("content_hash", "sha256:bad", "content hash"),
        ("source_timestamp", mod.MARKER_COMMITTED_AT_UTC, "strictly after"),
        ("reproducible_evidence", False, "reproducible"),
        ("primary_or_first_party", False, "primary"),
        ("local_executable_consequence", False, "local executable"),
        ("watch_only", True, "watch-only"),
    ):
        row = deepcopy(accepted)
        row[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_accepted_candidate(row)

    block = mod.execution_delta_block([accepted])
    source = _references()
    inserted = mod.insert_after_marker(source, block)
    assert mod.EXECUTION_DELTA_HEADING in inserted
    assert mod.insert_after_marker(inserted, block) == inserted
    assert mod.insert_after_marker("no marker", block).endswith(block)

    root = _make_repo(tmp_path / "append-repo")
    artifact_root = tmp_path / "append-artifacts"
    artifact_root.mkdir()
    report = mod.write_freeze(
        root,
        date="20260811",
        candidates=[accepted],
        duration_s=2.0,
        search_completed_utc="2026-08-11T11:40:56Z",
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )
    assert report["accepted_count"] == 1
    assert report["roadmap_scope_delta"]["delta_kind"] == "accepted_source_delta"
    assert mod.EXECUTION_DELTA_HEADING in (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )

    report = mod.build_report(
        REPO,
        date="20260811",
        duration_s=2.0,
        search_completed_utc="2026-08-11T11:40:56Z",
    )
    for mutator, error in (
        (lambda data: data.update({"accepted_count": 1}), "accepted_count"),
        (lambda data: data.update({"source_queries_by_channel": {}}), "source_queries_by_channel"),
        (lambda data: data.update({"source_receipts": []}), "source_receipts"),
        (
            lambda data: data["frozen_model_local_surface_contract"].update({"version": "bad"}),
            "frozen_model_local_surface_contract",
        ),
        (
            lambda data: data.update({"semantic_scholar_ebt_and_arm_ebm_receipts": {}}),
            "semantic_scholar_ebt_and_arm_ebm_receipts",
        ),
        (lambda data: data.update({"extropic_status": None}), "extropic_status"),
        (
            lambda data: data["protected_files_unchanged"].update({"all_unchanged": False}),
            "protected_files_unchanged",
        ),
        (lambda data: data.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda data: data.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (
            lambda data: data.update({"search_completed_utc": "2026-08-11T11:15:25Z"}),
            "search_window",
        ),
        (
            lambda data: data["field_provenance"]["status"].update({"principle": "bad"}),
            "field_provenance:status",
        ),
    ):
        malformed = deepcopy(report)
        mutator(malformed)
        malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
        assert error in mod.validate_report(malformed)

    invalid_root = _make_repo(tmp_path / "invalid-repo")
    invalid_artifact_root = tmp_path / "invalid-artifacts"
    invalid_artifact_root.mkdir()
    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp6311 freeze"):
        mod.write_freeze(
            invalid_root,
            date="20260811",
            duration_s=2.0,
            search_completed_utc="2026-08-11T11:40:56Z",
            env={ARTIFACT_ROOT_ENV: str(invalid_artifact_root)},
        )
