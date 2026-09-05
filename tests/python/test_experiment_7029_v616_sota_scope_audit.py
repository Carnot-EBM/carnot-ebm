"""Tests for the V616 source audit.

Spec refs: REQ-REPORT-7029 and SCENARIO-REPORT-7029-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7029_v616_sota_scope_audit as mod


def _fake_root(path: Path, *, marker: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for relative in mod.REQUIRED_LOCAL_INPUTS:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = "{}\n"
        if relative == mod.REFERENCE_PATH:
            text = f"{mod.REFERENCE_MARKER}\n" if marker else "# no marker\n"
        elif relative == mod.EXCLUSION_PATH:
            text = "retired_experiments: []\n"
        target.write_text(text, encoding="utf-8")
    return path


def _artifact(
    tmp_path: Path,
    *,
    source_groups: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    root = _fake_root(tmp_path / "root")
    return mod.build_artifact(
        root,
        "20260905",
        output_path=tmp_path / "artifact.json",
        network_is_available=True,
        duration_s=0.2,
        source_groups=source_groups,
    )


def _refresh_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_scenario_report_7029_cutoff_requires_proved_ordering() -> None:
    """SCENARIO-REPORT-7029-CUTOFF keeps an un-timed marker day uncertain."""

    assert mod.cutoff_relation("2026-09-04") == "pre_marker"
    assert mod.cutoff_relation("2026-09-05") == "same_day_order_uncertain"
    assert mod.cutoff_relation("2026-09-05T20:00:00Z") == "same_day_order_uncertain"
    assert mod.cutoff_relation("2026-09-06") == "post_marker"
    assert (
        mod.cutoff_relation(
            "2026-09-05T20:00:00Z",
            marker_timestamp="2026-09-05T19:00:00Z",
        )
        == "post_marker"
    )
    assert (
        mod.cutoff_relation(
            "2026-09-05T18:00:00Z",
            marker_timestamp="2026-09-05T19:00:00Z",
        )
        == "pre_marker"
    )
    assert mod.cutoff_relation("not-a-date") == "unknown"


def test_scenario_report_7029_receipts_preserve_rate_limits_and_boundaries() -> None:
    """SCENARIO-REPORT-7029-RECEIPTS records access without readiness promotion."""

    row = mod.access_receipt(
        family="semantic_scholar",
        query="EBT citations",
        url="https://api.semanticscholar.org/example",
        accessed_on="2026-09-05",
        http_status=429,
        access_outcome="rate_limited",
        evidence_kind="secondary_index",
    )
    assert row["terminal"] is True
    assert row["content_verified"] is False
    assert row["implementation_evidence_claimed"] is False
    assert row["http_status"] == 429
    with pytest.raises(ValueError, match="canonical https URL"):
        mod.access_receipt(
            family="arxiv",
            query="snippet",
            url="search result only",
            accessed_on="2026-09-05",
            http_status=None,
            access_outcome="search_snippet_only",
            evidence_kind="search_result",
        )


def test_scenario_report_7029_duplicates_are_suppressed_and_classified() -> None:
    """SCENARIO-REPORT-7029-DUPLICATES records source, finding, and marker repeats."""

    rows = [
        {"row_id": "a", "source_identity": "paper:a", "finding_id": "new"},
        {"row_id": "b", "source_identity": "paper:a", "finding_id": "other"},
        {"row_id": "c", "source_identity": "paper:c", "finding_id": "new"},
        {"row_id": "d", "source_identity": "paper:d", "finding_id": "marker"},
    ]
    unique, duplicates, classes = mod.deduplicate_candidates(
        rows,
        marker_finding_ids={"marker"},
    )

    assert [row["row_id"] for row in unique] == ["a"]
    assert {row["reason"] for row in duplicates} == {
        "duplicate_source_identity",
        "duplicate_finding_identity",
        "already_present_at_v616_marker",
    }
    assert {row["candidate_id"] for row in classes} == {"b", "c", "d"}
    assert all(row["classification"] == "duplicate" for row in classes)
    assert all(row["suppressed"] is True for row in duplicates)

    blank_unique, blank_duplicates, blank_classes = mod.deduplicate_candidates(
        [{"row_id": "blank", "source_identity": "", "finding_id": ""}]
    )
    assert blank_unique == [{"row_id": "blank", "source_identity": "", "finding_id": ""}]
    assert blank_duplicates == []
    assert blank_classes == []


def test_scenario_report_7029_classification_enforces_evidence_floors() -> None:
    """SCENARIO-REPORT-7029-CLASSIFY rejects prose and downgrades missing evidence."""

    build = mod.classify_candidate(
        candidate_id="method_with_code",
        source_ids=["paper", "git"],
        proposed_action="build",
        primary_document_opened=True,
        primary_method_verified=True,
        reproducible_implementation_identity=True,
        bounded_local_test=True,
        evidence_kind="primary_paper",
        reason="The method and immutable code identity are available.",
    )
    no_code = mod.classify_candidate(
        candidate_id="method_without_code",
        source_ids=["paper"],
        proposed_action="build",
        primary_document_opened=True,
        primary_method_verified=True,
        reproducible_implementation_identity=False,
        bounded_local_test=True,
        evidence_kind="primary_paper",
        reason="The implementation identity is missing.",
    )
    abstract = mod.classify_candidate(
        candidate_id="abstract_only",
        source_ids=["index"],
        proposed_action="build",
        primary_document_opened=False,
        primary_method_verified=False,
        reproducible_implementation_identity=False,
        bounded_local_test=False,
        evidence_kind="paper_abstract",
        reason="An abstract does not prove local readiness.",
    )
    product = mod.classify_candidate(
        candidate_id="product_only",
        source_ids=["kona"],
        proposed_action="test",
        primary_document_opened=True,
        primary_method_verified=False,
        reproducible_implementation_identity=False,
        bounded_local_test=False,
        evidence_kind="first_party_product_page",
        reason="The page exposes no runner.",
        false_positive_kind="product_only_readiness",
    )
    bounded = mod.classify_candidate(
        candidate_id="bounded_method",
        source_ids=["paper"],
        proposed_action="test",
        primary_document_opened=True,
        primary_method_verified=True,
        reproducible_implementation_identity=False,
        bounded_local_test=True,
        evidence_kind="primary_paper",
        reason="Run a sealed local falsification.",
    )

    assert build["classification"] == "build"
    assert no_code["classification"] == "watch"
    assert abstract["classification"] == "reject"
    assert product["classification"] == "reject"
    assert bounded["classification"] == "test"
    assert all(
        row["implementation_readiness_claimed"] is False
        for row in (build, no_code, abstract, product, bounded)
    )
    with pytest.raises(ValueError, match="unknown classification"):
        mod.classify_candidate(
            candidate_id="bad",
            source_ids=[],
            proposed_action="promote",
            primary_document_opened=False,
            primary_method_verified=False,
            reproducible_implementation_identity=False,
            bounded_local_test=False,
            evidence_kind="search_result",
            reason="Invalid class.",
        )


def test_scenario_report_7029_source_families_are_terminal() -> None:
    """SCENARIO-REPORT-7029-RECEIPTS covers all requested source families."""

    groups = mod.source_row_groups()
    combined = [row for rows in groups.values() for row in rows]

    assert set(groups) == set(mod.REQUIRED_SOURCE_FAMILIES)
    assert all(groups.values())
    assert all(row["terminal"] is True for row in combined)
    assert all(str(row["url"]).startswith("https://") for row in combined)
    assert all(row["accessed_on"] == "2026-09-05" for row in combined)
    assert {row["topic"] for row in mod.source_query_rows()} == set(mod.ARXIV_TOPICS)
    assert all(row["primary_document_opened"] is True for row in mod.primary_source_rows())
    assert {row["paper_id"] for row in mod.semantic_scholar_rows()} == {
        "2507.02092",
        "2512.15605",
    }
    assert all(row["citation_count_claimed"] is False for row in mod.semantic_scholar_rows())
    assert all(row["hardware_readiness_claimed"] is False for row in mod.extropic_rows())
    assert all(
        row["reproducible_local_runner_claimed"] is False for row in mod.logical_intelligence_rows()
    )


def test_scenario_report_7029_scope_audit_preserves_tasks_and_retirements() -> None:
    """SCENARIO-REPORT-7029-SCOPE keeps ten tasks and five closed mechanisms."""

    rows = mod.scope_audit_rows()
    task_row = next(row for row in rows if row["scope_id"] == "v616_task_contract")

    assert task_row["expected_task_count"] == 10
    assert task_row["observed_task_count"] == 10
    assert task_row["expected_id_order"] == list(mod.V616_TASK_ORDER)
    assert task_row["observed_id_order"] == list(mod.V616_TASK_ORDER)
    assert task_row["scope_expanded"] is False
    retired = {row["scope_id"]: row for row in rows if row.get("retired_technique")}
    assert set(retired) == set(mod.RETIRED_TECHNIQUES)
    assert all(
        row["reopened"] is False and row["disposition"] == "reject" for row in retired.values()
    )


def test_scenario_report_7029_append_requires_later_primary_and_scope() -> None:
    """SCENARIO-REPORT-7029-APPEND permits only scoped post-marker primary evidence."""

    base = {
        "candidate_id": "candidate",
        "source_id": "paper",
        "source_kind": "primary_paper",
        "primary_document_opened": True,
        "primary_method_verified": True,
        "cutoff_relation": "post_marker",
        "canonical_url": "https://arxiv.org/abs/2609.99999",
        "finding": "A bounded method changed.",
        "scope_disposition": "watch_without_task_change",
    }
    assert mod.reference_append_rows([])[0]["action"] == "no_change"
    assert (
        mod.reference_append_rows([base | {"cutoff_relation": "same_day_order_uncertain"}])[0][
            "action"
        ]
        == "no_change"
    )
    assert mod.reference_append_rows([base | {"scope_disposition": ""}])[0]["action"] == "no_change"
    rows = mod.reference_append_rows([base])
    assert rows[0]["action"] == "append"
    assert rows[0]["scope_disposition"] == "watch_without_task_change"


def test_scenario_report_7029_preflight_blocks_with_complete_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7029-PREFLIGHT returns one exact blocked artifact."""

    root = _fake_root(tmp_path / "missing", marker=False)
    artifact = mod.build_artifact(
        root,
        "20260905",
        output_path=tmp_path / "blocked.json",
        network_is_available=True,
        duration_s=0.1,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["v616_sota_scope_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_v616_sota_scope_audit"
    assert artifact["gate_check_summary"] == {
        "failed_check": "v616_reference_marker",
        "expected_value": mod.REFERENCE_MARKER,
        "observed_value": "missing",
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7029_all_source_families_unavailable_is_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7029-PREFLIGHT blocks when all routes lack usable content."""

    unavailable = {
        family: [
            mod.access_receipt(
                family=family,
                query=f"{family} route",
                url=f"https://example.com/{family}",
                accessed_on="2026-09-05",
                http_status=429,
                access_outcome="rate_limited",
                evidence_kind="secondary_index",
            )
        ]
        for family in mod.REQUIRED_SOURCE_FAMILIES
    }
    artifact = _artifact(tmp_path, source_groups=unavailable)

    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "all_source_families_unavailable"
    assert artifact["v616_sota_scope_complete_score"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7029_artifact_recomputes_contract(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7029-ARTIFACT rejects forged scores, scope, and readiness."""

    artifact = _artifact(tmp_path)

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["v616_sota_scope_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert "no_scientific_improvement" in str(artifact["honest_verdict"])
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    mutations = {
        "field_principles": {},
        "source_query_rows": [],
        "classification_rows": artifact["classification_rows"][:-1],
        "scope_audit_rows": artifact["scope_audit_rows"][:-1],
        "cutoff_rate_limit_and_same_day_uncertainty_receipts": [],
        "v616_sota_scope_complete_score": 0,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_unfinished",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert mod.validate_artifact(_refresh_checksum(changed)), field

    expanded = deepcopy(artifact)
    expanded["scope_audit_rows"][0]["observed_task_count"] = 11
    expanded["scope_audit_rows"][0]["scope_expanded"] = True
    assert "scope_audit_invalid" in mod.validate_artifact(_refresh_checksum(expanded))

    readiness = deepcopy(artifact)
    readiness["github_rows"][0]["implementation_readiness_claimed"] = True
    assert "unsupported_readiness_claim" in mod.validate_artifact(_refresh_checksum(readiness))

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_fields:rows"]
    changed_hash = deepcopy(artifact)
    changed_hash["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed_hash)


def test_req_report_7029_hashes_and_zero_delta_leave_reference_stable(tmp_path: Path) -> None:
    """REQ-REPORT-7029 binds inputs and keeps the ledger stable for zero delta."""

    root = _fake_root(tmp_path / "root")
    before = (root / mod.REFERENCE_PATH).read_bytes()
    hashes = mod.source_artifact_hashes(root)
    artifact = mod.build_artifact(
        root,
        "20260905",
        output_path=tmp_path / "artifact.json",
        network_is_available=True,
        duration_s=0.2,
    )

    assert {row["path"] for row in hashes} == {
        path.as_posix() for path in mod.REQUIRED_LOCAL_INPUTS
    }
    assert all(str(row["sha256"]).startswith("sha256:") for row in hashes)
    assert artifact["post_marker_delta_rows"] == []
    assert artifact["reference_append_rows"][0]["action"] == "no_change"
    assert (root / mod.REFERENCE_PATH).read_bytes() == before


def test_req_report_7029_writer_validator_and_cli(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-7029 writes, reloads, validates, and rejects a bad date."""

    artifact = _artifact(tmp_path)
    path = tmp_path / "nested" / "artifact.json"

    assert mod.write_artifact(artifact, path) == path
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(path) == []
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]

    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    writes: list[Path] = []
    monkeypatch.setattr(
        mod, "write_artifact", lambda _artifact, target: writes.append(target) or target
    )
    assert mod.main(["--date", "20260905", "--output", str(path)]) == 0
    assert writes == [path]
    assert mod.main(["--date", "bad", "--output", str(path)]) == 2
    assert mod.main(["--validate", str(path)]) == 0


def test_req_report_7029_failure_and_malformed_paths(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-7029 preserves network, file, and validation failures."""

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: Response())
    assert mod.network_available() is True
    monkeypatch.setattr(
        mod,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(mod.HTTPError("x", 429, "rate", {}, None)),
    )
    assert mod.network_available() is True
    monkeypatch.setattr(
        mod,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(mod.URLError("offline")),
    )
    assert mod.network_available() is False

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod.writable_path(tmp_path / "artifact.json") is False
    monkeypatch.undo()
    assert mod.writable_path(tmp_path / "not-created" / "artifact.json") is True

    class BrokenFile:
        def is_file(self) -> bool:
            return True

        def read_bytes(self) -> bytes:
            raise OSError("unreadable")

    assert mod._readable_nonempty(BrokenFile()) is False  # type: ignore[arg-type]

    root = _fake_root(tmp_path / "marker-io")
    original_read_text = Path.read_text

    def fail_marker(path: Path, *args: object, **kwargs: object) -> str:
        if path == root / mod.REFERENCE_PATH:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_marker)
    checks = mod.check_preconditions(
        root,
        tmp_path / "marker-io.json",
        network_is_available=False,
    )
    assert next(row for row in checks if row["check"] == "v616_reference_marker")["passed"] is False
    monkeypatch.undo()

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.validate_artifact(malformed) == ["artifact_unreadable"]
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    assert mod.validate_artifact(not_object) == ["artifact_not_object"]
    with pytest.raises(ValueError, match="missing_required_fields"):
        mod.write_artifact({}, tmp_path / "invalid.json")

    assert mod.main(["--validate", str(tmp_path / "missing.json")]) == 1
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["bad_artifact"])
    assert mod.main(["--date", "20260905", "--output", str(tmp_path / "bad-out.json")]) == 1


def test_req_report_7029_validator_rejects_duplicate_leaks_and_bad_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7029-ARTIFACT rejects duplicate, append, and gate defects."""

    artifact = _artifact(tmp_path)
    duplicate = deepcopy(artifact)
    duplicate["primary_source_rows"] = [
        *duplicate["primary_source_rows"],
        deepcopy(duplicate["primary_source_rows"][0]),
    ]
    assert "unsuppressed_duplicate_source_identity" in mod.validate_artifact(
        _refresh_checksum(duplicate)
    )

    forged_append = deepcopy(artifact)
    forged_append["reference_append_rows"] = [
        {
            "action": "append",
            "verified_source_id": "snippet",
            "cutoff_relation": "same_day_order_uncertain",
            "primary_method_verified": False,
            "scope_disposition": "expand_tasks",
            "appended": True,
            "terminal": True,
        }
    ]
    assert "reference_append_invalid" in mod.validate_artifact(_refresh_checksum(forged_append))

    assert (
        mod._reference_appends_valid({"reference_append_rows": [], "post_marker_delta_rows": []})
        is False
    )
    assert (
        mod._reference_appends_valid(
            {"reference_append_rows": ["bad"], "post_marker_delta_rows": []}
        )
        is False
    )
    eligible = {
        "source_id": "new",
        "source_kind": "primary_paper",
        "primary_document_opened": True,
        "primary_method_verified": True,
        "cutoff_relation": "post_marker",
        "canonical_url": "https://arxiv.org/abs/2609.99999",
        "scope_disposition": "watch_without_task_change",
    }
    assert (
        mod._reference_appends_valid(
            {
                "reference_append_rows": [
                    {
                        "action": "append",
                        "verified_source_id": "new",
                        "primary_method_verified": True,
                        "cutoff_relation": "post_marker",
                        "scope_disposition": "watch_without_task_change",
                    }
                ],
                "post_marker_delta_rows": [eligible],
            }
        )
        is True
    )
    assert (
        mod._reference_appends_valid(
            {
                "reference_append_rows": [{"action": "no_change"}],
                "post_marker_delta_rows": [eligible],
            }
        )
        is False
    )
    assert mod._classifications_valid({"classification_rows": ()}) is False

    bad_gate = deepcopy(artifact)
    bad_gate["gate_check_summary"]["passed"] = False
    assert "gate_check_summary_inconsistent" in mod.validate_artifact(_refresh_checksum(bad_gate))

    bad_shapes = (
        ("duration_s", -1, "duration_s_invalid"),
        ("source_artifact_hashes", [], "source_artifact_hashes_invalid"),
        ("primary_source_rows", (), "primary_source_rows_invalid"),
        ("verdict_class", "alien", "verdict_class_invalid"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
    )
    for field, value, expected in bad_shapes:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in mod.validate_artifact(_refresh_checksum(changed))

    empty_principle = deepcopy(artifact)
    empty_principle["field_principles"]["rows"] = ""
    assert "field_principles_empty" in mod.validate_artifact(_refresh_checksum(empty_principle))

    false_primary = deepcopy(artifact)
    false_primary["primary_source_rows"][0]["implementation_evidence_claimed"] = True
    assert "primary_source_boundary_invalid" in mod.validate_artifact(
        _refresh_checksum(false_primary)
    )

    groups = mod.source_row_groups()
    groups["logical_intelligence"] = []
    disqualified = _artifact(tmp_path / "disqualified", source_groups=groups)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"] == "v616_sota_scope_contract"
    assert {
        "source_family_coverage_invalid",
        "cutoff_receipts_invalid",
    } <= set(mod.validate_artifact(disqualified))
