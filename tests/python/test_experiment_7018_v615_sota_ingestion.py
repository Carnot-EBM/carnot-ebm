"""Tests for REQ-REPORT-7018 and its source-ingestion scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7018_v615_sota_ingestion as mod


ROOT = Path(__file__).resolve().parents[2]


def _fake_root(path: Path, *, marker: bool = True) -> Path:
    """Create only the local evidence that the preflight is allowed to trust."""

    for relative in mod.REQUIRED_LOCAL_INPUTS:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == mod.REFERENCE_PATH:
            text = f"{mod.REFERENCE_MARKER}\n" if marker else "## older marker\n"
        elif relative.suffix == ".json":
            text = '{"honest_verdict":"complete_positive_prior"}\n'
        else:
            text = "required evidence\n"
        target.write_text(text, encoding="utf-8")
    return path


def _artifact(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    """Build one complete fixture without making a test depend on the network."""

    root = _fake_root(tmp_path / "root")
    return mod.build_artifact(
        root,
        "20260905",
        output_path=tmp_path / "artifact.json",
        network_available=True,
        duration_s=0.25,
        **kwargs,
    )


def _refresh_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_scenario_report_7018_cutoff_requires_proved_order() -> None:
    """SCENARIO-REPORT-7018-CUTOFF rejects old, uncertain, and malformed dates."""

    assert mod.cutoff_relation("2026-09-04") == "pre_marker"
    assert mod.cutoff_relation("2026-09-05") == "same_day_order_uncertain"
    assert mod.cutoff_relation("2026-09-05T23:59:59Z") == "same_day_order_uncertain"
    assert mod.cutoff_relation("2026-09-06") == "post_marker"
    assert (
        mod.cutoff_relation(
            "2026-09-05T08:00:01Z",
            marker_timestamp="2026-09-05T08:00:00Z",
        )
        == "post_marker"
    )
    assert mod.cutoff_relation("not-a-date") == "unknown"


def test_scenario_report_7018_receipts_keep_rate_limits_terminal() -> None:
    """SCENARIO-REPORT-7018-RECEIPTS preserves access limits without claims."""

    row = mod.access_receipt(
        family="semantic_scholar",
        query="ARXIV:2507.02092 citations",
        url="https://api.semanticscholar.org/test",
        accessed_on="2026-09-05",
        http_status=429,
        access_outcome="rate_limited",
        evidence_kind="secondary_index",
    )
    assert row["terminal"] is True
    assert row["content_verified"] is False
    assert row["readiness_claimed"] is False
    assert row["http_status"] == 429
    with pytest.raises(ValueError, match="canonical https URL"):
        mod.access_receipt(
            family="arxiv",
            query="bad",
            url="snippet only",
            accessed_on="2026-09-05",
            http_status=None,
            access_outcome="search_snippet_only",
            evidence_kind="search_result",
        )


def test_scenario_report_7018_duplicates_suppress_identity_and_finding() -> None:
    """SCENARIO-REPORT-7018-DUPLICATES keeps one identity and one finding."""

    rows = [
        {"row_id": "a", "source_identity": "arXiv:2609.00455v1", "finding_id": "bbwm"},
        {"row_id": "b", "source_identity": "arXiv:2609.00455v1", "finding_id": "other"},
        {"row_id": "c", "source_identity": "arXiv:2606.11521v1", "finding_id": "bbwm"},
        {"row_id": "d", "source_identity": "arXiv:2502.00271v1", "finding_id": "control"},
    ]
    unique, duplicates = mod.deduplicate_findings(rows)

    assert [row["row_id"] for row in unique] == ["a", "d"]
    assert {row["reason"] for row in duplicates} == {
        "duplicate_source_identity",
        "duplicate_finding_identity",
    }
    assert all(row["suppressed"] is True for row in duplicates)


def test_scenario_report_7018_false_positives_and_readiness_fail_closed() -> None:
    """SCENARIO-REPORT-7018-CLASSIFY rejects snippets and product-only readiness."""

    snippet = mod.classify_finding(
        finding_id="snippet",
        source_ids=["search"],
        proposed_action="build",
        primary_verified=False,
        reproducible_code_identity=False,
        bounded_local_test=False,
        reason="A search snippet is not a primary source.",
        false_positive_kind="search_snippet",
    )
    hardware = mod.classify_finding(
        finding_id="z1_hardware_readiness",
        source_ids=["z1t"],
        proposed_action="build",
        primary_verified=True,
        reproducible_code_identity=True,
        bounded_local_test=False,
        reason="No authenticated Z1 or TSU runner exists.",
        false_positive_kind="unsupported_hardware_readiness",
    )
    bounded_test = mod.classify_finding(
        finding_id="counterexample_clusters",
        source_ids=["cegl"],
        proposed_action="test",
        primary_verified=True,
        reproducible_code_identity=False,
        bounded_local_test=True,
        reason="Test contradiction clusters on sealed later transitions.",
    )

    assert snippet["classification"] == "reject"
    assert hardware["classification"] == "reject"
    assert bounded_test["classification"] == "test"
    assert all(row["local_readiness_claimed"] is False for row in (snippet, hardware))
    assert (
        mod.classify_finding(
            finding_id="missing_code",
            source_ids=["paper"],
            proposed_action="build",
            primary_verified=True,
            reproducible_code_identity=False,
            bounded_local_test=True,
            reason="Code identity is missing.",
        )["classification"]
        == "watch"
    )
    assert (
        mod.classify_finding(
            finding_id="missing_test",
            source_ids=["paper"],
            proposed_action="test",
            primary_verified=True,
            reproducible_code_identity=False,
            bounded_local_test=False,
            reason="A bounded local test is missing.",
        )["classification"]
        == "watch"
    )
    with pytest.raises(ValueError, match="unknown classification"):
        mod.classify_finding(
            finding_id="bad",
            source_ids=[],
            proposed_action="promote",
            primary_verified=False,
            reproducible_code_identity=False,
            bounded_local_test=False,
            reason="Invalid action.",
        )


def test_scenario_report_7018_classifications_cover_selected_scope() -> None:
    """SCENARIO-REPORT-7018-CLASSIFY maps every selected V615 finding once."""

    rows = mod.classification_rows()
    by_id = {row["finding_id"]: row for row in rows}

    assert by_id["bbwm_queryable_belief"]["classification"] == "build"
    assert by_id["counterexample_guided_updates"]["classification"] == "test"
    assert by_id["compute_matched_verifier_controls"]["classification"] == "test"
    assert by_id["kan_forgetting_caution"]["classification"] == "watch"
    assert by_id["z1t_public_software"]["classification"] == "watch"
    assert by_id["kona_architecture_only"]["classification"] == "watch"
    assert {
        "retired_generated_text",
        "retired_grammar_decoding",
        "retired_pair_centered_latent",
        "unchanged_kan_scope",
        "z1_hardware_readiness",
    } <= {row["finding_id"] for row in rows if row["classification"] == "reject"}
    assert all(row["classification"] in mod.CLASSIFICATIONS for row in rows)


def test_scenario_report_7018_source_families_are_terminal_and_bounded() -> None:
    """SCENARIO-REPORT-7018-RECEIPTS covers every requested public route."""

    groups = mod.source_row_groups()
    combined = [row for rows in groups.values() for row in rows]

    assert set(groups) == set(mod.REQUIRED_SOURCE_FAMILIES)
    assert all(rows for rows in groups.values())
    assert all(row["terminal"] is True for row in combined)
    assert all(str(row["url"]).startswith("https://") for row in combined)
    assert all(row["accessed_on"] == "2026-09-05" for row in combined)
    assert {row["topic"] for row in mod.source_query_rows()} == set(mod.ARXIV_TOPICS)
    assert all(row["readiness_claimed"] is False for row in mod.secondary_source_rows())
    assert all(row["citation_count_claimed"] is False for row in mod.semantic_scholar_rows())
    assert all(row["hardware_readiness_claimed"] is False for row in mod.extropic_rows())
    assert all(
        row["reproducible_local_runner_claimed"] is False for row in mod.logical_intelligence_rows()
    )


def test_scenario_report_7018_append_requires_verified_post_marker_primary() -> None:
    """SCENARIO-REPORT-7018-APPEND permits only proved later primary evidence."""

    old = {
        "delta_id": "old",
        "source_id": "old",
        "source_kind": "primary_paper",
        "primary_verified": True,
        "cutoff_relation": "pre_marker",
        "canonical_url": "https://arxiv.org/abs/2502.00271",
        "finding": "Old finding.",
    }
    same_day = old | {"delta_id": "same", "cutoff_relation": "same_day_order_uncertain"}
    new = old | {"delta_id": "new", "cutoff_relation": "post_marker"}

    assert mod.reference_append_rows([])[0]["action"] == "no_change"
    assert mod.reference_append_rows([old, same_day])[0]["action"] == "no_change"
    rows = mod.reference_append_rows([new])
    assert rows[0]["action"] == "append"
    assert rows[0]["verified_source_id"] == "old"


def test_scenario_report_7018_preflight_blocks_with_complete_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7018-PREFLIGHT returns an exact blocked artifact."""

    root = _fake_root(tmp_path / "missing", marker=False)
    artifact = mod.build_artifact(
        root,
        "20260905",
        output_path=tmp_path / "blocked.json",
        network_available=True,
        duration_s=0.1,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["v615_sota_ingestion_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_v615_sota_ingestion"
    assert artifact["gate_check_summary"] == {
        "failed_check": "v615_reference_marker",
        "expected_value": mod.REFERENCE_MARKER,
        "observed_value": "missing",
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7018_all_routes_unavailable_is_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7018-PREFLIGHT blocks when every public route lacks content."""

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
    assert artifact["gate_check_summary"]["failed_check"] == "all_source_routes_unavailable"
    assert artifact["v615_sota_ingestion_complete_score"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7018_artifact_recomputes_contract_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7018-ARTIFACT rejects forged fields and promotions."""

    artifact = _artifact(tmp_path)

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["v615_sota_ingestion_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    mutations = {
        "field_principles": {},
        "source_query_rows": [],
        "classification_rows": artifact["classification_rows"][:-1],
        "cutoff_rate_limit_and_same_day_uncertainty_receipts": [],
        "v615_sota_ingestion_complete_score": 0,
        "inference_substrate": "llm_summary",
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_unfinished",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert mod.validate_artifact(_refresh_checksum(changed)), field

    forged_append = deepcopy(artifact)
    forged_append["reference_append_rows"] = [
        {
            "action": "append",
            "verified_source_id": "snippet",
            "cutoff_relation": "same_day_order_uncertain",
            "primary_verified": False,
            "appended": True,
        }
    ]
    assert "reference_append_without_verified_post_marker_primary" in mod.validate_artifact(
        _refresh_checksum(forged_append)
    )

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_fields:rows"]
    changed_hash = deepcopy(artifact)
    changed_hash["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed_hash)


def test_req_report_7018_source_hashes_and_reference_nochange_are_stable(tmp_path: Path) -> None:
    """REQ-REPORT-7018 binds local inputs and preserves the ledger without a delta."""

    root = _fake_root(tmp_path / "root")
    before = (root / mod.REFERENCE_PATH).read_bytes()
    hashes = mod.source_artifact_hashes(root)
    artifact = mod.build_artifact(
        root,
        "20260905",
        output_path=tmp_path / "result.json",
        network_available=True,
        duration_s=0.2,
    )

    assert {row["path"] for row in hashes} == {
        path.as_posix() for path in mod.REQUIRED_LOCAL_INPUTS
    }
    assert all(str(row["sha256"]).startswith("sha256:") for row in hashes)
    assert artifact["reference_append_rows"][0]["action"] == "no_change"
    assert (root / mod.REFERENCE_PATH).read_bytes() == before


def test_req_report_7018_writer_validator_and_cli(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-7018 writes, reloads, validates, and rejects a bad date."""

    artifact = _artifact(tmp_path)
    path = tmp_path / "nested" / "artifact.json"

    assert mod.write_artifact(artifact, path) == path
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(path) == []
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]

    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    writes: list[Path] = []
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda _artifact, target: writes.append(target) or target,
    )
    assert mod.main(["--date", "20260905", "--output", str(path)]) == 0
    assert writes == [path]
    assert mod.main(["--date", "bad", "--output", str(path)]) == 2
    assert mod.main(["--validate", str(path)]) == 0


def test_req_report_7018_io_and_network_failure_paths(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-7018 turns local probe failures into explicit preconditions."""

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

    class BrokenFile:
        def is_file(self) -> bool:
            return True

        def read_bytes(self) -> bytes:
            raise OSError("unreadable")

    assert mod._readable_nonempty(BrokenFile()) is False  # type: ignore[arg-type]

    nested = tmp_path / "not-created" / "deeper" / "artifact.json"
    monkeypatch.undo()
    assert mod.writable_path(nested) is True

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
    assert next(row for row in checks if row["check"] == "v615_reference_marker")["passed"] is False
    monkeypatch.undo()

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.validate_artifact(malformed) == ["artifact_unreadable"]
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    assert mod.validate_artifact(not_object) == ["artifact_not_object"]
    with pytest.raises(ValueError, match="missing_required_fields"):
        mod.write_artifact({}, tmp_path / "invalid.json")


def test_req_report_7018_validator_catches_family_rows_and_duplicate_leaks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7018-ARTIFACT catches missing families and unsuppressed duplicates."""

    artifact = _artifact(tmp_path)
    missing_family = deepcopy(artifact)
    missing_family["github_rows"] = []
    assert "source_family_coverage_invalid" in mod.validate_artifact(
        _refresh_checksum(missing_family)
    )

    duplicate = deepcopy(artifact)
    duplicate["primary_source_rows"] = [
        *duplicate["primary_source_rows"],
        deepcopy(duplicate["primary_source_rows"][0]),
    ]
    assert "unsuppressed_duplicate_source_identity" in mod.validate_artifact(
        _refresh_checksum(duplicate)
    )

    empty_principle = deepcopy(artifact)
    empty_principle["field_principles"]["rows"] = ""
    assert "field_principles_empty" in mod.validate_artifact(_refresh_checksum(empty_principle))


def test_req_report_7018_validator_covers_malformed_terminal_shapes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7018-ARTIFACT rejects malformed terminal row shapes."""

    artifact = _artifact(tmp_path)
    assert mod.completion_score(artifact | {"classification_rows": ()}) == 0
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
        "primary_verified": True,
        "cutoff_relation": "post_marker",
        "canonical_url": "https://arxiv.org/abs/2609.99999",
    }
    assert (
        mod._reference_appends_valid(
            {
                "reference_append_rows": [
                    {"action": "no_change", "terminal": True},
                ],
                "post_marker_delta_rows": [eligible],
            }
        )
        is False
    )

    mutations = (
        ("duration_s", -1, "duration_s_invalid"),
        ("source_artifact_hashes", [], "source_artifact_hashes_invalid"),
        ("primary_source_rows", (), "primary_source_rows_invalid"),
        ("verdict_class", "alien", "verdict_class_invalid"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in mod.validate_artifact(_refresh_checksum(changed))

    positive_bad_gate = deepcopy(artifact)
    positive_bad_gate["gate_check_summary"]["passed"] = False
    assert "gate_check_summary_inconsistent" in mod.validate_artifact(
        _refresh_checksum(positive_bad_gate)
    )

    blocked = _artifact(
        tmp_path / "blocked",
        source_groups={
            family: [
                mod.access_receipt(
                    family=family,
                    query=family,
                    url=f"https://example.com/{family}",
                    accessed_on="2026-09-05",
                    http_status=429,
                    access_outcome="rate_limited",
                    evidence_kind="secondary_index",
                )
            ]
            for family in mod.REQUIRED_SOURCE_FAMILIES
        },
    )
    blocked["gate_check_summary"]["passed"] = True
    assert "gate_check_summary_inconsistent" in mod.validate_artifact(_refresh_checksum(blocked))


def test_req_report_7018_disqualified_and_cli_error_branches(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-7018 reports contract defects and CLI validation failures."""

    groups = mod.source_row_groups()
    groups["logical_intelligence"] = []
    disqualified = _artifact(tmp_path, source_groups=groups)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"] == "source_ingestion_contract"

    assert mod.main(["--validate", str(tmp_path / "missing.json")]) == 1
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["bad_artifact"])
    assert mod.main(["--date", "20260905", "--output", str(tmp_path / "bad.json")]) == 1
