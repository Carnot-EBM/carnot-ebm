"""Tests for REQ-REPORT-7092 and SCENARIO-REPORT-7092-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7092_v622_sota_ingestion as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _root(tmp_path: Path, *, marker: bool = True) -> Path:
    """Create only the local evidence boundary that the experiment may write."""

    references = tmp_path / exp.REFERENCE_PATH
    references.parent.mkdir(parents=True, exist_ok=True)
    text = "# References\n"
    if marker:
        text += f"\n{exp.PLANNER_HEADING}\n\n<!-- planner body -->\n"
    references.write_text(text, encoding="utf-8")
    (tmp_path / exp.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _routes(**changes: bool) -> dict[str, bool]:
    routes = {
        "arxiv": True,
        "openreview": False,
        "semantic_scholar": True,
        "huggingface_papers": True,
    }
    routes.update(changes)
    return routes


def _valid_artifact(tmp_path: Path) -> dict[str, object]:
    root = _root(tmp_path)
    return exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        duration_s=1.25,
        update_references=True,
    )


def _promotable_candidate() -> dict[str, object]:
    return {
        "candidate_id": "candidate-post-cutoff-control",
        "source_id": "arxiv:2609.99999",
        "title": "Bounded Post-Cutoff Control",
        "source_url": "https://arxiv.org/abs/2609.99999",
        "publication_or_revision_date": "2026-09-07",
        "source_receipt_id": "receipt-post-cutoff-control",
        "source_role": "primary_paper",
        "identity_verified": True,
        "core_claim_verified": True,
        "classification": "control",
        "decision_relevant": True,
        "experiment_hook": "exp7095-entrance-energy-matched-controls",
        "claim_boundary": "The paper defines a control; exact Carnot outcomes remain authority.",
        "classification_reason": "A verified post-cutoff method changes an existing control cell.",
        "terminal": True,
    }


def test_req_report_7092_spec_precedes_implementation() -> None:
    """REQ-REPORT-7092 owns preflight, identity, dedup, vendor, and append rules."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7092") :]
    for scenario in (
        "PREFLIGHT",
        "IDENTITY",
        "DEDUPLICATION",
        "VENDOR",
        "EMPTY-DELTA",
        "APPEND",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7092-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7092_identity_validates_urls_and_date_window() -> None:
    """SCENARIO-REPORT-7092-IDENTITY rejects mismatched IDs and window edges."""

    assert exp.identifier_matches_url("arxiv:2608.28128", "https://arxiv.org/abs/2608.28128")
    assert exp.identifier_matches_url(
        "openreview:MtKSNKnNzN", "https://openreview.net/forum?id=MtKSNKnNzN"
    )
    assert exp.identifier_matches_url("github:alexiglad/EBT", "https://github.com/alexiglad/EBT")
    assert exp.identifier_matches_url(
        "hf-paper:2608.28128", "https://huggingface.co/papers/2608.28128"
    )
    assert exp.identifier_matches_url(
        "logical-intelligence:kona-1.0",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
    )
    assert not exp.identifier_matches_url(
        "arxiv:2608.28128", "https://huggingface.co/papers/2608.28128"
    )
    assert not exp.identifier_matches_url("arxiv:2608.28128", "not-a-url")
    assert not exp.identifier_matches_url("unknown:identifier", "https://example.com/item")
    assert not exp.identifier_matches_url("arxiv:2608.28128", None)  # type: ignore[arg-type]

    window = exp.search_window(exp.RUN_DATE)
    assert exp.in_execution_delta("2026-09-07", window)
    assert not exp.in_execution_delta("2026-09-06", window)
    assert not exp.in_execution_delta("2026-09-08", window)
    assert not exp.in_execution_delta("bad-date", window)


def test_scenario_report_7092_deduplicates_and_prefers_primary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7092-DEDUPLICATION keeps indexes below primary sources."""

    artifact = _valid_artifact(tmp_path)
    assert exp.validate_artifact(artifact) == []
    candidates = artifact["candidate_rows"]
    duplicates = artifact["deduplication_rows"]
    assert isinstance(candidates, list)
    assert isinstance(duplicates, list)
    duplicate_ids = {
        row["candidate_id"] for row in candidates if row["classification"] == "duplicate"
    }
    assert duplicate_ids == {row["candidate_id"] for row in duplicates}

    forged = deepcopy(artifact)
    forged_candidate = next(
        row for row in forged["candidate_rows"] if row["classification"] == "duplicate"
    )
    forged_candidate["classification"] = "adopt"
    forged_candidate["decision_relevant"] = True
    forged_candidate["experiment_hook"] = "exp7095-entrance-energy-matched-controls"
    forged["adoption_rows"] = exp.adoption_rows(forged["candidate_rows"])
    forged["rows"] = exp.combined_rows(forged)
    forged["v622_sota_ingestion_complete_score"] = exp.completion_score(forged)
    forged["reproducibility_checksum"] = exp.payload_checksum(forged)
    errors = exp.validate_artifact(forged)
    assert any("duplicate" in error for error in errors)

    secondary = deepcopy(artifact)
    watched = next(row for row in secondary["candidate_rows"] if row["classification"] == "watch")
    watched.update(_promotable_candidate())
    watched["source_role"] = "secondary_index"
    secondary["adoption_rows"] = exp.adoption_rows(secondary["candidate_rows"])
    secondary["rows"] = exp.combined_rows(secondary)
    secondary["v622_sota_ingestion_complete_score"] = exp.completion_score(secondary)
    secondary["reproducibility_checksum"] = exp.payload_checksum(secondary)
    assert any("primary source" in error for error in exp.validate_artifact(secondary))


def test_scenario_report_7092_vendor_labels_block_vendor_claims(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7092-VENDOR keeps official product prose non-scientific."""

    artifact = _valid_artifact(tmp_path)
    vendor_rows = [*artifact["extropic_rows"], *artifact["logical_intelligence_rows"]]
    assert vendor_rows
    assert all(row["claim_boundary_label"] == exp.VENDOR_BOUNDARY for row in vendor_rows)
    assert all(row["scientific_evidence_promoted"] is False for row in vendor_rows)

    forged = deepcopy(artifact)
    forged["extropic_rows"][0]["runtime_claimed"] = True
    forged["rows"] = exp.combined_rows(forged)
    forged["v622_sota_ingestion_complete_score"] = exp.completion_score(forged)
    forged["reproducibility_checksum"] = exp.payload_checksum(forged)
    assert any("vendor" in error.lower() for error in exp.validate_artifact(forged))

    forged_label = deepcopy(artifact)
    forged_label["logical_intelligence_rows"][0]["claim_boundary_label"] = "primary_evidence"
    forged_label["rows"] = exp.combined_rows(forged_label)
    forged_label["v622_sota_ingestion_complete_score"] = exp.completion_score(forged_label)
    forged_label["reproducibility_checksum"] = exp.payload_checksum(forged_label)
    assert any("vendor boundary" in error for error in exp.validate_artifact(forged_label))


def test_scenario_report_7092_empty_delta_is_positive_and_byte_stable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7092-EMPTY-DELTA completes without rewriting references."""

    root = _root(tmp_path)
    references = root / exp.REFERENCE_PATH
    before = references.read_bytes()
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        duration_s=0.5,
        update_references=True,
    )
    assert artifact["v622_sota_ingestion_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert artifact["adoption_rows"] == []
    assert references.read_bytes() == before
    assert artifact["references_append_hash"] == exp.file_sha256(references)
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7092_append_markers_are_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7092-APPEND writes one complete marker pair at most once."""

    root = _root(tmp_path)
    references = root / exp.REFERENCE_PATH
    candidate = _promotable_candidate()
    assert exp.append_references_delta(references, [candidate], exp.RUN_DATE)
    once = references.read_text(encoding="utf-8")
    assert once.count(exp.REFERENCE_START_MARKER) == 1
    assert once.count(exp.REFERENCE_END_MARKER) == 1
    assert candidate["source_id"] in once
    assert candidate["experiment_hook"] in once
    assert not exp.append_references_delta(references, [candidate], exp.RUN_DATE)
    assert references.read_text(encoding="utf-8") == once
    assert not exp.append_references_delta(references, [], exp.RUN_DATE)


@pytest.mark.parametrize(
    ("routes", "marker", "failed_check"),
    [
        (_routes(arxiv=False), True, "arxiv_network_reachability"),
        (
            _routes(openreview=False, semantic_scholar=False, huggingface_papers=False),
            True,
            "secondary_index_reachability",
        ),
        (_routes(), False, "v622_planner_marker"),
    ],
)
def test_scenario_report_7092_preflight_blocks_exactly(
    tmp_path: Path,
    routes: dict[str, bool],
    marker: bool,
    failed_check: str,
) -> None:
    """SCENARIO-REPORT-7092-PREFLIGHT records the first failed execution gate."""

    root = _root(tmp_path, marker=marker)
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=routes,
        duration_s=0.1,
        update_references=True,
    )
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["gate_check_summary"]["expected_value"] is not None
    assert artifact["gate_check_summary"]["observed_value"] is not None
    assert artifact["v622_sota_ingestion_complete_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7092_artifact_recomputes_rows_score_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7092-ARTIFACT rejects forged terminal and row fields."""

    artifact = _valid_artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["rows"] == exp.combined_rows(artifact)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)

    mutations = (
        ("rows", []),
        ("v622_sota_ingestion_complete_score", 0),
        ("inference_substrate_class", "aggregation"),
        ("execution_venue", "gpu"),
        ("verifier_is_oracle", True),
        ("verdict_class", "partial"),
        ("honest_verdict", "blocked_forged"),
        ("reproducibility_checksum", "0" * 64),
    )
    for field, value in mutations:
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    missing = deepcopy(artifact)
    missing.pop("query_rows")
    assert "artifact fields mismatch" in exp.validate_artifact(missing)[0]
    assert exp.validate_artifact([]) == ["artifact_not_object"]

    scalar_mutations = (
        ("field_principles", {}),
        ("inference_substrate", "model inference"),
        ("inference_substrate_class", "aggregation"),
        ("duration_s", -1),
        ("verdict_class", "not-a-class"),
        ("search_window", {}),
    )
    for field, value in scalar_mutations:
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field


def test_coverage_rules_reject_each_evidence_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-7092 recomputes every source, identity, and promotion boundary."""

    artifact = _valid_artifact(tmp_path)

    mutations: list[tuple[str, object]] = []
    missing_class = deepcopy(artifact)
    missing_class["source_class_rows"] = []
    mutations.append(("source class", missing_class))

    missing_family = deepcopy(artifact)
    missing_family["query_rows"] = [
        row
        for row in missing_family["query_rows"]
        if row.get("query_family") != exp.QUERY_FAMILIES[0]
    ]
    mutations.append(("query family", missing_family))

    missing_route = deepcopy(artifact)
    missing_route["query_rows"] = [
        row for row in missing_route["query_rows"] if row.get("route") != "extropic"
    ]
    mutations.append(("query route", missing_route))

    missing_collection = deepcopy(artifact)
    missing_collection["arxiv_rows"] = []
    mutations.append(("row collection", missing_collection))

    nonterminal = deepcopy(artifact)
    nonterminal["arxiv_rows"][0]["terminal"] = False
    mutations.append(("nonterminal", nonterminal))

    duplicate_receipt = deepcopy(artifact)
    duplicate_receipt["primary_source_receipts"].append(
        deepcopy(duplicate_receipt["primary_source_receipts"][0])
    )
    mutations.append(("receipt id", duplicate_receipt))

    non_mapping_rows = deepcopy(artifact)
    non_mapping_rows["primary_source_receipts"].append("not-a-row")
    non_mapping_rows["candidate_rows"].append("not-a-row")
    non_mapping_rows["extropic_rows"].append("not-a-row")
    mutations.append(("non-mapping rows", non_mapping_rows))

    bad_receipt_url = deepcopy(artifact)
    bad_receipt_url["primary_source_receipts"][0]["source_url"] = "https://example.com/wrong"
    mutations.append(("receipt URL", bad_receipt_url))

    bad_receipt_hash = deepcopy(artifact)
    bad_receipt_hash["primary_source_receipts"][0]["metadata_receipt_sha256"] = "sha256:bad"
    mutations.append(("receipt hash", bad_receipt_hash))

    bad_receipt_title = deepcopy(artifact)
    bad_receipt_title["primary_source_receipts"][0]["title"] = ""
    mutations.append(("receipt title", bad_receipt_title))

    duplicate_candidate = deepcopy(artifact)
    duplicate_candidate["candidate_rows"].append(deepcopy(duplicate_candidate["candidate_rows"][0]))
    mutations.append(("candidate id", duplicate_candidate))

    bad_candidate_class = deepcopy(artifact)
    bad_candidate_class["candidate_rows"][0]["classification"] = "selected"
    mutations.append(("candidate class", bad_candidate_class))

    bad_candidate_receipt = deepcopy(artifact)
    bad_candidate_receipt["candidate_rows"][0]["title"] = "wrong title"
    mutations.append(("candidate receipt", bad_candidate_receipt))

    bad_candidate_url = deepcopy(artifact)
    bad_candidate_url["candidate_rows"][0]["source_url"] = "https://example.com/wrong"
    mutations.append(("candidate URL", bad_candidate_url))

    bad_boundary = deepcopy(artifact)
    bad_boundary["candidate_rows"][0]["claim_boundary"] = ""
    mutations.append(("claim boundary", bad_boundary))

    bad_promotion = deepcopy(artifact)
    bad_promotion["candidate_rows"][0].update(
        {
            "classification": "control",
            "source_role": "secondary_index",
            "identity_verified": False,
            "core_claim_verified": False,
            "publication_or_revision_date": "2026-09-07",
            "experiment_hook": "not-an-active-task",
            "decision_relevant": False,
        }
    )
    mutations.append(("promotion", bad_promotion))

    bad_adoption = deepcopy(artifact)
    bad_adoption["adoption_rows"] = [{"forged": True}]
    mutations.append(("adoption", bad_adoption))

    bad_github = deepcopy(artifact)
    bad_github["github_rows"][0]["stars_used_as_evidence"] = True
    mutations.append(("github", bad_github))

    bad_huggingface = deepcopy(artifact)
    bad_huggingface["huggingface_papers_rows"][0]["generated_summary_used_as_evidence"] = True
    mutations.append(("huggingface", bad_huggingface))

    bad_hash = deepcopy(artifact)
    bad_hash["references_append_hash"] = None
    mutations.append(("append hash", bad_hash))

    bad_path = deepcopy(artifact)
    bad_path["references_append_path"] = "elsewhere.md"
    mutations.append(("append path", bad_path))

    bad_games = deepcopy(artifact)
    bad_games["per_game_results"] = [{"game": "forged"}]
    mutations.append(("game rows", bad_games))

    for label, forged in mutations:
        assert exp.completion_score(forged) == 0, label


def test_disqualified_and_blocked_terminal_states_recompute(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7092-ARTIFACT accepts honest non-positive terminal receipts."""

    root = _root(tmp_path)
    bad_candidates = exp.candidate_rows()
    bad_candidates[0]["classification"] = "adopt"
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=exp.RESULT_PATH,
        route_reachability=_routes(),
        candidates=bad_candidates,
        duration_s=0.2,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert exp.validate_artifact(artifact) == []

    blocked = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(arxiv=False),
        duration_s=0.1,
    )
    blocked_mutations = (
        ("gate_check_summary", {}),
        ("inference_substrate_class", "no_model_load"),
        ("v622_sota_ingestion_complete_score", 1),
        ("verdict_class", "positive"),
        ("honest_verdict", "complete_positive_forged"),
    )
    for field, value in blocked_mutations:
        forged = deepcopy(blocked)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    no_checks = deepcopy(artifact)
    no_checks["preconditions_checked"] = []
    assert "preconditions_checked missing" in exp.validate_artifact(no_checks)


def test_write_validate_and_cli_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7092 validates atomic output and supports build and validate modes."""

    root = _root(tmp_path)
    artifact = _valid_artifact(tmp_path)
    output = root / "results" / "custom.json"
    assert exp.write_artifact(artifact, output) == output
    assert exp.validate_artifact(output) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", lambda: _routes())
    cli_output = root / "results" / "cli.json"
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(cli_output)]) == 0
    assert cli_output.is_file()
    assert "complete_positive_v622_sota_ingestion" in capsys.readouterr().out
    assert exp.main(["--validate", str(cli_output)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    assert exp.main(["--date", "bad"]) == 2


def test_helpers_fail_closed_on_io_network_and_invalid_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7092 covers network exceptions and malformed stored artifacts."""

    def _bad_url(_value: object) -> object:
        raise TypeError("invalid URL")

    monkeypatch.setattr(exp, "urlparse", _bad_url)
    assert not exp.identifier_matches_url("arxiv:2608.28128", "invalid")
    monkeypatch.undo()

    class _Response:
        status = 200

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: _Response())
    assert exp.http_reachable("https://arxiv.org/")

    from urllib.error import HTTPError

    def _http_error(*_args: object, **_kwargs: object) -> object:
        raise HTTPError("https://openreview.net/", 403, "challenge", {}, None)

    monkeypatch.setattr(exp, "urlopen", _http_error)
    assert exp.http_reachable("https://openreview.net/")

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("offline")

    monkeypatch.setattr(exp, "urlopen", _raise)
    assert not exp.http_reachable("https://arxiv.org/")
    monkeypatch.setattr(exp, "http_reachable", lambda url: "arxiv.org" in url)
    assert exp.probe_routes() == {
        "arxiv": True,
        "openreview": False,
        "semantic_scholar": False,
        "huggingface_papers": False,
    }

    class _Unreadable:
        def is_file(self) -> bool:
            raise OSError("unreadable")

    assert not exp._readable_nonempty(_Unreadable())  # type: ignore[arg-type]
    assert not exp._writable_target(tmp_path / "absent" / "file.json")

    monkeypatch.setattr(exp.tempfile, "NamedTemporaryFile", _raise)
    assert not exp._writable_target(tmp_path / "file.json")

    missing = tmp_path / "missing.json"
    assert exp.validate_artifact(missing) == ["artifact_missing"]
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp.validate_artifact(malformed) == ["artifact_unreadable"]
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(array) == ["artifact_not_object"]

    artifact = _valid_artifact(tmp_path / "valid")
    artifact["random_seed"] = 0
    with pytest.raises(ValueError, match="random_seed"):
        exp.write_artifact(artifact, tmp_path / "bad.json")


def test_precondition_read_error_and_main_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7092 keeps read failures and internal validation failures terminal."""

    root = _root(tmp_path)
    original_read_text = Path.read_text

    def _read_error(path: Path, *args: object, **kwargs: object) -> str:
        if path == root / exp.REFERENCE_PATH:
            raise OSError("unreadable after stat")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(exp, "_readable_nonempty", lambda _path: True)
    monkeypatch.setattr(Path, "read_text", _read_error)
    checks = exp.check_preconditions(
        root,
        root / exp.RESULT_PATH,
        route_reachability=_routes(),
    )
    assert next(row for row in checks if row["check"] == "v622_planner_marker")["passed"] is False

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", lambda: _routes())
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: ["forced validation failure"])
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(root / exp.RESULT_PATH)]) == 1
    assert "forced validation failure" in capsys.readouterr().out
