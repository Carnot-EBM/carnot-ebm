"""Tests for REQ-REPORT-7098 and SCENARIO-REPORT-7098-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7098_v623_sota_ingestion as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _root(tmp_path: Path, *, marker: bool = True) -> Path:
    """Create the writable evidence boundary without touching tracked files."""

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
        "openreview": True,
        "semantic_scholar": False,
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
        "publication_or_revision_date": "2026-09-08",
        "source_receipt_id": "receipt-post-cutoff-control",
        "source_role": "primary_paper",
        "identity_verified": True,
        "core_claim_verified": True,
        "classification": "control",
        "decision_relevant": True,
        "experiment_hook": "exp7102-feasibility-projected-action-energy",
        "claim_boundary": "The paper defines a control; exact Carnot outcomes remain authority.",
        "classification_reason": "A verified later method changes an existing control cell.",
        "terminal": True,
    }


def test_req_report_7098_spec_precedes_implementation() -> None:
    """REQ-REPORT-7098 owns every required field and named scenario."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7098") :]
    for scenario in (
        "PREFLIGHT",
        "IDENTITY",
        "DEDUPLICATION",
        "VENDOR",
        "EMPTY-DELTA",
        "APPEND",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7098-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7098_identity_validates_urls_and_date_window() -> None:
    """SCENARIO-REPORT-7098-IDENTITY rejects bad routes and date edges."""

    assert exp.identifier_matches_url("arxiv:2609.00581", "https://arxiv.org/abs/2609.00581")
    assert exp.identifier_matches_url(
        "openreview:oum1txoy1D", "https://openreview.net/forum?id=oum1txoy1D"
    )
    assert exp.identifier_matches_url("github:alexiglad/EBT", "https://github.com/alexiglad/EBT")
    assert exp.identifier_matches_url(
        "hf-paper:2609.00581", "https://huggingface.co/papers/2609.00581"
    )
    assert exp.identifier_matches_url("extropic:z1t-2026-09-04", "https://extropic.ai/writing/z1t")
    assert exp.identifier_matches_url(
        "logical-intelligence:kona-1.0",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
    )
    assert not exp.identifier_matches_url(
        "arxiv:2609.00581", "https://huggingface.co/papers/2609.00581"
    )
    assert not exp.identifier_matches_url("unknown:item", "https://example.com/item")
    assert not exp.identifier_matches_url("arxiv:2609.00581", None)  # type: ignore[arg-type]

    window = exp.search_window(exp.RUN_DATE)
    assert not exp.in_execution_delta("2026-09-07", window)
    assert not exp.in_execution_delta("2026-09-08", window)
    later_window = exp.search_window("20260908")
    assert exp.in_execution_delta("2026-09-08", later_window)
    assert not exp.in_execution_delta("bad-date", later_window)


def test_scenario_report_7098_empty_delta_is_positive_and_byte_stable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7098-EMPTY-DELTA completes without a ledger rewrite."""

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
    assert artifact["v623_sota_ingestion_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert artifact["adoption_rows"] == []
    assert references.read_bytes() == before
    assert artifact["references_append_hash"] == exp.file_sha256(references)
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7098_append_markers_are_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7098-APPEND writes one complete marker pair."""

    references = _root(tmp_path) / exp.REFERENCE_PATH
    candidate = _promotable_candidate()
    assert exp.append_references_delta(references, [candidate], "20260908")
    once = references.read_text(encoding="utf-8")
    assert once.count(exp.REFERENCE_START_MARKER) == 1
    assert once.count(exp.REFERENCE_END_MARKER) == 1
    assert candidate["source_id"] in once
    assert candidate["experiment_hook"] in once
    assert not exp.append_references_delta(references, [candidate], "20260908")
    assert references.read_text(encoding="utf-8") == once
    assert not exp.append_references_delta(references, [], "20260908")


def test_scenario_report_7098_deduplicates_and_prefers_primary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7098-DEDUPLICATION suppresses known canonical IDs."""

    artifact = _valid_artifact(tmp_path)
    duplicate_ids = {
        row["candidate_id"]
        for row in artifact["candidate_rows"]
        if row["classification"] == "duplicate"
    }
    assert duplicate_ids == {row["candidate_id"] for row in artifact["deduplication_rows"]}

    forged = deepcopy(artifact)
    duplicate = next(
        row for row in forged["candidate_rows"] if row["classification"] == "duplicate"
    )
    duplicate.update(
        classification="adopt",
        decision_relevant=True,
        experiment_hook="exp7102-feasibility-projected-action-energy",
    )
    assert exp.completion_score(forged) == 0
    exp.recompute_artifact(forged)
    assert forged["verdict_class"] == "disqualified"
    assert exp.validate_artifact(forged) == []

    secondary = deepcopy(artifact)
    watched = next(row for row in secondary["candidate_rows"] if row["classification"] == "watch")
    watched.update(_promotable_candidate(), source_role="secondary_index")
    assert exp.completion_score(secondary) == 0
    exp.recompute_artifact(secondary)
    assert secondary["verdict_class"] == "disqualified"
    assert exp.validate_artifact(secondary) == []


def test_scenario_report_7098_vendor_labels_block_vendor_claims(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7098-VENDOR prevents vendor projections becoming evidence."""

    artifact = _valid_artifact(tmp_path)
    vendor_rows = [*artifact["extropic_rows"], *artifact["logical_intelligence_rows"]]
    assert vendor_rows
    assert all(row["claim_boundary_label"] == exp.VENDOR_BOUNDARY for row in vendor_rows)
    assert all(row["scientific_evidence_promoted"] is False for row in vendor_rows)

    forged = deepcopy(artifact)
    forged["extropic_rows"][0]["runtime_claimed"] = True
    assert exp.completion_score(forged) == 0
    exp.recompute_artifact(forged)
    assert forged["verdict_class"] == "disqualified"
    assert exp.validate_artifact(forged) == []

    forged_label = deepcopy(artifact)
    forged_label["logical_intelligence_rows"][0]["claim_boundary_label"] = "primary_evidence"
    assert exp.completion_score(forged_label) == 0
    exp.recompute_artifact(forged_label)
    assert forged_label["verdict_class"] == "disqualified"
    assert exp.validate_artifact(forged_label) == []


@pytest.mark.parametrize(
    ("routes", "marker", "failed_check"),
    [
        (_routes(arxiv=False), True, "arxiv_network_reachability"),
        (
            _routes(openreview=False, semantic_scholar=False, huggingface_papers=False),
            True,
            "secondary_index_reachability",
        ),
        (_routes(), False, "v623_planner_marker"),
    ],
)
def test_scenario_report_7098_preflight_blocks_exactly(
    tmp_path: Path,
    routes: dict[str, bool],
    marker: bool,
    failed_check: str,
) -> None:
    """SCENARIO-REPORT-7098-PREFLIGHT records the first failed gate."""

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
    assert str(artifact["honest_verdict"]).startswith("complete_blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["gate_check_summary"]["expected_value"] is not None
    assert artifact["gate_check_summary"]["observed_value"] is not None
    assert artifact["v623_sota_ingestion_complete_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7098_artifact_recomputes_rows_score_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7098-ARTIFACT rejects forged result fields."""

    artifact = _valid_artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["rows"] == exp.combined_rows(artifact)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)

    mutations = (
        ("rows", []),
        ("v623_sota_ingestion_complete_score", 0),
        ("field_principles", {}),
        ("inference_substrate", "model inference"),
        ("inference_substrate_class", "aggregation"),
        ("execution_venue", "gpu"),
        ("duration_s", -1),
        ("random_seed", 0),
        ("verifier_is_oracle", True),
        ("verdict_class", "partial"),
        ("verdict_class", "not-a-class"),
        ("honest_verdict", "blocked_forged"),
        ("search_window", {}),
        ("gate_check_summary", {}),
        ("reproducibility_checksum", "sha256:bad"),
    )
    for field, value in mutations:
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    missing = deepcopy(artifact)
    missing.pop("query_rows")
    assert "artifact fields mismatch" in exp.validate_artifact(missing)[0]
    assert exp.validate_artifact([]) == ["artifact_not_object"]

    positive_with_bad_coverage = deepcopy(artifact)
    positive_with_bad_coverage["source_class_rows"] = []
    positive_with_bad_coverage["rows"] = exp.combined_rows(positive_with_bad_coverage)
    positive_with_bad_coverage["reproducibility_checksum"] = exp.payload_checksum(
        positive_with_bad_coverage
    )
    assert "requested source class receipt coverage mismatch" in exp.validate_artifact(
        positive_with_bad_coverage
    )


def test_req_report_7098_rejects_each_evidence_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-7098 recomputes source, receipt, and discovery boundaries."""

    artifact = _valid_artifact(tmp_path)
    mutations: list[tuple[str, dict[str, object]]] = []

    for label, field in (
        ("source class", "source_class_rows"),
        ("primary receipt", "primary_source_receipts"),
        ("arxiv", "arxiv_rows"),
    ):
        forged = deepcopy(artifact)
        forged[field] = []
        mutations.append((label, forged))

    missing_family = deepcopy(artifact)
    missing_family["query_rows"] = [
        row
        for row in missing_family["query_rows"]
        if row.get("query_family") != exp.QUERY_FAMILIES[0]
    ]
    mutations.append(("query family", missing_family))

    bad_receipt = deepcopy(artifact)
    bad_receipt["primary_source_receipts"][0]["metadata_receipt_sha256"] = "sha256:bad"
    mutations.append(("receipt hash", bad_receipt))

    bad_candidate = deepcopy(artifact)
    bad_candidate["candidate_rows"][0]["claim_boundary"] = ""
    mutations.append(("claim boundary", bad_candidate))

    non_mapping_candidate = deepcopy(artifact)
    non_mapping_candidate["candidate_rows"].append("not-a-row")
    mutations.append(("non-mapping candidate", non_mapping_candidate))

    bad_github = deepcopy(artifact)
    bad_github["github_rows"][0]["stars_used_as_evidence"] = True
    mutations.append(("github", bad_github))

    bad_hf = deepcopy(artifact)
    bad_hf["huggingface_papers_rows"][0]["generated_summary_used_as_evidence"] = True
    mutations.append(("huggingface", bad_hf))

    bad_path = deepcopy(artifact)
    bad_path["references_append_path"] = "elsewhere.md"
    mutations.append(("append path", bad_path))

    for label, forged in mutations:
        assert exp.completion_score(forged) == 0, label


def test_disqualified_and_blocked_terminal_states_recompute(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7098-ARTIFACT accepts honest non-positive receipts."""

    root = _root(tmp_path)
    bad_candidates = exp.candidate_rows()
    bad_candidates[0]["classification"] = "adopt"
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        candidates=bad_candidates,
        duration_s=0.2,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert str(artifact["honest_verdict"]).startswith("complete_disqualified_")
    assert exp.validate_artifact(artifact) == []

    blocked = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(arxiv=False),
        duration_s=0.1,
    )
    for field, value in (
        ("gate_check_summary", {}),
        ("inference_substrate_class", "no_model_load"),
        ("v623_sota_ingestion_complete_score", 1),
        ("verdict_class", "positive"),
        ("honest_verdict", "complete_positive_forged"),
    ):
        forged = deepcopy(blocked)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    no_checks = deepcopy(artifact)
    no_checks["preconditions_checked"] = []
    assert "preconditions_checked missing" in exp.validate_artifact(no_checks)


def test_io_network_write_and_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7098 covers network, storage, and command-line paths."""

    root = _root(tmp_path)
    artifact = _valid_artifact(tmp_path / "valid")
    output = root / "results" / "custom.json"
    assert exp.write_artifact(artifact, output) == output
    assert exp.validate_artifact(output) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    relative_artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=exp.RESULT_PATH,
        route_reachability=_routes(),
        duration_s=0.2,
    )
    assert exp.validate_artifact(relative_artifact) == []

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

    missing = tmp_path / "missing.json"
    assert exp.validate_artifact(missing) == ["artifact_missing"]
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp.validate_artifact(malformed) == ["artifact_unreadable"]
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(array) == ["artifact_not_object"]

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", lambda: _routes())
    cli_output = root / "results" / "cli.json"
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(cli_output)]) == 0
    assert cli_output.is_file()
    assert "complete_positive_v623_sota_ingestion" in capsys.readouterr().out
    assert exp.main(["--validate", str(cli_output)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    assert exp.main(["--date", "bad"]) == 2

    bad = deepcopy(artifact)
    bad["random_seed"] = 0
    with pytest.raises(ValueError, match="random_seed"):
        exp.write_artifact(bad, tmp_path / "bad.json")


def test_precondition_io_failures_and_main_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7098 keeps I/O failures and invalid builds terminal."""

    root = _root(tmp_path)
    original_read_text = Path.read_text
    original_readable = exp._readable_nonempty

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
    marker = next(row for row in checks if row["check"] == "v623_planner_marker")
    assert marker["passed"] is False
    monkeypatch.setattr(exp, "_readable_nonempty", original_readable)

    class _Unreadable:
        def is_file(self) -> bool:
            raise OSError("unreadable")

    assert not exp._readable_nonempty(_Unreadable())  # type: ignore[arg-type]
    assert not exp._writable_target(tmp_path / "absent" / "file.json")
    monkeypatch.setattr(exp.tempfile, "NamedTemporaryFile", lambda **_kwargs: _raise_oserror())
    assert not exp._writable_target(tmp_path / "file.json")

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", lambda: _routes())
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: ["forced validation failure"])
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(root / exp.RESULT_PATH)]) == 1
    assert "forced validation failure" in capsys.readouterr().out


def _raise_oserror() -> object:
    """Raise the storage error used by the writable-path regression test."""

    raise OSError("not writable")
