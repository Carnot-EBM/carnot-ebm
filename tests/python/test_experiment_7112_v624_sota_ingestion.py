"""Tests for REQ-REPORT-7112 and SCENARIO-REPORT-7112-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7112_v624_sota_ingestion as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _root(tmp_path: Path, *, marker: bool = True) -> Path:
    """Create private evidence paths so tests cannot change the research record."""

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
        "semantic_scholar": False,
        "huggingface_papers": True,
    }
    routes.update(changes)
    return routes


def _verified_delta() -> dict[str, object]:
    return {
        "candidate_id": "arxiv-2609-05638",
        "source_id": "arxiv:2609.05638",
        "title": "Test-Time Training via Energy-Guided Diffusion for Reasoning",
        "source_url": "https://arxiv.org/abs/2609.05638",
        "publication_or_revision_date": "2026-09-07",
        "publication_or_revision_time": "2026-09-07T12:00:00Z",
        "first_observed_at": "2026-09-07T12:05:00Z",
        "source_receipt_id": "receipt-arxiv-2609.05638",
        "source_role": "primary_paper",
        "identity_verified": True,
        "core_claim_verified": True,
        "classification": "control",
        "decision_relevant": True,
        "experiment_hook": "exp7117-exact-verify-revise-loop",
        "claim_boundary": "The paper supplies a decoding control, not proof of Carnot correctness.",
        "classification_reason": "The verified method changes one bounded V624 control cell.",
        "terminal": True,
    }


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


def test_req_report_7112_spec_precedes_implementation() -> None:
    """REQ-REPORT-7112 owns every required field and named scenario."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7112") :]
    for scenario in (
        "PREFLIGHT",
        "IDENTITY",
        "DEDUPLICATION",
        "VENDOR",
        "EMPTY-DELTA",
        "APPEND",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7112-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7112_identity_validates_urls_and_time_window() -> None:
    """SCENARIO-REPORT-7112-IDENTITY rejects bad routes and time edges."""

    assert exp.identifier_matches_url("arxiv:2609.05638", "https://arxiv.org/abs/2609.05638")
    assert exp.identifier_matches_url(
        "openreview:oum1txoy1D", "https://openreview.net/forum?id=oum1txoy1D"
    )
    assert exp.identifier_matches_url("github:alexiglad/EBT", "https://github.com/alexiglad/EBT")
    assert exp.identifier_matches_url(
        "hf-paper:2609.05638", "https://huggingface.co/papers/2609.05638"
    )
    assert exp.identifier_matches_url("extropic:z1t-2026-09-04", "https://extropic.ai/writing/z1t")
    assert exp.identifier_matches_url(
        "logical-intelligence:kona-1.0",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
    )
    assert not exp.identifier_matches_url(
        "arxiv:2609.05638", "https://huggingface.co/papers/2609.05638"
    )
    assert not exp.identifier_matches_url("unknown:item", "https://example.com/item")

    window = exp.search_window(exp.RUN_DATE)
    assert not exp.in_execution_delta("2026-09-07T10:28:40Z", window)
    assert exp.in_execution_delta("2026-09-07T10:28:41Z", window)
    assert exp.in_execution_delta("2026-09-07T23:59:59Z", window)
    assert not exp.in_execution_delta("2026-09-08T00:00:00Z", window)
    assert not exp.in_execution_delta("bad-time", window)


def test_scenario_report_7112_empty_delta_is_positive_and_byte_stable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7112-EMPTY-DELTA permits a complete empty delta."""

    root = _root(tmp_path)
    references = root / exp.REFERENCE_PATH
    before = references.read_bytes()
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        route_reachability=_routes(),
        candidates=[],
        duration_s=0.5,
        update_references=True,
    )
    assert artifact["run_date"] == exp.RUN_DATE
    assert artifact["v624_sota_ingestion_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert artifact["adoption_rows"] == []
    assert references.read_bytes() == before
    assert artifact["references_append_hash"] == exp.file_sha256(references)
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7112_append_markers_are_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7112-APPEND writes one complete marker pair."""

    references = _root(tmp_path) / exp.REFERENCE_PATH
    candidate = _verified_delta()
    assert exp.append_references_delta(references, [candidate], exp.RUN_DATE)
    once = references.read_text(encoding="utf-8")
    assert once.count(exp.REFERENCE_START_MARKER) == 1
    assert once.count(exp.REFERENCE_END_MARKER) == 1
    assert str(candidate["source_id"]) in once
    assert str(candidate["experiment_hook"]) in once
    assert not exp.append_references_delta(references, [candidate], exp.RUN_DATE)
    assert references.read_text(encoding="utf-8") == once
    assert not exp.append_references_delta(references, [], exp.RUN_DATE)


def test_scenario_report_7112_deduplicates_and_prefers_primary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7112-DEDUPLICATION suppresses known and secondary rows."""

    artifact = _valid_artifact(tmp_path)
    duplicate_ids = {
        row["candidate_id"]
        for row in artifact["candidate_rows"]
        if row["classification"] == "duplicate"
    }
    assert duplicate_ids == {row["candidate_id"] for row in artifact["deduplication_rows"]}

    forged = deepcopy(artifact)
    duplicate = next(row for row in forged["candidate_rows"] if row["classification"] == "duplicate")
    duplicate.update(
        classification="adopt",
        decision_relevant=True,
        experiment_hook="exp7117-exact-verify-revise-loop",
    )
    assert exp.completion_score(forged) == 0

    secondary = deepcopy(artifact)
    secondary["candidate_rows"].append({**_verified_delta(), "source_role": "secondary_index"})
    assert exp.completion_score(secondary) == 0


def test_scenario_report_7112_vendor_labels_block_vendor_claims(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7112-VENDOR keeps vendor material below evidence."""

    artifact = _valid_artifact(tmp_path)
    vendor_rows = [*artifact["extropic_rows"], *artifact["logical_intelligence_rows"]]
    assert vendor_rows
    assert all(row["claim_boundary_label"] == exp.VENDOR_BOUNDARY for row in vendor_rows)
    assert all(row["scientific_evidence_promoted"] is False for row in vendor_rows)

    forged = deepcopy(artifact)
    forged["extropic_rows"][0]["runtime_claimed"] = True
    assert exp.completion_score(forged) == 0

    forged_label = deepcopy(artifact)
    forged_label["logical_intelligence_rows"][0]["claim_boundary_label"] = "primary_evidence"
    assert exp.completion_score(forged_label) == 0


@pytest.mark.parametrize(
    ("routes", "marker", "failed_check"),
    [
        (_routes(arxiv=False), True, "arxiv_network_reachability"),
        (
            _routes(openreview=False, semantic_scholar=False, huggingface_papers=False),
            True,
            "secondary_index_reachability",
        ),
        (_routes(), False, "v624_planner_marker"),
    ],
)
def test_scenario_report_7112_preflight_blocks_exactly(
    tmp_path: Path,
    routes: dict[str, bool],
    marker: bool,
    failed_check: str,
) -> None:
    """SCENARIO-REPORT-7112-PREFLIGHT records the first failed gate."""

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
    assert artifact["v624_sota_ingestion_complete_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7112_artifact_recomputes_rows_score_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7112-ARTIFACT rejects forged result fields."""

    artifact = _valid_artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["rows"] == exp.combined_rows(artifact)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)

    for field, value in (
        ("rows", []),
        ("v624_sota_ingestion_complete_score", 0),
        ("field_principles", {}),
        ("run_date", "2026-09-07"),
        ("inference_substrate", "model inference"),
        ("inference_substrate_class", "aggregation"),
        ("execution_venue", "gpu"),
        ("duration_s", -1),
        ("random_seed", 0),
        ("verifier_is_oracle", True),
        ("verdict_class", "partial"),
        ("honest_verdict", "blocked_forged"),
        ("search_window", {}),
        ("gate_check_summary", {}),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    missing = deepcopy(artifact)
    missing.pop("query_rows")
    assert "artifact fields mismatch" in exp.validate_artifact(missing)[0]
    assert exp.validate_artifact([]) == ["artifact_not_object"]


def test_req_report_7112_io_and_command_line_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7112 covers storage, network checks, and command-line output."""

    root = _root(tmp_path)
    artifact = _valid_artifact(tmp_path / "valid")
    output = root / "results" / "custom.json"
    assert exp.write_artifact(artifact, output) == output
    assert exp.validate_artifact(output) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    class _Response:
        status = 200

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: _Response())
    assert exp.http_reachable("https://example.com")
    monkeypatch.setattr(exp, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    assert not exp.http_reachable("https://example.com")

    monkeypatch.setattr(exp, "find_repo_root", lambda: root)
    monkeypatch.setattr(exp, "probe_routes", lambda: _routes())
    monkeypatch.setattr(exp.time, "monotonic", iter((10.0, 11.5)).__next__)
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 0
    stdout = capsys.readouterr().out
    assert '"v624_sota_ingestion_complete_score": 1' in stdout
    assert '"verdict_class": "positive"' in stdout

    assert exp.main(["--date", "20260908", "--output", str(output)]) == 2
    assert exp.main(["--date", "bad", "--output", str(output)]) == 2
