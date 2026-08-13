"""Tests for Exp6364 V548 source and scope freeze.

Spec refs: REQ-REPORT-6364, SCENARIO-REPORT-6364-1,
SCENARIO-REPORT-6364-2, SCENARIO-REPORT-6364-3,
SCENARIO-REPORT-6364-4, SCENARIO-REPORT-6364-5,
SCENARIO-REPORT-6364-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6364_v548_post_marker_source_scope_freeze as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _atom_feed() -> str:
    entries = []
    for paper in mod.DIRECT_ARXIV_PAPERS:
        entries.append(
            "<entry>"
            f"<id>https://arxiv.org/abs/{paper['arxiv_id']}v1</id>"
            f"<published>{paper['planner_submitted_date']}T12:00:00Z</published>"
            f"<title>{paper['title']}</title>"
            "</entry>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join(entries) + "</feed>"
    )


def _fake_fetcher(url: str) -> mod.JsonDict:
    if "export.arxiv.org" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"x-arxiv-test": "ok"},
            "body": _atom_feed(),
            "error": None,
        }
    if "openreview" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {},
            "body": json.dumps(
                {
                    "notes": [
                        {
                            "id": "DyDTtBUBEd",
                            "content": {
                                "title": {
                                    "value": (
                                        "Learning the Inverse Temperature of Ising Models "
                                        "under Hard Constraints Using One Sample"
                                    )
                                }
                            },
                        }
                    ]
                }
            ),
            "error": None,
        }
    if "huggingface.co" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {},
            "body": json.dumps([{"paper": {"id": "2607.05391", "title": "LLM-as-a-Verifier"}}]),
            "error": None,
        }
    if "2507.02092" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"x-ratelimit-remaining": "99"},
            "body": json.dumps(
                {
                    "total": 33,
                    "data": [
                        {
                            "citingPaper": {
                                "title": "Explorative Modeling",
                                "publicationDate": "2026-08-01",
                                "url": "https://www.semanticscholar.org/paper/ebt",
                            }
                        }
                    ],
                }
            ),
            "error": None,
        }
    if "2512.15605" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"x-ratelimit-remaining": "98"},
            "body": json.dumps(
                {
                    "total": 8,
                    "data": [
                        {
                            "citingPaper": {
                                "title": "Distributional Energy-Based Models",
                                "publicationDate": "2026-07-02",
                                "url": "https://www.semanticscholar.org/paper/arm",
                            }
                        }
                    ],
                }
            ),
            "error": None,
        }
    if "api.github.com" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"x-ratelimit-remaining": "57", "x-ratelimit-reset": "1786589999"},
            "body": json.dumps({"total_count": 0, "incomplete_results": False, "items": []}),
            "error": None,
        }
    if "extropic.ai" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"x-nextjs-date": "2026-08-11T07:12:45Z"},
            "body": "TSU XTR-0 DTM thrml",
            "error": None,
        }
    if "logicalintelligence.com" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"last-modified": "Fri, 26 Jun 2026 23:48:05 GMT"},
            "body": "Kona globally scored reasoning traces",
            "error": None,
        }
    raise AssertionError(f"unexpected URL {url}")


def test_req_report_6364_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-REPORT-6364: OpenSpec records the V548 source-freeze contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6364") : text.index("REQ-REPORT-5335")]

    for token in (
        "SCENARIO-REPORT-6364-1",
        "SCENARIO-REPORT-6364-2",
        "SCENARIO-REPORT-6364-3",
        "SCENARIO-REPORT-6364-4",
        "SCENARIO-REPORT-6364-5",
        "SCENARIO-REPORT-6364-6",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=false`",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6364_marker_and_live_arxiv_receipts() -> None:
    """SCENARIO-REPORT-6364-1 and 2: marker and arXiv receipts are pinned."""

    marker = mod.v548_marker_snapshot(REPO)

    assert marker["marker_text"] == mod.PLANNER_MARKER
    assert marker["marker_line"] == 34255
    assert marker["marker_count"] == 1
    assert marker["marker_commit"] == mod.MARKER_COMMIT
    assert marker["marker_committed_at_utc"] == mod.MARKER_COMMITTED_AT_UTC
    assert (
        marker["section_heading"]
        == "## V548 Planner Refresh (2026-08-12, after milestone 2026.08.547)"
    )
    assert marker["section_sha256"].startswith("sha256:")

    receipts = mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        network_receipt={"reachable": True, "method": "test", "error": None},
        accessed_at_utc="2026-08-13T01:15:00Z",
    )
    direct = receipts["direct_arxiv_source_receipts"]

    assert [row["arxiv_id"] for row in direct] == [
        "2608.10843",
        "2608.10502",
        "2608.10676",
        "2608.10627",
        "2608.10729",
        "2604.07480",
        "2605.15333",
    ]
    for row in direct:
        assert row["direct_url"] == f"https://arxiv.org/abs/{row['arxiv_id']}"
        assert row["endpoint_outcome"] == "live_metadata"
        assert row["planner_fallback_used"] is False
        assert row["metadata_valid"] is True
        assert row["submitted_date"] == row["planner_submitted_date"]
        assert row["classification"] in {"control_only", "deferred"}

    assert (
        receipts["semantic_scholar_ebt_and_arm_ebm_receipts"]["ebt"]["visible_citation_count"] == 33
    )
    assert (
        receipts["semantic_scholar_ebt_and_arm_ebm_receipts"]["arm_ebm"]["visible_citation_count"]
        == 8
    )
    assert receipts["github_discovery_receipts"][0]["rate_limit_remaining"] == "57"


def test_scenario_report_6364_source_classification_edges() -> None:
    """SCENARIO-REPORT-6364-4: source classes control executable scope."""

    base = {
        "stable_id": "arxiv:2608.99999",
        "title": "Fresh Executable Source",
        "url": "https://arxiv.org/abs/2608.99999",
        "source_timestamp": "2026-08-13T00:53:58Z",
        "publicly_available": True,
        "primary_or_first_party": True,
        "local_executable_route": True,
        "classification_reason": "fixture",
    }

    assert mod.classify_finding(base)["classification"] == "executable_now"

    at_marker = deepcopy(base)
    at_marker["source_timestamp"] = mod.MARKER_COMMITTED_AT_UTC
    assert mod.classify_finding(at_marker)["classification"] == "control_only"

    control = deepcopy(base)
    control["local_executable_route"] = False
    assert mod.classify_finding(control)["classification"] == "control_only"

    deferred = deepcopy(base)
    deferred["deferred"] = True
    assert mod.classify_finding(deferred)["classification"] == "deferred"

    retired = deepcopy(base)
    retired["retired_scope"] = True
    assert mod.classify_finding(retired)["classification"] == "retired_scope"

    unavailable = deepcopy(base)
    unavailable["unavailable"] = True
    assert mod.classify_finding(unavailable)["classification"] == "unavailable"

    unstable = deepcopy(base)
    unstable["url"] = "https://github.com/search?q=ebm"
    assert mod.classify_finding(unstable)["classification"] == "unavailable"

    for row in (base, at_marker, control, deferred, retired, unavailable, unstable):
        classified = mod.classify_finding(row)
        assert classified["classification_reason"].endswith(".")


def test_scenario_report_6364_unavailable_sources_do_not_create_claims() -> None:
    """SCENARIO-REPORT-6364-3: unavailable sources preserve the fallback boundary."""

    calls: list[str] = []

    def forbidden_fetcher(url: str) -> mod.JsonDict:
        calls.append(url)
        return {"ok": True, "status_code": 200, "url": url, "headers": {}, "body": ""}

    receipts = mod.collect_source_receipts(
        fetcher=forbidden_fetcher,
        network_receipt={"reachable": False, "method": "test", "error": "offline"},
        accessed_at_utc="2026-08-13T01:16:00Z",
    )
    report = mod.build_report(
        REPO,
        date="20260813",
        source_receipts=receipts,
        duration_s=1.0,
        source_window_end_utc="2026-08-13T01:16:00Z",
        command_receipts=[{"command": "full", "exit_code": 3}],
    )

    assert calls == []
    assert report["new_actionable_findings"] == []
    assert report["post_marker_findings_count"] == 0
    assert report["executable_scope_change_required"] is False
    assert report["unavailable_or_rate_limited_sources"]
    assert all(row["planner_fallback_used"] for row in receipts["direct_arxiv_source_receipts"])
    assert report["source_claim_boundaries"]["no_broad_literature_claim"] is True
    assert "verification caveat: 1 command(s) returned nonzero" in report["honest_verdict"]


def test_scenario_report_6364_closed_scope_and_schema_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6364-5 and 6: closed scopes stay closed and output validates."""

    receipts = mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        network_receipt={"reachable": True, "method": "test", "error": None},
        accessed_at_utc="2026-08-13T01:17:00Z",
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.write_freeze(
        REPO,
        date="20260813",
        source_receipts=receipts,
        duration_s=1.0,
        source_window_end_utc="2026-08-13T01:17:00Z",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )

    target = artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert mod.validate_report(report) == []
    assert report["status"] == "complete_no_scope_change"
    assert report["honest_verdict"].startswith("complete_no_scope_change:")
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["protected_files_unchanged"]["all_unchanged"] is True

    assert [lane["lane_id"] for lane in report["active_lane_freeze"]["lanes"]] == [
        "observable_local_gguf_proposal_generation",
        "verified_frontier_certified_factor_learning_with_dependency_rollback",
        "live_path_arc_two_sided_goal_evidence",
    ]
    closed = report["closed_and_deferred_scope_freeze"]["closed_patterns"]
    assert [row["pattern"] for row in closed] == list(mod.CLOSED_SCOPE_PATTERNS)
    assert all(row["reopen_allowed"] is False for row in closed)
    assert report["retired_scope_reopened"] is False

    for mutator, error in (
        (lambda data: data.pop("status"), "missing required field: status"),
        (
            lambda data: data.update({"inference_substrate": "wrong"}),
            "inference_substrate",
        ),
        (
            lambda data: data.update({"verifier_is_oracle": True}),
            "verifier_is_oracle must be false",
        ),
        (
            lambda data: data.update({"field_principles": {}}),
            "field_principles must cover exactly required fields",
        ),
        (
            lambda data: data.update({"field_provenance": {}}),
            "field_provenance must cover exactly required fields",
        ),
        (
            lambda data: data.update({"retired_scope_reopened": True}),
            "retired_scope_reopened must be false",
        ),
        (
            lambda data: data.update({"executable_scope_change_required": True}),
            "executable_scope_change_required",
        ),
        (
            lambda data: data["protected_files_unchanged"].update({"all_unchanged": False}),
            "protected_files_unchanged",
        ),
        (
            lambda data: data.update({"honest_verdict": "ok"}),
            "honest_verdict lacks terminal prefix",
        ),
    ):
        bad = deepcopy(report)
        mutator(bad)
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(error in err for err in mod.validate_report(bad))

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    with pytest.raises(ValueError, match="invalid Exp6364 freeze"):
        mod.write_report({"status": "complete"}, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=mod.REPO_ROOT, write=True, command_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_report_6364_fetch_and_receipt_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6364: helper edges are explicit and deterministic."""

    class FakeResponse:
        status = 200
        headers = {"retry-after": "3"}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    monkeypatch.setattr(mod.urllib_request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    fetched = mod._fetch_url("https://example.com/source")
    assert fetched["ok"] is True
    assert fetched["status_code"] == 200
    assert fetched["headers"]["retry-after"] == "3"

    monkeypatch.setattr(
        mod.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    failed = mod._fetch_url("https://example.com/source")
    assert failed["ok"] is False
    assert failed["error"] == "offline"

    assert mod._json_body({"body": "{bad"}) == {}
    assert mod._read_text(tmp_path / "missing.md") == ""
    assert mod._parse_timestamp("not-a-time") is None
    assert mod._parse_timestamp("") is None
    assert mod._parse_timestamp("2026-08-13T00:53:58").tzinfo is not None
    assert mod._is_stable_url("ftp://example.com") is False
    assert mod._endpoint_status({"ok": False, "status_code": 429}) == "http_429_unavailable"
    monkeypatch.setattr(mod, "MARKER_COMMITTED_AT_UTC", "not-a-time")
    with pytest.raises(ValueError, match="bad V548 marker timestamp"):
        mod._marker_dt()
    monkeypatch.setattr(mod, "MARKER_COMMITTED_AT_UTC", "2026-08-13T00:53:57Z")

    assert mod._arxiv_entries({"ok": True, "body": "<bad"}) == {}
    assert (
        mod._arxiv_entries(
            {
                "ok": True,
                "body": (
                    '<feed xmlns="http://www.w3.org/2005/Atom">'
                    "<entry><id>bad</id><title>No Id</title></entry></feed>"
                ),
            }
        )
        == {}
    )
    unavailable = mod._single_receipt(
        channel="test",
        url="https://example.com",
        accessed_at_utc="2026-08-13T01:18:00Z",
        receipt={
            "ok": False,
            "status_code": 429,
            "url": "https://example.com",
            "headers": {"retry-after": "2"},
            "error": "rate",
        },
        classification="control_only",
        reason="rate limited",
    )
    assert unavailable["classification"] == "unavailable"
    assert unavailable["retry_after"] == "2"

    class FakeSocket:
        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mod.socket, "create_connection", lambda *_args, **_kwargs: FakeSocket())
    assert mod.network_reachability_receipt()["reachable"] is True
    monkeypatch.setattr(
        mod.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    assert mod.network_reachability_receipt()["error"] == "offline"

    def rate_limited_fetcher(url: str) -> mod.JsonDict:
        return {
            "ok": False,
            "status_code": 429,
            "url": url,
            "headers": {"retry-after": "2"},
            "body": "",
            "error": "rate limited",
        }

    failed_receipts = mod.collect_source_receipts(
        fetcher=rate_limited_fetcher,
        network_receipt={"reachable": True, "method": "test", "error": None},
        accessed_at_utc="2026-08-13T01:18:00Z",
    )
    assert {
        row["classification"] for row in failed_receipts["unavailable_or_rate_limited_sources"]
    } == {"unavailable"}
    assert len(failed_receipts["unavailable_or_rate_limited_sources"]) >= 8
    semantic_without_total = mod._semantic_receipt(
        "ebt",
        mod.SEMANTIC_SCHOLAR_URLS["ebt"],
        "2026-08-13T01:18:00Z",
        {
            "ok": True,
            "status_code": 200,
            "url": mod.SEMANTIC_SCHOLAR_URLS["ebt"],
            "headers": {},
            "body": json.dumps({"data": []}),
            "error": None,
        },
    )
    assert semantic_without_total["visible_citation_count"] is None
    assert semantic_without_total["planner_dated_visible_citation_count"] == 33

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert mod._git_status(REPO) == []

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path.write_text(json.dumps({"focused": 0, "coverage": 2}), encoding="utf-8")
    assert mod.read_external_test_receipts() == [
        {"command": "focused", "exit_code": 0},
        {"command": "coverage", "exit_code": 2},
    ]

    receipt_path.write_text("{bad", encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path.write_text("[{}]", encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    report = mod.build_report(
        REPO,
        date="20260813",
        source_receipts=mod.collect_source_receipts(
            fetcher=_fake_fetcher,
            network_receipt={"reachable": True, "method": "test", "error": None},
            accessed_at_utc="2026-08-13T01:18:00Z",
        ),
        duration_s=1.0,
        source_window_end_utc="2026-08-13T01:18:00Z",
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    for mutator, error in (
        (lambda data: data.update({"random_seed": 6364}), "random_seed must be null"),
        (lambda data: data.update({"post_marker_findings_count": 1}), "post_marker_findings_count"),
        (lambda data: data.update({"status": "wrong"}), "status"),
        (
            lambda data: data["source_window_start_and_end_utc"].update(
                {"post_marker_lower_bound_utc": "2026-08-13T00:53:56Z"}
            ),
            "source_window_start_and_end_utc",
        ),
        (
            lambda data: data["source_window_start_and_end_utc"].update(
                {"source_request_end_utc": "2026-08-13T00:53:57Z"}
            ),
            "source_window_start_and_end_utc",
        ),
        (
            lambda data: data["planner_marker_path_and_hash"].update({"marker_count": 2}),
            "planner_marker_path_and_hash",
        ),
        (
            lambda data: data["direct_arxiv_source_receipts"][0].update({"arxiv_id": "bad"}),
            "direct_arxiv_source_receipts",
        ),
        (
            lambda data: data["active_lane_freeze"].update({"lanes": []}),
            "active_lane_freeze",
        ),
        (
            lambda data: data["closed_and_deferred_scope_freeze"].update({"closed_patterns": []}),
            "closed_and_deferred_scope_freeze",
        ),
        (
            lambda data: data["source_claim_boundaries"].update(
                {"no_broad_literature_claim": False}
            ),
            "source_claim_boundaries",
        ),
    ):
        bad = deepcopy(report)
        mutator(bad)
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(error in err for err in mod.validate_report(bad))

    writes: list[dict[str, object]] = []

    def fake_write_report(
        report: dict[str, object], root: Path = REPO, *, env: object = None
    ) -> Path:
        writes.append(report)
        return tmp_path / mod.RESULT_RELATIVE_PATH.name

    monkeypatch.setattr(mod, "write_report", fake_write_report)
    monkeypatch.setattr(
        mod,
        "network_reachability_receipt",
        lambda: {"reachable": False, "method": "test", "error": "offline"},
    )
    monkeypatch.setattr(
        mod,
        "collect_source_receipts",
        lambda **_kwargs: mod.planner_fallback_source_receipts(
            "2026-08-13T01:18:00Z",
            {"reachable": False, "method": "test", "error": "offline"},
        ),
    )
    run_report = mod.run(
        date="20260813",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert writes and run_report["status"] == "complete_no_scope_change"

    no_write_report = mod.run(
        date="20260813",
        root=REPO,
        write=False,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert no_write_report["status"] == "complete_no_scope_change"
