"""Tests for Exp6411 V552 source and scope freeze.

Spec refs: REQ-REPORT-6411, SCENARIO-REPORT-6411-1,
SCENARIO-REPORT-6411-2, SCENARIO-REPORT-6411-3,
SCENARIO-REPORT-6411-4, SCENARIO-REPORT-6411-5,
SCENARIO-REPORT-6411-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6411_v552_post_marker_source_scope_freeze as mod
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
            f"<summary>{paper['abstract_fixture']}</summary>"
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
            "headers": {"x-ratelimit-remaining": "17"},
            "body": json.dumps(
                {
                    "notes": [
                        {
                            "id": "control",
                            "content": {
                                "title": {"value": "Agent-as-a-Router: Agentic Model Routing"},
                                "venue": {"value": "OpenReview public listing"},
                            },
                        }
                    ]
                }
            ),
            "error": None,
        }
    if "huggingface.co/api/papers" in url:
        return {
            "ok": False,
            "status_code": 404,
            "url": url,
            "headers": {"ratelimit": '"api";r=498;t=256'},
            "body": "",
            "error": "HTTP Error 404: Not Found",
        }
    if "2507.02092" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {},
            "body": json.dumps(
                {
                    "data": [
                        {
                            "citingPaper": {
                                "title": "Distributional EBMs for Structured Reasoning",
                                "publicationDate": "2026-06-01",
                                "url": "https://www.semanticscholar.org/paper/ebt",
                                "externalIds": {"ArXiv": "2605.18871"},
                            }
                        }
                    ]
                }
            ),
            "error": None,
        }
    if "2512.15605" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {},
            "body": json.dumps(
                {
                    "data": [
                        {
                            "citingPaper": {
                                "title": "LoopUS: Energy-Aware Reasoning Control",
                                "publicationDate": "2026-07-02",
                                "url": "https://www.semanticscholar.org/paper/arm",
                                "externalIds": {"ArXiv": "2606.00001"},
                            }
                        }
                    ]
                }
            ),
            "error": None,
        }
    if "github.com/trending" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {},
            "body": (
                "<html><article class='Box-row'><h2>"
                "<a href='/example/agent-tools'>example / agent-tools</a>"
                "</h2><p>Agent utilities</p></article></html>"
            ),
            "error": None,
        }
    if "extropic.ai" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"last-modified": "Thu, 13 Aug 2026 21:54:00 GMT"},
            "body": "TSUs XTR-0 X0 Z1 Stick Z1 Card early access 2027 hardware",
            "error": None,
        }
    if "logicalintelligence.com" in url:
        return {
            "ok": True,
            "status_code": 200,
            "url": url,
            "headers": {"last-modified": "Fri, 26 Jun 2026 23:48:05 GMT"},
            "body": "Kona 1.0 Energy-Based Models Aleph formal verification product",
            "error": None,
        }
    raise AssertionError(f"unexpected URL {url}")


def test_req_report_6411_spec_declares_fields_and_scenarios() -> None:
    """REQ-REPORT-6411: OpenSpec records the V552 freeze contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6411") : text.index("REQ-REPORT-6143")]

    for token in (
        "SCENARIO-REPORT-6411-1",
        "SCENARIO-REPORT-6411-2",
        "SCENARIO-REPORT-6411-3",
        "SCENARIO-REPORT-6411-4",
        "SCENARIO-REPORT-6411-5",
        "SCENARIO-REPORT-6411-6",
        mod.PLANNER_MARKER,
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=false`",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6411_marker_and_live_arxiv_receipts() -> None:
    """SCENARIO-REPORT-6411-1 and 2: marker and arXiv receipts are pinned."""

    marker = mod.v552_marker_snapshot(REPO)

    assert marker["marker_text"] == mod.PLANNER_MARKER
    assert marker["marker_count"] == 1
    assert marker["marker_line"] == 34679
    assert marker["marker_commit"] == mod.MARKER_COMMIT
    assert marker["marker_committed_at_utc"] == mod.MARKER_COMMITTED_AT_UTC
    assert (
        marker["section_heading"]
        == "## V552 Planner Refresh (2026-08-13, after milestone 2026.08.551)"
    )
    assert marker["section_sha256"].startswith("sha256:")

    receipts = mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        network_receipt={"reachable": True, "method": "test", "error": None},
        accessed_at_utc="2026-08-14T01:00:00Z",
    )
    direct = receipts["direct_arxiv_source_receipts"]

    assert [row["arxiv_id"] for row in direct] == ["2608.09184", "2608.10005", "2608.10725"]
    assert [row["classification"] for row in direct] == ["executable_now"] * 3
    assert [row["strict_post_marker"] for row in direct] == [False] * 3
    for row in direct:
        assert row["direct_url"] == f"https://arxiv.org/abs/{row['arxiv_id']}"
        assert row["endpoint_outcome"] == "live_metadata"
        assert row["planner_fallback_used"] is False
        assert row["metadata_valid"] is True
        assert row["submitted_date"] == row["planner_submitted_date"]
        assert row["abstract_sha256"].startswith("sha256:")
        assert row["abstract_available"] is True

    assert direct[0]["active_lane_id"] == "prospective_dual_path_csl"
    assert direct[1]["active_lane_id"] == "ccg_kernelization"
    assert direct[2]["active_lane_id"] == "selective_refinement"
    assert receipts["openreview_receipts"][0]["classification"] == "diagnostic_control"
    assert receipts["huggingface_papers_receipts"][0]["classification"] == "unavailable"
    assert (
        receipts["semantic_scholar_ebt_and_arm_ebm_receipts"]["ebt"]["sampled_citation_count"] == 1
    )
    assert receipts["github_trending_receipts"][0]["classification"] == "diagnostic_control"
    assert receipts["extropic_first_party_receipts"][0]["classification"] == "product_status"
    assert receipts["logical_intelligence_first_party_receipts"][0]["classification"] == (
        "product_status"
    )


def test_scenario_report_6411_source_classification_edges() -> None:
    """SCENARIO-REPORT-6411-3: source classes control new scope."""

    base = {
        "stable_id": "arxiv:2608.99999",
        "title": "Fresh Executable Source",
        "url": "https://arxiv.org/abs/2608.99999",
        "source_timestamp": "2026-08-14T00:35:22Z",
        "publicly_available": True,
        "primary_or_first_party": True,
        "local_executable_route": True,
        "classification_reason": "fixture",
    }

    executable = mod.classify_finding(base)
    assert executable["classification"] == "executable_now"
    assert executable["strict_post_marker"] is True

    no_local_route = deepcopy(base)
    no_local_route["local_executable_route"] = False
    assert mod.classify_finding(no_local_route)["classification"] == "diagnostic_control"

    at_marker = deepcopy(base)
    at_marker["source_timestamp"] = mod.MARKER_COMMITTED_AT_UTC
    classified_at_marker = mod.classify_finding(at_marker)
    assert classified_at_marker["classification"] == "executable_now"
    assert classified_at_marker["strict_post_marker"] is False

    control = deepcopy(base)
    control["control_only"] = True
    assert mod.classify_finding(control)["classification"] == "diagnostic_control"

    deferred = deepcopy(base)
    deferred["deferred"] = True
    assert mod.classify_finding(deferred)["classification"] == "deferred"

    product = deepcopy(base)
    product["product_status"] = True
    assert mod.classify_finding(product)["classification"] == "product_status"

    retired = deepcopy(base)
    retired["retired_scope"] = True
    assert mod.classify_finding(retired)["classification"] == "retired_scope"

    unavailable = deepcopy(base)
    unavailable["unavailable"] = True
    assert mod.classify_finding(unavailable)["classification"] == "unavailable"

    unstable = deepcopy(base)
    unstable["url"] = "https://github.com/search?q=ebm"
    assert mod.classify_finding(unstable)["classification"] == "unavailable"

    for row in (
        base,
        no_local_route,
        at_marker,
        control,
        deferred,
        product,
        retired,
        unavailable,
        unstable,
    ):
        classified = mod.classify_finding(row)
        assert classified["classification_reason"].endswith(".")


def test_scenario_report_6411_unavailable_sources_do_not_create_claims() -> None:
    """SCENARIO-REPORT-6411-3: unavailable sources preserve the boundary."""

    calls: list[str] = []

    def forbidden_fetcher(url: str) -> mod.JsonDict:
        calls.append(url)
        return {"ok": True, "status_code": 200, "url": url, "headers": {}, "body": ""}

    receipts = mod.collect_source_receipts(
        fetcher=forbidden_fetcher,
        network_receipt={"reachable": False, "method": "test", "error": "offline"},
        accessed_at_utc="2026-08-14T01:01:00Z",
    )
    report = mod.build_report(
        REPO,
        date="20260814",
        source_receipts=receipts,
        duration_s=1.0,
        source_window_end_utc="2026-08-14T01:01:00Z",
        command_receipts=[{"command": "focused", "exit_code": 3}],
    )

    assert calls == []
    assert report["post_marker_delta"]["strict_post_marker_executable_count"] == 0
    assert report["scope_changed_after_marker"] is False
    assert report["queue_edit_required"] is False
    assert report["unavailable_sources_and_rate_limits"]
    assert all(row["planner_fallback_used"] for row in receipts["direct_arxiv_source_receipts"])
    assert report["diagnostic_controls"]["no_source_metadata_proves_carnot_works"] is True
    assert "verification caveat: 1 command(s) returned nonzero" in report["honest_verdict"]


def test_scenario_report_6411_closed_scope_and_schema_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6411-4 to 6: freezes and schema stay stable."""

    receipts = mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        network_receipt={"reachable": True, "method": "test", "error": None},
        accessed_at_utc="2026-08-14T01:02:00Z",
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.write_freeze(
        REPO,
        date="20260814",
        source_receipts=receipts,
        duration_s=1.0,
        source_window_end_utc="2026-08-14T01:02:00Z",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )

    target = artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert mod.validate_report(report) == []
    assert report["status"] == "complete_no_scope_change"
    assert report["honest_verdict"].startswith("complete_no_scope_change:")
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] == mod.EXPERIMENT_SEED
    assert report["model_specs"]["model_invoked"] is False
    assert report["model_specs"]["substrate"] == mod.INFERENCE_SUBSTRATE
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"].values()) <= {
        "measured",
        "derived",
        "constant",
        "upstream",
    }
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["protected_files_unchanged"]["all_unchanged"] is True

    assert [lane["lane_id"] for lane in report["executable_scope"]["active_lanes"]] == [
        "authentic_gguf_receipts",
        "fresh_exact_event_capture",
        "ccg_kernelization",
        "selective_refinement",
        "prospective_dual_path_csl",
        "default_off_arc_policy_influence",
    ]
    closed = report["retired_or_closed_scope"]["closed_patterns"]
    assert [row["pattern"] for row in closed] == list(mod.CLOSED_SCOPE_PATTERNS)
    assert all(row["reopen_allowed"] is False for row in closed)
    assert report["diagnostic_controls"]["diagnostic_control_count"] >= 1
    assert report["deferred_product_and_hardware_scope"]["product_status_count"] == 2
    assert report["scope_changed_after_marker"] is False
    assert report["queue_edit_required"] is False

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
            lambda data: data.update({"scope_changed_after_marker": True}),
            "scope_changed_after_marker must be false",
        ),
        (
            lambda data: data.update({"queue_edit_required": True}),
            "queue_edit_required must be false",
        ),
        (
            lambda data: data.update({"random_seed": None}),
            "random_seed must equal deterministic Exp6411 seed",
        ),
        (
            lambda data: data["field_provenance"].update({"status": "direct"}),
            "field_provenance contains unsupported category",
        ),
        (
            lambda data: data["post_marker_delta"].update(
                {"strict_post_marker_executable_count": 1}
            ),
            "post_marker_delta",
        ),
        (
            lambda data: data.update({"status": "complete_scope_change_required"}),
            "status",
        ),
        (
            lambda data: data["source_window_start_and_end_utc"].update(
                {"post_marker_lower_bound_utc": "2026-08-14T00:35:22Z"}
            ),
            "source_window_start_and_end_utc",
        ),
        (
            lambda data: data["source_window_start_and_end_utc"].update(
                {"source_request_end_utc": "2026-08-14T00:35:21Z"}
            ),
            "source_window_start_and_end_utc",
        ),
        (
            lambda data: data["planner_marker_path_and_hash"].update({"marker_count": 2}),
            "planner_marker_path_and_hash",
        ),
        (
            lambda data: data["direct_arxiv_source_receipts"].pop(),
            "direct_arxiv_source_receipts",
        ),
        (
            lambda data: data["executable_scope"].update({"active_lanes": []}),
            "executable_scope",
        ),
        (
            lambda data: data["retired_or_closed_scope"].update({"closed_patterns": []}),
            "retired_or_closed_scope",
        ),
        (
            lambda data: data["diagnostic_controls"].update(
                {"no_source_metadata_proves_carnot_works": False}
            ),
            "diagnostic_controls",
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

    with pytest.raises(ValueError, match="invalid Exp6411 freeze"):
        mod.write_report({"status": "complete"}, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=mod.REPO_ROOT, write=True, command_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260814"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_report_6411_fetch_and_receipt_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6411: helper edges are explicit and deterministic."""

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

    http_error = mod.HTTPError(
        "https://example.com/missing",
        404,
        "Not Found",
        {"ratelimit": '"api";r=498;t=256'},
        None,
    )
    monkeypatch.setattr(
        mod.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error),
    )
    http_failed = mod._fetch_url("https://example.com/missing")
    assert http_failed["status_code"] == 404
    assert http_failed["headers"]["ratelimit"] == '"api";r=498;t=256'

    assert mod._json_body({"body": "{bad"}) == {}
    assert mod._read_text(tmp_path / "missing.md") == ""
    assert mod._parse_timestamp("not-a-time") is None
    assert mod._parse_timestamp("") is None
    assert mod._parse_timestamp("2026-08-14T00:35:22").tzinfo is not None
    assert mod._is_stable_url("ftp://example.com") is False
    assert mod._endpoint_status({"ok": False, "status_code": 429}) == "http_429_unavailable"
    monkeypatch.setattr(mod, "MARKER_COMMITTED_AT_UTC", "not-a-time")
    with pytest.raises(ValueError, match="bad V552 marker timestamp"):
        mod._marker_dt()
    monkeypatch.setattr(mod, "MARKER_COMMITTED_AT_UTC", "2026-08-14T00:35:21Z")

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
        accessed_at_utc="2026-08-14T01:03:00Z",
        receipt={
            "ok": False,
            "status_code": 429,
            "url": "https://example.com",
            "headers": {"retry-after": "2"},
            "error": "rate",
        },
        classification="diagnostic_control",
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no route")),
    )
    assert mod.network_reachability_receipt()["error"] == "no route"


def test_req_report_6411_endpoint_failure_and_run_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6411: endpoint and run-wrapper edge cases are explicit."""

    def failing_fetcher(url: str) -> mod.JsonDict:
        status = 429 if "openreview" in url else 503
        return {
            "ok": False,
            "status_code": status,
            "url": url,
            "headers": {"retry-after": "9"},
            "body": "",
            "error": "rate" if status == 429 else "unavailable",
        }

    receipts = mod.collect_source_receipts(
        fetcher=failing_fetcher,
        network_receipt={"reachable": True, "method": "test", "error": None},
        accessed_at_utc="2026-08-14T01:04:00Z",
    )
    channels = {row["channel"] for row in receipts["unavailable_sources_and_rate_limits"]}
    assert {
        "arxiv",
        "openreview",
        "semantic_scholar_ebt",
        "semantic_scholar_arm_ebm",
        "github_trending",
        "extropic",
        "logical_intelligence",
    } <= channels

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git missing")),
    )
    assert mod._git_status(REPO) == []

    good_report = mod.build_report(
        REPO,
        date="20260814",
        source_receipts=receipts,
        duration_s=1.0,
        source_window_end_utc="2026-08-14T01:04:00Z",
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    changed = deepcopy(good_report)
    changed["post_marker_delta"]["strict_post_marker_executable_rows"] = [
        {"classification": "executable_now", "strict_post_marker": True}
    ]
    changed["post_marker_delta"]["strict_post_marker_executable_count"] = 1
    changed["status"] = "complete_scope_change_required"
    changed["reproducibility_checksum"] = mod.payload_checksum(changed)
    errors = mod.validate_report(changed)
    assert "scope_changed_after_marker" in errors
    assert "queue_edit_required" in errors

    bad_promoted = deepcopy(good_report)
    bad_promoted["promoted_method_classifications"]["source_metadata_is_not_carnot_evidence"] = (
        False
    )
    bad_promoted["reproducibility_checksum"] = mod.payload_checksum(bad_promoted)
    assert "promoted_method_classifications" in mod.validate_report(bad_promoted)

    missing_receipt = tmp_path / "missing.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", missing_receipt)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", bad_json)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    not_mapping = tmp_path / "list.json"
    not_mapping.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", not_mapping)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"cmd": 0}), encoding="utf-8")
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", mapping)
    assert mod.read_external_test_receipts() == [{"command": "cmd", "exit_code": 0}]

    monkeypatch.setattr(
        mod,
        "network_reachability_receipt",
        lambda: {"reachable": True, "method": "test", "error": None},
    )
    monkeypatch.setattr(
        mod,
        "collect_source_receipts",
        lambda **_kwargs: receipts,
    )
    wrote: list[str] = []
    monkeypatch.setattr(
        mod,
        "write_report",
        lambda report, root=mod.REPO_ROOT, env=None: (
            wrote.append(str(report["status"])) or (tmp_path / "artifact.json")
        ),
    )
    report = mod.run(
        date="20260814",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert report["status"] == "complete_no_scope_change"
    assert wrote == ["complete_no_scope_change"]
