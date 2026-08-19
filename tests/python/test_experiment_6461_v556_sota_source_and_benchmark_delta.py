"""Tests for Exp6461 V556 SOTA source and benchmark receipt.

Spec refs: REQ-REPORT-6461, SCENARIO-REPORT-6461-1,
SCENARIO-REPORT-6461-2, SCENARIO-REPORT-6461-3,
SCENARIO-REPORT-6461-4, SCENARIO-REPORT-6461-5,
SCENARIO-REPORT-6461-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6461_v556_sota_source_and_benchmark_delta as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _receipt(url: str, body: str, *, status_code: int = 200, error: str | None = None) -> mod.JsonDict:
    return {
        "ok": error is None and 200 <= status_code < 400,
        "status_code": status_code,
        "url": url,
        "headers": {"content-type": "application/test"},
        "body": body,
        "error": error,
    }


def _atom_feed() -> str:
    entries = [
        (
            "2608.13417",
            "Beyond Final Scores: A Systematic Evaluation of Agents for Long-Horizon AI Research and Development",
            "2026-08-13T16:11:22Z",
            "Long-horizon process evaluation.",
        ),
        (
            "2608.13545",
            "LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure",
            "2026-08-13T17:56:12Z",
            "Controlled knowledge exposure.",
        ),
        (
            "2608.13560",
            "AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design",
            "2026-08-13T17:59:57Z",
            "Recursive harness learning.",
        ),
    ]
    body = "".join(
        "<entry>"
        f"<id>https://arxiv.org/abs/{paper_id}v1</id>"
        f"<published>{published}</published>"
        f"<title>{title}</title>"
        f"<summary>{summary}</summary>"
        "</entry>"
        for paper_id, title, published, summary in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">' + body + "</feed>"
    )


def _arc_v3_json() -> str:
    return json.dumps(
        {
            "version": "v3",
            "generatedAt": "2026-08-13T23:42:30.422Z",
            "evaluations": [
                {
                    "modelDisplayName": "Claude Opus 5 (High)",
                    "providerDisplayName": "Anthropic",
                    "score": 0.3016,
                    "resultsUrl": "/results/anthropic-claude-opus-5",
                },
                {
                    "modelDisplayName": "GPT-5.6 Sol (Max)",
                    "providerDisplayName": "OpenAI",
                    "score": 0.0778,
                    "resultsUrl": "/results/openai-gpt-5-6-sol-max",
                },
            ],
        }
    )


def _fake_fetcher(url: str, source_id: str) -> mod.JsonDict:
    if source_id == "arxiv_v556_ids":
        return _receipt(url, _atom_feed())
    if source_id.startswith("arxiv_abs_"):
        paper_id = source_id.removeprefix("arxiv_abs_").replace("_", ".")
        return _receipt(
            url,
            (
                "<html><h1>Title: "
                f"{mod.V556_ARXIV_PAPERS[paper_id]['title']}</h1>"
                "<div class='dateline'>[Submitted on 13 Aug 2026]</div></html>"
            ),
        )
    if source_id.startswith("arxiv_"):
        return _receipt(url, "rate limited", status_code=429, error="HTTP Error 429")
    if source_id == "semantic_scholar_ebt_citations":
        return _receipt(
            url,
            json.dumps(
                {
                    "total": 33,
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "ebt-row-1",
                                "title": "Distributional EBMs for Structured Reasoning",
                                "url": "https://www.semanticscholar.org/paper/ebt-row-1",
                                "year": 2026,
                                "publicationDate": "2026-06-01",
                                "externalIds": {"ArXiv": "2605.18871", "DOI": "10.0000/ebt"},
                            }
                        }
                    ],
                }
            ),
        )
    if source_id == "semantic_scholar_arm_ebm_citations":
        return _receipt(
            url,
            json.dumps(
                {
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "arm-row-1",
                                "title": "LoopUS: Energy-Aware Reasoning Control",
                                "url": "https://www.semanticscholar.org/paper/arm-row-1",
                                "year": 2026,
                                "publicationDate": "2026-07-02",
                                "externalIds": {"ArXiv": "2606.00001"},
                            }
                        }
                    ]
                }
            ),
        )
    if source_id == "openreview_relevance":
        return _receipt(
            url,
            json.dumps(
                {
                    "notes": [
                        {
                            "id": "OR-1",
                            "number": 1,
                            "content": {
                                "title": {"value": "Agentic Process Evaluation"},
                                "venue": {"value": "OpenReview public listing"},
                            },
                        }
                    ]
                }
            ),
        )
    if source_id.startswith("huggingface_paper_"):
        paper_id = source_id.removeprefix("huggingface_paper_").replace("_", ".")
        return _receipt(
            url,
            json.dumps(
                {
                    "id": paper_id,
                    "title": mod.V556_ARXIV_PAPERS[paper_id]["title"],
                    "publishedAt": "2026-08-13T20:00:00.000Z",
                    "authors": [{"name": "Example Author"}],
                    "url": f"https://huggingface.co/papers/{paper_id}",
                }
            ),
        )
    if source_id == "github_relevance":
        return _receipt(
            url,
            json.dumps(
                {
                    "total_count": 1,
                    "items": [
                        {
                            "full_name": "example/autodesign-harness",
                            "html_url": "https://github.com/example/autodesign-harness",
                            "description": "Harness learning notes",
                            "pushed_at": "2026-08-13T22:00:00Z",
                            "stargazers_count": 4,
                            "archived": False,
                        }
                    ],
                }
            ),
        )
    if source_id.startswith("extropic_"):
        return _receipt(url, "Extropic TSU XTR-0 Z1 Stick and Z1 Card early access hardware")
    if source_id == "logical_intelligence_kona":
        return _receipt(url, "Logical Intelligence Kona 1.0 and Aleph product page")
    if source_id == "arc_leaderboard_page":
        return _receipt(url, "<html><body>ARC-AGI-3 Leaderboard</body></html>")
    if source_id == "arc_leaderboard_data_js":
        return _receipt(url, "export const leaderboard = [];")
    if source_id == "arc_leaderboard_v3_json":
        return _receipt(url, _arc_v3_json())
    raise AssertionError(f"unexpected source {source_id} {url}")


def _receipts() -> mod.JsonDict:
    return mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        network_receipt={
            "reachable": True,
            "method": "test socket",
            "queried_at_utc": "2026-08-19T14:00:00Z",
            "error": None,
        },
        accessed_at_utc="2026-08-19T14:01:00Z",
    )


def _report() -> mod.JsonDict:
    return mod.build_report(
        REPO,
        date="20260819",
        source_receipts=_receipts(),
        duration_s=1.25,
        tests_run=[{"command": "pytest exp6461", "exit_code": 0, "status": "passed"}],
    )


def test_req_report_6461_spec_declares_fields_and_scenarios() -> None:
    """REQ-REPORT-6461: OpenSpec records the source receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6461") :]

    for token in (
        "SCENARIO-REPORT-6461-1",
        "SCENARIO-REPORT-6461-2",
        "SCENARIO-REPORT-6461-3",
        "SCENARIO-REPORT-6461-4",
        "SCENARIO-REPORT-6461-5",
        "SCENARIO-REPORT-6461-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle` SHALL be false",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6461_source_receipts_have_http_state_and_hash() -> None:
    """SCENARIO-REPORT-6461-1: every queried URL gets a reproducible receipt."""

    receipts = _receipts()

    assert receipts["network_reachability"]["reachable"] is True
    assert len(receipts["source_timestamps_and_hashes"]) == len(mod.SOURCE_QUERIES)
    by_id = {row["source_id"]: row for row in receipts["source_timestamps_and_hashes"]}
    assert by_id["arxiv_v556_ids"]["http_state"] == "http_200"
    assert by_id["arxiv_kan"]["http_state"] == "http_429"
    for row in by_id.values():
        assert row["queried_at_utc"] == "2026-08-19T14:01:00Z"
        assert row["source_url"].startswith("https://")
        assert row["response_sha256"].startswith("sha256:")
        assert isinstance(row["byte_count"], int)


def test_scenario_report_6461_arxiv_and_secondary_rows_are_parsed() -> None:
    """SCENARIO-REPORT-6461-2 and 3: arXiv and citation identities are kept."""

    report = _report()

    assert report["arxiv_release_boundary"]["latest_observed_arxiv_id"] == "2608.13560"
    assert report["arxiv_release_boundary"]["latest_observed_submitted_utc"] == "2026-08-13T17:59:57Z"
    assert report["arxiv_release_boundary"]["status"] == "verified_from_arxiv_api_and_abs_pages"
    assert [row["arxiv_id"] for row in report["promoted_findings"]] == [
        "2608.13417",
        "2608.13545",
        "2608.13560",
    ]
    assert {row["disposition"] for row in report["promoted_findings"]} == {"experiment_hook"}

    assert report["ebt_citation_rows"][0]["returned_total"] == 33
    assert report["ebt_citation_rows"][0]["title"] == "Distributional EBMs for Structured Reasoning"
    assert report["ebt_citation_rows"][0]["external_ids"]["ArXiv"] == "2605.18871"
    assert report["arm_ebm_citation_rows"][0]["returned_total"] is None
    assert report["arm_ebm_citation_rows"][0]["count_invented"] is False
    assert report["openreview_rows"][0]["title"] == "Agentic Process Evaluation"
    assert report["huggingface_rows"][0]["paper_id"] == "2608.13417"
    assert report["github_rows"][0]["full_name"] == "example/autodesign-harness"


def test_scenario_report_6461_arc_receipt_uses_primary_loaded_data() -> None:
    """SCENARIO-REPORT-6461-4: ARC score is sourced from first-party data."""

    report = _report()
    arc = report["rendered_arc_leaderboard_receipt"]

    assert arc["primary_page_url"] == "https://arcprize.org/leaderboard"
    assert arc["score_basis"] == "rendered_primary_loaded_data"
    assert arc["leader_model"] == "Claude Opus 5 (High)"
    assert arc["displayed_public_score_percent"] == pytest.approx(30.16)
    assert arc["displayed_public_score_text"] == "30.2%"
    assert arc["not_search_snippet"] is True
    assert arc["cached_number_used"] is False
    assert arc["source_hash"].startswith("sha256:")


def test_scenario_report_6461_schema_classifications_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6461-5 and 6: artifact schema is complete and honest."""

    report = _report()

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["status"] == "complete_primary_source_receipt"
    assert report["blocked_reason"] is None
    assert report["gate_check_summary"]["status"] == "passed"
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["honest_verdict"].startswith("complete_primary_source_receipt:")
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)

    dispositions = {row["disposition"] for row in report["per_unit_rows"]}
    assert dispositions == {
        "experiment_hook",
        "watch-only",
        "duplicate",
        "retired scope",
        "unavailable substrate",
    }
    for row in report["per_unit_rows"]:
        assert {"previous_state", "current_state", "evidence_hash", "relevance", "disposition"} <= set(row)
        assert row["evidence_hash"].startswith("sha256:")

    assert report["duplicates_and_retired_scopes"]
    assert report["unavailable_substrates"]
    assert mod.recompute_reproducibility_checksum(report) == report["reproducibility_checksum"]

    target = mod.write_report(report, root=REPO, env={ARTIFACT_ROOT_ENV: str(tmp_path)})
    assert target == tmp_path / mod.RESULT_RELATIVE_PATH.name
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["reproducibility_checksum"] == report["reproducibility_checksum"]


def test_scenario_report_6461_validation_fails_closed() -> None:
    """REQ-REPORT-6461: invalid source receipts and blocked artifacts fail."""

    report = _report()
    missing = deepcopy(report)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_report(missing)

    oracle = deepcopy(report)
    oracle["verifier_is_oracle"] = True
    oracle["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_report(oracle)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_report(bad_checksum)

    empty_rows = deepcopy(report)
    empty_rows["per_unit_rows"] = []
    empty_rows["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(empty_rows)
    with pytest.raises(ValueError, match="per_unit_rows"):
        mod.validate_report(empty_rows)

    blocked = deepcopy(report)
    blocked["status"] = "blocked_arc_render"
    blocked["blocked_reason"] = None
    blocked["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_reason"):
        mod.validate_report(blocked)

    bad_source = deepcopy(report)
    bad_source["source_timestamps_and_hashes"][0].pop("response_sha256")
    bad_source["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_source)
    with pytest.raises(ValueError, match="source receipt"):
        mod.validate_report(bad_source)


def test_scenario_report_6461_fallback_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6461: fallback parsers keep unavailable evidence explicit."""

    assert mod.utc_now_iso().endswith("Z")
    assert mod.path_sha256(tmp_path / "missing.txt") is None

    def one_arg_fetcher(url: str) -> mod.JsonDict:
        return _receipt(url, b"bytes-body")

    fetched = mod._fetch(one_arg_fetcher, "https://example.test/source", "source")
    normalised = mod._normalise_receipt(
        fetched,
        {
            "source_id": "source",
            "surface": "test",
            "query": "bytes",
            "url": "https://example.test/source",
        },
        "2026-08-19T15:00:00Z",
    )
    assert normalised["body"] == "bytes-body"

    blocked_receipts = mod.collect_source_receipts(
        fetcher=lambda url, source_id: (_ for _ in ()).throw(AssertionError("must not fetch")),
        network_receipt={"reachable": False, "method": "test", "error": "offline"},
        accessed_at_utc="2026-08-19T15:01:00Z",
    )
    blocked_report = mod.build_report(
        REPO,
        date="20260819",
        source_receipts=blocked_receipts,
        duration_s=0.01,
        tests_run=[],
    )
    assert blocked_report["status"] == "blocked_primary_source_receipt"
    assert blocked_report["blocked_reason"] == "arxiv_boundary_primary_sourced"

    assert mod._json_body({"body": "not-json"}) is None
    assert mod._parse_arxiv_atom("", "s", "u", "h") == []
    assert mod._parse_arxiv_atom("<bad", "s", "u", "h") == []
    assert mod._parse_arxiv_atom("<feed><entry><id>not-an-arxiv-id</id></entry></feed>", "s", "u", "h") == []

    paper_id = "2608.13417"
    span_rows = mod._parse_abs_pages(
        {
            f"arxiv_abs_{paper_id.replace('.', '_')}": {
                "ok": True,
                "body": "<span>Title:</span> Span Matched Title <",
                "source_url": f"https://arxiv.org/abs/{paper_id}",
                "response_sha256": "sha256:" + "2" * 64,
            }
        }
    )
    assert span_rows[0]["title"] == "Span Matched Title"
    assert mod.arxiv_release_boundary({})["status"] == "blocked_no_primary_arxiv_release_rows"
    direct_only = mod.arxiv_release_boundary(
        {
            "arxiv_v556_ids": {"body": "", "source_url": "u", "response_sha256": "sha256:" + "0" * 64},
            f"arxiv_abs_{paper_id.replace('.', '_')}": {
                "ok": True,
                "body": "<h1>Title: Direct Only</h1>",
                "source_url": f"https://arxiv.org/abs/{paper_id}",
                "response_sha256": "sha256:" + "3" * 64,
            },
        }
    )
    assert direct_only["status"] == "verified_from_arxiv_abs_pages_api_unavailable"

    no_json_semantic = mod.semantic_citation_rows(
        {"body": "429", "status_code": 429, "source_id": "s", "source_url": "u", "response_sha256": "h"},
        source_paper="arXiv:2507.02092",
        trail="EBT",
    )
    assert no_json_semantic[0]["rate_limited"] is True
    assert mod.semantic_citation_rows(
        {"body": json.dumps({"data": ["bad"]}), "source_id": "s"},
        source_paper="arXiv:2507.02092",
        trail="EBT",
    ) == []
    assert mod.openreview_rows({"body": "bad"}) == []
    assert mod.openreview_rows({"body": json.dumps({"notes": ["bad"]})}) == []
    assert all(row["parse_state"] == "no_json_row" for row in mod.huggingface_rows({}))
    assert mod.github_rows({"body": "bad"}) == []
    assert mod.github_rows({"body": json.dumps({"items": ["bad"]})}) == []
    assert mod._leaderboard_rows_from_payload([{"score": 1}, "bad"]) == [{"score": 1}]
    assert mod._leaderboard_rows_from_payload({"rows": [{"score": 2}, "bad"]}) == [{"score": 2}]
    assert mod._leaderboard_rows_from_payload("bad") == []
    assert mod.rendered_arc_leaderboard_receipt({})["score_basis"] == "blocked_no_primary_loaded_data"

    gate = mod.gate_check_summary(
        {
            "source_timestamps_and_hashes": [{}],
            "arxiv_release_boundary": {"status": "blocked", "boundary_is_primary_sourced": False},
            "rendered_arc_leaderboard_receipt": {"score_basis": "blocked_no_primary_loaded_data"},
            "protected_files_unchanged": {"all_unchanged": False},
        }
    )
    assert {row["check"] for row in gate["failed_checks"]} == {
        "source_hashes_present",
        "arxiv_boundary_primary_sourced",
        "arc_primary_score",
        "protected_files_unchanged",
    }
    empty_gate = mod.gate_check_summary(
        {
            "source_timestamps_and_hashes": [],
            "arxiv_release_boundary": {"boundary_is_primary_sourced": True},
            "rendered_arc_leaderboard_receipt": {"score_basis": "rendered_primary_loaded_data"},
            "protected_files_unchanged": {"all_unchanged": True},
        }
    )
    assert empty_gate["failed_checks"][0]["check"] == "source_receipts_present"

    good = _report()
    bad_substrate = deepcopy(good)
    bad_substrate["inference_substrate"] = "runtime_execution"
    bad_substrate["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_report(bad_substrate)

    bad_principles = deepcopy(good)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_report(bad_principles)

    bad_provenance = deepcopy(good)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_report(bad_provenance)

    bad_gate = deepcopy(good)
    bad_gate["gate_check_summary"] = "not a mapping"
    bad_gate["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_gate)
    with pytest.raises(ValueError, match="gate_check_summary"):
        mod.validate_report(bad_gate)

    bad_verdict = deepcopy(good)
    bad_verdict["honest_verdict"] = "wrong prefix"
    bad_verdict["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_report(bad_verdict)

    missing_path = tmp_path / "missing-tests.json"
    invalid_path = tmp_path / "invalid-tests.json"
    dict_path = tmp_path / "dict-tests.json"
    list_path = tmp_path / "list-tests.json"
    invalid_path.write_text("bad", encoding="utf-8")
    dict_path.write_text(json.dumps({"command": "pytest"}), encoding="utf-8")
    list_path.write_text(json.dumps([{"command": "pytest"}]), encoding="utf-8")
    assert mod.tests_run_from_file(missing_path) == []
    assert mod.tests_run_from_file(invalid_path) == []
    assert mod.tests_run_from_file(dict_path) == []
    assert mod.tests_run_from_file(list_path) == [{"command": "pytest"}]
