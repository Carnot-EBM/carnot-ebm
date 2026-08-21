"""Tests for Exp6475 V557 primary source and product state receipt.

Spec refs: REQ-REPORT-6475, SCENARIO-REPORT-6475-RECEIPTS,
SCENARIO-REPORT-6475-ARXIV, SCENARIO-REPORT-6475-CITATIONS,
SCENARIO-REPORT-6475-PRODUCTS, SCENARIO-REPORT-6475-ARC,
SCENARIO-REPORT-6475-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.modules.setdefault("jax", None)
sys.modules.setdefault("jax.numpy", None)

from carnot import experiment_6475_v557_primary_source_and_product_state as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _receipt(
    url: str, body: str | bytes, *, status_code: int = 200, error: str | None = None
) -> mod.JsonDict:
    return {
        "ok": error is None and 200 <= status_code < 400,
        "status_code": status_code,
        "url": url,
        "headers": {"content-type": "application/test"},
        "body": body,
        "error": error,
    }


def _atom_feed(entries: list[tuple[str, str, str, str]]) -> str:
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


def _v557_atom_feed() -> str:
    return _atom_feed(
        [
            (
                "2608.17956",
                "An Omitted Mode Is a Rare Rule",
                "2026-08-18T19:01:00Z",
                "Protocol identifiability and rare modes.",
            ),
            (
                "2608.17687",
                "Mixture-of-Expert Blocks Contain Strong Hallucination Detection Signals",
                "2026-08-18T12:30:00Z",
                "MoE routing signals for hallucination detection.",
            ),
            (
                "2608.15143",
                "Translating finite-domain integer constraint models",
                "2026-08-15T09:00:00Z",
                "Backend-neutral constraint translation.",
            ),
            (
                "2608.13959",
                "Repair, Not Improvement",
                "2026-08-14T08:00:00Z",
                "Constrained decoding repairs format.",
            ),
            (
                "2608.13326",
                "Beyond Local Accuracy",
                "2026-08-13T21:00:00Z",
                "Older protocol audit outside this release boundary.",
            ),
        ]
    )


def _arc_v3_json() -> str:
    return json.dumps(
        {
            "generatedAt": "2026-08-21T12:00:00.000Z",
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
    if source_id == "arxiv_v557_ids":
        return _receipt(url, _v557_atom_feed())
    if source_id.startswith("arxiv_abs_"):
        paper_id = source_id.removeprefix("arxiv_abs_").replace("_", ".")
        title = mod.V557_ARXIV_PAPERS[paper_id]["title"]
        return _receipt(url, f"<html><span>Title:</span> {title} <div>Submitted</div></html>")
    if source_id == "arxiv_ising":
        return _receipt(url, "rate limited", status_code=429, error="HTTP Error 429")
    if source_id.startswith("arxiv_"):
        return _receipt(
            url,
            _atom_feed(
                [
                    (
                        "2608.17956",
                        "An Omitted Mode Is a Rare Rule",
                        "2026-08-18T19:01:00Z",
                        "Protocol identifiability.",
                    )
                ]
            ),
        )
    if source_id == "semantic_scholar_ebt_citations":
        return _receipt(
            url,
            json.dumps(
                {
                    "total": 31,
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "ebt-arxiv-row",
                                "title": "Self-Explainable Latent Reasoning",
                                "url": "https://www.semanticscholar.org/paper/ebt-arxiv-row",
                                "year": 2026,
                                "publicationDate": "2026-08-14",
                                "externalIds": {"ArXiv": "2608.13570"},
                            }
                        },
                        {
                            "citingPaper": {
                                "paperId": "ebt-nonarxiv-row",
                                "title": "Venue Only EBT Row",
                                "url": "https://www.semanticscholar.org/paper/ebt-nonarxiv-row",
                                "year": 2026,
                                "publicationDate": None,
                                "externalIds": {"DOI": "10.0000/example"},
                            }
                        },
                    ],
                }
            ),
        )
    if source_id == "semantic_scholar_arm_ebm_citations":
        return _receipt(
            url,
            json.dumps(
                {
                    "total": 8,
                    "data": [
                        {
                            "citingPaper": {
                                "paperId": "arm-row",
                                "title": "Distributional Energy-Based Models",
                                "url": "https://www.semanticscholar.org/paper/arm-row",
                                "year": 2026,
                                "publicationDate": "2026-05-15",
                                "externalIds": {"ArXiv": "2605.18871"},
                            }
                        }
                    ],
                }
            ),
        )
    if source_id.startswith("openreview_"):
        return _receipt(
            url,
            json.dumps(
                {
                    "notes": [
                        {
                            "id": "OR-1",
                            "number": 7,
                            "content": {
                                "title": {"value": "Spilled Energy"},
                                "venue": {"value": "OpenReview public submission"},
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
                    "title": mod.V557_ARXIV_PAPERS[paper_id]["title"],
                    "publishedAt": "2026-08-18T20:00:00.000Z",
                    "authors": [{"name": "Example Author"}],
                    "url": f"https://huggingface.co/papers/{paper_id}",
                }
            ),
        )
    if source_id == "github_trending_python_weekly":
        return _receipt(
            url,
            """
            <html><article class="Box-row">
            <h2><a href="/example/research-agent">example / research-agent</a></h2>
            <p>Agent research tools</p><span>1,234 stars</span>
            </article></html>
            """,
        )
    if source_id == "extropic_full_stack_update":
        return _receipt(
            url,
            (
                "Torx and Thermalizers compiler preview. Z1 taped out with "
                "269,568 p-bits, 16-neighbor connectivity, over 50 MHz "
                "sampling, and under one watt. A live early-access GPU "
                "simulator API is available. Z1 sticks, cards, and clusters "
                "are planned for 2027 early access."
            ),
        )
    if source_id == "extropic_api_landing":
        return _receipt(url, "Extropic API early access waitlist for the GPU simulator")
    if source_id == "extropic_writing_index":
        return _receipt(url, "Extropic writing index mentions Z1 and Thermalizers")
    if source_id.startswith("logical_intelligence_"):
        return _receipt(
            url, "Kona 1.0 and Aleph global constraint scoring product page with no public weights"
        )
    if source_id == "arc_leaderboard_page":
        return _receipt(url, "<html><body>ARC-AGI-3 leaderboard 30.2%</body></html>")
    if source_id == "arc_leaderboard_v3_json":
        return _receipt(url, _arc_v3_json())
    raise AssertionError(f"unexpected source {source_id} {url}")


def _receipts(tmp_path: Path) -> mod.JsonDict:
    return mod.collect_source_receipts(
        fetcher=_fake_fetcher,
        network_receipt={
            "reachable": True,
            "method": "test socket",
            "queried_at_utc": "2026-08-21T14:00:00Z",
            "error": None,
        },
        accessed_at_utc="2026-08-21T14:01:00Z",
        cache_dir=tmp_path / "cache",
    )


def _report(tmp_path: Path) -> mod.JsonDict:
    return mod.build_report(
        REPO,
        date="20260821",
        source_receipts=_receipts(tmp_path),
        duration_s=1.25,
        tests_run=[{"command": "pytest exp6475", "exit_code": 0, "status": "passed"}],
        render_snapshot={
            "attempted": True,
            "tool": "test-renderer",
            "rendered_dom_available": True,
            "rendered_html_sha256": "sha256:" + "a" * 64,
            "screenshot_sha256": "sha256:" + "b" * 64,
            "displayed_text_hash": "sha256:" + "c" * 64,
            "displayed_text_excerpt": "ARC-AGI-3 leaderboard 30.2%",
        },
    )


def test_req_report_6475_spec_declares_fields_and_scenarios() -> None:
    """REQ-REPORT-6475: OpenSpec records the V557 receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6475") :]

    for token in (
        "SCENARIO-REPORT-6475-RECEIPTS",
        "SCENARIO-REPORT-6475-ARXIV",
        "SCENARIO-REPORT-6475-CITATIONS",
        "SCENARIO-REPORT-6475-PRODUCTS",
        "SCENARIO-REPORT-6475-ARC",
        "SCENARIO-REPORT-6475-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle` SHALL be false",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6475_receipts_have_cache_hash_and_blocked_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6475-RECEIPTS: every source URL is accounted for."""

    receipts = _receipts(tmp_path)

    assert receipts["network_reachability"]["reachable"] is True
    assert len(receipts["source_timestamps_and_hashes"]) == len(mod.SOURCE_QUERIES)
    by_id = {row["source_id"]: row for row in receipts["source_timestamps_and_hashes"]}
    assert by_id["arxiv_v557_ids"]["http_state"] == "http_200"
    assert by_id["arxiv_ising"]["http_state"] == "http_429"
    assert by_id["arxiv_ising"]["source_blocked"] is True
    for row in by_id.values():
        assert row["queried_at_utc"] == "2026-08-21T14:01:00Z"
        assert row["source_url"].startswith("https://")
        assert row["response_sha256"].startswith("sha256:")
        assert Path(row["cache_path"]).is_file()
        assert isinstance(row["byte_count"], int)


def test_scenario_report_6475_arxiv_boundary_and_promotions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6475-ARXIV: release rows stay inside the dated window."""

    report = _report(tmp_path)

    boundary = report["arxiv_release_boundary"]
    assert boundary["release_window_start"] == "2026-08-14"
    assert boundary["release_window_end"] == "2026-08-18"
    assert boundary["latest_observed_arxiv_id"] == "2608.17956"
    assert "2608.13326" in {row["arxiv_id"] for row in boundary["out_of_boundary_rows"]}
    assert "2608.13326" not in {row["arxiv_id"] for row in boundary["observed_rows"]}
    assert boundary["boundary_is_primary_sourced"] is True

    promoted = {row["arxiv_id"]: row for row in report["promoted_findings"]}
    assert {"2608.17956", "2608.17687", "2608.15143", "2608.13959"} <= set(promoted)
    assert promoted["2608.17956"]["disposition"] == "bounded_experiment_hook"
    assert promoted["2608.17956"]["novelty"] == "already_in_v557_primary_refresh"
    assert report["preconditions_checked"]["research_references_append_performed"] is False


def test_scenario_report_6475_citation_and_secondary_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6475-CITATIONS: citation identities and counts stay separate."""

    report = _report(tmp_path)

    ebt_rows = report["ebt_citation_rows"]
    assert [row["title"] for row in ebt_rows] == [
        "Self-Explainable Latent Reasoning",
        "Venue Only EBT Row",
    ]
    assert {row["trail_arxiv_indexed_count"] for row in ebt_rows} == {1}
    assert {row["trail_returned_total"] for row in ebt_rows} == {31}
    assert [row["arxiv_indexed"] for row in ebt_rows] == [True, False]
    assert report["arm_ebm_citation_rows"][0]["trail_arxiv_indexed_count"] == 1
    assert report["arm_ebm_citation_rows"][0]["trail_returned_total"] == 8
    assert report["openreview_rows"][0]["venue"] == "OpenReview public submission"
    assert report["huggingface_rows"][0]["paper_id"] == "2608.17956"
    assert report["github_rows"][0]["full_name"] == "example/research-agent"


def test_scenario_report_6475_products_and_unavailable_substrates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6475-PRODUCTS: product pages never become execution proof."""

    report = _report(tmp_path)
    extropic = report["extropic_first_party_status"]
    logical = report["logical_intelligence_first_party_status"]

    assert extropic["execution_claim_made"] is False
    assert extropic["z1_status_claim"]["claim"] == "Z1 taped out"
    assert extropic["simulator_api_status_claim"]["claim"] == "early-access GPU simulator API"
    assert logical["execution_claim_made"] is False
    assert logical["public_weights_or_runner_found"] is False
    missing_names = {row["name"] for row in report["unavailable_substrates"]}
    assert {
        "Extropic Z1 device",
        "Extropic simulator API credentials",
        "Logical Intelligence Kona weights or runner",
    } <= missing_names


def test_scenario_report_6475_arc_receipt_and_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6475-ARC/SCHEMA: the artifact is complete and bounded."""

    report = _report(tmp_path)
    arc = report["rendered_arc_leaderboard_receipt"]

    assert arc["score_basis"] == "rendered_snapshot_and_first_party_loaded_data"
    assert arc["leader_model"] == "Claude Opus 5 (High)"
    assert arc["displayed_public_score_percent"] == pytest.approx(30.16)
    assert arc["displayed_public_score_text"] == "30.2%"
    assert arc["not_search_snippet"] is True
    assert arc["cached_local_record_used"] is False
    assert arc["execution_claim_made"] is False

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["status"] == "complete_primary_source_receipt"
    assert report["gate_check_summary"]["status"] == "passed"
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["honest_verdict"].startswith("complete_primary_source_receipt:")
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.recompute_reproducibility_checksum(report) == report["reproducibility_checksum"]
    for row in report["per_unit_rows"]:
        assert {
            "previous_state",
            "current_state",
            "evidence_hash",
            "novelty",
            "relevance",
            "disposition",
        } <= set(row)
        assert row["execution_evidence"] is False

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = mod.write_report(report, root=REPO, env={ARTIFACT_ROOT_ENV: str(out_dir)})
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["reproducibility_checksum"] == report["reproducibility_checksum"]


def test_scenario_report_6475_validation_and_fallbacks_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6475: malformed receipts and unsupported claims fail closed."""

    tmp_out = tmp_path / "out"
    tmp_out.mkdir()
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
        "2026-08-21T15:00:00Z",
        cache_dir=tmp_path / "single-cache",
    )
    assert normalised["body"] == "bytes-body"
    assert Path(normalised["cache_path"]).is_file()

    blocked_receipts = mod.collect_source_receipts(
        fetcher=lambda url, source_id: (_ for _ in ()).throw(AssertionError("must not fetch")),
        network_receipt={"reachable": False, "method": "test", "error": "offline"},
        accessed_at_utc="2026-08-21T15:01:00Z",
        cache_dir=tmp_path / "blocked-cache",
    )
    blocked_report = mod.build_report(
        REPO,
        date="20260821",
        source_receipts=blocked_receipts,
        duration_s=0.01,
        tests_run=[],
    )
    assert blocked_report["status"] == "blocked_primary_source_receipt"
    assert blocked_report["gate_check_summary"]["failed_checks"][0]["failed_url"].startswith(
        "https://"
    )

    assert mod._json_body({"body": "not-json"}) is None
    assert mod._parse_arxiv_atom("", "s", "u", "h") == []
    assert mod._parse_arxiv_atom("<bad", "s", "u", "h") == []
    assert (
        mod._parse_arxiv_atom("<feed><entry><id>not-an-arxiv-id</id></entry></feed>", "s", "u", "h")
        == []
    )
    assert mod._parse_abs_title("<h1>Title: Direct Only</h1>", "Fallback") == "Direct Only"
    assert mod._parse_abs_title("<html>No title</html>", "Fallback") == "Fallback"
    assert mod._inside_release_window("2026-08-14T00:00:00Z") is True
    assert mod._inside_release_window("2026-08-13T23:59:59Z") is False
    assert mod._inside_release_window("") is False
    direct_only = mod.arxiv_release_boundary(
        {
            "arxiv_abs_2608_17956": {
                "ok": True,
                "body": "<h1>Title: Direct Only</h1>",
                "source_url": "https://arxiv.org/abs/2608.17956",
                "response_sha256": "sha256:" + "4" * 64,
                "cache_path": "/tmp/direct-only",
            }
        }
    )
    assert direct_only["status"] == "verified_from_arxiv_abs_pages_api_unavailable"
    assert mod._leaderboard_rows_from_payload([{"score": 1}, "bad"]) == [{"score": 1}]
    assert mod._leaderboard_rows_from_payload({"rows": [{"score": 2}, "bad"]}) == [{"score": 2}]
    assert mod._leaderboard_rows_from_payload("bad") == []
    assert (
        mod.rendered_arc_leaderboard_receipt({})["score_basis"] == "blocked_no_primary_loaded_data"
    )
    arc_no_render = mod.rendered_arc_leaderboard_receipt(
        {
            "arc_leaderboard_v3_json": {
                "body": _arc_v3_json(),
                "response_sha256": "sha256:" + "5" * 64,
                "source_url": "https://arcprize.org/media/data/leaderboard/v3.json",
            }
        }
    )
    assert arc_no_render["score_basis"] == "first_party_loaded_data_without_render"
    assert mod.openreview_rows({"body": "bad"}) == []
    assert mod.openreview_rows({"body": json.dumps({"notes": ["bad"]})}) == []
    assert all(row["parse_state"] == "no_json_row" for row in mod.huggingface_rows({}))
    assert mod.github_rows({"body": "bad"}) == []
    assert mod.github_rows({"body": json.dumps({"items": ["bad"]})}) == []
    github_api = mod.github_rows(
        {
            "body": json.dumps(
                {
                    "items": [
                        {
                            "full_name": "example/api",
                            "html_url": "https://github.com/example/api",
                            "description": "API row",
                            "pushed_at": "2026-08-21T00:00:00Z",
                            "stargazers_count": 9,
                            "archived": False,
                        }
                    ]
                }
            ),
            "source_id": "github_api",
        }
    )
    assert github_api[0]["full_name"] == "example/api"
    assert (
        mod._source_previous_state("unknown_source")
        == "not previously recorded in an Exp6475 receipt"
    )

    no_json_semantic = mod.semantic_citation_rows(
        {
            "body": "429",
            "status_code": 429,
            "source_id": "s",
            "source_url": "u",
            "response_sha256": "h",
        },
        source_paper="arXiv:2507.02092",
        trail="EBT",
    )
    assert no_json_semantic[0]["rate_limited"] is True
    assert no_json_semantic[0]["count_invented"] is False
    assert (
        mod.semantic_citation_rows(
            {"body": json.dumps({"data": ["bad"]}), "source_id": "s"},
            source_paper="arXiv:2507.02092",
            trail="EBT",
        )
        == []
    )

    report = _report(tmp_path)
    missing = deepcopy(report)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_report(missing)

    oracle = deepcopy(report)
    oracle["verifier_is_oracle"] = True
    oracle["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_report(oracle)

    bad_substrate = deepcopy(report)
    bad_substrate["inference_substrate"] = "runtime_execution"
    bad_substrate["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(
        bad_substrate
    )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_report(bad_substrate)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_report(bad_checksum)

    bad_source = deepcopy(report)
    bad_source["source_timestamps_and_hashes"][0].pop("cache_path")
    bad_source["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_source)
    with pytest.raises(ValueError, match="source receipt"):
        mod.validate_report(bad_source)

    bad_gate = deepcopy(report)
    bad_gate["gate_check_summary"] = "not a mapping"
    bad_gate["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_gate)
    with pytest.raises(ValueError, match="gate_check_summary"):
        mod.validate_report(bad_gate)

    bad_verdict = deepcopy(report)
    bad_verdict["honest_verdict"] = "wrong prefix"
    bad_verdict["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_report(bad_verdict)

    bad_row = deepcopy(report)
    bad_row["per_unit_rows"][0]["execution_evidence"] = True
    bad_row["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_row)
    with pytest.raises(ValueError, match="execution evidence"):
        mod.validate_report(bad_row)

    empty_rows = deepcopy(report)
    empty_rows["per_unit_rows"] = []
    empty_rows["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(empty_rows)
    with pytest.raises(ValueError, match="per_unit_rows"):
        mod.validate_report(empty_rows)

    bad_product = deepcopy(report)
    bad_product["extropic_first_party_status"]["execution_claim_made"] = True
    bad_product["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_product)
    with pytest.raises(ValueError, match="execution claim"):
        mod.validate_report(bad_product)

    bad_status = deepcopy(report)
    bad_status["status"] = "blocked_without_gate"
    bad_status["gate_check_summary"] = {"status": "passed", "failed_checks": []}
    bad_status["honest_verdict"] = "blocked_without_gate: missing gate row"
    bad_status["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="blocked status"):
        mod.validate_report(bad_status)

    bad_principles = deepcopy(report)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(
        bad_principles
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_report(bad_principles)

    bad_provenance = deepcopy(report)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.recompute_reproducibility_checksum(
        bad_provenance
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_report(bad_provenance)

    gate = mod.gate_check_summary(
        {
            "source_timestamps_and_hashes": [
                "not-a-row",
                {
                    "source_id": "arxiv_v557_ids",
                    "source_url": "https://export.arxiv.org/api/query?id_list=2608.17956",
                    "http_state": "http_200",
                    "cache_path": "",
                    "source_blocked": False,
                },
            ],
            "arxiv_release_boundary": {"status": "blocked", "boundary_is_primary_sourced": False},
            "rendered_arc_leaderboard_receipt": {"score_basis": "blocked_no_primary_loaded_data"},
            "protected_files_unchanged": {"all_unchanged": False},
            "preconditions_checked": {"network_availability": {"reachable": True}},
            "per_unit_rows": [{"execution_evidence": True}],
        }
    )
    assert {
        "all_requested_sources_accounted",
        "source_hash_and_cache_present",
        "arxiv_boundary_primary_sourced",
        "arc_rendered_primary_score",
        "protected_files_unchanged",
        "no_source_row_execution_evidence",
    } <= {row["check"] for row in gate["failed_checks"]}
    empty_gate = mod.gate_check_summary(
        {
            "source_timestamps_and_hashes": [],
            "arxiv_release_boundary": {"boundary_is_primary_sourced": True},
            "rendered_arc_leaderboard_receipt": {
                "score_basis": "rendered_snapshot_and_first_party_loaded_data"
            },
            "protected_files_unchanged": {"all_unchanged": True},
            "preconditions_checked": {"network_availability": {"reachable": True}},
            "per_unit_rows": [],
        }
    )
    assert empty_gate["failed_checks"][0]["check"] == "source_receipts_present"

    assert mod.tests_run_from_file(tmp_path / "missing.json") == []
    bad_json = tmp_path / "bad.json"
    dict_json = tmp_path / "dict.json"
    list_json = tmp_path / "list.json"
    bad_json.write_text("bad", encoding="utf-8")
    dict_json.write_text(json.dumps({"command": "pytest"}), encoding="utf-8")
    list_json.write_text(json.dumps([{"command": "pytest"}]), encoding="utf-8")
    assert mod.tests_run_from_file(bad_json) == []
    assert mod.tests_run_from_file(dict_json) == []
    assert mod.tests_run_from_file(list_json) == [{"command": "pytest"}]
