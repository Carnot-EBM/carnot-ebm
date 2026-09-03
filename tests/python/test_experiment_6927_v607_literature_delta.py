"""Tests for the V607 execution-time literature delta.

Spec refs: REQ-REPORT-6927, SCENARIO-REPORT-6927-PREFLIGHT,
SCENARIO-REPORT-6927-PLAN, SCENARIO-REPORT-6927-TERMINAL,
SCENARIO-REPORT-6927-COMPATIBILITY, SCENARIO-REPORT-6927-LEDGER, and
SCENARIO-REPORT-6927-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6927_v607_literature_delta as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _fake_repo(tmp_path: Path, *, include_policy: bool = True) -> Path:
    root = tmp_path / "repo"
    required = {
        "research-roadmap.yaml": "milestone: 2026.09.607\n",
        "research-references.md": "# Reference ledger\n",
        "CLAUDE.md": "## SOTA-Ingestion Cycle Discipline (MANDATORY)\nlow-concurrency\n",
    }
    if not include_policy:
        required.pop("CLAUDE.md")
    for relative, text in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def _artifact(tmp_path: Path) -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260903",
        duration_s=1.0,
        output_path=tmp_path / "result.json",
    )


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_report_6927_spec_declares_complete_advisory_contract() -> None:
    """REQ-REPORT-6927: OpenSpec owns the complete literature contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6927") :]
    for marker in (
        "SCENARIO-REPORT-6927-PREFLIGHT",
        "SCENARIO-REPORT-6927-PLAN",
        "SCENARIO-REPORT-6927-TERMINAL",
        "SCENARIO-REPORT-6927-COMPATIBILITY",
        "SCENARIO-REPORT-6927-LEDGER",
        "SCENARIO-REPORT-6927-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6927_plan_is_dated_bounded_and_arxiv_first() -> None:
    """SCENARIO-REPORT-6927-PLAN: the plan fixes bounded query order."""

    plan = mod.frozen_query_plan()
    assert mod.validate_query_plan(plan, "20260903") == []
    assert [row["source_family"] for row in plan[:8]] == list(mod.ARXIV_FAMILIES)
    assert {row["source_family"] for row in plan} == set(mod.SOURCE_FAMILIES)
    assert len(plan) == len({row["query_id"] for row in plan}) == 15
    assert all(row["planned_utc"].startswith("2026-09-03T") for row in plan)
    assert all(row["max_attempts"] == 2 for row in plan)
    assert all(row["concurrency_limit"] == 1 for row in plan)
    assert all(row["timeout_s"] == 20 for row in plan)
    assert all(row["query_text"] and row["allowed_domains"] for row in plan)
    assert all(row["acceptance_criteria"] for row in plan)

    plan[0]["query_text"] = "changed by caller"
    assert mod.frozen_query_plan()[0]["query_text"] != "changed by caller"

    malformed = mod.frozen_query_plan()
    malformed[0]["max_attempts"] = 3
    malformed[0]["concurrency_limit"] = 2
    malformed[0]["timeout_s"] = 21
    malformed[0]["query_text"] = ""
    malformed[0], malformed[8] = malformed[8], malformed[0]
    malformed.pop()
    errors = mod.validate_query_plan(malformed, "20260903")
    assert "query arxiv_ebm_verification must allow exactly two attempts" in errors
    assert "query arxiv_ebm_verification must use concurrency one" in errors
    assert "query arxiv_ebm_verification must use the 20 second timeout" in errors
    assert "query arxiv_ebm_verification lacks a frozen source field" in errors
    assert "arXiv source families must be first" in errors
    assert "query plan source family coverage is incomplete" in errors
    assert mod.validate_query_plan(mod.frozen_query_plan(), "20260904") == [
        "query plan date does not match run date"
    ]


def test_scenario_report_6927_preflight_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6927-PREFLIGHT: any missing input blocks research."""

    root = _fake_repo(tmp_path)
    output = root / mod.RESULT_RELATIVE_PATH
    assert all(row["available"] for row in mod.check_preconditions(root, "20260903", output))

    missing_policy = _fake_repo(tmp_path / "missing", include_policy=False)
    blocked = mod.build_artifact(
        root=missing_policy,
        run_date="20260903",
        duration_s=0.1,
        output_path=missing_policy / mod.RESULT_RELATIVE_PATH,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_v607_literature_delta"
    assert blocked["v607_literature_delta_complete_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == "network_policy"
    assert mod.validate_artifact(blocked) == []

    missing_results = _fake_repo(tmp_path / "missing-results")
    blocked_rows = mod.check_preconditions(
        missing_results,
        "20260903",
        missing_results / "absent" / "result.json",
    )
    assert blocked_rows[-1]["resource"] == "writable_results_path"
    assert blocked_rows[-1]["available"] is False


def test_scenario_report_6927_all_queries_and_candidates_terminate() -> None:
    """SCENARIO-REPORT-6927-TERMINAL: each required recheck has one row."""

    queries = mod.query_rows()
    assert {row["source_family"] for row in queries} == set(mod.SOURCE_FAMILIES)
    assert all(row["terminal"] for row in queries)
    assert all(1 <= row["attempt_count"] <= row["max_attempts"] for row in queries)

    candidates = mod.candidate_rows()
    assert {row["candidate_id"] for row in candidates} == set(mod.NAMED_CANDIDATES)
    for row in candidates:
        assert row["canonical_url"].startswith("https://")
        assert row["date"]
        assert row["evidence_grade"] in {"A", "B"}
        assert row["code_state"]
        assert row["local_gguf_compatibility"] in {
            "compatible",
            "partial",
            "incompatible",
        }
        assert row["disposition"] in {"accepted_existing", "accepted_new", "rejected"}
        assert row["exclusion_reason"]
        assert row["terminal"] is True

    accepted = mod.accepted_finding_rows()
    rejected = mod.rejected_finding_rows()
    assert len(accepted) + len(rejected) == len(candidates)
    assert {row["candidate_id"] for row in accepted + rejected} == set(mod.NAMED_CANDIDATES)


def test_scenario_report_6927_compatibility_keeps_unavailable_work_advisory() -> None:
    """SCENARIO-REPORT-6927-COMPATIBILITY: unavailable methods do not promote."""

    compatibility = {row["candidate_id"]: row for row in mod.compatibility_rows()}
    assert compatibility["ism"]["constraint"] == "hosted_services"
    assert compatibility["hsrm"]["constraint"] == "repository_state"
    assert compatibility["thermalizers"]["constraint"] == "hardware_access"
    assert all(row["effect"] and row["observed"] for row in compatibility.values())

    candidates = {row["candidate_id"]: row for row in mod.candidate_rows()}
    for edge in mod.citation_edge_rows():
        assert edge["canonical_url"].startswith("https://")
        assert edge["source_id"]
        assert edge["target_id"]
    for row in mod.implementation_rows():
        assert row["candidate_id"] in candidates
        assert row["source_url"] == candidates[row["candidate_id"]]["canonical_url"]
        assert row["implementation_boundary"]
    assert all(row["target_milestone"] == "V608+" for row in mod.v608_candidate_rows())


def test_scenario_report_6927_ledger_is_append_only_and_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6927-LEDGER: only verified new rows append once."""

    root = _fake_repo(tmp_path)
    ledger = root / mod.LEDGER_RELATIVE_PATH
    before = ledger.read_bytes()
    artifact = _artifact(tmp_path)
    artifact["ledger_append_rows"] = []
    assert mod.append_verified_findings(root, artifact) == []
    assert ledger.read_bytes() == before

    update = {
        "candidate_id": "verified_new_fact",
        "title": "Verified new primary fact",
        "canonical_url": "https://arxiv.org/abs/2609.99999",
        "date": "2026-09-03",
        "finding": "A primary source changes one local execution boundary.",
        "v608_use": "Test the boundary in V608 without changing V607.",
    }
    artifact["ledger_append_rows"] = [update]
    assert mod.append_verified_findings(root, artifact) == ["verified_new_fact"]
    appended = ledger.read_bytes()
    assert appended.startswith(before.rstrip())
    assert b"Verified new primary fact" in appended
    assert mod.append_verified_findings(root, artifact) == []
    assert ledger.read_bytes() == appended


def test_scenario_report_6927_artifact_recomputes_score_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6927-ARTIFACT: rows independently prove completion."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["v607_literature_delta_complete_score"] == 1
    assert artifact["verdict_class"] in mod.VERDICT_CLASSES
    assert artifact["honest_verdict"].startswith("complete_")
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["query_rows"] = bad["query_rows"][:-1]
    assert "source family coverage is incomplete" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["candidate_rows"][0]["terminal"] = False
    assert "candidate coverage is incomplete" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "wrong"
    assert "reproducibility checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "unexpected"
    assert "complete artifact honest verdict lacks complete_ prefix" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "unspecified"
    bad["verifier_is_oracle"] = True
    bad["verdict_class"] = "unknown"
    bad["field_principles"] = {}
    errors = mod.validate_artifact(_with_checksum(bad))
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "verdict_class is invalid" in errors
    assert "field principles do not cover required fields" in errors

    bad = deepcopy(artifact)
    bad.pop("rows")
    assert mod.validate_artifact(bad) == ["missing required field: rows"]

    blocked = mod.build_artifact(
        root=_fake_repo(tmp_path / "blocked", include_policy=False),
        run_date="20260903",
        duration_s=0.1,
        output_path=tmp_path / "blocked" / "repo" / "results" / "blocked.json",
    )
    for field, value in (
        ("v607_literature_delta_complete_score", 1),
        ("verdict_class", "positive"),
        ("honest_verdict", "blocked_wrong"),
    ):
        bad = deepcopy(blocked)
        bad[field] = value
        assert "blocked artifact fields are inconsistent" in mod.validate_artifact(
            _with_checksum(bad)
        )


def test_req_report_6927_write_validate_and_cli(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-6927: the wrapper writes one stable validated JSON file."""

    monkeypatch.delenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", raising=False)
    artifact = _artifact(tmp_path)
    path = tmp_path / "result.json"
    assert mod.write_artifact(artifact, path) == path
    assert mod.validate_artifact(path) == []
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact missing"]

    cli_root = _fake_repo(tmp_path / "cli")
    monkeypatch.setattr(mod, "find_repo_root", lambda: cli_root)
    cli_path = cli_root / "results" / "cli.json"
    assert mod.main(["--date", "20260903", "--output", str(cli_path)]) == 0
    assert cli_path.exists()
    assert mod.main(["--validate", str(cli_path)]) == 0

    blocked_root = _fake_repo(tmp_path / "blocked-cli", include_policy=False)
    monkeypatch.setattr(mod, "find_repo_root", lambda: blocked_root)
    blocked_path = blocked_root / "results" / "blocked.json"
    assert mod.main(["--date", "20260903", "--output", str(blocked_path)]) == 0
    assert json.loads(blocked_path.read_text(encoding="utf-8"))["status"] == "blocked"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", str(invalid)]) == 1
    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, tmp_path / "never-written.json")

    monkeypatch.setattr(
        sys,
        "argv",
        [mod.MODULE_RELATIVE_PATH.as_posix(), "--validate", str(path)],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(REPO / mod.MODULE_RELATIVE_PATH, run_name="__main__")
