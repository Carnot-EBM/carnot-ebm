"""Tests for the V608 post-marker source delta.

Spec refs: REQ-REPORT-6941, SCENARIO-REPORT-6941-PREFLIGHT,
SCENARIO-REPORT-6941-PLAN, SCENARIO-REPORT-6941-TERMINAL,
SCENARIO-REPORT-6941-COMPATIBILITY, SCENARIO-REPORT-6941-LEDGER, and
SCENARIO-REPORT-6941-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6941_v608_source_delta as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _fake_repo(tmp_path: Path, *, missing: str | None = None) -> Path:
    root = tmp_path / "repo"
    required = {
        "research-references.md": (
            "# Reference ledger\n\n"
            "<!-- V608-PLANNER-REFRESH-20260903-START -->\n"
            "planner facts\n"
            "<!-- V608-PLANNER-REFRESH-20260903-END -->\n"
        ),
        "research-roadmap.yaml": (
            "milestone: 2026.09.608\ntasks:\n- id: exp6941-v608-source-delta\n"
        ),
        "openspec/change-proposals/research-roadmap-vNEXT.md": (
            "**Milestone:** 2026.09.608\n### Exp6941\n"
        ),
        "CLAUDE.md": (
            "## SOTA-Ingestion Cycle Discipline (MANDATORY)\nUse the low-concurrency channel.\n"
        ),
    }
    if missing:
        required.pop(missing)
    for relative, content in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
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


def test_req_report_6941_spec_declares_complete_advisory_contract() -> None:
    """REQ-REPORT-6941: OpenSpec owns the complete source-delta contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6941") :]
    for marker in (
        "SCENARIO-REPORT-6941-PREFLIGHT",
        "SCENARIO-REPORT-6941-PLAN",
        "SCENARIO-REPORT-6941-TERMINAL",
        "SCENARIO-REPORT-6941-COMPATIBILITY",
        "SCENARIO-REPORT-6941-LEDGER",
        "SCENARIO-REPORT-6941-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6941_plan_is_dated_bounded_and_arxiv_first() -> None:
    """SCENARIO-REPORT-6941-PLAN: the plan fixes bounded query order."""

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
    malformed[0]["allowed_domains"] = []
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


def test_scenario_report_6941_preflight_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6941-PREFLIGHT: any missing input blocks research."""

    root = _fake_repo(tmp_path)
    output = root / mod.RESULT_RELATIVE_PATH
    assert all(row["available"] for row in mod.check_preconditions(root, "20260903", output))

    missing_design = _fake_repo(
        tmp_path / "missing",
        missing="openspec/change-proposals/research-roadmap-vNEXT.md",
    )
    blocked = mod.build_artifact(
        root=missing_design,
        run_date="20260903",
        duration_s=0.1,
        output_path=missing_design / mod.RESULT_RELATIVE_PATH,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_v608_source_delta"
    assert blocked["v608_source_delta_complete_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == "design_document"
    assert mod.validate_artifact(blocked) == []

    missing_results = _fake_repo(tmp_path / "missing-results")
    blocked_rows = mod.check_preconditions(
        missing_results,
        "20260903",
        missing_results / "absent" / "result.json",
    )
    assert blocked_rows[-1]["resource"] == "writable_results_path"
    assert blocked_rows[-1]["available"] is False


def test_scenario_report_6941_all_sources_and_selected_papers_terminate() -> None:
    """SCENARIO-REPORT-6941-TERMINAL: every required recheck has one row."""

    queries = mod.query_rows()
    assert {row["source_family"] for row in queries} == set(mod.SOURCE_FAMILIES)
    assert all(row["terminal"] for row in queries)
    assert all(1 <= row["attempt_count"] <= row["max_attempts"] for row in queries)

    candidates = mod.candidate_rows()
    assert {row["arxiv_id"] for row in candidates} == set(mod.SELECTED_ARXIV_IDS)
    assert all(row["terminal"] is True for row in candidates)
    for row in candidates:
        assert row["canonical_url"] == f"https://arxiv.org/abs/{row['arxiv_id']}"
        assert row["title"] and row["version"] and row["date"]
        assert row["evidence_grade"] in {"A", "B"}
        assert row["code_state"]
        assert row["local_compatibility"] in {"compatible", "partial", "incompatible"}
        assert row["disposition"] in {"accepted_existing", "accepted_new", "rejected"}
        assert row["exclusion_reason"]

    accepted = mod.accepted_finding_rows()
    rejected = mod.rejected_finding_rows()
    assert len(accepted) + len(rejected) == len(candidates)
    assert {row["arxiv_id"] for row in accepted + rejected} == set(mod.SELECTED_ARXIV_IDS)


def test_scenario_report_6941_compatibility_keeps_sources_advisory() -> None:
    """SCENARIO-REPORT-6941-COMPATIBILITY: source checks cannot add gates."""

    candidates = {row["arxiv_id"]: row for row in mod.candidate_rows()}
    compatibility = {row["arxiv_id"]: row for row in mod.compatibility_rows()}
    implementation = {row["arxiv_id"]: row for row in mod.implementation_rows()}
    assert set(compatibility) == set(candidates)
    assert set(implementation) == set(candidates)
    assert all(row["observed"] and row["effect"] for row in compatibility.values())
    assert all(row["advisory_only"] is True for row in implementation.values())
    assert all(
        row["source_url"] == candidates[key]["canonical_url"] for key, row in implementation.items()
    )
    assert all(row["target_milestone"] == "V609+" for row in mod.v609_candidate_rows())
    for edge in mod.citation_edge_rows():
        assert edge["canonical_url"].startswith("https://")
        assert edge["source_id"] and edge["target_id"] and edge["terminal"] is True


def test_scenario_report_6941_ledger_is_post_marker_and_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6941-LEDGER: only verified post-marker facts append."""

    root = _fake_repo(tmp_path)
    ledger = root / mod.LEDGER_RELATIVE_PATH
    artifact = _artifact(tmp_path)
    artifact["ledger_append_rows"] = [
        {
            "finding_id": "older_fact",
            "title": "Older fact",
            "canonical_url": "https://arxiv.org/abs/2609.99990",
            "observed_utc": "2026-09-03T16:22:59Z",
            "evidence_grade": "A",
            "source_type": "primary",
            "verified": True,
            "finding": "This fact predates the marker.",
            "v609_use": "None.",
        },
        {
            "finding_id": "unverified_fact",
            "title": "Unverified fact",
            "canonical_url": "https://example.com/claim",
            "observed_utc": "2026-09-03T16:24:00Z",
            "evidence_grade": "C",
            "source_type": "announcement",
            "verified": False,
            "finding": "This claim lacks primary evidence.",
            "v609_use": "None.",
        },
        {
            "finding_id": "verified_new_fact",
            "title": "Verified new primary fact",
            "canonical_url": "https://arxiv.org/abs/2609.99999",
            "observed_utc": "2026-09-03T16:24:00Z",
            "evidence_grade": "A",
            "source_type": "primary",
            "verified": True,
            "finding": "A primary source changes one local execution boundary.",
            "v609_use": "Test the boundary after V608 without changing V608.",
        },
    ]
    before = ledger.read_bytes()
    assert mod.append_verified_findings(root, artifact) == ["verified_new_fact"]
    appended = ledger.read_bytes()
    assert appended.startswith(before.rstrip())
    assert b"Verified new primary fact" in appended
    assert b"Older fact" not in appended
    assert b"Unverified fact" not in appended
    assert mod.append_verified_findings(root, artifact) == []
    assert ledger.read_bytes() == appended

    artifact["ledger_append_rows"] = []
    assert mod.append_verified_findings(root, artifact) == []
    assert ledger.read_bytes() == appended


def test_scenario_report_6941_artifact_recomputes_score_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6941-ARTIFACT: rows independently prove completion."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["v608_source_delta_complete_score"] == 1
    assert artifact["verdict_class"] in mod.VERDICT_CLASSES
    assert str(artifact["honest_verdict"]).startswith("complete_")
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["query_rows"] = bad["query_rows"][:-1]
    assert "source family coverage is incomplete" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["candidate_rows"][0]["terminal"] = False
    assert "selected paper coverage is incomplete" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "wrong"
    assert "reproducibility checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "unexpected"
    assert "verdict prefix is inconsistent" in mod.validate_artifact(_with_checksum(bad))

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
        root=_fake_repo(tmp_path / "blocked", missing="CLAUDE.md"),
        run_date="20260903",
        duration_s=0.1,
        output_path=tmp_path / "blocked" / "repo" / "results" / "blocked.json",
    )
    for field, value in (
        ("v608_source_delta_complete_score", 1),
        ("verdict_class", "positive"),
        ("honest_verdict", "blocked_wrong"),
    ):
        bad = deepcopy(blocked)
        bad[field] = value
        assert "blocked artifact fields are inconsistent" in mod.validate_artifact(
            _with_checksum(bad)
        )


def test_req_report_6941_write_validate_and_cli(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-6941: the wrapper writes one stable validated JSON file."""

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

    blocked_root = _fake_repo(tmp_path / "blocked-cli", missing="CLAUDE.md")
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
