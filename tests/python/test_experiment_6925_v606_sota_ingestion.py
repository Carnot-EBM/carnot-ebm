"""Tests for the V606 execution-time SOTA ingestion.

Spec refs: REQ-REPORT-6925, SCENARIO-REPORT-6925-PREFLIGHT,
SCENARIO-REPORT-6925-PLAN, SCENARIO-REPORT-6925-TERMINAL,
SCENARIO-REPORT-6925-MAP, SCENARIO-REPORT-6925-LEDGER, and
SCENARIO-REPORT-6925-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6925_v606_sota_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _artifact(tmp_path: Path) -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260903",
        duration_s=1.0,
        output_root=tmp_path,
    )


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def _fake_repo(tmp_path: Path, *, include_policy: bool = True) -> Path:
    root = tmp_path / "repo"
    required = {
        "research-roadmap.yaml": "milestone: 2026.09.606\n",
        "research-references.md": "# Reference ledger\n",
        "CLAUDE.md": "## SOTA-Ingestion Cycle Discipline (MANDATORY)\nlow-concurrency\n",
        "openspec/change-proposals/research-roadmap-vNEXT.md": "# V606\n",
    }
    if not include_policy:
        required.pop("CLAUDE.md")
    for relative, text in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_req_report_6925_spec_declares_bounded_advisory_contract() -> None:
    """REQ-REPORT-6925: OpenSpec owns the complete ingestion contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6925") :]
    for marker in (
        "SCENARIO-REPORT-6925-PREFLIGHT",
        "SCENARIO-REPORT-6925-PLAN",
        "SCENARIO-REPORT-6925-TERMINAL",
        "SCENARIO-REPORT-6925-MAP",
        "SCENARIO-REPORT-6925-LEDGER",
        "SCENARIO-REPORT-6925-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6925_plan_is_dated_exact_and_arxiv_first() -> None:
    """SCENARIO-REPORT-6925-PLAN: the plan freezes bounded query order."""

    plan = mod.frozen_query_plan()
    assert mod.validate_query_plan(plan, "20260903") == []
    assert len(plan) >= len(mod.SOURCE_FAMILIES)
    assert [row["source_family"] for row in plan[:8]] == list(mod.ARXIV_FAMILIES)
    assert len({row["query_id"] for row in plan}) == len(plan)
    assert all(row["planned_utc"].startswith("2026-09-03T") for row in plan)
    assert all(row["max_attempts"] == 2 for row in plan)
    assert all(row["concurrency_limit"] == 1 for row in plan)
    assert all(row["timeout_s"] == 20 for row in plan)
    assert all(row["query_text"] and row["allowed_domains"] for row in plan)
    assert all(row["acceptance_criteria"] for row in plan)

    bad = deepcopy(plan)
    bad[0]["max_attempts"] = 3
    assert "query arxiv_ebm must allow exactly two attempts" in mod.validate_query_plan(
        bad, "20260903"
    )
    assert mod.validate_query_plan(plan, "20260904") == ["query plan date does not match run date"]

    malformed = deepcopy(plan)
    malformed[0]["concurrency_limit"] = 2
    malformed[0]["timeout_s"] = 21
    malformed[0]["query_text"] = ""
    malformed[0], malformed[8] = malformed[8], malformed[0]
    malformed.pop()
    errors = mod.validate_query_plan(malformed, "20260903")
    assert "query arxiv_ebm must use concurrency one" in errors
    assert "query arxiv_ebm must use the 20 second timeout" in errors
    assert "query arxiv_ebm lacks a frozen source field" in errors
    assert "arXiv source families must be first" in errors
    assert "query plan source family coverage is incomplete" in errors


def test_scenario_report_6925_preflight_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6925-PREFLIGHT: missing policy blocks before research."""

    root = _fake_repo(tmp_path)
    rows = mod.check_preconditions(root, "20260903")
    assert all(row["available"] for row in rows)

    missing_policy = _fake_repo(tmp_path / "missing", include_policy=False)
    blocked = mod.build_artifact(
        root=missing_policy,
        run_date="20260903",
        duration_s=0.1,
        output_root=tmp_path,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "complete_blocked_v606_sota_ingestion"
    assert blocked["v606_sota_ingestion_complete_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == "network_policy"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_6925_candidates_have_terminal_evidence_rows() -> None:
    """SCENARIO-REPORT-6925-TERMINAL: every named method terminates."""

    candidates = mod.candidate_rows()
    assert set(mod.NAMED_CANDIDATES) == {row["candidate_id"] for row in candidates}
    for row in candidates:
        assert row["canonical_url"].startswith("https://")
        assert row["date"]
        assert row["source_type"] in {"paper", "first_party_repository"}
        assert row["relevance"]
        assert row["evidence_grade"] in {"A", "B"}
        assert isinstance(row["code_available"], bool)
        assert row["local_compatibility"] in {"compatible", "partial", "incompatible"}
        assert row["disposition"] in {"accepted_existing", "accepted_new", "rejected"}
        assert row["exclusion_reason"]
        assert row["terminal"] is True

    queries = mod.query_rows()
    assert {row["source_family"] for row in queries} == set(mod.SOURCE_FAMILIES)
    assert all(row["terminal"] for row in queries)
    assert all(1 <= row["attempt_count"] <= row["max_attempts"] for row in queries)
    assert all(row["canonical_url"].startswith("https://") for row in queries)


def test_scenario_report_6925_method_map_cites_v606_and_incompatibilities() -> None:
    """SCENARIO-REPORT-6925-MAP: checked methods map without changing V606."""

    candidates = {row["candidate_id"]: row for row in mod.candidate_rows()}
    mappings = mod.implementation_rows()
    assert {row["method_family"] for row in mappings} == {
        "grounded_acquisition",
        "continual_memory",
        "verified_variant",
        "sequential_monte_carlo",
        "hardware_sampling",
    }
    for row in mappings:
        assert row["candidate_id"] in candidates
        assert row["source_url"] == candidates[row["candidate_id"]]["canonical_url"]
        assert row["v606_experiment"].startswith("Exp69")
        assert row["reusable_method"]
        assert row["implementation_boundary"]

    edges = mod.citation_edge_rows()
    assert len(edges) >= len(mappings)
    assert all(edge["canonical_url"].startswith("https://") for edge in edges)
    assert mod.compatibility_rows()
    assert all(row["constraint"] and row["effect"] for row in mod.compatibility_rows())
    assert all(row["target_milestone"] == "V607+" for row in mod.v607_candidate_rows())


def test_scenario_report_6925_ledger_is_append_only_and_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6925-LEDGER: only accepted-new rows append once."""

    root = _fake_repo(tmp_path)
    ledger = root / mod.LEDGER_RELATIVE_PATH
    before = ledger.read_bytes()
    no_updates = _artifact(tmp_path)
    no_updates["ledger_append_rows"] = []
    assert mod.append_verified_findings(root, no_updates) == []
    assert ledger.read_bytes() == before

    update = {
        "candidate_id": "new_primary_finding",
        "title": "A verified new primary finding",
        "canonical_url": "https://arxiv.org/abs/2609.99999",
        "date": "2026-09-03",
        "finding": "This source changes one future executable contract.",
        "v607_use": "Test the changed contract in V607.",
    }
    no_updates["ledger_append_rows"] = [update]
    appended = mod.append_verified_findings(root, no_updates)
    assert appended == ["new_primary_finding"]
    text = ledger.read_text(encoding="utf-8")
    assert "A verified new primary finding" in text
    assert mod.append_verified_findings(root, no_updates) == []
    assert ledger.read_text(encoding="utf-8") == text


def test_scenario_report_6925_artifact_recomputes_score_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6925-ARTIFACT: rows support independent validation."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["v606_sota_ingestion_complete_score"] == 1
    assert artifact["verdict_class"] in {"null", "positive"}
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
    bad["inference_substrate"] = "unspecified"
    bad["verifier_is_oracle"] = True
    bad["field_principles"] = {}
    errors = mod.validate_artifact(_with_checksum(bad))
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "field principles do not cover required fields" in errors

    blocked_root = _fake_repo(tmp_path / "blocked", include_policy=False)
    bad = mod.build_artifact(root=blocked_root, run_date="20260903", duration_s=0.1)
    bad["v606_sota_ingestion_complete_score"] = 1
    assert "blocked artifact fields are inconsistent" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad.pop("rows")
    assert mod.validate_artifact(bad) == ["missing required field: rows"]


def test_scenario_report_6925_write_validate_and_cli(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-6925: the wrapper writes one stable, validated JSON file."""

    monkeypatch.delenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", raising=False)
    artifact = _artifact(tmp_path)
    path = tmp_path / "result.json"
    assert mod.write_artifact(artifact, path) == path
    assert mod.validate_artifact(path) == []
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
    assert mod.validate_artifact(tmp_path / "missing.json") == ["artifact missing"]

    cli_root = _fake_repo(tmp_path / "cli")
    monkeypatch.setattr(mod, "find_repo_root", lambda: cli_root)
    cli_path = tmp_path / "cli.json"
    assert mod.main(["--date", "20260903", "--output", str(cli_path)]) == 0
    assert cli_path.exists()
    assert mod.main(["--validate", str(cli_path)]) == 0

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", str(invalid)]) == 1
    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, tmp_path / "never-written.json")

    blocked_root = _fake_repo(tmp_path / "blocked-cli", include_policy=False)
    monkeypatch.setattr(mod, "find_repo_root", lambda: blocked_root)
    blocked_path = tmp_path / "blocked-cli.json"
    assert mod.main(["--date", "20260903", "--output", str(blocked_path)]) == 0
    assert json.loads(blocked_path.read_text(encoding="utf-8"))["status"] == "blocked"

    monkeypatch.setattr(
        sys,
        "argv",
        [mod.MODULE_RELATIVE_PATH.as_posix(), "--validate", str(path)],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(REPO / mod.MODULE_RELATIVE_PATH, run_name="__main__")
