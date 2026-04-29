"""Tests for the conductor deliverable-exists fast-path status check.

Spec: REQ-INFRA-080, SCENARIO-INFRA-080-A through SCENARIO-INFRA-080-E

Background: the original `_deliverable_exists` returned True for any file at
the deliverable path, which caused milestone .80 (and earlier) to wedge:
Sonnet's "CRITICAL write artifact FIRST" pattern landed bootstrap-only
artifacts with `status: "running"` and `pre_test_fixed: False`, and the
fast-path then short-circuited every retry. Downstream gated tasks read
`False` forever. This test guards the fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import research_conductor  # noqa: E402


def _make_task(deliverable: str | None) -> dict:
    return {"id": "test-exp", "deliverable": deliverable}


def test_no_deliverable_field_returns_false():
    """SCENARIO-INFRA-080-A: tasks without a deliverable key are never 'done'."""
    assert research_conductor._deliverable_exists(_make_task(None)) is False


def test_missing_file_returns_false(tmp_path, monkeypatch):
    """SCENARIO-INFRA-080-B: deliverable path that doesn't exist returns False."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    assert research_conductor._deliverable_exists(_make_task("results/nope.json")) is False


def test_bootstrap_running_status_not_skipped(tmp_path, monkeypatch):
    """SCENARIO-INFRA-080-C: status='running' is bootstrap-only, must not skip."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"experiment": 1, "status": "running"}))
    assert research_conductor._deliverable_exists(_make_task("results/exp.json")) is False


@pytest.mark.parametrize("status", ["blocked", "partial", "in_progress", "RUNNING", "Blocked"])
def test_other_bootstrap_statuses_not_skipped(tmp_path, monkeypatch, status):
    """SCENARIO-INFRA-080-D: blocked/partial/in_progress (any case) not skipped."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"status": status}))
    assert research_conductor._deliverable_exists(_make_task("results/exp.json")) is False


def test_success_status_is_skipped(tmp_path, monkeypatch):
    """SCENARIO-INFRA-080-E: status='success' is genuinely done, fast-path skips."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"status": "success", "result": 42}))
    assert research_conductor._deliverable_exists(_make_task("results/exp.json")) is True


def test_legacy_no_status_field_is_skipped(tmp_path, monkeypatch):
    """Legacy artifacts without status field preserve old behaviour (skip = True)."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"result": "old artifact"}))
    assert research_conductor._deliverable_exists(_make_task("results/exp.json")) is True


def test_non_json_deliverable_is_skipped(tmp_path, monkeypatch):
    """Non-JSON deliverables (.md, .txt, .pdf) preserve old behaviour."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "docs" / "out.md"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text("# Some markdown")
    assert research_conductor._deliverable_exists(_make_task("docs/out.md")) is True


def test_malformed_json_falls_back_to_skip(tmp_path, monkeypatch):
    """Malformed JSON should not crash the fast-path; preserve old behaviour."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text("{not valid json")
    assert research_conductor._deliverable_exists(_make_task("results/exp.json")) is True


# ---------------------------------------------------------------------------
# _artifact_is_finished — Signal 1 re-validation guard
# ---------------------------------------------------------------------------
#
# A prior log "OK" can be poisoned: the conductor's pytest self-heal passed
# (logged OK) but Sonnet hit max-turns before updating the artifact's status
# from "running" to "success". The fix is for `pick_next_task` to consult
# `_artifact_is_finished` after Signal 1 to detect the poisoned-OK case.


def test_artifact_finished_no_deliverable():
    """Tasks without a deliverable trivially trust the log OK."""
    assert research_conductor._artifact_is_finished({"id": "plan"}) is True


def test_artifact_finished_missing_file(tmp_path, monkeypatch):
    """Tasks with a deliverable path that doesn't exist yet trust the log OK
    (the OK may be a planning step that doesn't write the artifact yet)."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    assert research_conductor._artifact_is_finished(_make_task("results/x.json")) is True


def test_artifact_finished_running_status_poisoned(tmp_path, monkeypatch):
    """The poisoned-OK case: artifact has status=running. Must invalidate."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"status": "running"}))
    assert research_conductor._artifact_is_finished(_make_task("results/exp.json")) is False


@pytest.mark.parametrize("status", ["blocked", "partial", "in_progress"])
def test_artifact_finished_other_bootstrap_statuses_poisoned(tmp_path, monkeypatch, status):
    """All bootstrap statuses invalidate a prior log OK."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"status": status}))
    assert research_conductor._artifact_is_finished(_make_task("results/exp.json")) is False


def test_artifact_finished_success_status_trusts_ok(tmp_path, monkeypatch):
    """status=success means the OK is real."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"status": "success", "result": 42}))
    assert research_conductor._artifact_is_finished(_make_task("results/exp.json")) is True


def test_artifact_finished_no_status_trusts_ok(tmp_path, monkeypatch):
    """Legacy artifacts (no status field) trust the log OK."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text(json.dumps({"data": "old"}))
    assert research_conductor._artifact_is_finished(_make_task("results/exp.json")) is True
