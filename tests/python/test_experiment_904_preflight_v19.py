"""Tests for Exp 904: Pre-flight v19 — Milestone 2026.04.70 Gate Audit.

Spec: REQ-INFRA-072, REQ-INFRA-073, SCENARIO-INFRA-072

WHY THESE TESTS:
    REQ-INFRA-072 requires that the exclusion manifest is consulted before any
    experiment runs and that RETRO items are formally recorded with statuses.
    These tests verify the pre-flight logic (root-cause extraction, retro auditing,
    escalation writing, prereqs update) without hitting the real filesystem
    unnecessarily, so they run fast in CI without GPU or heavy dependencies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from experiment_904_preflight_v19 import (  # noqa: E402
    CONDUCTOR_LOG_PATTERN,
    ESCALATION_BLOCK,
    EXPECTED_ROOT_CAUSE,
    OPEN_RETROS,
    PREREQS_SECTION,
    RETRO_STATUSES,
    _escalate_known_issues,
    _extract_root_cause,
    _load_retro,
    _update_prereqs,
    run_preflight,
)


# ---------------------------------------------------------------------------
# REQ-INFRA-072 / SCENARIO-INFRA-072: root-cause extraction
# ---------------------------------------------------------------------------


class TestExtractRootCause:
    """SCENARIO-INFRA-072: root-cause detector finds the yaml_key_error_title line."""

    def test_finds_title_key_error(self, tmp_path):
        """The conductor log line 'Failed to load research-roadmap.yaml: 'title'' maps
        to EXPECTED_ROOT_CAUSE.

        WHY: The .69 milestone ran zero experiments because of exactly this error.
        The pre-flight must confirm it, not guess.
        """
        log = tmp_path / "conductor.log"
        log.write_text(
            "2026-04-25 21:56:18 [conductor] Failed to load research-roadmap.yaml: 'title'\n"
        )
        assert _extract_root_cause(log) == "yaml_key_error_title"

    def test_returns_not_found_when_error_absent(self, tmp_path):
        """When the log has no title-key error, returns 'not_found'.

        WHY: We must not falsely claim the root cause is known when the log
        does not contain the expected pattern (e.g., a different conductor run).
        """
        log = tmp_path / "conductor.log"
        log.write_text("2026-04-25 [conductor] Everything OK\n")
        assert _extract_root_cause(log) == "not_found"

    def test_returns_log_not_found_when_file_missing(self, tmp_path):
        """Missing log file returns 'log_not_found' instead of crashing.

        WHY: CI environments may not have the full logs/ directory.
        """
        assert _extract_root_cause(tmp_path / "nonexistent.log") == "log_not_found"

    def test_pattern_matches_exact_conductor_line(self):
        """The compiled regex matches the exact conductor log format.

        WHY: A typo in the pattern would cause all root-cause checks to fail silently.
        """
        line = "2026-04-25 21:56:18,044 [conductor] Failed to load research-roadmap.yaml: 'title'"
        assert CONDUCTOR_LOG_PATTERN.search(line) is not None

    def test_pattern_does_not_match_unrelated_lines(self):
        """The regex does not fire on unrelated log lines.

        WHY: We must not produce false positives — other load errors exist.
        """
        assert CONDUCTOR_LOG_PATTERN.search("Failed to load config.yaml: 'version'") is None


# ---------------------------------------------------------------------------
# REQ-INFRA-072: retro loading
# ---------------------------------------------------------------------------


class TestLoadRetro:
    """Tests for _load_retro helper."""

    def test_loads_valid_json(self, tmp_path):
        """Valid JSON file is parsed correctly."""
        p = tmp_path / "retro.json"
        p.write_text(json.dumps({"experiments_in_milestone": 11, "wall_time_minutes": 13.9}))
        d = _load_retro(p)
        assert d["experiments_in_milestone"] == 11

    def test_returns_empty_dict_when_missing(self, tmp_path):
        """Missing retro file returns an empty dict (not an exception).

        WHY: The retro may not exist in fresh environments; the pre-flight
        must still complete and write its own artifact.
        """
        assert _load_retro(tmp_path / "missing.json") == {}


# ---------------------------------------------------------------------------
# REQ-INFRA-073: escalation to known-issues.md
# ---------------------------------------------------------------------------


class TestEscalateKnownIssues:
    """SCENARIO-INFRA-072: RETRO-MANIFEST-FULL-SCOPE block is appended idempotently."""

    def test_appends_block_to_existing_file(self, tmp_path):
        """The escalation block is appended after existing content.

        WHY: ops/known-issues.md has prior entries; we must not overwrite them
        (CLAUDE.md rule: never remove existing content).
        """
        ki = tmp_path / "known-issues.md"
        ki.write_text("# Known Issues\n\nExisting entry.\n")
        _escalate_known_issues(ki)
        content = ki.read_text()
        assert "RETRO-MANIFEST-FULL-SCOPE: CRITICAL" in content
        assert "Existing entry." in content

    def test_idempotent_when_block_already_present(self, tmp_path):
        """Running twice does not duplicate the block.

        WHY: The conductor may run the pre-flight multiple times if a checkpoint
        is resumed; duplicate escalation blocks would be confusing.
        """
        ki = tmp_path / "known-issues.md"
        ki.write_text("")
        _escalate_known_issues(ki)
        size_after_first = len(ki.read_text())
        _escalate_known_issues(ki)
        assert len(ki.read_text()) == size_after_first

    def test_returns_true_on_success(self, tmp_path):
        """Return value is True when escalation succeeds."""
        ki = tmp_path / "known-issues.md"
        ki.write_text("")
        assert _escalate_known_issues(ki) is True

    def test_block_contains_enforcement_wired_false(self, tmp_path):
        """The escalation block explicitly states enforcement_wired: false.

        WHY: Downstream tools and humans must know the wiring status without
        having to parse the full artifact.
        """
        ki = tmp_path / "known-issues.md"
        ki.write_text("")
        _escalate_known_issues(ki)
        assert "enforcement_wired: false" in ki.read_text()


# ---------------------------------------------------------------------------
# REQ-INFRA-073: MILESTONE_PREREQS.md update
# ---------------------------------------------------------------------------


class TestUpdatePrereqs:
    """Tests for _update_prereqs helper."""

    def test_adds_70_section(self, tmp_path):
        """The .70 pre-flight section is appended to the prereqs file.

        WHY: The conductor reads MILESTONE_PREREQS.md to know which gate
        conditions block each experiment.
        """
        p = tmp_path / "MILESTONE_PREREQS.md"
        p.write_text("# Prerequisites\n")
        _update_prereqs(p)
        content = p.read_text()
        assert "Milestone 2026.04.70 Pre-flight" in content

    def test_idempotent(self, tmp_path):
        """Adding the section twice does not duplicate it."""
        p = tmp_path / "MILESTONE_PREREQS.md"
        p.write_text("")
        _update_prereqs(p)
        first = p.read_text()
        _update_prereqs(p)
        assert p.read_text() == first

    def test_gate_for_exp906_present(self, tmp_path):
        """The gate condition for Exp 906 (signed_improvement > 0) is present.

        WHY: Exp 906 is gated on Exp 905 results; the gate must be machine-readable.
        """
        p = tmp_path / "MILESTONE_PREREQS.md"
        p.write_text("")
        _update_prereqs(p)
        assert "signed_improvement > 0" in p.read_text()

    def test_gate_for_exp908_present(self, tmp_path):
        """The gate condition for Exp 908 (labeling_mismatch_confirmed) is present."""
        p = tmp_path / "MILESTONE_PREREQS.md"
        p.write_text("")
        _update_prereqs(p)
        assert "labeling_mismatch_confirmed == True" in p.read_text()

    def test_abort_condition_for_exp914_present(self, tmp_path):
        """The abort condition for Exp 914 (exclusion manifest check) is present."""
        p = tmp_path / "MILESTONE_PREREQS.md"
        p.write_text("")
        _update_prereqs(p)
        assert "iCE40 PIMI research" in p.read_text()


# ---------------------------------------------------------------------------
# REQ-INFRA-072: retro status constants
# ---------------------------------------------------------------------------


class TestRetroConstants:
    """Verify the RETRO registry has the correct entries and statuses."""

    def test_all_four_retros_present(self):
        """OPEN_RETROS contains all four expected identifiers.

        WHY: Missing an entry would cause the artifact to under-report open work.
        """
        expected = {
            "RETRO-MANIFEST-FULL-SCOPE",
            "RETRO-SVAMP-ZERO-AUC",
            "RETRO-XILINX-TOOLS-UNAVAILABLE",
            "RETRO-INERTIA-SWEEPS-TARGET-MISSED",
        }
        assert set(OPEN_RETROS) == expected

    def test_manifest_retro_is_human_required(self):
        """RETRO-MANIFEST-FULL-SCOPE status is HUMAN_REQUIRED.

        WHY: It cannot be fixed programmatically; a human must grant permission
        to modify scripts/research_conductor.py.
        """
        assert RETRO_STATUSES["RETRO-MANIFEST-FULL-SCOPE"] == "HUMAN_REQUIRED"

    def test_xilinx_retro_is_human_required(self):
        """RETRO-XILINX-TOOLS-UNAVAILABLE status is HUMAN_REQUIRED (Vivado install)."""
        assert RETRO_STATUSES["RETRO-XILINX-TOOLS-UNAVAILABLE"] == "HUMAN_REQUIRED"

    def test_svamp_retro_is_targeted(self):
        """RETRO-SVAMP-ZERO-AUC is TARGETED — addressed by Exp 907+908."""
        assert RETRO_STATUSES["RETRO-SVAMP-ZERO-AUC"] == "TARGETED"

    def test_inertia_retro_is_targeted(self):
        """RETRO-INERTIA-SWEEPS-TARGET-MISSED is TARGETED — addressed by Exp 914."""
        assert RETRO_STATUSES["RETRO-INERTIA-SWEEPS-TARGET-MISSED"] == "TARGETED"


# ---------------------------------------------------------------------------
# Integration: run_preflight output shape
# ---------------------------------------------------------------------------


class TestRunPreflightOutput:
    """Tests that run_preflight returns a well-formed payload."""

    def _make_env(self, tmp_path):
        """Create a minimal repo structure for run_preflight."""
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "conductor.log").write_text(
            "2026-04-25 21:56:18 [conductor] Failed to load research-roadmap.yaml: 'title'\n"
        )
        (tmp_path / "results").mkdir()
        retro = {
            "milestone": "2026.04.69",
            "wall_time_minutes": 13.9255,
            "experiments_in_milestone": 11,
        }
        (tmp_path / "results" / "operational_retro_2026_04_69.json").write_text(json.dumps(retro))
        (tmp_path / "ops").mkdir()
        (tmp_path / "ops" / "known-issues.md").write_text("# Known Issues\n")
        return tmp_path

    def test_zero_run_root_cause_correct(self, tmp_path):
        """run_preflight sets zero_run_root_cause='yaml_key_error_title'."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["zero_run_root_cause"] == "yaml_key_error_title"

    def test_n_exps_run_is_zero(self, tmp_path):
        """n_exps_run_in_69 is 0 (the planned .69 experiments never ran).

        WHY: Even though the retro records 11 experiments (from .68 roadmap
        carry-over), the PLANNED .69 experiments were zero due to the yaml error.
        """
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["n_exps_run_in_69"] == 0

    def test_enforcement_wired_false(self, tmp_path):
        """enforcement_wired is False — the wiring change requires human approval."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["enforcement_wired"] is False

    def test_honest_verdict_preflight_complete(self, tmp_path):
        """honest_verdict is 'preflight_complete' on a clean run."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["honest_verdict"] == "preflight_complete"

    def test_escalation_written_true(self, tmp_path):
        """escalation_written is True when the known-issues file is updated."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["escalation_written"] is True

    def test_open_retros_all_present(self, tmp_path):
        """All four open retros appear in the output."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert set(result["open_retros"]) == set(OPEN_RETROS)

    def test_retro_statuses_populated(self, tmp_path):
        """retro_statuses dict has an entry for every open retro."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        for retro in OPEN_RETROS:
            assert retro in result["retro_statuses"]

    def test_milestone_field(self, tmp_path):
        """milestone is '2026.04.70'."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["milestone"] == "2026.04.70"

    def test_preflight_version(self, tmp_path):
        """preflight_version is 19."""
        env = self._make_env(tmp_path)
        result = run_preflight(env)
        assert result["preflight_version"] == 19
