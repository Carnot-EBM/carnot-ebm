"""Tests for Experiment 831: Governance pre-flight — RETRO closure audit.

Traces to:
    REQ-INFRA-063: Governance pre-flight MUST audit RETRO closure status against
        actual experiment result JSONs before any new milestone experiments begin.
    SCENARIO-INFRA-071: Exp N+1 RETRO listed as open; Exp N result JSON shows
        closure field=True; governance pre-flight corrects status to CLOSED in
        MILESTONE_PREREQS.md.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.experiment_831_governance_preflight import (
    EXPERIMENT_CAP,
    _RETRO_GGUF,
    _RETRO_ISING,
    _load_json,
    audit_retro_closures,
    run_audit,
    update_milestone_prereqs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def exp819_closed() -> dict:
    """Minimal Exp 819 artifact where RETRO-ISING is confirmed closed."""
    # REQ-INFRA-063: the closure field name is retro_injection_closed.
    return {"retro_injection_closed": True}


@pytest.fixture()
def exp819_open() -> dict:
    """Minimal Exp 819 artifact where RETRO-ISING is NOT closed."""
    return {"retro_injection_closed": False}


@pytest.fixture()
def exp820_closed() -> dict:
    """Minimal Exp 820 artifact where RETRO-GGUF is confirmed closed."""
    return {"honest_verdict": "import_fixed_repair_positive"}


@pytest.fixture()
def exp820_open() -> dict:
    """Minimal Exp 820 artifact where RETRO-GGUF is NOT closed."""
    return {"honest_verdict": "import_blocked"}


@pytest.fixture()
def exp830_with_both_open() -> dict:
    """Minimal Exp 830 retro listing both target RETROs as still-open.

    This mirrors the actual reporting-lag scenario that Exp 831 is designed to fix:
    the retro was written before the closure experiments completed, so both RETROs
    appear in retros_still_open even though they were actually closed.
    """
    return {
        "experiments_completed": 728,
        "retros_still_open": [
            {"id": "RETRO-MANIFEST-FULL-SCOPE", "status": "open"},
            {"id": "RETRO-ISING-INJECTION-NO-DISCRIMINATION", "status": "open (lag)"},
            {"id": "RETRO-GGUF-CACHE-IMPORT", "status": "open (lag)"},
            {"id": "RETRO-ARBITER-FLAT-ENERGY", "status": "open"},
        ],
    }


@pytest.fixture()
def exp830_minimal() -> dict:
    """Exp 830 retro with no retros_still_open (edge case)."""
    return {"experiments_completed": 700, "retros_still_open": []}


# ── Tests for audit_retro_closures() ─────────────────────────────────────────

class TestAuditRetroClosures:
    """REQ-INFRA-063: audit reads retro_injection_closed and honest_verdict correctly."""

    def test_both_retros_confirmed_closed(
        self, exp819_closed, exp820_closed, exp830_with_both_open
    ):
        """SCENARIO-INFRA-071: both RETROs listed as open in retro; result JSONs confirm closed."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_with_both_open)

        # Both should appear in confirmed_closed.
        assert _RETRO_ISING in result["retros_confirmed_closed"]
        assert _RETRO_GGUF in result["retros_confirmed_closed"]

    def test_corrected_open_retros_excludes_confirmed_closed(
        self, exp819_closed, exp820_closed, exp830_with_both_open
    ):
        """SCENARIO-INFRA-071: corrected_open_retros must not include the closed RETROs."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_with_both_open)

        assert _RETRO_ISING not in result["corrected_open_retros"]
        assert _RETRO_GGUF not in result["corrected_open_retros"]

    def test_corrected_open_retros_preserves_genuinely_open(
        self, exp819_closed, exp820_closed, exp830_with_both_open
    ):
        """Retros not mentioned in the closure experiments must remain in corrected list."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_with_both_open)

        assert "RETRO-MANIFEST-FULL-SCOPE" in result["corrected_open_retros"]
        assert "RETRO-ARBITER-FLAT-ENERGY" in result["corrected_open_retros"]

    def test_ising_not_closed_when_field_false(
        self, exp819_open, exp820_closed, exp830_with_both_open
    ):
        """If retro_injection_closed=False, RETRO-ISING stays in corrected_open_retros."""
        result = audit_retro_closures(exp819_open, exp820_closed, exp830_with_both_open)

        assert _RETRO_ISING in result["corrected_open_retros"]
        assert _RETRO_ISING not in result["retros_confirmed_closed"]

    def test_gguf_not_closed_when_verdict_wrong(
        self, exp819_closed, exp820_open, exp830_with_both_open
    ):
        """If honest_verdict != import_fixed_repair_positive, RETRO-GGUF stays open."""
        result = audit_retro_closures(exp819_closed, exp820_open, exp830_with_both_open)

        assert _RETRO_GGUF in result["corrected_open_retros"]
        assert _RETRO_GGUF not in result["retros_confirmed_closed"]

    def test_experiments_completed_extracted(
        self, exp819_closed, exp820_closed, exp830_with_both_open
    ):
        """experiments_completed must be read from exp830 artifact."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_with_both_open)
        assert result["experiments_completed"] == 728

    def test_experiments_over_cap(
        self, exp819_closed, exp820_closed, exp830_with_both_open
    ):
        """experiments_over_cap = max(0, experiments_completed - EXPERIMENT_CAP)."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_with_both_open)
        expected = max(0, 728 - EXPERIMENT_CAP)
        assert result["experiments_over_cap"] == expected

    def test_experiments_over_cap_zero_when_within_cap(
        self, exp819_closed, exp820_closed, exp830_minimal
    ):
        """experiments_over_cap must be 0 when completed <= cap."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_minimal)
        assert result["experiments_over_cap"] == 0

    def test_source_open_retros_preserved(
        self, exp819_closed, exp820_closed, exp830_with_both_open
    ):
        """retro_source_open_retros must contain the raw list from exp830."""
        result = audit_retro_closures(exp819_closed, exp820_closed, exp830_with_both_open)
        assert _RETRO_ISING in result["retro_source_open_retros"]
        assert _RETRO_GGUF in result["retro_source_open_retros"]

    def test_cascade_retros_closed_with_ising(
        self, exp819_closed, exp820_open, exp830_minimal
    ):
        """When RETRO-ISING closes, its cascade dependents also appear in confirmed_closed."""
        result = audit_retro_closures(exp819_closed, exp820_open, exp830_minimal)
        # RETRO-CONSTRAINT-ZERO-DELTA and RETRO-TIER1-PLATEAU are cascade-closed.
        assert "RETRO-CONSTRAINT-ZERO-DELTA" in result["retros_confirmed_closed"]
        assert "RETRO-TIER1-PLATEAU" in result["retros_confirmed_closed"]


# ── Tests for update_milestone_prereqs() ─────────────────────────────────────

class TestUpdateMilestonePrereqs:
    """SCENARIO-INFRA-071: MILESTONE_PREREQS.md updated with corrected RETRO status."""

    def test_prereqs_section_written(self, tmp_path):
        """update_milestone_prereqs writes the pre-flight section and returns True."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("# Existing content\n", encoding="utf-8")

        result = update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=["RETRO-ARBITER-FLAT-ENERGY"],
            confirmed_closed=[_RETRO_ISING, _RETRO_GGUF],
            experiments_completed=728,
            experiments_over_cap=28,
        )

        assert result is True
        content = prereqs_file.read_text(encoding="utf-8")
        assert "## Milestone 2026.04.64 Pre-flight" in content

    def test_closed_retros_marked_in_prereqs(self, tmp_path):
        """CLOSED RETROs must appear with CLOSED label in the written section."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=[],
            confirmed_closed=[_RETRO_ISING, _RETRO_GGUF],
            experiments_completed=728,
            experiments_over_cap=28,
        )

        content = prereqs_file.read_text(encoding="utf-8")
        # Both closed RETROs must appear with CLOSED marker.
        assert _RETRO_ISING in content
        assert _RETRO_GGUF in content
        assert "CLOSED" in content

    def test_open_retros_listed_in_prereqs(self, tmp_path):
        """Genuinely open RETROs must appear in the written section."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=["RETRO-MANIFEST-FULL-SCOPE"],
            confirmed_closed=[],
            experiments_completed=728,
            experiments_over_cap=28,
        )

        content = prereqs_file.read_text(encoding="utf-8")
        assert "RETRO-MANIFEST-FULL-SCOPE" in content

    def test_existing_content_preserved(self, tmp_path):
        """update_milestone_prereqs must NEVER remove pre-existing content (REQ: no pruning)."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        existing = "# Original section\nSome existing prereqs text.\n"
        prereqs_file.write_text(existing, encoding="utf-8")

        update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=[],
            confirmed_closed=[_RETRO_ISING],
            experiments_completed=728,
            experiments_over_cap=28,
        )

        content = prereqs_file.read_text(encoding="utf-8")
        assert "# Original section" in content
        assert "Some existing prereqs text." in content

    def test_idempotent_write(self, tmp_path):
        """Calling update_milestone_prereqs twice must not duplicate the section."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        kwargs = dict(
            prereqs_path=prereqs_file,
            corrected_open_retros=[],
            confirmed_closed=[_RETRO_ISING],
            experiments_completed=728,
            experiments_over_cap=28,
        )
        update_milestone_prereqs(**kwargs)
        update_milestone_prereqs(**kwargs)

        content = prereqs_file.read_text(encoding="utf-8")
        # Header must appear exactly once.
        assert content.count("## Milestone 2026.04.64 Pre-flight") == 1

    def test_experiment_cap_note_in_prereqs(self, tmp_path):
        """The over-cap status must appear in the written section."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=[],
            confirmed_closed=[],
            experiments_completed=728,
            experiments_over_cap=28,
        )

        content = prereqs_file.read_text(encoding="utf-8")
        assert "728" in content
        assert "700" in content

    def test_creates_file_if_not_exists(self, tmp_path):
        """update_milestone_prereqs must create MILESTONE_PREREQS.md if it does not exist."""
        prereqs_file = tmp_path / "NEW_PREREQS.md"
        assert not prereqs_file.exists()

        result = update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=[],
            confirmed_closed=[_RETRO_ISING],
            experiments_completed=728,
            experiments_over_cap=28,
        )

        assert result is True
        assert prereqs_file.exists()

    def test_within_cap_note_shown(self, tmp_path):
        """When experiments_completed <= cap, a within-cap note must appear."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        update_milestone_prereqs(
            prereqs_path=prereqs_file,
            corrected_open_retros=[],
            confirmed_closed=[],
            experiments_completed=700,
            experiments_over_cap=0,
        )

        content = prereqs_file.read_text(encoding="utf-8")
        assert "within" in content or "700" in content


# ── Tests for _load_json() ────────────────────────────────────────────────────

class TestLoadJson:
    """_load_json reads and parses a JSON file from disk."""

    def test_load_json_reads_file(self, tmp_path):
        """_load_json must return parsed dict from a valid JSON file."""
        data = {"key": "value", "number": 42}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = _load_json(json_file)

        assert result == data


# ── Tests for run_audit() ─────────────────────────────────────────────────────

class TestRunAudit:
    """run_audit integrates _load_json, audit_retro_closures, and update_milestone_prereqs."""

    def _make_results_dir(self, tmp_path: Path, exp819: dict, exp820: dict, exp830: dict) -> Path:
        """Write the three experiment JSON files into a temp results dir."""
        rdir = tmp_path / "results"
        rdir.mkdir()
        (rdir / "experiment_819_injection_field_fix.json").write_text(
            json.dumps(exp819), encoding="utf-8"
        )
        (rdir / "experiment_820_gguf_import_fix_code_repair_v5.json").write_text(
            json.dumps(exp820), encoding="utf-8"
        )
        (rdir / "operational_retro_2026_04_63.json").write_text(
            json.dumps(exp830), encoding="utf-8"
        )
        return rdir

    def test_governance_ready_when_both_closed(self, tmp_path):
        """honest_verdict must be governance_ready when >= 2 RETROs confirmed closed."""
        rdir = self._make_results_dir(
            tmp_path,
            {"retro_injection_closed": True},
            {"honest_verdict": "import_fixed_repair_positive"},
            {"experiments_completed": 728, "retros_still_open": []},
        )
        prereqs_file = tmp_path / "PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        result = run_audit(results_dir=rdir, prereqs_path=prereqs_file)

        assert result["honest_verdict"] == "governance_ready"
        assert result["prereqs_updated"] is True

    def test_governance_partial_when_one_closed(self, tmp_path):
        """honest_verdict must be governance_partial when prereqs updated but < 2 closed."""
        rdir = self._make_results_dir(
            tmp_path,
            {"retro_injection_closed": False},
            {"honest_verdict": "import_blocked"},
            {"experiments_completed": 728, "retros_still_open": []},
        )
        prereqs_file = tmp_path / "PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        result = run_audit(results_dir=rdir, prereqs_path=prereqs_file)

        assert result["honest_verdict"] == "governance_partial"

    def test_governance_issues_when_prereqs_not_updated(self, tmp_path):
        """honest_verdict must be governance_issues when update_milestone_prereqs returns False."""
        rdir = self._make_results_dir(
            tmp_path,
            {"retro_injection_closed": True},
            {"honest_verdict": "import_fixed_repair_positive"},
            {"experiments_completed": 728, "retros_still_open": []},
        )
        prereqs_file = tmp_path / "PREREQS.md"
        prereqs_file.write_text("", encoding="utf-8")

        # Patch update_milestone_prereqs to simulate a write failure.
        with patch(
            "scripts.experiment_831_governance_preflight.update_milestone_prereqs",
            return_value=False,
        ):
            result = run_audit(results_dir=rdir, prereqs_path=prereqs_file)

        assert result["honest_verdict"] == "governance_issues"
